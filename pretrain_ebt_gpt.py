# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain EBT (Energy-Based Transformer) with masked diffusion + MCMC refinement."""

import torch
import math
from contextlib import nullcontext
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import EBTGPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group, update_rotary_pos_emb
from megatron.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

import os
import subprocess
import torch.nn.functional as F
from torch.nn import Parameter


class PureRMSNorm(torch.nn.Module):
    """Pure-PyTorch RMSNorm that supports double backward (second-order gradients).

    Apex's MixedFusedRMSNorm uses a custom CUDA kernel whose backward is not
    differentiable, silently zeroing out second-order gradients needed by EBT's
    create_graph=True MCMC chain. This implementation uses only standard PyTorch
    ops, so torch.autograd can differentiate through the backward pass.
    """

    def __init__(self, dim, eps=1e-6, sequence_parallel=False):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))
        self.sequence_parallel = sequence_parallel
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def forward(self, x):
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


def _replace_fused_rmsnorm(model):
    """Replace all fused RMSNorm modules with PureRMSNorm for double-backward support."""
    try:
        from apex.normalization import MixedFusedRMSNorm
        fused_cls = MixedFusedRMSNorm
    except ImportError:
        return  # no apex, nothing to replace

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, fused_cls):
            # Build replacement with same dim, eps, and copy weight
            pure = PureRMSNorm(
                module.weight.shape[0],
                eps=module.eps if hasattr(module, 'eps') else 1e-6,
                sequence_parallel=getattr(module.weight, 'sequence_parallel', False),
            )
            pure.weight = module.weight  # share the same Parameter
            # Set the replacement on the parent module
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], pure)
            count += 1

    if count > 0:
        print_rank_0(f'  > replaced {count} fused RMSNorm layers with PureRMSNorm '
                     f'(required for EBT second-order gradients)')


def model_provider(pre_process=True, post_process=True):
    """Build the EBT model."""
    print_rank_0('building EBT (Energy-Based Transformer) model ...')
    see_memory_usage("Before Building Model", force=True)

    args = get_args()
    config = core_transformer_config_from_args(args)

    if hasattr(mpu, 'get_sequence_data_parallel_group'):
        dpg = mpu.get_sequence_data_parallel_group()
    elif hasattr(mpu, 'get_data_parallel_group'):
        dpg = mpu.get_data_parallel_group()
    else:
        dpg = None

    with deepspeed.zero.Init(data_parallel_group=dpg,
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config_dict if args.deepspeed else None,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        # EBT always uses non-pipeline model (requires direct forward calls for MCMC)
        model = EBTGPTModel(
            config=config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    # Replace fused RMSNorm with pure-PyTorch version so that second-order
    # gradients (create_graph=True in MCMC) flow correctly through the encoder.
    _replace_fused_rmsnorm(model)

    see_memory_usage("After Building Model", force=True)
    return model


def get_batch(data_iterator, eps=1e-3):
    """Generate a batch with random masking (same as MDM)."""
    args = get_args()
    tokenizer = get_tokenizer()

    mask_token_id = tokenizer.vocab_size

    keys = ['text']
    datatype = torch.int64

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    tokens_ = data_b['text'].long()
    tokens = tokens_[:, :-1].contiguous()  # use token itself as label (like MDM)

    micro_batch_size, seq_length = tokens.size()
    t = torch.rand(micro_batch_size, device=tokens.device)

    if args.low_noise_start and args.iteration < args.low_noise_iter:
        t = t * args.low_noise_ratio

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, seq_length)

    masked_indices = torch.rand((micro_batch_size, seq_length), device=tokens.device) < p_mask

    noisy_input = torch.where(masked_indices, mask_token_id, tokens)
    labels = tokens

    # Bidirectional attention mask (all ones)
    attention_mask = torch.ones(1, 1, seq_length, seq_length, device=tokens.device)
    attention_mask = (attention_mask < 0.5)

    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        True)

    # Sequence parallel handling
    seq_parallel_world_size = mpu.get_sequence_parallel_world_size()
    seq_parallel_world_rank = mpu.get_sequence_parallel_rank()

    if args.sequence_parallel:
        seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
        seq_parallel_world_rank = mpu.get_tensor_model_parallel_rank()

    seq_length = noisy_input.size(1)
    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    noisy_input = noisy_input[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]
    if mpu.get_sequence_parallel_world_size() > 1:
        labels = labels[:, sub_seq_start:sub_seq_end]

    return noisy_input, labels, loss_mask, attention_mask, position_ids, masked_indices, p_mask


def ebt_mcmc_step(model_module, predicted_logits, real_embeddings, masked_indices,
                  position_ids, attention_mask, alpha, step_idx, num_steps, args):
    """Perform one MCMC optimization step on predicted logits.

    Args:
        model_module: unwrapped EBTGPTModel
        predicted_logits: [B, S, V] current predicted logits (detached, requires_grad)
        real_embeddings: [B, S, H] embeddings of real (unmasked) tokens
        masked_indices: [B, S] boolean mask of which positions are masked
        position_ids: [B, S]
        attention_mask: [1, 1, S, S]
        alpha: step size (scalar or tensor)
        step_idx: current MCMC step index
        num_steps: total number of MCMC steps
        args: global args

    Returns:
        updated_logits: [B, S, V] updated predicted logits
        energies: [B, S, 1] energy values from this step
    """
    is_final_step = (step_idx == num_steps - 1)
    truncate = args.ebt_truncate_mcmc

    predicted_logits = predicted_logits.detach().requires_grad_(True)

    # Optional Langevin noise
    if args.ebt_langevin_noise > 0:
        noise = torch.randn_like(predicted_logits) * args.ebt_langevin_noise
        predicted_logits_noisy = predicted_logits + noise
    else:
        predicted_logits_noisy = predicted_logits

    # Optional RMSNorm on predictions (all but final step)
    if model_module.pred_rmsnorm is not None: #TODO possibly add and not is_final_step clause as well?
        predicted_logits_noisy = model_module.pred_rmsnorm(predicted_logits_noisy)

    # Convert logits to embeddings
    # In hybrid mode (bf16 + fp32 MCMC), disable autocast so energy computation
    # runs in fp32 — required for second-order gradients via create_graph=True
    use_hybrid = args.bf16 and not args.ebt_fp32
    ctx = torch.amp.autocast('cuda', enabled=False) if use_hybrid else nullcontext()

    with ctx:
        if use_hybrid:
            predicted_logits_noisy = predicted_logits_noisy.float()

        pred_embeddings = model_module.logits_to_embeddings(
            predicted_logits_noisy, normalize=args.ebt_normalize_predictions)

        # Merge: real embeddings for unmasked, predicted for masked
        if use_hybrid:
            real_embeddings_cast = real_embeddings.float()
        else:
            real_embeddings_cast = real_embeddings
        masked_expanded = masked_indices.unsqueeze(-1).expand_as(real_embeddings_cast)
        merged_embeddings = torch.where(masked_expanded, pred_embeddings, real_embeddings_cast)

        # Get energies from transformer (runs in fp32 under disabled autocast)
        energies = model_module.forward_energy(merged_embeddings, position_ids, attention_mask)

        # Compute gradient of energy sum w.r.t. predicted logits
        create_graph = is_final_step if truncate else True
        grad = torch.autograd.grad(
            energies.sum(), predicted_logits,
            create_graph=create_graph)[0]

    # Gradient descent step
    updated_logits = predicted_logits - alpha * grad

    return updated_logits, energies


def ebt_mcmc_step_embed(model_module, predicted_embeds, real_embeddings, masked_indices,
                        position_ids, attention_mask, alpha, step_idx, num_steps, args):
    """Perform one MCMC optimization step directly in embedding space.

    Like ebt_mcmc_step but operates on [B, S, H] embeddings instead of [B, S, V] logits.

    Returns:
        updated_embeds: [B, S, H] updated predicted embeddings
        energies: [B, S, 1] energy values from this step
    """
    is_final_step = (step_idx == num_steps - 1)
    truncate = args.ebt_truncate_mcmc

    predicted_embeds = predicted_embeds.detach().requires_grad_(True)

    # Optional Langevin noise
    if args.ebt_langevin_noise > 0:
        predicted_embeds_noisy = predicted_embeds + torch.randn_like(predicted_embeds) * args.ebt_langevin_noise
    else:
        predicted_embeds_noisy = predicted_embeds

    # Optional RMSNorm on predictions
    if model_module.pred_rmsnorm is not None and not is_final_step:
        predicted_embeds_noisy = model_module.pred_rmsnorm(predicted_embeds_noisy)

    use_hybrid = args.bf16 and not args.ebt_fp32
    ctx = torch.amp.autocast('cuda', enabled=False) if use_hybrid else nullcontext()

    with ctx:
        if use_hybrid:
            predicted_embeds_noisy = predicted_embeds_noisy.float()
            real_embeddings_cast = real_embeddings.float()
        else:
            real_embeddings_cast = real_embeddings

        # Merge: real embeddings for unmasked, predicted for masked
        masked_expanded = masked_indices.unsqueeze(-1).expand_as(real_embeddings_cast)
        merged_embeddings = torch.where(masked_expanded, predicted_embeds_noisy, real_embeddings_cast)

        # Get energies from transformer
        energies = model_module.forward_energy(merged_embeddings, position_ids, attention_mask)

        # Compute gradient of energy sum w.r.t. predicted embeddings
        create_graph = is_final_step if truncate else True
        grad = torch.autograd.grad(
            energies.sum(), predicted_embeds,
            create_graph=create_graph)[0]

    # Gradient descent step
    updated_embeds = predicted_embeds - alpha * grad

    return updated_embeds, energies


def loss_func(loss_mask, moe_loss, mos_loss, masked_indices, p_mask, output_tensor):
    """Compute EBT loss on masked positions from the final predicted distribution.

    output_tensor here contains the NLL losses per token (from cross entropy
    of final predicted logits vs true labels).
    """
    args = get_args()
    # Zero out losses at positions after EOD tokens
    output_tensor = output_tensor * loss_mask
    losses = output_tensor[masked_indices].float() / p_mask[masked_indices].float()
    loss = losses.sum() / (output_tensor.shape[0] * output_tensor.shape[1])

    averaged_loss = average_losses_across_data_parallel_group([loss])
    if max(args.num_experts) <= 1:
        return loss, {'lm loss': averaged_loss[0]}
    else:
        loss = loss + moe_loss
        return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}


def forward_step(data_iterator, model):
    """Forward step with MCMC energy-based refinement."""
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, masked_indices, p_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    # Unwrap model to access EBT-specific methods
    # DeepSpeed engine wraps as: engine.module = Float16Module.module = EBTGPTModel
    model_module = model
    while hasattr(model_module, 'module'):
        model_module = model_module.module

    B, S = tokens.shape
    vocab_size = args.padded_vocab_size

    # Get real token embeddings
    real_embeddings = model_module.embed_tokens(tokens)

    # MCMC step size
    alpha = torch.clamp(model_module.alpha, min=1e-4)

    # Optional step size randomization
    if args.ebt_randomize_step_size_scale != 1.0 and model_module.training:
        scale = args.ebt_randomize_step_size_scale
        low = alpha / scale
        high = alpha * scale
        alpha = low + torch.rand(1, device=tokens.device) * (high - low)

    # Determine number of MCMC steps
    num_steps = args.ebt_num_mcmc_steps
    if args.ebt_randomize_num_steps > 0 and model_module.training:
        extra = torch.randint(0, args.ebt_randomize_num_steps + 1, (1,)).item()
        num_steps = num_steps + extra

    if args.ebt_use_mask_token:
        # === Embedding-space MCMC mode ===
        # Initialize from mask token embeddings (same as what MDM sees)
        tokenizer = get_tokenizer()
        mask_token_id = tokenizer.vocab_size
        mask_embeds = model_module.embed_tokens(
            torch.full((B, S), mask_token_id, device=tokens.device, dtype=torch.long))
        predicted_embeds = mask_embeds.float()

        # Run MCMC in embedding space
        with torch.set_grad_enabled(True):
            for step_idx in range(num_steps):
                predicted_embeds, energies = ebt_mcmc_step_embed(
                    model_module, predicted_embeds, real_embeddings, masked_indices,
                    position_ids, attention_mask, alpha, step_idx, num_steps, args)

        # Map refined embeddings to logits via embed_to_vocab
        predicted_logits = model_module.embed_to_vocab(predicted_embeds.float())
    else:
        # === Logit-space MCMC mode (original) ===
        predicted_logits = torch.randn(
            B, S, vocab_size, device=tokens.device,
            dtype=torch.float32)

        # Run MCMC optimization loop
        with torch.set_grad_enabled(True):
            for step_idx in range(num_steps):
                predicted_logits, energies = ebt_mcmc_step(
                    model_module, predicted_logits, real_embeddings, masked_indices,
                    position_ids, attention_mask, alpha, step_idx, num_steps, args)

    # Compute loss on final predicted distribution
    final_log_probs = F.log_softmax(predicted_logits.float(), dim=-1)  # [B, S, V]

    labels_for_loss = labels.long()

    # Compute NLL loss per token: -log_prob[true_token]
    output_tensor = -final_log_probs.gather(
        dim=-1, index=labels_for_loss.unsqueeze(-1)).squeeze(-1)  # [B, S]

    moe_loss = torch.tensor(0.0, device=tokens.device)
    mos_loss = torch.tensor(0.0, device=tokens.device)

    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss, masked_indices, p_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets for EBT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating EBT datasets ...")
    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


if __name__ == "__main__":
    git_ds_info()
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
