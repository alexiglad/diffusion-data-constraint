# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# EBT (Energy-Based Transformer) model for masked diffusion with energy-based refinement.

"""EBT GPT model — bidirectional transformer with energy head for MCMC refinement."""

import torch
from torch import nn
import torch.nn.functional as F

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits, get_language_model
from .utils import init_method_normal, scaled_init_method_normal
from megatron.model import RMSNorm


class EBTGPTModel(MegatronModule):
    """Energy-Based Transformer with bidirectional attention and energy head.

    Uses the same transformer backbone as DiffGPTModel (bidirectional attention),
    but adds an energy head that outputs scalar energy per position.
    During training, MCMC optimization refines predicted logits by descending
    the energy landscape, then NLL loss is computed on the refined predictions.
    """

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        args = get_args()
        super().__init__(config=config,
                         share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.hidden_size = args.hidden_size

        # Same bidirectional transformer as MDM (AttnMaskType.padding)
        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

        self.input_tensor = None

        # === EBT-specific components ===

        # Energy head: maps hidden states to scalar energy per position
        self.energy_head = nn.Linear(args.hidden_size, 1, bias=False)
        nn.init.xavier_uniform_(self.energy_head.weight)

        self.ebt_weight_tie = getattr(args, 'ebt_weight_tie', False)

        # Vocab-to-embed: maps predicted logits (vocab_size) to embedding space
        vocab_size = args.padded_vocab_size
        if args.ebt_vocab_to_embed_method == 'linear' and not self.ebt_weight_tie:
            self.vocab_to_embed = nn.Linear(vocab_size, args.hidden_size, bias=False)
            nn.init.xavier_uniform_(self.vocab_to_embed.weight)
        else:
            # weighted_sum mode, or weight-tied (uses word_emb.weight directly)
            self.vocab_to_embed = None

        # Embed-to-vocab: maps refined embeddings back to logit space
        # Used when --ebt-use-mask-token (MCMC in embedding space)
        # When weight-tied, no separate parameter — uses word_emb.weight directly.
        if args.ebt_use_mask_token and not self.ebt_weight_tie:
            self.embed_to_vocab = nn.Linear(args.hidden_size, vocab_size, bias=False)
            nn.init.xavier_uniform_(self.embed_to_vocab.weight)
        else:
            self.embed_to_vocab = None

        # MCMC step size (alpha)
        self.alpha = nn.Parameter(
            torch.tensor(float(args.ebt_step_size)),
            requires_grad=args.ebt_step_size_learnable)

        # Optional RMSNorm on predictions
        if args.ebt_pred_rmsnorm:
            self.pred_rmsnorm = RMSNorm(vocab_size if not args.ebt_use_mask_token else args.hidden_size, eps=1e-6)
        else:
            self.pred_rmsnorm = None

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def get_word_embeddings(self):
        """Get word embedding weight for weighted_sum vocab_to_embed."""
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            return self.word_embeddings.weight

    def forward_energy(self, embeddings, position_ids, attention_mask):
        """Forward pass that returns per-position energy scalars.

        Args:
            embeddings: [batch, seq, hidden] — merged real + predicted embeddings
            position_ids: [batch, seq]
            attention_mask: [1, 1, seq, seq]

        Returns:
            energies: [batch, seq, 1] — scalar energy per position
        """
        # Run through transformer encoder (bypass embedding layer)
        # We need to pass embeddings directly to the encoder
        encoder_input = embeddings.transpose(0, 1).contiguous()  # [seq, batch, hidden]

        args = get_args()
        # Get rotary embeddings if needed
        rotary_pos_emb = None
        if args.use_rotary_position_embeddings:
            rotary_pos_emb = self.language_model.rotary_pos_emb(embeddings.shape[1])

        # Cast encoder_input to the encoder's parameter dtype (e.g. bf16 in hybrid mode).
        # The MCMC step passes fp32 embeddings (for gradient stability), but the encoder
        # weights may be bf16. The cast is differentiable so second-order gradients still flow.
        encoder_param_dtype = next(self.language_model.encoder.parameters()).dtype
        encoder_input = encoder_input.to(encoder_param_dtype)

        # Run encoder (final_layernorm is applied inside when post_process=True)
        encoder_output, *moe_losses = self.language_model.encoder(
            encoder_input,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb)

        # [seq, batch, hidden] -> [batch, seq, hidden]
        hidden_states = encoder_output.transpose(0, 1).contiguous()

        # Energy head: [batch, seq, hidden] -> [batch, seq, 1]
        # Use float32 for energy computation (needed for autograd stability)
        energies = F.linear(hidden_states.float(), self.energy_head.weight.float())

        return energies

    def embed_tokens(self, input_ids):
        """Look up token embeddings."""
        if self.pre_process:
            return self.language_model.embedding.word_embeddings(input_ids)
        else:
            return self.word_embeddings(input_ids)

    def logits_to_embeddings(self, predicted_logits, normalize=True):
        """Convert predicted logits to embedding space.

        Args:
            predicted_logits: [batch, seq, vocab_size]
            normalize: whether to apply softmax before conversion

        Returns:
            embeddings: [batch, seq, hidden_size]
        """
        if normalize:
            probs = torch.softmax(predicted_logits, dim=-1)
        else:
            probs = predicted_logits

        if self.vocab_to_embed is not None:
            # Linear projection — cast to match weight dtype
            return self.vocab_to_embed(probs.to(self.vocab_to_embed.weight.dtype))
        else:
            # Weight-tied or weighted_sum: probs @ word_embeddings
            # Both paths are equivalent: matmul(probs, W) where W=[vocab, hidden]
            word_emb = self.get_word_embeddings()  # [vocab, hidden]
            return torch.matmul(probs.to(word_emb.dtype), word_emb)

    def forward(self, input_ids, position_ids, attention_mask,
                labels=None, tokentype_ids=None, inference_params=None,
                curriculum_seqlen=None):
        """Standard forward pass — returns logits or loss like DiffGPTModel.

        This is used for validation loss computation (not the MCMC training loop).
        """
        lm_output, moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inference_params=inference_params)

        if self.post_process:
            lm_output = parallel_lm_logits(
                lm_output,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights
                else self.shared_embedding_or_output_weight(),
                self.parallel_output)

            if labels is None:
                # [s b h] => [b s h]
                return lm_output.transpose(0, 1).contiguous()
            else:
                # Compute per-token cross entropy for validation
                labels_t = labels.transpose(0, 1).contiguous()
                loss = tensor_parallel.vocab_parallel_cross_entropy(
                    lm_output.float(), labels_t)
                loss = loss.transpose(0, 1).contiguous()
                return loss, moe_losses if self.return_moe_loss else loss

        return lm_output, moe_losses if self.return_moe_loss else lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save EBT-specific parameters
        state_dict_['energy_head'] = self.energy_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.vocab_to_embed is not None:
            state_dict_['vocab_to_embed'] = self.vocab_to_embed.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.embed_to_vocab is not None:
            state_dict_['embed_to_vocab'] = self.embed_to_vocab.state_dict(prefix=prefix, keep_vars=keep_vars)
        state_dict_['alpha'] = self.alpha.data
        if self.pred_rmsnorm is not None:
            state_dict_['pred_rmsnorm'] = self.pred_rmsnorm.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] = \
                self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Load EBT-specific parameters
        if 'energy_head' in state_dict:
            self.energy_head.load_state_dict(state_dict.pop('energy_head'), strict=strict)
        if 'vocab_to_embed' in state_dict and self.vocab_to_embed is not None:
            self.vocab_to_embed.load_state_dict(state_dict.pop('vocab_to_embed'), strict=strict)
        if 'embed_to_vocab' in state_dict and self.embed_to_vocab is not None:
            self.embed_to_vocab.load_state_dict(state_dict.pop('embed_to_vocab'), strict=strict)
        if 'alpha' in state_dict:
            self.alpha.data = state_dict.pop('alpha')
        if 'pred_rmsnorm' in state_dict and self.pred_rmsnorm is not None:
            self.pred_rmsnorm.load_state_dict(state_dict.pop('pred_rmsnorm'), strict=strict)
        # Load language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)
