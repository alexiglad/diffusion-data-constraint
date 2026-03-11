#!/bin/bash
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --job-name=ebt_alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-cpu=15G
#SBATCH --time=72:00:00
#SBATCH --output=output/logs/ebt_alpha_%A_%a.out
#SBATCH --error=output/logs/ebt_alpha_%A_%a.err
#SBATCH --array=2

module load cuda/12.4.1-fasrc01 gcc/12.2.0-fasrc01
source ~/miniconda3/etc/profile.d/conda.sh
conda activate data_eff

export MASTER_PORT=$((RANDOM%16384+49152))
# export CUDA_VISIBLE_DEVICES=0 #TODO TEMP REMOVE

# Step size sweep
STEP_SIZES=(0.005 0.05 0.5 5 50 500)
STEP_SIZE=${STEP_SIZES[$SLURM_ARRAY_TASK_ID]}

# Fixed model: 74M
source utils/model_params.sh
MODEL_PARAM=("${PARAM_74M[@]}")
NHIDDEN=${MODEL_PARAM[0]}
FFN_HIDDEN_SIZE=${MODEL_PARAM[1]}
KV_SIZE=${MODEL_PARAM[2]}
NHEADS=${MODEL_PARAM[3]}
NLAYERS=${MODEL_PARAM[4]}
SIZE=74M

VARIANT=ebt_25m_512_2_mcmc_bf_rms_after_wt_better_init_longer_warm_beta2_0.999_${SIZE}_alpha${STEP_SIZE}_500ep
echo "Running variant: $VARIANT"

CHECKPOINT_PATH=output/checkpoints_$VARIANT
TENSORBOARD_PATH=output/tensorboard_$VARIANT
KILL_SWITCH_PATH=output/kill-switch-$VARIANT
WANDB_PATH=output/wandb_$VARIANT
WANDB_GROUP=ebt_alpha_sweep
WANDB_PROJECT="ebt_data_eff"
WANDB_EXP_NAME=$VARIANT

VOCAB_FILE=utils/data/gpt2-vocab.json
MERGE_FILE=utils/data/gpt2-merges.txt
TRAIN_DATA_PATH=utils/datapaths/train_c4_25m.txt
VALID_DATA_PATH=utils/datapaths/val_c4.txt

PP_SIZE=1
TP_SIZE=1

SEQ_LEN=512
MICRO_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256

source utils/epoch_tokens.sh
DATA_CNT=${DATA_25M[@]}
EPOCH_CNT=500

echo "Model: $SIZE, alpha: $STEP_SIZE, d_model $NHIDDEN, n_heads $NHEADS, n_layers $NLAYERS"

SAVE_INTERVAL=1000
EVAL_INTERVAL=1000
LOG_INTERVAL=10
EVAL_ITERS=10

TRAIN_SAMPLES=$((DATA_CNT*EPOCH_CNT/SEQ_LEN))
WARMUP_SAMPLES=$((TRAIN_SAMPLES/10))
echo "Training samples: $TRAIN_SAMPLES, Epochs: $EPOCH_CNT, Data: $DATA_CNT"

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 2e-4 \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $TRAIN_SAMPLES \
    --lr-warmup-samples $WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --log-grad-stats \
    "

EBT_ARGS=" \
    --ebt-num-mcmc-steps 2 \
    --ebt-step-size $STEP_SIZE \
    --ebt-vocab-to-embed-method linear \
    --ebt-normalize-predictions \
    --ebt-truncate-mcmc \
    --ebt-fp32 \
    --ebt-use-mask-token \
    --ebt-pred-rmsnorm \
    --ebt-weight-tie \
    "
    # --ebt-fp32 \
    # --ebt-truncate-mcmc \

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --kv-channels $KV_SIZE \
    --seq-length $SEQ_LEN \
    --use-rotary-position-embeddings \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --disable-bias-linear \
    --kill-switch-file $KILL_SWITCH_PATH \
    --normalization rmsnorm \
    --swiglu \
    $OPTIMIZER_ARGS \
    $EBT_ARGS \
    "
    # --bf16 \

OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --keep-best-n-checkpoints 3 \
    --eval-iters $EVAL_ITERS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    --wandb-save-dir $WANDB_PATH \
    --wandb-group $WANDB_GROUP \
    --online-eval-tasks lambada_standard,hellaswag,piqa \
    --online-eval-interval $EVAL_INTERVAL \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --train-data-path-file $TRAIN_DATA_PATH \
    --valid-data-path-file $VALID_DATA_PATH \
    --data-impl mmap \
    "

ZERO_STAGE=0
mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$VARIANT.json"

# bf16 enabled : true
cat <<EOF > $DS_CONFIG_PATH
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "bf16": {
        "enabled": false
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE
    "

mkdir -p output/logs

CMD=" pretrain_ebt_gpt.py \
    --no-pipeline-parallel \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $DEEPSPEED_ARGS \
    "

LAUNCHER="deepspeed --master_port $MASTER_PORT"

$LAUNCHER $CMD
