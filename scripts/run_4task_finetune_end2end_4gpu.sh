#!/usr/bin/env bash
# 4-task DIAL finetune (end2end) — for 4×24GB RTX4090 (师弟版)
# 从预训练 checkpoint 续跑：默认 pretrain-golden-4task/checkpoint-20000（用 BASE_CKPT 覆盖）
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"$ROOT/Datasets/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-python}
OUT=${OUT:-"$ROOT/outputs/finetune-end2end-4task"}
LOG=/tmp/finetune_4task.log
MAX_STEPS=${MAX_STEPS:-20000}
BASE_CKPT=${BASE_CKPT:-"$ROOT/outputs/pretrain-golden-4task/checkpoint-20000"}
mkdir -p "$OUT"

DS=(
  "$DATA_ROOT/gr1_unified.PnPCupToDrawerClose"
  "$DATA_ROOT/gr1_unified.PnPPotatoToMicrowaveClose"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA"
)

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODEL_BF16=1
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type end2end \
  --dataset-path "${DS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus 4 --batch-size 2 --gradient-accumulation-steps 16 \
  --dataloader-num-workers 2 --max-steps "$MAX_STEPS" --save-steps 5000 \
  --base_model_path "$BASE_CKPT" \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --output_dir "$OUT" > "$LOG" 2>&1
echo "4TASK-FINETUNE DONE rc=$?" >> "$LOG"
