#!/usr/bin/env bash
# 4-task DIAL pretrain WITH state-future — 4×24GB (师弟版)
#   PnPCupToDrawerClose / PnPPotatoToMicrowaveClose /
#   CuttingboardToBasketSplitA / PlacematToPlateSplitA
#
# 配置说明（4×24GB 必须）:
#   --num-gpus 4 --batch-size 2 --gradient-accumulation-steps 16  → 每卡有效 32，合计 128
#   3 个环境变量已内置: MODEL_BF16=1 GRADIENT_CHECKPOINTING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# 用法:
#   先改 DATA_ROOT 指向你本机 LeRobot-AugPosRot-Correct 目录（若与默认相同则无需改）
#   MAX_STEPS 可调步数（默认 20000）: MAX_STEPS=30000 bash run_4task_pretrain_golden_sf_4gpu.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"           # 代码根目录（含 gr00t/ 的那层）
DATA_ROOT=${DATA_ROOT:-"$ROOT/Datasets/LeRobot-AugPosRot-Correct"}
PYTHON=${PYTHON:-python}
OUT=${OUT:-"$ROOT/outputs/pretrain-golden-4task"}
LOG=/tmp/pretrain_4task.log
MAX_STEPS=${MAX_STEPS:-20000}
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
  --bridge_type golden \
  --dataset-path "${DS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus 4 --batch-size 2 --gradient-accumulation-steps 16 \
  --dataloader-num-workers 2 --max-steps "$MAX_STEPS" --save-steps 5000 \
  --base_model_path "$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json" \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --use-state-future --compute-state-future-loss --state-future-loss-weight 0.5 \
  --output_dir "$OUT" > "$LOG" 2>&1
echo "4TASK-PRETRAIN DONE rc=$?" >> "$LOG"
