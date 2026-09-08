#!/usr/bin/env bash
# 4-task DIAL pretrain (golden) — PnPCupToDrawerClose / PnPPotatoToMicrowaveClose /
#                           CuttingboardToBasketSplitA / PlacematToPlateSplitA
# Usage: adjust MAX_STEPS / OUT below; run from repo root.
# 本机单卡版：--num-gpus 1 --batch-size 4 --accum 16（有效 64）
# 师弟 4×24G 版：--num-gpus 4 --batch-size 2 --accum 16（每卡有效32，合计128）
#   并 export MODEL_BF16=1 GRADIENT_CHECKPOINTING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -u
ROOT=/home/weibin/DIAL-master
DATA_ROOT=$ROOT/Datasets/LeRobot-AugPosRot-Correct
PYTHON=/home/weibin/miniconda3/envs/dial/bin/python
OUT=$ROOT/outputs/pretrain-golden-4task
LOG=/tmp/opencode/pretrain_4task.log
MAX_STEPS=${MAX_STEPS:-20000}      # 4任务数据≈24任务1/6，建议 10k-30k
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
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type golden \
  --dataset-path "${DS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus 1 --batch-size 4 --gradient-accumulation-steps 16 \
  --dataloader-num-workers 2 --max-steps "$MAX_STEPS" --save-steps 5000 \
  --base_model_path "$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json" \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --output_dir "$OUT" > "$LOG" 2>&1
echo "4TASK-PRETRAIN DONE rc=$?" >> "$LOG"
