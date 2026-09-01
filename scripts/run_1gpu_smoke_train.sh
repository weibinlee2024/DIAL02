#!/usr/bin/env bash
# 1-task, short-run smoke train (decoupled / golden). Needs ≥1 preprocessed task.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_1gpu.sh"

TASK="${TASK:-PnPCupToDrawerClose}"
DS="$GR1_DIR/gr1_unified.${TASK}"
if [[ ! -d "$DS" ]]; then
  # fall back to any available task
  build_gr1_datasets
  DS="${GR1_DATASETS[0]}"
  echo "Using available dataset: $DS"
fi

MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
# bs=1, high accum, gradient checkpointing to fit on 24GB 4090
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

# Memory optimizations for single GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIENT_CHECKPOINTING=1

OUT="${OUT:-$ROOT/outputs/smoke-decoupled-1task}"

echo "=== smoke train: $DS  steps=$MAX_STEPS  bs=$BATCH_SIZE accum=$GRAD_ACCUM ==="

bash examples/train.sh decoupled \
  --dataset-path "$DS" \
  --data-config "$GR1_DATA_CONFIG" \
  --embodiment_tag gr1 \
  --data_split "[:20]" \
  --num-gpus "$NUM_GPUS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --dataloader-num-workers "$DATALOADER_WORKERS" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --base_model_path gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json \
  --compute-bridge-loss \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --no-tune-bridge-visual \
  --no-tune-bridge-goal \
  --tune-bridge-embedding \
  --select_layer 12 \
  --use_image_type_embedding \
  --ignore_lang_prefix \
  --report_to "$REPORT_TO" \
  --output_dir "$OUT"

echo "Smoke train finished: $OUT"
