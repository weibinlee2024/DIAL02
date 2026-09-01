#!/usr/bin/env bash
# LIBERO smoke: Stage1 decoupled on one suite, 1×4090 (~48GB).
# Outputs go to weibin02 (system disk is nearly full).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIBERO_DIR="${LIBERO_DIR:-$ROOT/data/libero}"
SUITE="${SUITE:-libero_spatial}"
DS="$LIBERO_DIR/$SUITE"
[[ -d "$DS" ]] || { echo "ERROR: missing dataset $DS" >&2; exit 1; }

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUT="${OUT:-/media/weibin/weibin02/DIAL/outputs/libero-smoke-decoupled-bs4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIENT_CHECKPOINTING=1

mkdir -p "$(dirname "$OUT")" "$ROOT/outputs/logs"

echo "=== LIBERO smoke decoupled: $DS ==="
echo "    steps=$MAX_STEPS bs=$BATCH_SIZE accum=$GRAD_ACCUM out=$OUT"

bash examples/train.sh decoupled \
  --dataset-path "$DS" \
  --data-config libero \
  --embodiment-tag libero \
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
  --output-dir "$OUT"

echo "Smoke train finished: $OUT"
