#!/usr/bin/env bash
# LIBERO Stage1: decoupled pretrain on all 4 suites, 1×4090.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIBERO_DIR="${LIBERO_DIR:-$ROOT/data/libero}"
SUITES=(libero_spatial libero_object libero_goal libero_10)
DATASETS=()
for s in "${SUITES[@]}"; do
  p="$LIBERO_DIR/$s"
  [[ -d "$p" ]] || { echo "ERROR: missing $p" >&2; exit 1; }
  DATASETS+=("$p")
done

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-80000}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
REPORT_TO="${REPORT_TO:-tensorboard}"
OUT="${OUT:-/media/weibin/weibin02/DIAL/outputs/libero-pretrain-decoupled-bs4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIENT_CHECKPOINTING=1

mkdir -p "$(dirname "$OUT")" "$ROOT/outputs/logs"

echo "=== LIBERO pretrain decoupled: ${#DATASETS[@]} suites, steps=$MAX_STEPS ==="
echo "    global_batch ≈ $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)) out=$OUT"

bash examples/train.sh decoupled \
  --dataset-path "${DATASETS[@]}" \
  --data-config libero libero libero libero \
  --embodiment-tag libero libero libero libero \
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

echo "Pretrain done: $OUT"
