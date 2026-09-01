#!/usr/bin/env bash
# Scenario B Stage2: end2end finetune from sim-only pretrain on 1×4090.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_1gpu.sh"
build_gr1_datasets

PRETRAIN_CKPT="${PRETRAIN_CKPT:-$ROOT/outputs/pretrain-decoupled-sim-only-1gpu/checkpoint-80000}"
MAX_STEPS="${MAX_STEPS:-80000}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
OUT="${OUT:-$ROOT/outputs/finetune-end2end-sim-only-1gpu}"

if [[ ! -d "$PRETRAIN_CKPT" && ! -f "$PRETRAIN_CKPT/config.json" ]]; then
  # allow either directory checkpoint or HF-style folder
  if [[ ! -e "$PRETRAIN_CKPT" ]]; then
    echo "ERROR: missing pretrain ckpt: $PRETRAIN_CKPT" >&2
    echo "Set PRETRAIN_CKPT=... or finish run_1gpu_pretrain_sim.sh" >&2
    exit 1
  fi
fi

echo "=== finetune end2end from $PRETRAIN_CKPT ==="

bash examples/train.sh end2end \
  --dataset-path "${GR1_DATASETS[@]}" \
  --data-config "$GR1_DATA_CONFIG" \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  "${COMMON_FLAGS[@]}" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --base_model_path "$PRETRAIN_CKPT" \
  --use_separate_projector_for_loss \
  --output_dir "$OUT"

echo "Finetune done: $OUT"
