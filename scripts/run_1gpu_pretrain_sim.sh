#!/usr/bin/env bash
# Scenario B Stage1: sim-only decoupled pretrain on 1×4090.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_1gpu.sh"
build_gr1_datasets

MAX_STEPS="${MAX_STEPS:-80000}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
OUT="${OUT:-$ROOT/outputs/pretrain-decoupled-sim-only-1gpu}"

echo "=== pretrain decoupled: ${#GR1_DATASETS[@]} tasks, steps=$MAX_STEPS ==="
echo "    global_batch ≈ $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)) (target ~256)"

bash examples/train.sh decoupled \
  --dataset-path "${GR1_DATASETS[@]}" \
  --data-config "$GR1_DATA_CONFIG" \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  "${COMMON_FLAGS[@]}" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --base_model_path gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json \
  --output_dir "$OUT"

echo "Pretrain done: $OUT"
