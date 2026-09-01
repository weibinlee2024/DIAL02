#!/usr/bin/env bash
# Shared env for 1×4090 training. Source from other run_*.sh scripts.
# Official: 8 GPUs × bs32 = global 256. We use bs × accum ≈ 256.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_local_paths.sh >/dev/null

export DIAL_ROOT="$ROOT"
export GR1_DIR="${GR1_DIR:-$ROOT/data/LeRobot-AugPosRot-Correct}"
export GR1_DATA_CONFIG="${GR1_DATA_CONFIG:-fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop}"

# Per-GPU micro-batch / accumulation (override as needed)
export NUM_GPUS="${NUM_GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"   # 1*16=16 effective batch; OOM-safe on 24GB
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

# Memory optimizations (must be set before torch/CUDA init)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRADIENT_CHECKPOINTING=1

GR1_TASK_NAMES=(
  PnPPotatoToMicrowaveClose
  PnPMilkToMicrowaveClose
  PnPCanToDrawerClose
  PnPCupToDrawerClose
  PnPBottleToCabinetClose
  PnPWineToCabinetClose
  PosttrainPnPNovelFromPlacematToBowlSplitA
  PosttrainPnPNovelFromPlateToPlateSplitA
  PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToTieredshelfSplitA
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA
  PosttrainPnPNovelFromPlateToCardboardboxSplitA
  PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlateToPanSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA
  PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromCuttingboardToBasketSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA
)

build_gr1_datasets() {
  GR1_DATASETS=()
  local missing=0
  for t in "${GR1_TASK_NAMES[@]}"; do
    local p="$GR1_DIR/gr1_unified.${t}"
    if [[ -d "$p" ]]; then
      GR1_DATASETS+=("$p")
    else
      echo "WARN: missing $p" >&2
      missing=$((missing + 1))
    fi
  done
  if [[ ${#GR1_DATASETS[@]} -eq 0 ]]; then
    echo "ERROR: no augmented GR1 datasets under $GR1_DIR" >&2
    echo "Run download + preprocess first (see REPRO_1x4090.md)." >&2
    exit 1
  fi
  if [[ "$missing" -gt 0 ]]; then
    echo "WARN: ${missing} tasks missing; training on ${#GR1_DATASETS[@]} available tasks." >&2
  fi
}

COMMON_FLAGS=(
  --compute-bridge-loss
  --no-tune-llm
  --no-tune-visual
  --tune-projector
  --tune-diffusion-model
  --no-tune-bridge-visual
  --no-tune-bridge-goal
  --tune-bridge-embedding
  --select_layer 12
  --use_image_type_embedding
  --ignore_lang_prefix
  --report_to "$REPORT_TO"
  --num-gpus "$NUM_GPUS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --dataloader-num-workers "$DATALOADER_WORKERS"
)
