#!/usr/bin/env bash
# 2-card cross-machine pretrain (golden / decoupled), local machine (10.10.70.153)
set -euo pipefail

ROOT=/home/weibin/DIAL-master
DATA_ROOT=$ROOT/Datasets/LeRobot-AugPosRot-Correct
PYTHON=/home/weibin/miniconda3/envs/dial/bin/python

TASKS=(
  PnPBottleToCabinetClose PnPCanToDrawerClose PnPCupToDrawerClose PnPMilkToMicrowaveClose
  PnPPotatoToMicrowaveClose PnPWineToCabinetClose
  PosttrainPnPNovelFromCuttingboardToBasketSplitA PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
  PosttrainPnPNovelFromCuttingboardToPanSplitA PosttrainPnPNovelFromCuttingboardToPotSplitA
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlacematToBowlSplitA PosttrainPnPNovelFromPlacematToPlateSplitA
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromPlateToCardboardboxSplitA PosttrainPnPNovelFromPlateToPanSplitA
  PosttrainPnPNovelFromPlateToPlateSplitA PosttrainPnPNovelFromTrayToCardboardboxSplitA
  PosttrainPnPNovelFromTrayToPlateSplitA PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA PosttrainPnPNovelFromTrayToTieredshelfSplitA
)

DS_ARGS=()
for t in "${TASKS[@]}"; do DS_ARGS+=("$DATA_ROOT/gr1_unified.$t"); done

OUT=$ROOT/outputs/pretrain-golden-2card
LOG=/tmp/opencode/pretrain_local.log
mkdir -p "$OUT"

export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
export IS_TORCHRUN=1

exec "$PYTHON" -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=10.10.70.153 --master_port=29500 --node_rank=0 \
  --max_restarts=0 \
  "$ROOT/scripts/dual_system_train.py" \
  --bridge_type golden \
  --dataset-path "${DS_ARGS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus 1 --batch-size 4 --gradient-accumulation-steps 16 \
  --dataloader-num-workers 2 --max-steps 40000 --save-steps 1000 \
  --base_model_path "$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json" \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --output_dir "$OUT" > "$LOG" 2>&1
