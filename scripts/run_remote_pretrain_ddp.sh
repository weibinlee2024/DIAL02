#!/usr/bin/env bash
# 2-card pretrain (golden), REMOTE machine rank1
set -u
ROOT=/home/weibin/projects3/DIAL-master_Libero
DATA_ROOT=/media/weibin/C4E60209E601FC84/DIAL/datasets/LeRobot-AugPosRot-Correct
PYTHON=/home/weibin/anaconda3/envs/dial/bin/python
OUT=/media/weibin/C4E60209E601FC84/DIAL/outputs-ditfm/pretrain-golden-2card
LOG=/tmp/pretrain_remote.log
mkdir -p "$OUT"

TASKS="PnPBottleToCabinetClose PnPCanToDrawerClose PnPCupToDrawerClose PnPMilkToMicrowaveClose PnPPotatoToMicrowaveClose PnPWineToCabinetClose PosttrainPnPNovelFromCuttingboardToBasketSplitA PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA PosttrainPnPNovelFromCuttingboardToPanSplitA PosttrainPnPNovelFromCuttingboardToPotSplitA PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA PosttrainPnPNovelFromPlacematToBasketSplitA PosttrainPnPNovelFromPlacematToBowlSplitA PosttrainPnPNovelFromPlacematToPlateSplitA PosttrainPnPNovelFromPlacematToTieredshelfSplitA PosttrainPnPNovelFromPlateToBowlSplitA PosttrainPnPNovelFromPlateToCardboardboxSplitA PosttrainPnPNovelFromPlateToPanSplitA PosttrainPnPNovelFromPlateToPlateSplitA PosttrainPnPNovelFromTrayToCardboardboxSplitA PosttrainPnPNovelFromTrayToPlateSplitA PosttrainPnPNovelFromTrayToPotSplitA PosttrainPnPNovelFromTrayToTieredbasketSplitA PosttrainPnPNovelFromTrayToTieredshelfSplitA"

DS=()
for t in $TASKS; do DS+=("$DATA_ROOT/gr1_unified.$t"); done

cd "$ROOT"
export MASTER_ADDR=10.10.70.153 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0
export GRADIENT_CHECKPOINTING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_SOCKET_IFNAME=enp7s0

"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type golden \
  --dataset-path "${DS[@]}" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:-10]" \
  --num-gpus 1 --batch-size 4 --gradient-accumulation-steps 16 \
  --dataloader-num-workers 2 --max-steps 80000 --save-steps 1000 \
  --base_model_path "$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json" \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --output_dir "$OUT" > "$LOG" 2>&1
echo "REMOTE TRAIN DONE rc=$?" >> "$LOG"
