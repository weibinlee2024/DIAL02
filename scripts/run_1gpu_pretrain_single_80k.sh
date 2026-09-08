#!/usr/bin/env bash
# Stage B: continue pretrain 40k -> 80k (resume from checkpoint-40000, same output dir)
set -u
ROOT=/home/weibin/DIAL-master
DATA_ROOT=$ROOT/Datasets/LeRobot-AugPosRot-Correct
PYTHON=/home/weibin/miniconda3/envs/dial/bin/python
OUT=$ROOT/outputs/pretrain-golden-single
LOG=/tmp/opencode/pretrain_single_80k.log
mkdir -p "$OUT"

TASKS="PnPBottleToCabinetClose PnPCanToDrawerClose PnPCupToDrawerClose PnPMilkToMicrowaveClose PnPPotatoToMicrowaveClose PnPWineToCabinetClose PosttrainPnPNovelFromCuttingboardToBasketSplitA PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA PosttrainPnPNovelFromCuttingboardToPanSplitA PosttrainPnPNovelFromCuttingboardToPotSplitA PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA PosttrainPnPNovelFromPlacematToBasketSplitA PosttrainPnPNovelFromPlacematToBowlSplitA PosttrainPnPNovelFromPlacematToPlateSplitA PosttrainPnPNovelFromPlacematToTieredshelfSplitA PosttrainPnPNovelFromPlateToBowlSplitA PosttrainPnPNovelFromPlateToCardboardboxSplitA PosttrainPnPNovelFromPlateToPanSplitA PosttrainPnPNovelFromPlateToPlateSplitA PosttrainPnPNovelFromTrayToCardboardboxSplitA PosttrainPnPNovelFromTrayToPlateSplitA PosttrainPnPNovelFromTrayToPotSplitA PosttrainPnPNovelFromTrayToTieredbasketSplitA PosttrainPnPNovelFromTrayToTieredshelfSplitA"

DS=()
for t in $TASKS; do DS+=("$DATA_ROOT/gr1_unified.$t"); done

cd "$ROOT"
export GRADIENT_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --resume \
  --output_dir "$OUT" > "$LOG" 2>&1
echo "STAGE-B DONE rc=$?" >> "$LOG"
