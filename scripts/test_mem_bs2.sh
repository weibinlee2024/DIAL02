#!/usr/bin/env bash
# GPU memory self-check: run 10 training steps with bs=2 + MODEL_BF16=1
# Prints peak memory. If peak < 20GB, 24GB cards are safe for bs=2/accum=32.
set -u
ROOT=/home/weibin/DIAL-master
PYTHON=/home/weibin/miniconda3/envs/dial/bin/python
LOG=/tmp/opencode/mem_test.log
export GRADIENT_CHECKPOINTING=1
export MODEL_BF16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT"
"$PYTHON" -u scripts/dual_system_train.py \
  --bridge_type golden \
  --dataset-path "$ROOT/Datasets/LeRobot-AugPosRot-Correct/gr1_unified.PnPCanToDrawerClose" \
  --data-config fourier_gr1_arms_waist_aug_pos_rot_flip_wrist_only_gausNorm_crop \
  --embodiment_tag gr1 \
  --data_split "[:10]" \
  --num-gpus 1 --batch-size 2 --gradient-accumulation-steps 1 \
  --dataloader-num-workers 2 --max-steps 10 --save-steps 100000 \
  --base_model_path "$ROOT/gr00t/model/configs/gr00t_n1.5_dial_augPosRot.json" \
  --compute-bridge-loss --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-tune-bridge-visual --no-tune-bridge-goal --tune-bridge-embedding \
  --select_layer 12 --use_image_type_embedding --ignore_lang_prefix \
  --report_to tensorboard \
  --output_dir /tmp/opencode/mem_test_out > "$LOG" 2>&1
echo "MEM_TEST_DONE rc=$?" >> "$LOG"
PEAK=$(/home/weibin/miniconda3/envs/dial/bin/python -c "
import re
d=open('$LOG',errors='replace').read()
m=re.findall(r'peak memory usage: ([0-9.]+) GiB|([0-9.]+) GiB', d)
print('peak_giB:', m[-1] if m else 'unknown')
" 2>/dev/null)
echo "PEAK_MEM $PEAK" >> "$LOG"
