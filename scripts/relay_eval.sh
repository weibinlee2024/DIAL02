#!/usr/bin/env bash
# Relay eval: after finetune 80k finishes -> offline eval -> online smoke -> full ID eval
set -u
ROOT=/home/weibin/DIAL-master
CKPT=$ROOT/outputs/finetune-end2end-single/checkpoint-80000
LOG=/tmp/opencode/relay_eval.log
N_EPISODES_FULL=${N_EPISODES_FULL:-50}
SMOKE_TASK=${SMOKE_TASK:-"PnPCupToDrawerClose"}

echo "[relay_eval] started $(date)" >> "$LOG"

# wait for finetune to finish: checkpoint-80000 exists AND no training process
while true; do
  if [ -d "$CKPT" ]; then
    if ! ps aux | grep -E "python -u scripts/dual_system_train" | grep -v grep > /dev/null 2>&1; then
      echo "[relay_eval] finetune done, starting validation $(date)" >> "$LOG"
      break
    fi
  fi
  sleep 300
done

cd "$ROOT"

# ---- Step 0: GPU memory self-check (bs=2 + MODEL_BF16) ----
echo "[relay_eval] Step0 mem test bs=2+MODEL_BF16 $(date)" >> "$LOG"
bash scripts/test_mem_bs2.sh >> "$LOG" 2>&1
PEAK=$(grep -ao "max_memory[^ ]*\|[0-9.]* GB" "$LOG" | tail -3)
echo "[relay_eval] mem test peak: $PEAK" >> "$LOG"

# ---- Step 1: offline eval (action MSE + bridge loss + PCA) ----
echo "[relay_eval] Step1 offline eval $(date)" >> "$LOG"
bash examples/eval_loss.sh "$CKPT" >> "$LOG" 2>&1 || echo "[relay_eval] offline eval failed rc=$?" >> "$LOG"

# ---- Step 2: online smoke (1 task x 5 episodes) ----
echo "[relay_eval] Step2 online smoke: $SMOKE_TASK x5 $(date)" >> "$LOG"
N_EPISODES=5 PORT=50052 bash examples/eval.sh "$CKPT" id >> "$LOG" 2>&1 \
  && echo "[relay_eval] smoke done" >> "$LOG" \
  || echo "[relay_eval] smoke failed rc=$?" >> "$LOG"

# ---- Step 3: full ID eval (24 tasks x N episodes) ----
echo "[relay_eval] Step3 full ID eval: ${N_EPISODES_FULL} eps/task $(date)" >> "$LOG"
N_EPISODES=$N_EPISODES_FULL PORT=50051 bash examples/eval.sh "$CKPT" id >> "$LOG" 2>&1 \
  && echo "[relay_eval] full ID done" >> "$LOG" \
  || echo "[relay_eval] full ID failed rc=$?" >> "$LOG"

echo "[relay_eval] ALL DONE $(date)" >> "$LOG"
