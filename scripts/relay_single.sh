#!/usr/bin/env bash
# Relay watchdog: Stage A(40k done) -> Stage B(80k pretrain) -> Stage C(40k finetune)
# Runs in background; checks every 300s.
set -u
ROOT=/home/weibin/DIAL-master
PRETRAIN_DIR=$ROOT/outputs/pretrain-golden-single
LOG=/tmp/opencode/relay.log

check_idle() {
  ps aux | grep -E "python -u scripts/dual_system_train" | grep -v grep | grep -v relay > /dev/null 2>&1
  return $?
}

echo "[relay] started $(date)" >> "$LOG"
while true; do
  # Stage C trigger: pretrain 80k checkpoint exists AND no training running
  if [ -d "$PRETRAIN_DIR/checkpoint-80000" ]; then
    if ! check_idle; then
      echo "[relay] 80k done, launching Stage C (finetune 40k) $(date)" >> "$LOG"
      setsid nohup bash "$ROOT/scripts/run_1gpu_finetune_single.sh" > /dev/null 2>&1 < /dev/null &
      exit 0
    fi
  # Stage B trigger: pretrain 40k checkpoint exists AND no training running
  elif [ -d "$PRETRAIN_DIR/checkpoint-40000" ]; then
    if ! check_idle; then
      echo "[relay] 40k done, launching Stage B (resume to 80k) $(date)" >> "$LOG"
      setsid nohup bash "$ROOT/scripts/run_1gpu_pretrain_single_80k.sh" > /dev/null 2>&1 < /dev/null &
      exit 0
    fi
  fi
  sleep 300
done
