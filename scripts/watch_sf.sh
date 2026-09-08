#!/usr/bin/env bash
# SF 训练实时监控（循环刷新）。用法: bash scripts/watch_sf.sh [finetune|pretrain] [间隔秒]
set -u
MODE="${1:-finetune}"
INTERVAL="${2:-10}"
case "$MODE" in
  finetune) LOG=/tmp/opencode/finetune_sf_80k.log; CKPT=/home/weibin/DIAL-master/outputs/finetune-end2end-sf-80k ;;
  pretrain) LOG=/tmp/opencode/pretrain_sf_80k.log;  CKPT=/home/weibin/DIAL-master/outputs/pretrain-golden-sf-80k ;;
  *) echo "usage: $0 [finetune|pretrain] [sec]"; exit 1 ;;
esac

while true; do
  clear
  echo "==== SF-$MODE $(date '+%F %H:%M:%S') ===="
  echo "--- 进度 ---"
  tr '\r' '\n' < "$LOG" | grep -a -o "[0-9]*/80000 \[[0-9:]*<[^]]*\]" | tail -1
  echo "--- loss (bridge/action/state_future) 最近3条 ---"
  tr '\r' '\n' < "$LOG" | grep -a -o "{'bridge_loss': [0-9.]*, 'action_loss': [0-9.]*, 'state_future_loss': [0-9.]*" | tail -3
  echo "--- GPU / 进程 ---"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  ps aux | grep "[d]ual_system_train" | wc -l | awk '{print "训练进程数:", $1}'
  echo "--- checkpoint ---"
  ls "$CKPT" 2>/dev/null | grep "^checkpoint" | tr '\n' ' '; echo
  sleep "$INTERVAL"
done
