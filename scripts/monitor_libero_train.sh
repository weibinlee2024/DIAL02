#!/usr/bin/env bash
# Live / one-shot DIAL LIBERO training observer.
#
# Usage (from repo root):
#   bash scripts/monitor_libero_train.sh              # one-shot summary (latest log)
#   bash scripts/monitor_libero_train.sh watch         # refresh every 10s
#   bash scripts/monitor_libero_train.sh watch 5       # refresh every 5s
#   bash scripts/monitor_libero_train.sh csv            # also dump CSV under outputs/logs/
#   bash scripts/monitor_libero_train.sh tail           # raw log follow + nvidia-smi
#   bash scripts/monitor_libero_train.sh tb             # print tensorboard launch cmd
#
# Env overrides:
#   LOG=.../libero_pretrain_bs4_....log
#   OUT_DIR=/media/weibin/weibin02/DIAL/outputs/libero-pretrain-decoupled-bs4

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /home/weibin/anaconda3/envs/dial/bin/python ]]; then
    PY=/home/weibin/anaconda3/envs/dial/bin/python
  else
    PY=python3
  fi
fi

MODE="${1:-once}"
ARG2="${2:-}"

# Prefer active bs4 pretrain log if LOG not set
if [[ -z "${LOG:-}" ]]; then
  LOG="$(ls -t outputs/logs/libero_pretrain_bs4_*.log 2>/dev/null | head -1 || true)"
  if [[ -z "$LOG" ]]; then
    LOG="$(ls -t outputs/logs/libero_pretrain_*.log 2>/dev/null | head -1 || true)"
  fi
fi

OUT_DIR="${OUT_DIR:-/media/weibin/weibin02/DIAL/outputs/libero-pretrain-decoupled-bs4}"
TS="$(date +%Y%m%d_%H%M%S)"

common=(scripts/monitor_libero_train.py)
[[ -n "${LOG:-}" ]] && common+=(--log "$LOG")
[[ -d "$OUT_DIR" ]] && common+=(--out-dir "$OUT_DIR")

case "$MODE" in
  once|"")
    "$PY" "${common[@]}" --recent 15
    ;;
  watch)
    interval="${ARG2:-10}"
    "$PY" "${common[@]}" --watch "$interval" --recent 15
    ;;
  csv)
    csv_path="outputs/logs/libero_metrics_${TS}.csv"
    micro_path="outputs/logs/libero_microbatch_${TS}.csv"
    json_path="outputs/logs/libero_summary_${TS}.json"
    "$PY" "${common[@]}" \
      --csv "$csv_path" \
      --csv-micro "$micro_path" \
      --json-summary "$json_path" \
      --recent 20
    echo
    echo "CSV optimizer steps : $csv_path"
    echo "CSV microbatches    : $micro_path"
    echo "JSON summary        : $json_path"
    ;;
  tail)
    echo "=== nvidia-smi ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
    echo
    echo "=== process ==="
    pgrep -af dual_system_train | grep -v pgrep || echo "(no train process)"
    echo
    if [[ -z "${LOG:-}" ]]; then
      echo "ERROR: no log found under outputs/logs/libero_pretrain*.log"
      exit 1
    fi
    echo "=== tail -F $LOG ==="
    # Strip tqdm carriage returns for readability
    tail -F "$LOG" | tr '\r' '\n'
    ;;
  tb)
    runs="$OUT_DIR/runs"
    echo "TensorBoard event dir: $runs"
    if [[ -d "$runs" ]]; then
      ls -la "$runs" | head -20
    else
      echo "(runs/ not created yet)"
    fi
    echo
    echo "Launch:"
    echo "  tensorboard --logdir $runs --port 6006 --bind_all"
    echo "Then open http://<host>:6006"
    ;;
  *)
    cat <<EOF
Unknown mode: $MODE

Usage:
  bash scripts/monitor_libero_train.sh [once|watch|csv|tail|tb] [watch_interval]
EOF
    exit 1
    ;;
esac
