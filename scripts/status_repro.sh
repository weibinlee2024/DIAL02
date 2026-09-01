#!/usr/bin/env bash
# Quick status for DIAL reproduction jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="${DIAL_EXT_ROOT:-/media/weibin/C4E60209E601FC84/DIAL}"

echo "=== env ==="
conda info --envs 2>/dev/null | rg 'dial|^\*' || true
echo "=== links ==="
ls -l "$ROOT/data" "$ROOT/checkpoints" "$ROOT/outputs" 2>/dev/null || true
echo "=== dataset ==="
du -sh "$EXT/datasets/raw/PhysicalAI-Robotics-GR00T-Teleop-Sim" 2>/dev/null || echo "(no raw yet)"
du -sh "$EXT/datasets/LeRobot-AugPosRot-Correct" 2>/dev/null || echo "(no aug yet)"
if [[ -f "$EXT/datasets/download_gr1.pid" ]]; then
  pid=$(cat "$EXT/datasets/download_gr1.pid")
  if ps -p "$pid" >/dev/null 2>&1; then echo "download RUNNING pid=$pid"; else echo "download pid=$pid not running"; fi
  tail -5 "$EXT/datasets/download_gr1.log" 2>/dev/null || true
fi
echo "=== checkpoints ==="
ls "$EXT/checkpoints" 2>/dev/null | head
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
echo "=== disk ==="
df -h /home/weibin "$EXT" | tail -n +1
