#!/usr/bin/env bash
for p in $(pgrep -f "python -u /tmp/opencode/ddp_smoke.py"); do kill -9 $p 2>/dev/null; done
sleep 1
MASTER_ADDR=10.10.70.153 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \
  /home/weibin/miniconda3/envs/dial/bin/python -u /tmp/opencode/ddp_smoke.py > /tmp/opencode/ddp_local.log 2>&1
echo "LOCAL DONE" >> /tmp/opencode/ddp_local.log
