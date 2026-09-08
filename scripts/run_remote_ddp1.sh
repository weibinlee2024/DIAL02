#!/usr/bin/env bash
for p in $(pgrep -f "python -u /tmp/ddp_smoke.py"); do kill -9 $p 2>/dev/null; done
sleep 1
MASTER_ADDR=10.10.70.153 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \
  /home/weibin/anaconda3/envs/dial/bin/python -u /tmp/ddp_smoke.py > /tmp/ddp_remote.log 2>&1
echo "REMOTE DONE" >> /tmp/ddp_remote.log
