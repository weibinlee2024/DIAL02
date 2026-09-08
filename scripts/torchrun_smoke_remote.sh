#!/usr/bin/env bash
cd /tmp
for p in $(pgrep -f "torch.distributed.run"); do kill -9 $p 2>/dev/null; done
sleep 1
/home/weibin/anaconda3/envs/dial/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=10.10.70.153 --master_port=29500 --node_rank=1 \
  --max_restarts=0 \
  /tmp/ddp_smoke.py > /tmp/torchrun_smoke_remote.log 2>&1
echo "SMOKE DONE rc=$?" >> /tmp/torchrun_smoke_remote.log
