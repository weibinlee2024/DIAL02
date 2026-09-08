#!/usr/bin/env bash
cd /tmp/opencode
for p in $(pgrep -f "torch.distributed.run"); do kill -9 $p 2>/dev/null; done
sleep 1
/home/weibin/miniconda3/envs/dial/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=10.10.70.153 --master_port=29500 --node_rank=0 \
  --max_restarts=0 \
  ddp_smoke.py > /tmp/opencode/torchrun_smoke_local.log 2>&1
echo "SMOKE DONE rc=$?" >> /tmp/opencode/torchrun_smoke_local.log
