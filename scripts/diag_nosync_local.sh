#!/usr/bin/env bash
cd /home/weibin/DIAL-master/scripts
/home/weibin/miniconda3/envs/dial/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=10.10.70.153 --master_port=29500 --node_rank=0 \
  --max_restarts=0 \
  diag_nosync.py > /tmp/opencode/diag_local.log 2>&1
echo "DIAG LOCAL DONE rc=$?" >> /tmp/opencode/diag_local.log
