#!/usr/bin/env bash
cd /home/weibin/projects3/DIAL-master_Libero/scripts
/home/weibin/anaconda3/envs/dial/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 \
  --master_addr=10.10.70.153 --master_port=29500 --node_rank=1 \
  --max_restarts=0 \
  diag_nosync.py > /tmp/diag_remote.log 2>&1
echo "DIAG REMOTE DONE rc=$?" >> /tmp/diag_remote.log
