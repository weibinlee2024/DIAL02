import os
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel as DDP

print("=== PHASE 1: Accelerator state ===", flush=True)
acc = Accelerator()
print(f"distributed_type={acc.distributed_type} use_distributed={acc.use_distributed} "
      f"dist_initialized={dist.is_initialized()} world={dist.get_world_size() if dist.is_initialized() else 0}",
      flush=True)

print("=== PHASE 2: prepare model ===", flush=True)
m = nn.Linear(512, 512)
m, = acc.prepare(m)
print(f"model type after prepare: {type(m).__name__} has_no_sync={hasattr(m, 'no_sync')}", flush=True)

print("=== PHASE 3: no_sync check ===", flush=True)
import contextlib
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
for step in range(2):
    opt.zero_grad()
    for i in range(8):
        x = torch.randn(4, 512, device="cuda")
        ctx = acc.no_sync(m) if i != 7 else contextlib.nullcontext()
        with ctx():
            loss = m(x).pow(2).mean()
            acc.backward(loss)
    opt.step()
    dist.barrier()
    print(f"step {step} ok", flush=True)
print("DIAG2 DONE", flush=True)
