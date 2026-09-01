import contextlib
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8192, 8192)

    def forward(self, x):
        return self.lin(x)


acc = Accelerator()
model = M()
model, = acc.prepare(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

rank = dist.get_rank()
print(f"[rank{rank}] distributed_type={acc.distributed_type} "
      f"use_distributed={acc.use_distributed} model={type(model).__name__} "
      f"has_no_sync={hasattr(model, 'no_sync')}", flush=True)

t0 = time.time()
for step in range(4):
    opt.zero_grad()
    for i in range(16):
        x = torch.randn(2, 8192, device="cuda")
        ctx = acc.no_sync(model) if i != 15 else contextlib.nullcontext()
        with ctx():
            loss = model(x).pow(2).mean()
            acc.backward(loss)
    opt.step()
    dist.barrier()
    print(f"[rank{rank}] step {step} done in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
print(f"[rank{rank}] DIAG DONE", flush=True)
