#!/usr/bin/env python
# Regenerate AugPosRot-Correct episode_000701 (161-dim) from raw(44-dim) + replay EEF(157 frames)
# EEF is linearly interpolated from the replay frame grid onto the raw timestamp grid.
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUG = Path('/home/weibin/DIAL-master/Datasets/LeRobot-AugPosRot-Correct/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA')
RAW = Path('/home/weibin/DIAL-master/Datasets/raw/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA')
REPLAY = Path('/tmp/opencode/replay_701/parquet/demo_701.parquet')

raw = pd.read_parquet(RAW / 'data/chunk-000/episode_000701.parquet')
rep = pd.read_parquet(REPLAY)

# EEF columns in modality.json order == replay column order
eef_cols = [c for c in rep.columns if c != 'timestep']
raw_t = np.array(raw['timestamp']).flatten()
T = float(raw_t[-1])
n_raw = len(raw)
n_rep = len(rep)
print(f'raw rows={n_raw} (t_end={T:.2f}s), replay rows={n_rep}, eef_cols={len(eef_cols)}')

rep_t = np.arange(n_rep) * (T / max(1, n_rep - 1))
# build (n_rep, D) EEF matrix in col order (each cell is a list)
rows = []
for i in range(n_rep):
    rows.append(np.concatenate([np.asarray(rep[c].iloc[i]).flatten() for c in eef_cols]))
eef = np.array(rows).astype(np.float64)
D = eef.shape[1]
print(f'eef matrix {eef.shape}, total dim={D}')
# interpolate every dim onto raw grid
eef_i = np.stack([np.interp(raw_t, rep_t, eef[:, d]) for d in range(D)], axis=1)  # (n_raw, D)

# state/action: raw part unchanged + EEF appended (action uses next-frame EEF, last padded)
state_list, action_list = [], []
for i in range(n_raw):
    s_raw = np.asarray(raw['observation.state'].iloc[i]).flatten().astype(np.float64)
    a_raw = np.asarray(raw['action'].iloc[i]).flatten().astype(np.float64)
    j = min(i + 1, n_raw - 1)
    state_list.append(np.concatenate([s_raw, eef_i[i]]))
    action_list.append(np.concatenate([a_raw, eef_i[j]]))
print('state dim:', len(state_list[0]), 'action dim:', len(action_list[0]))

out = pd.DataFrame({
    'observation.state': state_list,
    'action': action_list,
    'timestamp': raw['timestamp'],
    'next.reward': raw['next.reward'],
    'next.done': raw['next.done'],
    'task_index': raw['task_index'],
    'annotation.human.fine_action': raw['annotation.human.fine_action'],
    'annotation.human.coarse_action': raw['annotation.human.coarse_action'],
    'episode_index': raw['episode_index'],
    'index': raw['index'],
})
out['observation.state'] = out['observation.state'].apply(lambda x: np.asarray(x, dtype=np.float64))
out['action'] = out['action'].apply(lambda x: np.asarray(x, dtype=np.float64))
dst = AUG / 'data/chunk-000/episode_000701.parquet'
out.to_parquet(dst, index=False)
print('saved:', dst)

# verify vs neighbor 700
d700 = pd.read_parquet(AUG / 'data/chunk-000/episode_000700.parquet')
s701 = out['observation.state'].iloc[0]
s700 = d700['observation.state'].iloc[0]
print('701 dim:', len(s701), '700 dim:', len(s700))
print('701[44:50]:', np.round(s701[44:50], 4))
print('700[44:50]:', np.round(np.asarray(s700)[44:50], 4))
