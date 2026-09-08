import sys
sys.path.insert(0, '/home/weibin/DIAL-master')

import gymnasium as gym
import robocasa
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: triggers registration

TASKS = [
    "gr1_unified/EvalPnPNovelFromPlacematToBowlSplitB_GR1ArmsAndWaistFourierHands_Env",   # OOD 外观
    "gr1_unified/PretrainPnPNovelFromPlacematToPanSplitA_GR1ArmsAndWaistFourierHands_Env",  # OOD 容器
    "gr1_unified/PretrainPnPBaseFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env", # OOD 物体类型
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env", # ID
]
for t in TASKS:
    try:
        env = gym.make(t, enable_render=True)
        obs, info = env.reset()
        env.close()
        print(f"[OK] {t.split('/')[-1]}")
    except Exception as e:
        print(f"[FAIL] {t.split('/')[-1]}: {type(e).__name__}: {str(e)[:150]}")
print("CHECK DONE")
