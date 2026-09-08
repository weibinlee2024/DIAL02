import sys
sys.path.insert(0, '/home/weibin/DIAL-master')

import gymnasium as gym
import robocasa
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv

print(f"robocasa={robocasa.__version__} robosuite={robosuite.__version__}", flush=True)
print("assets dir:", robocasa.__path__[0] + "/models/assets", flush=True)

env_name = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
print(f"creating env: {env_name}", flush=True)
env = gym.make(env_name, enable_render=True)
print("env created:", type(env).__name__, flush=True)
obs, info = env.reset()
print("reset OK, obs keys:", list(obs.keys())[:8], flush=True)
env.close()
print("ENV TEST PASSED", flush=True)
