import sys
sys.path.insert(0, '/home/weibin/DIAL-master')

import gymnasium as gym
from pathlib import Path
import numpy as np
import robocasa
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401  (triggers env registration)
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecordingWrapper, VideoRecorder)
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper

env_name = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
print(f"creating env: {env_name}", flush=True)
env = gym.make(env_name, enable_render=True)

video_recorder = VideoRecorder.create_h264(
    fps=20,
    codec="h264",
    input_pix_fmt="bgr24",
    crf=23,
    thread_type="FRAME",
    thread_count=2,
)
env = VideoRecordingWrapper(
    env, video_recorder,
    video_dir=Path("outputs/viz_test"),
    steps_per_render=1,
    state_modality_keys=["state.right_arm", "state.left_arm"],
)
env = MultiStepWrapper(
    env,
    video_delta_indices=np.array([0]),
    state_delta_indices=np.array([0]),
    n_action_steps=4,
    max_episode_steps=600,
    state_modality_keys=["state.right_arm", "state.left_arm"],
)

obs, info = env.reset()
print("reset OK, obs keys:", list(obs.keys()), flush=True)
print("action space:", env.action_space, flush=True)

# run 10 steps with dummy (zero) actions
for i in range(10):
    action = {k: np.zeros(v.shape, dtype=np.float32) for k, v in env.action_space.items()}
    obs, rew, term, trunc, info = env.step(action)
    if i in (0, 4, 9):
        print(f"step {i}: reward={rew} term={term} trunc={trunc}", flush=True)

env.close()
print("VIZ TEST DONE", flush=True)
