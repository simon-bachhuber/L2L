import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import rand_dyn_env  # noqa: F401

env = gym.make(
    "RandDyn-v0",
    on_reset_draw_motion=False,
    on_reset_draw_sys=False,
)
env = VecFrameStack(make_vec_env(env, 1), 6)
model = PPO("MultiInputPolicy", env)
model.learn(total_timesteps=1_000_000)
