import os

from gymnasium import Wrapper
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

import rand_dyn_env  # noqa: F401


class SaveImageOnResetWrapper(Wrapper):
    def __init__(self, env, save_path: str, every: int = 5):
        super(SaveImageOnResetWrapper, self).__init__(env)
        self.save_path = save_path
        self.episode_counter = 0
        self.every = every
        os.makedirs(self.save_path, exist_ok=True)  # Ensure the directory exists

    def reset(self, **kwargs):
        self.episode_counter += 1

        if (self.episode_counter % self.every) == 0:
            image = Image.fromarray(self.env.render())
            image.save(
                os.path.join(self.save_path, f"episode_{self.episode_counter}.png")
            )
        # Call the original reset method
        return self.env.reset(**kwargs)


env_fn = lambda: SaveImageOnResetWrapper(FlattenObservation(gym.make("RandDyn-v0")), "images")

env = VecFrameStack(make_vec_env(env_fn, 1), 6)
model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64]*3), device="cpu")
model.learn(total_timesteps=1_000_000, progress_bar=True)
