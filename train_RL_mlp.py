import os

from gymnasium import Wrapper
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
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
        out = self.env.reset(**kwargs)

        # If the environment supports rendering before reset, save the frame
        frame = self.env.unwrapped._frame
        self.episode_counter += 1

        if frame is not None and (self.episode_counter % self.every) == 0:
            # Save the frame as an image
            image = Image.fromarray(frame)
            image.save(
                os.path.join(self.save_path, f"episode_{self.episode_counter}.png")
            )

        # Call the original reset method
        return out


# Custom callback for logging returns
class ReturnLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if an episode is done
        if "episode" in self.locals.get("infos", [{}])[0]:
            reward = self.locals["infos"][0]["episode"]["r"]  # Episode return
            self.episode_rewards.append(reward)
            if self.verbose > 0:
                print(f"Episode return: {np.mean(self.episode_rewards[-10:])}")
        return True


env_fn = lambda: SaveImageOnResetWrapper(
    Monitor(
        gym.make(
            "RandDyn-v0",
            on_reset_draw_motion=True,
            on_reset_draw_sys=True,
        )
    ),
    "images",
)

env = VecFrameStack(make_vec_env(env_fn, 1), 6)

# Create callbacks
return_callback = ReturnLoggingCallback(verbose=1)
progress_bar = ProgressBarCallback()
callback = CallbackList([return_callback, progress_bar])

model = PPO("MultiInputPolicy", env)
model.learn(total_timesteps=1_000_000, callback=callback)
