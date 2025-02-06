import os
from gymnasium import Wrapper
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, Tuple
import rand_dyn_env  # noqa: F401
import gymnasium as gym

class KeepLastWrapper(Wrapper):
    def __init__(self, env, features: list[str] = []):
        super().__init__(env)
        self.features = features
        for feature in features:
            old_space = self.observation_space[feature]
            assert isinstance(old_space, spaces.Box)
            self.observation_space[feature] = spaces.Box(old_space.low[-1], old_space.high[-1], old_space.shape[1:], old_space.dtype)

    def step(self, action):
        obs, rew, term, trunc, infos  = super().step(action)
        self._tranform_obs(obs)
        return obs, rew, term, trunc, infos
    
    def reset(self, *, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        self._tranform_obs(obs)
        return obs, info

    def _tranform_obs(self, obs: dict):
        for feature in self.features:
            obs[feature] = obs[feature][-1]

class ActionFeedbackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) \
            if isinstance(env.action_space, Box) else 0

        # Modify observation space to include last action
        obs_space = env.observation_space
        if isinstance(obs_space, Box):
            low = np.concatenate([obs_space.low, np.full(self.last_action.shape, env.action_space.low.min())])
            high = np.concatenate([obs_space.high, np.full(self.last_action.shape, env.action_space.high.max())])
            self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)
        elif isinstance(obs_space, Discrete):
            self.observation_space = Tuple((obs_space, env.action_space))
        elif isinstance(obs_space, Dict):
            new_spaces = dict(obs_space.spaces)
            new_spaces["last_action"] = env.action_space
            self.observation_space = Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros_like(self.last_action)  # Reset last action
        return self._augment_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = action
        return self._augment_observation(obs), reward, terminated, truncated, info

    def _augment_observation(self, obs):
        if isinstance(self.observation_space, Box):
            return np.concatenate([obs, np.array(self.last_action, dtype=obs.dtype)])
        elif isinstance(self.observation_space, Tuple):
            return (obs, self.last_action)
        elif isinstance(self.observation_space, Dict):
            obs = dict(obs)  # Copy to avoid modifying original
            obs["last_action"] = self.last_action
            return obs
        return obs

class MultiInputTCNExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for dictionary observations:
    - Scalar (vector) features are passed unchanged.
    - Time-series features are processed with Temporal CNN (TCN).
    """

    def __init__(self, observation_space: spaces.Dict):
        super(MultiInputTCNExtractor, self).__init__(observation_space, features_dim=1)  # Temporary

        self.extractors = nn.ModuleDict()
        self.passthrough_keys = []  # Keys that remain unchanged
        total_output_dim = 0  # Final concatenated feature size

        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 1:  # Scalar/Vector features
                self.passthrough_keys.append(key)
                total_output_dim += subspace.shape[0]  # Directly added

            elif len(subspace.shape) == 2:  # Time-series features (N, F)
                n_features = subspace.shape[1]  # Number of input channels (F)

                tcn = nn.Sequential(
                    nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),  # Reduce to shape (batch, 64, 1)
                    nn.Flatten()  # Output shape: (batch, 64)
                )

                self.extractors[key] = tcn
                total_output_dim += 64  # Fixed TCN output per time-series

        self._features_dim = total_output_dim  # Set final output feature dimension

    def forward(self, observations: dict) -> torch.Tensor:
        extracted_features = []

        for key in self.passthrough_keys:
            extracted_features.append(observations[key])  # Directly pass scalar features

        for key, extractor in self.extractors.items():
            x = observations[key]  # Time-series tensor (batch, N, F)
            x = x.permute(0, 2, 1)  # Change to (batch, F, N) for Conv1D
            extracted_features.append(extractor(x))

        return torch.cat(extracted_features, dim=1)  # Concatenate all features


# Custom Policy that integrates the MultiInputTCNExtractor
class MultiInputTCNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, use_sde, **kwargs):
        super(MultiInputTCNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde=use_sde,  # ✅ Pass use_sde explicitly
            features_extractor_class=MultiInputTCNExtractor,  # ✅ Use the custom feature extractor
            **kwargs  # ✅ Pass all remaining policy arguments
        )


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
    

def env_fn():
    Ts = 0.04
    env = gym.make("RandDyn-v0", Ts=Ts)
    env = SaveImageOnResetWrapper(env, "images")
    env = ActionFeedbackWrapper(env)
    env = FrameStackObservation(env, int(60 / 0.04))
    env = KeepLastWrapper(env, ["countdown", "ref"])
    return env


model = PPO(MultiInputTCNPolicy, make_vec_env(env_fn))
model.learn(total_timesteps=1_000_000, progress_bar=True)
