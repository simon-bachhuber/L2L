from datetime import datetime

import fire
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers import NormalizeReward
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

import rand_dyn_env  # noqa: F401


def generate_filename():
    """Generates a filename based on the current timestamp."""
    return datetime.now().strftime("file_%Y%m%d_%H%M%S")


class RenderToTensorboardCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.writer = None  # Will be initialized when training starts

    def _on_training_start(self) -> None:
        """Manually initialize TensorBoard SummaryWriter."""
        log_dir = self.logger.get_dir()  # Get SB3 log directory
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            raise ValueError("TensorBoard log directory not found!")

    def _on_step(self) -> bool:
        """Every `log_freq` steps, run a single episode and log a rendered frame."""
        if self.num_timesteps % self.log_freq == 0:
            print(f"Logging image at step {self.num_timesteps}")

            obs, _ = self.eval_env.reset()
            done = False
            state = None
            frame = None

            while not done:
                action, state = self.model.predict(obs, state=state, deterministic=True)
                obs, _, done, _, _ = self.eval_env.step(action)

            frame = self.eval_env.render()  # Get RGB frame

            # Ensure the writer is properly initialized
            if self.writer and frame is not None:
                frame = np.transpose(frame, (2, 0, 1))  # Convert to CHW format
                self.writer.add_image("render/eval", frame, self.num_timesteps)
            else:
                print("Warning: Writer is None or frame is invalid.")

        return True  # Continue training


def env_fn(on_reset_draw: bool = True):
    return NormalizeReward(
        NormalizeObservation(
            FlattenObservation(
                gym.make(
                    "RandDyn-v0",
                    on_reset_draw_motion=on_reset_draw,
                    on_reset_draw_sys=on_reset_draw,
                    on_reset_draw_transition_time=on_reset_draw,
                    draw_random_motion_method="rff",
                )
            )
        )
    )


def main(
    device: str,
    n_envs: int,
    n_lstm_layers: int = 1,
    lstm_hidden_size: int = 256,
    total_timesteps: int = 1_000_000,
    seed: int = 1,
    lr: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        tensorboard_log="tensorboard_logs",
        device=device,
        seed=1000 + seed,
        policy_kwargs=dict(
            n_lstm_layers=n_lstm_layers, lstm_hidden_size=lstm_hidden_size
        ),
    )
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=RenderToTensorboardCallback(
            env_fn(on_reset_draw=False), log_freq=10_000
        ),
    )
    model.save(path=generate_filename())


if __name__ == "__main__":
    fire.Fire(main)
