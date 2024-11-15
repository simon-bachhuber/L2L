from typing import Optional

import control as ctrl
import gymnasium as gym
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from rss import rss


class RandDynEnv(gym.Env):
    metadata = dict(render_modes=["rgb_array"])
    render_mode = "rgb_array"

    def __init__(
        self,
        state_dim_min: int = 3,
        state_dim_max: int = 3,
        T: float = 60,
        Ts: float = 0.02,
        n_inputs: int = 1,
        n_outputs: int = 1,
        reward_ramp: float = 1.0,
        on_reset_draw_sys: bool = True,
        on_reset_draw_motion: bool = True,
        on_reset_draw_transition_time: bool = True,
        transition_time: float = 30,
        action_limit: float = 1.0,
    ):
        self.state_dim_min = state_dim_min
        self.state_dim_max = state_dim_max
        self.T = T
        self.Ts = Ts
        self.ts = np.arange(T, step=Ts)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.reward_ramp = reward_ramp
        self.on_reset_draw_sys = on_reset_draw_sys
        self.on_reset_draw_motion = on_reset_draw_motion
        self.on_reset_draw_transition_time = on_reset_draw_transition_time
        self._ss = None
        self._ref = None
        self.transition_time = np.array([transition_time])
        self._obss = None
        self._frame = None

        assert (
            n_inputs == 1
        ), ">1 is not yet supported, GaussianProcessRegressor doesn't work"

        assert action_limit > 0.0

        self.action_space = gym.spaces.Box(
            low=-action_limit, high=action_limit, shape=(self.n_inputs,)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "countdown": gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([0.8 * T]),
                    shape=(1,),
                    dtype=np.float64,
                ),
                "ref": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_outputs,), dtype=np.float64
                ),
                "obs": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_outputs,), dtype=np.float64
                ),
            }
        )

    def draw_rand_system(self):
        state_dim = self.np_random.integers(self.state_dim_min, self.state_dim_max + 1)

        valid_sys = False
        while not valid_sys:
            # this makes it be continuous time
            self._ss = rss(
                self.np_random, state_dim, self.n_outputs, self.n_inputs, dt=0
            )

            A, B, C = self._ss.A, self._ss.B, self._ss.C
            # Check controllability
            ctrb_matrix = ctrl.ctrb(A, B)
            rank_ctrb = np.linalg.matrix_rank(ctrb_matrix)
            controllable = rank_ctrb == A.shape[0]

            # Check observability
            obsv_matrix = ctrl.obsv(A, C)
            rank_obsv = np.linalg.matrix_rank(obsv_matrix)
            observable = rank_obsv == A.shape[0]

            valid_sys = observable and controllable

    def draw_rand_motion(self):
        seed = self.np_random.integers(0, int(1e8))
        us = GaussianProcessRegressor(kernel=0.15 * RBF(0.75)).sample_y(
            self.ts[:, np.newaxis], random_state=seed
        )
        us = (us - np.mean(us)) / np.std(us)
        us *= 0.25

        # apply action limits such that reference motion *is feasible*
        us = np.clip(us, self.action_space.low, self.action_space.high)

        yout = ctrl.forced_response(self._ss, T=self.ts, U=us.T).outputs
        self._us = us
        self._ref = yout[:, None]

    def draw_rand_transition_time(self):
        self.transition_time = self.np_random.uniform(
            0.2 * self.T, 0.8 * self.T, size=(1,)
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._frame = self.render()

        if self.on_reset_draw_sys or self._ss is None:
            self.draw_rand_system()
        if self.on_reset_draw_motion or self._ref is None:
            self.draw_rand_motion()
        if self.on_reset_draw_transition_time:
            self.draw_rand_transition_time()
        self._t = 0
        self._x = np.zeros((self._ss.A.shape[0],))
        self._u = np.zeros((self.n_inputs,))

        observation = self._get_obs()
        info = self._get_info()

        # only for rendering
        self._obss = [observation["obs"]]

        return observation, info

    def step(self, action):
        self._t += 1

        self._u = action
        self._simulate_one_timestep()

        truncated = self.ts[self._t] >= (self.T - 1)
        terminated = truncated
        obs = self._get_obs()
        info = self._get_info()
        reward = -np.linalg.norm(obs["ref"] - obs["obs"])
        # smoothen the step of the reward function
        cd = obs["countdown"]
        if cd > self.reward_ramp:
            reward = 0.0
        else:
            reward *= (self.reward_ramp - cd) / self.reward_ramp

        # only done for rendering
        self._obss.append(obs["obs"])

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "countdown": np.clip(
                self.transition_time - self.ts[self._t][None], a_max=None, a_min=0
            ),
            "ref": self._ref[self._t],
            "obs": self._ss.C @ self._x + self._ss.D @ self._u,
        }

    def _get_info(self):
        return {"x": self._x}

    def _simulate_one_timestep(self):
        rhs = lambda t, x: self._ss.A @ x + self._ss.B @ self._u
        self._x = self._runge_kutta(rhs, 0, self._x, self.Ts)

    @staticmethod
    def _runge_kutta(rhs, t, x, dt):
        h = dt
        k1 = rhs(t, x)
        k2 = rhs(t + h / 2, x + h * k1 / 2)
        k3 = rhs(t + h / 2, x + h * k2 / 2)
        k4 = rhs(t + h, x + h * k3)
        dx = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x + dt * dx

    def _update_figure(self):
        self._fig.gca().scatter(self.ts[self._t], self._get_obs()["obs"], c="black")

    def render(self):
        if self.render_mode != "rgb_array":
            raise NotImplementedError

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.set_xlabel("Time [s]")

        if self._ref is not None and self._obss is not None:
            ax.plot(self.ts, self._ref, label="ref")
            obs = np.vstack(self._obss)
            ax.plot(self.ts[: len(obs)], obs, label="obs")
            ax.axvline(
                x=self.transition_time, color="red", linestyle="--", label="transition"
            )
            ax.legend()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return frame


gym.register(id="RandDyn-v0", entry_point=RandDynEnv)
