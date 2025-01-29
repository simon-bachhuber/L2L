import math
from typing import Optional

from control import StateSpace
import control as ctrl
from control.iosys import _process_iosys_keywords
from control.iosys import _process_signal_list
import gymnasium as gym
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class RandDynEnv(gym.Env):
    metadata = dict(render_modes=["rgb_array"], render_fps=None)
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
        draw_random_motion_method: str = "gp",
    ):
        """
        Initialize a randomized dynamic environment for reinforcement learning.

        Parameters
        ----------
        state_dim_min : int, optional (default=3)
            Minimum number of state variables for the generated system.

        state_dim_max : int, optional (default=3)
            Maximum number of state variables for the generated system.

        T : float, optional (default=60)
            Total duration of the environment in seconds.

        Ts : float, optional (default=0.02)
            Sampling time step in seconds.

        n_inputs : int, optional (default=1)
            Number of control inputs for the system.

        n_outputs : int, optional (default=1)
            Number of outputs for the system.

        reward_ramp : float, optional (default=1.0)
            Time in seconds over which the reward function ramps up after a transition.

        on_reset_draw_sys : bool, optional (default=True)
            If True, draw a new random system upon environment reset.

        on_reset_draw_motion : bool, optional (default=True)
            If True, draw a new random reference motion upon environment reset.

        on_reset_draw_transition_time : bool, optional (default=True)
            If True, draw a new random transition time upon environment reset.

        transition_time : float, optional (default=30)
            Initial transition time in seconds, used when `on_reset_draw_transition_time` is False.

        action_limit : float, optional (default=1.0)
            Upper and lower limit for actions. Actions are clipped to [-action_limit, action_limit].
        draw_random_motion_method : str, optional (default=gp)
            Specifies the method that is used to generate a random feedforward signal `us` which is
            then applied to the system to generate a feasible reference trajectory. Possible values
            are `gp` (Gaussian Process), `rff` (Random Fourier Features), `lpf-noise` (Low-Pass-Filtered-Whitenoise),
            and `ou-noise` (Ornstein-Uhlenbeck Process)

        Raises
        ------
        AssertionError
            If `n_inputs` is greater than 1, as currently only single-input systems are supported.
            If `action_limit` is not positive.

        Notes
        -----
        - The environment generates a random controllable and observable state-space system.
        - It supports dynamic system generation, random reference motion, and varying transition times.
        """  # noqa: E501
        self.state_dim_min = state_dim_min
        self.state_dim_max = state_dim_max
        self.T = T
        self.Ts = Ts
        self.metadata["render_fps"] = int(1 / Ts)
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
        self._draw_random_motion_method = draw_random_motion_method

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
        if self._draw_random_motion_method == "gp":
            us = self._draw_us_gp()
        elif self._draw_random_motion_method == "rff":
            us = self._draw_us_random_fourier_features()
        elif self._draw_random_motion_method == "lpf-noise":
            us = self._draw_us_lpf_noise()
        elif self._draw_random_motion_method == "ou-noise":
            us = self._draw_us_ou_noise()
        else:
            raise Exception(f"{self._draw_random_motion_method} not valid")

        # apply action limits such that reference motion *is feasible*
        us = np.clip(us, self.action_space.low, self.action_space.high)
        yout = ctrl.forced_response(self._ss, T=self.ts, U=us.T).outputs
        self._us = us
        self._ref = yout[:, None]

    def _draw_us_gp(self):
        seed = self.np_random.integers(0, int(1e8))
        us = GaussianProcessRegressor(kernel=0.15 * RBF(0.75)).sample_y(
            self.ts[:, np.newaxis], random_state=seed
        )
        us = ((us - np.mean(us)) / np.std(us)) * 0.25
        return us

    def _draw_us_random_fourier_features(self):
        num_features = 10  # Controls smoothness (higher = less smooth)
        length_scale = 0.75  # Similar to RBF kernel length scale

        omega = (
            self.np_random.standard_normal(num_features) / length_scale
        )  # Random frequencies
        phi = self.np_random.uniform(0, 2 * np.pi, num_features)  # Random phases
        us = np.sum(np.sin(np.outer(self.ts, omega) + phi), axis=1)

        # Normalize and scale
        us = ((us - np.mean(us)) / np.std(us)) * 0.25
        return us

    def _draw_us_lpf_noise(self):
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            raise Exception("draw method `lpf-noise` requires `scipy`")

        us = self.np_random.standard_normal(len(self.ts))  # White noise
        us = gaussian_filter1d(
            us, sigma=15
        )  # Smooth with Gaussian filter, smoother with higher sigma
        us = ((us - np.mean(us)) / np.std(us)) * 0.25  # Normalize and scale
        return us

    def _draw_us_ou_noise(self):
        dt = self.ts[1] - self.ts[0]  # Time step
        theta = 0.5  # Mean-reverting speed (lower = smoother)
        sigma = 0.1  # Noise intensity
        us = np.zeros_like(self.ts)

        for i in range(1, len(self.ts)):
            us[i] = (
                us[i - 1]
                + theta * (-us[i - 1]) * dt
                + sigma * np.sqrt(dt) * self.np_random.standard_normal()
            )

        us = ((us - np.mean(us)) / np.std(us)) * 0.25
        return us

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

        truncated = bool(self.ts[self._t] >= (self.T - 1))
        terminated = truncated
        obs = self._get_obs()
        info = self._get_info()
        reward = -np.linalg.norm(obs["ref"] - obs["obs"])
        # smoothen the step of the reward function
        cd = obs["countdown"]
        if cd > self.reward_ramp:
            reward = 0.0
        else:
            reward *= (self.reward_ramp - cd[0]) / self.reward_ramp

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

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("`render` requires `matplotlib`")

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


def _rand(np_random, *args):
    return np_random.uniform(low=0, high=1, size=args)


def _randn(np_random, *args):
    return np_random.standard_normal(size=args)


def rss(np_random, states=1, outputs=1, inputs=1, strictly_proper=False, **kwargs):
    """Create a stable random state space object.

    Parameters
    ----------
    states, outputs, inputs : int, list of str, or None
        Description of the system states, outputs, and inputs. This can be
        given as an integer count or as a list of strings that name the
        individual signals.  If an integer count is specified, the names of
        the signal will be of the form 's[i]' (where 's' is one of 'x',
        'y', or 'u').
    strictly_proper : bool, optional
        If set to 'True', returns a proper system (no direct term).
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous
        time, True indicates discrete time with unspecified sampling
        time, positive number is discrete time with specified
        sampling time, None indicates unspecified timebase (either
        continuous or discrete time).
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

    Returns
    -------
    sys : StateSpace
        The randomly created linear system.

    Raises
    ------
    ValueError
        if any input is not a positive integer.

    Notes
    -----
    If the number of states, inputs, or outputs is not specified, then the
    missing numbers are assumed to be 1.  If dt is not specified or is given
    as 0 or None, the poles of the returned system will always have a
    negative real part.  If dt is True or a postive float, the poles of the
    returned system will have magnitude less than 1.

    """
    # Process keyword arguments
    kwargs.update({"states": states, "outputs": outputs, "inputs": inputs})
    name, inputs, outputs, states, dt = _process_iosys_keywords(kwargs)

    # Figure out the size of the sytem
    nstates, _ = _process_signal_list(states)
    ninputs, _ = _process_signal_list(inputs)
    noutputs, _ = _process_signal_list(outputs)

    sys = _rss_generate(
        np_random,
        nstates,
        ninputs,
        noutputs,
        "c" if not dt else "d",
        name=name,
        strictly_proper=strictly_proper,
    )

    return StateSpace(
        sys, name=name, states=states, inputs=inputs, outputs=outputs, dt=dt, **kwargs
    )


def _rss_generate(
    np_random, states, inputs, outputs, cdtype, strictly_proper=False, name=None
):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

    """

    rand = lambda *args: _rand(np_random, *args)
    randn = lambda *args: _randn(np_random, *args)

    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.6
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." % states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." % inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." % outputs)
    if cdtype not in ["c", "d"]:
        raise ValueError("cdtype must be `c` or `d`")

    # Make some poles for A.  Preallocate a complex array.
    poles = np.zeros(states) + np.zeros(states) * 0.0j
    i = 0

    while i < states:
        if rand() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last
            # element.
            if poles[i - 1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i - 1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i : i + 2] = poles[i - 2 : i]  # noqa: E203
                i += 2
        elif rand() < pReal or i == states - 1:
            # No-oscillation pole.
            if cdtype == "c":
                poles[i] = -np.exp(randn()) + 0.0j
            else:
                poles[i] = 2.0 * rand() - 1.0
            i += 1
        else:
            # Complex conjugate pair of oscillating poles.
            if cdtype == "c":
                poles[i] = complex(-np.exp(randn()), 3.0 * np.exp(randn()))
            else:
                mag = rand()
                phase = 2.0 * math.pi * rand()
                poles[i] = complex(mag * np.cos(phase), mag * np.sin(phase))
            poles[i + 1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = np.zeros((states, states))
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i + 1, i + 1] = poles[i].real
            A[i, i + 1] = poles[i].imag
            A[i + 1, i] = -poles[i].imag
            i += 2
    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = randn(states, states)
        try:
            A = np.linalg.solve(T, A) @ T  # A = T \ A @ T
            break
        except np.linalg.LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = randn(states, inputs)
    C = randn(outputs, states)
    D = randn(outputs, inputs)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rand(states, inputs) < pBCmask
        if np.any(Bmask):  # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if np.any(Cmask):  # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = np.zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else np.zeros(D.shape)

    if cdtype == "c":
        ss_args = (A, B, C, D)
    else:
        ss_args = (A, B, C, D, True)
    return StateSpace(*ss_args, name=name)
