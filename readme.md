# RandDynEnv: Randomized Dynamic Environment

## Overview

`RandDynEnv` is a Gymnasium-based environment for reinforcement learning with dynamic, randomized, controllable, and observable state-space systems. It provides tools to simulate complex environments with customizable system dynamics, reference motions, and transition times.

## Features

- Randomized controllable and observable state-space systems.
- Adjustable state, input, and output dimensions.
- Dynamic reference motion generation using Gaussian Process Regression.
- Customizable transition times and action limits.
- Fully integrated with Gym's API for easy RL integration.

## Installation

Requires `gymnasium, numpy, scikit-learn, matplotlib, python-control`

## Usage

### Environment Setup
```python
import rand_dyn_env  # noqa: F401
import gymnasium as gym

env = gym.make("RandDyn-v0", 
               state_dim_min=3, 
               state_dim_max=5, 
               T=60, 
               Ts=0.02, 
               n_inputs=1, 
               n_outputs=1)
```

### Runing an Episode
```python
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your control policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

    if done:
        env.reset()
```

### Structure of Observations
```python
# `obs` is a dictionary
{
    # the remaining time before the reward transitions into punishing tracking error
    "countdown": np.ndarray,  # Shape: (1,)
    # the current desired reference
    "ref": np.ndarray,        # Shape: (n_outputs,)
    # the current system output
    "obs": np.ndarray         # Shape: (n_outputs,)
}
# `action` is a np.ndarray with shape (n_inputs,)
```

## Parameters
- **state_dim_min** *(int, default=3)*: Minimum number of state variables.  
- **state_dim_max** *(int, default=3)*: Maximum number of state variables.  
- **T** *(float, default=60)*: Total duration of the environment in seconds.  
- **Ts** *(float, default=0.02)*: Sampling time step in seconds.  
- **n_inputs** *(int, default=1)*: Number of control inputs.  
- **n_outputs** *(int, default=1)*: Number of outputs.  
- **reward_ramp** *(float, default=1.0)*: Reward ramp-up time in seconds.  
- **on_reset_draw_sys** *(bool, default=True)*: If True, draw a new random system upon environment reset.  
- **on_reset_draw_motion** *(bool, default=True)*: If True, draw a new random reference motion upon environment reset.  
- **on_reset_draw_transition_time** *(bool, default=True)*: If True, draw a new random transition time upon environment reset.  
- **transition_time** *(float, default=30)*: Initial transition time in seconds, used when `on_reset_draw_transition_time` is False.  
- **action_limit** *(float, default=1.0)*: Upper and lower limit for actions. Actions are clipped to [-action_limit, action_limit]. 
- **draw_random_motion_method** *(str, default='rff')*: Specifies the method used to generate a random feedforward signal `us`, which is then applied to the system to create a feasible reference trajectory. Possible values are `gp` (Gaussian Process), `rff` (Random Fourier Features), `lpf-noise` (Low-Pass-Filtered-Whitenoise), and `ou-noise` (Ornstein-Uhlenbeck Process).  
- **render_mode** *(str | None, default='rgb_array')*: Specifies how the `env.render()` function behaves. If `None`, no rendering takes place.  
- **draw_step_function_reference** *(bool, default=False)*: If `True`, the reference will become a constant step after the transition time has passed.  
- **scale_by_step_response** *(bool, default=True)*: Scales the output of the internally generated random dynamics by its step response. This ensures that the output will always be of a similar scale between different dynamics.  

# Simple PPO example `train.py`
This is the result of `train.py` when the `env.reset` does not change the dynamics and motion, i.e., it is for a single dynamics and motion.
![training-progress](assets/video.gif)