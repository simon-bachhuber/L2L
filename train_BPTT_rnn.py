from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from rand_dyn_env import RandDynEnv



def _step_lti(A, B, C, action, x, Ts):
    "Returns: (y, next_x)"

    @jax.vmap
    def _timestep(A, B, C, x, u):
        rhs = lambda t, x: A @ x + B @ u
        next_x = _runge_kutta(rhs, 0, x, Ts)
        y = C @ next_x
        return next_x, y

    next_x, y = _timestep(A, B, C, x, action)
    return y, next_x

def _runge_kutta(rhs, t, x, dt):
    h = dt
    k1 = rhs(t, x)
    k2 = rhs(t + h / 2, x + h * k1 / 2)
    k3 = rhs(t + h / 2, x + h * k2 / 2)
    k4 = rhs(t + h, x + h * k3)
    dx = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x + dt * dx


def generate_random_lti_systems(batch_size: int, state_dim: int, **kwargs):
    env = RandDynEnv(state_dim_min=state_dim, state_dim_max=state_dim, **kwargs)
    sss, refs = [], []
    for _ in range(batch_size):
        env.reset()
        sss.append(env._ss)
        refs.append(env._ref)

    A = jnp.stack([ss.A for ss in sss], axis=0)
    B = jnp.stack([ss.B for ss in sss], axis=0)
    C = jnp.stack([ss.C for ss in sss], axis=0)
    refs = jnp.stack(refs, axis=0)
    return A, B, C, refs, jnp.array(env.Ts)


class Controller(nn.Module):
    hidden_dim: int
    num_layers: int = 1
    celltype = nn.OptimizedLSTMCell

    @nn.compact
    def __call__(self, x, carry=None):
        cells = [self.celltype(self.hidden_dim) for _ in range(self.num_layers)]

        if carry is None:
            carry = [
                cell.initialize_carry(jax.random.key(0), x.shape) for cell in cells
            ]

        new_carry = []
        for cell, _carry in zip(cells, carry):
            _carry, x = cell(_carry, x)
            new_carry.append(_carry)
        u_t = nn.Dense(1)(x)  # Control output
        yhat = nn.Dense(1)(x)  # Predict next states
        return (u_t, yhat), new_carry


T = 3000


def stability_loss(ys, refs):
    errors = refs - ys  # Compute tracking error e_t
    errors_diff = jnp.diff(errors, axis=0)  # Compute (e_t - e_{t-1})
    loss_stability = jnp.mean(errors_diff**2)
    return loss_stability


def exploration_loss(u_seq):
    std = jnp.mean(jnp.std(u_seq, axis=0))
    return -jnp.log(std + 1e-6)  # Maximize std, which maximizes exploration


@partial(
    jax.jit,
    static_argnames=[
        "apply_fn",
        "optimizer",
        "T1",
        "T2",
        "lam_id_loss",
        "lam_ex_loss",
        "lam_tr_loss",
        "lam_st_loss",
    ],
)
def update(
    apply_fn,
    params,
    A, B, C, Ts,
    opt_state,
    refs,
    carry,
    optimizer,
    T1,
    T2,
    lam_id_loss,
    lam_ex_loss,
    lam_tr_loss,
    lam_st_loss,
):

    def loss_fn(params, refs, carry):
        def closed_loop(carry, ref_t):
            controller_carry, x_tm1, y_tm1, u_tm1, t = carry
            (u_t, yhat_t), controller_carry = apply_fn(
                params,
                jnp.concatenate(
                    [y_tm1, ref_t, jax.lax.stop_gradient(u_tm1), t], axis=-1
                ),
                controller_carry,
            )
            y_t, x_t = _step_lti(A, B, C, u_t, x_tm1, Ts)
            return (controller_carry, x_t, y_t, u_t, (t + 1) / T), (u_t, y_t, yhat_t)

        carry, (us, ys, yhats) = jax.lax.scan(closed_loop, carry, refs)

        id_loss = jnp.mean((ys[:T1] - yhats[:T1]) ** 2)
        expl_loss = exploration_loss(us[:T1])
        track_loss = jnp.mean((ys[T1:T2] - refs[T1:T2]) ** 2)
        stabl_loss = stability_loss(ys[T2:], refs[T2:])

        loss_term = (
            id_loss * lam_id_loss
            + expl_loss * lam_ex_loss
            + track_loss * lam_tr_loss
            + stabl_loss * lam_st_loss
        )
        loss_logging = {
            "stabl_loss": stabl_loss,
            "track_mae_after_40s": jnp.mean(jnp.abs(ys - refs)),
            "id_loss": id_loss,
            "expl_loss": expl_loss,
            "track_loss": track_loss,
        }
        return loss_term, (carry, loss_logging)

    (loss, (carry, loss_terms)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, refs, carry
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, carry, loss_terms


def train_controller(
    controller,
    params,
    opt_state,
    F,
    batch_size,
    state_dim,
    steps,
    optimizer,
    T1,
    T2,
    lam_id_loss,
    lam_ex_loss,
    lam_tr_loss,
    lam_st_loss,
    logging=False,
    **kwargs,
):
    _, carry = controller.apply(params, jnp.zeros((batch_size, F)))
    x = jnp.zeros((batch_size, state_dim))
    init = (
        carry,
        x,
        jnp.zeros((batch_size, 1)),
        jnp.zeros((batch_size, 1)),
        jnp.zeros((batch_size, 1)),
    )

    for step in range(steps):
        A, B, C, refs, Ts = generate_random_lti_systems(batch_size, state_dim, **kwargs)
        refs = refs.transpose(1, 0, 2)

        params, opt_state, loss, _, loss_terms = update(
            controller.apply,
            params,
            A, B, C, Ts,
            opt_state,
            refs,
            init,
            optimizer,
            T1,
            T2,
            lam_id_loss,
            lam_ex_loss,
            lam_tr_loss,
            lam_st_loss,
        )

        if (step % 5 == 0) and logging:
            print(
                f"Step {step}: Loss = {jnp.mean(loss):.4f}",
                ", ".join(
                    [f"{key} = {float(val):.4f}" for key, val in loss_terms.items()]
                ),
            )

    return params, opt_state, loss_terms["track_mae_after_40s"]


max_steps = 10_000


def objective(config):
    F = 4
    controller = Controller(config["num_hidden_nodes"], num_layers=config["num_layers"])
    params = controller.init(jax.random.key(1), jnp.zeros((config["batch_size"], F)))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["grad_clip"]),
        optax.adam(
            learning_rate=config["lr"],
            b1=config["b1"],
            b2=config["b2"],
            eps=config["eps"],
        ),
    )
    opt_state = optimizer.init(params)

    for step in range(max_steps):
        params, opt_state, tracking_mae = train_controller(
            controller,
            params,
            opt_state,
            F,
            config["batch_size"],
            config["state_dim"],
            1,
            optimizer,
            config["T1"],
            config["T2"],
            config["lam_id_loss"],
            config["lam_ex_loss"],
            config["lam_tr_loss"],
            config["lam_st_loss"],
            logging=False,
            draw_step_function_reference=config["draw_step_function_reference"],
        )
        train.report({"tracking_mae": float(tracking_mae), "step": step})


param_space = {
    "num_hidden_nodes": tune.choice([16, 32, 64, 128, 256]),
    "num_layers": tune.randint(1, 4),
    "batch_size": tune.choice([32, 64, 128]),
    "grad_clip": tune.loguniform(0.01, 10.0),
    "lr": tune.loguniform(1e-4, 1e-2),
    "b1": tune.uniform(0.8, 0.99),
    "b2": tune.uniform(0.9, 0.999),
    "eps": tune.loguniform(1e-8, 1e-5),
    "T1": tune.randint(10, 2000),
    "T2": tune.randint(1500, 2900),
    "lam_id_loss": tune.loguniform(1e-4, 10.0),
    "lam_ex_loss": tune.loguniform(1e-4, 10.0),
    "lam_tr_loss": tune.loguniform(1e-4, 10.0),
    "lam_st_loss": tune.loguniform(1e-4, 10.0),
    "state_dim": tune.choice([1, 2, 3, 4]),
    "draw_step_function_reference": tune.choice([True, False]),
}

tuner = tune.Tuner(
    tune.with_resources(objective, {"gpu": 0.5, "cpu": 4}),
    param_space=param_space,
    tune_config=tune.TuneConfig(
        mode="min",
        metric="tracking_mae",
        time_budget_s=24 * 3600,
        num_samples=-1,
        scheduler=ASHAScheduler(
            "step",
            max_t=max_steps,
            grace_period=50,
        ),
        max_concurrent_trials=8,
    ),
)

tuner.fit()
