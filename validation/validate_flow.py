"""Validation stage 2: the three solvers, checked against closed-form Gaussian OT.

This pins down (a) the correct entropic-map constant, (b) which ICNN potential
carries the forward map, (c) workable default hyper-parameters.
"""
import functools
import numpy as np, jax, jax.numpy as jnp
from reference import *

ok = lambda name, cond, extra="": print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
rng = np.random.default_rng(0)

# --- problem: two 2-D Gaussians, equal covariance -> Brenier map is a shift ---
D = 2
m0 = np.zeros(D); S0 = 0.5 ** 2 * np.eye(D)
m1 = np.array([3.0, 0.0]); S1 = 0.5 ** 2 * np.eye(D)
A_true, b_true = brenier_gaussian(m0, S0, m1, S1)
W2_true = w2_gaussian(m0, S0, m1, S1)
L0 = np.linalg.cholesky(S0); L1 = np.linalg.cholesky(S1)


def sample_mu(key, n):
    return jnp.array(L0) @ jax.random.normal(key, (D, n)) + jnp.array(m0)[:, None]


def sample_nu(key, n):
    return jnp.array(L1) @ jax.random.normal(key, (D, n)) + jnp.array(m1)[:, None]


def map_error(T_fn, key, n=4000):
    X = sample_mu(key, n)
    T = T_fn(X)
    ref = jnp.array(A_true) @ X + jnp.array(b_true)[:, None]
    return float(jnp.sqrt(jnp.mean(jnp.sum((T - ref) ** 2, axis=0))))


print("=" * 74)
print("problem: 2D Gaussians, W2^2 = %.4f" % W2_true)
print("=" * 74)

# C. flow_match
# ---------------------------------------------------------------------------
print("\nC. flow_match: conditional flow matching + ODE integration")
print("-" * 74)


def swish(x):
    return x * jax.nn.sigmoid(x)


def fm_loss(params, X0, X1, t):
    Xt = (1 - t) * X0 + t * X1
    inp = jnp.concatenate([t, Xt], axis=0)
    pred = mlp_forward(params, inp, act=swish)
    return jnp.mean((pred - (X1 - X0)) ** 2)


def ot_pair(X0, X1, key, eps=0.05):
    """minibatch OT coupling: resample X1 columns from the Sinkhorn plan rows."""
    C = np.asarray(pairwise_sqeuclidean(X0, X1))
    fp, gp, _, _ = sinkhorn_potentials(C, eps=eps, n_iter=100, tol=1e-8)
    P = sinkhorn_plan(C, fp, gp, eps=eps)
    P = P / P.sum(axis=1, keepdims=True)
    cdf = np.cumsum(P, axis=1)
    u = np.random.rand(P.shape[0], 1)
    idx = (cdf < u).sum(axis=1).clip(0, P.shape[1] - 1)
    return X1[:, idx]


def train_flow(steps=3000, batch=256, lr=1e-3, coupling="independent", seed=0):
    key = jax.random.PRNGKey(seed)
    k1, key = jax.random.split(key)
    params = mlp_init(k1, [D + 1, 128, 128, D])
    state = adam_init(params)

    @jax.jit
    def step(params, state, X0, X1, t):
        loss, g = jax.value_and_grad(fm_loss)(params, X0, X1, t)
        params, state = adam_step(params, g, state, lr=lr)
        return params, state, loss

    hist = []
    for i in range(steps):
        key, kx, ky, kt, kc = jax.random.split(key, 5)
        X0, X1 = sample_mu(kx, batch), sample_nu(ky, batch)
        if coupling == "ot":
            X1 = ot_pair(X0, X1, kc)
        t = jax.random.uniform(kt, (1, batch))
        params, state, loss = step(params, state, X0, X1, t)
        if i % 200 == 0:
            hist.append(float(loss))
    return params, hist


def integrate(params, x0, n_steps=100, solver="euler"):
    x = x0
    dt = 1.0 / n_steps
    vf = lambda t, x: mlp_forward(params, jnp.concatenate(
        [jnp.full((1, x.shape[1]), t), x], axis=0), act=swish)
    for k in range(n_steps):
        t = k * dt
        if solver == "euler":
            x = x + dt * vf(t, x)
        elif solver == "heun":
            k1 = vf(t, x)
            k2 = vf(t + dt, x + dt * k1)
            x = x + dt * 0.5 * (k1 + k2)
        elif solver == "rk4":
            k1 = vf(t, x)
            k2 = vf(t + dt / 2, x + dt / 2 * k1)
            k3 = vf(t + dt / 2, x + dt / 2 * k2)
            k4 = vf(t + dt, x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


params_i, hist_i = train_flow(coupling="independent")
print(f"  independent coupling: loss {hist_i[0]:.4f} -> {hist_i[-1]:.4f}")
Xtest = sample_mu(jax.random.PRNGKey(31), 2000)
Ytest = sample_nu(jax.random.PRNGKey(32), 2000)
for solver in ["euler", "heun", "rk4"]:
    P = integrate(params_i, Xtest, n_steps=50, solver=solver)
    sd = sinkhorn_divergence(np.asarray(P), np.asarray(Ytest), eps=0.1, n_iter=1500, tol=1e-9)
    print(f"    {solver:5s}: S(pushforward, nu) = {sd:.4f}   mean = {np.asarray(P).mean(1)}")
P = integrate(params_i, Xtest, n_steps=50, solver="rk4")
sd_flow = sinkhorn_divergence(np.asarray(P), np.asarray(Ytest), eps=0.1, n_iter=1500, tol=1e-9)
sd_src = sinkhorn_divergence(np.asarray(Xtest), np.asarray(Ytest), eps=0.1, n_iter=1500, tol=1e-9)
ok("flow pushforward matches nu", sd_flow < 0.05 * sd_src,
   f"S={sd_flow:.4f} (source {sd_src:.4f})")
# reverse integration should invert the flow
back = integrate(params_i, P, n_steps=50, solver="rk4")  # placeholder, replaced below


def integrate_rev(params, x1, n_steps=50):
    x = x1
    dt = 1.0 / n_steps
    vf = lambda t, x: mlp_forward(params, jnp.concatenate(
        [jnp.full((1, x.shape[1]), t), x], axis=0), act=swish)
    for k in range(n_steps):
        t = 1.0 - k * dt
        x = x - dt * vf(t, x)
    return x


back = integrate_rev(params_i, P)
rt = float(jnp.sqrt(jnp.mean(jnp.sum((back - Xtest) ** 2, axis=0))))
ok("reverse integration inverts the flow", rt < 0.3, f"round-trip RMSE={rt:.4f}")

params_o, hist_o = train_flow(coupling="ot", steps=1200)
params_i2, hist_i2 = train_flow(coupling="independent", steps=1200)
Po = integrate(params_o, Xtest, n_steps=50, solver="rk4")
Pi = integrate(params_i2, Xtest, n_steps=50, solver="rk4")
# straightness: transport cost of the learned coupling vs the true W2
cost_o = float(jnp.mean(jnp.sum((Po - Xtest) ** 2, axis=0)))
cost_i = float(jnp.mean(jnp.sum((Pi - Xtest) ** 2, axis=0)))
print(f"  transport cost E||T(x)-x||^2 : OT coupling {cost_o:.4f} | "
      f"independent {cost_i:.4f} | true W2^2 {W2_true:.4f}")
ok("OT coupling gives a straighter (lower-cost) flow",
   abs(cost_o - W2_true) <= abs(cost_i - W2_true) + 1e-6,
   f"|gap| {abs(cost_o-W2_true):.4f} vs {abs(cost_i-W2_true):.4f}")
