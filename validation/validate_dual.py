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
print(f"problem: N({m0}, .25I) -> N({m1}, .25I);  W2^2 = {W2_true:.4f}; "
      f"true map T(x) = x + {b_true}")
print("=" * 74)

# ---------------------------------------------------------------------------
# A. solve_dual  (Seguy et al.)
# ---------------------------------------------------------------------------
print("\nA. solve_dual: entropic dual potentials")
print("-" * 74)


def dual_loss(params, X, Y, eps, formulation):
    u, v = params["u"], params["v"]
    ux = mlp_forward(u, X)[0]
    vy = mlp_forward(v, Y)[0]
    C = pairwise_sqeuclidean(X, Y)
    M = (ux[:, None] + vy[None, :] - C) / eps
    if formulation == "exp":
        pen = eps * jnp.mean(jnp.exp(M))
    else:  # numerically-stable log-sum-exp variant
        mx = jax.lax.stop_gradient(jnp.max(M))
        pen = eps * (mx + jnp.log(jnp.mean(jnp.exp(M - mx))))
    return -(jnp.mean(ux) + jnp.mean(vy) - pen)


def train_dual(eps=0.1, steps=3000, batch=256, lr=1e-3, formulation="exp", seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)
    params = dict(u=mlp_init(k1, [D, 128, 128, 1]), v=mlp_init(k2, [D, 128, 128, 1]))
    state = adam_init(params)

    @jax.jit
    def step(params, state, kx, ky):
        X, Y = sample_mu(kx, batch), sample_nu(ky, batch)
        loss, g = jax.value_and_grad(dual_loss)(params, X, Y, eps, formulation)
        params, state = adam_step(params, g, state, lr=lr)
        return params, state, loss

    losses = []
    for t in range(steps):
        key, kx, ky = jax.random.split(key, 3)
        params, state, loss = step(params, state, kx, ky)
        if t % 200 == 0:
            losses.append(float(loss))
    return params, losses


for form in ["exp", "lse"]:
    params, losses = train_dual(formulation=form)
    dual_val = -losses[-1]
    grad_u = jax.jit(jax.jacrev(lambda x: jnp.sum(mlp_forward(params["u"], x))))
    err_half = map_error(lambda X: X - 0.5 * grad_u(X), jax.random.PRNGKey(99))
    err_eps = map_error(lambda X: X - grad_u(X) / (2 * 0.1), jax.random.PRNGKey(99))
    # reference entropic OT value from Sinkhorn on large samples
    Xs = np.asarray(sample_mu(jax.random.PRNGKey(11), 800))
    Ys = np.asarray(sample_nu(jax.random.PRNGKey(12), 800))
    ref_val = sinkhorn_value(np.asarray(pairwise_sqeuclidean(Xs, Ys)), eps=0.1,
                             n_iter=5000, tol=1e-11)
    print(f"  formulation={form:4s}  loss {losses[0]:+8.3f} -> {losses[-1]:+8.3f}  "
          f"dual value={dual_val:6.3f}  (Sinkhorn ref {ref_val:6.3f}, W2^2 {W2_true:.3f})")
    print(f"      map RMSE  T = x - grad_u/2      : {err_half:.4f}   <- textbook constant")
    print(f"      map RMSE  T = x - grad_u/(2*eps): {err_eps:.4f}   <- constant used in v0.1")
    ok(f"[{form}] entropic map with /2 recovers the Brenier map", err_half < 0.25,
       f"RMSE={err_half:.4f}")
    ok(f"[{form}] the /(2 eps) variant is materially worse", err_eps > 3 * err_half,
       f"{err_eps:.3f} vs {err_half:.3f}")
    ok(f"[{form}] dual value close to Sinkhorn reference", abs(dual_val - ref_val) < 0.6,
       f"|diff|={abs(dual_val-ref_val):.3f}")

# barycentric projection from samples (model-free, more accurate)
params, _ = train_dual(formulation="lse")


def barycentric(X, Y, params, eps=0.1):
    ux = mlp_forward(params["u"], X)[0]
    vy = mlp_forward(params["v"], Y)[0]
    C = pairwise_sqeuclidean(X, Y)
    logw = (ux[:, None] + vy[None, :] - C) / eps
    logw = logw - jax.scipy.special.logsumexp(logw, axis=1, keepdims=True)
    return Y @ jnp.exp(logw).T


Yb = sample_nu(jax.random.PRNGKey(77), 2000)
err_bary = map_error(lambda X: barycentric(X, Yb, params), jax.random.PRNGKey(99), n=2000)
ok("sample-based barycentric projection recovers the map", err_bary < 0.25,
   f"RMSE={err_bary:.4f}")

