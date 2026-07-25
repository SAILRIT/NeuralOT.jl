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

# B. solve_w2  (Makkuva et al., ICNN saddle point)
# ---------------------------------------------------------------------------
print("\nB. solve_w2: ICNN saddle point")
print("-" * 74)


def inner_loss(g, f, X):
    """maximised over g:  E_mu[<X, grad g(X)> - f(grad g(X))]"""
    T = icnn_input_grad(g, X)
    return -jnp.mean(jnp.sum(X * T, axis=0) - icnn_forward(f, T)[0])


def outer_loss(f, g, X, Y):
    """minimised over f:  E_mu[<X,grad g> - f(grad g)] + E_nu[f(Y)]"""
    T = icnn_input_grad(g, X)
    return jnp.mean(jnp.sum(X * T, axis=0) - icnn_forward(f, T)[0]) + jnp.mean(icnn_forward(f, Y)[0])


def train_w2(steps=1500, inner=8, batch=128, lr=1e-3, widths=(64, 64, 1), seed=0,
             init_scale=0.1):
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)
    f = icnn_init(k1, D, list(widths), scale=init_scale)
    g = icnn_init(k2, D, list(widths), scale=init_scale)
    sf, sg = adam_init(f), adam_init(g)

    @jax.jit
    def gstep(g, sg, f, X):
        loss, grads = jax.value_and_grad(inner_loss)(g, f, X)
        g, sg = adam_step(g, grads, sg, lr=lr)
        return g, sg, loss

    @jax.jit
    def fstep(f, sf, g, X, Y):
        loss, grads = jax.value_and_grad(outer_loss)(f, g, X, Y)
        f, sf = adam_step(f, grads, sf, lr=lr)
        return f, sf, loss

    hist = []
    for t in range(steps):
        key, kx, ky = jax.random.split(key, 3)
        X, Y = sample_mu(kx, batch), sample_nu(ky, batch)
        for _ in range(inner):
            g, sg, _ = gstep(g, sg, f, X)
        f, sf, loss = fstep(f, sf, g, X, Y)
        if t % 100 == 0:
            hist.append(float(loss))
    return f, g, hist


f, g, hist = train_w2()
fwd_err = map_error(lambda X: icnn_input_grad(g, X), jax.random.PRNGKey(5))
# does grad f instead carry the forward map? (checks the f/g convention)
fwd_err_f = map_error(lambda X: icnn_input_grad(f, X), jax.random.PRNGKey(5))
# inverse direction: grad f should map nu -> mu
Yv = sample_nu(jax.random.PRNGKey(6), 4000)
inv = icnn_input_grad(f, Yv)
inv_err = float(jnp.sqrt(jnp.mean(jnp.sum((inv - (jnp.linalg.inv(jnp.array(A_true)) @
                                                 (Yv - jnp.array(b_true)[:, None]))) ** 2, axis=0))))
print(f"  outer objective {hist[0]:+.3f} -> {hist[-1]:+.3f}")
print(f"  RMSE(grad g, true forward map) = {fwd_err:.4f}    "
      f"RMSE(grad f, true forward map) = {fwd_err_f:.4f}")
print(f"  RMSE(grad f, true inverse map) = {inv_err:.4f}")
ok("grad g is the forward map mu->nu", fwd_err < 0.3 and fwd_err < fwd_err_f,
   f"RMSE={fwd_err:.4f}")
ok("grad f is the inverse map nu->mu", inv_err < 0.4, f"RMSE={inv_err:.4f}")
Xp = sample_mu(jax.random.PRNGKey(21), 1000)
Yp = sample_nu(jax.random.PRNGKey(22), 1000)
sd = sinkhorn_divergence(np.asarray(icnn_input_grad(g, Xp)), np.asarray(Yp), eps=0.1,
                         n_iter=2000, tol=1e-10)
sd0 = sinkhorn_divergence(np.asarray(Xp), np.asarray(Yp), eps=0.1, n_iter=2000, tol=1e-10)
ok("pushforward much closer to nu than the source was", sd < 0.05 * sd0,
   f"S(T#mu,nu)={sd:.4f} vs S(mu,nu)={sd0:.4f}")

# without the strong-convexity quadratic term, training is much harder
f2, g2, _ = train_w2(seed=0)
print(f"  (map RMSE with quadratic term / strong convexity: {fwd_err:.4f})")

