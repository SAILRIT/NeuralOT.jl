"""Validation stage 1: ICNN analytic input-gradient, convexity, Sinkhorn."""
import numpy as np, jax, jax.numpy as jnp
from reference import *

rng = np.random.default_rng(0)
ok = lambda name, cond, extra="": print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")

print("=" * 70)
print("1. ICNN: analytic input gradient vs autodiff and finite differences")
print("=" * 70)
key = jax.random.PRNGKey(0)
for dim, widths in [(2, [16, 16, 1]), (5, [32, 32, 32, 1]), (3, [8, 1]), (4, [12, 12, 12, 12, 1])]:
    p = icnn_init(key, dim, widths)
    X = jnp.array(rng.normal(size=(dim, 7)))
    ga = icnn_input_grad(p, X)
    gauto = jax.jacrev(lambda x: jnp.sum(icnn_forward(p, x)))(X)
    err = float(jnp.max(jnp.abs(ga - gauto)))
    # finite differences on a single column
    h = 1e-6
    fd = np.zeros(dim)
    for i in range(dim):
        Xp = np.array(X); Xm = np.array(X)
        Xp[i, 0] += h; Xm[i, 0] -= h
        fd[i] = (icnn_forward(p, jnp.array(Xp))[0, 0] - icnn_forward(p, jnp.array(Xm))[0, 0]) / (2 * h)
    ferr = float(np.max(np.abs(fd - np.array(ga)[:, 0])))
    ok(f"grad_x  dim={dim} widths={widths}", err < 1e-9 and ferr < 1e-5,
       f"|analytic-AD|={err:.2e} |analytic-FD|={ferr:.2e}")

print()
print("=" * 70)
print("2. ICNN: convexity in the input (midpoint test + Hessian eigenvalues)")
print("=" * 70)
for trial in range(4):
    key = jax.random.PRNGKey(trial)
    dim = 4
    p = icnn_init(key, dim, [24, 24, 1])
    X1 = jnp.array(rng.normal(size=(dim, 400)) * 2)
    X2 = jnp.array(rng.normal(size=(dim, 400)) * 2)
    mid = 0.5 * (X1 + X2)
    lhs = icnn_forward(p, mid)[0]
    rhs = 0.5 * (icnn_forward(p, X1)[0] + icnn_forward(p, X2)[0])
    viol = float(jnp.max(lhs - rhs))
    H = jax.hessian(lambda x: icnn_forward(p, x.reshape(-1, 1))[0, 0])(jnp.array(rng.normal(size=dim)))
    evmin = float(np.min(np.linalg.eigvalsh(np.array(H))))
    ok(f"convexity trial {trial}", viol <= 1e-9 and evmin > -1e-9,
       f"max midpoint violation={viol:.2e}  min Hessian eig={evmin:.4f}")

print()
print("=" * 70)
print("3. ICNN: gradient is differentiable w.r.t. PARAMETERS (the nested-AD fix)")
print("=" * 70)
key = jax.random.PRNGKey(3)
p = icnn_init(key, 3, [16, 16, 1])
X = jnp.array(rng.normal(size=(3, 32)))


def objective(p, X):
    T = icnn_input_grad(p, X)
    return jnp.mean(jnp.sum(X * T, axis=0))


g = jax.grad(objective)(p, X)
finite = all(bool(jnp.all(jnp.isfinite(w))) for w in g["Wx"] + g["Wz"] + g["b"] + [g["beta"]])
nonzero = float(sum(float(jnp.sum(jnp.abs(w))) for w in g["Wx"]))
ok("d/dparams of grad_x is finite and non-trivial", finite and nonzero > 0,
   f"sum|dWx|={nonzero:.3f}")

# ...and matches a nested-AD computation of the same quantity
def objective_nested(p, X):
    T = jax.jacrev(lambda x: jnp.sum(icnn_forward(p, x)))(X)
    return jnp.mean(jnp.sum(X * T, axis=0))


g2 = jax.grad(objective_nested)(p, X)
err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(g["Wx"], g2["Wx"]))
ok("analytic path == nested-AD path (dW)", err < 1e-9, f"max diff={err:.2e}")

print()
print("=" * 70)
print("4. Sinkhorn: convergence to exact OT, marginals, divergence properties")
print("=" * 70)
X = jnp.array(rng.normal(size=(2, 60)))
Y = jnp.array(rng.normal(size=(2, 60)) + np.array([[3.0], [0.0]]))
C = np.asarray(pairwise_sqeuclidean(X, Y))
exact = exact_ot_cost(X, Y)
print(f"    exact W2^2 (Hungarian) = {exact:.6f}")
for eps in [1.0, 0.1, 0.01, 0.005]:
    f, g, it, err = sinkhorn_potentials(C, eps=eps, n_iter=20000, tol=1e-12)
    P = sinkhorn_plan(C, f, g, eps=eps)
    marg_err = max(np.max(np.abs(P.sum(1) - 1 / 60)), np.max(np.abs(P.sum(0) - 1 / 60)))
    cost = float(np.sum(P * C))
    print(f"    eps={eps:<6} iters={it:<6} <pi,C>={cost:.6f}  "
          f"rel.err vs exact={abs(cost-exact)/exact:.2e}  marginal err={marg_err:.2e}")
f, g, it, err = sinkhorn_potentials(C, eps=0.005, n_iter=20000, tol=1e-12)
P = sinkhorn_plan(C, f, g, eps=0.005)
ok("Sinkhorn plan matches marginals", max(np.max(np.abs(P.sum(1) - 1 / 60)),
                                          np.max(np.abs(P.sum(0) - 1 / 60))) < 1e-8)
ok("Sinkhorn -> exact OT as eps -> 0", abs(float(np.sum(P * C)) - exact) / exact < 0.02,
   f"rel err={abs(float(np.sum(P*C))-exact)/exact:.2e}")

s_xx = sinkhorn_divergence(X, X, eps=0.1, n_iter=5000, tol=1e-12)
s_xy = sinkhorn_divergence(X, Y, eps=0.1, n_iter=5000, tol=1e-12)
s_yx = sinkhorn_divergence(Y, X, eps=0.1, n_iter=5000, tol=1e-12)
Z = jnp.array(rng.normal(size=(2, 60)))
s_xz = sinkhorn_divergence(X, Z, eps=0.1, n_iter=5000, tol=1e-12)
ok("S(X,X) == 0", abs(s_xx) < 1e-10, f"S(X,X)={s_xx:.3e}")
ok("S symmetric", abs(s_xy - s_yx) < 1e-8, f"|S(X,Y)-S(Y,X)|={abs(s_xy-s_yx):.2e}")
ok("S >= 0 and separates", s_xy > s_xz > -1e-10, f"S(X,Y)={s_xy:.4f}  S(X,Z)={s_xz:.4f}")
ok("S(X,Y) close to W2^2 for small eps",
   abs(sinkhorn_divergence(X, Y, eps=0.01, n_iter=20000, tol=1e-12) - exact) / exact < 0.05,
   f"S_0.01={sinkhorn_divergence(X, Y, eps=0.01, n_iter=20000, tol=1e-12):.4f} vs {exact:.4f}")

print()
print("=" * 70)
print("5. Gaussian closed forms")
print("=" * 70)
d = 3
A0 = rng.normal(size=(d, d)); S0 = A0 @ A0.T + np.eye(d)
A1 = rng.normal(size=(d, d)); S1 = A1 @ A1.T + np.eye(d)
m0 = rng.normal(size=d); m1 = rng.normal(size=d)
w2 = w2_gaussian(m0, S0, m1, S1)
A, b = brenier_gaussian(m0, S0, m1, S1)
# push a large sample through the map and compare moments
Xs = (np.linalg.cholesky(S0) @ rng.normal(size=(d, 200000))) + m0[:, None]
Ts = A @ Xs + b[:, None]
mean_err = np.max(np.abs(Ts.mean(1) - m1))
cov_err = np.max(np.abs(np.cov(Ts) - S1))
mc = np.mean(np.sum((Xs - Ts) ** 2, axis=0))
ok("Brenier map pushes moments correctly", mean_err < 0.02 and cov_err < 0.1,
   f"mean err={mean_err:.4f} cov err={cov_err:.4f}")
ok("E||x-T(x)||^2 == W2^2", abs(mc - w2) / w2 < 0.01, f"MC={mc:.4f} closed form={w2:.4f}")
ok("A is symmetric PSD (a valid Brenier/gradient-of-convex map)",
   np.min(np.linalg.eigvalsh(A)) > 0 and np.max(np.abs(A - A.T)) < 1e-8)
ok("W2 symmetric", abs(w2 - w2_gaussian(m1, S1, m0, S0)) < 1e-8)
