"""
Reference implementation of every algorithm that goes into NeuralOT.jl v0.2.

This mirrors the Julia code formula-for-formula so the numerics can be
validated (against finite differences, autodiff, exact LP/Hungarian OT and
closed-form Gaussian optimal transport) before the Julia source is written.

Column convention matches Julia: data is (dim, batch).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import linear_sum_assignment
from scipy.linalg import sqrtm

jax.config.update("jax_enable_x64", True)

softplus = jax.nn.softplus
sigmoid = jax.nn.sigmoid


# ----------------------------------------------------------------------------
# Gaussian closed forms  (src/gaussian.jl)
# ----------------------------------------------------------------------------
def w2_gaussian(m0, S0, m1, S1):
    """Squared 2-Wasserstein distance between two Gaussians (Bures form)."""
    s0h = np.real(sqrtm(S0))
    cross = np.real(sqrtm(s0h @ S1 @ s0h))
    return float(np.sum((m0 - m1) ** 2) + np.trace(S0 + S1 - 2 * cross))


def brenier_gaussian(m0, S0, m1, S1):
    """Return (A, b) with the optimal map T(x) = A x + b from N(m0,S0)->N(m1,S1)."""
    s0h = np.real(sqrtm(S0))
    s0hi = np.linalg.inv(s0h)
    A = s0hi @ np.real(sqrtm(s0h @ S1 @ s0h)) @ s0hi
    A = 0.5 * (A + A.T)
    b = m1 - A @ m0
    return A, b


# ----------------------------------------------------------------------------
# ICNN  (src/icnn.jl)
# ----------------------------------------------------------------------------
def icnn_init(key, dim, widths, quadratic=True, beta0=1.0, scale=1.0):
    """Glorot-uniform init.  params = dict(Wx=[...], Wz=[...], b=[...], beta=..)."""
    keys = jax.random.split(key, 3 * len(widths) + 1)
    Wx, Wz, b = [], [], []
    prev = dim
    for l, w in enumerate(widths):
        lim = np.sqrt(6.0 / (w + dim))
        Wx.append(scale * jax.random.uniform(keys[3 * l], (w, dim), minval=-lim, maxval=lim))
        if l == 0:
            Wz.append(jnp.zeros((w, 0)))
        else:
            limz = np.sqrt(6.0 / (w + prev))
            # inverse-softplus of a small positive number keeps the map near
            # identity at initialisation
            Wz.append(jax.random.uniform(keys[3 * l + 1], (w, prev), minval=-limz, maxval=limz))
        b.append(jnp.zeros((w,)))
        prev = w
    beta = jnp.array(np.log(np.exp(beta0) - 1.0))  # softplus^-1(beta0)
    if not quadratic:
        beta = jnp.array(-1e30)  # softplus(-inf) = 0 -> no quadratic term
    return dict(Wx=Wx, Wz=Wz, b=b, beta=beta)


def icnn_forward(p, x):
    """Scalar potential, (1, B).  Hidden layers use softplus; the head is linear."""
    L = len(p["Wx"])
    a = p["Wx"][0] @ x + p["b"][0][:, None]
    z = softplus(a)
    for l in range(1, L):
        a = softplus(p["Wz"][l]) @ z + p["Wx"][l] @ x + p["b"][l][:, None]
        z = softplus(a) if l < L - 1 else a
    return z + 0.5 * softplus(p["beta"]) * jnp.sum(x * x, axis=0, keepdims=True)


def icnn_input_grad(p, x):
    """Analytic d(out)/dx via manual reverse accumulation - no nested AD needed."""
    L = len(p["Wx"])
    # forward, caching pre-activations
    acts = []
    a = p["Wx"][0] @ x + p["b"][0][:, None]
    acts.append(a)
    z = softplus(a)
    for l in range(1, L):
        a = softplus(p["Wz"][l]) @ z + p["Wx"][l] @ x + p["b"][l][:, None]
        acts.append(a)
        z = softplus(a) if l < L - 1 else a
    # backward
    e = jnp.ones_like(acts[L - 1])          # d out / d a_L  (linear head)
    gx = p["Wx"][L - 1].T @ e
    for l in range(L - 1, 0, -1):
        s = softplus(p["Wz"][l]).T @ e      # d out / d z_{l-1}
        e = s * sigmoid(acts[l - 1])        # chain through softplus'
        gx = gx + p["Wx"][l - 1].T @ e
    return gx + softplus(p["beta"]) * x


# ----------------------------------------------------------------------------
# Sinkhorn  (src/sinkhorn.jl)
# ----------------------------------------------------------------------------
def pairwise_sqeuclidean(X, Y):
    sx = jnp.sum(X * X, axis=0)
    sy = jnp.sum(Y * Y, axis=0)
    return sx[:, None] + sy[None, :] - 2.0 * (X.T @ Y)


def sinkhorn_potentials(C, a=None, b=None, eps=0.1, n_iter=1000, tol=1e-9):
    """Log-domain Sinkhorn. Returns (f, g, n_used, err) with f,g the *scaled*
    potentials so that pi = exp((f_i + g_j - C_ij)/eps) * a_i * b_j."""
    n, m = C.shape
    a = np.full(n, 1.0 / n) if a is None else np.asarray(a, float)
    b = np.full(m, 1.0 / m) if b is None else np.asarray(b, float)
    loga, logb = np.log(a), np.log(b)
    f = np.zeros(n)
    g = np.zeros(m)
    C = np.asarray(C, float)
    err = np.inf
    used = n_iter
    for it in range(n_iter):
        # f_i = -eps * logsumexp_j( (g_j - C_ij)/eps + log b_j )
        M = (g[None, :] - C) / eps + logb[None, :]
        f_new = -eps * _lse(M, axis=1)
        M2 = (f_new[:, None] - C) / eps + loga[:, None]
        g_new = -eps * _lse(M2, axis=0)
        err = max(np.max(np.abs(f_new - f)), np.max(np.abs(g_new - g)))
        f, g = f_new, g_new
        if err < tol:
            used = it + 1
            break
    return f, g, used, err


def _lse(M, axis):
    mx = np.max(M, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(M - mx), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def sinkhorn_plan(C, f, g, a=None, b=None, eps=0.1):
    n, m = C.shape
    a = np.full(n, 1.0 / n) if a is None else a
    b = np.full(m, 1.0 / m) if b is None else b
    return np.exp((f[:, None] + g[None, :] - C) / eps) * a[:, None] * b[None, :]


def sinkhorn_value(C, a=None, b=None, eps=0.1, **kw):
    """Regularised OT value  <f,a> + <g,b>  (equals <pi,C> + eps*KL(pi|a x b))."""
    n, m = C.shape
    a = np.full(n, 1.0 / n) if a is None else a
    b = np.full(m, 1.0 / m) if b is None else b
    f, g, _, _ = sinkhorn_potentials(C, a, b, eps=eps, **kw)
    return float(f @ a + g @ b)


def sinkhorn_divergence(X, Y, eps=0.1, **kw):
    """Debiased Sinkhorn divergence (Feydy et al. 2019)."""
    v_xy = sinkhorn_value(np.asarray(pairwise_sqeuclidean(X, Y)), eps=eps, **kw)
    v_xx = sinkhorn_value(np.asarray(pairwise_sqeuclidean(X, X)), eps=eps, **kw)
    v_yy = sinkhorn_value(np.asarray(pairwise_sqeuclidean(Y, Y)), eps=eps, **kw)
    return v_xy - 0.5 * (v_xx + v_yy)


def exact_ot_cost(X, Y):
    """Exact W2^2 between equal-size uniform empirical measures (Hungarian)."""
    C = np.asarray(pairwise_sqeuclidean(X, Y))
    r, c = linear_sum_assignment(C)
    return C[r, c].mean()


# ----------------------------------------------------------------------------
# Generic MLP potential  (src/potentials.jl)
# ----------------------------------------------------------------------------
def mlp_init(key, dims, scale=1.0):
    ks = jax.random.split(key, len(dims) - 1)
    ps = []
    for i, k in enumerate(ks):
        fan_in, fan_out = dims[i], dims[i + 1]
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        ps.append((scale * jax.random.uniform(k, (fan_out, fan_in), minval=-lim, maxval=lim),
                   jnp.zeros((fan_out,))))
    return ps


def mlp_forward(ps, x, act=jax.nn.softplus):
    for W, b in ps[:-1]:
        x = act(W @ x + b[:, None])
    W, b = ps[-1]
    return W @ x + b[:, None]


# ----------------------------------------------------------------------------
# Adam  (mirrors Flux.Adam defaults)
# ----------------------------------------------------------------------------
def adam_init(params):
    return (jax.tree.map(jnp.zeros_like, params),
            jax.tree.map(jnp.zeros_like, params), 0)


def adam_step(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree.map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree.map(lambda vi, gi: b2 * vi + (1 - b2) * gi * gi, v, grads)
    mh = jax.tree.map(lambda mi: mi / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda vi: vi / (1 - b2 ** t), v)
    params = jax.tree.map(lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps),
                          params, mh, vh)
    return params, (m, v, t)
