"""Line-by-line transcription of the FINAL Julia source, re-verified.

Mirrors src/icnn.jl (_sweep with Wz[l-1] indexing), src/sinkhorn.jl (array
shapes) and src/maps.jl (barycentric weights) exactly as written, to catch
transcription/indexing errors introduced while porting.
"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sp, sig = jax.nn.softplus, jax.nn.sigmoid
ok = lambda n, c, e="": print(f"{'PASS' if c else 'FAIL'}  {n}  {e}")
rng = np.random.default_rng(7)

# ---- src/icnn.jl ----------------------------------------------------------
def make(dim, widths, seed=0, init_scale=0.1, quad=1.0):
    r = np.random.default_rng(seed)
    L = len(widths); Wx=[]; Wz=[]; b=[]
    prev = dim
    for l in range(L):
        lim = np.sqrt(6/(widths[l]+dim))
        Wx.append(jnp.array(init_scale*r.uniform(-lim,lim,(widths[l],dim))))
        if l > 0:                                   # Wz has length L-1
            limz = np.sqrt(6/(widths[l]+prev))
            Wz.append(jnp.array(r.uniform(-limz,limz,(widths[l],prev))))
        b.append(jnp.zeros(widths[l]))
        prev = widths[l]
    logbeta = jnp.array([np.log(np.expm1(quad))])
    return dict(Wx=Wx, Wz=Wz, b=b, logbeta=logbeta)

def forward(f, x):                                  # (f::ICNN)(x)
    L = len(f["Wx"])
    a = f["Wx"][0] @ x + f["b"][0][:,None]
    for l in range(1, L):                           # julia l = 2:L
        z = sp(a)
        a = sp(f["Wz"][l-1]) @ z + f["Wx"][l] @ x + f["b"][l][:,None]
    return a + (0.5*sp(f["logbeta"]))[:,None]*jnp.sum(x*x,axis=0,keepdims=True)

def sweep(f, x, z, l, L):                           # 1-based, as in Julia
    if l == 1:
        a = f["Wx"][0] @ x + f["b"][0][:,None]
    else:
        a = sp(f["Wz"][l-2]) @ z + f["Wx"][l-1] @ x + f["b"][l-1][:,None]
    if l == L:
        e  = jnp.ones_like(a)
        gx = f["Wx"][L-1].T @ e
        bz = None if L == 1 else sp(f["Wz"][L-2]).T @ e
        return a, gx, bz
    out, gx_rest, bz_next = sweep(f, x, sp(a), l+1, L)
    e  = bz_next * sig(a)
    gx = gx_rest + f["Wx"][l-1].T @ e
    bz = None if l == 1 else sp(f["Wz"][l-2]).T @ e
    return out, gx, bz

def input_gradient(f, x):
    _, gx, _ = sweep(f, x, None, 1, len(f["Wx"]))
    return gx + sp(f["logbeta"])[:,None]*x

print("transcription check: src/icnn.jl")
for dim, widths in [(2,[16,16,1]), (4,[32,32,32,1]), (3,[1]), (5,[8,8,8,8,8,1]), (2,[7,1])]:
    f = make(dim, widths)
    X = jnp.array(rng.normal(size=(dim,6)))
    ga = input_gradient(f, X)
    gr = jax.jacrev(lambda z: jnp.sum(forward(f,z)))(X)
    d  = float(jnp.max(jnp.abs(ga-gr)))
    ok(f"  grad dim={dim} widths={widths}", d < 1e-10, f"max|diff|={d:.2e}")
    # convexity of the transcribed forward pass
    A = jnp.array(rng.normal(size=(dim,200))); B = jnp.array(rng.normal(size=(dim,200)))
    viol = float(jnp.max(forward(f,0.5*(A+B))[0] - 0.5*(forward(f,A)[0]+forward(f,B)[0])))
    ok(f"  convex dim={dim} widths={widths}", viol <= 1e-10, f"max violation={viol:.2e}")
# derivative w.r.t. parameters must exist (this is the whole point of _sweep)
f = make(3,[16,16,1]); X = jnp.array(rng.normal(size=(3,32)))
gp = jax.grad(lambda p, X: jnp.mean(jnp.sum(X*input_gradient(p,X),axis=0)))(f, X)
ok("  d/dparams finite", all(bool(jnp.all(jnp.isfinite(w))) for w in gp["Wx"]+gp["Wz"]))

# ---- src/sinkhorn.jl (shapes exactly as written) --------------------------
print("\ntranscription check: src/sinkhorn.jl")
def lse(A, axis):
    mx = np.max(A, axis=axis, keepdims=True)
    mxs = np.where(np.isfinite(mx), mx, 0.0)
    return mxs + np.log(np.sum(np.exp(A-mxs), axis=axis, keepdims=True))

def potentials(C, a=None, b=None, eps=0.1, n_iter=2000, tol=1e-9):
    n, m = C.shape
    av = np.full(n,1/n) if a is None else np.asarray(a,float)
    bv = np.full(m,1/m) if b is None else np.asarray(b,float)
    loga, logb = np.log(av), np.log(bv)
    f = np.zeros(n); g = np.zeros(m); conv=False; used=n_iter
    for it in range(n_iter):
        M  = (g[None,:] - C)/eps + logb[None,:]          # julia: (g' .- Cf)./eps .+ logb'
        fn = -eps*np.ravel(lse(M, 1))
        M2 = (fn[:,None] - C)/eps + loga[:,None]         # julia: (fnew .- Cf)./eps .+ loga
        gn = -eps*np.ravel(lse(M2, 0))
        err = max(np.max(np.abs(fn-f)), np.max(np.abs(gn-g)))
        f, g = fn, gn
        if err < tol: conv=True; used=it+1; break
    return f, g, used, conv

X = rng.normal(size=(2,50)); Y = rng.normal(size=(2,50)) + np.array([[3.],[0.]])
C = ((X*X).sum(0)[:,None] + (Y*Y).sum(0)[None,:] - 2*X.T@Y).clip(0)
f,g,it,conv = potentials(C, eps=0.1)
P = np.exp((f[:,None]+g[None,:]-C)/0.1)*(1/50)*(1/50)
ok("  shapes/marginals", conv and max(np.max(np.abs(P.sum(1)-1/50)), np.max(np.abs(P.sum(0)-1/50)))<1e-9,
   f"iters={it} row err={np.max(np.abs(P.sum(1)-1/50)):.2e}")
from scipy.optimize import linear_sum_assignment
r_,c_ = linear_sum_assignment(C)
ok("  <P,C> near exact OT", abs(np.sum(P*C)-C[r_,c_].mean())/C[r_,c_].mean() < 0.02,
   f"{np.sum(P*C):.4f} vs exact {C[r_,c_].mean():.4f}")
# non-uniform weights
a = rng.random(50); a/=a.sum(); b = rng.random(50); b/=b.sum()
f2,g2,_,cv2 = potentials(C, a, b, eps=0.1)
P2 = np.exp((f2[:,None]+g2[None,:]-C)/0.1)*a[:,None]*b[None,:]
ok("  non-uniform marginals", cv2 and np.max(np.abs(P2.sum(1)-a))<1e-9 and np.max(np.abs(P2.sum(0)-b))<1e-9)
# value identity  <f,a>+<g,b> == <P,C> + eps*KL(P|ab')
val = f@np.full(50,1/50) + g@np.full(50,1/50)
kl = np.sum(P*np.log(np.maximum(P/((1/50)*(1/50)),1e-300))) - P.sum() + 1
ok("  value == <P,C> + eps*KL", abs(val - (np.sum(P*C)+0.1*kl)) < 1e-6,
   f"{val:.6f} vs {np.sum(P*C)+0.1*kl:.6f}")

# ---- src/maps.jl barycentric ---------------------------------------------
print("\ntranscription check: src/maps.jl barycentric weights")
ux = rng.normal(size=25); vy = rng.normal(size=40)
Xb = rng.normal(size=(2,25)); Yb = rng.normal(size=(2,40))
Cb = ((Xb*Xb).sum(0)[:,None] + (Yb*Yb).sum(0)[None,:] - 2*Xb.T@Yb).clip(0)
logw = (ux[:,None] + vy[None,:] - Cb)/0.1
logw = logw - lse(logw, 1)
T = Yb @ np.exp(logw).T
ok("  weights normalised per row", np.max(np.abs(np.exp(logw).sum(1)-1)) < 1e-12)
ok("  output shape", T.shape == Xb.shape, str(T.shape))
ok("  invariant to constant shift of both potentials",
   np.allclose(T, Yb @ np.exp(((ux+5)[:,None]+(vy+5)[None,:]-Cb)/0.1 -
                              lse(((ux+5)[:,None]+(vy+5)[None,:]-Cb)/0.1,1)).T))
