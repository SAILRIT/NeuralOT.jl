# Evaluating a transport map

A neural OT model can look plausible and still be wrong. This package provides
three levels of check.

## 1. Moments

[`moment_error`](@ref) compares empirical means and covariances — cheap, and it
catches gross failures like mode collapse or a map that only learned a shift.

```@example ev
using NeuralOT
mu = gaussian_sampler(2; sigma = 0.5)
nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)
res = flow_match(mu, nu; dim = 2, hidden = [64, 64], steps = 800, seed = 0)
T = monge_map(res, mu(500))
moment_error(T, nu(500))
```

## 2. Distributional distances

[`sinkhorn_divergence`](@ref) is the debiased entropic divergence: zero exactly
when the two samples coincide, non-negative otherwise, and interpolating between
maximum mean discrepancy (large `ε`) and squared Wasserstein distance (small
`ε`). [`energy_distance`](@ref) and [`mmd`](@ref) are cheaper alternatives that
need no iteration.

```@example ev
round(sinkhorn_divergence(T, nu(500); ε = 0.5); digits = 4)
```

!!! warning "Choose `n_iter` to match `ε`"
    Sinkhorn converges more slowly as `ε` shrinks. Two Gaussians three standard
    deviations apart need roughly 1900 iterations at `ε = 0.1` to reach
    `tol = 1e-9`. Call [`sinkhorn_potentials`](@ref) directly and inspect
    `sol.converged` if you are unsure — an unconverged divergence is biased.

## 3. Against the exact answer

For Gaussians the optimal map and the exact `W₂²` are known, so a solver can be
*scored*:

```@example ev
S = 0.25 * [1.0 0.0; 0.0 1.0]
ref = gaussian_ot(zeros(2), S, [3.0, 0.0], S)
X = mu(500)
round(transport_error(monge_map(res, X), ref.map(X)); digits = 3)
```

This is what the test suite does, and what `validation/` in the repository
records for each solver. When you add a method, add a check of this kind: a test
that only asserts "the code ran" will not catch a map that is off by a constant
factor — which is exactly the bug that v0.1's entropic map contained.

## Which metric when

| Situation | Use |
|---|---|
| quick sanity check during training | [`moment_error`](@ref) |
| comparing two models on the same target | [`sinkhorn_divergence`](@ref) |
| large batches, speed matters | [`energy_distance`](@ref) or [`mmd`](@ref) |
| Gaussian problem, want absolute accuracy | [`gaussian_ot`](@ref) + [`transport_error`](@ref) |
