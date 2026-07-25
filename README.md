# NeuralOT.jl

[![CI](https://github.com/YOUR_USERNAME/NeuralOT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/YOUR_USERNAME/NeuralOT.jl/actions/workflows/CI.yml)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YOUR_USERNAME.github.io/NeuralOT.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Neural optimal transport in Julia.** Estimate Monge maps, dual potentials and
transport-based generative models with neural networks — and check the answer
against exact optimal transport, because the package ships the references too.

NeuralOT.jl fills a gap in the Julia OT ecosystem: while
[OptimalTransport.jl](https://github.com/JuliaOptimalTransport/OptimalTransport.jl)
provides excellent discrete-measure solvers, there is no Julia package dedicated
to *neural* OT — continuous methods that scale to high dimensions by
parameterising potentials or maps as neural networks.

| Method | Reference | Best for |
|---|---|---|
| `solve_dual` | Seguy et al., ICLR 2018 | high-dimensional entropic OT from samples |
| `solve_w2` | Makkuva et al., ICML 2020 | W₂ Monge maps via input convex networks |
| `flow_match` | Lipman et al., ICLR 2023 | transport-based generative models |
| `rectify` | Liu et al., ICLR 2023 | straightening a trained flow |

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/YOUR_USERNAME/NeuralOT.jl")
```

## Quickstart

```julia
using NeuralOT

# Samplers return dim x n matrices, one sample per column.
sample_mu = gaussian_sampler(2; sigma = 0.5)
sample_nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)

result = solve_w2(sample_mu, sample_nu; dim = 2, widths = [64, 64, 1],
                  steps = 2_000, inner_steps = 8, lr = 1e-3, seed = 42)

X   = sample_mu(500)
T_X = monge_map(result, X)                       # push samples forward
sinkhorn_divergence(T_X, sample_nu(500))         # should be small
```

Because this problem is Gaussian, the exact answer is available for comparison:

```julia
ref = gaussian_ot(zeros(2), 0.25I(2), [3.0, 0.0], 0.25I(2))
ref.w2                              # 9.0, the true squared Wasserstein distance
transport_error(T_X, ref.map(X))    # RMSE against the exact Brenier map
```

## Which method should I use?

- **`solve_dual`** — you need the OT cost or the potentials between samples from
  high-dimensional distributions. Map recovery via `barycentric_map` is
  available and accurate; `entropic_map` gives a closed-form approximation.
- **`solve_w2`** — you need a genuine Monge map with convexity structure for the
  squared cost. `monge_map` gives `μ → ν` and `inverse_map` gives `ν → μ`.
- **`flow_match`** — you want a generative model transporting noise to data, or
  simulation-free training on very large data sets. Use `coupling = :ot` to
  bring the learned coupling closer to the optimal one, and `rectify` to
  straighten the paths so that fewer integration steps suffice.

See [`examples/`](examples/) for worked scripts, including a benchmark that
scores all three solvers against closed-form Gaussian optimal transport.

## What changed in v0.2

v0.2 is a substantial rewrite. The correctness-relevant parts:

- **The entropic map constant was wrong.** `monge_map` on a `:dual` result
  divided `∇u` by `2ε` instead of `2`. On a Gaussian benchmark with a known
  exact map, the RMSE went from **27.14** to **0.076**.
- **`solve_w2` no longer needs nested automatic differentiation.** ICNN input
  gradients are now computed analytically, so the saddle-point objective needs
  only one level of reverse-mode AD. In v0.1 this path was documented as
  experimental and the test for it was `@test_broken`.
- **The Sinkhorn divergence is now the standard debiased one**, built from the
  regularised OT value rather than the raw transport cost, so `S(X, X) = 0`
  exactly and `S ≥ 0`.

Plus discrete Sinkhorn solvers, cost objects, closed-form Gaussian references,
toy data sets, extra metrics, OT couplings, higher-order ODE solvers, callbacks
and a 14-file test suite. See [CHANGELOG.md](CHANGELOG.md) for the full list.

## Validation

Every algorithm in this package was checked numerically against something
independently known to be correct — analytic gradients against finite
differences and autodiff, Sinkhorn against exact Hungarian-algorithm optimal
transport, and all three solvers against closed-form Gaussian optimal transport.
The cross-validation scripts and their recorded output live in
[`validation/`](validation/); `validation/README.md` explains what was measured
and reports the numbers.

## Comparison to the Julia OT ecosystem

- **OptimalTransport.jl** — discrete/entropic OT with mature solvers
- **ExactOptimalTransport.jl** — exact Kantorovich LP
- **PythonOT.jl** — wrapper around POT

Use those when you have weighted point clouds and need discrete solvers. Use
NeuralOT.jl when you have sample access to continuous distributions and need a
parameterised map or potential that generalises beyond the training points.

## License

MIT. See [LICENSE](LICENSE).
