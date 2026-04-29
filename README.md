# NeuralOT.jl

[![CI](https://github.com/YOUR_USERNAME/NeuralOT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/YOUR_USERNAME/NeuralOT.jl/actions/workflows/CI.yml)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YOUR_USERNAME.github.io/NeuralOT.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Neural optimal transport in Julia.** Estimate Monge maps, dual potentials,
and transport-based generative models using neural networks.

NeuralOT.jl fills a gap in the Julia OT ecosystem: while
[OptimalTransport.jl](https://github.com/JuliaOptimalTransport/OptimalTransport.jl)
provides discrete-measure solvers (Sinkhorn, exact LP, unbalanced
OT), there is no Julia package dedicated to *neural* OT — continuous methods
that scale to high dimensions by parameterising potentials or maps as neural
networks. This package implements three established families:

| Method | Reference | Best for |
|---|---|---|
| `solve_dual` | Seguy et al., ICLR 2018 | High-dim entropic OT from samples |
| `solve_w2` | Makkuva et al., ICML 2020 | W₂ Monge maps via ICNNs |
| `flow_match` | Lipman et al., ICLR 2023 | Transport-based generative models |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/YOUR_USERNAME/NeuralOT.jl")
```

After registration in the Julia General registry:

```julia
Pkg.add("NeuralOT")
```

## Quickstart

Transport samples from one Gaussian to another using a W₂ Monge map:

```julia
using NeuralOT, Random
Random.seed!(0)

# Source and target samplers (each returns dim × n matrices)
sample_μ(n) = 0.5f0 .* randn(Float32, 2, n)
sample_ν(n) = 0.5f0 .* randn(Float32, 2, n) .+ Float32[3.0; 0.0]

# Train via the Makkuva et al. ICNN saddle-point formulation
result = solve_w2(sample_μ, sample_ν;
                  dim=2, widths=[64, 64, 1],
                  steps=2_000, inner_steps=10, lr=1e-4, seed=42)

# Push source samples through the learned map
X = sample_μ(500)
T_X = monge_map(result, X)       # ≈ samples from ν

# Evaluate transport quality
using Statistics
sinkhorn_divergence(T_X, sample_ν(500); ε=0.1)   # should be small
```

## Which method should I use?

- **`solve_dual`** — you just need the OT cost/potentials between samples from
  high-dimensional distributions. No map recovery needed, or map recovery via
  the barycentric projection is enough.
- **`solve_w2`** — you need a valid Monge map (T: R^d → R^d) with provable
  convexity structure, squared-Euclidean cost. The gold standard for neural
  OT map estimation.
- **`flow_match`** — you want a generative model that transports noise to
  data, or you care about simulation-free training on very large datasets.

See [`examples/`](examples/) for fully worked scripts.

## Comparison to the Julia OT ecosystem

NeuralOT.jl is designed to complement, not replace, existing packages:

- **OptimalTransport.jl** — discrete/entropic OT with exact solvers
- **ExactOptimalTransport.jl** — exact Kantorovich LP
- **PythonOT.jl** — wrapper around POT

Use those when you have weighted point clouds and need discrete solvers. Use
NeuralOT.jl when you have sample access to continuous distributions and need
a parameterised map or potential that generalises beyond the training points.



## License

MIT. See [LICENSE](LICENSE).
