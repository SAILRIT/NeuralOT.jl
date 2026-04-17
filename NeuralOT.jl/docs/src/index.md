# NeuralOT.jl

Neural optimal transport algorithms for Julia, built on [Flux.jl](https://fluxml.ai).

## Scope

NeuralOT.jl implements three families of neural OT methods that share a
common premise: rather than solving an OT problem on a fixed weighted point
cloud, parameterise the transport map or dual potentials as neural networks
that generalise to unseen points and scale to high dimensions by stochastic
optimisation.

Three entry points:

- [`solve_dual`](@ref) – entropic dual OT with MLP potentials
  (Seguy et al., 2018)
- [`solve_w2`](@ref) – W₂ Monge map via Input Convex Neural Networks
  (Makkuva et al., 2020)
- [`flow_match`](@ref) – conditional flow matching / rectified flow
  (Lipman et al., 2023; Liu et al., 2023)

## When to use NeuralOT.jl vs. OptimalTransport.jl

The Julia ecosystem already has
[OptimalTransport.jl](https://github.com/JuliaOptimalTransport/OptimalTransport.jl)
for discrete OT: weighted sums of Diracs, Sinkhorn iterations on an explicit
cost matrix, exact Kantorovich LPs, unbalanced OT, GPU-accelerated entropic
solvers. If that matches your problem, use it.

NeuralOT.jl is for the complementary regime:

- You have *sample access* to continuous distributions, not a fixed point
  cloud.
- You need a map or potential that *generalises* beyond the training samples
  (e.g. to push new samples at test time).
- Your dimension is high enough that materialising a pairwise cost matrix
  is impractical.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/YOUR_USERNAME/NeuralOT.jl")
```

## Minimal example

```julia
using NeuralOT
sample_μ(n) = randn(Float32, 2, n)
sample_ν(n) = randn(Float32, 2, n) .+ Float32[3.0; 0.0]

result = solve_w2(sample_μ, sample_ν; dim=2, steps=2_000, seed=0)
T_X = monge_map(result, sample_μ(100))
```

See the [Methods](methods/dual.md) pages for detailed walkthroughs of each
algorithm, and the [API](api.md) reference for full docstrings.
