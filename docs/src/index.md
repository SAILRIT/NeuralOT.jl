# NeuralOT.jl

Neural optimal transport for Julia: Monge maps, dual potentials and
transport-based generative models, with the discrete solvers and closed-form
references needed to check that the result is right.

## Why neural OT

Discrete solvers compute a transport plan between two *fixed* point clouds. That
plan says nothing about a new point. Neural OT instead learns a parameterised
map or potential from samples, so it

- generalises to points never seen during training,
- scales to high dimension, where a full cost matrix is impossible,
- gives a reusable object you can apply, invert, and compose.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/YOUR_USERNAME/NeuralOT.jl")
```

## The three solvers

```@example intro
using NeuralOT

sample_mu = gaussian_sampler(2; sigma = 0.5)
sample_nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)

res = solve_w2(sample_mu, sample_nu; dim = 2, widths = [64, 64, 1],
               steps = 500, inner_steps = 5, seed = 1)

X = sample_mu(200)
T = monge_map(res, X)
round(sinkhorn_divergence(T, sample_nu(200)); digits = 3)
```

| Solver | Learns | Map recovery |
|---|---|---|
| [`solve_dual`](@ref) | dual potentials `u`, `v` | [`barycentric_map`](@ref), [`entropic_map`](@ref) |
| [`solve_w2`](@ref) | convex potentials `f`, `g` | [`monge_map`](@ref) = `∇g`, [`inverse_map`](@ref) = `∇f` |
| [`flow_match`](@ref) | vector field `v(t, x)` | [`monge_map`](@ref) integrates the ODE |

## Conventions

- **Column-major data.** A batch of `n` points in `d` dimensions is a `d × n`
  matrix. Samplers are functions `n -> Matrix{Float32}` of that shape.
- **`Float32` throughout.** Inputs are converted on entry.
- **Costs are not halved.** [`SqEuclideanCost`](@ref) is `‖x - y‖²`, so the
  entropic barycentric map is `x - ∇u(x)/2`.

## Checking your results

Because the package ships closed-form Gaussian optimal transport, you can score
a solver rather than eyeball it:

```@example intro
ref = gaussian_ot(zeros(2), 0.25 * [1.0 0.0; 0.0 1.0],
                  [3.0, 0.0], 0.25 * [1.0 0.0; 0.0 1.0])
round(ref.w2; digits = 3), round(transport_error(T, ref.map(X)); digits = 3)
```
