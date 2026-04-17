# Dual OT (Seguy et al., 2018)

## Problem

Given two distributions μ and ν on ℝᵈ with sample access, and a cost
function `c(x, y)` (default: squared Euclidean), we solve the entropy-
regularised OT problem

```math
\mathrm{OT}_\varepsilon(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)}
  \int c(x, y)\, \mathrm{d}\pi(x, y)
  + \varepsilon \, \mathrm{KL}(\pi \,\|\, \mu \otimes \nu).
```

The Fenchel dual admits a saddle-point formulation that can be optimised
stochastically by sampling mini-batches:

```math
\mathrm{OT}_\varepsilon(\mu, \nu) = \sup_{u, v}
  \mathbb{E}_\mu[u(X)] + \mathbb{E}_\nu[v(Y)]
  - \varepsilon\, \mathbb{E}_{\mu \otimes \nu}\!\left[
    e^{(u(X) + v(Y) - c(X, Y))/\varepsilon}
  \right].
```

Parameterising `u` and `v` as MLPs and maximising by SGD gives the Seguy
et al. (2018) algorithm. Unlike Sinkhorn on a cost matrix, this scales to
arbitrary sample sizes and dimensions — memory is O(batch²), not O(N²).

## Usage

```julia
using NeuralOT

sample_μ(n) = randn(Float32, 5, n)
sample_ν(n) = randn(Float32, 5, n) .+ Float32.(vcat(2.0, zeros(4)))

result = solve_dual(sample_μ, sample_ν;
                    dim=5, ε=0.5, hidden=[128, 128],
                    steps=3_000, batch=256, lr=1e-4)
```

The returned [`NeuralOTResult`](@ref) contains `result.models.u` and
`result.models.v` — scalar-valued networks you can evaluate on new points.

## Map recovery

Given the dual potential `u`, the barycentric map for squared cost is
approximately

```math
T(x) \approx x - \tfrac{1}{2\varepsilon}\, \nabla u(x),
```

which [`monge_map`](@ref) computes automatically when
`result.method === :dual`. For a Monge map with sharper theoretical
guarantees, use [`solve_w2`](@ref) instead.

## Tuning notes

- **ε (regularisation)**: smaller ε gives a sharper, closer-to-unregularised
  solution but destabilises training. Start at 0.1–1.0.
- **batch**: the penalty term is estimated over batch² pairs, so larger
  batches markedly reduce gradient variance.
- **steps**: typically 3–10k for low dimensions, more for d ≥ 50.
