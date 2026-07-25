# Dual optimal transport (Seguy et al., 2018)

## The problem

For measures `μ`, `ν` and cost `c`, the entropy-regularised Kantorovich dual is

```math
\max_{u, v} \; \mathbb{E}_\mu[u(X)] + \mathbb{E}_\nu[v(Y)]
  - \varepsilon\, \mathbb{E}_{\mu \otimes \nu}
    \left[\exp\!\left(\frac{u(X) + v(Y) - c(X, Y)}{\varepsilon}\right)\right].
```

Both expectations are over samples, so nothing here needs a full cost matrix
over the data set: the method scales to large sample sizes and high dimension.
[`solve_dual`](@ref) parameterises `u` and `v` as MLPs and maximises stochastically.

## Two formulations

`formulation = :exp` is the objective above verbatim. `formulation = :logsumexp`
(the default) replaces the penalty with

```math
\varepsilon \log \mathbb{E}\left[\exp\!\left(\frac{u + v - c}{\varepsilon}\right)\right],
```

the self-normalising variant. It is invariant to adding a constant to both
potentials and is far better conditioned: the raw exponential penalty explodes
whenever `u + v` overshoots `c` early in training.

The difference is not subtle. On a 2-D Gaussian pair with `W₂² = 9.0`, matched
budgets and `ε = 0.1`:

| formulation | dual value reached | Sinkhorn reference |
|---|---|---|
| `:logsumexp` | 9.23 | 9.12 |
| `:exp` | 2.60 | 9.12 |

## Recovering a map

The dual gives potentials, not a map. Two routes:

[`barycentric_map`](@ref) averages target samples under the learned plan,

```math
T(x) = \sum_j w_j(x)\, y_j, \qquad
w_j(x) \propto \exp\!\big((u(x) + v(y_j) - c(x, y_j))/\varepsilon\big),
```

which is accurate and works for any cost. [`entropic_map`](@ref) is the
closed-form version for the squared cost,

```math
T(x) = x - \tfrac{1}{2}\nabla u(x),
```

which follows from `∇u(x) = E[∇ₓ c(x, Y) | x] = 2(x - T(x))`.

!!! warning "Changed in v0.2"
    v0.1 used `x - ∇u(x)/(2ε)`. The extra `1/ε` is wrong. On the Gaussian
    benchmark the RMSE against the exact Brenier map was 27.14 with the old
    constant and 0.076 with the correct one.

## Example

```julia
using NeuralOT
mu = gaussian_sampler(2; sigma = 0.5)
nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)

res = solve_dual(mu, nu; dim = 2, ε = 0.1, steps = 3_000, batch = 256,
                 hidden = [128, 128], lr = 1e-3, seed = 0, verbose = true)

-res.losses[end]                          # estimated regularised OT cost
barycentric_map(res, mu(256), nu(4_096))  # push samples forward
```

## Choosing `ε`

Smaller `ε` gives a sharper plan and a value closer to the unregularised
transport cost, at the price of slower convergence and a harder optimisation
problem. `ε` should be read relative to the scale of your cost: for data with
unit variance and squared cost, `0.01`–`0.5` is the useful range.
