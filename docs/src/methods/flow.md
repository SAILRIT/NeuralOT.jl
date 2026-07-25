# Flow matching (Lipman et al., 2023)

## The idea

Train a vector field `v(t, x)` so that the ODE `dx/dt = v(t, x)` carries `μ` at
`t = 0` to `ν` at `t = 1`. Along the straight interpolant
`xₜ = (1-t)x₀ + t x₁` the target velocity is simply `x₁ - x₀`, so training is a
regression:

```math
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_1}
  \left\| v_\theta\big(t, (1-t)x_0 + t x_1\big) - (x_1 - x_0)\right\|^2.
```

No simulation during training, and the loss is an honest regression objective
that decreases monotonically — unlike the `solve_w2` saddle point.

## Couplings

With `x₀` and `x₁` drawn independently, the flow reproduces `ν` but its paths
are not the optimal-transport ones. `coupling = :ot` instead pairs each
minibatch through an entropic transport plan before regressing, which
straightens the paths and moves the induced map towards the Monge map
(Tong et al., 2023). The cost is one small Sinkhorn solve per step.

## Integration

[`integrate_flow`](@ref) offers `:euler`, `:heun` and `:rk4`. Higher order costs
more evaluations per step but tolerates far fewer steps. If `:euler` and `:rk4`
disagree on your trained model, you are under-resolving the ODE.

Integrating from `t = 1` back to `t = 0` inverts the map, which
[`inverse_map`](@ref) does for you.

## Reflow

[`rectify`](@ref) retrains a fresh field on the pairs `(x₀, T(x₀))` induced by an
already-trained flow. Because those pairs are already coupled, the new field's
paths are straighter, and after one or two rounds very few integration steps
suffice.

```julia
using NeuralOT
mu = gaussian_sampler(2; sigma = 0.5)
nu = eight_gaussians(; radius = 3.0)

res1 = flow_match(mu, nu; dim = 2, hidden = [128, 128], steps = 5_000, seed = 0)
res2 = rectify(res1, mu; dim = 2, hidden = [128, 128], steps = 5_000, seed = 1)

X = mu(1_000)
monge_map(res2, X; n_flow_steps = 4, solver = :euler)   # few steps now suffice
```

## Time conditioning

By default `t` is concatenated to the state. `n_fourier > 0` adds
`sin`/`cos` features at frequencies `1 … n_fourier`, which helps when the
velocity changes quickly in time — typically for multimodal targets.

## Noise

`sigma > 0` perturbs the interpolant with Gaussian noise, giving a probability
path of positive width. The regression target is unchanged. This makes the
learned field smoother away from the data.
