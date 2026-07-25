# Troubleshooting

## `solve_w2` loss oscillates and does not decrease

Expected. The objective is a saddle point: `f` minimises while `g` maximises, so
the logged outer objective is not a convergence diagnostic. Judge the map
instead — [`sinkhorn_divergence`](@ref) between the pushforward and the target,
or [`transport_error`](@ref) against a known map.

## `solve_w2` produces a nearly constant map

Usually too few inner steps: the inner maximisation approximates a convex
conjugate, and a poor approximation biases the map. Raise `inner_steps` (5–15 is
the usual range) or `lr_inner`. Check that `quadratic` is greater than zero —
the strong-convexity term is what keeps the map well scaled at initialisation.

## Training diverges or produces `NaN`

- Lower `lr`. The defaults (`1e-3`) suit standardised data; unstandardised data
  with large magnitudes needs less.
- For `solve_dual` with `formulation = :exp`, the exponential penalty is
  unforgiving: switch to the default `:logsumexp`, which is self-normalising and
  far better conditioned.
- Standardise your data. Optimal transport is not scale-invariant and neither is
  Adam's default step size.

## `solve_dual`'s dual value is far from the true transport cost

The dual value is the *regularised* cost, which exceeds `W₂²` by an amount
growing with `ε`. Reduce `ε` for a closer match, and expect slower convergence.
Also confirm the potentials have enough capacity: two hidden layers of 128 units
is a reasonable starting point in low dimension.

## The flow pushforward is blurry or misses modes

- Increase `n_flow_steps` in [`monge_map`](@ref); an under-resolved ODE blurs
  the target. Compare `:euler` against `:rk4` — if they disagree, use more steps.
- Train longer. Flow matching is a regression problem and benefits from more
  steps far past the point where the loss curve looks flat.
- Try `coupling = :ot`, then [`rectify`](@ref) to straighten the paths.

## Sinkhorn results look wrong at small `ε`

Check `sinkhorn_potentials(...).converged`. The default budget is not enough for
small `ε` on separated measures. Either raise `n_iter` or raise `ε`.

## "Mutating arrays is not supported"

Something inside a differentiated function writes into an array. Build results
functionally instead — `vcat`, broadcasting, `reduce` — or move the computation
outside the gradient, as the solvers do with cost matrices. This is also why
[`input_gradient`](@ref) for an `ICNN` is written as an explicit non-mutating
reverse sweep.

## An `ICNN` rejects my activation

Input convexity requires a convex, non-decreasing activation, and the analytic
gradient needs its derivative. `softplus` (the default), `relu` and `elu` are
recognised; `tanh` is not convex and is rejected deliberately. For anything
else, pass `dactivation` explicitly.
