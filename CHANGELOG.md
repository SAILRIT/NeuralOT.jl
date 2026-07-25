# Changelog

## v0.2.0

### Fixed (correctness)

- **Entropic map constant.** `monge_map` on a `:dual` result used
  `x - grad_u(x) / (2 * eps)`. The barycentric projection of the entropic plan
  for the squared cost is `x - grad_u(x) / 2`; the extra `1/eps` made the map
  wrong by a factor of `1/eps` (a factor of 10 at the default `eps = 0.1`).
  Measured on a 2-D Gaussian pair with known Brenier map, the RMSE against the
  closed-form map dropped from **27.14** to **0.076**.
- **`solve_w2` no longer needs nested automatic differentiation.** `ICNN`
  gradients with respect to the input are now computed by an analytic reverse
  sweep (`input_gradient`) built only from non-mutating array operations, so a
  single level of reverse-mode AD suffices for the saddle-point objective. The
  old path relied on Zygote-over-Zygote and was documented as "experimental /
  may error".
- **Sinkhorn divergence is now the standard debiased one.** `_sinkhorn_cost`
  previously returned the raw transport cost `<pi, C>`, which makes
  `S(X, X) = 0` hold only trivially and is not positive-definite. Divergences
  are now built from the regularised OT *value* `<f, a> + <g, b>`, giving
  `S(X, X) = 0` exactly, symmetry to machine precision, and convergence to the
  exact optimal transport cost as `eps -> 0`.
- **`logu` shape bug** in the Sinkhorn loop: `loga .- logsumexp(M; dims=2)`
  produced an `(n, 1)` matrix rather than an `(n,)` vector, which only worked
  by accidental broadcasting.
- **Package layout.** The distributed archive duplicated `Project.toml`,
  `README.md`, `LICENSE` and `CONTRIBUTING.md` at the top level *and* inside a
  nested `NeuralOT.jl/` directory, with no `src/` at the root. The package root
  is now the archive root.
- **`ICNN` head is linear.** The output layer previously applied the
  activation, forcing the potential to be positive and its gradient to be
  bounded, which restricts the representable transport maps.

### Added

- `src/sinkhorn.jl` - log-domain Sinkhorn with non-uniform weights, tolerance
  based early stopping, `sinkhorn_potentials`, `sinkhorn_plan`,
  `sinkhorn_value`, `sinkhorn_cost`.
- `src/costs.jl` - `SqEuclidean`, `Euclidean`, `GenericCost` cost objects with
  an allocation-light squared-Euclidean path.
- `src/gaussian.jl` - closed-form Gaussian optimal transport (`gaussian_ot`,
  `gaussian_brenier_map`, `w2_gaussian`) used as ground truth in tests and
  benchmarks.
- `src/metrics.jl` - `energy_distance`, `mmd`, `moment_error`,
  `transport_error`.
- `src/datasets.jl` - toy samplers: `gaussian_sampler`, `two_moons`,
  `eight_gaussians`, `checkerboard`, `swiss_roll`, `circles`, `uniform_box`.
- `src/maps.jl` - `monge_map`, `inverse_map`, `barycentric_map`,
  `entropic_map`, `pushforward`.
- Optimal-transport minibatch couplings for flow matching (`coupling = :ot`,
  Tong et al. 2023) and `rectify` for reflow / path straightening.
- Heun and RK4 ODE integrators plus reverse-time integration.
- `ICNN` strong-convexity term `beta/2 * ||x||^2` with learnable `beta`, and an
  `init_scale` keyword; both substantially stabilise `solve_w2`.
- Training callbacks, `verbose` progress reporting, early stopping, held-out
  evaluation losses, wall-clock timing in `NeuralOTResult`.
- `formulation = :logsumexp | :exp` for `solve_dual`, and `cycle` regularisation
  for `solve_w2`.
- A test suite of 14 files covering every exported function, including
  end-to-end accuracy tests against closed-form Gaussian optimal transport.

### Changed

- `NeuralOTResult` gained `logged_steps`, `eval_losses` and `elapsed` fields.
  The first four fields keep their names, order and meaning.
- `solve_dual` updates both potentials from a single gradient evaluation.
- Flow-matching vector fields default to `swish`; time is concatenated as
  before, with optional Fourier time features.
