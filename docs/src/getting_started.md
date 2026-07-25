# Getting started

## Samplers

Every solver takes *samplers*: functions mapping a batch size to a `dim × n`
matrix. This is what lets the solvers train on unlimited fresh data rather than
a fixed array.

```@example gs
using NeuralOT
sample_mu = gaussian_sampler(2; sigma = 0.5)
size(sample_mu(128))
```

Wrapping a fixed data set is a one-liner:

```julia
data = randn(Float32, 2, 10_000)                    # your data, dim x N
sample_data(n) = data[:, rand(1:size(data, 2), n)]  # bootstrap batches
```

Built-in toy distributions: [`gaussian_sampler`](@ref), [`two_moons`](@ref),
[`eight_gaussians`](@ref), [`checkerboard`](@ref), [`swiss_roll`](@ref),
[`circles`](@ref), [`uniform_box`](@ref).

## Training

```@example gs
sample_nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)
res = solve_dual(sample_mu, sample_nu; dim = 2, steps = 300, batch = 128,
                 hidden = [64, 64], seed = 0)
res.method, length(res.losses)
```

All solvers accept `seed` (reproducible initialisation), `verbose` (progress
printing), `log_every`, `eval_batch` (an independent held-out loss) and
`callback`.

## Monitoring and early stopping

A callback receives `(step, loss, models)` and stops training when it returns
`false`:

```@example gs
history = Float64[]
res2 = solve_dual(sample_mu, sample_nu; dim = 2, steps = 2_000, batch = 64,
                  hidden = [32], log_every = 50, seed = 0,
                  callback = function (step, loss, models)
                      push!(history, loss)
                      loss > -8.5          # stop once the dual value exceeds 8.5
                  end)
last(res2.logged_steps)
```

## Using the result

```@example gs
X = sample_mu(64)
T = monge_map(res, X)
size(T)
```

[`NeuralOTResult`](@ref) carries the models, the loss history, the wall-clock
time and the full configuration, so a saved result documents itself.

## Reproducibility

`seed` seeds a *private* RNG for network initialisation and for any sampling the
solver does internally; it never touches the global RNG. To make the data stream
reproducible too, give the samplers their own RNG:

```@example gs
using Random
s = gaussian_sampler(2; rng = MersenneTwister(0))
s(3) == gaussian_sampler(2; rng = MersenneTwister(0))(3)
```
