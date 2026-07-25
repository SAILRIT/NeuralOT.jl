# Contributing to NeuralOT.jl

Thanks for considering a contribution.

## Getting started

```julia
julia --project=.
julia> using Pkg; Pkg.instantiate(); Pkg.test()
```

## Guidelines

- Every new solver or metric needs a test that checks it against something
  independently known to be true: a closed-form result (see `src/gaussian.jl`),
  an exact discrete solver, or a finite-difference check. Tests that only
  assert "the code ran" are not enough.
- Keep everything differentiable by Zygote: no array mutation inside anything
  that ends up under `Flux.gradient`. Build results functionally
  (`vcat`, broadcasting, `reduce`) rather than pre-allocating and writing.
- Public functions need a docstring with an `# Arguments` or
  `# Keyword arguments` section and, where useful, an `# Example`.
- Run the test suite before opening a pull request. If you add an exported
  name, add it to `docs/src/api.md`.

## Numerical conventions

- Data is stored **column-major**: a batch of `n` points in `d` dimensions is a
  `d x n` matrix. This matches Flux and keeps samples contiguous.
- Internally everything is `Float32`. Inputs are converted on entry.
- Costs are *not* halved: `SqEuclidean` is `||x - y||^2`, so the entropic map
  for that cost is `x - grad_u(x) / 2`.
