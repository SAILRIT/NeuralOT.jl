# Contributing to NeuralOT.jl

Thanks for your interest! Contributions of all kinds are welcome: bug
reports, documentation fixes, new examples, and new algorithms.

## Reporting bugs

Open a GitHub issue with:

- A minimal working example that reproduces the problem
- Julia version (`versioninfo()`) and `Pkg.status()` output
- What you expected vs. what happened

## Proposing changes

1. Fork and branch off `main`.
2. For non-trivial changes, open an issue first to discuss scope.
3. Run the tests locally: `julia --project -e 'using Pkg; Pkg.test()'`.
4. Add tests covering new behaviour.
5. Format code with consistent 4-space indents (no trailing whitespace).
6. Update docstrings and, if relevant, the docs under `docs/src/`.
7. Submit a PR describing the change and linking any related issue.

## Adding a new algorithm

The package is structured so each method lives in its own file under
`src/`, exports a single solver function returning a `NeuralOTResult`,
and dispatches through `monge_map` for inference. To add a new method:

1. Create `src/your_method.jl` implementing `your_solver(...)` that
   returns `NeuralOTResult(models, losses, :your_method, config)`.
2. Add `include("your_method.jl")` to `src/NeuralOT.jl` and export the
   solver.
3. Add a branch to `monge_map` if your method defines a transport map.
4. Add a test in `test/runtests.jl`.
5. Add a docs page in `docs/src/methods/`.

## Code of conduct

Be kind. This project follows the
[Julia Community Standards](https://julialang.org/community/standards/).
