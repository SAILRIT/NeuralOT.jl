# Cross-validation

## Why this directory exists

This package was developed in an environment where Julia could not be
installed: the network policy blocked `julialang.org`, `julialang-s3.julialang.org`
and `pkg.julialang.org`, Julia publishes no Linux binaries as GitHub release
assets, and it is no longer in the Ubuntu archives. **The Julia test suite in
`test/` has therefore not been executed.** Run it before trusting the package:

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

To keep the work honest in the meantime, every algorithm was validated two
different ways, and the recorded output is in `results.txt`.

## 1. Numerical validation against independent ground truth

`reference.py` is a NumPy/JAX implementation that mirrors the Julia source
formula for formula. The check scripts then compare it against things that are
known to be correct independently of this package:

| Component | Checked against | Result |
|---|---|---|
| `input_gradient` for `ICNN` | JAX autodiff and central finite differences | agrees to 1.4e-13 (AD) and 1e-6 (FD) |
| differentiability of that gradient w.r.t. parameters | the nested-AD computation of the same quantity | agrees to 8.9e-16 |
| `ICNN` convexity | midpoint inequality and Hessian eigenvalues | no violation; min eigenvalue > 0 |
| `sinkhorn_*` | exact optimal transport by the Hungarian algorithm | 0.013% relative error at `eps = 0.005` |
| `sinkhorn_divergence` | its defining properties | `S(X,X) = 0` exactly, symmetric to 1.8e-15, non-negative |
| `gaussian_ot` | Monte-Carlo moments of the pushforward | mean error 0.009, `E‖x-T(x)‖² = W₂²` to 0.1% |
| `solve_dual` | closed-form Brenier map, Sinkhorn value | map RMSE 0.076; dual value 9.23 vs reference 9.12 |
| `solve_w2` | closed-form Brenier map | forward map RMSE 0.27, inverse 0.32 |
| `flow_match` | closed-form target, round-trip integration | divergence 0.019 vs 8.88 for the source; round trip 0.020 |

`validate_transcription.py` then re-derives the *final* Julia source — including
the `Wz[l-1]` index shift, the `_sweep` recursion and the exact array shapes in
the Sinkhorn loop — and re-runs the gradient, convexity and marginal checks on
five different architectures. This is what catches indexing errors introduced
during porting, as opposed to errors in the algorithm itself.

## 2. Static analysis of the Julia sources

`jlcheck.py` parses every `.jl` file with the tree-sitter Julia grammar and
reports parse errors, unresolved `include()` targets and exported names that are
never defined. `apicheck.py` verifies that every exported symbol has a docstring
and appears in `docs/src/api.md`, which is what Documenter's
`checkdocs = :exports` requires.

Both pass on the whole package.

## What this does and does not establish

**Established:** the mathematics is right, the formulas as written in the Julia
source produce the claimed numbers, the bug fixes are real and measurable, the
files parse, and the module structure is consistent.

**Not established:** that the Flux API calls behave as intended on a live
install. `Flux.@layer`, `Flux.trainable`, `Flux.setup` / `Flux.update!` return
conventions and Zygote's handling of the recursive `_sweep` are used according to
their documented contracts for Flux 0.14.12–0.16, but they have not been
exercised. If something fails on first run, it will almost certainly be in that
layer rather than in the numerics.

## Reproducing

```bash
pip install numpy scipy jax tree_sitter tree_sitter_julia
python3 validate_core.py
python3 validate_transcription.py
python3 validate_dual.py      # a few minutes
python3 validate_w2.py
python3 validate_flow.py
python3 jlcheck.py ..
python3 apicheck.py ..
```
