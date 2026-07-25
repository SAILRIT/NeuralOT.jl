# Evaluation metrics for comparing a pushforward T#mu against the target nu.

"""
    sinkhorn_divergence(X, Y; ε=0.1, n_iter=2_000, cost=SqEuclideanCost(), tol=1e-9, a, b)

Debiased Sinkhorn divergence between the empirical measures on the columns of
`X` (`d x n`) and `Y` (`d x m`):

```math
S_\\varepsilon(X, Y) = \\mathrm{OT}_\\varepsilon(X, Y)
    - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(X, X)
    - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(Y, Y)
```

where each term is the regularised OT *value* (Feydy et al., 2019). This
debiasing is what makes `S(X, X) = 0` exactly and `S >= 0`, and it interpolates
between the maximum-mean-discrepancy (large `ε`) and the squared Wasserstein
distance (small `ε`).

Both `ε` and `epsilon` are accepted as the keyword name.

# Keyword arguments
- `ε = 0.1` (or `epsilon`): entropic regularisation.
- `n_iter = 2_000`: maximum Sinkhorn iterations per term.
- `tol = 1e-9`: stopping tolerance on the potentials.
- `cost = SqEuclideanCost()`: ground cost.
- `a`, `b`: optional marginal weights, uniform by default.

!!! note
    Small `ε` needs more iterations. If you request `ε` below roughly `0.01`,
    raise `n_iter` accordingly; the returned value is otherwise biased by
    incomplete convergence.

# Example
```julia
sinkhorn_divergence(monge_map(res, X), Y; ε = 0.05)
```
"""
function sinkhorn_divergence(X::AbstractMatrix, Y::AbstractMatrix;
                             ε::Real = 0.1, epsilon::Real = ε,
                             n_iter::Int = 2_000, tol::Real = 1e-9,
                             cost = SqEuclideanCost(), a = nothing, b = nothing)
    c = _as_cost(cost)
    kw = (epsilon = epsilon, n_iter = n_iter, tol = tol)
    vxy = sinkhorn_value(cost_matrix(c, X, Y), a, b; kw...)
    vxx = sinkhorn_value(cost_matrix(c, X, X), a, a; kw...)
    vyy = sinkhorn_value(cost_matrix(c, Y, Y), b, b; kw...)
    return vxy - 0.5 * (vxx + vyy)
end

"""
    energy_distance(X, Y) -> Float64

Székely-Rizzo energy distance

```math
E = 2\\,\\mathbb{E}\\|X - Y\\| - \\mathbb{E}\\|X - X'\\| - \\mathbb{E}\\|Y - Y'\\|
```

estimated from the columns of `X` and `Y`. Zero if and only if the two
distributions coincide, and much cheaper than [`sinkhorn_divergence`](@ref)
since it needs no iteration.
"""
function energy_distance(X::AbstractMatrix, Y::AbstractMatrix)
    dxy = sqrt.(cost_matrix(SqEuclideanCost(), X, Y))
    dxx = sqrt.(cost_matrix(SqEuclideanCost(), X, X))
    dyy = sqrt.(cost_matrix(SqEuclideanCost(), Y, Y))
    return Float64(2 * mean(dxy) - mean(dxx) - mean(dyy))
end

"""
    mmd(X, Y; sigma=nothing) -> Float64

Unbiased estimate of the squared maximum mean discrepancy with a Gaussian
kernel `k(x, y) = exp(-||x - y||^2 / (2 sigma^2))`.

`sigma = nothing` uses the median heuristic: the median pairwise distance of the
pooled sample. The diagonal is excluded from the `XX` and `YY` terms, which is
what makes the estimator unbiased (and lets it be slightly negative).
"""
function mmd(X::AbstractMatrix, Y::AbstractMatrix; sigma = nothing)
    n, m = size(X, 2), size(Y, 2)
    (n > 1 && m > 1) || throw(ArgumentError("mmd needs at least two samples per input"))
    Dxx = cost_matrix(SqEuclideanCost(), X, X)
    Dyy = cost_matrix(SqEuclideanCost(), Y, Y)
    Dxy = cost_matrix(SqEuclideanCost(), X, Y)
    s = sigma === nothing ? sqrt(median(vcat(vec(Dxx), vec(Dyy), vec(Dxy)))) : Float32(sigma)
    s = max(Float32(s), 1f-8)
    kern(D) = exp.(.-D ./ (2f0 * s^2))
    Kxx, Kyy, Kxy = kern(Dxx), kern(Dyy), kern(Dxy)
    sxx = (sum(Kxx) - sum(diag(Kxx))) / (n * (n - 1))
    syy = (sum(Kyy) - sum(diag(Kyy))) / (m * (m - 1))
    return Float64(sxx + syy - 2 * mean(Kxy))
end

"""
    moment_error(X, Y) -> (mean=..., cov=...)

Maximum absolute difference between the empirical means and covariances of two
sample matrices. A fast first check that a pushforward landed on the target.
"""
function moment_error(X::AbstractMatrix, Y::AbstractMatrix)
    mx = vec(mean(X; dims = 2))
    my = vec(mean(Y; dims = 2))
    cx = cov(Float64.(X'); dims = 1)
    cy = cov(Float64.(Y'); dims = 1)
    return (mean = maximum(abs.(mx .- my)), cov = maximum(abs.(cx .- cy)))
end

"""
    transport_error(T, Tref) -> Float64

Root-mean-square deviation between a learned map's output and a reference map's
output on the same inputs, `sqrt(mean(sum((T - Tref).^2, dims=1)))`.

# Example
```julia
ref = gaussian_ot(m0, S0, m1, S1)
X = sample_mu(2000)
transport_error(monge_map(res, X), ref.map(X))
```
"""
function transport_error(T::AbstractMatrix, Tref::AbstractMatrix)
    size(T) == size(Tref) || throw(DimensionMismatch(
        "sizes $(size(T)) and $(size(Tref)) do not match"))
    return Float64(sqrt(mean(sum(abs2, _f32(T) .- _f32(Tref); dims = 1))))
end
