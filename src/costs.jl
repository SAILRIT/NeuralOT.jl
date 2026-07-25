# Ground costs.
#
# A cost is anything that can produce a pairwise cost matrix `C[i, j] =
# c(X[:, i], Y[:, j])`. Two fast paths are provided; anything else is wrapped in
# `GenericCost` and evaluated column-by-column.
#
# NOTE ON CONVENTIONS: `SqEuclideanCost` is ||x - y||^2, *not* half of it. The
# entropic barycentric map for this cost is `x - grad_u(x) / 2`.

"""
    OTCost

Abstract supertype of ground costs understood by the solvers.
"""
abstract type OTCost end

"""
    SqEuclideanCost()

Squared Euclidean cost `c(x, y) = ||x - y||^2`. This is the cost for which the
2-Wasserstein theory (Brenier's theorem, [`solve_w2`](@ref)) applies.
"""
struct SqEuclideanCost <: OTCost end

"""
    EuclideanCost()

Euclidean cost `c(x, y) = ||x - y||`.
"""
struct EuclideanCost <: OTCost end

"""
    GenericCost(f)

Wrap an arbitrary two-argument function `f(x, y) -> Real` acting on columns.
Slower than the built-in costs because the cost matrix is filled entrywise.

# Example
```julia
c = GenericCost((x, y) -> sum(abs, x .- y))   # L1 cost
```
"""
struct GenericCost{F} <: OTCost
    f::F
end

"""
    cost_matrix(cost, X, Y)

Pairwise cost matrix between the columns of `X` (`d x n`) and `Y` (`d x m`),
returned as an `n x m` `Matrix{Float32}`.

The `SqEuclideanCost` path uses the expansion
`||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>` and is clamped at zero to remove
the small negative values that expansion can produce for near-identical points.

!!! note
    Cost matrices do not depend on any network parameter, so they are computed
    *outside* the differentiated loss functions. That keeps mutation-based
    implementations (including `Distances.pairwise`) out of Zygote's path.
"""
function cost_matrix(::SqEuclideanCost, X::AbstractMatrix, Y::AbstractMatrix)
    size(X, 1) == size(Y, 1) || throw(DimensionMismatch(
        "cost_matrix: X has $(size(X, 1)) rows, Y has $(size(Y, 1))"))
    Xf, Yf = _f32(X), _f32(Y)
    sx = _colnorm2(Xf)          # 1 x n
    sy = _colnorm2(Yf)          # 1 x m
    C = (sx' .+ sy) .- 2f0 .* (Xf' * Yf)
    return max.(C, 0f0)
end

function cost_matrix(::EuclideanCost, X::AbstractMatrix, Y::AbstractMatrix)
    return sqrt.(cost_matrix(SqEuclideanCost(), X, Y))
end

function cost_matrix(c::GenericCost, X::AbstractMatrix, Y::AbstractMatrix)
    size(X, 1) == size(Y, 1) || throw(DimensionMismatch(
        "cost_matrix: X has $(size(X, 1)) rows, Y has $(size(Y, 1))"))
    n, m = size(X, 2), size(Y, 2)
    C = Matrix{Float32}(undef, n, m)
    @inbounds for j in 1:m
        yj = view(Y, :, j)
        for i in 1:n
            C[i, j] = Float32(c.f(view(X, :, i), yj))
        end
    end
    return C
end

# Bare functions are accepted for backwards compatibility with v0.1, where the
# `cost` keyword took `Distances.sqeuclidean`. Recognised functions are routed
# to their fast paths.
_as_cost(c::OTCost) = c
_as_cost(::typeof(Distances.sqeuclidean)) = SqEuclideanCost()
_as_cost(::typeof(Distances.euclidean)) = EuclideanCost()
_as_cost(::Distances.SqEuclidean) = SqEuclideanCost()
_as_cost(::Distances.Euclidean) = EuclideanCost()
_as_cost(f::Function) = GenericCost(f)

cost_matrix(c, X::AbstractMatrix, Y::AbstractMatrix) = cost_matrix(_as_cost(c), X, Y)

"""
    is_squared_euclidean(cost) -> Bool

Whether `cost` is the squared Euclidean cost. Map-recovery routines that rely on
Brenier's theorem check this and refuse to guess for other costs.
"""
is_squared_euclidean(c) = _as_cost(c) isa SqEuclideanCost
