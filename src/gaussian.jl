# Closed-form Gaussian optimal transport.
#
# These are the ground truth the test suite and the benchmarks measure against:
# for Gaussians the Brenier map and the 2-Wasserstein distance are known
# exactly, so a neural solver can be scored rather than merely smoke-tested.

"""
    _sqrtm_psd(S)

Symmetric positive-semidefinite matrix square root via an eigendecomposition,
with negative eigenvalues (numerical noise) clamped to zero.
"""
function _sqrtm_psd(S::AbstractMatrix)
    Ssym = Symmetric((S .+ S') ./ 2)
    F = eigen(Ssym)
    vals = max.(F.values, zero(eltype(F.values)))
    return F.vectors * Diagonal(sqrt.(vals)) * F.vectors'
end

"""
    w2_gaussian(m0, S0, m1, S1) -> Float64

Squared 2-Wasserstein distance between `N(m0, S0)` and `N(m1, S1)`:

```math
W_2^2 = \\|m_0 - m_1\\|^2 + \\mathrm{tr}\\left(S_0 + S_1
        - 2 (S_0^{1/2} S_1 S_0^{1/2})^{1/2}\\right)
```

# Example
```julia
w2_gaussian(zeros(2), 0.25I(2), [3.0, 0.0], 0.25I(2))   # 9.0
```
"""
function w2_gaussian(m0::AbstractVector, S0::AbstractMatrix,
                     m1::AbstractVector, S1::AbstractMatrix)
    length(m0) == length(m1) ||
        throw(DimensionMismatch("means have lengths $(length(m0)) and $(length(m1))"))
    size(S0) == size(S1) ||
        throw(DimensionMismatch("covariances have sizes $(size(S0)) and $(size(S1))"))
    A0 = Matrix{Float64}(S0)
    A1 = Matrix{Float64}(S1)
    s0h = _sqrtm_psd(A0)
    cross = _sqrtm_psd(s0h * A1 * s0h)
    return sum(abs2, Float64.(m0) .- Float64.(m1)) + tr(A0 + A1 - 2 .* cross)
end

"""
    gaussian_brenier_map(m0, S0, m1, S1) -> (A, b)

The optimal transport map from `N(m0, S0)` to `N(m1, S1)` for the squared
Euclidean cost, as `T(x) = A x + b` with

```math
A = S_0^{-1/2} (S_0^{1/2} S_1 S_0^{1/2})^{1/2} S_0^{-1/2}.
```

`A` is symmetric positive-definite, so `T` really is the gradient of a convex
function - which is exactly what [`solve_w2`](@ref) tries to learn.
"""
function gaussian_brenier_map(m0::AbstractVector, S0::AbstractMatrix,
                              m1::AbstractVector, S1::AbstractMatrix)
    A0 = Matrix{Float64}(S0)
    A1 = Matrix{Float64}(S1)
    s0h = _sqrtm_psd(A0)
    s0hi = inv(s0h)
    A = s0hi * _sqrtm_psd(s0h * A1 * s0h) * s0hi
    A = (A .+ A') ./ 2
    b = Float64.(m1) .- A * Float64.(m0)
    return A, b
end

"""
    gaussian_ot(m0, S0, m1, S1)

Everything known in closed form about optimal transport between two Gaussians,
as a named tuple `(A, b, w2, map)` where `map(X)` applies `T(x) = A x + b` to a
`d x n` batch and `w2` is the squared 2-Wasserstein distance.

# Example
```julia
ref = gaussian_ot(zeros(2), 0.25 * I(2), [3.0, 0.0], 0.25 * I(2))
ref.w2                       # 9.0
ref.map(randn(Float32, 2, 5))
```
"""
function gaussian_ot(m0::AbstractVector, S0::AbstractMatrix,
                     m1::AbstractVector, S1::AbstractMatrix)
    A, b = gaussian_brenier_map(m0, S0, m1, S1)
    w2 = w2_gaussian(m0, S0, m1, S1)
    Af = Float32.(A)
    bf = Float32.(b)
    return (A = A, b = b, w2 = w2, map = X -> Af * _f32(X) .+ bf)
end
