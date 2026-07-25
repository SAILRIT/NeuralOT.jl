# Small shared helpers. Nothing here is exported.

"""
    _f32(X)

Convert to `Float32` without copying when the input is already `Float32`.
"""
_f32(X::AbstractArray{Float32}) = X
_f32(X::AbstractArray) = Float32.(X)

"""
    _rng(seed)

Return an RNG. `nothing` gives the default global RNG (and does not reseed it);
an integer gives a fresh, independent `MersenneTwister` so that results are
reproducible without disturbing global state.
"""
_rng(seed::Nothing) = Random.default_rng()
_rng(seed::Integer) = MersenneTwister(seed)
_rng(rng::AbstractRNG) = rng

"""
    _check_batch(X, dim, name)

Validate that a sampler returned a `dim x n` matrix of finite numbers.
"""
function _check_batch(X, dim::Int, name::AbstractString)
    X isa AbstractMatrix ||
        throw(ArgumentError("$name must return a matrix of size (dim, n); got $(typeof(X))"))
    size(X, 1) == dim || throw(DimensionMismatch(
        "$name returned $(size(X, 1)) rows but dim = $dim. " *
        "Samplers must return `dim x n` matrices (one sample per column)."))
    size(X, 2) > 0 || throw(ArgumentError("$name returned an empty batch"))
    return X
end

"""
    _logsumexp(A; dims)

Numerically stable `log.(sum(exp.(A); dims))`. Keeps the reduced dimensions,
matching `sum`.
"""
function _logsumexp(A::AbstractArray; dims)
    mx = maximum(A; dims = dims)
    # `mx` can contain -Inf if a whole slice is -Inf; guard so that -Inf - -Inf
    # never produces NaN.
    mxs = map(m -> isfinite(m) ? m : zero(m), mx)
    return mxs .+ log.(sum(exp.(A .- mxs); dims = dims))
end

_logsumexp(v::AbstractVector) = only(_logsumexp(reshape(v, :, 1); dims = 1))

"""
    _softplus_inv(y)

Inverse of `softplus`, used to initialise positively-constrained parameters at a
prescribed value.
"""
_softplus_inv(y::Real) = log(expm1(float(y)))

"""
    _colnorm2(X)

Squared Euclidean norm of every column, as a `1 x n` matrix.
"""
_colnorm2(X::AbstractMatrix) = sum(abs2, X; dims = 1)

"""
    _onelike(A)

An array of ones with the same size and element type as `A`, built without
mutation so that it is safe inside a Zygote-differentiated function.
"""
_onelike(A::AbstractArray) = one(eltype(A)) .+ zero(A)

"""
    _sample_categorical(rng, P)

For a row-stochastic matrix `P` (`n x m`), draw one column index per row.
Returns a `Vector{Int}` of length `n`. Used for minibatch OT couplings.
"""
function _sample_categorical(rng::AbstractRNG, P::AbstractMatrix)
    n, m = size(P)
    idx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        u = rand(rng) * sum(view(P, i, :))
        acc = zero(eltype(P))
        j = m
        for k in 1:m
            acc += P[i, k]
            if acc >= u
                j = k
                break
            end
        end
        idx[i] = j
    end
    return idx
end

"""
    _fmt(x)

Compact fixed-point formatting for progress lines.
"""
_fmt(x::Real) = @sprintf("%.5g", x)
