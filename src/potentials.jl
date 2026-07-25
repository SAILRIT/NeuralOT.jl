# Generic (non-convex) potential networks and time-conditioned vector fields.

"""
    _mlp(dims; activation, final_activation=identity, rng=nothing)

Build a `Chain` of `Dense` layers through `dims`, applying `activation` to every
layer except the last.
"""
function _mlp(dims::Vector{Int}; activation = relu, final_activation = identity,
              rng = nothing, init = Flux.glorot_uniform)
    length(dims) >= 2 || throw(ArgumentError("need at least an input and output width"))
    _init(a, b) = rng === nothing ? init(a, b) : init(_rng(rng), a, b)
    layers = Any[]
    for i in 1:(length(dims) - 2)
        push!(layers, Dense(_init(dims[i + 1], dims[i]), zeros(Float32, dims[i + 1]),
                            activation))
    end
    push!(layers, Dense(_init(dims[end], dims[end - 1]), zeros(Float32, dims[end]),
                        final_activation))
    return Chain(layers...)
end

"""
    DualPotentialNet(dim; hidden=[128, 128], activation=softplus, rng=nothing)

Generic MLP parameterising a scalar potential `R^dim -> R`, used by
[`solve_dual`](@ref) where the potentials need not be convex.

`softplus` is the default activation because the dual objective differentiates
the potential when recovering a map, and `relu` networks have piecewise-constant
gradients.

# Example
```julia
u = DualPotentialNet(2; hidden = [64, 64])
u(randn(Float32, 2, 10))     # 1 x 10
```
"""
struct DualPotentialNet{C}
    net::C
end

Flux.@layer DualPotentialNet

function DualPotentialNet(dim::Int; hidden::Vector{Int} = [128, 128],
                          activation = softplus, rng = nothing)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    return DualPotentialNet(_mlp(vcat(dim, hidden, 1); activation = activation, rng = rng))
end

(p::DualPotentialNet)(x::AbstractMatrix) = p.net(x)
(p::DualPotentialNet)(x::AbstractVector) = vec(p.net(reshape(x, :, 1)))

"""
    VelocityNet(dim; hidden=[128, 128], activation=swish, n_fourier=0, rng=nothing)

Time-conditioned vector field `v(t, x): [0, 1] x R^dim -> R^dim`, used by
[`flow_match`](@ref).

The time argument is concatenated to the state. With `n_fourier > 0`, `t` is
additionally expanded into `sin`/`cos` features at frequencies `1, ..., n_fourier`
(multiplied by `2 pi`), which helps when the velocity varies quickly in time.

# Example
```julia
v = VelocityNet(2; hidden = [64, 64])
t = rand(Float32, 1, 16)
x = randn(Float32, 2, 16)
v(t, x)     # 2 x 16
```
"""
struct VelocityNet{C}
    net::C
    dim::Int
    n_fourier::Int
end

Flux.@layer VelocityNet
Flux.trainable(m::VelocityNet) = (; net = m.net)

function VelocityNet(dim::Int; hidden::Vector{Int} = [128, 128], activation = swish,
                     n_fourier::Int = 0, rng = nothing)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    n_fourier >= 0 || throw(ArgumentError("n_fourier must be non-negative"))
    in_dim = dim + 1 + 2 * n_fourier
    return VelocityNet(_mlp(vcat(in_dim, hidden, dim); activation = activation, rng = rng),
                       dim, n_fourier)
end

"""
    _time_features(t, n_fourier)

Expand a `1 x B` row of times into `(1 + 2 n_fourier) x B` features.
"""
function _time_features(t::AbstractMatrix, n_fourier::Int)
    n_fourier == 0 && return t
    freqs = Float32.(reshape(1:n_fourier, :, 1))
    ang = (2f0 * Float32(pi)) .* (freqs .* t)
    return vcat(t, sin.(ang), cos.(ang))
end

function (m::VelocityNet)(t::AbstractMatrix, x::AbstractMatrix)
    size(t, 1) == 1 || throw(DimensionMismatch("t must be a 1 x B matrix"))
    size(t, 2) == size(x, 2) || throw(DimensionMismatch(
        "t has $(size(t, 2)) columns but x has $(size(x, 2))"))
    return m.net(vcat(_time_features(t, m.n_fourier), x))
end

# Scalar time, broadcast across the batch.
function (m::VelocityNet)(t::Real, x::AbstractMatrix)
    tt = fill(Float32(t), 1, size(x, 2))
    return m(tt, x)
end

# Accept a stacked [t; x] matrix, as v0.1's raw `Chain` vector field did.
function (m::VelocityNet)(tx::AbstractMatrix)
    size(tx, 1) == m.dim + 1 || throw(DimensionMismatch(
        "expected $(m.dim + 1) rows of [t; x], got $(size(tx, 1))"))
    return m(tx[1:1, :], tx[2:end, :])
end
