# Input Convex Neural Networks (ICNNs)
#
# Amos, Xu, Kolter (2017), "Input Convex Neural Networks", ICML.
# Makkuva, Taghvaei, Oh, Lee (2020), "Optimal Transport Mapping via Input
# Convex Neural Networks", ICML.
#
# The network computes
#
#   a_1 = Wx_1 x + b_1,                          z_1 = sigma(a_1)
#   a_l = Wz_l^+ z_{l-1} + Wx_l x + b_l,         z_l = sigma(a_l),  1 < l < L
#   a_L = Wz_L^+ z_{L-1} + Wx_L x + b_L
#   f(x) = a_L + (beta / 2) ||x||^2
#
# with Wz^+ = softplus(Wz) >= 0. Each `a_l` is a non-negative combination of
# convex functions plus an affine term, so convexity in `x` is preserved layer
# by layer provided sigma is convex and non-decreasing. The head is *linear*:
# applying the activation there would force f > 0 and bound its gradient, which
# needlessly restricts the family of representable transport maps.
#
# The quadratic term makes f beta-strongly convex, so grad f is invertible and
# is well scaled at initialisation. This matters a lot for `solve_w2`.

"""
    ICNN(dim, widths; activation=softplus, quadratic=1.0, init_scale=0.1, ...)

Input convex neural network representing a scalar potential `R^dim -> R`.

The output is convex in the input for any parameter values: the `z`-weights are
passed through `softplus` to keep them non-negative, and the activation must be
convex and non-decreasing.

# Arguments
- `dim::Int`: input dimensionality.
- `widths::Vector{Int}`: layer widths. The last entry is the output width and
  must be `1` for a potential.

# Keyword arguments
- `activation = softplus`: convex, non-decreasing activation. `relu` and `elu`
  are also recognised; anything else needs `dactivation`.
- `dactivation = nothing`: derivative of `activation`. Supplied automatically
  for `softplus`, `relu`, `elu` and `identity`.
- `quadratic::Real = 1.0`: initial strength `beta` of the `beta/2 ||x||^2`
  term. Set to `0` to disable it (the parameter stays learnable but starts at
  zero); `beta` is kept non-negative by a `softplus` reparameterisation.
- `init_scale::Real = 0.1`: multiplier on the `x`-path weights at
  initialisation. Small values start the potential close to `||x||^2 / 2`,
  i.e. the transport map starts close to the identity.
- `init = Flux.glorot_uniform`: weight initialiser.
- `rng = nothing`: optional RNG for reproducible initialisation.

# Example
```julia
f = ICNN(2, [64, 64, 1])
x = randn(Float32, 2, 16)
f(x)                  # 1 x 16 potential values
input_gradient(f, x)  # 2 x 16 Brenier-style map
```

# See also
[`input_gradient`](@ref), [`solve_w2`](@ref).
"""
struct ICNN{A, D}
    Wx::Vector{Matrix{Float32}}
    Wz::Vector{Matrix{Float32}}     # length L-1; Wz[l-1] feeds layer l
    b::Vector{Vector{Float32}}
    logbeta::Vector{Float32}        # 1-element; beta = softplus(logbeta)
    activation::A
    dactivation::D
end

Flux.@layer ICNN
Flux.trainable(m::ICNN) = (; Wx = m.Wx, Wz = m.Wz, b = m.b, logbeta = m.logbeta)

# Derivatives of the supported convex, non-decreasing activations.
_dactivation(::typeof(softplus)) = sigmoid
_dactivation(::typeof(relu)) = _drelu
_dactivation(::typeof(identity)) = _done
_dactivation(::typeof(elu)) = _delu
_dactivation(f) = throw(ArgumentError(
    "no known derivative for activation $(f); pass `dactivation = <derivative>` " *
    "to the ICNN constructor. The activation must be convex and non-decreasing."))

_drelu(x) = ifelse(x > zero(x), one(x), zero(x))
_done(x) = one(x)
_delu(x) = ifelse(x > zero(x), one(x), exp(x))

function ICNN(dim::Int, widths::Vector{Int};
              activation = softplus,
              dactivation = nothing,
              quadratic::Real = 1.0,
              init_scale::Real = 0.1,
              init = Flux.glorot_uniform,
              rng = nothing)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    isempty(widths) && throw(ArgumentError("widths must be non-empty"))
    all(>(0), widths) || throw(ArgumentError("all widths must be positive, got $widths"))
    quadratic >= 0 || throw(ArgumentError("quadratic must be non-negative"))

    L = length(widths)
    _init(dims...) = rng === nothing ? init(dims...) : init(_rng(rng), dims...)

    Wx = Vector{Matrix{Float32}}(undef, L)
    Wz = Vector{Matrix{Float32}}(undef, max(L - 1, 0))
    b = Vector{Vector{Float32}}(undef, L)

    prev = dim
    for l in 1:L
        Wx[l] = Float32(init_scale) .* Float32.(_init(widths[l], dim))
        if l > 1
            Wz[l - 1] = Float32.(_init(widths[l], prev))
        end
        b[l] = zeros(Float32, widths[l])
        prev = widths[l]
    end

    # softplus(logbeta) = quadratic; softplus is ~0 for very negative inputs.
    lb = quadratic > 0 ? Float32(_softplus_inv(quadratic)) : -30f0
    return ICNN(Wx, Wz, b, Float32[lb],
                activation,
                dactivation === nothing ? _dactivation(activation) : dactivation)
end

"""
    n_layers(f::ICNN)

Number of affine layers, i.e. `length(widths)`.
"""
n_layers(f::ICNN) = length(f.Wx)

"""
    input_dim(f::ICNN)

Dimensionality of the input the network expects.
"""
input_dim(f::ICNN) = size(f.Wx[1], 2)

"""
    beta(f::ICNN)

Current strength of the strong-convexity term `beta/2 * ||x||^2`.
"""
beta(f::ICNN) = softplus(f.logbeta[1])

# Non-negative reparameterisation of the z-path weights.
_pos(W) = softplus.(W)

function (f::ICNN)(x::AbstractMatrix)
    size(x, 1) == input_dim(f) || throw(DimensionMismatch(
        "ICNN expects $(input_dim(f)) rows, got $(size(x, 1))"))
    L = n_layers(f)
    a = f.Wx[1] * x .+ f.b[1]
    for l in 2:L
        z = f.activation.(a)
        a = _pos(f.Wz[l - 1]) * z .+ f.Wx[l] * x .+ f.b[l]
    end
    return a .+ (0.5f0 .* softplus.(f.logbeta)) .* _colnorm2(x)
end

(f::ICNN)(x::AbstractVector) = vec(f(reshape(x, :, 1)))

# ---------------------------------------------------------------------------
# Analytic gradient with respect to the input.
# ---------------------------------------------------------------------------
#
# `_sweep` walks forward to the head, then accumulates the reverse pass on the
# way back out of the recursion. Everything is built from matrix products and
# broadcasts, with no array mutation, so the result is itself differentiable
# with respect to the parameters by a *single* level of reverse-mode AD. That
# is what lets `solve_w2` avoid Zygote-over-Zygote.
#
# Returns (out, dout/dx, dout/dz_{l-1}); the last entry is `nothing` at l = 1.
function _sweep(f::ICNN, x::AbstractMatrix, z, l::Int, L::Int)
    a = l == 1 ? (f.Wx[1] * x .+ f.b[1]) :
                 (_pos(f.Wz[l - 1]) * z .+ f.Wx[l] * x .+ f.b[l])
    if l == L
        e = _onelike(a)                       # d out / d a_L (linear head)
        gx = f.Wx[L]' * e
        bz = L == 1 ? nothing : _pos(f.Wz[L - 1])' * e
        return a, gx, bz
    else
        out, gx_rest, bz_next = _sweep(f, x, f.activation.(a), l + 1, L)
        e = bz_next .* f.dactivation.(a)      # chain through the activation
        gx = gx_rest .+ f.Wx[l]' * e
        bz = l == 1 ? nothing : _pos(f.Wz[l - 1])' * e
        return out, gx, bz
    end
end

"""
    input_gradient(f::ICNN, x::AbstractMatrix)

Gradient of the scalar potential with respect to its input, column by column.
Returns a matrix the same size as `x`.

For an ICNN this is computed analytically rather than by automatic
differentiation, which makes it fast *and* differentiable with respect to the
network parameters under one level of reverse-mode AD. `solve_w2` depends on
that: the naive route (`Zygote.gradient` inside a `Flux.gradient`) needs nested
AD and is both slow and fragile.

Because `f` is convex, `input_gradient(f, .)` is the gradient of a convex
function and is therefore a valid Brenier map.

# Example
```julia
f = ICNN(3, [32, 32, 1])
x = randn(Float32, 3, 8)
T = input_gradient(f, x)     # 3 x 8
```
"""
function input_gradient(f::ICNN, x::AbstractMatrix)
    X = _f32(x)
    size(X, 1) == input_dim(f) || throw(DimensionMismatch(
        "ICNN expects $(input_dim(f)) rows, got $(size(X, 1))"))
    n_layers(f) >= 1 || throw(ArgumentError("ICNN has no layers"))
    size(f.Wx[end], 1) == 1 || throw(ArgumentError(
        "input_gradient requires a scalar output (widths[end] == 1), " *
        "got width $(size(f.Wx[end], 1))"))
    _, gx, _ = _sweep(f, X, nothing, 1, n_layers(f))
    return gx .+ softplus.(f.logbeta) .* X
end

input_gradient(f::ICNN, x::AbstractVector) = vec(input_gradient(f, reshape(x, :, 1)))

"""
    input_gradient(f, x)

Fallback for callables that are not `ICNN`s: uses reverse-mode AD.
"""
function input_gradient(f, x::AbstractMatrix)
    X = _f32(x)
    g = Zygote.gradient(xin -> sum(f(xin)), X)[1]
    return g === nothing ? zero(X) : g
end

input_gradient(f, x::AbstractVector) = vec(input_gradient(f, reshape(x, :, 1)))

"""
    grad_x(f, x)

Alias for [`input_gradient`](@ref), kept for compatibility with v0.1.
"""
grad_x(f, x) = input_gradient(f, x)
