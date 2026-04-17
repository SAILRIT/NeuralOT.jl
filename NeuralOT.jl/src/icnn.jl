# Input Convex Neural Networks (ICNNs)
#
# Amos, Xu, Kolter (2017) "Input Convex Neural Networks"
# Makkuva et al. (2020) "Optimal Transport Mapping via Input Convex Neural Networks"

"""
    ICNN(dim, widths; activation=softplus)

Input Convex Neural Network. The output is convex in the input `x` provided
that all `W_z` weights are non-negative and the activation is convex and
non-decreasing (softplus is the standard choice).

# Arguments
- `dim::Int`: input dimensionality
- `widths::Vector{Int}`: hidden layer widths; the last element is the output
  layer width (typically 1 for a scalar potential)
- `activation`: convex, non-decreasing activation (default `softplus`)
"""
struct ICNN{A}
    Wx::Vector{Matrix{Float32}}
    Wz::Vector{Matrix{Float32}}
    b::Vector{Vector{Float32}}
    activation::A
end

# Flux ≥ 0.14 functor macro. `activation` is excluded from training by the
# explicit `trainable` method below (Flux sees all fields by default, but
# `trainable` narrows the update set to numeric arrays only).
Flux.@functor ICNN
Flux.trainable(m::ICNN) = (Wx = m.Wx, Wz = m.Wz, b = m.b)

function ICNN(dim::Int, widths::Vector{Int}; activation=softplus,
              init=Flux.glorot_uniform)
    @assert !isempty(widths) "widths must be non-empty"
    L = length(widths)
    Wx = Vector{Matrix{Float32}}(undef, L)
    Wz = Vector{Matrix{Float32}}(undef, L)
    b  = Vector{Vector{Float32}}(undef, L)
    prev = dim
    for l in 1:L
        Wx[l] = init(widths[l], dim)
        Wz[l] = l == 1 ? zeros(Float32, widths[l], 0) : init(widths[l], prev)
        b[l]  = zeros(Float32, widths[l])
        prev  = widths[l]
    end
    ICNN(Wx, Wz, b, activation)
end

# Forward pass. `x` is deliberately un-typed beyond AbstractMatrix so that
# ForwardDiff.Dual matrices also work (used by `grad_x` below).
function (f::ICNN)(x::AbstractMatrix)
    z = f.activation.(f.Wx[1] * x .+ f.b[1])
    for l in 2:length(f.Wx)
        Wz_pos = softplus.(f.Wz[l])
        z = f.activation.(Wz_pos * z .+ f.Wx[l] * x .+ f.b[l])
    end
    return z
end

(f::ICNN)(x::AbstractVector) = vec(f(reshape(x, :, 1)))

"""
    grad_x(f, x::AbstractMatrix)

Per-column gradient of a scalar-output network with respect to its input.
Returns a matrix of the same shape as `x`.

For `solve_w2`, callers should differentiate through `grad_x` using the
ICNN-specific convex-conjugate trick rather than nested AD — see
`solve_w2` for how this is handled.
"""
function grad_x(f, x::AbstractMatrix)
    return Zygote.gradient(xin -> sum(f(xin)), x)[1]
end
