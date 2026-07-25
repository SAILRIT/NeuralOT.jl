# Toy distributions. Every constructor returns a *sampler*: a function
# `n -> Matrix{Float32}` of size `dim x n`, which is the interface all solvers
# expect.

"""
    gaussian_sampler(dim; mean=zeros(dim), cov=I, sigma=nothing, rng=nothing)

Sampler for a multivariate normal. Pass either a full covariance `cov` or an
isotropic standard deviation `sigma`.

# Example
```julia
mu = gaussian_sampler(2; sigma = 0.5)
nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)
mu(128)     # 2 x 128
```
"""
function gaussian_sampler(dim::Int; mean = zeros(dim), cov = nothing,
                          sigma = nothing, rng = nothing)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    m = Float32.(collect(mean))
    length(m) == dim || throw(DimensionMismatch("mean has length $(length(m)), expected $dim"))
    L = if cov === nothing
        s = sigma === nothing ? 1.0 : sigma
        Float32(s) .* Matrix{Float32}(I, dim, dim)
    else
        C = Matrix{Float64}(cov)
        size(C) == (dim, dim) || throw(DimensionMismatch("cov must be $dim x $dim"))
        Float32.(_sqrtm_psd(C))
    end
    r = _rng(rng)
    return n -> L * randn(r, Float32, dim, n) .+ m
end

"""
    two_moons(; noise=0.05, rng=nothing)

The classic two interleaving half-circles, as a single 2-D distribution
(both moons pooled). `noise` is the standard deviation of additive Gaussian
noise.
"""
function two_moons(; noise::Real = 0.05, rng = nothing)
    r = _rng(rng)
    return function (n::Int)
        out = Matrix{Float32}(undef, 2, n)
        @inbounds for i in 1:n
            t = Float32(pi) * rand(r, Float32)
            if rand(r) < 0.5
                out[1, i] = cos(t)
                out[2, i] = sin(t)
            else
                out[1, i] = 1f0 - cos(t)
                out[2, i] = 0.5f0 - sin(t)
            end
        end
        return out .+ Float32(noise) .* randn(r, Float32, 2, n)
    end
end

"""
    eight_gaussians(; radius=4.0, sigma=0.35, n_modes=8, rng=nothing)

Mixture of `n_modes` isotropic Gaussians spaced evenly on a circle - the
standard multimodal benchmark for transport-based generative models.
"""
function eight_gaussians(; radius::Real = 4.0, sigma::Real = 0.35,
                         n_modes::Int = 8, rng = nothing)
    n_modes > 0 || throw(ArgumentError("n_modes must be positive"))
    r = _rng(rng)
    angles = Float32.(range(0, 2 * pi; length = n_modes + 1)[1:n_modes])
    centres = vcat((Float32(radius) .* cos.(angles))', (Float32(radius) .* sin.(angles))')
    return function (n::Int)
        idx = rand(r, 1:n_modes, n)
        return centres[:, idx] .+ Float32(sigma) .* randn(r, Float32, 2, n)
    end
end

"""
    checkerboard(; scale=4.0, rng=nothing)

Uniform density on the black squares of a 4x4 checkerboard scaled to
`[-scale, scale]^2`. Multimodal with sharp edges, so it is a hard target for
smooth vector fields.
"""
function checkerboard(; scale::Real = 4.0, rng = nothing)
    r = _rng(rng)
    return function (n::Int)
        out = Matrix{Float32}(undef, 2, n)
        filled = 0
        while filled < n
            x = 2f0 * rand(r, Float32) - 1f0
            y = 2f0 * rand(r, Float32) - 1f0
            # keep the point if its 4x4 cell is "black"
            ix = floor(Int, (x + 1f0) * 2f0)
            iy = floor(Int, (y + 1f0) * 2f0)
            if iseven(ix + iy)
                filled += 1
                out[1, filled] = Float32(scale) * x
                out[2, filled] = Float32(scale) * y
            end
        end
        return out
    end
end

"""
    swiss_roll(; noise=0.1, scale=0.25, rng=nothing)

Two-dimensional Swiss roll (a spiral), a curved manifold that transport maps
must bend rather than merely translate.
"""
function swiss_roll(; noise::Real = 0.1, scale::Real = 0.25, rng = nothing)
    r = _rng(rng)
    return function (n::Int)
        t = 1.5f0 * Float32(pi) .* (1f0 .+ 2f0 .* rand(r, Float32, n))
        x = t .* cos.(t)
        y = t .* sin.(t)
        return Float32(scale) .* vcat(x', y') .+ Float32(noise) .* randn(r, Float32, 2, n)
    end
end

"""
    circles(; radius=1.0, noise=0.05, rng=nothing)

Points on a circle of the given radius with Gaussian jitter.
"""
function circles(; radius::Real = 1.0, noise::Real = 0.05, rng = nothing)
    r = _rng(rng)
    return function (n::Int)
        θ = 2f0 * Float32(pi) .* rand(r, Float32, n)
        return Float32(radius) .* vcat(cos.(θ)', sin.(θ)') .+
               Float32(noise) .* randn(r, Float32, 2, n)
    end
end

"""
    uniform_box(dim; lo=-1.0, hi=1.0, rng=nothing)

Uniform distribution on the box `[lo, hi]^dim`.
"""
function uniform_box(dim::Int; lo::Real = -1.0, hi::Real = 1.0, rng = nothing)
    hi > lo || throw(ArgumentError("hi must exceed lo"))
    r = _rng(rng)
    span = Float32(hi - lo)
    return n -> Float32(lo) .+ span .* rand(r, Float32, dim, n)
end
