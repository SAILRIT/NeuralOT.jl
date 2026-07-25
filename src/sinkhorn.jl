# Discrete entropic optimal transport, in the log domain.
#
# Cuturi (2013), "Sinkhorn Distances"; Feydy et al. (2019), "Interpolating
# between Optimal Transport and MMD using Sinkhorn Divergences".
#
# Conventions. For weights a, b and cost C the regularised problem is
#
#   OT_eps(a, b) = min_{P in U(a,b)} <P, C> + eps * KL(P | a b^T)
#
# with dual potentials f, g satisfying
#
#   P_ij = exp((f_i + g_j - C_ij) / eps) a_i b_j,
#   OT_eps(a, b) = <f, a> + <g, b>.
#
# The *value* `<f,a> + <g,b>` (not the raw transport cost `<P,C>`) is what makes
# the debiased Sinkhorn divergence positive-definite and exactly zero between a
# measure and itself, which is why `sinkhorn_divergence` is built from it.

"""
    sinkhorn_potentials(C, a=nothing, b=nothing; epsilon=0.1, n_iter=2_000, tol=1e-9)

Log-domain Sinkhorn iterations on the cost matrix `C` (`n x m`) with marginals
`a` (length `n`) and `b` (length `m`), both defaulting to uniform.

Returns a named tuple `(f, g, iterations, error, converged)` where `f`, `g` are
the dual potentials on the same scale as `C`.

The log-domain form is stable for small `epsilon`, where the plain
matrix-scaling form underflows. Iteration stops once the largest change in either potential falls below `tol`;
`sol.converged` reports whether that happened.

Well-separated measures at small `epsilon` need many iterations - two Gaussians
three standard deviations apart need roughly 1900 iterations at
`epsilon = 0.1` to reach `tol = 1e-9`. Check `sol.converged` rather than
assuming the default budget was enough.

# Example
```julia
C = cost_matrix(SqEuclideanCost(), X, Y)
sol = sinkhorn_potentials(C; epsilon = 0.05)
sol.converged, sol.iterations
```
"""
function sinkhorn_potentials(C::AbstractMatrix, a = nothing, b = nothing;
                             epsilon::Real = 0.1, n_iter::Int = 2_000,
                             tol::Real = 1e-9)
    epsilon > 0 || throw(ArgumentError("epsilon must be positive, got $epsilon"))
    n_iter >= 1 || throw(ArgumentError("n_iter must be at least 1"))
    Cf = _f32(C)
    n, m = size(Cf)
    av = a === nothing ? fill(1f0 / n, n) : _f32(collect(a))
    bv = b === nothing ? fill(1f0 / m, m) : _f32(collect(b))
    length(av) == n || throw(DimensionMismatch("a has length $(length(av)), expected $n"))
    length(bv) == m || throw(DimensionMismatch("b has length $(length(bv)), expected $m"))
    all(>=(0), av) && all(>=(0), bv) ||
        throw(ArgumentError("marginal weights must be non-negative"))

    eps32 = Float32(epsilon)
    loga = log.(av)
    logb = log.(bv)
    f = zeros(Float32, n)
    g = zeros(Float32, m)
    err = Inf32
    used = n_iter
    converged = false

    for it in 1:n_iter
        # f_i = -eps * logsumexp_j( (g_j - C_ij)/eps + log b_j )
        M = (g' .- Cf) ./ eps32 .+ logb'
        fnew = -eps32 .* vec(_logsumexp(M; dims = 2))
        M2 = (fnew .- Cf) ./ eps32 .+ loga
        gnew = -eps32 .* vec(_logsumexp(M2; dims = 1))
        err = max(maximum(abs.(fnew .- f)), maximum(abs.(gnew .- g)))
        f, g = fnew, gnew
        if err < Float32(tol)
            used = it
            converged = true
            break
        end
    end
    return (f = f, g = g, iterations = used, error = Float64(err), converged = converged)
end

"""
    sinkhorn_plan(C, a=nothing, b=nothing; epsilon=0.1, kwargs...)

The entropic transport plan `P` (`n x m`), whose row and column sums reproduce
`a` and `b` to the accuracy of the Sinkhorn iteration.
"""
function sinkhorn_plan(C::AbstractMatrix, a = nothing, b = nothing;
                       epsilon::Real = 0.1, kwargs...)
    Cf = _f32(C)
    n, m = size(Cf)
    av = a === nothing ? fill(1f0 / n, n) : _f32(collect(a))
    bv = b === nothing ? fill(1f0 / m, m) : _f32(collect(b))
    sol = sinkhorn_potentials(Cf, av, bv; epsilon = epsilon, kwargs...)
    eps32 = Float32(epsilon)
    return exp.((sol.f .+ sol.g' .- Cf) ./ eps32) .* av .* bv'
end

"""
    sinkhorn_value(C, a=nothing, b=nothing; epsilon=0.1, kwargs...)

The regularised optimal transport value `<f, a> + <g, b>`, equal to
`<P, C> + epsilon * KL(P | a b')`. This is the quantity that behaves like a
proper divergence; use [`sinkhorn_cost`](@ref) if you want the raw transport
cost instead.
"""
function sinkhorn_value(C::AbstractMatrix, a = nothing, b = nothing;
                        epsilon::Real = 0.1, kwargs...)
    Cf = _f32(C)
    n, m = size(Cf)
    av = a === nothing ? fill(1f0 / n, n) : _f32(collect(a))
    bv = b === nothing ? fill(1f0 / m, m) : _f32(collect(b))
    sol = sinkhorn_potentials(Cf, av, bv; epsilon = epsilon, kwargs...)
    return Float64(dot(sol.f, av) + dot(sol.g, bv))
end

"""
    sinkhorn_cost(C, a=nothing, b=nothing; epsilon=0.1, kwargs...)

The raw transport cost `<P, C>` of the entropic plan. Converges to the exact
optimal transport cost as `epsilon -> 0`, but is not itself a divergence: it
does not vanish between a measure and itself.
"""
function sinkhorn_cost(C::AbstractMatrix, a = nothing, b = nothing;
                       epsilon::Real = 0.1, kwargs...)
    Cf = _f32(C)
    P = sinkhorn_plan(Cf, a, b; epsilon = epsilon, kwargs...)
    return Float64(sum(P .* Cf))
end
