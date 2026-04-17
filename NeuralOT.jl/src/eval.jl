# Post-training utilities: extract the transport map and evaluate quality
# against an entropic reference (Sinkhorn divergence).

"""
    monge_map(result::NeuralOTResult, x; n_flow_steps=100)

Apply the learned transport map to points `x` (`dim × B`).

Dispatches on `result.method`:
- `:w2_icnn` — returns `∇g(x)`, the ICNN Brenier map
- `:flow`    — integrates the learned ODE from `t=0` to `t=1`
- `:dual`    — returns `x - 0.5 ∇u(x) / ε` (entropic map heuristic; valid only
               for squared cost)
"""
function monge_map(result::NeuralOTResult, x::AbstractMatrix; n_flow_steps::Int = 100)
    X = Float32.(x)
    if result.method === :w2_icnn
        return grad_x(result.models.g, X)
    elseif result.method === :flow
        return integrate_flow(result.models.vfield, X; n_steps=n_flow_steps)
    elseif result.method === :dual
        ε = Float32(result.config.ε)
        gu = grad_x(result.models.u, X)
        # Seguy et al. barycentric map approximation for squared cost.
        return X .- gu ./ (2f0 * ε)
    else
        error("Unknown method $(result.method)")
    end
end

monge_map(result::NeuralOTResult, x::AbstractVector; kwargs...) =
    vec(monge_map(result, reshape(x, :, 1); kwargs...))

"""
    sinkhorn_divergence(X, Y; ε=0.1, n_iter=200, cost=sqeuclidean)

Entropic Sinkhorn divergence `S_ε(X, Y) = OT_ε(X, Y) - ½(OT_ε(X,X) + OT_ε(Y,Y))`
between two empirical measures given as `dim × N` and `dim × M` matrices.

Provided as a lightweight evaluation metric so users can compare the output
of `monge_map(result, X)` to the target `Y`. Not intended as a replacement
for OptimalTransport.jl.
"""
function sinkhorn_divergence(X::AbstractMatrix, Y::AbstractMatrix;
                             ε::Real = 0.1, n_iter::Int = 200,
                             cost = sqeuclidean)
    εf = Float32(ε)
    otxy = _sinkhorn_cost(X, Y, εf, n_iter, cost)
    otxx = _sinkhorn_cost(X, X, εf, n_iter, cost)
    otyy = _sinkhorn_cost(Y, Y, εf, n_iter, cost)
    return otxy - 0.5f0 * (otxx + otyy)
end

function _pairwise_cost(X, Y, cost)
    if cost === sqeuclidean
        sx = sum(abs2, X; dims=1)
        sy = sum(abs2, Y; dims=1)
        return Float32.(sx' .+ sy .- 2f0 .* (X' * Y))
    else
        B1 = size(X, 2); B2 = size(Y, 2)
        C = Matrix{Float32}(undef, B1, B2)
        for j in 1:B2, i in 1:B1
            C[i, j] = Float32(cost(view(X, :, i), view(Y, :, j)))
        end
        return C
    end
end

# Log-domain Sinkhorn for numerical stability.
function _sinkhorn_cost(X, Y, ε::Float32, n_iter::Int, cost)
    C = _pairwise_cost(X, Y, cost)
    n, m = size(C)
    loga = fill(-log(Float32(n)), n)
    logb = fill(-log(Float32(m)), m)
    logK = -C ./ ε                          # (n, m)
    logu = zeros(Float32, n)
    logv = zeros(Float32, m)

    for _ in 1:n_iter
        # log u = log a - logsumexp(logK + log v, dims=2)
        M = logK .+ logv'
        logu = loga .- _logsumexp(M; dims=2)
        M2 = logK .+ logu
        logv = logb .- _logsumexp(M2; dims=1)[:]
    end
    # Transport plan π = exp(logu + logK + logv); cost = Σ π .* C
    logπ = logu .+ logK .+ logv'
    π = exp.(logπ)
    return sum(π .* C)
end

function _logsumexp(A; dims)
    mx = maximum(A; dims=dims)
    return mx .+ log.(sum(exp.(A .- mx); dims=dims))
end
