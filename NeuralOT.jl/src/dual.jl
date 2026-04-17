# Entropy/L2-regularised dual OT à la Seguy et al. (2018),
# "Large-Scale Optimal Transport and Mapping Estimation", ICLR.
#
# The entropic dual is
#   max_{u,v} E_μ[u(X)] + E_ν[v(Y)] - ε E_{μ⊗ν}[exp((u(X)+v(Y)-c(X,Y))/ε)]
# which can be maximised stochastically from samples of μ and ν — no cost
# matrix required, and it scales to arbitrary sample sizes and dimensions.

"""
    NeuralOTResult

Container for trained neural OT models and metadata.

# Fields
- `models`: NamedTuple of trained networks (e.g. `(u=..., v=...)` or `(f=..., g=...)`)
- `losses::Vector{Float64}`: training loss per logged step
- `method::Symbol`: `:dual`, `:w2_icnn`, or `:flow`
- `config::NamedTuple`: hyperparameters used
"""
struct NeuralOTResult
    models::NamedTuple
    losses::Vector{Float64}
    method::Symbol
    config::NamedTuple
end

"""
    solve_dual(sample_μ, sample_ν; dim, cost=sqeuclidean, ε=0.1, ...)

Solve the entropy-regularised OT dual problem with neural potentials.

Given samplers `sample_μ(n)` and `sample_ν(n)` returning `dim × n` matrices,
trains two potential networks `u, v` maximising

    E[u(X)] + E[v(Y)] - ε E[exp((u(X) + v(Y) - c(X,Y)) / ε)]

which recovers the regularised OT dual in the limit of infinite capacity.

# Keyword arguments
- `dim::Int` (required): ambient dimension
- `cost`: function `(x, y) -> scalar` applied column-wise (default sqeuclidean)
- `ε::Real = 0.1`: entropic regularisation strength
- `hidden = [128, 128]`: hidden widths for the two MLPs
- `batch::Int = 256`: batch size per step
- `steps::Int = 5_000`: number of optimisation steps
- `lr::Real = 1e-4`: Adam learning rate
- `log_every::Int = 100`: record loss every N steps
- `seed = nothing`: RNG seed for reproducibility

# Returns
A [`NeuralOTResult`](@ref) with `models = (u=..., v=...)`.
"""
function solve_dual(sample_μ, sample_ν;
                    dim::Int,
                    cost = sqeuclidean,
                    ε::Real = 0.1,
                    hidden::Vector{Int} = [128, 128],
                    batch::Int = 256,
                    steps::Int = 5_000,
                    lr::Real = 1e-4,
                    log_every::Int = 100,
                    seed = nothing)
    seed === nothing || Random.seed!(seed)

    u = DualPotentialNet(dim; hidden=hidden)
    v = DualPotentialNet(dim; hidden=hidden)

    opt_u = Flux.setup(Adam(lr), u)
    opt_v = Flux.setup(Adam(lr), v)

    losses = Float64[]
    εf = Float32(ε)

    # Vectorised pairwise cost between two batches (columns are samples).
    function pairwise_cost(X, Y)
        # For sqeuclidean: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x'y
        # Stay generic by falling back to explicit loop for arbitrary cost.
        if cost === sqeuclidean
            sx = sum(abs2, X; dims=1)            # (1, B)
            sy = sum(abs2, Y; dims=1)            # (1, B)
            return sx' .+ sy .- 2f0 .* (X' * Y)  # (B, B)
        else
            B1 = size(X, 2); B2 = size(Y, 2)
            C = Matrix{Float32}(undef, B1, B2)
            for j in 1:B2, i in 1:B1
                C[i, j] = Float32(cost(view(X, :, i), view(Y, :, j)))
            end
            return C
        end
    end

    function loss_fn(u, v, X, Y)
        ux = vec(u(X))                     # (B,)
        vy = vec(v(Y))                     # (B,)
        C  = pairwise_cost(X, Y)           # (B, B)
        # Broadcast: U[i,j] = ux[i], V[i,j] = vy[j]
        M  = (ux .+ vy' .- C) ./ εf
        # Stable exponential via log-sum-exp-style shift
        mmax = maximum(M)
        pen  = εf * (mmax + log(mean(exp.(M .- mmax))))
        return -(mean(ux) + mean(vy) - pen) # minimise negative dual
    end

    for step in 1:steps
        X = Float32.(sample_μ(batch))
        Y = Float32.(sample_ν(batch))

        gs_u = Flux.gradient(m -> loss_fn(m, v, X, Y), u)[1]
        Flux.update!(opt_u, u, gs_u)

        gs_v = Flux.gradient(m -> loss_fn(u, m, X, Y), v)[1]
        Flux.update!(opt_v, v, gs_v)

        if step % log_every == 0 || step == 1
            push!(losses, Float64(loss_fn(u, v, X, Y)))
        end
    end

    NeuralOTResult(
        (u=u, v=v),
        losses,
        :dual,
        (ε=ε, hidden=hidden, batch=batch, steps=steps, lr=lr),
    )
end
