# W2 Monge map estimation via Input Convex Neural Networks.
#
# Makkuva, Taghvaei, Oh, Lee (2020)
# "Optimal Transport Mapping via Input Convex Neural Networks", ICML.
#
# Formulation: parameterise two convex potentials f, g with ICNNs. The
# optimal W2 map from μ to ν is T(x) = ∇f*(x), where f* is the convex
# conjugate. The saddle-point objective is
#
#   inf_f sup_g  E_μ[<X, ∇g(X)> - f(∇g(X))] + E_ν[f(Y)]  + const
#
# which is solved by alternating maximisation over g (inner) and minimisation
# over f (outer). At optimum, ∇g approximately inverts ∇f and is itself a
# valid transport map.

"""
    solve_w2(sample_μ, sample_ν; dim, widths=[64,64,1], ...)

Estimate the squared-Euclidean (W2) Monge map between μ and ν using the
ICNN saddle-point formulation of Makkuva et al. (2020).

Both `sample_μ(n)` and `sample_ν(n)` should return `dim × n` matrices.

# Keyword arguments
- `dim::Int` (required)
- `widths::Vector{Int} = [64, 64, 1]`: ICNN hidden/output widths (last must be 1)
- `batch::Int = 256`
- `steps::Int = 5_000`: outer steps
- `inner_steps::Int = 10`: inner maximisation steps per outer step
- `lr::Real = 1e-4`
- `log_every::Int = 100`
- `seed = nothing`

# Returns
[`NeuralOTResult`](@ref) with `models = (f=..., g=...)`. The forward map μ→ν
is recovered via `monge_map(result, x)` (uses `∇g`).

!!! warning "Experimental"
    The inner loop requires second-order differentiation (gradient through
    `∇g`). This is supported by Zygote in principle but can be fragile for
    certain ICNN configurations. If you see `ERROR: Mutating arrays is not
    supported` or similar, it's the nested AD path — please open an issue
    with a minimal reproducer. We plan to add an `Enzyme.jl` backend.

!!! note
    The inner loop approximates the supremum over `g`; stability is sensitive
    to `inner_steps` and `lr`. Start with small `widths` and increase.
"""
function solve_w2(sample_μ, sample_ν;
                  dim::Int,
                  widths::Vector{Int} = [64, 64, 1],
                  batch::Int = 256,
                  steps::Int = 5_000,
                  inner_steps::Int = 10,
                  lr::Real = 1e-4,
                  log_every::Int = 100,
                  seed = nothing)
    @assert last(widths) == 1 "ICNN potential must have scalar output (widths[end] == 1)"
    seed === nothing || Random.seed!(seed)

    f = ICNN(dim, widths)
    g = ICNN(dim, widths)

    opt_f = Flux.setup(Adam(lr), f)
    opt_g = Flux.setup(Adam(lr), g)

    losses = Float64[]

    # Inner objective (maximised over g):
    #   J(g; f) = E_μ[<X, ∇g(X)> - f(∇g(X))]
    # Outer objective (minimised over f):
    #   L(f) = J(g*(f); f) + E_ν[f(Y)]
    function inner_loss(g, f, X)
        Tg = grad_x(g, X)                      # candidate map ∇g(X)
        xg = sum(X .* Tg; dims=1)              # <X, ∇g(X)> per column
        fT = f(Tg)                             # f(∇g(X))
        return -mean(xg .- fT)                 # minimise negative -> maximise
    end

    function outer_loss(f, g, X, Y)
        Tg = grad_x(g, X)
        xg = sum(X .* Tg; dims=1)
        fT = f(Tg)
        fY = f(Y)
        return mean(xg .- fT) + mean(fY)
    end

    for step in 1:steps
        X = Float32.(sample_μ(batch))
        Y = Float32.(sample_ν(batch))

        # Inner: maximise over g
        for _ in 1:inner_steps
            gs_g = Flux.gradient(m -> inner_loss(m, f, X), g)[1]
            Flux.update!(opt_g, g, gs_g)
        end

        # Outer: minimise over f
        gs_f = Flux.gradient(m -> outer_loss(m, g, X, Y), f)[1]
        Flux.update!(opt_f, f, gs_f)

        if step % log_every == 0 || step == 1
            push!(losses, Float64(outer_loss(f, g, X, Y)))
        end
    end

    NeuralOTResult(
        (f=f, g=g),
        losses,
        :w2_icnn,
        (widths=widths, batch=batch, steps=steps,
         inner_steps=inner_steps, lr=lr),
    )
end
