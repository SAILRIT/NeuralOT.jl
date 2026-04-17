# Flow matching (aka rectified flow / conditional flow matching).
#
# Lipman, Chen, Ben-Hamu, Nickel, Le (2023)
# "Flow Matching for Generative Modeling", ICLR.
# Liu, Gong, Liu (2023) "Flow Straight and Fast", ICLR.
#
# Trains a vector field v_θ(t, x) such that the ODE dx/dt = v_θ(t, x) from
# t=0 to t=1 transports μ to ν. The straightest interpolant between paired
# samples (x_0, x_1) is simply x_t = (1-t) x_0 + t x_1, giving the target
# velocity x_1 - x_0 for the regression loss
#
#   L(θ) = E_{t, x_0, x_1} || v_θ(t, (1-t)x_0 + t x_1) - (x_1 - x_0) ||^2
#
# This is closely related to neural OT: the learned flow is a
# transport-style generative model, and iterating the procedure (rectified
# flow) straightens the paths toward the true OT map.

"""
    flow_match(sample_μ, sample_ν; dim, hidden=[128,128], ...)

Train a flow-matching vector field transporting μ to ν.

The model is an MLP `v_θ(t, x): [0,1] × R^dim -> R^dim` trained by
regressing the straight-line velocity between independent samples from μ
(at `t=0`) and ν (at `t=1`).

# Keyword arguments
- `dim::Int` (required)
- `hidden::Vector{Int} = [128, 128]`
- `batch::Int = 256`
- `steps::Int = 10_000`
- `lr::Real = 1e-4`
- `log_every::Int = 100`
- `seed = nothing`

# Returns
[`NeuralOTResult`](@ref) with `models = (vfield=...,)`. To sample: integrate
`dx/dt = vfield([t; x])` from `t=0` with `x ~ μ` (see `examples/flow.jl`).
"""
function flow_match(sample_μ, sample_ν;
                    dim::Int,
                    hidden::Vector{Int} = [128, 128],
                    batch::Int = 256,
                    steps::Int = 10_000,
                    lr::Real = 1e-4,
                    log_every::Int = 100,
                    seed = nothing)
    seed === nothing || Random.seed!(seed)

    # Build MLP taking [t; x] of size dim+1 and returning a vector of size dim
    layers = Any[]
    prev = dim + 1
    for h in hidden
        push!(layers, Dense(prev, h, relu))
        prev = h
    end
    push!(layers, Dense(prev, dim))
    vfield = Chain(layers...)

    opt = Flux.setup(Adam(lr), vfield)
    losses = Float64[]

    function loss_fn(net, X0, X1, t)
        # t: (1, B) in [0,1]. Interpolate and regress straight-line velocity.
        Xt = (1f0 .- t) .* X0 .+ t .* X1
        inp = vcat(t, Xt)               # (dim+1, B)
        pred = net(inp)
        target = X1 .- X0
        return mean(abs2, pred .- target)
    end

    for step in 1:steps
        X0 = Float32.(sample_μ(batch))
        X1 = Float32.(sample_ν(batch))
        t  = rand(Float32, 1, batch)

        gs = Flux.gradient(m -> loss_fn(m, X0, X1, t), vfield)[1]
        Flux.update!(opt, vfield, gs)

        if step % log_every == 0 || step == 1
            # Evaluation loss on a fresh draw of t
            t_eval = rand(Float32, 1, batch)
            push!(losses, Float64(loss_fn(vfield, X0, X1, t_eval)))
        end
    end

    NeuralOTResult(
        (vfield=vfield,),
        losses,
        :flow,
        (hidden=hidden, batch=batch, steps=steps, lr=lr),
    )
end

"""
    integrate_flow(vfield, x0; n_steps=100)

Integrate `dx/dt = vfield([t; x])` from `t=0` to `t=1` starting at `x0`
using explicit Euler. `x0` is `dim × B`; returns `dim × B`.

A small helper for the [`flow_match`](@ref) method — users needing stiff
solvers should couple this with DifferentialEquations.jl externally.
"""
function integrate_flow(vfield, x0::AbstractMatrix; n_steps::Int = 100)
    x = Float32.(copy(x0))
    B = size(x, 2)
    dt = 1f0 / n_steps
    for k in 0:n_steps-1
        t = fill(Float32(k * dt), 1, B)
        inp = vcat(t, x)
        x = x .+ dt .* vfield(inp)
    end
    return x
end
