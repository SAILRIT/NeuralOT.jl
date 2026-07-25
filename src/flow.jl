# Flow matching / rectified flow.
#
# Lipman, Chen, Ben-Hamu, Nickel, Le (2023), "Flow Matching for Generative
# Modeling", ICLR. Liu, Gong, Liu (2023), "Flow Straight and Fast", ICLR.
# Tong et al. (2023), "Improving and Generalizing Flow-Based Generative Models
# with Minibatch Optimal Transport".
#
# Train v(t, x) so that the ODE dx/dt = v(t, x) carries mu (t = 0) to nu
# (t = 1). Along the straight interpolant x_t = (1-t) x_0 + t x_1 the target
# velocity is simply x_1 - x_0, giving the regression loss
#
#   L = E_{t, x0, x1} || v(t, (1-t) x_0 + t x_1) - (x_1 - x_0) ||^2.
#
# With independently drawn pairs the flow matches nu but its paths are not the
# optimal-transport ones. Coupling each minibatch with an entropic OT plan
# (`coupling = :ot`) straightens them and brings the induced map closer to the
# Monge map.

"""
    _pair_independent(sample_mu, sample_nu, dim)

Pairing function drawing `x0` and `x1` independently (vanilla flow matching).
"""
function _pair_independent(sample_mu, sample_nu, dim::Int)
    return function (n::Int)
        X0 = _f32(_check_batch(sample_mu(n), dim, "sample_mu"))
        X1 = _f32(_check_batch(sample_nu(n), dim, "sample_nu"))
        return X0, X1
    end
end

"""
    _pair_ot(sample_mu, sample_nu, dim; epsilon, n_iter, cost, rng)

Pairing function that reorders the target batch by sampling from the entropic
optimal transport plan between the two minibatches.
"""
function _pair_ot(sample_mu, sample_nu, dim::Int; epsilon::Real, n_iter::Int,
                  cost, rng::AbstractRNG)
    c = _as_cost(cost)
    return function (n::Int)
        X0 = _f32(_check_batch(sample_mu(n), dim, "sample_mu"))
        X1 = _f32(_check_batch(sample_nu(n), dim, "sample_nu"))
        C = cost_matrix(c, X0, X1)
        P = sinkhorn_plan(C; epsilon = epsilon, n_iter = n_iter)
        idx = _sample_categorical(rng, P)
        return X0, X1[:, idx]
    end
end

"""
    _flow_core(pair, dim; kwargs...)

Shared training loop behind [`flow_match`](@ref) and [`rectify`](@ref). `pair`
is a function `n -> (X0, X1)` returning a matched batch.
"""
function _flow_core(pair, dim::Int;
                    hidden::Vector{Int}, activation, n_fourier::Int,
                    batch::Int, steps::Int, lr::Real, sigma::Real,
                    log_every::Int, eval_batch::Int, rng::AbstractRNG,
                    verbose::Bool, callback, method_tag::Symbol)
    vfield = VelocityNet(dim; hidden = hidden, activation = activation,
                         n_fourier = n_fourier, rng = rng)
    opt = Flux.setup(Adam(lr), vfield)
    sig = Float32(sigma)

    losses = Float64[]
    evals = Float64[]
    logged = Int[]

    function loss_fn(net, X0, X1, t, noise)
        Xt = (1f0 .- t) .* X0 .+ t .* X1
        sig > 0 && (Xt = Xt .+ sig .* noise)
        return mean(abs2, net(t, Xt) .- (X1 .- X0))
    end

    t0 = time()
    for step in 1:steps
        X0, X1 = pair(batch)
        t = rand(rng, Float32, 1, batch)
        noise = sig > 0 ? randn(rng, Float32, dim, batch) : X0
        gs = Flux.gradient(m -> loss_fn(m, X0, X1, t, noise), vfield)[1]
        opt, vfield = Flux.update!(opt, vfield, gs)

        if _log_step(step, log_every, steps)
            teval = rand(rng, Float32, 1, batch)
            l = Float64(loss_fn(vfield, X0, X1, teval, noise))
            push!(losses, l)
            push!(logged, step)
            ev = nothing
            if eval_batch > 0
                E0, E1 = pair(eval_batch)
                te = rand(rng, Float32, 1, eval_batch)
                ne = sig > 0 ? randn(rng, Float32, dim, eval_batch) : E0
                ev = Float64(loss_fn(vfield, E0, E1, te, ne))
                push!(evals, ev)
            end
            _report(verbose, method_tag, step, steps, l, ev)
            _run_callback(callback, step, l, (vfield = vfield,)) || break
        end
    end
    return vfield, losses, evals, logged, time() - t0
end

"""
    flow_match(sample_mu, sample_nu; dim, kwargs...)

Train a flow-matching vector field transporting `mu` to `nu`.

# Keyword arguments
- `dim::Int` (required).
- `hidden::Vector{Int} = [128, 128]`, `activation = swish`.
- `n_fourier::Int = 0`: number of Fourier time features (0 concatenates raw `t`).
- `coupling::Symbol = :independent`: `:independent` for vanilla flow matching,
  `:ot` to couple each minibatch with an entropic OT plan, which straightens the
  learned paths and moves the induced map towards the true Monge map.
- `coupling_epsilon::Real = 0.05`, `coupling_iter::Int = 100`: the minibatch
  Sinkhorn settings used when `coupling = :ot`.
- `cost = SqEuclideanCost()`: cost for the coupling.
- `sigma::Real = 0.0`: standard deviation of Gaussian noise added to the
  interpolant, giving a probability path of positive width.
- `batch::Int = 256`, `steps::Int = 10_000`, `lr::Real = 1e-3`.
- `log_every::Int = 100`, `eval_batch::Int = 0`, `seed = nothing`,
  `verbose::Bool = false`, `callback = nothing`.

# Returns
A [`NeuralOTResult`](@ref) with `models = (vfield = ...,)`. Push samples forward
with [`monge_map`](@ref) (which integrates the ODE) or [`integrate_flow`](@ref).

# Example
```julia
res = flow_match(gaussian_sampler(2), two_moons(); dim = 2, steps = 4000)
samples = monge_map(res, gaussian_sampler(2)(1000); n_flow_steps = 100)
```
"""
function flow_match(sample_mu, sample_nu;
                    dim::Int,
                    hidden::Vector{Int} = [128, 128],
                    activation = swish,
                    n_fourier::Int = 0,
                    coupling::Symbol = :independent,
                    coupling_epsilon::Real = 0.05,
                    coupling_iter::Int = 100,
                    cost = SqEuclideanCost(),
                    sigma::Real = 0.0,
                    batch::Int = 256,
                    steps::Int = 10_000,
                    lr::Real = 1e-3,
                    log_every::Int = 100,
                    eval_batch::Int = 0,
                    seed = nothing,
                    verbose::Bool = false,
                    callback = nothing)
    _validate_common(steps, batch, lr, log_every)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    coupling in (:independent, :ot) || throw(ArgumentError(
        "coupling must be :independent or :ot, got :$coupling"))
    sigma >= 0 || throw(ArgumentError("sigma must be non-negative"))

    rng = _rng(seed)
    pair = coupling === :ot ?
        _pair_ot(sample_mu, sample_nu, dim; epsilon = coupling_epsilon,
                 n_iter = coupling_iter, cost = cost, rng = rng) :
        _pair_independent(sample_mu, sample_nu, dim)

    vfield, losses, evals, logged, el = _flow_core(
        pair, dim; hidden = hidden, activation = activation, n_fourier = n_fourier,
        batch = batch, steps = steps, lr = lr, sigma = sigma, log_every = log_every,
        eval_batch = eval_batch, rng = rng, verbose = verbose, callback = callback,
        method_tag = :flow)

    return NeuralOTResult(
        (vfield = vfield,), losses, :flow,
        (hidden = hidden, batch = batch, steps = steps, lr = lr, dim = dim,
         coupling = coupling, sigma = sigma, n_fourier = n_fourier),
        logged, evals, el,
    )
end

"""
    rectify(result, sample_mu; dim, n_flow_steps=100, kwargs...)

Reflow: retrain a vector field on the pairs `(x0, T(x0))` induced by an already
trained flow, where `T` integrates `result`'s ODE. Repeating this straightens
the trajectories, so that fewer integration steps are needed and the induced
coupling moves towards the optimal one (Liu et al., 2023).

Accepts the same keyword arguments as [`flow_match`](@ref) apart from the
coupling options, which do not apply: the pairing comes from the previous flow.

# Example
```julia
res1 = flow_match(mu, nu; dim = 2, steps = 4000)
res2 = rectify(res1, mu; dim = 2, steps = 4000)
# res2 needs far fewer integration steps for the same accuracy
```
"""
function rectify(result::NeuralOTResult, sample_mu;
                 dim::Int,
                 n_flow_steps::Int = 100,
                 solver::Symbol = :rk4,
                 hidden::Vector{Int} = [128, 128],
                 activation = swish,
                 n_fourier::Int = 0,
                 sigma::Real = 0.0,
                 batch::Int = 256,
                 steps::Int = 10_000,
                 lr::Real = 1e-3,
                 log_every::Int = 100,
                 eval_batch::Int = 0,
                 seed = nothing,
                 verbose::Bool = false,
                 callback = nothing)
    result.method === :flow || throw(ArgumentError(
        "rectify expects a :flow result, got :$(result.method)"))
    _validate_common(steps, batch, lr, log_every)
    rng = _rng(seed)
    vf = result.models.vfield

    pair = function (n::Int)
        X0 = _f32(_check_batch(sample_mu(n), dim, "sample_mu"))
        X1 = integrate_flow(vf, X0; n_steps = n_flow_steps, solver = solver)
        return X0, X1
    end

    vfield, losses, evals, logged, el = _flow_core(
        pair, dim; hidden = hidden, activation = activation, n_fourier = n_fourier,
        batch = batch, steps = steps, lr = lr, sigma = sigma, log_every = log_every,
        eval_batch = eval_batch, rng = rng, verbose = verbose, callback = callback,
        method_tag = :rectify)

    return NeuralOTResult(
        (vfield = vfield,), losses, :flow,
        (hidden = hidden, batch = batch, steps = steps, lr = lr, dim = dim,
         coupling = :rectified, sigma = sigma, n_fourier = n_fourier,
         reflow_of = result.config),
        logged, evals, el,
    )
end

"""
    _eval_vfield(vfield, t, x)

Evaluate a vector field. `VelocityNet` takes `(t, x)`; anything else (a bare
`Flux.Chain`, as v0.1 produced) is called with the stacked `[t; x]` matrix.
"""
_eval_vfield(vf::VelocityNet, t, x) = vf(t, x)
_eval_vfield(vf, t, x) = vf(vcat(t, x))

"""
    integrate_flow(vfield, x0; n_steps=100, solver=:rk4, t0=0.0, t1=1.0)

Integrate `dx/dt = vfield(t, x)` from `t0` to `t1` starting at `x0` (`dim x B`).

`solver` is `:euler` (1st order), `:heun` (2nd) or `:rk4` (4th). Higher-order
solvers cost more evaluations per step but let you use far fewer steps; on a
trained flow all three agree once the step count is adequate.

Set `t0 = 1, t1 = 0` to integrate backwards, which inverts the map.

# Example
```julia
X1 = integrate_flow(res.models.vfield, X0; n_steps = 50, solver = :rk4)
X0r = integrate_flow(res.models.vfield, X1; n_steps = 50, t0 = 1.0, t1 = 0.0)
```
"""
function integrate_flow(vfield, x0::AbstractMatrix; n_steps::Int = 100,
                        solver::Symbol = :rk4, t0::Real = 0.0, t1::Real = 1.0)
    n_steps >= 1 || throw(ArgumentError("n_steps must be at least 1, got $n_steps"))
    solver in (:euler, :heun, :rk4) || throw(ArgumentError(
        "solver must be :euler, :heun or :rk4, got :$solver"))
    x = _f32(x0)
    B = size(x, 2)
    dt = Float32(t1 - t0) / n_steps
    v(t, z) = _eval_vfield(vfield, fill(Float32(t), 1, B), z)

    for k in 0:(n_steps - 1)
        t = Float32(t0) + k * dt
        if solver === :euler
            x = x .+ dt .* v(t, x)
        elseif solver === :heun
            k1 = v(t, x)
            k2 = v(t + dt, x .+ dt .* k1)
            x = x .+ (dt / 2f0) .* (k1 .+ k2)
        else
            k1 = v(t, x)
            k2 = v(t + dt / 2f0, x .+ (dt / 2f0) .* k1)
            k3 = v(t + dt / 2f0, x .+ (dt / 2f0) .* k2)
            k4 = v(t + dt, x .+ dt .* k3)
            x = x .+ (dt / 6f0) .* (k1 .+ 2f0 .* k2 .+ 2f0 .* k3 .+ k4)
        end
    end
    return x
end
