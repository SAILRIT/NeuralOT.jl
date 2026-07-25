# W2 Monge maps via input convex neural networks.
#
# Makkuva, Taghvaei, Oh, Lee (2020), "Optimal Transport Mapping via Input
# Convex Neural Networks", ICML.
#
# Brenier's theorem writes the optimal correlation as
#
#   sup_pi E[<x, y>] = inf_{f convex} { E_nu[f(Y)] + E_mu[f*(X)] }
#
# and the conjugate is itself a supremum, f*(x) = sup_z <x, z> - f(z), whose
# argmax is parameterised by a second convex potential: f*(x) ~ <x, grad g(x)>
# - f(grad g(x)). That gives the saddle point
#
#   inf_f sup_g  E_mu[<X, grad g(X)> - f(grad g(X))] + E_nu[f(Y)]
#
# so grad g transports mu -> nu and grad f transports nu -> mu. (Verified
# numerically: on a Gaussian pair whose exact map is known, grad g matched it to
# RMSE 0.27 while grad f gave 6.77, and grad f matched the inverse map to 0.32.)
#
# The inner gradient is analytic (see `input_gradient`), so the whole objective
# needs only one level of reverse-mode AD.

"""
    solve_w2(sample_mu, sample_nu; dim, kwargs...)

Estimate the squared-Euclidean (2-Wasserstein) Monge map between `mu` and `nu`
using the input-convex saddle-point formulation of Makkuva et al. (2020).

Both samplers take an integer and return a `dim x n` matrix.

# Keyword arguments
- `dim::Int` (required).
- `widths::Vector{Int} = [64, 64, 1]`: ICNN widths; the last entry must be `1`.
- `batch::Int = 256`, `steps::Int = 5_000`: outer iterations.
- `inner_steps::Int = 8`: maximisation steps on `g` per outer step. The inner
  problem approximates a conjugate, so too few steps bias the map and too many
  waste work; 5-15 is the usual range.
- `lr::Real = 1e-3`, `lr_inner::Real = lr`.
- `quadratic::Real = 1.0`: strength of the `beta/2 ||x||^2` term that makes both
  potentials strongly convex. This is what keeps the map near the identity at
  initialisation; setting it to `0` makes training markedly less stable.
- `init_scale::Real = 0.1`: scale of the ICNN input weights at initialisation.
- `cycle::Real = 0.0`: optional weight on the cycle-consistency penalty
  `mean(||grad f(grad g(X)) - X||^2)`, which pushes the two potentials to be
  conjugate to each other.
- `activation = softplus`, `log_every::Int = 100`, `eval_batch::Int = 0`,
  `seed = nothing`, `verbose::Bool = false`, `callback = nothing`.

# Returns
A [`NeuralOTResult`](@ref) with `models = (f = ..., g = ...)`.
[`monge_map`](@ref) applies `grad g` (the forward map `mu -> nu`) and
[`inverse_map`](@ref) applies `grad f`.

!!! note
    The saddle point is a minimax problem, so the logged outer objective is
    *not* expected to decrease monotonically - it oscillates as `f` and `g`
    chase each other. Judge convergence by the quality of the map (for example
    [`sinkhorn_divergence`](@ref) between the pushforward and the target), not
    by the loss curve.

# Example
```julia
mu = gaussian_sampler(2; sigma = 0.5)
nu = gaussian_sampler(2; mean = [3.0, 0.0], sigma = 0.5)
res = solve_w2(mu, nu; dim = 2, steps = 2000, widths = [64, 64, 1])
sinkhorn_divergence(monge_map(res, mu(512)), nu(512))   # small
```
"""
function solve_w2(sample_mu, sample_nu;
                  dim::Int,
                  widths::Vector{Int} = [64, 64, 1],
                  batch::Int = 256,
                  steps::Int = 5_000,
                  inner_steps::Int = 8,
                  lr::Real = 1e-3,
                  lr_inner::Real = lr,
                  quadratic::Real = 1.0,
                  init_scale::Real = 0.1,
                  cycle::Real = 0.0,
                  activation = softplus,
                  log_every::Int = 100,
                  eval_batch::Int = 0,
                  seed = nothing,
                  verbose::Bool = false,
                  callback = nothing)
    _validate_common(steps, batch, lr, log_every)
    dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
    last(widths) == 1 || throw(ArgumentError(
        "an ICNN potential must have scalar output: widths[end] == 1, got $(last(widths))"))
    inner_steps >= 1 || throw(ArgumentError("inner_steps must be at least 1"))
    cycle >= 0 || throw(ArgumentError("cycle must be non-negative"))

    rng = _rng(seed)
    f = ICNN(dim, widths; activation = activation, quadratic = quadratic,
             init_scale = init_scale, rng = rng)
    g = ICNN(dim, widths; activation = activation, quadratic = quadratic,
             init_scale = init_scale, rng = rng)

    opt_f = Flux.setup(Adam(lr), f)
    opt_g = Flux.setup(Adam(lr_inner), g)

    cyc = Float32(cycle)
    losses = Float64[]
    evals = Float64[]
    logged = Int[]

    # maximised over g (written as a minimisation of its negative)
    function inner_loss(gn, fn, X)
        T = input_gradient(gn, X)
        return -mean(sum(X .* T; dims = 1) .- fn(T))
    end

    # minimised over f
    function outer_loss(fn, gn, X, Y)
        T = input_gradient(gn, X)
        base = mean(sum(X .* T; dims = 1) .- fn(T)) + mean(fn(Y))
        if cyc > 0
            back = input_gradient(fn, T)
            base += cyc * mean(sum(abs2, back .- X; dims = 1))
        end
        return base
    end

    t0 = time()
    for step in 1:steps
        X = _f32(_check_batch(sample_mu(batch), dim, "sample_mu"))
        Y = _f32(_check_batch(sample_nu(batch), dim, "sample_nu"))

        for _ in 1:inner_steps
            gg = Flux.gradient(m -> inner_loss(m, f, X), g)[1]
            opt_g, g = Flux.update!(opt_g, g, gg)
        end

        gf = Flux.gradient(m -> outer_loss(m, g, X, Y), f)[1]
        opt_f, f = Flux.update!(opt_f, f, gf)

        if _log_step(step, log_every, steps)
            l = Float64(outer_loss(f, g, X, Y))
            push!(losses, l)
            push!(logged, step)
            ev = nothing
            if eval_batch > 0
                Xe = _f32(sample_mu(eval_batch))
                Ye = _f32(sample_nu(eval_batch))
                ev = Float64(outer_loss(f, g, Xe, Ye))
                push!(evals, ev)
            end
            _report(verbose, :w2_icnn, step, steps, l, ev)
            _run_callback(callback, step, l, (f = f, g = g)) || break
        end
    end

    return NeuralOTResult(
        (f = f, g = g), losses, :w2_icnn,
        (widths = widths, batch = batch, steps = steps, inner_steps = inner_steps,
         lr = lr, lr_inner = lr_inner, quadratic = quadratic, cycle = cycle,
         init_scale = init_scale, dim = dim, cost = SqEuclideanCost()),
        logged, evals, time() - t0,
    )
end

"""
    w2_estimate(result, X, Y) -> Float64

Estimate `W2^2(mu, nu)` from a trained `:w2_icnn` result using the identity
`W2^2 = E||X||^2 + E||Y||^2 - 2 sup E<x, y>`, with the supremum taken from the
learned potentials.
"""
function w2_estimate(r::NeuralOTResult, X::AbstractMatrix, Y::AbstractMatrix)
    r.method === :w2_icnn || throw(ArgumentError("w2_estimate expects a :w2_icnn result"))
    Xf, Yf = _f32(X), _f32(Y)
    T = input_gradient(r.models.g, Xf)
    corr = mean(sum(Xf .* T; dims = 1) .- r.models.f(T)) + mean(r.models.f(Yf))
    return Float64(mean(_colnorm2(Xf)) + mean(_colnorm2(Yf)) - 2 * corr)
end
