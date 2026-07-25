# Container returned by every solver.

"""
    NeuralOTResult

Trained neural OT models together with the training history and the exact
configuration that produced them.

# Fields
- `models::NamedTuple`: the trained networks. `(u, v)` for `:dual`, `(f, g)`
  for `:w2_icnn`, `(vfield,)` for `:flow`.
- `losses::Vector{Float64}`: training loss at each logged step.
- `method::Symbol`: `:dual`, `:w2_icnn` or `:flow`.
- `config::NamedTuple`: every hyper-parameter used, including the cost object,
  so a result is self-describing.
- `logged_steps::Vector{Int}`: the step index each loss belongs to.
- `eval_losses::Vector{Float64}`: loss on a freshly drawn batch at each logged
  step (empty if evaluation was disabled). Because every batch is new, the
  training loss is already an out-of-sample estimate; `eval_losses` uses an
  independent draw and a fixed evaluation size, which makes it less noisy.
- `elapsed::Float64`: wall-clock training time in seconds.

The first four fields keep the names, order and meaning they had in v0.1, so
`NeuralOTResult(models, losses, method, config)` still works.

# Example
```julia
res = solve_dual(sample_mu, sample_nu; dim = 2, steps = 500)
res.method                # :dual
res.losses[end]           # final training loss
res.config.epsilon        # the regularisation actually used
monge_map(res, x)
```
"""
struct NeuralOTResult
    models::NamedTuple
    losses::Vector{Float64}
    method::Symbol
    config::NamedTuple
    logged_steps::Vector{Int}
    eval_losses::Vector{Float64}
    elapsed::Float64
end

function NeuralOTResult(models::NamedTuple, losses::Vector{Float64}, method::Symbol,
                        config::NamedTuple)
    return NeuralOTResult(models, losses, method, config,
                          collect(1:length(losses)), Float64[], NaN)
end

"""
    converged(result; window=5, rtol=1e-3) -> Bool

Crude convergence heuristic: whether the mean logged loss over the last `window`
entries differs from the preceding `window` by less than `rtol` in relative
terms. Useful in scripts; not a substitute for looking at the curve.
"""
function converged(r::NeuralOTResult; window::Int = 5, rtol::Real = 1e-3)
    length(r.losses) < 2 * window && return false
    tail = r.losses[(end - window + 1):end]
    prev = r.losses[(end - 2 * window + 1):(end - window)]
    m1, m0 = mean(tail), mean(prev)
    scale = max(abs(m0), abs(m1), eps())
    return abs(m1 - m0) / scale < rtol
end

"""
    loss_history(result) -> (steps, losses)

The logged training curve as a tuple of vectors, convenient for plotting.
"""
loss_history(r::NeuralOTResult) = (r.logged_steps, r.losses)

function Base.show(io::IO, ::MIME"text/plain", r::NeuralOTResult)
    println(io, "NeuralOTResult(:", r.method, ")")
    println(io, "  models      : ", join(string.(keys(r.models)), ", "))
    if isempty(r.losses)
        println(io, "  losses      : (none logged)")
    else
        println(io, "  losses      : ", length(r.losses), " logged, ",
                "first ", _fmt(first(r.losses)), " -> last ", _fmt(last(r.losses)))
    end
    if !isempty(r.eval_losses)
        println(io, "  eval losses : last ", _fmt(last(r.eval_losses)))
    end
    isnan(r.elapsed) || println(io, "  elapsed     : ", _fmt(r.elapsed), " s")
    print(io, "  config      : ", r.config)
    return nothing
end

Base.show(io::IO, r::NeuralOTResult) =
    print(io, "NeuralOTResult(:", r.method, ", ", length(r.losses), " logged steps)")
