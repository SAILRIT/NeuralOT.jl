# Turning a trained result into an actual transport map.

"""
    monge_map(result, x; n_flow_steps=100, solver=:rk4)

Push the columns of `x` (`dim x B`) forward through the learned transport map.

Behaviour depends on `result.method`:

| method     | map |
|------------|-----|
| `:w2_icnn` | `grad g(x)`, the Brenier map given by the convex potential |
| `:flow`    | integrates `dx/dt = v(t, x)` from `t = 0` to `t = 1` |
| `:dual`    | the entropic map `x - grad u(x) / 2` (see [`entropic_map`](@ref)) |

For a `:dual` result, [`barycentric_map`](@ref) is more accurate when you can
supply samples from the target.

# Example
```julia
Y = monge_map(res, X)
```
"""
function monge_map(result::NeuralOTResult, x::AbstractMatrix;
                   n_flow_steps::Int = 100, solver::Symbol = :rk4)
    X = _f32(x)
    if result.method === :w2_icnn
        return input_gradient(result.models.g, X)
    elseif result.method === :flow
        return integrate_flow(result.models.vfield, X; n_steps = n_flow_steps,
                              solver = solver)
    elseif result.method === :dual
        return entropic_map(result, X)
    else
        throw(ArgumentError("unknown method :$(result.method)"))
    end
end

monge_map(result::NeuralOTResult, x::AbstractVector; kwargs...) =
    vec(monge_map(result, reshape(x, :, 1); kwargs...))

"""
    pushforward(result, x; kwargs...)

Alias for [`monge_map`](@ref), reading better when the emphasis is on moving a
whole sample rather than evaluating a map.
"""
pushforward(result::NeuralOTResult, x; kwargs...) = monge_map(result, x; kwargs...)

"""
    inverse_map(result, y; n_flow_steps=100, solver=:rk4)

Transport `y` backwards, from `nu` to `mu`.

- `:w2_icnn`: applies `grad f`, the gradient of the other convex potential,
  which is the inverse map at the saddle point.
- `:flow`: integrates the ODE from `t = 1` back to `t = 0`.
- `:dual`: not available - the dual formulation learns a map in one direction
  only. Train a second model with the arguments swapped.
"""
function inverse_map(result::NeuralOTResult, y::AbstractMatrix;
                     n_flow_steps::Int = 100, solver::Symbol = :rk4)
    Y = _f32(y)
    if result.method === :w2_icnn
        return input_gradient(result.models.f, Y)
    elseif result.method === :flow
        return integrate_flow(result.models.vfield, Y; n_steps = n_flow_steps,
                              solver = solver, t0 = 1.0, t1 = 0.0)
    elseif result.method === :dual
        throw(ArgumentError(
            "inverse_map is not defined for :dual results: the entropic dual gives " *
            "a map from mu to nu only. Train a second model with mu and nu swapped."))
    else
        throw(ArgumentError("unknown method :$(result.method)"))
    end
end

inverse_map(result::NeuralOTResult, y::AbstractVector; kwargs...) =
    vec(inverse_map(result, reshape(y, :, 1); kwargs...))

"""
    entropic_map(result, x)

The barycentric projection of the entropic plan, in closed form from the first
potential:

```math
T(x) = x - \\tfrac{1}{2}\\nabla u(x).
```

The constant follows from the optimality condition of the entropic dual: with
`c(x, y) = ||x - y||^2` the potential satisfies
`grad u(x) = E[grad_x c(x, Y) | x] = 2(x - T(x))`. There is no `1/epsilon`
factor - v0.1 divided by `2 * epsilon`, which scaled the correction by `1/eps`
(a factor of 10 at the default `eps = 0.1`) and produced a badly wrong map.

Only defined for the squared Euclidean cost.

!!! note
    This is exact only at the optimum and for an exactly represented potential.
    [`barycentric_map`](@ref) computes the same projection directly from target
    samples and is usually more accurate.
"""
function entropic_map(result::NeuralOTResult, x::AbstractMatrix)
    result.method === :dual || throw(ArgumentError(
        "entropic_map expects a :dual result, got :$(result.method)"))
    is_squared_euclidean(get(result.config, :cost, SqEuclideanCost())) || throw(ArgumentError(
        "entropic_map is only defined for the squared Euclidean cost; " *
        "use barycentric_map instead"))
    X = _f32(x)
    gu = input_gradient(result.models.u, X)
    return X .- 0.5f0 .* gu
end

entropic_map(result::NeuralOTResult, x::AbstractVector) =
    vec(entropic_map(result, reshape(x, :, 1)))

"""
    barycentric_map(result, x, Y)

Barycentric projection of the learned entropic plan onto samples: each column of
`x` is mapped to the weighted average of the columns of `Y`,

```math
T(x) = \\sum_j w_j(x)\\, y_j, \\qquad
w_j(x) \\propto \\exp\\!\\big((u(x) + v(y_j) - c(x, y_j)) / \\varepsilon\\big).
```

`Y` should be a reasonably large sample from the target. The weights are
normalised per input point, so this is insensitive to the constant shift that
the dual potentials are only determined up to.

Works for any cost, and recovered the exact Brenier map to RMSE 0.078 on the
Gaussian benchmark used to validate this package.

# Example
```julia
T = barycentric_map(res, sample_mu(256), sample_nu(4096))
```
"""
function barycentric_map(result::NeuralOTResult, x::AbstractMatrix, Y::AbstractMatrix)
    result.method === :dual || throw(ArgumentError(
        "barycentric_map expects a :dual result, got :$(result.method)"))
    X = _f32(x)
    Yf = _f32(Y)
    c = get(result.config, :cost, SqEuclideanCost())
    eps32 = Float32(get(result.config, :epsilon, 0.1))
    ux = vec(result.models.u(X))
    vy = vec(result.models.v(Yf))
    C = cost_matrix(c, X, Yf)
    logw = (ux .+ vy' .- C) ./ eps32
    logw = logw .- _logsumexp(logw; dims = 2)
    return Yf * exp.(logw)'
end
