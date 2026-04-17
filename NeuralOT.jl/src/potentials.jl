# Generic (non-convex) potential network for Seguy-style dual OT.

"""
    DualPotentialNet(dim; hidden=[128, 128], activation=relu)

Generic MLP parameterising a scalar potential `R^dim -> R`. Used for the
Seguy et al. (2018) dual regularised-OT formulation where the potentials
need not be convex.

# Example
```julia
u = DualPotentialNet(2; hidden=[64, 64])
v = DualPotentialNet(2; hidden=[64, 64])
```
"""
struct DualPotentialNet
    net::Chain
end

Flux.@layer DualPotentialNet

function DualPotentialNet(dim::Int; hidden::Vector{Int}=[128, 128],
                          activation=relu)
    layers = Any[]
    prev = dim
    for h in hidden
        push!(layers, Dense(prev, h, activation))
        prev = h
    end
    push!(layers, Dense(prev, 1))
    DualPotentialNet(Chain(layers...))
end

(p::DualPotentialNet)(x::AbstractMatrix) = p.net(x)
(p::DualPotentialNet)(x::AbstractVector) = vec(p.net(reshape(x, :, 1)))
