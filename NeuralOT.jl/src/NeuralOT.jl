"""
    NeuralOT

Neural optimal transport for Julia. Estimate Monge maps, dual potentials, and
transport-based generative models using neural networks with Flux.jl.

# Exports

## Models
- [`ICNN`](@ref) — Input Convex Neural Network (Amos et al., 2017)
- [`DualPotentialNet`](@ref) — Generic MLP potential network

## Solvers
- [`solve_dual`](@ref) — Seguy-style dual OT (Seguy et al., 2018)
- [`solve_w2`](@ref) — W2 Monge map via ICNNs (Makkuva et al., 2020)
- [`flow_match`](@ref) — Rectified flow / flow matching (Lipman et al., 2023)

## Utilities
- [`monge_map`](@ref) — push-forward from a trained potential
- [`sinkhorn_divergence`](@ref) — for evaluation / comparison
"""
module NeuralOT

using Flux
using Flux: Chain, Dense, relu, softplus, Adam
using LinearAlgebra
using Statistics
using Random
using Distances
using Zygote

include("icnn.jl")
include("potentials.jl")
include("dual.jl")
include("monge.jl")
include("flow.jl")
include("eval.jl")

export ICNN, DualPotentialNet
export solve_dual, solve_w2, flow_match
export monge_map, sinkhorn_divergence
export NeuralOTResult

end # module
