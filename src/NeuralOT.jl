"""
    NeuralOT

Neural optimal transport for Julia. Estimate Monge maps, dual potentials and
transport-based generative models with neural networks, plus the discrete
solvers and closed-form references needed to check that the result is right.

# Models
- [`ICNN`](@ref) - input convex neural network (Amos et al., 2017)
- [`DualPotentialNet`](@ref) - generic MLP potential
- [`VelocityNet`](@ref) - time-conditioned vector field

# Solvers
- [`solve_dual`](@ref) - regularised dual OT (Seguy et al., 2018)
- [`solve_w2`](@ref) - W2 Monge map via ICNNs (Makkuva et al., 2020)
- [`flow_match`](@ref) - flow matching / rectified flow (Lipman et al., 2023)
- [`rectify`](@ref) - reflow to straighten a trained flow

# Maps and evaluation
- [`monge_map`](@ref), [`inverse_map`](@ref), [`barycentric_map`](@ref)
- [`sinkhorn_divergence`](@ref), [`energy_distance`](@ref), [`mmd`](@ref)
- [`gaussian_ot`](@ref) - closed-form Gaussian OT for validation
"""
module NeuralOT

using Flux
using Flux: Chain, Dense, relu, softplus, sigmoid, swish, elu, Adam
using LinearAlgebra
using Statistics
using Random
using Printf
import Distances
using Zygote

include("utils.jl")
include("costs.jl")
include("icnn.jl")
include("potentials.jl")
include("result.jl")
include("sinkhorn.jl")
include("gaussian.jl")
include("metrics.jl")
include("datasets.jl")
include("training.jl")
include("dual.jl")
include("monge.jl")
include("flow.jl")
include("maps.jl")

# models
export ICNN, DualPotentialNet, VelocityNet
export input_gradient, grad_x

# costs
export SqEuclideanCost, EuclideanCost, GenericCost, cost_matrix

# solvers
export solve_dual, solve_w2, flow_match, rectify
export NeuralOTResult

# maps
export monge_map, inverse_map, barycentric_map, entropic_map, pushforward
export integrate_flow

# discrete OT
export sinkhorn_potentials, sinkhorn_plan, sinkhorn_value, sinkhorn_cost
export sinkhorn_divergence

# metrics and references
export energy_distance, mmd, moment_error, transport_error
export gaussian_ot, gaussian_brenier_map, w2_gaussian

# datasets
export gaussian_sampler, two_moons, eight_gaussians, checkerboard
export swiss_roll, circles, uniform_box

end # module
