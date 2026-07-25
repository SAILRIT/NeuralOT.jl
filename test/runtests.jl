using NeuralOT
using Test
using Random
using LinearAlgebra
using Statistics
using Flux
using Zygote
import Distances

# Set NEURALOT_QUICK=1 to run a reduced version of the training-heavy tests.
const QUICK = get(ENV, "NEURALOT_QUICK", "0") == "1"
scale_steps(n) = QUICK ? max(50, n ÷ 4) : n

include("testhelpers.jl")

@testset "NeuralOT.jl" begin
    include("test_utils.jl")
    include("test_costs.jl")
    include("test_icnn.jl")
    include("test_potentials.jl")
    include("test_sinkhorn.jl")
    include("test_gaussian.jl")
    include("test_metrics.jl")
    include("test_datasets.jl")
    include("test_result.jl")
    include("test_dual.jl")
    include("test_w2.jl")
    include("test_flow.jl")
    include("test_maps.jl")
    include("test_integration.jl")
end
