# Example: transport a standard Gaussian to the two-moons distribution
# using flow matching.
#
# Two moons is the canonical low-dim benchmark for neural OT generative
# models. We train a flow-matching network and then sample by integrating
# the learned ODE forward from noise.

using NeuralOT
using Random
using Statistics

Random.seed!(42)

# Two-moons sampler (standard formulation from scikit-learn)
function sample_moons(n; noise=0.05f0)
    n1 = n ÷ 2
    n2 = n - n1
    θ1 = Float32.(π .* rand(n1))
    θ2 = Float32.(π .* rand(n2))
    X1 = vcat(cos.(θ1)', sin.(θ1)')
    X2 = vcat((1f0 .- cos.(θ2))', (0.5f0 .- sin.(θ2))')
    X  = hcat(X1, X2) .+ noise .* randn(Float32, 2, n)
    return X[:, randperm(n)]
end

sample_μ(n) = randn(Float32, 2, n)           # source: standard normal
sample_ν(n) = sample_moons(n)                 # target: two moons

println("Training flow-matching model (this takes ~30s on CPU)...")
result = flow_match(sample_μ, sample_ν;
                    dim=2, hidden=[128, 128, 128],
                    steps=8_000, batch=256, lr=5e-4,
                    log_every=200, seed=0)

println("Initial loss: ", round(result.losses[1]; digits=4))
println("Final loss:   ", round(result.losses[end]; digits=4))

# Sample by integrating the ODE
X0 = sample_μ(2048)
X1 = monge_map(result, X0; n_flow_steps=100)

# Evaluate fit against fresh target samples
Y = sample_ν(2048)
sd = sinkhorn_divergence(X1, Y; ε=0.05, n_iter=200)
println("Sinkhorn divergence of generated samples vs. target: ",
        round(sd; digits=4))

# Save samples for plotting (CSV for language-agnostic visualization)
open(joinpath(@__DIR__, "moons_generated.csv"), "w") do io
    println(io, "x,y")
    for j in 1:size(X1, 2)
        println(io, X1[1, j], ",", X1[2, j])
    end
end
println("Saved generated samples to moons_generated.csv")
