# Estimating a W2 Monge map between two Gaussians, scored against the exact
# Brenier map.
#
#   julia --project=. examples/gaussian_w2.jl

using NeuralOT
using LinearAlgebra
using Statistics
using Random

Random.seed!(0)

const D = 2
m0, m1 = zeros(D), [3.0, 0.0]
S0 = Matrix(0.25I, D, D)
S1 = Matrix(0.25I, D, D)

sample_mu = gaussian_sampler(D; mean = m0, cov = S0, rng = MersenneTwister(1))
sample_nu = gaussian_sampler(D; mean = m1, cov = S1, rng = MersenneTwister(2))

reference = gaussian_ot(m0, S0, m1, S1)
println("exact W2^2            : ", round(reference.w2; digits = 4))

result = solve_w2(sample_mu, sample_nu; dim = D, widths = [64, 64, 1],
                  steps = 2_000, inner_steps = 8, batch = 256, lr = 1e-3,
                  log_every = 200, seed = 42, verbose = true)

X = sample_mu(2_000)
Y = sample_nu(2_000)
T = monge_map(result, X)

println()
println("learned vs exact map (RMSE) : ", round(transport_error(T, reference.map(X)); digits = 4))
println("W2^2 estimate               : ", round(NeuralOT.w2_estimate(result, X, Y); digits = 4))
println("S(T#mu, nu)                 : ", round(sinkhorn_divergence(T, Y; ε = 0.5); digits = 4))
println("S(mu,   nu)                 : ", round(sinkhorn_divergence(X, Y; ε = 0.5); digits = 4))

err = moment_error(T, Y)
println("moment error (mean, cov)    : ", round(err.mean; digits = 4), ", ",
        round(err.cov; digits = 4))

# The inverse direction comes for free from the second potential.
back = inverse_map(result, Y)
println("inverse map RMSE            : ",
        round(transport_error(back, (inv(reference.A) * (Float64.(Y) .- reference.b))); digits = 4))
