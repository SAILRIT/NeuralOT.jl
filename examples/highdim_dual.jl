# Entropic dual OT in a dimension where a full cost matrix over the data set
# would be hopeless, with both map-recovery routes compared.
#
#   julia --project=. examples/highdim_dual.jl

using NeuralOT
using LinearAlgebra
using Statistics
using Random

Random.seed!(0)

const D = 32
shift = fill(1.0, D)
S = Matrix(1.0I, D, D)

sample_mu = gaussian_sampler(D; sigma = 1.0, rng = MersenneTwister(1))
sample_nu = gaussian_sampler(D; mean = shift, sigma = 1.0, rng = MersenneTwister(2))

reference = gaussian_ot(zeros(D), S, shift, S)
println("dimension              : ", D)
println("exact W2^2             : ", round(reference.w2; digits = 4))

result = solve_dual(sample_mu, sample_nu; dim = D, ε = 0.5, steps = 4_000,
                    batch = 256, hidden = [256, 256], lr = 1e-3,
                    log_every = 400, seed = 0, verbose = true, eval_batch = 512)

println()
println("dual value (regularised): ", round(NeuralOT.dual_value(result); digits = 4))

X = sample_mu(512)
Y = sample_nu(4_096)

T_closed = entropic_map(result, X)         # x - grad u(x) / 2
T_bary   = barycentric_map(result, X, Y)   # weighted average of target samples

for (name, T) in (("entropic_map  ", T_closed), ("barycentric_map", T_bary))
    println(name, " RMSE vs exact map: ", round(transport_error(T, reference.map(X)); digits = 4),
            "   S(T#mu, nu): ", round(sinkhorn_divergence(T, sample_nu(512); ε = 1.0); digits = 4))
end
