# Independent versus minibatch-OT couplings in flow matching.
#
# Both reach the target; the OT coupling produces a straighter flow whose
# induced transport cost is closer to the true W2.
#
#   julia --project=. examples/ot_cfm.jl

using NeuralOT
using LinearAlgebra
using Statistics
using Random

Random.seed!(0)

const D = 2
S = Matrix(0.25I, D, D)
sample_mu = gaussian_sampler(D; cov = S, rng = MersenneTwister(1))
sample_nu = gaussian_sampler(D; mean = [3.0, 0.0], cov = S, rng = MersenneTwister(2))
reference = gaussian_ot(zeros(D), S, [3.0, 0.0], S)

X = sample_mu(2_000)
Y = sample_nu(2_000)

println("true W2^2 = ", round(reference.w2; digits = 4))
for coupling in (:independent, :ot)
    res = flow_match(sample_mu, sample_nu; dim = D, hidden = [128, 128],
                     steps = 3_000, batch = 128, lr = 1e-3, coupling = coupling,
                     coupling_epsilon = 0.1, seed = 0)
    T = monge_map(res, X; n_flow_steps = 100)
    induced_cost = mean(sum(abs2, T .- X; dims = 1))
    println(rpad(string(coupling), 12),
            "  S(T#mu, nu) = ", round(sinkhorn_divergence(T, Y; ε = 0.5); digits = 4),
            "  induced cost = ", round(induced_cost; digits = 4),
            "  |gap to W2^2| = ", round(abs(induced_cost - reference.w2); digits = 4))
end
