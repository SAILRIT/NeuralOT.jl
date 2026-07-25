# Flow matching from a Gaussian to the two-moons distribution, comparing
# integrators and showing what reflow buys.
#
#   julia --project=. examples/moons_flow.jl

using NeuralOT
using Statistics
using Random

Random.seed!(0)

sample_mu = gaussian_sampler(2; sigma = 1.0, rng = MersenneTwister(1))
sample_nu = two_moons(; noise = 0.05, rng = MersenneTwister(2))

result = flow_match(sample_mu, sample_nu; dim = 2, hidden = [128, 128],
                    steps = 5_000, batch = 256, lr = 1e-3, log_every = 500,
                    seed = 0, verbose = true)

X = sample_mu(2_000)
Y = sample_nu(2_000)

println()
println("baseline S(mu, nu) : ", round(sinkhorn_divergence(X, Y; ε = 0.1); digits = 4))
for solver in (:euler, :heun, :rk4), n in (4, 20, 100)
    T = monge_map(result, X; n_flow_steps = n, solver = solver)
    println("  ", rpad(string(solver), 6), " n=", lpad(n, 3), "  S = ",
            round(sinkhorn_divergence(T, Y; ε = 0.1); digits = 4))
end

# Reflow: retrain on the pairs the current flow induces. The paths straighten,
# so a coarse integration becomes as good as a fine one.
rect = rectify(result, sample_mu; dim = 2, hidden = [128, 128], steps = 5_000,
               batch = 256, lr = 1e-3, n_flow_steps = 100, seed = 1, verbose = true)

println()
println("after one reflow:")
for n in (4, 20, 100)
    T = monge_map(rect, X; n_flow_steps = n, solver = :euler)
    println("  euler  n=", lpad(n, 3), "  S = ",
            round(sinkhorn_divergence(T, Y; ε = 0.1); digits = 4))
end
