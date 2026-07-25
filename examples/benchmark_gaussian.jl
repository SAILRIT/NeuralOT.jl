# Score all three solvers on the same problem, against the exact answer.
#
#   julia --project=. examples/benchmark_gaussian.jl

using NeuralOT
using LinearAlgebra
using Statistics
using Printf
using Random

Random.seed!(0)

# An anisotropic target, so that a solver cannot pass by learning a shift.
m0, S0 = [0.0, 0.0], [0.5 0.0; 0.0 0.5]
m1, S1 = [1.0, -1.0], [2.0 0.4; 0.4 0.3]

sample_mu = gaussian_sampler(2; mean = m0, cov = S0, rng = MersenneTwister(1))
sample_nu = gaussian_sampler(2; mean = m1, cov = S1, rng = MersenneTwister(2))
reference = gaussian_ot(m0, S0, m1, S1)

X = sample_mu(4_000)
Y = sample_nu(4_000)
Texact = reference.map(X)

@printf("exact W2^2 = %.4f\n", reference.w2)
@printf("%-14s %10s %10s %10s %10s\n", "method", "map RMSE", "S(T,nu)", "mean err", "seconds")
@printf("%-14s %10.4f %10.4f %10.4f %10s\n", "identity",
        transport_error(X, Texact), sinkhorn_divergence(X, Y; ε = 0.5),
        moment_error(X, Y).mean, "-")

runs = (
    ("dual", () -> solve_dual(sample_mu, sample_nu; dim = 2, ε = 0.05, steps = 4_000,
                              batch = 256, hidden = [128, 128], lr = 1e-3, seed = 0),
     (r, X) -> monge_map(r, X)),
    ("w2_icnn", () -> solve_w2(sample_mu, sample_nu; dim = 2, widths = [64, 64, 1],
                               steps = 3_000, inner_steps = 8, batch = 256,
                               lr = 1e-3, seed = 0),
     (r, X) -> monge_map(r, X)),
    ("flow", () -> flow_match(sample_mu, sample_nu; dim = 2, hidden = [128, 128],
                              steps = 6_000, batch = 256, lr = 1e-3, seed = 0),
     (r, X) -> monge_map(r, X; n_flow_steps = 100)),
    ("flow (OT)", () -> flow_match(sample_mu, sample_nu; dim = 2, hidden = [128, 128],
                                   steps = 6_000, batch = 256, lr = 1e-3,
                                   coupling = :ot, seed = 0),
     (r, X) -> monge_map(r, X; n_flow_steps = 100)),
)

for (name, train, apply) in runs
    res = train()
    T = apply(res, X)
    @printf("%-14s %10.4f %10.4f %10.4f %10.1f\n", name,
            transport_error(T, Texact), sinkhorn_divergence(T, Y; ε = 0.5),
            moment_error(T, Y).mean, res.elapsed)
end
