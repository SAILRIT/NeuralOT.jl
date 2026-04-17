# Example: Seguy-style entropic dual OT on high-dimensional Gaussians.
#
# Demonstrates that the dual approach scales to dimensions where building
# a full cost matrix is infeasible. We use d=50 with 1024 samples per step
# (dense discrete solvers would need a 50k × 50k cost matrix for comparable
# effective sample size; the dual method never materializes one).

using NeuralOT
using Random
using LinearAlgebra
using Statistics

Random.seed!(7)
d = 50

# Source: standard normal. Target: shifted, correlated.
shift = Float32.(vcat(2.0, zeros(d - 1)))
A = Float32.(0.3 * randn(d, d) + I)   # mild anisotropy

sample_μ(n) = randn(Float32, d, n)
sample_ν(n) = A * randn(Float32, d, n) .+ shift

println("Training dual potentials (d=$d)...")
result = solve_dual(sample_μ, sample_ν;
                    dim=d, ε=1.0, hidden=[256, 256],
                    steps=3_000, batch=512,
                    lr=3e-4, log_every=100, seed=0)

println("Dual objective (negative): ", round(result.losses[end]; digits=4))

# The learned potentials give an approximate map via the barycentric
# projection — useful for visualizing where the first coordinates land.
X = sample_μ(1024)
T_X = monge_map(result, X)

# Check the first coordinate shifts toward the target mean ~2.0
println("Source first-coord mean: ", round(mean(X[1, :]); digits=3))
println("Mapped first-coord mean: ", round(mean(T_X[1, :]); digits=3),
        " (target ≈ $(shift[1]))")
