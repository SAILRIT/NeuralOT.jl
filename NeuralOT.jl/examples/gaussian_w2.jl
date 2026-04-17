# Example: W2 Monge map between two Gaussians via ICNNs.
#
# Between two Gaussians μ = N(m_μ, Σ_μ) and ν = N(m_ν, Σ_ν), the true W2
# Monge map is known in closed form (Bures map). We use this as a ground-
# truth sanity check for the ICNN solver.

using NeuralOT
using Random
using LinearAlgebra
using Statistics

Random.seed!(0)
d = 2

# Source: isotropic Gaussian at origin
m_μ = zeros(Float32, d)
Σ_μ = Matrix{Float32}(I, d, d)
L_μ = cholesky(Σ_μ).L

# Target: shifted, anisotropic Gaussian
m_ν = Float32[3.0, 1.0]
Σ_ν = Float32[2.0 0.5; 0.5 1.0]
L_ν = cholesky(Σ_ν).L

sample_μ(n) = L_μ * randn(Float32, d, n) .+ m_μ
sample_ν(n) = L_ν * randn(Float32, d, n) .+ m_ν

# Closed-form Bures map: T(x) = m_ν + A (x - m_μ)
# where A = Σ_μ^{-1/2} (Σ_μ^{1/2} Σ_ν Σ_μ^{1/2})^{1/2} Σ_μ^{-1/2}
# (for Σ_μ = I this simplifies to A = Σ_ν^{1/2})
A_true = sqrt(Σ_ν)
T_true(x) = m_ν .+ A_true * (x .- m_μ)

println("Training ICNN-based W2 solver...")
result = solve_w2(sample_μ, sample_ν;
                  dim=d, widths=[64, 64, 1],
                  steps=3_000, inner_steps=10,
                  batch=256, lr=1e-4, log_every=100, seed=1)

println("Final outer loss: ", round(result.losses[end]; digits=4))

# Compare learned map to analytic map on a test batch
X_test = sample_μ(1024)
T_learned = monge_map(result, X_test)
T_analytic = T_true(X_test)

err = mean(sum(abs2, T_learned .- T_analytic; dims=1))
println("Mean squared error vs. analytic Bures map: ", round(err; digits=4))

# Target fit: Sinkhorn divergence between pushed samples and true target samples
sd = sinkhorn_divergence(T_learned, sample_ν(1024); ε=0.1)
println("Sinkhorn divergence to target: ", round(sd; digits=4))
