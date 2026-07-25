# Shared fixtures. Every sampler is seeded so failures are reproducible.

"""
    gaussian_problem(; dim=2, shift=3.0, sigma=0.5, seed=0)

A source and a target Gaussian whose optimal transport map is known exactly.
Returns `(sample_mu, sample_nu, reference)` where `reference` comes from
`gaussian_ot` and carries the true map and `W2^2`.
"""
function gaussian_problem(; dim::Int = 2, shift::Real = 3.0, sigma::Real = 0.5,
                          seed::Int = 0)
    m0 = zeros(dim)
    m1 = vcat(Float64(shift), zeros(dim - 1))
    S0 = Matrix((sigma^2) * I, dim, dim)
    S1 = Matrix((sigma^2) * I, dim, dim)
    smu = gaussian_sampler(dim; mean = m0, sigma = sigma, rng = MersenneTwister(seed))
    snu = gaussian_sampler(dim; mean = m1, sigma = sigma, rng = MersenneTwister(seed + 1))
    return smu, snu, gaussian_ot(m0, S0, m1, S1)
end

"""
    anisotropic_problem(; seed=0)

A 2-D problem whose exact map is a genuine linear transformation rather than a
translation, so a solver cannot pass by learning a constant shift.
"""
function anisotropic_problem(; seed::Int = 0)
    m0 = [0.0, 0.0]
    m1 = [1.0, -1.0]
    S0 = [0.5 0.0; 0.0 0.5]
    S1 = [2.0 0.4; 0.4 0.3]
    smu = gaussian_sampler(2; mean = m0, cov = S0, rng = MersenneTwister(seed))
    snu = gaussian_sampler(2; mean = m1, cov = S1, rng = MersenneTwister(seed + 1))
    return smu, snu, gaussian_ot(m0, S0, m1, S1)
end

"""
    finite_difference_gradient(f, x; h=1e-3)

Central-difference gradient of a scalar-valued network at a single point,
used to check analytic gradients.
"""
function finite_difference_gradient(f, x::AbstractVector; h::Real = 1e-3)
    g = zeros(Float32, length(x))
    for i in eachindex(x)
        xp = copy(x); xp[i] += Float32(h)
        xm = copy(x); xm[i] -= Float32(h)
        g[i] = (f(reshape(xp, :, 1))[1] - f(reshape(xm, :, 1))[1]) / (2 * Float32(h))
    end
    return g
end
