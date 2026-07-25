@testset "metrics" begin
    rng = MersenneTwister(0)
    X = randn(rng, Float32, 2, 64)
    Z = randn(rng, Float32, 2, 64)
    Y = randn(rng, Float32, 2, 64) .+ Float32[5.0; 0.0]

    @testset "sinkhorn_divergence properties" begin
        kw = (n_iter = 20_000, tol = 1e-10)
        s_xx = sinkhorn_divergence(X, X; ε = 0.5, kw...)
        s_xy = sinkhorn_divergence(X, Y; ε = 0.5, kw...)
        s_yx = sinkhorn_divergence(Y, X; ε = 0.5, kw...)
        s_xz = sinkhorn_divergence(X, Z; ε = 0.5, kw...)
        @test abs(s_xx) < 1e-6                      # exactly zero, not just small
        @test s_xy ≈ s_yx rtol = 1e-5               # symmetric
        @test s_xy > s_xz                           # separates
        @test s_xz > -1e-6                          # non-negative
        # `epsilon` is accepted as an alias for `ε`
        @test sinkhorn_divergence(X, Y; epsilon = 0.5, kw...) ≈ s_xy rtol = 1e-8
    end

    @testset "sinkhorn_divergence approaches W2^2 for small eps" begin
        # two well-separated clouds: the divergence should land near the
        # closed-form Gaussian value
        smu, snu, ref = gaussian_problem(; shift = 3.0, sigma = 0.5, seed = 3)
        A = smu(400); B = snu(400)
        s = sinkhorn_divergence(A, B; ε = 0.05, n_iter = 40_000, tol = 1e-9)
        @test isapprox(s, ref.w2; rtol = 0.25)
    end

    @testset "energy_distance" begin
        @test energy_distance(X, X) ≈ 0 atol = 1e-5
        @test energy_distance(X, Y) > energy_distance(X, Z)
        @test energy_distance(X, Y) ≈ energy_distance(Y, X) rtol = 1e-6
        @test energy_distance(X, Z) > -1e-5
    end

    @testset "mmd" begin
        @test abs(mmd(X, X)) < 0.05
        @test mmd(X, Y) > mmd(X, Z)
        @test mmd(X, Y) ≈ mmd(Y, X) rtol = 1e-5
        @test mmd(X, Y; sigma = 1.0) > 0
        @test_throws ArgumentError mmd(X[:, 1:1], Y)
    end

    @testset "moment_error" begin
        e_same = moment_error(X, X)
        @test e_same.mean < 1e-6 && e_same.cov < 1e-6
        e_diff = moment_error(X, Y)
        @test e_diff.mean > 4                       # the means differ by 5
    end

    @testset "transport_error" begin
        @test transport_error(X, X) ≈ 0 atol = 1e-6
        shifted = X .+ Float32[3.0; 4.0]            # every point moves by 5
        @test transport_error(X, shifted) ≈ 5 rtol = 1e-4
        @test_throws DimensionMismatch transport_error(X, Y[:, 1:10])
    end
end
