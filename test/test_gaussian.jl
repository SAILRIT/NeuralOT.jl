@testset "closed-form Gaussian OT" begin
    @testset "W2 between translated standard Gaussians" begin
        d = 3
        S = Matrix(1.0I, d, d)
        @test w2_gaussian(zeros(d), S, [3.0, 0.0, 0.0], S) ≈ 9.0 rtol = 1e-8
        @test w2_gaussian(zeros(d), S, zeros(d), S) ≈ 0.0 atol = 1e-8
    end

    @testset "W2 between centred Gaussians with different scales" begin
        # for commuting covariances W2^2 = sum (sqrt(s0) - sqrt(s1))^2
        S0 = Diagonal([4.0, 1.0]) |> Matrix
        S1 = Diagonal([1.0, 9.0]) |> Matrix
        expected = (2 - 1)^2 + (1 - 3)^2
        @test w2_gaussian(zeros(2), S0, zeros(2), S1) ≈ expected rtol = 1e-8
    end

    @testset "symmetry and non-negativity" begin
        rng = MersenneTwister(0)
        for _ in 1:5
            A = randn(rng, 3, 3); S0 = A * A' + I
            B = randn(rng, 3, 3); S1 = B * B' + I
            m0 = randn(rng, 3); m1 = randn(rng, 3)
            w = w2_gaussian(m0, S0, m1, S1)
            @test w >= 0
            @test w ≈ w2_gaussian(m1, S1, m0, S0) rtol = 1e-8
        end
    end

    @testset "Brenier map pushes the moments correctly" begin
        rng = MersenneTwister(1)
        m0 = [0.5, -1.0]; S0 = [1.0 0.2; 0.2 0.5]
        m1 = [2.0, 1.0];  S1 = [0.8 -0.1; -0.1 2.0]
        ref = gaussian_ot(m0, S0, m1, S1)
        A, b = ref.A, ref.b
        @test A ≈ A' rtol = 1e-8
        @test minimum(eigvals(Symmetric(A))) > 0        # gradient of a convex function
        @test A * S0 * A' ≈ S1 rtol = 1e-6              # pushes the covariance exactly
        @test A * m0 + b ≈ m1 rtol = 1e-8

        smu = gaussian_sampler(2; mean = m0, cov = S0, rng = MersenneTwister(2))
        X = smu(40_000)
        T = ref.map(X)
        @test vec(mean(T; dims = 2)) ≈ m1 atol = 0.05
        @test cov(Float64.(T'); dims = 1) ≈ S1 atol = 0.1
        # E||x - T(x)||^2 equals W2^2
        @test mean(sum(abs2, X .- T; dims = 1)) ≈ ref.w2 rtol = 0.05
    end

    @testset "map output type and shape" begin
        ref = gaussian_ot(zeros(2), Matrix(1.0I, 2, 2), [1.0, 1.0], Matrix(1.0I, 2, 2))
        out = ref.map(randn(Float32, 2, 7))
        @test size(out) == (2, 7)
        @test eltype(out) === Float32
        @test ref.w2 ≈ 2.0 rtol = 1e-8
    end

    @testset "dimension checks" begin
        @test_throws DimensionMismatch w2_gaussian(zeros(2), Matrix(1.0I, 2, 2),
                                                   zeros(3), Matrix(1.0I, 3, 3))
    end
end
