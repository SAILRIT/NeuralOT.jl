@testset "datasets" begin
    samplers = Dict(
        "gaussian" => (gaussian_sampler(3; sigma = 0.5, rng = MersenneTwister(0)), 3),
        "two_moons" => (two_moons(; rng = MersenneTwister(0)), 2),
        "eight_gaussians" => (eight_gaussians(; rng = MersenneTwister(0)), 2),
        "checkerboard" => (checkerboard(; rng = MersenneTwister(0)), 2),
        "swiss_roll" => (swiss_roll(; rng = MersenneTwister(0)), 2),
        "circles" => (circles(; rng = MersenneTwister(0)), 2),
        "uniform_box" => (uniform_box(4; rng = MersenneTwister(0)), 4),
    )

    @testset "shape, type and finiteness: $name" for (name, (s, d)) in samplers
        X = s(37)
        @test X isa Matrix{Float32}
        @test size(X) == (d, 37)
        @test all(isfinite, X)
        @test size(s(1)) == (d, 1)
    end

    @testset "gaussian_sampler moments" begin
        s = gaussian_sampler(2; mean = [1.0, -2.0], sigma = 0.5, rng = MersenneTwister(1))
        X = s(200_000)
        @test vec(mean(X; dims = 2)) ≈ [1.0, -2.0] atol = 0.02
        @test cov(Float64.(X'); dims = 1) ≈ 0.25 * I(2) atol = 0.02
    end

    @testset "gaussian_sampler with a full covariance" begin
        S = [2.0 0.5; 0.5 1.0]
        s = gaussian_sampler(2; cov = S, rng = MersenneTwister(2))
        X = s(200_000)
        @test cov(Float64.(X'); dims = 1) ≈ S atol = 0.05
        @test_throws DimensionMismatch gaussian_sampler(2; cov = Matrix(1.0I, 3, 3))
        @test_throws DimensionMismatch gaussian_sampler(2; mean = [1.0])
    end

    @testset "reproducibility with an explicit RNG" begin
        a = two_moons(; rng = MersenneTwister(5))(50)
        b = two_moons(; rng = MersenneTwister(5))(50)
        @test a == b
    end

    @testset "support constraints" begin
        X = uniform_box(3; lo = -2.0, hi = 5.0, rng = MersenneTwister(0))(5_000)
        @test minimum(X) >= -2 && maximum(X) <= 5
        @test_throws ArgumentError uniform_box(2; lo = 1.0, hi = 0.0)

        C = circles(; radius = 2.0, noise = 0.0, rng = MersenneTwister(0))(1_000)
        @test all(abs.(sqrt.(vec(sum(abs2, C; dims = 1))) .- 2) .< 1e-4)

        E = eight_gaussians(; radius = 4.0, sigma = 0.05, rng = MersenneTwister(0))(2_000)
        radii = sqrt.(vec(sum(abs2, E; dims = 1)))
        @test all(abs.(radii .- 4) .< 0.5)
        @test_throws ArgumentError eight_gaussians(; n_modes = 0)

        B = checkerboard(; scale = 4.0, rng = MersenneTwister(0))(2_000)
        @test maximum(abs.(B)) <= 4
    end
end
