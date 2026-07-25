@testset "sinkhorn" begin
    rng = MersenneTwister(0)
    X = randn(rng, Float32, 2, 40)
    Y = randn(rng, Float32, 2, 40) .+ Float32[3.0; 0.0]
    C = cost_matrix(SqEuclideanCost(), X, Y)

    @testset "potentials converge and reproduce the marginals" begin
        sol = sinkhorn_potentials(C; epsilon = 0.5, n_iter = 5_000, tol = 1e-9)
        @test sol.converged
        @test sol.iterations < 5_000
        @test length(sol.f) == 40 && length(sol.g) == 40
        @test all(isfinite, sol.f) && all(isfinite, sol.g)

        P = sinkhorn_plan(C; epsilon = 0.5, n_iter = 5_000, tol = 1e-9)
        @test size(P) == (40, 40)
        @test all(P .>= 0)
        @test sum(P) ≈ 1 rtol = 1e-5
        @test maximum(abs.(vec(sum(P; dims = 2)) .- 1 / 40)) < 1e-6
        @test maximum(abs.(vec(sum(P; dims = 1)) .- 1 / 40)) < 1e-6
    end

    @testset "non-uniform marginals" begin
        a = rand(rng, 40); a ./= sum(a)
        b = rand(rng, 40); b ./= sum(b)
        P = sinkhorn_plan(C, a, b; epsilon = 0.5, n_iter = 5_000, tol = 1e-9)
        @test maximum(abs.(vec(sum(P; dims = 2)) .- a)) < 1e-5
        @test maximum(abs.(vec(sum(P; dims = 1)) .- b)) < 1e-5
    end

    @testset "the value identity <f,a> + <g,b> = <P,C> + eps KL" begin
        eps = 0.5
        sol = sinkhorn_potentials(C; epsilon = eps, n_iter = 20_000, tol = 1e-11)
        @test sol.converged
        P = sinkhorn_plan(C; epsilon = eps, n_iter = 20_000, tol = 1e-11)
        val = sinkhorn_value(C; epsilon = eps, n_iter = 20_000, tol = 1e-11)
        ab = fill(1 / 40 * 1 / 40, 40, 40)
        kl = sum(P .* log.(max.(P ./ ab, 1e-30))) - sum(P) + 1
        @test val ≈ sinkhorn_cost(C; epsilon = eps, n_iter = 20_000, tol = 1e-11) + eps * kl rtol = 1e-3
    end

    @testset "converges to the exact transport cost as eps -> 0" begin
        # For equal-size uniform empirical measures the exact optimal cost is
        # attained by a permutation; the entropic cost must approach it.
        c_big = sinkhorn_cost(C; epsilon = 1.0, n_iter = 20_000, tol = 1e-10)
        c_small = sinkhorn_cost(C; epsilon = 0.05, n_iter = 40_000, tol = 1e-10)
        @test c_small < c_big
        # a permutation-based upper bound the entropic solution must not beat
        best_diag = sum(C[i, i] for i in 1:40) / 40
        @test c_small <= best_diag + 1e-3
    end

    @testset "argument validation" begin
        @test_throws ArgumentError sinkhorn_potentials(C; epsilon = 0.0)
        @test_throws ArgumentError sinkhorn_potentials(C; epsilon = -1.0)
        @test_throws ArgumentError sinkhorn_potentials(C; n_iter = 0)
        @test_throws DimensionMismatch sinkhorn_potentials(C, ones(3) ./ 3)
        @test_throws ArgumentError sinkhorn_potentials(C, fill(-1.0, 40))
    end

    @testset "symmetric problems give symmetric potentials" begin
        Cs = cost_matrix(SqEuclideanCost(), X, X)
        sol = sinkhorn_potentials(Cs; epsilon = 0.5, n_iter = 20_000, tol = 1e-11)
        @test sol.f ≈ sol.g rtol = 1e-3
    end
end
