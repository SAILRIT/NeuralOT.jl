@testset "ICNN" begin
    @testset "construction and shapes" begin
        f = ICNN(3, [16, 16, 1])
        @test NeuralOT.input_dim(f) == 3
        @test NeuralOT.n_layers(f) == 3
        @test length(f.Wz) == 2                       # one fewer than the layers
        x = randn(Float32, 3, 8)
        y = f(x)
        @test size(y) == (1, 8)
        @test all(isfinite, y)
        @test size(f(randn(Float32, 3))) == (1,)
        @test_throws DimensionMismatch f(randn(Float32, 4, 8))
    end

    @testset "argument validation" begin
        @test_throws ArgumentError ICNN(0, [4, 1])
        @test_throws ArgumentError ICNN(2, Int[])
        @test_throws ArgumentError ICNN(2, [4, 0, 1])
        @test_throws ArgumentError ICNN(2, [4, 1]; activation = tanh)   # not convex
    end

    @testset "convexity in the input" begin
        for (dim, widths) in ((2, [16, 16, 1]), (4, [32, 32, 32, 1]), (3, [8, 1]))
            f = ICNN(dim, widths; rng = MersenneTwister(1))
            x1 = 3f0 .* randn(MersenneTwister(2), Float32, dim, 200)
            x2 = 3f0 .* randn(MersenneTwister(3), Float32, dim, 200)
            mid = 0.5f0 .* (x1 .+ x2)
            lhs = vec(f(mid))
            rhs = 0.5f0 .* (vec(f(x1)) .+ vec(f(x2)))
            @test all(lhs .<= rhs .+ 1f-3)
        end
    end

    @testset "convexity survives adversarial parameters" begin
        # convexity must hold for ANY weights, not just at initialisation
        f = ICNN(2, [8, 8, 1]; rng = MersenneTwister(4))
        for W in f.Wz
            W .= 5f0 .* randn(MersenneTwister(5), Float32, size(W)...)
        end
        for W in f.Wx
            W .= 3f0 .* randn(MersenneTwister(6), Float32, size(W)...)
        end
        x1 = randn(MersenneTwister(7), Float32, 2, 300)
        x2 = randn(MersenneTwister(8), Float32, 2, 300)
        lhs = vec(f(0.5f0 .* (x1 .+ x2)))
        rhs = 0.5f0 .* (vec(f(x1)) .+ vec(f(x2)))
        @test all(lhs .<= rhs .+ 1f-3)
    end

    @testset "analytic input gradient matches finite differences" begin
        for (dim, widths) in ((2, [16, 16, 1]), (4, [12, 12, 12, 1]), (3, [1]))
            f = ICNN(dim, widths; rng = MersenneTwister(9))
            x = randn(MersenneTwister(10), Float32, dim, 4)
            g = input_gradient(f, x)
            @test size(g) == size(x)
            @test all(isfinite, g)
            for col in 1:size(x, 2)
                fd = finite_difference_gradient(f, x[:, col])
                @test g[:, col] ≈ fd rtol = 1e-2 atol = 1e-3
            end
        end
    end

    @testset "analytic gradient matches reverse-mode AD" begin
        f = ICNN(3, [16, 16, 1]; rng = MersenneTwister(11))
        x = randn(MersenneTwister(12), Float32, 3, 5)
        auto = Zygote.gradient(z -> sum(f(z)), x)[1]
        @test input_gradient(f, x) ≈ auto rtol = 1e-4
        @test grad_x(f, x) ≈ auto rtol = 1e-4          # v0.1 alias
    end

    @testset "gradient is differentiable w.r.t. parameters (single-level AD)" begin
        # This is the property that lets solve_w2 avoid nested AD.
        f = ICNN(2, [16, 16, 1]; rng = MersenneTwister(13))
        x = randn(MersenneTwister(14), Float32, 2, 16)
        gs = Flux.gradient(m -> sum(abs2, input_gradient(m, x)), f)[1]
        @test gs !== nothing
        @test all(w === nothing || all(isfinite, w) for w in gs.Wx)
        @test all(w === nothing || all(isfinite, w) for w in gs.Wz)
        @test sum(sum(abs, w) for w in gs.Wx) > 0
    end

    @testset "input gradient is monotone (gradient of a convex function)" begin
        f = ICNN(3, [16, 16, 1]; rng = MersenneTwister(15))
        x1 = randn(MersenneTwister(16), Float32, 3, 200)
        x2 = randn(MersenneTwister(17), Float32, 3, 200)
        g1 = input_gradient(f, x1)
        g2 = input_gradient(f, x2)
        # <grad f(x) - grad f(y), x - y> >= 0 characterises convexity
        inner = vec(sum((g1 .- g2) .* (x1 .- x2); dims = 1))
        @test all(inner .>= -1f-3)
    end

    @testset "strong convexity term" begin
        f = ICNN(2, [8, 1]; quadratic = 2.0)
        @test NeuralOT.beta(f) ≈ 2.0 rtol = 1e-5
        f0 = ICNN(2, [8, 1]; quadratic = 0.0)
        @test NeuralOT.beta(f0) < 1e-6
        # with a large quadratic weight the map is close to a scaled identity
        fq = ICNN(2, [8, 1]; quadratic = 50.0, init_scale = 1e-4)
        x = randn(MersenneTwister(18), Float32, 2, 32)
        @test input_gradient(fq, x) ./ 50 ≈ x rtol = 0.1
    end

    @testset "custom activation with supplied derivative" begin
        f = ICNN(2, [8, 1]; activation = relu, dactivation = NeuralOT._drelu)
        x = randn(MersenneTwister(19), Float32, 2, 6)
        auto = Zygote.gradient(z -> sum(f(z)), x)[1]
        @test input_gradient(f, x) ≈ auto rtol = 1e-4
    end

    @testset "scalar-output requirement" begin
        f = ICNN(2, [8, 4])
        @test_throws ArgumentError input_gradient(f, randn(Float32, 2, 3))
    end

    @testset "generic fallback for non-ICNN callables" begin
        u = DualPotentialNet(2; hidden = [8, 8])
        x = randn(MersenneTwister(20), Float32, 2, 5)
        @test input_gradient(u, x) ≈ Zygote.gradient(z -> sum(u(z)), x)[1] rtol = 1e-5
    end
end
