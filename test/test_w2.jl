@testset "solve_w2" begin
    smu, snu, ref = gaussian_problem(; shift = 3.0, sigma = 0.5, seed = 11)

    @testset "argument validation" begin
        @test_throws ArgumentError solve_w2(smu, snu; dim = 2, widths = [16, 16, 4])
        @test_throws ArgumentError solve_w2(smu, snu; dim = 2, inner_steps = 0)
        @test_throws ArgumentError solve_w2(smu, snu; dim = 2, cycle = -1.0)
        @test_throws ArgumentError solve_w2(smu, snu; dim = 2, steps = 0)
    end

    @testset "training runs without nested AD" begin
        # v0.1 wrapped this in a try/catch and marked it @test_broken because the
        # Zygote-over-Zygote path could fail. With the analytic ICNN gradient it
        # is an ordinary single-level reverse-mode problem.
        res = solve_w2(smu, snu; dim = 2, widths = [32, 32, 1], steps = scale_steps(500),
                       inner_steps = 5, batch = 128, lr = 2e-3, log_every = 50, seed = 123)
        @test res.method === :w2_icnn
        @test res.models.f isa ICNN && res.models.g isa ICNN
        @test all(isfinite, res.losses)
        @test !isempty(res.losses)
        @test res.elapsed > 0

        X = smu(512)
        T = monge_map(res, X)
        @test size(T) == size(X)
        @test all(isfinite, T)
        # the map must move mass towards the target
        @test mean(T[1, :]) > mean(X[1, :])
        @test abs(mean(T[1, :]) - 3.0) < abs(mean(X[1, :]) - 3.0)
    end

    @testset "the learned map approaches the exact Brenier map" begin
        res = solve_w2(smu, snu; dim = 2, widths = [64, 64, 1], steps = scale_steps(1200),
                       inner_steps = 8, batch = 128, lr = 1e-3, seed = 5)
        X = smu(1_000)
        Y = snu(1_000)
        T = monge_map(res, X)
        @test transport_error(T, ref.map(X)) < 1.5           # exact map shifts by 3
        s_after = sinkhorn_divergence(T, Y; ε = 0.5, n_iter = 20_000)
        s_before = sinkhorn_divergence(X, Y; ε = 0.5, n_iter = 20_000)
        @test s_after < 0.4 * s_before
        # the W2 estimate should be in the right ballpark
        @test isapprox(NeuralOT.w2_estimate(res, X, Y), ref.w2; rtol = 0.6)
    end

    @testset "grad f inverts grad g" begin
        res = solve_w2(smu, snu; dim = 2, widths = [64, 64, 1], steps = scale_steps(1200),
                       inner_steps = 8, batch = 128, lr = 1e-3, seed = 6)
        Y = snu(500)
        back = inverse_map(res, Y)
        @test size(back) == size(Y)
        # inverse map must move mass back towards the source
        @test mean(back[1, :]) < mean(Y[1, :])
        @test abs(mean(back[1, :])) < abs(mean(Y[1, :]))
    end

    @testset "the potentials stay convex during training" begin
        res = solve_w2(smu, snu; dim = 2, widths = [32, 32, 1], steps = scale_steps(300),
                       inner_steps = 5, batch = 64, seed = 7)
        for pot in (res.models.f, res.models.g)
            x1 = randn(MersenneTwister(1), Float32, 2, 300)
            x2 = randn(MersenneTwister(2), Float32, 2, 300)
            lhs = vec(pot(0.5f0 .* (x1 .+ x2)))
            rhs = 0.5f0 .* (vec(pot(x1)) .+ vec(pot(x2)))
            @test all(lhs .<= rhs .+ 1f-3)
        end
    end

    @testset "cycle regularisation runs" begin
        res = solve_w2(smu, snu; dim = 2, widths = [16, 16, 1], steps = scale_steps(100),
                       inner_steps = 3, batch = 64, cycle = 0.1, seed = 8)
        @test res.config.cycle == 0.1
        @test all(isfinite, res.losses)
    end

    @testset "callback receives both potentials" begin
        got = Ref(false)
        solve_w2(smu, snu; dim = 2, widths = [8, 1], steps = 40, inner_steps = 2,
                 batch = 32, log_every = 20, seed = 9,
                 callback = (s, l, m) -> (got[] = haskey(m, :f) && haskey(m, :g); true))
        @test got[]
    end
end
