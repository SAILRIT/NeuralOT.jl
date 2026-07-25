# End-to-end checks: every solver on a problem whose answer is known exactly.

@testset "integration" begin
    @testset "all three solvers reduce the divergence to the target" begin
        smu, snu, ref = gaussian_problem(; shift = 3.0, sigma = 0.5, seed = 30)
        X = smu(600)
        Y = snu(600)
        baseline = sinkhorn_divergence(X, Y; ε = 0.5, n_iter = 20_000)

        dual = solve_dual(smu, snu; dim = 2, steps = scale_steps(2000), batch = 128,
                          hidden = [64, 64], lr = 2e-3, seed = 31)
        w2 = solve_w2(smu, snu; dim = 2, widths = [64, 64, 1], steps = scale_steps(1200),
                      inner_steps = 8, batch = 128, lr = 1e-3, seed = 32)
        flow = flow_match(smu, snu; dim = 2, hidden = [64, 64], steps = scale_steps(2000),
                          batch = 128, lr = 2e-3, seed = 33)

        for (name, T) in (("dual", monge_map(dual, X)),
                          ("w2", monge_map(w2, X)),
                          ("flow", monge_map(flow, X; n_flow_steps = 60)))
            s = sinkhorn_divergence(T, Y; ε = 0.5, n_iter = 20_000)
            @test all(isfinite, T)
            @test s < 0.5 * baseline
            e = moment_error(T, Y)
            @test e.mean < 0.6
        end
    end

    @testset "anisotropic target: the map must be more than a translation" begin
        smu, snu, ref = anisotropic_problem(; seed = 40)
        res = solve_w2(smu, snu; dim = 2, widths = [64, 64, 1],
                       steps = scale_steps(1500), inner_steps = 8, batch = 128,
                       lr = 1e-3, seed = 41)
        X = smu(1_000)
        Y = snu(1_000)
        T = monge_map(res, X)
        # a pure shift cannot match the target covariance, so this checks that
        # the solver learned the linear part too
        shift_only = X .+ Float32.(vec(mean(Y; dims = 2)) .- vec(mean(X; dims = 2)))
        @test moment_error(T, Y).cov < moment_error(shift_only, Y).cov
        @test sinkhorn_divergence(T, Y; ε = 0.5, n_iter = 20_000) <
              sinkhorn_divergence(X, Y; ε = 0.5, n_iter = 20_000)
    end

    @testset "higher dimensions run end to end" begin
        d = 8
        smu = gaussian_sampler(d; sigma = 1.0, rng = MersenneTwister(50))
        snu = gaussian_sampler(d; mean = fill(1.5, d), sigma = 1.0,
                               rng = MersenneTwister(51))
        res = solve_dual(smu, snu; dim = d, steps = scale_steps(600), batch = 128,
                         hidden = [64, 64], lr = 2e-3, seed = 52)
        X = smu(400)
        T = monge_map(res, X)
        @test size(T) == (d, 400)
        @test all(isfinite, T)
        @test mean(T) > mean(X)

        resw = solve_w2(smu, snu; dim = d, widths = [32, 32, 1],
                        steps = scale_steps(300), inner_steps = 5, batch = 128,
                        lr = 2e-3, seed = 53)
        Tw = monge_map(resw, X)
        @test size(Tw) == (d, 400)
        @test all(isfinite, Tw)
    end

    @testset "a non-Gaussian target is reachable" begin
        smu = gaussian_sampler(2; sigma = 0.5, rng = MersenneTwister(60))
        snu = eight_gaussians(; radius = 3.0, sigma = 0.3, rng = MersenneTwister(61))
        res = flow_match(smu, snu; dim = 2, hidden = [64, 64],
                         steps = scale_steps(3000), batch = 128, lr = 2e-3, seed = 62)
        X = smu(800)
        Y = snu(800)
        T = monge_map(res, X; n_flow_steps = 80)
        @test sinkhorn_divergence(T, Y; ε = 0.5, n_iter = 20_000) <
              0.5 * sinkhorn_divergence(X, Y; ε = 0.5, n_iter = 20_000)
        # the pushforward should spread out onto the ring rather than collapse
        @test mean(sqrt.(vec(sum(abs2, T; dims = 1)))) > 1.5
    end
end
