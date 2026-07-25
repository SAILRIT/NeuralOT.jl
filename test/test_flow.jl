@testset "flow_match" begin
    smu, snu, ref = gaussian_problem(; shift = 3.0, sigma = 0.5, seed = 12)

    @testset "argument validation" begin
        @test_throws ArgumentError flow_match(smu, snu; dim = 2, coupling = :nope)
        @test_throws ArgumentError flow_match(smu, snu; dim = 2, sigma = -1.0)
        @test_throws ArgumentError flow_match(smu, snu; dim = 2, steps = 0)
    end

    @testset "training decreases the regression loss" begin
        res = flow_match(smu, snu; dim = 2, hidden = [64, 64], steps = scale_steps(1500),
                         batch = 128, lr = 2e-3, log_every = 100, seed = 7,
                         eval_batch = 256)
        @test res.method === :flow
        @test res.models.vfield isa VelocityNet
        @test all(isfinite, res.losses)
        @test mean(res.losses[max(1, end - 2):end]) < res.losses[1]
        @test length(res.eval_losses) == length(res.losses)
    end

    @testset "the flow transports mu onto nu" begin
        res = flow_match(smu, snu; dim = 2, hidden = [64, 64], steps = scale_steps(2000),
                         batch = 128, lr = 2e-3, seed = 8)
        X = smu(1_000)
        Y = snu(1_000)
        T = monge_map(res, X; n_flow_steps = 60)
        @test size(T) == size(X)
        @test all(isfinite, T)
        s_after = sinkhorn_divergence(T, Y; ε = 0.5, n_iter = 20_000)
        s_before = sinkhorn_divergence(X, Y; ε = 0.5, n_iter = 20_000)
        @test s_after < 0.2 * s_before
        @test abs(mean(T[1, :]) - 3.0) < 0.4
    end

    @testset "integrators agree once the step count is adequate" begin
        res = flow_match(smu, snu; dim = 2, hidden = [64, 64], steps = scale_steps(1500),
                         batch = 128, lr = 2e-3, seed = 9)
        X = smu(256)
        e = integrate_flow(res.models.vfield, X; n_steps = 200, solver = :euler)
        h = integrate_flow(res.models.vfield, X; n_steps = 200, solver = :heun)
        r = integrate_flow(res.models.vfield, X; n_steps = 200, solver = :rk4)
        @test transport_error(h, r) < 0.05
        @test transport_error(e, r) < 0.2
        @test_throws ArgumentError integrate_flow(res.models.vfield, X; solver = :bogus)
        @test_throws ArgumentError integrate_flow(res.models.vfield, X; n_steps = 0)
    end

    @testset "reverse integration inverts the flow" begin
        res = flow_match(smu, snu; dim = 2, hidden = [64, 64], steps = scale_steps(2000),
                         batch = 128, lr = 2e-3, seed = 10)
        X = smu(256)
        T = monge_map(res, X; n_flow_steps = 100)
        back = inverse_map(res, T; n_flow_steps = 100)
        @test transport_error(back, X) < 0.3
    end

    @testset "integrate_flow accepts a bare Chain (v0.1 vector fields)" begin
        chain = Flux.Chain(Flux.Dense(3, 8, tanh), Flux.Dense(8, 2))
        X = randn(Float32, 2, 5)
        out = integrate_flow(chain, X; n_steps = 5, solver = :euler)
        @test size(out) == size(X)
        @test all(isfinite, out)
    end

    @testset "OT coupling runs and stays sane" begin
        res = flow_match(smu, snu; dim = 2, hidden = [32, 32], steps = scale_steps(400),
                         batch = 64, coupling = :ot, coupling_epsilon = 0.5,
                         coupling_iter = 200, lr = 2e-3, seed = 11)
        @test res.config.coupling === :ot
        @test all(isfinite, res.losses)
        T = monge_map(res, smu(200); n_flow_steps = 50)
        @test all(isfinite, T)
    end

    @testset "noisy probability paths" begin
        res = flow_match(smu, snu; dim = 2, hidden = [32, 32], steps = scale_steps(200),
                         batch = 64, sigma = 0.05, seed = 12)
        @test res.config.sigma == 0.05
        @test all(isfinite, res.losses)
    end

    @testset "rectify straightens an existing flow" begin
        base = flow_match(smu, snu; dim = 2, hidden = [64, 64], steps = scale_steps(1500),
                          batch = 128, lr = 2e-3, seed = 13)
        ref_flow = rectify(base, smu; dim = 2, hidden = [64, 64],
                           steps = scale_steps(1500), batch = 128, lr = 2e-3,
                           n_flow_steps = 50, seed = 14)
        @test ref_flow.method === :flow
        @test ref_flow.config.coupling === :rectified
        X = smu(500)
        fine = monge_map(base, X; n_flow_steps = 100)
        # a rectified flow should need far fewer integration steps: compare the
        # 4-step approximation of each against the well-resolved original.
        coarse_base = monge_map(base, X; n_flow_steps = 4, solver = :euler)
        coarse_rect = monge_map(ref_flow, X; n_flow_steps = 4, solver = :euler)
        @test transport_error(coarse_rect, fine) <= transport_error(coarse_base, fine) + 0.35
        @test_throws ArgumentError rectify(
            NeuralOTResult((u = 1,), [1.0], :dual, (;)), smu; dim = 2)
    end
end
