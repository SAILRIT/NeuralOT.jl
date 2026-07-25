@testset "solve_dual" begin
    smu, snu, ref = gaussian_problem(; shift = 3.0, sigma = 0.5, seed = 10)

    @testset "argument validation" begin
        @test_throws ArgumentError solve_dual(smu, snu; dim = 2, steps = 0)
        @test_throws ArgumentError solve_dual(smu, snu; dim = 2, batch = 1)
        @test_throws ArgumentError solve_dual(smu, snu; dim = 2, lr = 0.0)
        @test_throws ArgumentError solve_dual(smu, snu; dim = 2, ε = 0.0)
        @test_throws ArgumentError solve_dual(smu, snu; dim = 0)
        @test_throws ArgumentError solve_dual(smu, snu; dim = 2, formulation = :nope)
        # a sampler with the wrong dimension must fail loudly, not silently
        @test_throws DimensionMismatch solve_dual(smu, snu; dim = 3, steps = 2)
    end

    @testset "training runs and improves the dual value" begin
        res = solve_dual(smu, snu; dim = 2, steps = scale_steps(1200), batch = 128,
                         hidden = [64, 64], lr = 2e-3, log_every = 100, seed = 42,
                         eval_batch = 256)
        @test res.method === :dual
        @test res.models.u isa DualPotentialNet
        @test length(res.losses) > 2
        @test length(res.logged_steps) == length(res.losses)
        @test length(res.eval_losses) == length(res.losses)
        @test res.elapsed > 0
        @test all(isfinite, res.losses)
        # the dual is a supremum written as a minimisation of its negative
        @test mean(res.losses[max(1, end - 2):end]) < res.losses[1]
        @test res.config.epsilon == 0.1
        @test res.config.ε == 0.1                      # v0.1 key name kept
    end

    @testset "the dual value approaches the regularised transport cost" begin
        res = solve_dual(smu, snu; dim = 2, steps = scale_steps(2000), batch = 128,
                         hidden = [64, 64], lr = 2e-3, seed = 1)
        value = NeuralOT.dual_value(res)
        # reference: entropic OT between large samples, same epsilon
        C = cost_matrix(SqEuclideanCost(), smu(600), snu(600))
        reference = sinkhorn_value(C; epsilon = 0.1, n_iter = 20_000, tol = 1e-9)
        @test isapprox(value, reference; rtol = 0.35)
        @test isapprox(value, ref.w2; rtol = 0.4)
        @test_throws ArgumentError NeuralOT.dual_value(
            NeuralOTResult((f = 1,), [1.0], :flow, (;)))
    end

    @testset "both formulations run" begin
        for form in (:logsumexp, :exp)
            res = solve_dual(smu, snu; dim = 2, steps = scale_steps(200), batch = 64,
                             hidden = [32, 32], formulation = form, seed = 2)
            @test res.config.formulation === form
            @test all(isfinite, res.losses)
        end
    end

    @testset "callbacks and early stopping" begin
        seen = Int[]
        res = solve_dual(smu, snu; dim = 2, steps = 500, batch = 64, hidden = [16],
                         log_every = 50, seed = 3,
                         callback = (step, loss, models) -> (push!(seen, step); true))
        @test !isempty(seen)
        @test issorted(seen)

        stopped = solve_dual(smu, snu; dim = 2, steps = 5_000, batch = 64,
                             hidden = [16], log_every = 50, seed = 3,
                             callback = (step, loss, models) -> step < 150)
        @test last(stopped.logged_steps) <= 200        # stopped well before 5000
    end

    @testset "reproducibility from a seed" begin
        kw = (dim = 2, steps = 60, batch = 32, hidden = [16], seed = 7, log_every = 20)
        a = solve_dual(gaussian_problem(; seed = 20)[1], gaussian_problem(; seed = 20)[2]; kw...)
        b = solve_dual(gaussian_problem(; seed = 20)[1], gaussian_problem(; seed = 20)[2]; kw...)
        @test a.losses ≈ b.losses
    end

    @testset "log_every = 0 disables logging" begin
        res = solve_dual(smu, snu; dim = 2, steps = 20, batch = 32, hidden = [8],
                         log_every = 0, seed = 4)
        @test isempty(res.losses)
    end
end
