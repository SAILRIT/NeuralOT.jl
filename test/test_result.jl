@testset "NeuralOTResult" begin
    models = (u = DualPotentialNet(2; hidden = [4]),)
    losses = [3.0, 2.0, 1.0]

    @testset "v0.1 four-argument constructor still works" begin
        r = NeuralOTResult(models, losses, :dual, (ε = 0.1,))
        @test r.method === :dual
        @test r.losses == losses
        @test r.config.ε == 0.1
        @test r.logged_steps == [1, 2, 3]
        @test isempty(r.eval_losses)
        @test isnan(r.elapsed)
    end

    @testset "full constructor and accessors" begin
        r = NeuralOTResult(models, losses, :flow, (a = 1,), [1, 50, 100], [3.1, 2.1, 1.1], 4.2)
        @test r.logged_steps == [1, 50, 100]
        @test r.eval_losses == [3.1, 2.1, 1.1]
        @test r.elapsed == 4.2
        steps, ls = loss_history(r)
        @test steps == [1, 50, 100] && ls == losses
    end

    @testset "converged heuristic" begin
        flat = NeuralOTResult(models, fill(1.0, 20), :dual, (;))
        @test NeuralOT.converged(flat)
        falling = NeuralOTResult(models, collect(range(10.0, 1.0; length = 20)),
                                 :dual, (;))
        @test !NeuralOT.converged(falling)
        @test !NeuralOT.converged(NeuralOTResult(models, [1.0], :dual, (;)))
    end

    @testset "show does not error" begin
        r = NeuralOTResult(models, losses, :dual, (ε = 0.1,))
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        s = String(take!(io))
        @test occursin("NeuralOTResult", s)
        @test occursin("dual", s)
        io2 = IOBuffer()
        show(io2, r)
        @test occursin("dual", String(take!(io2)))
        # an empty history must not break printing
        io3 = IOBuffer()
        show(io3, MIME"text/plain"(), NeuralOTResult(models, Float64[], :flow, (;)))
        @test occursin("none logged", String(take!(io3)))
    end
end
