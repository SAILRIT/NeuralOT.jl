@testset "potentials" begin
    @testset "DualPotentialNet" begin
        u = DualPotentialNet(4; hidden = [8, 8])
        x = randn(Float32, 4, 10)
        @test size(u(x)) == (1, 10)
        @test size(u(randn(Float32, 4))) == (1,)
        @test all(isfinite, u(x))
        @test_throws ArgumentError DualPotentialNet(0)
        # trainable and differentiable
        gs = Flux.gradient(m -> sum(abs2, m(x)), u)[1]
        @test gs !== nothing
        @test !isempty(Flux.trainable(u))
    end

    @testset "DualPotentialNet is reproducible given an RNG" begin
        a = DualPotentialNet(3; hidden = [6], rng = MersenneTwister(0))
        b = DualPotentialNet(3; hidden = [6], rng = MersenneTwister(0))
        x = randn(Float32, 3, 5)
        @test a(x) ≈ b(x)
    end

    @testset "VelocityNet shapes and time handling" begin
        v = VelocityNet(3; hidden = [16, 16])
        x = randn(Float32, 3, 12)
        t = rand(Float32, 1, 12)
        @test size(v(t, x)) == (3, 12)
        @test size(v(0.3, x)) == (3, 12)
        @test v(0.3, x) ≈ v(fill(0.3f0, 1, 12), x)
        # stacked [t; x] form, as v0.1's raw Chain accepted
        @test v(vcat(t, x)) ≈ v(t, x)
        @test_throws DimensionMismatch v(rand(Float32, 2, 12), x)
        @test_throws DimensionMismatch v(rand(Float32, 1, 5), x)
    end

    @testset "Fourier time features" begin
        v = VelocityNet(2; hidden = [8], n_fourier = 3)
        x = randn(Float32, 2, 6)
        t = rand(Float32, 1, 6)
        @test size(v(t, x)) == (2, 6)
        feats = NeuralOT._time_features(t, 3)
        @test size(feats) == (7, 6)                       # t + 3 sin + 3 cos
        @test feats[1:1, :] == t
        @test all(abs.(feats[2:end, :]) .<= 1 + 1e-5)
        @test NeuralOT._time_features(t, 0) === t
    end

    @testset "_mlp builds the requested shape" begin
        net = NeuralOT._mlp([3, 5, 7, 2])
        @test length(net) == 3
        @test size(net(randn(Float32, 3, 4))) == (2, 4)
        @test_throws ArgumentError NeuralOT._mlp([4])
    end
end
