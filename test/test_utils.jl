@testset "utils" begin
    @testset "_f32 avoids copying" begin
        A = randn(Float32, 3, 4)
        @test NeuralOT._f32(A) === A
        B = randn(Float64, 3, 4)
        @test eltype(NeuralOT._f32(B)) === Float32
        @test NeuralOT._f32(B) ≈ Float32.(B)
    end

    @testset "_logsumexp is stable and shape-preserving" begin
        A = randn(Float32, 5, 7)
        lse = NeuralOT._logsumexp(A; dims = 2)
        @test size(lse) == (5, 1)
        @test lse ≈ log.(sum(exp.(A); dims = 2)) rtol = 1e-4
        # extreme values would overflow a naive implementation
        big = Float32[1000.0 1000.0; -1000.0 -1000.0]
        out = NeuralOT._logsumexp(big; dims = 2)
        @test all(isfinite, out)
        @test out[1] ≈ 1000 + log(2) rtol = 1e-5
        @test NeuralOT._logsumexp(Float32[1.0, 2.0, 3.0]) ≈ log(sum(exp.([1.0, 2.0, 3.0]))) rtol = 1e-5
    end

    @testset "_softplus_inv inverts softplus" begin
        for y in (0.1, 1.0, 5.0)
            @test NeuralOT.softplus(NeuralOT._softplus_inv(y)) ≈ y rtol = 1e-6
        end
    end

    @testset "_onelike has no mutation and right shape" begin
        A = randn(Float32, 3, 5)
        @test NeuralOT._onelike(A) == ones(Float32, 3, 5)
        @test eltype(NeuralOT._onelike(A)) === Float32
    end

    @testset "_check_batch rejects malformed samplers" begin
        @test NeuralOT._check_batch(randn(Float32, 3, 8), 3, "s") isa AbstractMatrix
        @test_throws DimensionMismatch NeuralOT._check_batch(randn(Float32, 2, 8), 3, "s")
        @test_throws ArgumentError NeuralOT._check_batch(randn(Float32, 3, 0), 3, "s")
        @test_throws ArgumentError NeuralOT._check_batch(randn(Float32, 3), 3, "s")
    end

    @testset "_rng is reproducible and independent" begin
        a = randn(NeuralOT._rng(42), 5)
        b = randn(NeuralOT._rng(42), 5)
        @test a == b
        @test NeuralOT._rng(nothing) isa Random.AbstractRNG
    end

    @testset "_sample_categorical respects the weights" begin
        rng = MersenneTwister(0)
        P = [1.0 0.0 0.0; 0.0 0.0 1.0]
        idx = NeuralOT._sample_categorical(rng, P)
        @test idx == [1, 3]
        # a uniform row should hit every column eventually
        P2 = fill(0.25, 1, 4)
        hits = Set(NeuralOT._sample_categorical(rng, P2)[1] for _ in 1:200)
        @test length(hits) > 1
    end
end
