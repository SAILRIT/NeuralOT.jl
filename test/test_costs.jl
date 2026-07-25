@testset "costs" begin
    X = randn(Float32, 3, 6)
    Y = randn(Float32, 3, 8)

    @testset "squared Euclidean matches the definition" begin
        C = cost_matrix(SqEuclideanCost(), X, Y)
        @test size(C) == (6, 8)
        @test eltype(C) === Float32
        for i in 1:6, j in 1:8
            @test C[i, j] ≈ sum(abs2, X[:, i] .- Y[:, j]) rtol = 1e-4
        end
        # the expansion can go slightly negative for identical points
        @test all(cost_matrix(SqEuclideanCost(), X, X) .>= 0)
        @test all(abs.(diag(cost_matrix(SqEuclideanCost(), X, X))) .< 1e-4)
    end

    @testset "Euclidean is the square root" begin
        C = cost_matrix(EuclideanCost(), X, Y)
        @test C ≈ sqrt.(cost_matrix(SqEuclideanCost(), X, Y)) rtol = 1e-5
    end

    @testset "GenericCost handles arbitrary functions" begin
        c = GenericCost((x, y) -> sum(abs, x .- y))
        C = cost_matrix(c, X, Y)
        @test size(C) == (6, 8)
        for i in 1:6, j in 1:8
            @test C[i, j] ≈ sum(abs, X[:, i] .- Y[:, j]) rtol = 1e-4
        end
        # a generic wrapper of the squared cost must agree with the fast path
        c2 = GenericCost((x, y) -> sum(abs2, x .- y))
        @test cost_matrix(c2, X, Y) ≈ cost_matrix(SqEuclideanCost(), X, Y) rtol = 1e-4
    end

    @testset "v0.1 compatibility: bare Distances functions" begin
        @test cost_matrix(Distances.sqeuclidean, X, Y) ≈
              cost_matrix(SqEuclideanCost(), X, Y) rtol = 1e-5
        @test NeuralOT.is_squared_euclidean(Distances.sqeuclidean)
        @test NeuralOT.is_squared_euclidean(SqEuclideanCost())
        @test !NeuralOT.is_squared_euclidean(EuclideanCost())
    end

    @testset "dimension mismatch is caught" begin
        @test_throws DimensionMismatch cost_matrix(SqEuclideanCost(), X, randn(Float32, 2, 4))
    end
end
