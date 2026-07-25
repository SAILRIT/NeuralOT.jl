@testset "maps" begin
    smu, snu, ref = gaussian_problem(; shift = 3.0, sigma = 0.5, seed = 13)

    @testset "entropic_map uses the textbook constant" begin
        res = solve_dual(smu, snu; dim = 2, steps = scale_steps(2000), batch = 128,
                         hidden = [64, 64], lr = 2e-3, seed = 21)
        X = smu(1_000)
        T = entropic_map(res, X)
        @test size(T) == size(X)
        @test all(isfinite, T)
        err = transport_error(T, ref.map(X))
        @test err < 1.2
        # v0.1 divided by 2 * epsilon instead of 2; that map is far worse
        gu = input_gradient(res.models.u, X)
        wrong = X .- gu ./ (2f0 * 0.1f0)
        @test err < transport_error(wrong, ref.map(X))
        # monge_map dispatches to entropic_map for :dual results
        @test monge_map(res, X) ≈ T
        @test size(monge_map(res, X[:, 1])) == (2,)
    end

    @testset "barycentric_map" begin
        res = solve_dual(smu, snu; dim = 2, steps = scale_steps(2000), batch = 128,
                         hidden = [64, 64], lr = 2e-3, seed = 22)
        X = smu(256)
        Y = snu(2_000)
        T = barycentric_map(res, X, Y)
        @test size(T) == size(X)
        @test all(isfinite, T)
        @test transport_error(T, ref.map(X)) < 1.2
        # every image is a convex combination of the target samples
        @test minimum(T[1, :]) >= minimum(Y[1, :]) - 1e-3
        @test maximum(T[1, :]) <= maximum(Y[1, :]) + 1e-3
    end

    @testset "method dispatch and error messages" begin
        dummy_flow = NeuralOTResult((vfield = VelocityNet(2; hidden = [4]),),
                                    [1.0], :flow, (;))
        @test_throws ArgumentError entropic_map(dummy_flow, randn(Float32, 2, 3))
        @test_throws ArgumentError barycentric_map(dummy_flow, randn(Float32, 2, 3),
                                                   randn(Float32, 2, 3))
        dummy_dual = NeuralOTResult((u = DualPotentialNet(2; hidden = [4]),
                                     v = DualPotentialNet(2; hidden = [4])),
                                    [1.0], :dual, (epsilon = 0.1, cost = SqEuclideanCost()))
        @test_throws ArgumentError inverse_map(dummy_dual, randn(Float32, 2, 3))
        bad = NeuralOTResult((u = 1,), [1.0], :mystery, (;))
        @test_throws ArgumentError monge_map(bad, randn(Float32, 2, 3))
        @test_throws ArgumentError inverse_map(bad, randn(Float32, 2, 3))
    end

    @testset "entropic_map refuses non-squared costs" begin
        res = NeuralOTResult((u = DualPotentialNet(2; hidden = [4]),
                              v = DualPotentialNet(2; hidden = [4])),
                             [1.0], :dual, (epsilon = 0.1, cost = EuclideanCost()))
        @test_throws ArgumentError entropic_map(res, randn(Float32, 2, 3))
    end

    @testset "pushforward is an alias for monge_map" begin
        res = solve_w2(smu, snu; dim = 2, widths = [16, 16, 1], steps = scale_steps(100),
                       inner_steps = 3, batch = 64, seed = 23)
        X = smu(32)
        @test pushforward(res, X) ≈ monge_map(res, X)
    end

    @testset "vector inputs round-trip" begin
        res = solve_w2(smu, snu; dim = 2, widths = [16, 16, 1], steps = scale_steps(100),
                       inner_steps = 3, batch = 64, seed = 24)
        x = smu(1)[:, 1]
        @test size(monge_map(res, x)) == (2,)
        @test monge_map(res, x) ≈ vec(monge_map(res, reshape(x, :, 1)))
        @test size(inverse_map(res, x)) == (2,)
    end
end
