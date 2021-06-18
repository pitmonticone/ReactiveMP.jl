module RulesAutoregressiveYTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:θ" begin

    @testset "Variational: (q_y_x::MultivariateNormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)" begin

        @test_rules [ with_float_conversions = false ] Autoregressive(:θ, Marginalisation) [
            (input = (q_y_x = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(1/2, 1/2)),
            (input = (q_y_x = MvNormalMeanPrecision([1/2, 1.0], [1.0 0.0; 0.0 1.0]), q_γ = GammaShapeRate(2.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(1/4, 1/4)),
        ]

        @test_rules [ with_float_conversions = false ] Autoregressive(:θ, Marginalisation) [
            (input = (q_y_x = MvNormalMeanCovariance([1.0, 1.0, 1/2, 1/2], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Multivariate, 2, ARsafe())), output = MvNormalMeanCovariance([1/3, 1/3], [15/18 -1/6; -1/6 15/18])),
            (input = (q_y_x = MvNormalMeanCovariance([1.0, 1.0, 1/2, 1.0], [1.0 2.0 0.0 0.0; 2.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]), q_γ = GammaShapeRate(2.0, 1.0), meta = ARMeta(Multivariate, 2, ARsafe())), output = MvNormalMeanCovariance([2/9, 4/9], [4/9 -1/9; -1/9 5/18])),
        ]
    end

end

end