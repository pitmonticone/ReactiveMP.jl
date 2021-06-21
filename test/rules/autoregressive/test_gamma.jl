module RulesAutoregressiveYTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:γ" begin

    @testset "Variational: (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, meta::ARMeta)" begin

        @test_rules [ with_float_conversions = true ] Autoregressive(:γ, Marginalisation) [
            (input = (q_y_x = MvNormalMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0]), q_θ = NormalMeanPrecision(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = GammaShapeRate(3/2, 2.0)),
            (input = (q_y_x = MvNormalMeanPrecision([1/2, 1.0], [1.0 0.0; 0.0 1.0]), q_θ = NormalMeanPrecision(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = GammaShapeRate(3/2, 17/8)),
        ]

        @test_rules [ with_float_conversions = true ] Autoregressive(:γ, Marginalisation) [
            (input = (q_y_x = MvNormalMeanCovariance([1.0, 1.0, 1/2, 1/2], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]), q_θ = MvNormalMeanPrecision(1/2*ones(2), [1.0 1/2; -1/2 1.0]), meta = ARMeta(Multivariate, 2, ARsafe())), output = GammaShapeRate(3/2, 19/8)),
            (input = (q_y_x = MvNormalMeanCovariance([1.0, 1.0, 1/2, 1.0], [1.0 2.0 0.0 0.0; 2.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]), q_θ = MvNormalMeanPrecision(1/2*ones(2), [1.0 1/2; -1/2 1.0]), meta = ARMeta(Multivariate, 2, ARsafe())), output = GammaShapeRate(3/2, 251/96)),
        ]
    end

end

end