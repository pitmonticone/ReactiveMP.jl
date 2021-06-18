module RulesAutoregressiveYTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:x" begin

    @testset "Variational: (m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)" begin

        @test_rules [ with_float_conversions = true ] Autoregressive(:x, Marginalisation) [
            (input = (m_y = NormalMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(1/3, 2/3)),
            (input = (m_y = NormalMeanVariance(2.0, 1.0), q_θ = NormalMeanVariance(1.0, 2.0), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(2/5, 2/5)),
            (input = (m_y = NormalMeanVariance(2.0, 2.0), q_θ = NormalMeanPrecision(2.0, 1.0), q_γ = GammaShapeRate(1/2, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(2/3, 2/3))
        ]

        @test_rules [ with_float_conversions = false ] Autoregressive(:x, Marginalisation) [
            (input = (m_y = MvNormalMeanPrecision([1.0], [1.0]), q_θ = MvNormalMeanPrecision([1.0], [1.0]), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Multivariate, 1, ARsafe())), output = MvNormalMeanCovariance([1/3], [2/3])),
            (input = (m_y = MvNormalMeanCovariance([1/2, 1/2], [1/2 1/4; 1/4 1/2]), q_θ = MvNormalMeanPrecision([1/2, -1/2], [2.0 0.0; 0.0 1.0]), q_γ = GammaShapeRate(2.0, 1.0), meta = ARMeta(Multivariate, 2, ARsafe())), output = MvNormalMeanCovariance([1/3, -1/16], [1/3 0.0; 0.0 7/16])),
            (input = (m_y = MvNormalMeanCovariance([1.0, 1.0, 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), q_θ = MvNormalMeanPrecision([1.0, 1.0, 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Multivariate, 3, ARsafe())), output = MvNormalMeanCovariance([1/2, 1/2, 0.0], [7/16 -1/16 -1/8; -1/16 7/16 -1/8; -1/8 -1/8 3/4])),
        ]
    end

end

end