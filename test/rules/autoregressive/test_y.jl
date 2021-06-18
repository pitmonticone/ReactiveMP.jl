module RulesAutoregressiveYTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:Autoregressive:y" begin

    @testset "Variational: (m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)" begin

        @test_rules [ with_float_conversions = true ] Autoregressive(:y, Marginalisation) [
            (input = (m_x = NormalMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(1/2, 3/2)),
            (input = (m_x = NormalMeanVariance(2.0, 2.0), q_θ = NormalMeanVariance(1/2, 1.0), q_γ = GammaShapeRate(1/2, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(1/2, 9/4)),
            (input = (m_x = NormalMeanVariance(2.0, 2.0), q_θ = NormalMeanPrecision(2.0, 1.0), q_γ = GammaShapeRate(1/2, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), output = NormalMeanVariance(2.0, 6.0))
        ]

        @test_rules [ with_float_conversions = false ] Autoregressive(:y, Marginalisation) [
            (input = (m_x = MvNormalMeanPrecision([1.0], [1.0]), q_θ = MvNormalMeanPrecision([1.0], [1.0]), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Multivariate, 1, ARsafe())), output = MvNormalMeanCovariance([1/2], [3/2])),
            (input = (m_x = MvNormalMeanPrecision([1.0, 2.0], [1.0 0.0; 0.0 1/2]), q_θ = MvNormalMeanPrecision([1/2, -1/2], [2.0 0.0; 0.0 1.0]), q_γ = GammaShapeRate(2.0, 1.0), meta = ARMeta(Multivariate, 2, ARsafe())), output = MvNormalMeanCovariance([1/20, 1/2], [29/40 1/4; 1/4 1/2])),
            (input = (m_x = MvNormalMeanPrecision([1.0, 1.0, 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), q_θ = MvNormalMeanPrecision([1/2, -1/2, 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), q_γ = GammaShapeRate(2.0, 1.0), meta = ARMeta(Multivariate, 3, ARsafe())), output = MvNormalMeanCovariance([1/3, 1/3, 1/3], [1.0 1/6 -1/6; 1/6 1/3 0.0; -1/6 0.0 1/3])),
        ]
    end

end

end