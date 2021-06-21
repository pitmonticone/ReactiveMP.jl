module RulesWishartMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_marginalrules, huge


@testset "marginalrules:Autoregressive" begin

    @testset ":y_x (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)" begin
        
        @test_marginalrules [ with_float_conversions = true, float64_atol=1e-3, float32_atol=1e-3, bigfloat_atol=1e-3] AR(:y_x) [
            (
                input = (m_y = NormalMeanPrecision(1.0, 1.0), m_x = NormalMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), 
                output = MvNormalMeanPrecision([0.8, 0.6], [2.0 -1.0; -1.0 3.0])
            ),
            (
                input = (m_y = MvNormalMeanCovariance([1.0, 1.0], [10.0 0.0; 0.0 10.0]), 
                         m_x = MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0]), 
                         q_θ = MvNormalMeanCovariance([1.0, 1.0], [10.0 0.0; 0.0 10.0]), 
                         q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Multivariate, 2, ARsafe())), 
                output = MvNormalMeanPrecision([0.276, 0.106, 0.106, 0.097], [1.1    0.0      -1.0     -1.0; 
                                                                              0.0    0.1+huge -huge    -0.0;
                                                                              -1.0  -huge      12+huge  1.0;
                                                                              -1.0  -0.0       1.0      12.0])
            ),
        ]

    end

end


end