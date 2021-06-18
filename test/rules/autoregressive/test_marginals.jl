module RulesWishartMarginalsTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_marginalrules


@testset "marginalrules:Autoregressive" begin

    @testset ":y_x (m_y::NormalDistributionsFamily, m_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta)" begin
        
        @test_marginalrules [ with_float_conversions = false ] AR(:y_x) [
            (
                input = (m_y = NormalMeanPrecision(1.0, 1.0), m_x = NormalMeanPrecision(1.0, 1.0), q_θ = NormalMeanPrecision(1.0, 1.0), q_γ = GammaShapeRate(1.0, 1.0), meta = ARMeta(Univariate, 1, ARsafe())), 
                output = MvNormalMeanPrecision([0.8, 0.6], [2.0 -1.0; -1.0 3.0])
            ),
        ]

    end

end


end