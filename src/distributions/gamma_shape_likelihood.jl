import SpecialFunctions: loggamma
using Distributions
using JuMP
using Ipopt

"""
    ν(x) ∝ exp(p*β*x - p*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real, A}
    p :: T
    γ :: T # p * β

    approximation :: A
end

Base.show(io::IO, distribution::GammaShapeLikelihood{T}) where T = print(io, "GammaShapeLikelihood{$T}(π = $(distribution.p), γ = $(distribution.γ))")

Distributions.logpdf(distribution::GammaShapeLikelihood, x) = distribution.γ * x - distribution.p * loggamma(x)

function approximate_prod_expectations(approximation::GaussLaguerreQuadrature, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    b = rate(left)

    """
    q(x)    ∝ v(x)*v(x)
            ∝ exp(γ*x - p*ln(Г(x))) * exp((a-1)*ln(x) - b*x)
            = exp(-x) * exp((γ-b+1)*x + (a-1)*ln(x) - p*ln(Г(x)))
    """
    f = let p = right.p, a = shape(left), γ = right.γ, b = b
        x -> exp((γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x))
    end

    logf = let p = right.p, a = shape(left), γ = right.γ, b = b
        x -> (γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x)
    end

    # calculate log-normalization constant
    logC = log_approximate(approximation, logf)

    # mean function without explicitly calculating the normalization constant
    mf = let logf = logf, logC = logC
        x -> x * exp(logf(x) - logC)
    end

    # calculate mean
    m = approximate(approximation, mf)

    # variance function without explicitly calculating the normalization constant
    vf = let logf = logf, logC = logC, m = m
        x -> (x - m) ^ 2 * exp(logf(x) - logC)
    end

    # calculate variance
    v = approximate(approximation, vf)

    return m, v
end

function approximate_prod_expectations(approximation::ImportanceSamplingApproximation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
 
    f = let p = right.p, γ = right.γ
        x -> exp(γ * x - p * loggamma(x))
    end

    m, v = approximate_meancov(approximation, f, GammaShapeScale(shape(left), scale(left)))

    return m, v
end

# Preserve Gamma Distribution

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    m, v = approximate_prod_expectations(right.approximation, left, right)

    a = m ^ 2 / v
    b = m / v

    return GammaShapeRate(a, b)
end

# Expectation maximisation

function prod(::ProdExpectationMaximisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdExpectationMaximisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdExpectationMaximisation(), right, left)
end

function prod(::ProdExpectationMaximisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)

    a, b = shape(left), rate(left)
    γ, p = right.γ, right.p

    model_a = Model(Ipopt.Optimizer)
    register(model_a, :loggamma, 1, loggamma; autodiff = true)
    @variable(model_a, x, start=mean(left))
    @constraint(model_a, x >= 1e-12)
    @NLobjective(model_a, Min, (a-1)*log(x) - b*x + γ*x - p*loggamma(x) - loggamma(a) + a*log(b))
    optimize!(model_a)

    â = value(x)

    return PointMass(â)
end


