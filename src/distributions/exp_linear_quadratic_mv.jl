export MvExponentialLinearQuadratic

import Distributions: pdf, logpdf, ContinuousMultivariateDistribution

struct MvExponentialLinearQuadratic{ A <: AbstractApproximationMethod, T <: Real , M <: AbstractVector{T}} <: ContinuousMultivariateDistribution
    approximation :: A
    a :: M
    b :: M
    c :: T
    d :: T
end

function MvExponentialLinearQuadratic(approximation, a::AbstractVector{<: Real}, b::AbstractVector{<: Real}, c::Real, d::Real)          
    T = promote_type(eltype(a), eltype(b))
    MvExponentialLinearQuadratic(approximation, convert(AbstractArray{T},a), convert(AbstractArray{T},b),promote(c),promote(d))
end

dims(dist::MvExponentialLinearQuadratic) = length(dist.a)

Distributions.pdf(dist::MvExponentialLinearQuadratic, x::Array{Float64})    = exp(logpdf(dist, x))
Distributions.logpdf(dist::MvExponentialLinearQuadratic, x::Array{Float64}) = -0.5 * (dist.a' * x + dist.b' * exp.(dist.c .* x + 0.5 .* dist.d .* x.^ 2 ))
Distributions.mean(dist::MvExponentialLinearQuadratic)            = approximate_meancov(dist.approximation, (z) -> pdf(dist, z) * exp(0.5 * z'*z), MvNormalMeanCovariance(zeros(dims(dist)),Diagonal(ones(dims(dist)))))[1]
Distributions.cov(dist::MvExponentialLinearQuadratic)             = approximate_meancov(dist.approximation, (z) -> pdf(dist, z) * exp(0.5 * z'*z), MvNormalMeanCovariance(zeros(dims(dist)),Diagonal(ones(dims(dist)))))[2]
Distributions.var(dist::MvExponentialLinearQuadratic)             = diag(cov(dist))
precision(dist::MvExponentialLinearQuadratic)                     = inv(cov(dist))
invcov(dist::MvExponentialLinearQuadratic)                        = precision(dist)

prod_analytical_rule(::Type{ <: MultivariateNormalDistributionsFamily }, ::Type{ <: MvExponentialLinearQuadratic }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::MultivariateNormalDistributionsFamily, right::MvExponentialLinearQuadratic)
    mean, variance = approximate_meancov(right.approximation, (z) -> pdf(right, z), left)
    return MvNormalMeanCovariance(mean, variance)
end

function prod(::ProdAnalytical, left::MvExponentialLinearQuadratic, right::MultivariateNormalDistributionsFamily)
    return prod(ProdPreserveParametrisation(),right,left)
end