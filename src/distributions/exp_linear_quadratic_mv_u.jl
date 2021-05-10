export MvUniExponentialLinearQuadratic

import Distributions: pdf, logpdf, ContinuousUnivariateDistribution

struct MvUniExponentialLinearQuadratic{ A <: AbstractApproximationMethod, T <: Real , M <: AbstractVector{T}} <: ContinuousUnivariateDistribution
    approximation :: A
    a :: T
    b :: T
    c :: M
    d :: M
    e :: M
    f :: M
end

function MvUniExponentialLinearQuadratic(approximation, a::Real, b::Real, c::AbstractVector{ <: Real}, d::AbstractVector{ <: Real},e::AbstractVector{ <: Real},f::AbstractVector{ <: Real})          
    T = promote_type(eltype(c),eltype(d), eltype(e),eltype(f))
    MvUniExponentialLinearQuadratic(approximation, promote(a),promote(b),convert(AbstractArray{T},c),convert(AbstractArray{T},d),convert(AbstractArray{T},e),convert(AbstractArray{T},f))
end


Distributions.pdf(dist::MvUniExponentialLinearQuadratic, x::Real)    = exp(logpdf(dist, x))
Distributions.logpdf(dist::MvUniExponentialLinearQuadratic, x::Real) = -0.5 * (dist.a * x + dist.b + dist.c' * exp.(dist.d .* x + dist.e + 0.5 .* dist.f .* x^2 ))
Distributions.mean(dist::MvUniExponentialLinearQuadratic)            = approximate_meancov(dist.approximation, (z) -> pdf(dist, z) * exp(0.5 * z^2)/sqrt(2pi), NormalMeanVariance(0.0,1.0))[1]
Distributions.cov(dist::MvUniExponentialLinearQuadratic)             = approximate_meancov(dist.approximation, (z) -> pdf(dist, z) * exp(0.5 * z^2)/sqrt(2pi), NormalMeanVariance(0.0,1.0))[2]
Distributions.var(dist::MvUniExponentialLinearQuadratic)             = diag(cov(dist))

precision(dist::MvUniExponentialLinearQuadratic)                    = cholinv(cov(dist))
invcov(dist::MvUniExponentialLinearQuadratic)                       = precision(dist)
function prod(::ProdPreserveParametrisation, left::UnivariateNormalDistributionsFamily, right::MvUniExponentialLinearQuadratic)
    mean, variance = approximate_meancov(right.approximation, (z) -> pdf(right, z), left)
    return NormalMeanVariance(mean, variance)
end


function prod(::ProdPreserveParametrisation, left::MvUniExponentialLinearQuadratic , right::UnivariateNormalDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right,left)
end