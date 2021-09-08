export ContinuousMultivariateLogPdf

using Distributions

import DomainSets
import DomainIntegrals
import Base: isapprox

struct ContinuousMultivariateLogPdf{ D <: DomainSets.Domain, F } <: ContinuousMultivariateDistribution
    domain :: D
    logpdf :: F
end

ContinuousMultivariateLogPdf(f::Function) = ContinuousMultivariateLogPdf(DomainSets.FullSpace(), f)

(dist::ContinuousMultivariateLogPdf)(x::AbstractVector{ <: Real }) = logpdf(dist, x)

Distributions.support(dist::ContinuousMultivariateLogPdf) = Distributions.RealInterval(DomainSets.infimum(dist.domain), DomainSets.supremum(dist.domain))

Distributions.mean(dist::ContinuousMultivariateLogPdf)    = error("mean() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.median(dist::ContinuousMultivariateLogPdf)  = error("median() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.mode(dist::ContinuousMultivariateLogPdf)    = error("mode() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.var(dist::ContinuousMultivariateLogPdf)     = error("var() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.std(dist::ContinuousMultivariateLogPdf)     = error("std() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.cov(dist::ContinuousMultivariateLogPdf)     = error("cov() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.invcov(dist::ContinuousMultivariateLogPdf)  = error("invcov() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.entropy(dist::ContinuousMultivariateLogPdf) = error("entropy() is not defined for `ContinuousUnivariateLogPdf`.")

# We don't expect neither `pdf` nor `logpdf` to be normalised
Distributions.pdf(dist::ContinuousMultivariateLogPdf, x::AbstractVector{ <: Real })    = exp(logpdf(dist, x))



Base.precision(dist::ContinuousMultivariateLogPdf) = error("precision() is not defined for `ContinuousUnivariateLogPdf`.")

Base.convert(::Type{ ContinuousMultivariateLogPdf }, domain::D, logpdf::F) where { D <: DomainSets.Domain, F } = ContinuousMultivariateLogPdf{D, F}(domain, logpdf)

convert_eltype(::Type{ ContinuousMultivariateLogPdf }, ::Type{ T }, dist::ContinuousMultivariateLogPdf) where { T <: Real } = convert(ContinuousMultivariateLogPdf, dist.domain, dist.logpdf)


prod_analytical_rule(::Type{ <: ContinuousMultivariateLogPdf }, ::Type{ <: ContinuousMultivariateLogPdf }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::ContinuousMultivariateLogPdf, right::ContinuousMultivariateLogPdf)
    @assert left.domain == right.domain "Different domain types in product of generic `ContinuousUnivariateLogPdf` distributions. Left domain is $(left.domain), right is $(right.domain)."
    plogpdf = let left = left, right = right
        (x) -> logpdf(left, x) + logpdf(right, x)
    end
    return ContinuousMultivariateLogPdf(left.domain, plogpdf)
end
