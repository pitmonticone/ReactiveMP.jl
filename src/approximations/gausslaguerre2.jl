export GaussLaguerreQuadrature2

using DomainIntegrals
using FastGaussQuadrature

import FastGaussQuadrature: gausslaguerre_GW, gausslaguerre_rec, gausslaguerre_asy

import Base: ==


struct GaussLaguerreQuadrature2 <: AbstractApproximationMethod
    n :: Int
end

getlength(approximation::GaussLaguerreQuadrature2) = approximation.n

approximation_name(approximation::GaussLaguerreQuadrature2)       = "GaussLaguerre2($(getlength(approximation)))"
approximation_short_name(approximation::GaussLaguerreQuadrature2) = "GL2$(getlength(approximation))"

function approximate(approximation::GaussLaguerreQuadrature2, alpha, fn::Function) 
    x, w = gausslaguerre(getlength(approximation), alpha, reduced = true)
    # return integral(DomainIntegrals.HalfLineRule(x, w), fn)
    return sum([ fn(x[i]) * w[i] for i in 1:getlength(approximation) ])
end

"""
This function calculates the log of the Gauss-laguerre integral by making use of the log of the integrable function.
    ln ( ∫ exp(-x)f(x) dx ) 
    ≈ ln ( ∑ wi * f(xi) ) 
    = ln ( ∑ exp( ln(wi) + logf(xi) ) )
    = ln ( ∑ exp( yi ) )
    = max(yi) + ln ( ∑ exp( yi - max(yi) ) )
    where we make use of the numerically stable log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp
"""
function log_approximate(approximation::GaussLaguerreQuadrature2, alpha, fn::Function)

    # get weights and points
    p    = getlength(approximation)
    x, w = gausslaguerre(p, alpha, reduced = true)
    logw = log.(w)
    T    = eltype(logw)

    # calculate the ln(wi) + logf(xi) terms
    logresult = Vector{T}(undef, p)
    for i = 1:p
        logresult[i] = logw[i] + fn(x[i])
    end

    # log-sum-exp trick, calculate maximum
    max_logresult = maximum(logresult)

    # return log sum exp
    return max_logresult + log(sum(exp.(logresult .- max_logresult)))
end

function Base.:(==)(left::GaussLaguerreQuadrature2, right::GaussLaguerreQuadrature2)
    return getlength(left) == getlength(right)
end