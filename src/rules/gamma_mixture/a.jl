
@rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_b::GammaDistributionsFamily, meta::GammaMixtureNodeMetadata) = begin
    p = probvec(q_switch)[k]
    β = logmean(q_out) + logmean(q_b)
    γ = p * β
    return GammaShapeLikelihood(p, γ, get_shape_likelihood_approximation(meta))
end

# @rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}, meta::GammaMixtureNodeMetadata) where { N1, N2 } = begin
#     # @show q_b[3]
#     p = probvec(q_switch)[k]
#     β = logmean(q_out) + logmean(q_b[k])
#     γ = p * β
#
#     f = let γ = abs(γ), p = p
#         x -> exp(2* γ * x - p * loggamma(x))
#     end
#
#     logCf = let f = f,γ = abs(γ)
#         x -> f(x/γ)/γ
#     end
#     approximation = GaussLaguerreQuadrature(120)
#
#     logC =log(approximate(approximation, logCf))
#
#     mf = let f = f, b = abs(γ), C = logC
#         x -> (x/b) * f(x/b) * exp(-C) / b
#     end
#
#     m = approximate(approximation, mf)
#
#     vf = let f = f, b = abs(γ), C = logC, m = m
#         x -> (x/b  - m) ^ 2 * f(x/b)*exp(-C)/b
#     end
#
#     v = approximate(approximation, vf)
#
#     a = m/v^2
#     b = m/a
#
#
#     return GammaShapeRate(a,b)
# end

# @rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}, meta::GammaMixtureNodeMetadata) where { N1, N2 } = begin
#     @show k
#     q_a = getrecent(getmarginal(connectedvar(__node.as[k])))
#     a_hat_prev = 0.5/(log(mean(q_a))-logmean(q_a))
#     α_prev = 1/a_hat_prev
#
#     for i=1:3
#         ratio = (logmean(q_out)-log(mean(q_out))+log(a_hat_prev)-digamma(a_hat_prev))/(a_hat_prev^2*(α_prev-trigamma(a_hat_prev)))
#         α = α_prev + ratio
#         α_prev = α
#         a_hat_prev = 1/α_prev
#         # @show a_hat_prev
#     end
#
#     return GammaShapeRate(a_hat_prev,1.0)
# end
