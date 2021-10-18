export rule

# @rule GCV(:z, Marginalisation) (q_y_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

#     m, v = mean(q_y_x), cov(q_y_x)
#     psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
#     A = exp(-mean(q_ω)+var(q_ω) / 2)

#     a = mean(q_κ)
#     b = psi * A
#     c = -mean(q_κ)
#     d = var(q_κ)

#     return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
# end


@rule GCV(:z, Marginalisation) (q_y_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    m_κ, v_κ = mean(q_κ), var(q_κ)
    A = approximate_expectation(get_approximation(meta),x -> exp(-x), q_ω)
    points = ReactiveMP.getpoints(get_approximation(meta),m_κ,v_κ)
    weights = ReactiveMP.getweights(get_approximation(meta),m_κ,v_κ)

    b = psi * A

    f = x -> -0.5*(m_κ*x + b*sum(exp.(points .* (-x)) .* weights))

    return ContinuousUnivariateLogPdf(f)
end


@rule GCV(:z, Marginalisation) (q_y_x::Any, m_z::UnivariateNormalDistributionsFamily, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    A = exp(-mean(q_ω)+var(q_ω) / 2)

    a = mean(q_κ)
    b = psi * A
    c = -mean(q_κ)
    d = var(q_κ)

    message = ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
    dist = prod(ProdAnalytical(), m_z, message)

    return divide_marginals(dist, m_z) 
end

@rule GCV(:z, Marginalisation) (q_y::Any,q_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)
    psi = (my - mx) ^ 2 + vy+ vx
    A = exp(-mean(q_ω)+var(q_ω) / 2)

    a = mean(q_κ)
    b = psi * A
    c = -mean(q_κ)
    d = var(q_κ)

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end