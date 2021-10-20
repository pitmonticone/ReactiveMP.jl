export rule

@rule GCV(:z, Marginalisation) (q_y_x::Any, q_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    A = exp(-mean(q_ω)+var(q_ω) / 2)

    a = mean(q_κ)
    b = psi * A
    c = -mean(q_κ)
    d = var(q_κ)

    return ExponentialLinearQuadratic(get_approximation(meta), a, b, c, d)
end


@rule GCV(:z, Marginalisation) (q_y_x::Any, m_κ::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    mean_κ, v_κ = mean(m_κ), var(v_κ)
    A = approximate_expectation(get_approximation(meta),x -> exp(-x), q_ω)
    points = ReactiveMP.getpoints(get_approximation(meta),mean_κ,v_κ)
    weights = ReactiveMP.getweights(get_approximation(meta),mean_κ,v_κ)

    b = psi * A

    f = z -> -0.5*(mean_κ*z + b*sum(exp.(points .* (-z)) .* weights))

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

@rule GCV(:z, Marginalisation) (q_y_x::Any, m_z::Any, m_κ::Any, m_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    mean_z, v_z = mean(m_z), var(v_z)
    A = approximate_expectation(get_approximation(meta),x -> exp(-x), m_ω)
    points = ReactiveMP.getpoints(get_approximation(meta),mean_z,v_z)
    weights = ReactiveMP.getweights(get_approximation(meta),mean_z,v_z)

    

    f = x -> exp(-0.5*(x[1]*x[2] + psi*exp(-x[1]x[2] - x[3])) + logpdf(m_z,x[1]) + logpdf(m_κ,x[2]) +logpdf(m_ω,x[3]))
    m_zκω = [mean(m_z); mean(m_κ); mean(m_ω)]
    v_zκω  = [var(m_z) 0.0 0.0; 0.0 var(m_κ) 0.0; 0.0 0.0 var(m_ω)]

    joint_m, joint_v = approximate_meancov(get_approximation(meta), f, m_zκω,v_zκω)
    
    return divide_marginals(NormalMeanVariance(joint_m[1],joint_v[1,1]), m_z)
end