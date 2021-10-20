export rule


@rule GCV(:ω, Marginalisation) (q_y_x::Any, q_z::Any, q_κ::Any, meta::GCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)

    γ = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    A = exp(-mean(q_κ) * mean(q_z) + γ / 2)
    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = one(typeof(γ))
    b = psi * A
    c = -one(typeof(γ))
    d = zero(typeof(γ))

    return ExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end

@rule GCV(:ω, Marginalisation) (q_y_x::Any, q_z_κ::Any, meta::GCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)
    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]), q_z_κ)
    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    b = psi * tmp1

    f = ω -> -0.5*(ω+ b*exp(-ω))

    return ContinuousUnivariateLogPdf(f)
end

@rule GCV(:ω, Marginalisation) (q_y::Any, q_x::Any, q_z::Any, q_κ::Any, meta::GCVMetadata) = begin
    
    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)

    γ = mean(q_z) ^ 2 * var(q_κ) + mean(q_κ) ^ 2 * var(q_z) + var(q_z) * var(q_κ)
    A = exp(-mean(q_κ) * mean(q_z) + γ / 2)
    psi = (my - mx) ^ 2 + vy + vx

    a = mean(q_z)
    b = psi * A
    c = -a
    d = var(q_z)

    return ExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end

@rule GCV(:ω, Marginalisation) (q_y_x::Any, m_z::Any, m_κ::Any, m_ω::Any, meta::GCVMetadata) = begin

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
    
    return divide_marginals(NormalMeanVariance(joint_m[3],joint_v[3,3]), m_ω)
end