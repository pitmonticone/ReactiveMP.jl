export rule

@rule GCV(:κ, Marginalisation) (q_y_x::Any, q_z::Any, q_ω::Any, meta::GCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)

    A = exp(-mean(q_ω) + var(q_ω) / 2)
    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = mean(q_z)
    b = psi * A
    c = -a
    d = var(q_z)

    return ExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end

@rule GCV(:κ, Marginalisation) (q_y_x::Any, m_z::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    mean_z, v_z = mean(m_z), var(m_z)
    # A = approximate_expectation(get_approximation(meta),x -> exp(-x), q_ω)
    # points = ReactiveMP.getpoints(get_approximation(meta),mean_z,v_z)
    # weights = ReactiveMP.getweights(get_approximation(meta),mean_z,v_z)
    A = expectation(x -> exp(-x), Normal(mean(q_ω),std(q_ω)))
    points = ReactiveMP.getpoints(get_approximation(meta),mean_z,v_z)
    weights = ReactiveMP.getweights(get_approximation(meta),mean_z,v_z)

    b = psi * A

    f = z -> log(sum(exp.(-0.5*(points .* z + b*exp.(points .* (-z)) )) .* weights))

    return ContinuousUnivariateLogPdf(f)
end

@rule GCV(:κ, Marginalisation) (q_y_x::Any, m_z::Any, m_κ::Any, m_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    mean_z, v_z = mean(m_z), var(m_z)
    A = approximate_expectation(get_approximation(meta),x -> exp(-x), m_ω)
    points = ReactiveMP.getpoints(get_approximation(meta),mean_z,v_z)
    weights = ReactiveMP.getweights(get_approximation(meta),mean_z,v_z)

    

    f = x -> exp(-0.5*(x[1]*x[2] + psi*exp(-x[1]x[2] - x[3])) + logpdf(m_z,x[1]) + logpdf(m_κ,x[2]) +logpdf(m_ω,x[3]))
    m_zκω = [mean(m_z); mean(m_κ); mean(m_ω)]
    v_zκω  = [var(m_z) 0.0 0.0; 0.0 var(m_κ) 0.0; 0.0 0.0 var(m_ω)] 

    joint_m, joint_v = approximate_meancov(get_approximation(meta), f, m_zκω,v_zκω)
    
    return divide_marginals(NormalMeanVariance(joint_m[2],joint_v[2,2]), m_κ)
end


@rule GCV(:κ, Marginalisation) (q_y::Any,q_x::Any, q_z::Any, q_ω::Any, meta::GCVMetadata) = begin
    
    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)

    A = exp(-mean(q_ω) + var(q_ω) / 2)
    psi = (my - mx) ^ 2 + vy + vx

    a = mean(q_z)
    b = psi * A
    c = -a
    d = var(q_z)

    return ExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end

@rule GCV(:κ, Marginalisation) (q_y_x_z::Any, q_ω::Any, meta::GCVMetadata) = begin

    m, v = mean(q_y_x_z), cov(q_y_x_z)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    A = approximate_expectation(get_approximation(meta),x -> exp(-x), NormalMeanVariance(mean(q_ω),std(q_ω)))
    points = ReactiveMP.getpoints(get_approximation(meta),m[3],v[3,3])
    weights = ReactiveMP.getweights(get_approximation(meta),m[3],v[3,3])

    b = psi * A
    h = κ -> sum(exp.(-0.5*(points .* κ  )) .* weights)
    f = κ -> -0.5*(m[3]*κ + b*h(κ))

    return ContinuousUnivariateLogPdf(f)
end