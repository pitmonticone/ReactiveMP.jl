export rule

@rule GCV(:y, Marginalisation) (m_x::UniNormalOrExpLinQuad, q_z::Any, q_κ::Any, q_ω::Any) = begin

    x_mean, x_var = mean_var(m_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean ^ 2 * z_var + z_mean ^ 2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(x_mean, x_var + inv(A * B))
end

@rule GCV(:y, Marginalisation) (m_x::Any, q_z_κ::Any, q_ω::Any,meta::GCVMetadata) = begin
    m_zk = mean(q_z_κ)
    v_zk = cov(q_z_κ)

    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]), q_z_κ)
    tmp2 = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)

    return NormalMeanVariance(mean(m_x), var(m_x) + inv(tmp1 * tmp2))
end

@rule GCV(:y, Marginalisation) (m_x::Any, q_z_κ_ω::Any,meta::GCVMetadata) = begin
    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]-x[3]), q_z_κ_ω)

    return NormalMeanVariance(mean(m_x), var(m_x) + inv(tmp1))
end


@rule GCV(:y, Marginalisation) (q_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    x_mean        = mean(q_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean ^ 2 * z_var + z_mean ^ 2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(x_mean, inv(A * B))
end

@rule GCV(:y, Marginalisation) (m_y::Any, m_x::Any, m_z::Any, q_κ::Any, q_ω::Any,meta::GCVMetadata) = begin

    tmp2 = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)
    mean_κ, v_κ = mean_cov(q_κ)
    points = ReactiveMP.getpoints(get_approximation(meta),mean_κ,v_κ)
    weights = ReactiveMP.getweights(get_approximation(meta),mean_κ,v_κ)

    h = z -> sum(exp.(-0.5*(points .* z  )) .* weights)
    f = x -> exp(-0.5*( mean(q_κ)*x[3] + h(x[3])*tmp2*(x[2]-x[1])^2  + logpdf(m_x,x[2])+logpdf(m_y, x[1])+logpdf(m_z,x[3])))

    m_yxz = [mean(m_y); mean(m_x); mean(m_z)]
    v_yxz  = [var(m_y) 0.0 0.0; 0.0 var(m_x) 0.0; 0.0 0.0 var(m_z)]
    dist =  MvNormalMeanCovariance(m_yxz,v_yxz)
    m_joint,v_joint = approximate_meancov(get_approximation(meta), f, dist)
    dist_proxy = NormalMeanVariance(m_joint[1],v_joint[1,1])

    return divide_marginals(dist_proxy, m_y) 
end


@rule GCV(:y, Marginalisation) (m_x::Any,  m_z::Any, q_κ::Any, q_ω::Any,meta::GCVMetadata) = begin
   

    tmp2 = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)
    mean_κ, v_κ = mean_cov(q_κ)
    points = ReactiveMP.getpoints(get_approximation(meta),mean_κ,v_κ)
    weights = ReactiveMP.getweights(get_approximation(meta),mean_κ,v_κ)

    h = z -> sum(exp.(-0.5*(points .* z  )) .* weights)

    points_z = ReactiveMP.getpoints(get_approximation(meta),mean(m_x),var(m_x))
    weights_z = ReactiveMP.getweights(get_approximation(meta),mean(m_x),var(m_x))

    f = x -> log(sum( exp.(-0.5.* tmp2 .* (x - mean(m_x)^2 ) ./ (var(m_x) .+ h.(points_z))) .* weights_z))

    return ContinuousUnivariateLogPdf(f)
end