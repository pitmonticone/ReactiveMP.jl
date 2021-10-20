export rule

@rule GCV(:x, Marginalisation) (m_y::UniNormalOrExpLinQuad, q_z::Any, q_κ::Any, q_ω::Any) = begin

    y_mean, y_var = mean_var(m_y)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)
    
    ksi = κ_mean ^ 2 * z_var + z_mean ^ 2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(y_mean, y_var + inv(A * B))
end


@rule GCV(:x, Marginalisation) (m_y::Any, q_z_κ::Any, q_ω::Any,meta::GCVMetadata) = begin
    m_zk = mean(q_z_κ)
    v_zk = cov(q_z_κ)

    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]), q_z_κ)
    tmp2 = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)

    return NormalMeanVariance(mean(m_y), var(m_y) + inv(tmp1 * tmp2))
end

@rule GCV(:x, Marginalisation) (m_y::Any, q_z_κ_ω::Any,meta::GCVMetadata) = begin

    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]-x[3]), q_z_κ_ω)

    return NormalMeanVariance(mean(m_y), var(m_y) + inv(tmp1))
end

@rule GCV(:x, Marginalisation) (q_y::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    y_mean        = mean(q_y)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ω_mean, ω_var = mean_var(q_ω)

    ksi = κ_mean ^ 2 * z_var + z_mean ^ 2 * κ_var + z_var * κ_var
    A = exp(-ω_mean + ω_var / 2)
    B = exp(-κ_mean * z_mean + ksi / 2)

    return NormalMeanVariance(y_mean, inv(A * B))
end