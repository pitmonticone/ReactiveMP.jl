export rule

# @rule GCV(:y, Marginalisation) (m_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

#     ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * cov(q_κ)
#     A = exp(-mean(q_ω) + var(q_ω) / 2)
#     B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

#     return NormalMeanVariance(mean(m_x), var(m_x) + inv(A * B))
# end

@rule GCV(:y, Marginalisation) (m_x::Any, q_z::Any, q_κ::Any, q_ω::Any,meta::GCVMetadata) = begin
    m_zk = [mean(q_z); mean(q_κ)]
    v_zk = [var(q_z) 0.0; 0.0 var(q_κ)]

    q_zk = MvNormalMeanCovariance(m_zk,v_zk)
    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]), q_zk)
    tmp2 = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)

    return NormalMeanVariance(mean(m_x), var(m_x) + inv(tmp1 * tmp2))
end

@rule GCV(:y, Marginalisation) (q_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * cov(q_κ)
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

    return NormalMeanVariance(mean(q_x), inv(A * B))
end