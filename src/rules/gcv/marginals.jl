export marginalrule

@marginalrule GCV(:y_x) (m_y::Any, m_x::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * var(q_κ)
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)
    W = [ precision(m_y) + A * B -A * B; -A * B precision(m_x) + A * B ]
    m = cholinv(W) * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

    return MvNormalMeanPrecision(m, W)
end

@marginalrule GCV(:y_x) (m_y::Any, m_x::Any, q_z_κ::Any, q_ω::Any) = begin

    A = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]), q_z_κ)
    B = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)


    W = [ precision(m_y) + A * B -A * B; -A * B precision(m_x) + A * B ]
    m = cholinv(W) * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

    return MvNormalMeanPrecision(m, W)
end

@marginalrule GCV(:y_x) (m_y::Any, m_x::Any, q_z_κ_ω::Any,met::GCVMetadata) = begin

    A = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]-x[3]), q_z_κ_ω)
    W = [ precision(m_y) + A  -A ; -A  precision(m_x) + A  ]
    m = cholinv(W) * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

    return MvNormalMeanPrecision(m, W)
end

@marginalrule GCV(:z_κ) (q_y_x::Any, m_z::Any,m_κ::Any, q_ω::Any,meta::GCVMetadata) = begin
    m_zκ = [mean(m_z); mean(m_κ)]
    v_zκ = [var(m_z) 0.0; 0.0 var(m_κ)]
    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]
    B = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)

    f = x -> exp(-0.5*(x[1]*x[2] + psi*B*exp(-x[1]*x[2])))
    dist =  MvNormalMeanCovariance(m_zκ,v_zκ)
    m_joint,v_joint = approximate_meancov(get_approximation(meta), f, dist)
    return MvNormalMeanCovariance(m_joint,v_joint)
end

@marginalrule GCV(:z_κ_ω) (q_y_x::Any, m_z::Any,m_κ::Any, m_ω::Any,meta::GCVMetadata) = begin
    m_zκω = [mean(m_z); mean(m_κ); mean(m_ω)]
    v_zκω  = [var(m_z) 0.0 0.0; 0.0 var(m_κ) 0.0; 0.0 0.0 var(m_ω)]
    m, v = mean(q_y_x), cov(q_y_x)
    psi = (m[1] - m[2]) ^ 2 + v[1,1]+ v[2,2] -v[1,2] -v[2,1]

    f = x -> exp(-0.5*(x[1]*x[2]+x[3] + psi*exp(-x[1]*x[2]-x[3]))+logpdf(m_z,x[1])+logpdf(m_κ,x[2])+logpdf(m_ω,x[3]))
    dist =  MvNormalMeanCovariance(m_zκω,v_zκω)
    m_joint,v_joint = approximate_meancov(get_approximation(meta), f, dist)
    return MvNormalMeanCovariance(m_joint,v_joint)
end