export rule

@rule SGCV(:x, Marginalisation) (m_y::Any, q_z::Any, q_κ::Any, q_ω::Any,q_s::Any) = begin
    return NormalMeanVariance(mean(m_y), var(m_y) + 1/ϕ(q_z,q_κ,q_ω,q_s))
end

@rule SGCV(:x, Marginalisation) (q_y::Any, q_z::Any, q_κ::Any, q_ω::Any,q_s::Any) = begin
    return NormalMeanVariance(mean(q_y), var(q_y) + 1/ϕ(q_z,q_κ,q_ω,q_s))
end
