export rule

@rule SGCV(:y, Marginalisation) (m_x::Any, q_z::Any, q_κ::Any, q_ω::Any,q_s::Any) = begin
    return NormalMeanVariance(mean(m_x), var(m_x) + 1/ϕ(q_z,q_κ,q_ω,q_s))
end
