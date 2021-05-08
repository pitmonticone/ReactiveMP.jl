export marginalrule

@marginalrule SGCV(:y_x) (m_y::Any, m_x::Any, q_z::Any, q_κ::Any, q_ω::Any,q_s::Any) = begin

    η̂ = ϕ(q_z,q_κ,q_ω,q_s)
    W = [ precision(m_y)+η̂ -η̂; -η̂  precision(m_x) + η̂ ]
    m = cholinv(W) * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

    return MvNormalMeanPrecision(m, W)
end
