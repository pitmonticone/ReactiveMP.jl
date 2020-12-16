export rule

@rule Wishart(:out, Marginalisation) (m_ν::Dirac, m_S::Dirac) = Wishart(mean(m_ν), mean(m_S))

@rule Wishart(:out, Marginalisation) (q_ν::Any, q_S::Any) = Wishart(mean(m_ν), mean(m_S))