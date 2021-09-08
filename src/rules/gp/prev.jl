@rule GaussianProcess(:prev, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_params::PointMass, meta::GaussianProcessMeta) = begin 
    append!(get_messages(meta),[m_out])
    
    return missing 

end

@rule GaussianProcess(:prev, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, q_params::Any, meta::GaussianProcessMeta) = begin 
    return missing
end