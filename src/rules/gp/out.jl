export rule

@rule GaussianProcess(:out, Marginalisation) (m_prev::GaussianProcessMeta  ,m_params::PointMass, meta::GaussianProcessMeta,) = begin
    append!(get_messages(meta),get_messages(m_prev))
    return meta 
end

@rule GaussianProcess(:out, Marginalisation) (m_prev::Any ,m_params::PointMass, meta::GaussianProcessMeta,) = begin
    # append!(get_messages(meta),[m_prev])
    return meta
end

@rule GaussianProcess(:out, Marginalisation) (m_prev::Tuple, m_params::PointMass, meta::GaussianProcessMeta) = begin 
    messages = get_messages(m_prev[1])

    append!(get_messages(meta), messages, [ m_prev[2] ])


    return meta
end


@rule GaussianProcess(:out, Marginalisation) (m_prev::Any, q_params::Any, meta::GaussianProcessMeta) = begin 
    # @show m_prev
    
    return meta
end

@rule GaussianProcess(:out, Marginalisation) (m_prev::Tuple, q_params::Any, meta::GaussianProcessMeta) = begin 
    messages = get_messages(m_prev[1])

    append!(get_messages(meta), messages, [ m_prev[2] ])

    return meta
end