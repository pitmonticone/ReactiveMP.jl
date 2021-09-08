ReactiveMP.@marginalrule GaussianProcess(:out_prev) (m_out::Any, m_prev::Tuple, q_params::Any, meta::GaussianProcessMeta) = begin 
    n = length(ReactiveMP.get_messages(m_prev[1]))
    input = ReactiveMP.get_input(meta)
    kernel = ReactiveMP.get_covariancefunction(meta)
    meanf = ReactiveMP.get_meanfunction(meta)
    if n>0 && n<length(input)-1
        message_prev = ReactiveMP.get_messages(m_prev[1])[end]
        joint_mean = [mean(m_out) ; mean(message_prev)]
        c = GaussianProcesses.cov(kernel, [input[n+2] input[n+1]])
        return (ProdAnalytical(),MvNormalMeanCovariance(ones(2)*meanf.β,c),MvNormalMeanCovariance(joint_mean,[var(m_out) 0; 0 var(message_prev)]))
    elseif n>0 && n==length(input)
        message_prev = ReactiveMP.get_messages(m_prev[1])[end]
        joint_mean = [mean(m_out) ; mean(message_prev)]
        c = GaussianProcesses.cov(kernel, [input[n] input[n-1]])
        return prod(ProdAnalytical(),MvNormalMeanCovariance(ones(2)*meanf.β,c),MvNormalMeanCovariance(joint_mean,[var(m_out) 0; 0 var(message_prev)]))
    else
        joint_mean = [mean(m_out) ; mean(m_prev[2])]
        c = GaussianProcesses.cov(kernel, [input[2] input[1]])
        return prod(ProdAnalytical(),MvNormalMeanCovariance(ones(2)*meanf.β,c),MvNormalMeanCovariance(joint_mean,[var(m_out) 0; 0 var(m_prev[2])]))
    end
    
end


ReactiveMP.@marginalrule GaussianProcess(:out_prev) (m_out::Any, m_prev::UnivariateNormalDistributionsFamily, q_params::Any, meta::GaussianProcessMeta) = begin 
    kernel = ReactiveMP.get_covariancefunction(meta)
    input = ReactiveMP.get_input(meta)
    meanf = ReactiveMP.get_meanfunction(meta)
    joint_mean = [mean(m_out) ; mean(m_prev)]
    c = GaussianProcesses.cov(kernel, [input[2] input[1]])
    return prod(ProdAnalytical(),MvNormalMeanCovariance(ones(2)*meanf.β,c), MvNormalMeanCovariance(joint_mean,[var(m_out) 0.0; 0.0 var(m_prev)]))
    
end