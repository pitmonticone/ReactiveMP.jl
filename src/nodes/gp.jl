import GaussianProcesses
export GaussianProcess, GaussianProcessMeta 

struct GaussianProcess end 

@node GaussianProcess Stochastic [out, prev, params]

struct GaussianProcessMeta{ I,R, D, M, K,F}
    input::I
    range::R
    index::D
    meanFunction::M
    covarianceFunction::K
    messages :: F
end

get_meanfunction(meta::GaussianProcessMeta)           = meta.meanFunction()
get_covariancefunction(meta::GaussianProcessMeta)     = meta.covarianceFunction()
get_input(meta::GaussianProcessMeta)                  = meta.input
get_range(meta::GaussianProcessMeta)                  = meta.range
get_messages(meta::GaussianProcessMeta)               = meta.messages
get_index(meta::GaussianProcessMeta)                  = meta.index






default_meta(::Type{ GaussianProcess}) = error("GP node requires meta flag to be explicitly specified")



ReactiveMP.prod_analytical_rule( ::Type{ <: GaussianProcessMeta },::Type{ <: UnivariateNormalDistributionsFamily }) = ProdPreserveTypeRight()

function prod(::ProdAnalytical, left::GaussianProcessMeta, right::UnivariateNormalDistributionsFamily)
    m, v = mean(right), var(right)
    kernel_gp = get_covariancefunction(left)
    mean_gp = get_meanfunction(left)
    input = get_input(left)
    n = length(get_messages(left))
    if n>0 && n<length(input)
        m_list = mean.(get_messages(left))
        append!(m_list,m)
        gp_estimate = GP(input[1:n+1], m_list , mean_gp,kernel_gp, log(v))
        m_est,v_est = predict_f(gp_estimate,range);
    elseif n>0 && n==length(input)
        m_list =  mean.(get_messages(left))
        gp_estimate = GP(input, m_list , mean_gp,kernel_gp, log(v))
        m_est,v_est = predict_f(gp_estimate,range);
    else
        m_est,v_est = [],[]
    end
    
    return left,right, (m_est,v_est)
   
end