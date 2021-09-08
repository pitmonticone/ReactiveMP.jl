@rule GaussianProcess(:params, Marginalisation) (q_out_prev::MultivariateNormalDistributionsFamily, meta::GaussianProcessMeta) = begin 
    n = length(ReactiveMP.get_messages(meta))
    input = ReactiveMP.get_input(meta)
    kernel = ReactiveMP.get_covariancefunction(meta)
    meanf = ReactiveMP.get_meanfunction(meta)
    vect = [input[1]; input[2]]
    
    m,c = mean_cov(q_out_prev)
    h = (x) -> x[1]*exp.(-vect*vect'/x[2])
    
    f = (θ) -> -0.5*( tr(h([θ[2] θ[3]])*c) - (θ[1]*ones(2))'*h([θ[2] θ[3]])*m - m'*h([θ[2] θ[3]])*(θ[1]*ones(2)) + (θ[1]*ones(2))'*h([θ[2] θ[3]])*(θ[1]*ones(2)))
    return ContinuousMultivariateLogPdf(f)
end