import SpecialFunctions: loggamma

"""
    ν(x) ∝ exp(p*β*x - π*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real, A}
    p :: T
    γ :: T # p * β

    approximation :: A
end

using FastGaussQuadrature: gausslaguerre
"""
    ν(x) ∝ exp(p*β*x - π*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""

function approximate_prod_expectations(approximation::GaussLaguerreQuadrature, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
    b = rate(left)

    """
    q(x)    ∝ v(x)*v(x)
            ∝ exp(γ*x - p*ln(Г(x))) * exp((a-1)*ln(x) - b*x)
            = exp(-x) * exp((γ-b+1)*x + (a-1)*ln(x) - p*ln(Г(x)))
    """
    f = let p = right.p, a = shape(left), γ = right.γ, b = b
        x -> exp((γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x))
    end

    logf = let p = right.p, a = shape(left), γ = right.γ, b = b
        x -> (γ - b + 1) * x - p * loggamma(x) + (a - 1) * log(x)
    end

    # calculate log-normalization constant
    logC = log_approximate(approximation, logf)

    # mean function without explicitly calculating the normalization constant
    mf = let logf = logf, logC = logC
        x -> x * exp(logf(x) - logC)
    end

    # calculate mean
    m = approximate(approximation, mf)

    # variance function without explicitly calculating the normalization constant
    vf = let logf = logf, logC = logC, m = m
        x -> (x - m) ^ 2 * exp(logf(x) - logC)
    end

    logmf = let logf = logf, logC = logC, m = m
        x -> log(x) * exp(logf(x) - logC)
    end

    logm = approximate(approximation,logmf)


    # calculate variance
    v = approximate(approximation, vf)

    return logC, m, v, logm
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    @assert left.approximation == right.approximation "Different approximation types for $(left) and $(right) messages"
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ, left.approximation)
end

function prod(::ProdPreserveParametrisation, left::GammaShapeLikelihood, right::GammaDistributionsFamily)
    return prod(ProdPreserveParametrisation(), right, left)
end

function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)

    logC, m, v, logm = approximate_prod_expectations(right.approximation,left,right)
    log_partition(x) = -loggamma(x[1]) + x[1]*log(x[2]) - (x[1]-m^2/v)^2 - (x[2]-m/v)^2 - (x[1]-m*x[2]) - (digamma(x[1]) - log(x[2]) - logm)
    # @show logC, m, v, logm,logm2
    θ = gradientOptimization(log_partition, natural_gradient, [1.5 ; 1.5], 0.01)
    @show θ[1], θ[2]
    return GammaShapeRate(θ[1], θ[2])
end

# function prod(::ProdPreserveParametrisation, left::GammaDistributionsFamily, right::GammaShapeLikelihood)
#     _, m, v,_ = approximate_prod_expectations(right.approximation, left, right)
#
#     a = m ^ 2 / v
#     b = m / v
#
#     return GammaShapeRate(a, b)
# end

using ForwardDiff

function natural_gradient(f::Function,point)
    n = length(point)
    grad_f(z) = ForwardDiff.gradient(f,z)
    hessian_f(z) = ForwardDiff.hessian(f,z)
    nat_grad = -inv(hessian_f(point))*grad_f(point)
    nat_grad, inv(hessian_f(point))
    return nat_grad
end

function gradientOptimization(log_partition::Function, natural_gradient::Function, m_initial, step_size)

    dim_tot = length(m_initial)
    m_total = zeros(dim_tot)
    m_average = zeros(dim_tot)
    m_new = ones(dim_tot)
    m_old = m_initial
    satisfied = false
    step_count = 0
    positive = false

    while !satisfied && !positive
        m_new = m_old .+ step_size.*natural_gradient(log_partition,m_old)
        if (m_new .> 0)[1] && (m_new .> 0)[2]
            positive = true
        else
            m_new = abs.(m_new)
        end
        if (log_partition(m_new) > log_partition(m_old))
            proposal_step_size = 10*step_size
            m_proposal = m_old .+ proposal_step_size.*natural_gradient(log_partition,m_old)
            if (m_proposal .> 0)[1] && (m_proposal .> 0)[2]
                positive = true
            else
                m_proposal = abs.(m_proposal)
            end
            if log_partition(m_proposal) > log_partition(m_new)
                m_new = m_proposal
                step_size = proposal_step_size
            end
        else
            step_size = 0.1*step_size
            m_new = m_old .+ step_size.*natural_gradient(log_partition,m_old)
        end

        step_count += 1
        m_total .+= m_old
        m_average = m_total ./ step_count
        if step_count > 10
            if sum(sqrt.(((m_new.-m_average)./m_average).^2)) < dim_tot*0.00001
                satisfied = true
            end
        end
        if step_count > dim_tot*250
            satisfied = true
        end

        if (m_new .> 0)[1] && (m_new .> 0)[2]
            positive = true
        else
            m_new = abs.(m_new)
        end

        m_old = m_new
    end

    return m_new
end
