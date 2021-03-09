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

    a = shape(left)
    b = rate(left)
    p = right.p
    g = right.γ
    λ = -0.1
    prior(x) = ((a .- 1) .* log.(x) .- b .* x .+ a .* log.(b) .- loggamma.(a))
    multip(x) = (p .* x .- g*loggamma.(x) .+ (a .- 1) .* log.(x) .- b .* x .+ a .* log.(b) .- loggamma.(a) ) + 2/λ*((x .- 0.001)^2)- λ*(log(0.001))

    d(x) = ForwardDiff.derivative(multip,x)
    dprime(x) = ForwardDiff.derivative(d,x)
    grad(x) = d(x)/dprime(x)
    x0 = gradientOptimization(multip,grad,5.0,0.001)

    η = -0.00001
    multip2(x) = (-g*trigamma.(x0) .- (x .- 1) .* 1/x0^2 )
    approximate_log_partition(x) =  2/η*((x[1] .- 0.001)^2 .+ (x[2] .- 0.001)^2)- η*(log(0.001) + log(0.001)) .+ 0.5*log(2*pi) - 0.5*log(abs(multip2(x[1])))+(p .* x0 .- g*loggamma.(x0) .+ (x[1] .- 1) .* log.(x0) .- x[2] .* x0 .+ x[1] .* log.(x[2]) .- loggamma.(x[1]) )
    dlog(x) = ForwardDiff.gradient(approximate_log_partition,x)
    ddlog(x) = ForwardDiff.hessian(approximate_log_partition,x)
    direction(x) = inv(ddlog(x))*dlog(x)
    est = gradientOptimization(approximate_log_partition,direction,[a,b],0.1)
    return GammaShapeRate(est[1], est[2])
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

using DataStructures
function gradientOptimization(log_joint::Function, d_log_joint::Function, m_initial, step_size)
    dim_tot = length(m_initial)
    m_total = zeros(dim_tot)
    m_average = zeros(dim_tot)
    m_new = zeros(dim_tot)
    m_old = m_initial
    satisfied = false
    step_count = 0
    m_latests = if dim_tot == 1 Queue{Float64}() else Queue{Vector}() end
    while !satisfied
        m_new = m_old .+ step_size.*d_log_joint(m_old)
        # @show m_new,d_log_joint(m_old)
        if log_joint(m_new) > log_joint(m_old)
            proposal_step_size = 10*step_size
            m_proposal = m_old .+ proposal_step_size.*d_log_joint(m_old)
            if log_joint(m_proposal) > log_joint(m_new)
                m_new = m_proposal
                step_size = proposal_step_size
            end
        else
            step_size = 0.1*step_size
            m_new = m_old .+ step_size.*d_log_joint(m_old)
        end
        step_count += 1
        enqueue!(m_latests, m_old)
        if step_count > 10
            m_average = sum(x for x in m_latests)./10
            if sum(sqrt.(((m_new.-m_average)./m_average).^2)) < dim_tot*0.1
                satisfied = true
            end
            dequeue!(m_latests);
        end
        if step_count > dim_tot*250
            satisfied = true
        end
        m_old = m_new
    end

    return m_new
end
