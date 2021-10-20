export GCV, GCVMetadata

import StatsFuns: log2π

const UniNormalOrExpLinQuad = Union{UnivariateGaussianDistributionsFamily, ExponentialLinearQuadratic}

struct GCVMetadata{ A <: AbstractApproximationMethod }
    approximation :: A
end

get_approximation(meta::GCVMetadata) = meta.approximation

struct GCV end

@node GCV Stochastic [ y, x, z, κ, ω ]

const DefaultGCVNodeMetadata = GCVMetadata(GaussHermiteCubature(101))

default_meta(::Type{ GCV }) = DefaultGCVNodeMetadata

function approximate_expectation(method::AbstractApproximationMethod, g::Function, m, v) 
    weights = ReactiveMP.getweights(method, m, v)
    points  = ReactiveMP.getpoints(method, m, v)

    cs   = Vector{eltype(m)}(undef, length(weights))
    norm = 0.0
    mean = 0.0

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv

        norm += cv

        @inbounds cs[index] = cv
    end

    return norm
end

function approximate_expectation(method::AbstractApproximationMethod, g::Function, distribution)
    return approximate_expectation(method, g, mean(distribution), cov(distribution))
end


function divide_marginals(marginal::UnivariateNormalDistributionsFamily, message::UnivariateNormalDistributionsFamily)

    mz, vz = mean_cov(message)
    mpz, vpz = mean_cov(marginal)

    vpz = clamp(vpz, tiny, vz)
    wz_out = clamp(1/vpz - 1/vz, tiny, huge)
    ξz_out = mpz/vpz - mz/vz

    return NormalWeightedMeanPrecision(ξz_out, wz_out)
end

@average_energy GCV (q_y_x::MultivariateNormalDistributionsFamily, q_z::NormalDistributionsFamily, q_κ::Any, q_ω::Any) = begin
    m, c = mean(q_y_x), cov(q_y_x)

    ksi = (mean(q_κ) ^ 2) * var(q_z) + (mean(q_z) ^ 2) * var(q_κ) + var(q_κ) * var(q_z)
    psi = (m[2] - m[1]) ^ 2 + c[1, 1] + c[2, 2] - c[1, 2] - c[2, 1]
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

    0.5 * log2π + 0.5 * (mean(q_z) * mean(q_κ) + mean(q_ω)) + 0.5 * (psi * A * B)
end


@average_energy GCV (q_y_x::MultivariateNormalDistributionsFamily, q_z_κ::Any, q_ω::Any) = begin
    m, c = mean(q_y_x), cov(q_y_x)

   
    psi = (m[2] - m[1]) ^ 2 + c[1, 1] + c[2, 2] - c[1, 2] - c[2, 1]
    

    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]), q_z_κ)
    tmp2 = approximate_expectation(get_approximation(meta), x -> exp(-x) , q_ω)
    tmp3 = approximate_expectation(get_approximation(meta), x -> x[1]*x[2],q_z_κ)

    0.5 * log2π + 0.5 * (tmp3+ mean(q_ω)) + 0.5 * (psi * tmp1 * tmp2)
end

@average_energy GCV (q_y_x::MultivariateNormalDistributionsFamily, q_z_κ::Any, q_ω::Any) = begin
    m, c = mean(q_y_x), cov(q_y_x)

   
    psi = (m[2] - m[1]) ^ 2 + c[1, 1] + c[2, 2] - c[1, 2] - c[2, 1]
    

    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]-x[3]), q_z_κ_ω)
    tmp2 = approximate_expectation(get_approximation(meta), x -> x[1]*x[2]+x[3],q_z_κ)

    0.5 * log2π + 0.5 * tmp2 + 0.5 * (psi * tmp1 )
end

@average_energy GCV (q_y_x::MultivariateNormalDistributionsFamily, q_z_κ_ω::Any) = begin
    m, c = mean(q_y_x), cov(q_y_x)

   
    psi = (m[2] - m[1]) ^ 2 + c[1, 1] + c[2, 2] - c[1, 2] - c[2, 1]
    

    tmp1 = approximate_expectation(get_approximation(meta), x -> exp(-x[1]*x[2]-x[3]), q_z_κ_ω)

    0.5 * log2π + 0.5 * (psi * tmp1 )
end

@average_energy GCV (q_y::NormalDistributionsFamily,q_x::NormalDistributionsFamily, q_z::NormalDistributionsFamily, q_κ::Any, q_ω::Any) = begin
    my,vy = mean(q_y), cov(q_y)
    mx,vx = mean(q_x), cov(q_x)

    ksi = (mean(q_κ) ^ 2) * var(q_z) + (mean(q_z) ^ 2) * var(q_κ) + var(q_κ) * var(q_z)
    psi = (my - mx) ^ 2 + vy+vx
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

    0.5 * log2π + 0.5 * (mean(q_z) * mean(q_κ) + mean(q_ω)) + 0.5 * (psi * A * B)
end