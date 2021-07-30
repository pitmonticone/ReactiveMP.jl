
export NARMAX, NonlinearAutoregressiveMovingAverageX, NonlinearARMAX, NARMAXMeta

import StatsFuns: log2π

"""
Description:
    A Nonlinear AutoRegressive model with moving average and exogeneous input (NARMAX)
    y_k = fθ(u_k, …, u_k-M1, y_k-1, …, y_k-M2, , e_k, …, e_k-M3)
    where M1 is the number of input delays, M2 the number of output delays
    and M3 the number of error delays. These histories are stored as the following 
    vectors:
    - x_k-1 = [u_k-1, …, u_k-M1]
    - z_k-1 = [y_k-1, …, y_k-M2] 
    - r_k-1 = [e_k-1, …, e_k-M3]
    Assume u_k, x_k-1, z_k-1 and r_k-1 are observed and e_k ~ N(0, τ^-1).
    !! Currently, fθ is assumed to be a linear product of coefficients θ 
    and a basis expansion ϕ of inputs, outputs and errors: 
    
        fθ(...) = θ'*ϕ(u_k, …, u_k-M1, y_k-1, …, y_k-M2, , e_k, …, e_k-M3)
Interfaces:
    1. y (output)
    2. θ (function coefficients)
    3. u (current input)
    4. x (previous inputs)
    5. z (previous outputs)
    6. r (previous errors)
    7. τ (precision)
"""

struct NARMAXMeta{Basis, Gradient, Hessian, Correction}
    ϕ          :: Basis
    ∇ϕ         :: Gradient
    ∇²ϕ        :: Hessian
    correction :: Correction
end

function NARMAXMeta(ϕ, ∇ϕ, ∇²ϕ)
    return NARMAXMeta(ϕ, ∇ϕ, ∇²ϕ, TinyCorrection())
end

struct NARMAX end

const NonlinearAutoregressiveMovingAverageX = NARMAX
const NonlinearARMAX                        = NARMAX


@node NARMAX Stochastic [ y, θ, x, u, z, r, τ]


@average_energy NARMAX (q_y::PointMass, q_θ::NormalDistributionsFamily, 
                        q_x::PointMass, q_u::PointMass, 
                        q_z::PointMass, q_r::PointMass, 
                        q_τ::GammaDistributionsFamily, meta::NARMAXMeta) = begin

    mθ, Vθ = mean_cov(q_θ)
    my, mu, mx, mz, mr, mτ = mean.((q_y, q_x, q_u, q_z, q_r, q_τ))
    
    ϕ = meta.ϕ
    ϕm = ϕ([mu; mx; mz; mr])

    1/2*log2π - 1/2*logmean(q_τ) + 1/2*mτ*(my^2 - 2*my*(mθ'*ϕm) + (mθ'*ϕm)^2 + ϕm'*Vθ*ϕm)
end