export rule
                        
@rule NARMAX(:τ, Marginalisation) (q_y::PointMass, 
                                   q_θ::NormalDistributionsFamily, 
                                   q_x::PointMass, q_u::PointMass, 
                                   q_z::PointMass, q_r::PointMass, 
                                   meta::NARMAXMeta) = begin

    
    mθ, Vθ = mean_cov(q_θ)

    my, mx, mu, mz, mr = mean.((q_y, q_x, q_u, q_z, q_r))

    ϕm = meta.ϕ([mu; mx; mz; mr])
    
    α = 3/2.
	β = (my^2 - 2*my*(mθ'*ϕm) + (mθ'*ϕm)^2 + ϕm'*Vθ*ϕm)/2.

    return GammaShapeRate(α, β)
end