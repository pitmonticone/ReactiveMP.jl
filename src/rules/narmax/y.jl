export rule

@rule NARMAX(:y, Marginalisation) (q_θ::NormalDistributionsFamily, 
                                   q_x::PointMass, q_u::PointMass, 
                                   q_z::PointMass, q_r::PointMass, 
                                   q_τ::GammaShapeRate, meta::NARMAXMeta) = begin

    mθ, mu, mx, mz, mr, mτ = mean.((q_θ, q_x, q_u, q_z, q_r, q_τ))

    ϕm = meta.ϕ([mu; mx; mz; mr])
    
    return NormalMeanPrecision(mθ'ϕm, mτ)
end
