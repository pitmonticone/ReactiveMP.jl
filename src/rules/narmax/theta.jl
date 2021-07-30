export rule

@rule NARMAX(:θ, Marginalisation) (q_y::Union{PointMass, UnivariateNormalDistributionsFamily}, 
                                   q_x::PointMass, q_u::PointMass, 
                                   q_z::PointMass, q_r::PointMass, 
                                   q_τ::GammaDistributionsFamily, meta::NARMAXMeta) = begin

    my, mu, mx, mz, mr, mτ = mean.((q_y, q_x, q_u, q_z, q_r, q_τ))

    ϕm = meta.ϕ([mu; mx; mz; mr])
    n_coef = size(ϕm, 1)
    
    Ψ = mτ*correction!(meta.correction, ϕm*ϕm')
	ψ = mτ*my*ϕm

    return MvNormalWeightedMeanPrecision(ψ, Ψ)
end
