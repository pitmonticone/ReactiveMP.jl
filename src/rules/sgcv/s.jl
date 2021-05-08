export rule

@rule SGCV(:s, Marginalisation) (q_y_x::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::SGCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)
    mω, vω = mean(q_ω), cov(q_ω)
    mz, vz = mean(q_z), cov(q_z)
    mκ, vκ = mean(q_κ), cov(q_κ)


    
    A = exp.(-mω+0.5diag(vω))
    B = exp.(-mκ .* mz .+ 0.5(mκ.^2*vz .+ mz^2 .* diag(vκ) + diag(vκ) .* vz))
    r = exp.(-0.5.*(mκ .* mz .+ mω + ψ(q_y_x) .* A .* B))
    ρ = clamp.(softmax(r), tiny, 1.0 - tiny)
    ρ = ρ ./ sum(ρ)

    return Categorical(ρ )

    
end
@rule SGCV(:s, Marginalisation) (q_y::Any,q_x::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::SGCVMetadata) = begin
    
    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)
    mω, vω = mean(q_ω), cov(q_ω)
    mz, vz = mean(q_z), cov(q_z)
    mκ, vκ = mean(q_κ), cov(q_κ)

    psi = (my - mx) ^ 2 + vy + vx
    
    A = exp.(-mω+0.5diag(vω))
    B = exp.(-mκ .* mz .+ 0.5(mκ.^2*vz .+ mz^2 .* diag(vκ) + diag(vκ) .* vz))
    r = exp.(-0.5.*(mκ .* mz .+ mω + psi .* A .* B))
    ρ = clamp.(softmax(r), tiny, 1.0 - tiny)
    ρ = ρ ./ sum(ρ)

    return Categorical(ρ )

    
end
