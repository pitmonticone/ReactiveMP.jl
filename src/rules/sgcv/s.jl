export rule

@rule SGCV(:s, Marginalisation) (q_y_x::Any, q_z::Any, q_κ::Any, q_ω::Any, meta::SGCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)
    mω, vω = mean(q_ω), cov(q_ω)
    mz, vz = mean(q_z), cov(q_z)
    mκ, vκ = mean(q_κ), cov(q_κ)


    A = exp.(-mω+0.5diag(vω))
    B = exp.(-mκ .* mz .+ 0.5(mκ.^2*vz .+ mz^2 .* diag(vκ) + diag(vκ) .* vz))
    r = exp.(-0.5.*(mκ .* mz .+ mω + ψ1(q_y_x) .* A .* B))
    ρ = r ./ sum(r)
    return Categorical( ρ)

    
end

function ψ1(yx)
    m, V = mean(yx),cov(yx)
    return (m[1] - m[2])*(m[1] - m[2]) + V[1] + V[4] - V[3] - V[2]
end