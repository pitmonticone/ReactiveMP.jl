export rule

@rule SGCV(:y, Marginalisation) (m_x::Any, q_z::Any, q_κ::Any, q_ω::Any,q_s::Any) = begin
    return NormalMeanVariance(mean(m_x), var(m_x) + 1/η(q_z,q_κ,q_ω,q_s))
end

function η(z, κ, ω, s)
    ms = probvec(s)
    mω, Vω = mean(ω), cov(ω)
    mz, vz = mean(z), cov(z)
    mκ, Vκ = mean(κ), cov(κ)

    return ms'*(exp.(-mκ*mz .- mω + 0.5((mκ).^2 .* vz .+ mz^2 .*diag(Vκ) .+ diag(Vκ).*vz .+ diag(Vω))))
end