export rule

@rule SGCV(:ω, Marginalisation) (q_y_x::Any, q_z::Any, q_κ::Any, q_s::Any, meta::GCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)
    mz, vz = mean(q_z), cov(q_z)
    mκ, vκ = mean(q_κ), cov(q_κ)
    ms = mean(q_s)

    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = ms
    b = psi .* ms 
    c = -1.0
    d = 0.0

    return MvExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end