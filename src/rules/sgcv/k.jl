export rule

@rule SGCV(:κ, Marginalisation) (q_y_x::Any, q_z::Any, q_ω::Any, q_s::Any, meta::GCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)
    mz, vz = mean(q_z), cov(q_z)
    mω, vω = mean(q_ω), cov(q_ω)
    ms = mean(q_s)

    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = mz .* ms
    b = psi .* ms 
    c = -mz
    d = vz

    return MvExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end