export rule

@rule SGCV(:κ, Marginalisation) (q_y_x::Any, q_z::Any, q_ω::Any, q_s::Any, meta::SGCVMetadata) = begin
    
    m, v = mean(q_y_x), cov(q_y_x)
    mz, vz = mean(q_z), cov(q_z)
    mω, vω = mean(q_ω), cov(q_ω)
    ms = probvec(q_s)

    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = mz .* ms
    b = psi .* ms 
    c = -mz
    d = vz

    return MvExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end


@rule SGCV(:κ, Marginalisation) (q_y::Any,q_x::Any, q_z::Any, q_ω::Any, q_s::Any, meta::SGCVMetadata) = begin
    
    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)
    mz, vz = mean(q_z), cov(q_z)
    mω, vω = mean(q_ω), cov(q_ω)
    ms = probvec(q_s)

    psi = (my - mx) ^ 2 + vy + vx

    a = mz .* ms
    b = psi .* ms 
    c = -mz
    d = vz

    return MvExponentialLinearQuadratic(get_approximation(meta),a, b, c, d)
end