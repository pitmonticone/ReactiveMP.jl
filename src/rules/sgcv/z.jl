export rule

@rule SGCV(:z, Marginalisation) (q_y_x::Any, q_κ::Any, q_ω::Any,q_s::Any, meta::SGCVMetadata) = begin

    m, v = mean(q_y_x), cov(q_y_x)
    mκ, Vκ = mean(q_κ), cov(q_κ)
    mω, Vω = mean(q_ω), cov(q_ω)
    ms = probvec(q_s)


    psi = (m[1] - m[2]) ^ 2 + v[1, 1] + v[2, 2] - v[1, 2] - v[2, 1]

    a = ms' * mκ
    b = ms' * mω
    c = ms .* psi
    d = -mκ
    e = -mω + 0.5*diag(Vω)
    f = diag(Vκ)

    return MvUniExponentialLinearQuadratic(get_approximation(meta), a, b, c, d,e,f)
end

@rule SGCV(:z, Marginalisation) (q_y::Any,q_x::Any, q_κ::Any, q_ω::Any,q_s::Any, meta::SGCVMetadata) = begin

    my, vy = mean(q_y), cov(q_y)
    mx, vx = mean(q_x), cov(q_x)
    mκ, Vκ = mean(q_κ), cov(q_κ)
    mω, Vω = mean(q_ω), cov(q_ω)
    ms = probvec(q_s)


    psi = (my - mx) ^ 2 + vy + vx

    a = ms' * mκ
    b = ms' * mω
    c = ms .* psi
    d = -mκ
    e = -mω + 0.5*diag(Vω)
    f = diag(Vκ)

    return MvUniExponentialLinearQuadratic(get_approximation(meta), a, b, c, d,e,f)
end