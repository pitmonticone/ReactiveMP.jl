export rule

@rule AR(:x, Marginalisation) (m_y::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta) = begin
    mθ, Vθ = mean_cov(q_θ)
    my, Vy = mean_cov(m_y)

    mγ = mean(q_γ)

    mA = as_companion_matrix(mθ)
    mV = ar_transition(getvform(meta), getorder(meta), mγ)

    D = mA'*inv(Vy + mV)*mA + mγ*Vθ
    Vx = inv(D)
    mx = inv(D)*mA'*inv(Vy + mV)*my

    return convert(promote_variate_type(getvform(meta), NormalMeanVariance), mx, Vx)
end
