export rule

@rule NormalMixture{2}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{2, NormalMeanVariance}, q_p::NTuple{2, Gamma}) = begin
    # TODO check
    U1 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[1], q_p[1])), nothing)
    U2 = -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[2], q_p[2])), nothing)
    return Bernoulli(clamp(softmax((U1, U2))[1], tiny, 1.0 - tiny))
end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{N, NormalMeanVariance}, q_p::NTuple{N, Gamma}) where { N } = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, m, p)), nothing)
    end
    return Categorical(softmax(U))
end

@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::NTuple{N, MvNormalMeanCovariance}, q_p::NTuple{N, Wishart}) where { N } = begin
    U = map(zip(q_m, q_p)) do (m, p)
        return -score(AverageEnergy(), MvNormalMeanPrecision, Val{ (:out, :μ, :Λ) }, map(as_marginal, (q_out, m, p)), nothing)
    end
    return Categorical(softmax(U))
end