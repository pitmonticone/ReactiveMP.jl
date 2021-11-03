export VariableBoundEntropy

struct VariableBoundEntropy end

function score(::Type{T}, objective::BetheFreeEnergy, ::VariableBoundEntropy, variable::RandomVariable, scheduler) where { T <: InfCountingReal }
    mapping = let d = degree(variable)
        (m) -> convert(promote_type(T, eltype(m)), (d - 1) * score(DifferentialEntropy(), m))
    end
    return getmarginal(variable, marginal_skip_strategy(objective)) |> schedule_on(scheduler) |> map(T, mapping)
end