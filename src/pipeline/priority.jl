export PriorityPipelineStage

import Rocket: PriorityHandler

"""
    PriorityPipelineStage <: AbstractPipelineStage

Applies the `prioritize()` operator from `Rocket.jl` library to the given pipeline using name of variables as a label.

See also: [`AbstractPipelineStage`](@ref), [`apply_pipeline_stage`](@ref), [`EmptyPipelineStage`](@ref), [`CompositePipelineStage`](@ref)
"""
struct PriorityPipelineStage <: AbstractPipelineStage 
    handler :: PriorityHandler
end

function apply_pipeline_stage(stage::PriorityPipelineStage, factornode, ::Type{Val{T}}, stream) where T
    interface = getinterface(factornode, T)
    variable  = connectedvar(interface)
    if israndom(variable)
        return stream |> prioritize(stage.handler, name(variable))
    else
        return stream 
    end
end