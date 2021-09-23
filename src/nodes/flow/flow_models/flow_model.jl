## TODO: create a forward-jacobian joint function that calculates both at the same time

export FlowModel, CompiledFlowModel
export compile, nr_params
export getlayers, getforward, getbackward, getjacobian, getinv_jacobian
export forward, forward!, backward, backward!
export jacobian, jacobian!, inv_jacobian, inv_jacobian!
export det_jacobian, absdet_jacobian, logdet_jacobian, logabsdet_jacobian
export detinv_jacobian, absdetinv_jacobian, logdetinv_jacobian, logabsdetinv_jacobian

import Base: length, eltype

@doc raw"""
The FlowModel structure is the most generic type of Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains the input dimensionality and a tuple of layers and can be constructed as `FlowModel( dim, (layer1, layer2, ...) )`.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct FlowModel{N,T<:NTuple{N,AbstractLayer}} <: AbstractFlowModel
    dim     :: Int
    layers  :: T
end

@doc raw"""
Creates a flow model with uninitialized parameters.

Input arguments:
- `dim::Int` - input dimensionality.
- `layers<:NTuple{N,AbstractLayer}` - arbitrarily sized tuple containing abstractlayers.

Return arguments:
- `::Flowmodel` - model containing layers of which are appropriately sized according to the input dimensionality.
"""
function FlowModel(dim::Int, layers::T) where { T <: NTuple{N,AbstractLayerPlaceholder} where { N } }
    return FlowModel(dim, flatten_tuple(prepare.(dim, layers)))
end
function FlowModel(layers::T) where { T <: NTuple{N, Union{AbstractLayer, AbstractLayerPlaceholder}} where { N } }
    @assert typeof(first(layers)) <: InputLayer "The FlowModel requires an input dimension to be specified. This can be achieved, either by preceding the layers tuple with an integer as `FlowModel(dim, layers)`, or by starting the tuple of layers with an `InputLayer(dim)` as `FlowModel((InputLayer(dim), layers...))`."
    return FlowModel(getdim(first(layers)), flatten_tuple(prepare.(getdim(first(layers)), Base.tail(layers))))
end

# prepare function for setting correct sizes in the layers (without assigning the parameters yet!)
prepare(dim::Int, layers::T) where { T <: NTuple{N,Union{AbstractLayer, AbstractLayerPlaceholder}} where { N }} = _prepare(dim, layers)
Broadcast.broadcasted(::typeof(prepare), dim::Int, layers::T) where { T <: NTuple{N,Union{AbstractLayer, AbstractLayerPlaceholder}} where { N }} = broadcast(_prepare, Ref(dim), layers)

@doc raw"""
The CompiledFlowModel structure is the most generic type of compiled Flow model, in which the layers are not constrained to be of a specific type. The FlowModel structure contains the input dimension and a tuple of compiled layers. Do not manually create a CompiledFlowModel! Instead create a FlowModel first and compile it with `compile(model::FlowModel)`. This will make sure that all layers/mappings are configured with the proper dimensionality and with randomly sampled parameters. Alternatively, if you would like to pass your own parameters, call `compile(model::FlowModel, params::Vector)`.

Note: this model can be specialized by constraining the types of layers. This potentially allows for more efficient specialized methods that can deal with specifics of these layers, such as triangular jacobian matrices.
"""
struct CompiledFlowModel{N,T<:NTuple{N,AbstractLayer}} <: AbstractCompiledFlowModel
    dim     :: Int
    layers  :: T
end

@doc raw"""
`compile()` compiles a model by setting its parameters. It randomly sets parameter values in the layers and flows such that inference in the model can be obtained.

Input arguments
- `model::FlowModel` - a model of which the dimensionality of its layers/flows has been initialized, but its parameters have not been set.

Return arguments
- `::CompiledFlowModel` - a compiled model with set parameters, such that it can be used for processing data.
"""
function compile(model::FlowModel)
    # do not create parameters here for decentralized initializers (TODO)
    return CompiledFlowModel(model.dim, compile.(model.layers))
end

@doc raw"""
`compile(model::FlowModel, params::Vector)` lets you initialize a model `model` with a vector of parameters `params`.

Input arguments
- `model::FlowModel` - a model of which the dimensionality of its layers/flows has been initialized, but its parameters have not been set.
- `params::Vector`   - a vector of parameters with which the model should be compiled.

Return arguments
- `::CompiledFlowModel` - a compiled model with set parameters, such that it can be used for processing data.
"""
function compile(model::FlowModel, params::Vector)
    
    # assert whether the parameter vector complies with the model
    @assert nr_params(model) == length(params) "The number of parameters in the model does not match the passed number of parameters."
    
    # compile layers
    return CompiledFlowModel(getdim(model), compile(getlayers(model), params))

end
function compile(x::Tuple, params::Vector)

    # fetch number of parameters in layer
    params_in_layer = nr_params(first(x))

    if params_in_layer > 0
        # if layer has parameters, compile that layer and the other layers with parameters
        return (compile(first(x), params[1:params_in_layer]), compile(Base.tail(x), params[params_in_layer+1:end])...)
    else
        # if layer has no parameters, compile that layer on its own and the other layers with parameters
        return (compile(first(x)), compile(Base.tail(x), params)...)
    end
    
end
function compile(x::Tuple)
    return (compile(first(x)), compile(Base.tail(x))...)
end
# compile functions to stop the recursion
compile(x::Tuple{}, params::Vector) = ()
compile(x::Tuple{})         = ()

# get-functions for the FlowModel structure
getdim(model::FlowModel)                    = model.dim
getdim(model::CompiledFlowModel)            = model.dim
getlayers(model::FlowModel)                 = model.layers
getlayers(model::CompiledFlowModel)         = model.layers
getforward(model::CompiledFlowModel)        = (x) -> forward(model, x)
getbackward(model::CompiledFlowModel)       = (x) -> backward(model, x)
getjacobian(model::CompiledFlowModel)       = (x) -> jacobian(model, x)
getinv_jacobian(model::CompiledFlowModel)   = (x) -> inv_jacobian(model, x)

# custom Base function for the FlowModel structure
eltype(model::CompiledFlowModel) = promote_type(map(eltype, model.layers)...)
length(model::CompiledFlowModel) = length(model.layers)

# fetch number of parameters of model
nr_params(model::FlowModel)         = mapreduce(nr_params, +, model.layers)
nr_params(model::CompiledFlowModel) = mapreduce(nr_params, +, model.layers)


# forward pass through the Flow model
function _forward(model::CompiledFlowModel, input::Vector{T1}) where { T1 <: Real }

    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for result
    output = zeros(T, size(input))

    # perform calculations 
    forward!(output, model, input)

    # return result
    return output
    
end

# when calling forward, redirect to _forward
forward(model::CompiledFlowModel, input::Vector{T}) where { T <: Real } = _forward(model, input)

# for broadcasting over forward, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(forward), model::CompiledFlowModel, input::Vector{Vector{T}}) where { T <: Real } = broadcast(_forward, Ref(model), input)

# inplace forward pass through the Flow model
function forward!(output::Vector{T1}, model::CompiledFlowModel, input::Vector{T2}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # decouple changing input from actual input
    input_new = zeros(T, size(input))
    input_new .= input

    # perform forward pass over all layers
    forward!(output, layers, input_new)
    
end

# inplace forward pass through a tuple of layers
function forward!(output::Vector{T1}, layers::T, input::Vector{T2}) where { T1 <: Real, T2 <: Real,  T <: NTuple{N,AbstractLayer} where N}

    # perform pass over first layer
    forward!(output, first(layers), input)

    # update intermediate input
    input .= output

    # perform pass over all remaining layers
    forward!(output, Base.tail(layers), input)

end

# when no layers are left, stop the inplace recursion
forward!(output::Vector{T1}, layers::Tuple{}, input::Vector{T2}) where { T1 <: Real, T2 <: Real } = return


# backward pass through the Flow model
function _backward(model::CompiledFlowModel, output::Vector{T1}) where { T1 <: Real }

    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for result
    input = zeros(T, size(output))

    # perform calculations 
    backward!(input, model, output)

    # return result
    return input
    
end

# when calling backward, redirect to _backward
backward(model::CompiledFlowModel, output::Vector{T}) where { T <: Real } = _backward(model, output)

# for broadcasting over backward, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(backward), model::CompiledFlowModel, output::Vector{Vector{T}}) where { T <: Real } = broadcast(_backward, Ref(model), output)

# inplace backward pass through the Flow model
function backward!(input::Vector{T1}, model::CompiledFlowModel, output::Vector{T2}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)

    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # decouple changing output from actual output
    output_new = zeros(T, size(output))
    output_new .= output

    # perform backward pass over all layers
    backward!(input, reverse(layers), output_new)
    
end

# inplace backward pass through a (reversed!) tuple of layers
function backward!(input::Vector{T1}, layers::T, output::Vector{T2}) where { T1 <: Real, T2 <: Real,  T <: NTuple{N,AbstractLayer} where N}

    # perform pass over last layer (but reversed!)
    backward!(input, first(layers), output)

    # update intermediate output
    output .= input

    # perform pass over all remaining layers
    backward!(input, Base.tail(layers), output)

end

# when no layers are left, stop the inplace recursion
backward!(input::Vector{T1}, layers::Tuple{}, output::Vector{T2}) where { T1 <: Real, T2 <: Real } = return


# jacobian of the Flow model
function _jacobian(model::CompiledFlowModel, input::Vector{T1}) where { T1 <: Real }

    # fetch layers
    dim    = getdim(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for jacobian
    J = zeros(T, dim, dim)
    for k = 1:dim
        J[k,k] = 1.0
    end

    # calculate jacobian
    jacobian!(J, model, input)

    # return result
    return J

end

# when calling jacobian, redirect to _jacobian
jacobian(model::CompiledFlowModel, input::Vector{T}) where { T <: Real } = _jacobian(model, input)

# for broadcasting over jacobian, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(jacobian), model::CompiledFlowModel, input::Vector{Vector{T}}) where { T <: Real } = broadcast(_jacobian, Ref(model), input)

# inplace jacobian of the Flow model
function jacobian!(J::Matrix{T1}, model::CompiledFlowModel, input::Vector{T2}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)
    dim = getdim(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # allocate space for intermediate output and jacobian
    input_new = zeros(T, dim)
    input_new .= input
    output = zeros(T, dim)
    J_new = zeros(T, dim, dim)
    J_old = zeros(T, dim, dim)
    for k = 1:dim
        J_old[k,k] = 1.0
    end

    # calculate jacobian over all layers
    jacobian!(J, J_new, J_old, output, input_new, layers)

end

# inplace jacobian calculation for a tuple of layers
function jacobian!(J::Matrix{T1}, J_new::Matrix{T2}, J_old::Matrix{T3}, output::Vector{T4}, input_new::Vector{T5}, layers::T) where { T1 <: Real, T2 <: Real,  T3 <: Real, T4 <: Real, T5 <: Real, T <: NTuple{N,AbstractLayer} where N}

    # check for specialized jacobians that are fixed (e.g. Permutation Matrix)
    if typeof(first(layers)) <: PermutationLayer
        
        # calculate total jacobian
        mul!(J, jacobian(first(layers), input_new), J_old)
        
        # else fall back to standard calculation
    else 
        
        # calculate new jacobian
        jacobian!(J_new, first(layers), input_new)
        
        # calculate total jacobian
        mul!(J, J_new, J_old)
        
    end

    # update old jacobian
    J_old .= J

    # if we did not arrive at the last layer yet, then also perform a forward pass
    if length(layers) > 1

        # perform forward pass over last layer
        forward!(output, first(layers), input_new)

        # update intermediate output
        input_new .= output

    end

    # calculate jacobian of remaining layers
    jacobian!(J, J_new, J_old, output, input_new, Base.tail(layers))

end

# when no layers are left, stop the inplace recursion
jacobian!(J::Matrix{T1}, J_new::Matrix{T2}, J_old::Matrix{T3}, output::Vector{T4}, input_new::Vector{T5}, layers::Tuple{}) where { T1 <: Real, T2 <: Real,  T3 <: Real, T4 <: Real, T5 <: Real } = return


# inverse jacobian of the Flow model
function _inv_jacobian(model::CompiledFlowModel, output::Vector{T1}) where { T1 <: Real }

    # fetch layers
    dim    = getdim(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T1)

    # allocate space for jacobian
    J = zeros(T, dim, dim)
    for k = 1:dim
        J[k,k] = 1.0
    end

    # calculate jacobian
    inv_jacobian!(J, model, output)

    # return result
    return J

end

# when calling inverse jacobian, redirect to _inv_jacobian
inv_jacobian(model::CompiledFlowModel, output::Vector{T}) where { T <: Real } = _inv_jacobian(model, output)

# for broadcasting over inverse jacobian, fix the model for multiple inputs
Broadcast.broadcasted(::typeof(inv_jacobian), model::CompiledFlowModel, output::Vector{Vector{T}}) where { T <: Real } = broadcast(_inv_jacobian, Ref(model), output)

# inplace inverse jacobian of the Flow model
function inv_jacobian!(J::Matrix{T1}, model::CompiledFlowModel, output::Vector{T2}) where { T1 <: Real, T2 <: Real }

    # fetch layers
    layers = getlayers(model)
    dim = getdim(model)
    
    # promote type for allocating output
    T = promote_type(eltype(model), T2)

    # allocate space for intermediate output and jacobian
    output_new = zeros(T, dim)
    output_new .= output
    input = zeros(T, dim)
    J_new = zeros(T, dim, dim)
    J_old = zeros(T, dim, dim)
    for k = 1:dim
        J_old[k,k] = 1.0
    end

    # calculate inverse jacobian over all layers (reversed!)
    inv_jacobian!(J, J_new, J_old, input, output_new, reverse(layers))

end

# inplace inverse jacobian calculation for a tuple of layers
function inv_jacobian!(J::Matrix{T1}, J_new::Matrix{T2}, J_old::Matrix{T3}, input::Vector{T4}, output_new::Vector{T5}, layers::T) where { T1 <: Real, T2 <: Real,  T3 <: Real, T4 <: Real, T5 <: Real, T <: NTuple{N,AbstractLayer} where N}

    # check for specialized jacobians that are fixed (e.g. Permutation Matrix)
    if typeof(first(layers)) <: PermutationLayer
        
        # calculate total jacobian
        mul!(J, inv_jacobian(first(layers), output_new), J_old)
        
    # else fall back to standard calculation
    else 
        
        # calculate new jacobian
        inv_jacobian!(J_new, first(layers), output_new)
        
        # calculate total jacobian
        mul!(J, J_new, J_old)
        
    end

    # update old jacobian
    J_old .= J

    # if we did not arrive at the last layer yet, then also perform a forward pass
    if length(layers) > 1

        # perform forward pass over last layer
        backward!(input, first(layers), output_new)

        # update intermediate output
        output_new .= input

    end

    # calculate jacobian of remaining layers
    inv_jacobian!(J, J_new, J_old, input, output_new, Base.tail(layers))

end

# when no layers are left, stop the inplace recursion
inv_jacobian!(J::Matrix{T1}, J_new::Matrix{T2}, J_old::Matrix{T3}, input::Vector{T4}, output_new::Vector{T5}, layers::Tuple{}) where { T1 <: Real, T2 <: Real,  T3 <: Real, T4 <: Real, T5 <: Real } = return


# extra utility functions
det_jacobian(model::CompiledFlowModel, input::Vector{T})           where { T <: Real} = det(jacobian(model, input))
absdet_jacobian(model::CompiledFlowModel, input::Vector{T})        where { T <: Real} = abs(det_jacobian(model, input))
logdet_jacobian(model::CompiledFlowModel, input::Vector{T})        where { T <: Real} = logdet(jacobian(model, input))
logabsdet_jacobian(model::CompiledFlowModel, input::Vector{T})     where { T <: Real} = logabsdet(jacobian(model, input))

detinv_jacobian(model::CompiledFlowModel, output::Vector{T})       where { T <: Real} = det(inv_jacobian(model, output))
absdetinv_jacobian(model::CompiledFlowModel, output::Vector{T})    where { T <: Real} = abs(detinv_jacobian(model, output))
logdetinv_jacobian(model::CompiledFlowModel, output::Vector{T})    where { T <: Real} = logdet(inv_jacobian(model, output))
logabsdetinv_jacobian(model::CompiledFlowModel, output::Vector{T}) where { T <: Real} = logabsdet(inv_jacobian(model, output))


# throw an error when the model has not yet been compiled
_forward(model::FlowModel, input::Vector{T1}) where { T1 <: Real }                                          = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
forward(model::FlowModel, input::Vector{T}) where { T <: Real }                                             = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(forward), model::FlowModel, input::Vector{Vector{T}}) where { T <: Real }    = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
forward!(output::Vector{T1}, model::FlowModel, input::Vector{T2}) where { T1 <: Real, T2 <: Real }          = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_backward(model::FlowModel, output::Vector{T1}) where { T1 <: Real }                                        = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
backward(model::FlowModel, output::Vector{T}) where { T <: Real }                                           = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(backward), model::FlowModel, output::Vector{Vector{T}}) where { T <: Real }  = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
backward!(input::Vector{T1}, model::FlowModel, output::Vector{T2}) where { T1 <: Real, T2 <: Real }         = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
jacobian!(J_new::Matrix{T1}, model::FlowModel, input::Vector{T2}) where { T1 <: Real, T2 <: Real }          = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
jacobian(model::FlowModel, input::Vector{T}) where { T <: Real }                                            = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
Broadcast.broadcasted(::typeof(jacobian), model::FlowModel, input::Vector{Vector{T}}) where { T <: Real }   = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_jacobian(model::FlowModel, input::Vector{T1}) where { T1 <: Real }                                         = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
_inv_jacobian(model::FlowModel, output::Vector{T1}) where { T1 <: Real }                                    = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
inv_jacobian!(J_new::Matrix{T1}, model::FlowModel, output::Vector{T2}) where { T1 <: Real, T2 <: Real }     = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))

det_jacobian(model::FlowModel, input::Vector{T})           where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
absdet_jacobian(model::FlowModel, input::Vector{T})        where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logdet_jacobian(model::FlowModel, input::Vector{T})        where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logabsdet_jacobian(model::FlowModel, input::Vector{T})     where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))

detinv_jacobian(model::FlowModel, output::Vector{T})       where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
absdetinv_jacobian(model::FlowModel, output::Vector{T})    where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logdetinv_jacobian(model::FlowModel, output::Vector{T})    where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))
logabsdetinv_jacobian(model::FlowModel, output::Vector{T}) where { T <: Real} = throw(ArgumentError("Please first compile your model using `compiled_model = compile(model)`."))