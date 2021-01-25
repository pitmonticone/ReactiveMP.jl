export NormalMixture, NormalMixtureNode, score

# Normal Mixture Functional Form
struct NormalMixture{N} end

# Special node
# Generic FactorNode implementation does not work with dynamic number of inputs
# We need to reimplement the following set of functions
# functionalform(factornode::FactorNode)          
# sdtype(factornode::FactorNode)                 
# interfaces(factornode::FactorNode)              
# factorisation(factornode::FactorNode)           
# localmarginals(factornode::FactorNode)          
# localmarginalnames(factornode::FactorNode)      
# metadata(factornode::FactorNode)                
# outbound_message_portal(factornode::FactorNode)       
#
# setmarginal!(factornode::FactorNode, cname::Symbol, marginal)
# getmarginal!(factornode::FactorNode, localmarginal::FactorNodeLocalMarginal)
#
# functional_dependencies(factornode::FactorNode, iindex::Int)
# get_messages_observable(factornode, message_dependencies)
# get_marginals_observable(factornode, marginal_dependencies)
#
# score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::AbstractFactorNode, scheduler) where T
#
# Base.show
# 
const NormalMixtureNodeFactorisationSupport = Union{MeanField, }

struct NormalMixtureNode{N, F <: NormalMixtureNodeFactorisationSupport, M, P} <: AbstractFactorNode
    factorisation :: F
    
    # Interfaces
    out    :: NodeInterface
    switch :: NodeInterface
    means  :: NTuple{N, IndexedNodeInterface}
    precs  :: NTuple{N, IndexedNodeInterface}

    meta   :: M
    portal :: P
end

functionalform(factornode::NormalMixtureNode{N}) where N = NormalMixture{N}
sdtype(factornode::NormalMixtureNode)                    = Stochastic()           
interfaces(factornode::NormalMixtureNode)                = (factornode.out, factornode.switch, factornode.means..., factornode.precs...)
factorisation(factornode::NormalMixtureNode)             = factornode.factorisation       
localmarginals(factornode::NormalMixtureNode)            = error("localmarginals() function is not implemented for NormalMixtureNode")           
localmarginalnames(factornode::NormalMixtureNode)        = error("localmarginalnames() function is not implemented for NormalMixtureNode")     
metadata(factornode::NormalMixtureNode)                  = factornode.meta            
outbound_message_portal(factornode::NormalMixtureNode)   = factornode.portal   

setmarginal!(factornode::NormalMixtureNode, cname::Symbol, marginal)                = error("setmarginal() function is not implemented for NormalMixtureNode")           
getmarginal!(factornode::NormalMixtureNode, localmarginal::FactorNodeLocalMarginal) = error("getmarginal() function is not implemented for NormalMixtureNode")           

## activate!

function functional_dependencies(factornode::NormalMixtureNode{N, F}, iindex::Int) where { N, F <: MeanField }
    message_dependencies = ()

    marginal_dependencies = if iindex === 1
        (factornode.switch, factornode.means, factornode.precs)
    elseif iindex === 2
        (factornode.out, factornode.means, factornode.precs)
    elseif 2 < iindex <= N + 2
        (factornode.out, factornode.switch, tuple(skipindex(factornode.means, iindex - 2)...), factornode.precs)
    elseif N + 2 < iindex <= 2N + 2
        (factornode.out, factornode.switch, factornode.means, tuple(skipindex(factornode.precs, iindex - N - 2)...))
    else
        error("Bad index in functional_dependencies for NormalMixtureNode")
    end

    return message_dependencies, marginal_dependencies
end

function get_messages_observable(factornode::NormalMixtureNode{N, F}, message_dependencies::Tuple{}) where { N, F <: MeanField }
    return nothing, of(nothing)
end

function get_marginals_observable(
    factornode::NormalMixtureNode{N, F}, 
    marginal_dependencies::Tuple{ NodeInterface, NTuple{N, IndexedNodeInterface}, NTuple{N, IndexedNodeInterface} }) where { N, F <: MeanField }

    varinterface    = marginal_dependencies[1]
    meansinterfaces = marginal_dependencies[2]
    precsinterfaces = marginal_dependencies[3]

    marginal_names = Val{ (name(varinterface), name(meansinterfaces[1]), name(precsinterfaces[1])) }
    marginals_observable = combineSourceUpdates((
        getmarginal(connectedvar(varinterface)),
        combineLatest(map((prec) -> getmarginal(connectedvar(prec)), reverse(precsinterfaces)), PushNew()),
        combineLatest(map((mean) -> getmarginal(connectedvar(mean)), reverse(meansinterfaces)), PushNew()),
    ), PushNew())

    return marginal_names, marginals_observable
end

function get_marginals_observable(
    factornode::NormalMixtureNode{N, F}, 
    marginal_dependencies::Tuple{ NodeInterface, NodeInterface, NTuple{N1, IndexedNodeInterface}, NTuple{N2, IndexedNodeInterface} }) where { N, N1, N2, F <: MeanField }

    outinterface    = marginal_dependencies[1]
    switchinterface = marginal_dependencies[2]
    meansinterfaces = marginal_dependencies[3]
    precsinterfaces = marginal_dependencies[4]

    marginal_names = Val{ (name(outinterface), name(switchinterface), name(meansinterfaces[1]), name(precsinterfaces[1])) }

    marginals_observable = combineSourceUpdates((
        getmarginal(connectedvar(outinterface)),
        getmarginal(connectedvar(switchinterface)),
        combineLatest(map((prec) -> getmarginal(connectedvar(prec)), reverse(precsinterfaces)), PushNew()),
        combineLatest(map((mean) -> getmarginal(connectedvar(mean)), reverse(meansinterfaces)), PushNew()),
    ), PushNew())

    return marginal_names, marginals_observable
end

# FreeEnergy related functions

@average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::NTuple{N, NormalMeanVariance}, q_p::NTuple{N, Gamma}) where N = begin
    z_bar = probvec(q_switch)
    return mapreduce((i) -> z_bar[i] * score(AverageEnergy(), NormalMeanPrecision, Val{ (:out, :μ, :τ) }, map(as_marginal, (q_out, q_m[i], q_p[i])), nothing), +, 1:N, init = 0.0)
end

@average_energy NormalMixture (q_out::Any, q_switch::Any, q_m::NTuple{N, MvNormalMeanCovariance}, q_p::NTuple{N, Wishart}) where N = begin
    z_bar = probvec(q_switch)
    return mapreduce((i) -> z_bar[i] * score(AverageEnergy(), MvNormalMeanPrecision, Val{ (:out, :μ, :Λ) }, map(as_marginal, (q_out, q_m[i], q_p[i])), nothing), +, 1:N, init = 0.0)
end

function score(::Type{T}, ::FactorBoundFreeEnergy, ::Stochastic, node::NormalMixtureNode{N, MeanField}, scheduler) where { T <: InfCountingReal, N }
    
    stream = combineLatest((
        getmarginal(connectedvar(node.out)),
        getmarginal(connectedvar(node.switch)),
        combineLatest(map((mean) -> getmarginal(connectedvar(mean)), node.means), PushEach()),
        combineLatest(map((prec) -> getmarginal(connectedvar(prec)), node.precs), PushEach())
    ), PushEach())

    mapping = let fform = functionalform(node), meta = metadata(node)
        (d) -> begin 
            average_energy  = score(AverageEnergy(), fform, Val{ (:out, :switch, :m, :p) }, d, meta)

            out_entropy     = score(DifferentialEntropy(), d[1])
            switch_entropy  = score(DifferentialEntropy(), d[2])
            means_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, d[3])
            precs_entropies = mapreduce((m) -> score(DifferentialEntropy(), m), +, d[4])

            return convert(T, average_energy - (out_entropy + switch_entropy + means_entropies + precs_entropies))
        end
    end

    return stream |> schedule_on(scheduler) |> map(T, mapping)
end

as_node_functional_form(::Type{ <: NormalMixture }) = ValidNodeFunctionalForm()

# Node creation related functions

sdtype(::Type{ <: NormalMixture }) = Stochastic()
        
function ReactiveMP.make_node(::Type{ <: NormalMixture{N} }; factorisation::F = MeanField(), meta::M = nothing, portal::P = EmptyPortal()) where { N, F, M, P }
    @assert N >= 2 "NormalMixtureNode requires at least two mixtures on input"
    @assert typeof(factorisation) <: NormalMixtureNodeFactorisationSupport "NormalMixtureNode supports only following factorisations: [ $(NormalMixtureNodeFactorisationSupport) ]"
    out    = NodeInterface(:out)
    switch = NodeInterface(:switch)
    means  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:m)), N)
    precs  = ntuple((index) -> IndexedNodeInterface(index, NodeInterface(:p)), N)
    return NormalMixtureNode{N, F, M, P}(factorisation, out, switch, means, precs, meta, portal)
end

function ReactiveMP.make_node(::Type{ <: NormalMixture }, out::AbstractVariable, switch::AbstractVariable, means::NTuple{N, AbstractVariable}, precs::NTuple{N, AbstractVariable}; factorisation = MeanField(), meta = nothing, portal = EmptyPortal()) where { N}
    node = make_node(NormalMixture{N}, factorisation = factorisation, meta = meta, portal = portal)

    # out
    out_index = getlastindex(out)
    connectvariable!(node.out, out, out_index)
    setmessagein!(out, out_index, messageout(node.out))

    # switch
    switch_index = getlastindex(switch)
    connectvariable!(node.switch, switch, switch_index)
    setmessagein!(switch, switch_index, messageout(node.switch))

    # means
    foreach(zip(node.means, means)) do (minterface, mvar)
        mean_index = getlastindex(mvar)
        connectvariable!(minterface, mvar, mean_index)
        setmessagein!(mvar, mean_index, messageout(minterface))
    end

    # precs
    foreach(zip(node.precs, precs)) do (pinterface, pvar)
        prec_index = getlastindex(pvar)
        connectvariable!(pinterface, pvar, prec_index)
        setmessagein!(pvar, prec_index, messageout(pinterface))
    end

    return node
end

function ReactiveMP.make_node(fform::Type{ <: NormalMixture }, autovar::AutoVar, args::Vararg{ <: AbstractVariable }; kwargs...)
    var  = randomvar(getname(autovar))
    node = make_node(fform, var, args...; kwargs...)
    return node, var
end