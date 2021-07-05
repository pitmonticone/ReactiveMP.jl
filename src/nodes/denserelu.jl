export DenseReLU, DenseReLUMeta

struct DenseReLU end

@node DenseReLU Stochastic [ output, input, w, z, f ]

mutable struct DenseReLUMeta{T}
    C :: T
    α :: T
    β :: T
    γ :: T
end

function DenseReLUMeta(C::T1, α::T2, β::T3, γ::T4) where { T1, T2, T3, T4 }
    T = promote_type(T1, T2, T3, T4)

    return DenseReLUMeta{T}(C, α, β, γ)
end

getC(meta::DenseReLUMeta)           = meta.C
getα(meta::DenseReLUMeta)           = meta.α
getβ(meta::DenseReLUMeta)           = meta.β
getγ(meta::DenseReLUMeta)           = meta.γ