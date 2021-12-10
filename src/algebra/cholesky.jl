export cholinv, cholsqrt

using LinearAlgebra
using PositiveFactorizations

cholinv(x)           = inv(cholesky(PositiveFactorizations.Positive, Hermitian(x)))
cholinv(x::Diagonal) = Diagonal(inv.(diag(x)))
cholinv(x::Real)     = inv(x)
function cholinv(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat }
    y = cholesky(PositiveFactorizations.Positive, Hermitian(x))
    LinearAlgebra.inv!(y)
    return y.factors
end

cholsqrt(x)           = Matrix(cholesky(PositiveFactorizations.Positive, Hermitian(x)).L)
cholsqrt(x::Diagonal) = Diagonal(sqrt.(diag(x)))
cholsqrt(x::Real)     = sqrt(x)

chollogdet(x)           = logdet(cholesky(PositiveFactorizations.Positive, Hermitian(x)))
chollogdet(x::Diagonal) = logdet(x)
chollogdet(x::Real)     = logdet(x)

function cholinv_logdet(x) 
    # calculate cholesky decomposition
    y = cholesky(PositiveFactorizations.Positive, Hermitian(x))
      
    # return inverse and log-determinant
    return inv(y), logdet(y)
end
function cholinv_logdet(x::AbstractMatrix{T}) where { T <: LinearAlgebra.BlasFloat } 
    # calculate cholesky decomposition
    y = cholesky(PositiveFactorizations.Positive, Hermitian(x))
    
    # calculate logdeterminant of cholesky decomposition
    ly = logdet(y)
    
    # calculate inplace inverse of A and store in y.factors
    LinearAlgebra.inv!(y)
    
    # return inverse and log-determinant
    return y.factors, ly
end
cholinv_logdet(x::Diagonal) = Diagonal(inv.(diag(x))), mapreduce(z -> log(z), +, diag(x))
cholinv_logdet(x::Real)     = inv(x), log(abs(x))
