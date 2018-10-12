# Perform exact and approximate euclidean lattice field calculations.
#
# Lin Lin
# Revision: 7/23/2018
include("./utils.jl")

module Field

using ..Utils:gauss_hermite 
using Printf
using LinearAlgebra

mutable struct Ham
  N::Int64
  A::Array{Float64,2}
  V::Array{Float64,2}
  Nb::Int64

  function Ham(N,A,V)
    @assert size(A,1) == size(A,2) == N
    @assert size(V,1) == size(V,2) == N

    new(N, A, V)
  end
end # struct Ham

mutable struct SCFOptions
  tol::Float64
  max_iter::Int
  alpha::Float64
  verbose::Int

  function SCFOptions()
    tol = 1e-5
    max_iter = 100
    alpha = 1.0
    verbose = 1
    new(tol,max_iter,verbose)
  end
end # struct SCFOptions

mutable struct EmbeddingOptions
  # Options for converging the G in the outer iteration
  tol_outer::Float64
  max_iter_outer::Int
  alpha_outer::Float64
  # Options for converging the structured Hamiltonian matrix in the
  # inner iteration
  tol_inner::Float64
  max_iter_inner::Int
  alpha_inner::Float64
  
  # Impurity definition
  impurity_width::Int64
  impurity_stride::Int64
  NGauss::Int64
  
  verbose::Int

  function EmbeddingOptions()
    tol_outer = 1e-5
    max_iter_outer = 100
    alpha_outer = 1.0
    tol_inner = 1e-5
    max_iter_inner = 100
    alpha_inner = 1.0

    impurity_width = 1
    impurity_stride = 1
    NGauss = 20

    verbose = 1


    new(tol_outer,max_iter_outer,alpha_outer,
        tol_inner,max_iter_inner,alpha_inner,
        impurity_width, impurity_stride, NGauss,
        verbose)
  end
end # struct EmbeddingOptions

include("direct_integration.jl")

include("diagram.jl")

include("dmet.jl")

include("dmft.jl")

end # Module Field
