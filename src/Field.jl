# Perform exact and approximate 
#
include("utils.jl")

module Field

export Ham, SCFOptions
export hartree_fock

mutable struct Ham
  N::Int64
  Amat::Array{Float64,2}
  Vmat::Array{Float64,2}
  Afunc
  Ufunc


  function Ham(N,Amat,Vmat)
    assert(size(Amat,1) == N)
    assert(size(Vmat,1) == N)

    new(N, Amat, Vmat)
  end
end # struct Ham

mutable struct SCFOptions
  tol::Float64
  max_iter::Int
  verbose::Int

  function SCFOptions()
    tol = 1e-5
    max_iter = 100
    verbose = 1
    new(tol,max_iter,verbose)
  end
end # struct SCFOptions

#function 


function hartree_fock(H::Ham, G0, opt::SCFOptions)
  assert(size(G0,1) == H.N)
  G = copy(G0)
  Gnew = copy(G0)

  for iter = 1 : opt.max_iter
    rho = diag(G)
    Sigma1 = -0.5*diagm(H.Vmat * rho) - (H.Vmat.*G)
    Sigma = Sigma1
    GNew = inv(H.Amat-Sigma)
    nrmerr = norm(G-GNew)/norm(G)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol )
      @printf("Convergence reached.\n\n")
      break
    end
    G = copy(GNew)
  end

  return G
end # function hartree_fock

end # Module Field
