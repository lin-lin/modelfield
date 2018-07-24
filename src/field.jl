# Perform exact and approximate euclidean lattice field calculations.
#
# Lin Lin
# Revision: 7/23/2018
include("utils.jl")

module Field

using Utils.gauss_hermite 

mutable struct Ham
  N::Int64
  Amat::Array{Float64,2}
  Vmat::Array{Float64,2}

  function Ham(N,Amat,Vmat)
    assert(size(Amat,1) == N)
    assert(size(Vmat,1) == N)

    new(N, Amat, Vmat)
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

# Direct integration by applying Gauss-Hermite polynomial to
# the transformed coordinate by diagonalizing A
function direct_integration_diagA(H::Ham, NGauss)
  N = H.N
  (xGauss,wGauss) = gauss_hermite(NGauss)
  # diagonalize the quadratic part and compute in the transformed
  # coordinate
  # CAVEAT: this is mainly for the positive definite case.
  # the eigenvalue of A cannot be zero.
  (DA,VA) = eig(H.Amat*0.5)
  sgnDA = sign.(DA)
  sqrtDA = sqrt.(abs.(DA))
  facDA = 1./prod(sqrtDA)

  assert(NGauss^N < typemax(Int64))
  inddim = ntuple(i->NGauss,N)
  x = zeros(N)
  y = zeros(N)
  w = zeros(N)

  G = zeros(N,N)
  Z = 0.0
  E = 0.0

  tol_wgt = 1e-8

  cnt = 0
  for gind = 1 : NGauss^N
    lind = ind2sub(inddim,gind)
    for i = 1 : N
      y[i] = xGauss[lind[i]]
      w[i] = wGauss[lind[i]]
    end
    intfac1 =  prod(w)
    # Only compute if weight is large enough
    if( intfac1 > tol_wgt )
      cnt += 1
      x = (VA*(y./sqrtDA))
      x2 = x.^2
      # note that Amat is already diagonalized
      Hfac1 = sum(sgnDA .* (y.^2))
      Hfac2 = 1.0/8.0 * (x2'*(H.Vmat*x2))
      Hfac = Hfac1 + Hfac2
      intfac2 = exp(-Hfac)
      intfac = exp(sum(y.^2)) * intfac1 * intfac2
      E += Hfac * intfac 
      Z += intfac
      G += (x * x') * intfac
    end
  end
  println("Percentage of evaluated configurations = ", 
          cnt / float(NGauss^N))

  Z = Z * facDA
  G = G * facDA / Z
  E = E * facDA / Z
  Omega = -log(Z)

  # Galitskii-Migdal formula
  E_GM = 0.25 * trace( H.Amat * G + eye(N) )
  println("E    = ", E)
  println("E_GM = ", E_GM)
  
  return (G, Omega)
end # function direct_integration_diagA

# Direct integration by applying Gauss-Hermite polynomial to
# the transformed coordinate by diagonalizing the Hartree-Fock part
#
# This can be much more effective than the quadrature by diagonalizing
# A.
function direct_integration_diagHF(H::Ham, NGauss)
  N = H.N
  (xGauss,wGauss) = gauss_hermite(NGauss)
  # Evaluate the Hartree contribution

  # diagonalize the Hartree-Fock part
  # Assume Amat is invertible as the initial guess
  G0 = inv(H.Amat)
  opt = SCFOptions()
  (G_HF, _) = hartree_fock(H, G0, opt)
  Amat_HF = inv(G_HF)

  # compute the transformed coordinate
  # Note that Amat_HF is guaranteed to be positive definite
  (DA,VA) = eig(Amat_HF*0.5)
  assert( all(DA .> 0.0) )
  sqrtDA = sqrt.(abs.(DA))
  facDA = 1./prod(sqrtDA)

  assert(NGauss^N < typemax(Int64))
  inddim = ntuple(i->NGauss,N)
  x = zeros(N)
  y = zeros(N)
  w = zeros(N)

  G = zeros(N,N)
  Z = 0.0
  E = 0.0

  tol_wgt = 1e-8

  cnt = 0
  for gind = 1 : NGauss^N
    lind = ind2sub(inddim,gind)
    for i = 1 : N
      y[i] = xGauss[lind[i]]
      w[i] = wGauss[lind[i]]
    end
    intfac1 = prod(w)
    # Only compute if weight is large enough
    if( intfac1 > tol_wgt )
      cnt += 1
      x = (VA*(y./sqrtDA))
      x2 = x.^2
      # note that Amat is already diagonalized
      Hfac1 = 0.5*(x'*(H.Amat*x))
      Hfac2 = 1.0/8.0 * (x2'*(H.Vmat*x2))
      Hfac = Hfac1 + Hfac2
      intfac2 = exp(-Hfac)
      intfac = exp(sum(y.^2)) * intfac1 * intfac2
      E += Hfac * intfac 
      Z += intfac
      G += (x * x') * intfac
    end
  end
  println("Percentage of evaluated configurations = ", 
          cnt / float(NGauss^N))

  Z = Z * facDA
  G = G * facDA / Z
  E = E * facDA / Z
  Omega = -log(Z)

  # Galitskii-Migdal formula
  E_GM = 0.25 * trace( H.Amat * G + eye(N) )
  println("E    = ", E)
  println("E_GM = ", E_GM)
  
  return (G, Omega)
end # function direct_integration_diagHF

# Hartree-Fock method
#
# Self-consistency is controlled through opt.maxiter
function hartree_fock(H::Ham, G0, opt::SCFOptions)
  N = H.N
  assert(size(G0,1) == N)
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
      @printf("Convergence reached. nrmerr = %g\n\n", nrmerr)
      break
    end
    G = (1-opt.alpha)*G + opt.alpha*GNew
  end
  
  # Luttinger-Ward functional
  Phi0 = N*(log(2*pi)+1.0)
  rho = diag(G)
  Sigma1 = -0.5*diagm(H.Vmat * rho) - (H.Vmat.*G)
  Phi = 1/2*trace(Sigma1*G);
  Omega = 0.5*(trace(H.Amat*G) - log(det(G)) - (Phi + Phi0))
  return (G, Omega)
end # function hartree_fock

end # Module Field
