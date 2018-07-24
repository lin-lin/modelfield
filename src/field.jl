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
  Nb::Int64

  function Ham(N,Amat,Vmat)
    assert(size(Amat,1) == size(Amat,2) == N)
    assert(size(Vmat,1) == size(Vmat,2) == N)

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
# the transformed coordinate by diagonalizing the quadratic matrix
# provided by Aquad.
#
# The default suggested Aquad is from the inverse of G of the
# Hartree-Fock solution.
function direct_integration(H::Ham, 
                            NGauss, 
                            Aquad::Array{Float64,2})
  N = H.N
  (xGauss,wGauss) = gauss_hermite(NGauss)

  # compute the transformed coordinate
  # Note that Aquad should be guaranteed to be positive definite
  (DA,VA) = eig(Aquad*0.5)
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
      # Undo the rotation 
      x = (VA*(y./sqrtDA))
      x2 = x.^2
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

  # Symmetrization
  G = (G+G') * 0.5

  # Galitskii-Migdal formula
  E_GM = 0.25 * trace( H.Amat * G + eye(N) )
  println("E    = ", E)
  println("E_GM = ", E_GM)
  
  return (G, Omega)
end # function direct_integration


# Direct integration by applying Gauss-Hermite polynomial to
# the transformed coordinate by diagonalizing the quadratic matrix
# provided by Aquad.
#
# The impurity problem is defined by the basis, which is a matrix with
# orthonormal columns.
#
# A good guess for Aquad should be (basis'*G*basis), where G is the
# guess of the Green's function for the global system.
function direct_integration(H::Ham, 
                            NGauss, 
                            Aquad::Array{Float64,2},
                            basis::Array{Float64,2})
  N = H.N
  Nimp = size(basis,2)
  assert( size(basis,1) == N )
  assert( size(Aquad,1) == size(Aquad,2) == Nimp )
  
  (xGauss,wGauss) = gauss_hermite(NGauss)

  # compute the transformed coordinate
  # Note that Aquad should be guaranteed to be positive definite
  (DA,VA) = eig(Aquad*0.5)
  assert( all(DA .> 0.0) )
  sqrtDA = sqrt.(abs.(DA))
  facDA = 1./prod(sqrtDA)

  assert(NGauss^Nimp < typemax(Int64))
  inddim = ntuple(i->NGauss,Nimp)
  xt = zeros(N)
  x = zeros(Nimp)
  y = zeros(Nimp)
  w = zeros(Nimp)

  G = zeros(Nimp,Nimp)
  Z = 0.0
  E = 0.0

  tol_wgt = 1e-8

  cnt = 0
  for gind = 1 : NGauss^Nimp
    lind = ind2sub(inddim,gind)
    for i = 1 : Nimp
      y[i] = xGauss[lind[i]]
      w[i] = wGauss[lind[i]]
    end
    intfac1 = prod(w)
    # Only compute if weight is large enough
    if( intfac1 > tol_wgt )
      cnt += 1
      # Undo the rotation and basis projection
      x = VA*(y./sqrtDA)
      xt = basis * x
      xt2 = xt.^2
      Hfac1 = 0.5*(xt'*(H.Amat*xt))
      Hfac2 = 1.0/8.0 * (xt2'*(H.Vmat*xt2))
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

  # Symmetrization
  G = (G+G') * 0.5

  # Galitskii-Migdal formula
  E_GM = 0.25 * trace( (basis'*H.Amat*basis) * G + eye(Nimp) )
  println("E    = ", E)
  println("E_GM = ", E_GM)
  
  return (G, Omega)
end # function direct_integration


# Hartree-Fock method
#
# Self-consistency is controlled through opt.maxiter
function hartree_fock(H::Ham, G0, opt::SCFOptions)
  N = H.N
  assert(size(G0,1) == N)
  G = copy(G0)
  Gnew = copy(G0)

  Sigma1 = zeros(N,N)

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
      @printf("Convergence reached. nrmerr = %g\n", nrmerr)
      break
    end
    G = (1-opt.alpha)*G + opt.alpha*GNew
  end

  # Symmetrization
  G = (G+G') * 0.5
  
  # Luttinger-Ward functional
  Phi0 = N*(log(2*pi)+1.0)
  Phi = 1/2*trace(Sigma1*G)
  Omega = 0.5*(trace(H.Amat*G) - log(det(G)) - (Phi + Phi0))
  return (G, Omega)
end # function hartree_fock


# 2nd order Green's function expansion method
#
# Self-consistency is controlled through opt.maxiter
function GF2(H::Ham, G0, opt::SCFOptions)
  N = H.N
  assert(size(G0,1) == N)
  G = copy(G0)
  Gnew = copy(G0)
  
  Sigma1 = zeros(N,N)
  Sigma2 = zeros(N,N)

  for iter = 1 : opt.max_iter
    rho = diag(G)
    Sigma1 = -0.5*diagm(H.Vmat * rho) - (H.Vmat.*G)
    Chi = G.*G
    # Ring term
    Sigma2 = 1/2 * G.*(H.Vmat * Chi * H.Vmat)
    # Second order exchange term
    for i = 1 : N
      for j = 1 : N
        tt1 = H.Vmat[:,i].*G[:,j]
        tt2 = H.Vmat[:,j].*G[:,i]
        Sigma2[i,j] += tt1'*G*tt2
      end
    end
    
    Sigma = Sigma1 + Sigma2
    GNew = inv(H.Amat-Sigma)
    nrmerr = norm(G-GNew)/norm(G)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol )
      @printf("Convergence reached. nrmerr = %g\n", nrmerr)
      break
    end
    G = (1-opt.alpha)*G + opt.alpha*GNew
  end

  # Symmetrization
  G = (G+G') * 0.5
  
  # Luttinger-Ward functional
  Phi0 = N*(log(2*pi)+1.0)
  Phi = 1/2*trace(Sigma1*G) + 1/4*trace(Sigma2*G)
  Omega = 0.5*(trace(H.Amat*G) - log(det(G)) - (Phi + Phi0))
  return (G, Omega)
end # function GF2


# GW
#
# Self-consistency is controlled through opt.maxiter
function GW(H::Ham, G0, opt::SCFOptions)
  N = H.N
  assert(size(G0,1) == N)
  G = copy(G0)
  Gnew = copy(G0)
  
  Sigma1 = zeros(N,N)
  Sigma2 = zeros(N,N)
  Chi = zeros(N,N)

  for iter = 1 : opt.max_iter
    rho = diag(G)
    Chi = G.*G
    W = inv(eye(N) + 1/2*H.Vmat*Chi) * H.Vmat
    Sigma1 = -0.5*diagm(H.Vmat * rho) - (W.*G)
    
    Sigma = Sigma1
    GNew = inv(H.Amat-Sigma)
    nrmerr = norm(G-GNew)/norm(G)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol )
      @printf("Convergence reached. nrmerr = %g\n", nrmerr)
      break
    end
    G = (1-opt.alpha)*G + opt.alpha*GNew
  end

  # Symmetrization
  G = (G+G') * 0.5
  
  # Luttinger-Ward functional
  Phi0 = N*(log(2*pi)+1.0)
  rho = diag(G)
  SigmaHartree = -1/2*diagm(H.Vmat * rho)
  Phi = 1/2*trace(SigmaHartree*G) - trace(logm(eye(N)+1/2*H.Vmat*Chi))
  Omega = 0.5*(trace(H.Amat*G) - log(det(G)) - (Phi + Phi0))
  return (G, Omega)
end # function GF2

end # Module Field
