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
