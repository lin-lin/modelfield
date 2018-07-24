# Density matrix embedding theory
function DMET(H::Ham, G0, opt::EmbeddingOptions)
  N = H.N
  assert(size(G0,1) == size(G0,2) == N)
  G = copy(G0)
  Gnew = copy(G0)

  # Outer iteration to converge G
  for iter = 1 : opt.max_iter_outer
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
end # function DMET

