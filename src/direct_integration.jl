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
      Hfac1 = 0.5*(x'*(H.A*x))
      Hfac2 = 1.0/8.0 * (x2'*(H.V*x2))
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
  E_GM = 0.25 * trace( H.A * G + eye(N) )
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
      Hfac1 = 0.5*(xt'*(H.A*xt))
      Hfac2 = 1.0/8.0 * (xt2'*(H.V*xt2))
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
  E_GM = 0.25 * trace( (basis'*H.A*basis) * G + eye(Nimp) )
  println("E    = ", E)
  println("E_GM = ", E_GM)
  
  return (G, Omega)
end # function direct_integration
