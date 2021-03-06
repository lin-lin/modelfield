# Density matrix embedding theory.
#
# Last revision: 10/11/2018 

# Currently, the embedding does not allow overlapping impurity elements.
function DMET_1(H::Ham, G_global0, opt::EmbeddingOptions)
  N = H.N
  @assert(size(G_global0,1) == size(G_global0,2) == N)
  G_global     = copy(G_global0)
  G_global_new = copy(G_global0)
  G_local      = zeros(N,N)
  Vimp = zeros(N,N)
  Omega = 0.0
  Phi = 0.0

  (partition_index, block_index, imp_index) = build_impurity_pattern(N, opt)
  num_imp_block = length(block_index)

  width = opt.impurity_width

  # Outer iteration to converge G_global
  for iter = 1 : opt.max_iter_outer
    Phi = 0.0
    for i_imp = 1 : num_imp_block
      cur_index = partition_index[i_imp]
      res_index = setdiff(1:N, cur_index)
      tmp_basis = Matrix(qr(G_global[res_index,cur_index]).Q)
      basis = zeros(N,2*width)
      basis[cur_index,1:width] = Matrix(1.0I,width,width)
      basis[res_index,width+1:2*width] = tmp_basis
      Aquad = inv(basis'*(G_global*basis))
      
      # Perform direct integration for impurity
      (imp_G, imp_Omega, imp_Phi) = direct_integration(H, opt.NGauss,
                                                       Aquad, basis,
                                                       opt.verbose)
      imp_G = imp_G[1:width,1:width]
      G_local[block_index[i_imp]] = imp_G[:]
      Phi += imp_Phi
    end

    # Hquad can be improved to HF or other guesses later
    Hquad = H.A
    Vimp = fit_impurity_potential(Hquad, G_local, imp_index,
                                  Vimp, opt)
    G_global_new = inv(Hquad + Vimp)

    nrmerr = norm(G_global - G_global_new)/norm(G_global)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol_outer )
      @printf("Convergence reached of DMET. nrmerr = %g\n", nrmerr)
      break
    end
    G_global = (1-opt.alpha_outer)*G_global + opt.alpha_outer*G_global_new
  end

  if( opt.verbose > 1 )
    println("G_global =")
    display(G_global)
    println("G_local =")
    display(G_local)
  end
  mismatch = norm(G_global[imp_index] - G_local[imp_index])
  @printf("Mismatch between global and local = %g\n", mismatch)

  # Symmetrization
  G = G_global
  G = (G+G') * 0.5

  Phi0 = N*(log(2*pi)+1.0)
  println("Phi = ", Phi)
  Omega = 0.5*(tr(H.A*G) - log(det(G)) - (Phi + Phi0))
  E_GM = 0.25 * tr( H.A * G + Matrix(1.0I,N,N) )
  return (G, Omega, E_GM)
end # function DMET

# Build the impurity sparse structure pattern
function build_impurity_pattern(N,opt::EmbeddingOptions)
  @assert( opt.impurity_stride == opt.impurity_width )
  num_imp_block = ceil(Int64, N/opt.impurity_stride)
  partition_index = Array{Any,1}(undef,num_imp_block)
  block_index = Array{Any,1}(undef,num_imp_block)
  global_index = zeros(Int64,0)
  idx_mat = reshape(1:N^2,N,N)
  for i_imp = 1 : num_imp_block
    idx_sta = (i_imp-1)*opt.impurity_stride + 1
    idx_end = min(idx_sta + opt.impurity_width - 1, N)
    partition_index[i_imp] = idx_sta:idx_end
    gi = idx_mat[partition_index[i_imp],partition_index[i_imp]] 
    block_index[i_imp] = gi[:]
    append!(global_index, gi[:])
  end

  return (partition_index, block_index, global_index)
end

# Fit the impurity potential to satisfy
#
# inv(A + Vimp) = G 
#
# restricted to the sparsity pattern given by imp_index
#
function fit_impurity_potential(A::Array{Float64,2},
                                G_target::Array{Float64,2},
                                imp_index::Array{Int64},
                                Vimp0::Array{Float64,2},
                                opt::EmbeddingOptions)
  
  N = size(A,1)

  G_target_vec = G_target[imp_index]

  Vimp = copy(Vimp0)
  Vimp_vec = Vimp[imp_index]

  for iter = 1 : opt.max_iter_inner
    G = inv(A + Vimp)
    G_vec = G[imp_index]

    res = G_vec - G_target_vec
    nrmerr = norm(res)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol_inner )
      @printf("Convergence reached for impurity fitting. nrmerr = %g\n", nrmerr)
      break
    end

    # Might need to try negative mixing constant
    Vimp_vec = Vimp_vec + opt.alpha_inner * res
    Vimp[imp_index] = Vimp_vec
  end

  return Vimp
end


# Currently, the embedding does not allow overlapping impurity elements.
# This version of DMET directly obtains G from the impurity problem and
# patch them together. The energy can then be computed directly from
# G_1RDM (i.e. G) and G_2RDM. The free energy should be computed via a
# thermodynamic integration approach.
#

function DMET_2(H::Ham, G_global0, opt::EmbeddingOptions)
  N = H.N
  @assert(size(G_global0,1) == size(G_global0,2) == N)
  G_global     = copy(G_global0)
  G_global_new = copy(G_global0)
  Vimp = zeros(N,N)
  Omega = 0.0
  Phi = 0.0

  (partition_index, block_index, imp_index) = build_impurity_pattern(N, opt)
  num_imp_block = length(block_index)

  width = opt.impurity_width

  # Outer iteration to converge G_global
  for iter = 1 : opt.max_iter_outer
#    println(eigvals(G_global))
    for i_imp = 1 : num_imp_block
      cur_index = partition_index[i_imp]
      res_index = setdiff(1:N, cur_index)
      basis = G_global[:,cur_index] * inv(G_global[cur_index,cur_index])
#      display(basis)

      Aquad = (basis'*(inv(G_global)*basis))
#      display(inv(G_global[cur_index,cur_index]))
#      display(Aquad)
#      sleep(1)
      
      # Perform direct integration for impurity
      (imp_G, imp_Omega, _) = direct_integration(H, opt.NGauss,
                                                       Aquad, basis,
                                                       opt.verbose)
      tmp_G = basis*imp_G[1:width,1:width]*basis'
      # Update the block row only

      G_global_new[cur_index,:] = tmp_G[cur_index,:]
    end

    G_global_new = (G_global_new + G_global_new') * 0.5

    nrmerr = norm(G_global - G_global_new)/norm(G_global)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol_outer )
      @printf("Convergence reached of DMET. nrmerr = %g\n", nrmerr)
      break
    end
    G_global = (1-opt.alpha_outer)*G_global + opt.alpha_outer*G_global_new
  end

  if( opt.verbose > 1 )
    println("G_global =")
    display(G_global)
  end

  # Symmetrization
  G = G_global
  G = (G+G') * 0.5

  Phi0 = N*(log(2*pi)+1.0)
  Phi = 0.0
  Omega = 0.5*(tr(H.A*G) - log(det(G)) - (Phi + Phi0))
  E_GM = 0.25 * tr( H.A * G + Matrix(1.0I,N,N) )
  return (G, Omega, E_GM)
end # function DMET_2

function DMET_3(H::Ham, G_global0, opt::EmbeddingOptions)
  N = H.N
  @assert(size(G_global0,1) == size(G_global0,2) == N)
  G_global     = copy(G_global0)
  G_global_new = copy(G_global0)
  Vimp = zeros(N,N)
  Omega = 0.0
  Phi = 0.0

  (partition_index, block_index, imp_index) = build_impurity_pattern(N, opt)
  num_imp_block = length(block_index)

  width = opt.impurity_width

  # Outer iteration to converge G_global
  for iter = 1 : opt.max_iter_outer
#    println(eigvals(G_global))
    for i_imp = 1 : num_imp_block
      cur_index = partition_index[i_imp]
      res_index = setdiff(1:N, cur_index)
      tmp_basis = Matrix(qr(G_global[res_index,cur_index]).Q)
      basis = zeros(N,2*width)
      basis[cur_index,1:width] = Matrix(1.0I,width,width)
      basis[res_index,width+1:2*width] = tmp_basis
#      display(basis)

      Aquad = (basis'*(inv(G_global)*basis))
#      display(inv(G_global[cur_index,cur_index]))
#      display(Aquad)
#      sleep(1)
      
      # Perform direct integration for impurity
      (imp_G, imp_Omega, _) = direct_integration(H, opt.NGauss,
                                                       Aquad, basis,
                                                       opt.verbose)
      tmp_G = basis*imp_G*basis'
#      tmp_G = basis[:,1:width]*imp_G[1:width,1:width]*basis[:,1:width]'
      # Update the block row only

      G_global_new[cur_index,:] = tmp_G[cur_index,:]
    end

    G_global_new = (G_global_new + G_global_new') * 0.5

    nrmerr = norm(G_global - G_global_new)/norm(G_global)
    if( opt.verbose > 1 )
      @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
    end
    if( nrmerr < opt.tol_outer )
      @printf("Convergence reached of DMET. nrmerr = %g\n", nrmerr)
      break
    end
    G_global = (1-opt.alpha_outer)*G_global + opt.alpha_outer*G_global_new
  end

  if( opt.verbose > 1 )
    println("G_global =")
    display(G_global)
  end

  # Symmetrization
  G = G_global
  G = (G+G') * 0.5

  Phi0 = N*(log(2*pi)+1.0)
  Phi = 0.0
  Omega = 0.5*(tr(H.A*G) - log(det(G)) - (Phi + Phi0))
  E_GM = 0.25 * tr( H.A * G + Matrix(1.0I,N,N) )
  return (G, Omega, E_GM)
end # function DMET_3
