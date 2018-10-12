# Dynamical mean field theory
#
# This also uses the build_impurity_pattern function in dmet
#
# Last revision: 10/11/2018

# This is just block diagonal Sigma. 
#
function DMFT_1(H::Ham, G_global0, opt::EmbeddingOptions)
    N = H.N
    @assert(size(G_global0,1) == size(G_global0,2) == N)
    G_global     = copy(G_global0)
    G_local      = zeros(N,N)
    Gamma_local  = zeros(N,N)
    Gamma_local_new = zeros(N,N)
    Sigma_local  = zeros(N,N)

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

            Aimp = H.A[cur_index,cur_index] + Gamma_local[cur_index, cur_index]
            Vimp = H.V[cur_index,cur_index]

            Himp = Field.Ham(width, Aimp, Vimp)

            Aquad = inv(G_global[cur_index, cur_index])

            # Perform direct integration for impurity
            (imp_G, imp_Omega, imp_E) = direct_integration(Himp, opt.NGauss, Aquad)

            G_local[cur_index,cur_index] = imp_G
            Sigma_local[cur_index,cur_index] = Aimp - inv(imp_G)

            # FIXME
            Phi += 0.0
        end
        
        G_global = inv(H.A - Sigma_local)
        for i_imp = 1 : num_imp_block
            cur_index = partition_index[i_imp]
            Gamma_local_new[cur_index, cur_index] =
                inv(G_global[cur_index, cur_index]) - 
                H.A[cur_index, cur_index] + 
                Sigma_local[cur_index, cur_index]
        end


        nrmerr = norm(Gamma_local - Gamma_local_new)/norm(Gamma_local)
        if( opt.verbose > 1 )
            @printf("iter = %4d,  nrmerr = %15.5e\n", iter, nrmerr)
        end
        if( nrmerr < opt.tol_outer )
            @printf("Convergence reached of DMFT. nrmerr = %g\n", nrmerr)
            break
        end
        Gamma_local = (1-opt.alpha_outer)*Gamma_local + 
            opt.alpha_outer*Gamma_local_new
    end

    # Symmetrization
    G = G_global
    G = (G+G') * 0.5

    Phi0 = N*(log(2*pi)+1.0)
    println("Phi = ", Phi)
    Omega = 0.5*(tr(H.A*G) - log(det(G)) - (Phi + Phi0))
    E_GM = 0.25 * tr( H.A * G + Matrix(1.0I,N,N) )
    return (G, Omega, E_GM)
end # function DMFT_1
