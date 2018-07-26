include("../src/field.jl")
# DO NOT use `using' to aovid naming conflict in the global scope when
# loaded
import Field

N = 3
Amat = 2*diagm(ones(N)) + diagm(ones(N-1),1) + diagm(ones(N-1),-1)
Vmat = 1.0*eye(N,N)


H = Field.Ham(N,Amat,Vmat)

G0 = ones(N,N)
opt = Field.SCFOptions()
opt.verbose = 1
(G_HF,Omega_HF) = Field.hartree_fock(H,G0,opt)
println("Omega (HF)          = ", Omega_HF)
A_HF = inv(G_HF)

(G_exact,Omega_exact) = Field.direct_integration(H,20,A_HF)
println("Omega (exact)       = ", Omega_exact)

#basis = zeros(N,2)
#basis[1,1] = 1.0
#basis[2:end,2] = qr(G_exact[2:end,1])[1]
#Gimp_HF = basis'*G_HF*basis
#Aimp_HF = inv(Gimp_HF)
#Field.direct_integration(H,20,Aimp_HF,basis)

optDMET = Field.EmbeddingOptions()
optDMET.verbose=1
(G_DMET, Omega_DMET) = Field.DMET(H,G_exact,optDMET)

