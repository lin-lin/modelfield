include("../src/field.jl")
# DO NOT use `using' to aovid naming conflict in the global scope when
# loaded
import ..Field

N = 3
Amat = 2*diagm(0=> ones(N)) + diagm(1=>ones(N-1)) + diagm(-1=>ones(N-1))
Vmat = 1.0*Matrix(1.0I,N,N)


H = Field.Ham(N,Amat,Vmat)

G0 = ones(N,N)
opt = Field.SCFOptions()
opt.verbose = 1
(G_HF,Omega_HF,E_HF) = Field.hartree_fock(H,G0,opt)
println("Omega  (HF)          = ", Omega_HF)
println("Energy (HF)          = ", E_HF)
A_HF = inv(G_HF)

(G_GW,Omega_GW) = Field.GW(H,G_HF,opt)
(G_GW,Omega_GW,E_GW) = Field.GW(H,G_HF,opt)
println("Omega  (GW)          = ", Omega_GW)
println("Energy (GW)          = ", E_GW)

(G_exact,Omega_exact, E_exact) = Field.direct_integration(H,20,A_HF)
println("Omega  (exact)       = ", Omega_exact)
println("Energy (exact)       = ", E_exact)


optDMET = Field.EmbeddingOptions()
optDMET.verbose=1
(G_DMET, Omega_DMET, E_DMET) = Field.DMET(H,G_HF,optDMET)
println("Energy (DMET)        = ", E_DMET)


