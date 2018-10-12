# Preliminary result shows that the bare DMFT is more accurate than
# DMET. GW only improves with respect to HF marginally.

using LinearAlgebra
include("../src/field.jl")
# DO NOT use `using' to aovid naming conflict in the global scope when
# loaded
import ..Field

N = 5
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
(G_DMET1, Omega_DMET, E_DMET1) = Field.DMET_1(H,G_HF,optDMET)
println("Energy (DMET)        = ", E_DMET1)

optDMET = Field.EmbeddingOptions()
optDMET.verbose=1
(G_DMET2, Omega_DMET, E_DMET2) = Field.DMET_2(H,G_HF,optDMET)
println("Energy (DMET)        = ", E_DMET2)


optDMET = Field.EmbeddingOptions()
optDMET.verbose=1
(G_DMET3, Omega_DMET, E_DMET3) = Field.DMET_3(H,G_HF,optDMET)
println("Energy (DMET)        = ", E_DMET3)


optDMFT = Field.EmbeddingOptions()
optDMFT.verbose=1
(G_DMFT1, Omega_DMFT, E_DMFT1) = Field.DMFT_1(H,G_HF,optDMFT)
println("Energy (DMFT)        = ", E_DMFT1)
