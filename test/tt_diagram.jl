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

(G_GF2,Omega_GF2) = Field.GF2(H,G_HF,opt)
println("Omega (GF2)         = ", Omega_GF2)

opt.verbose=1
(G_GW,Omega_GW) = Field.GW(H,G_HF,opt)
println("Omega (GW)          = ", Omega_GW)


(G_exact,Omega_exact) = Field.direct_integration(H,20,A_HF)
println("Omega (exact)       = ", Omega_exact)



