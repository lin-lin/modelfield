include("field.jl")
# DO NOT use `using' to aovid naming conflict in the global scope when
# loaded
import Field

N = 2
Amat = 2*diagm(ones(N)) + diagm(ones(N-1),1) + diagm(ones(N-1),-1)
Vmat = 1.0*eye(N,N)


H = Field.Ham(N,Amat,Vmat)

#(G,Omega) = Field.direct_integration_diagA(H,20)

(G_exact,Omega_exact) = Field.direct_integration_diagHF(H,20)
println("Omega (exact)       = ", Omega_exact)

G0 = ones(N,N)
opt = Field.SCFOptions()
opt.verbose = 1
(G_HF,Omega_HF) = Field.hartree_fock(H,G0,opt)
println("Omega (HF)          = ", Omega_HF)



