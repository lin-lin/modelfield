include("Field.jl")
# DO NOT use `using' to aovid naming conflict in the global scope when
# loaded
import Field

N = 2
Amat = [2.0 1.0;1.0 2.0]
Vmat = eye(N,N)


H = Field.Ham(N,Amat,Vmat)
G0 = ones(N,N)

opt = Field.SCFOptions()
opt.verbose = 2
Field.hartree_fock(H,G0,opt)

