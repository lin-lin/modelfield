module Utils
using LinearAlgebra

function gauss_hermite(n)
# Determines the abscisas (x) and weights (w) for the Gauss-Hermite
# quadrature of order n>1, on the interval [-INF, +INF].  This function
# is valid for any degree n>=2, as the companion matrix (of the n'th
# degree Hermite polynomial) is constructed as a symmetrical matrix,
# guaranteeing that all the eigenvalues (roots) will be real.
#
# Geert Van Damme

# Building the companion matrix CM
#
# CM is such that det(xI-CM)=L_n(x), with L_n the Hermite polynomial
# under consideration. Moreover, CM will be constructed in such a way
# that it is symmetrical.

a   = sqrt.((1:n-1)/2.0)
CM  = diagm(1=>a) + diagm(-1=>a)

# Determining the abscissas (x) and weights (w) - since
# det(xI-CM)=L_n(x), the abscissas are the roots of the characteristic
# polynomial, i.d. the eigenvalues of CM
# - the weights can be derived from the corresponding eigenvectors.
(L,V) = eigen(CM)
ind = sortperm(L)
x = L[ind]
V = V[:,ind]'
w = sqrt(pi) * V[:,1].^2

return (x,w)

end # function gauss_hermite
end # module Utils
