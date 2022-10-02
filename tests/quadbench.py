from scipy.integrate import quad, quad_vec, quadrature
from scipy.special import erf
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π, inf, sin
import timeit
import quadpy

def integrand(x):
    return - exp(-x**2/2)  + erf(-(x-3)/sqrt(2)) * exp(-(x-3)**2/2) / sqrt(2*π) + sin(x**2) / (1 + x**4)

# time the integral with quad, quad_vec, and quadrature

# quad
v = quad(integrand, -inf, inf)
t = timeit.timeit(lambda: quad(integrand, -inf, inf), number=100)
print(f'quad: {t}, v: {v}')

# quad_vec
v = quad_vec(integrand,  -inf, inf)
#t = timeit.timeit(lambda: quad_vec(integrand,  -inf, inf), number=100)
print(f'quad_vec: {t}, v: {v}')

# quadrature
v = quadrature(lambda y: integrand(log(y/(1-y))) / (y*(1-y)), 0, 1, maxiter=100)
t = timeit.timeit(lambda: quadrature(lambda y: integrand(log(y/(1-y))) / (y*(1-y)), 0, 1, maxiter=1000), number=100)
print(f'quadrature: {t}, v: {v}')

# quadpy
v = quadpy.quad(integrand, -inf, inf)
t = timeit.timeit(lambda: quadpy.quad(integrand,  -inf, inf, maxiter=100), number=100)
print(f'quadpy: {t}, v: {v}')
