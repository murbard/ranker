
from scipy.special import erf, comb
from scipy.integrate import quad, quadrature
import numpy as np
from timeit import timeit
from line_profiler import LineProfiler

"""
Compute the integral of exp(-kx) with a Gaussian distribution of mean μ and standard deviation σ,
between a and b, but a and b must
"""
def integral_exp_minus_kx_with_gaussian(k,μ,σ,a,b):

    ulo = np.exp(-(a * k + ((a - μ)/σ)**2 / 2))
    uhi = np.exp(-(b * k + ((b - μ)/σ)**2 / 2))

    omlo = erfc_ez2((a - μ + k * σ**2) / (np.sqrt(2) * σ))
    omhi = erfc_ez2((b - μ + k * σ**2) / (np.sqrt(2) * σ))

    # Problem, this causes catastrophic cancellation when
    # k * σ is very large, hi - lo becomes
    # very small and u very large

    # The solution is to use a series of erf towards infinity
    # Note that
    # erfc x = x/ sqrt(pi) exp(-z^2) 1 / (z^2 + a1 / (1 + a2 / (z^2 + (a3 /( 1 + ...)))))
    # with am = m/2

    # approx = quad(lambda z: np.exp(-0.5*((z-μ)/σ)**2)/(σ * np.sqrt(2*np.pi)) * np.exp(-k * z), a, b)[0]

    res = 0.5 * (ulo * omlo - uhi * omhi)
    # if np.abs(res - approx) > 1e-7:
    #    print('wot')

    return res

"""Return erfc(z) e^(z^2)"""
def erfc_ez2(z):
    if z == np.inf:
        return 0
    if z == -np.inf:
        return np.inf
    if z < 2:
        return (1 - erf(z)) * np.exp(z**2)

    z2 = z * z
    last = 1
    for m in range(9, 0, -1): # 11 is odd it matters because last=1 at first
        last = ((m&1) + (1-m&1)*z2) + (0.5*m) / last
        #print(last)
    return z / np.sqrt(np.pi) / last



"""
The integral of minus x against a gaussian between -inf and 0
"""
def integral_minus_x_with_gaussian(μ,σ):
    u =  μ/(np.sqrt(2) * σ)
    return np.exp(-u**2) * σ / np.sqrt(2 * np.pi) - 0.5 * μ * (1-erf(u))

"""
Compute the integral of log(1+exp(-x)) with a Gaussian distribution of mean μ and standard deviation σ.
We do this by approximating log(1+exp(-x)) with series in exp(-|x|) around ±∞ when |x|>1 and with
a Taylor series around 0 for |x|<1
"""

def integral(μ, σ):
    # The linear part of the approximation
    res = integral_minus_x_with_gaussian(μ,σ)
    # The constant coefficient for the middle of the approximation

    middle = [
        0.5 * (erf((1+μ)/(np.sqrt(2)*σ)) - erf(μ/(np.sqrt(2)*σ))) +
        0.5 * (erf(μ/(np.sqrt(2)*σ)) - erf((μ-1)/(np.sqrt(2)*σ))) ]
    res += np.log(2) * middle[0].sum()
    for k in range(1, 1000):
        if k == 60:
            print('foo')
        delta = 0
        # first between -inf and -1
        # log(1+exp(-x)) = log(1+exp(x)) - x
        # log(1+y) around y->0 is y - y²/2 + y³/3 ...
        delta += integral_exp_minus_kx_with_gaussian(k,-μ,σ,1,np.inf) / k
        delta += integral_exp_minus_kx_with_gaussian(k,μ,σ,1,np.inf) / k
        delta *= (-1)**(k-1)

        middle.append(
            integral_exp_minus_kx_with_gaussian(k,-μ,σ,0,1) +
            integral_exp_minus_kx_with_gaussian(k,μ,σ,0,1))

        c = 1
        m = 0.0
        for i in range(0, k+1):
            m -= c * middle[i]
            c = -(c * (k-i))//(i+1)
        m /=  k * 2**k

        delta += m
        res += delta
        if np.abs(delta) < 1.49e-8 and np.abs(delta/res) < 1.49e-8:
            return  res
    return res


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)



def time_mine():
    uu = 0
    for μ in np.linspace(-3,3,10):
        for σ in np.linspace(0.1,3,10):
            uu += integral(μ, σ)
    return uu

def time_quad():
    uu = 0
    for μ in np.linspace(-3,3,10):
        for σ in np.linspace(0.1,3,10):
            f = lambda x: np.exp(-0.5*((x-μ)/σ)**2)/(σ * np.sqrt(2 * np.pi)) * softplus_np(x)
            uu += quad(f, -np.inf, np.inf)[0]
    return uu


if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(integral)
    lp.add_function(integral_minus_x_with_gaussian)
    lp.add_function(integral_exp_minus_kx_with_gaussian)
    lp.add_function(erfc_ez2)

    lp_wrapper = lp(time_mine)
    lp_wrapper()
    lp.print_stats()

    #time_quad()



# alternative to all this
# use

# sigoid( mu  / sqrt(1 + pi s^2))
# the integral of the sigmoid and a random normal
# is basically the sigmoid of the expectation
# with an adjustment for variance
