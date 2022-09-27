#import pymc3
import numpy as np
import scipy
import scipy.stats


def softplus(x) :
    # always take the exp of a negative number for numerical
    # stability
    return np.log(1 + np.exp(-np.abs(x))) + np.clip(x, 0, None)

def integrand(x, mu, sigma):
    g = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
    i = softplus(x) * g
    expit = scipy.special.expit(x)
    # partial derivatives of integrand with respect to loc and scale?
    di_da = - expit * x * g
    di_db = expit * g
    return np.array([i, di_da, di_db])

def compute(a, b):
    # For mu=a and sigma=b, compute the integral
    # of the normal distribution times the logistic function
    lo = min(b/a - 30.0/a, -10) # at least from -10 but also look at 30 stdev?
    hi = max(b/a + 30.0/a, 10)
    x = np.linspace(lo, hi, 1000)
    return  np.trapz(integrand(x, a, b), x)

# Precompute the function for quick access
iab = np.zeros((1000,1000,3))
for i, a in enumerate(np.linspace(-20, 20, 1000)):
    for j, b in enumerate(np.linspace(-20, 20, 1000)):
        iab[i,j] = compute(a, b)
# todo, pickle the result to avoid recomputing each time


class Point(object):
    def __init__(self, mu, sigma):
        if len(mu) != len(sigma):
            raise ValueError
        self.n = len(mu)
        self.mu = mu
        self.sigma = sigma
        self.comparisons = []

    def compare(self, i, j):
        # observe i beating j
        self.comparisons.append((i,j))

    def likelihood(self):
        ll = 0
        dll_dmu = np.zeros(self.n),
        dll_dsigma = np.zeros(self.n)
        for comparison in self.comparisons:
            a, b = f(mu[i], mu[j], sigma[i], sigma[j])
            ll += compute(a, b)
            ll += compute(a_, b_)
            dll_dmu += ...
            dll_dsigma += ...


# get samples of vector mu and sigma by following HMC
# to find next comparison, for each possible comparison, for each possible result
# run HMC, compute expected improvement in metric (means comparison is informative)
# propose comparison, gather feedback


compute(1, 2)
print('foo')
