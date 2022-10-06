
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import ranker.models as models

from scipy.stats import invgamma, norm
from scipy.special import gammaln, gamma, polygamma, erf, expit, log_expit, erfc
from scipy.integrate import quad, dblquad
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π, inf
import numpy as np
import copy
from parameterized import parameterized

np.random.seed(42)

def get_params():
    for i in range(50):
        model = models.Model(4)
        instance = model.rvs()
        vbayes = models.VBayes(model)
        vbayes.α()[0] *= (1 + np.random.randn() * 0.1)
        vbayes.β()[0] *= (1 + np.random.randn() * 0.1)
        vbayes.μ()[:] += np.random.randn() * (sqrt(instance.v) / 3)
        vbayes.σ()[:] *= (1 + np.random.randn(model.n) * 0.1)

        obs = instance.observe(3)

        yield (f'case_{i}', vbayes, obs)

params = list(get_params())





class TestIntegral(unittest.TestCase):

    def assertClose(self, a, b, tol=1e-2):
        if a == 0 or b == 0:
            self.assertAlmostEqual(a, b, delta=tol)
        else:
            self.assertLess( 2 * np.abs(a-b) / (np.abs(a) + np.abs(b)), tol)

    def setUp(self):
        np.random.seed(42)

    @parameterized.expand(params)
    def test_eval(self, name, vbayes, obs):
        α, β = vbayes.α()[0], vbayes.β()[0]
        µ, σ = vbayes.µ(), vbayes.σ()

        minus_invgamma_entropy = lambda v, α, β:  invgamma(α, scale=1/β).pdf(v) * invgamma(α, scale=1/β).logpdf(v)
        minus_normal_entropy = lambda x, µ, σ: norm(loc=µ, scale=σ).pdf(x) * norm(loc=µ, scale=σ).logpdf(x)
        gamma_cross_entropy = lambda v, α, β: - invgamma(α, scale=1/β).pdf(v) * invgamma(vbayes.model.α, scale=1/vbayes.model.β).logpdf(v)
        normal_cross_entropy = lambda α, β, μ, σ : 1/2 * (log(2 * π / β) + α * β * (μ**2 + σ**2) - polygamma(0, α))
        observation = lambda δ, μδ, σδ: norm(loc=μδ, scale=σδ).pdf(δ) * (2 * exp(-π * δ**2 / 16) / π - (1/2) * δ * erfc(sqrt(π) * δ / 4))

        integral = (
              quad(minus_invgamma_entropy, 0, inf, args=(α, β))[0]
            + sum([quad(minus_normal_entropy, -inf, inf, args=(µ[i], σ[i]))[0] for i in range(vbayes.n)])
            + quad(gamma_cross_entropy, 0, inf, args=(α, β))[0]
            + sum([normal_cross_entropy(α, β, µ[i], σ[i]) for i in range(vbayes.n)])
            + sum([obs[i,j] * quad(observation, -inf, inf, args=(µ[i] - μ[j], sqrt(σ[i]**2 + σ[j]**2)))[0] for j in range(vbayes.n) for i in range(vbayes.n) if obs[i,j] != 0]))

        self.assertAlmostEqual(integral, vbayes.eval(obs=obs).val, places=4)

    @parameterized.expand(params)
    def test_gradients_hessian(self, name, vbayes, obs):


        funcs = [
            lambda v, _, cg=False, ch=False : v.__minus_invgamma_entropy__(compute_gradient = cg, compute_hessian = ch),
            lambda v, _, cg=False, ch=False : v.__minus_normal_entropy__(compute_gradient = cg, compute_hessian = ch),
            lambda v, _, cg=False, ch=False : v.__gamma_cross_entropy__(compute_gradient = cg, compute_hessian = ch),
            # lambda v, _, cg=False, ch=False : v.__normal_cross_entropy__(compute_gradient = cg, compute_hessian = ch),
            lambda v, obs, cg=False, ch=False : v.__minus_observations__(compute_gradient = cg, compute_hessian = ch, obs=obs)
        ]

        for func_name, func in zip([
            'minus_invgamma_entropy',
            'minus_normal_entropy',
            'gamma_cross_entropy',
            #'normal_cross_entropy'
            'minus_observations' ], funcs):
            gh = func(vbayes, obs, cg=True, ch=True)
            val = gh.val

            ε = 1e-4

            points = np.zeros((4,4))
            ss = [-2,-1,1,2]
            # test hessian and gradient
            for k in range(vbayes.n):

                for u in range(2):
                    vbayes.params[k] += (2*u - 1) * ε
                    points[u,0] = func(vbayes, obs).val
                    vbayes.params[k] -= (2*u - 1) * ε
                    vbayes.params[k] += 2 * (2*u - 1) * ε
                    points[u,1] = func(vbayes, obs).val
                    vbayes.params[k] -= 2* (2*u - 1) * ε

                g = (points[1,0] - points[0,0])/(2 * ε)
                g_lo = (points[1,1] - points[0,1])/(4 * ε)


                # g = true_g + c ε^3 + O(ε^4)
                # g_lo = true_g + c (2 ε)^3 + O(ε^4)
                # c ~ (g_lo - g) / (3 * ε^3)
                # error of g ~ (g_lo - g) / (3)

                error = max(2 * abs(g_lo - g) / 3, 1e-6)
                g = (4 * g - g_lo) / 3

                try:
                    self.assertAlmostEqual(g, gh.g[k], delta=error)
                except AssertionError as e:
                    print(f'gradient {func_name} {k} failed')
                    raise e

                h = (points[1,0] - 2 * val + points[0,0])/(ε**2)
                h_lo = (points[1,1] - 2 * val + points[0,1])/(4*ε**2)

                error = max(2 * abs(h_lo - h) / 3, 1e-6)
                h = (4 * h - h_lo) / 3

                try:
                    self.assertAlmostEqual(h, gh.h[k,k], delta=error)
                except AssertionError:
                    print(f'Failed for {name} on {func_name} for {k}')
                    raise

                for l in range(k + 1, vbayes.n):
                    for u in range(4):
                        for v in range(4):
                            vbayes.params[k] += ss[u] * ε
                            vbayes.params[l] += ss[v] * ε
                            points[u,v] = func(vbayes, obs).val
                            vbayes.params[k] -= ss[u] * ε
                            vbayes.params[l] -= ss[v] * ε
                    h = (points[2,2] + points[1,1] - points[1,2] - points[2,1])/(4*ε**2)
                    h_lo = (points[3,3] + points[0,0] - points[0,3] - points[3,0])/(16*ε**2)

                    error = max(2 * abs(h_lo - h) / 3, 1e-6)
                    h = (4 * h - h_lo) / 3

                    try:
                        self.assertAlmostEqual(h, gh.h[k,l], delta=error)
                    except AssertionError:
                        print(f'Failed for {name} on {func_name} for {k} {l}')
                        raise
