
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


def get_params():
    for i in range(50):
        model = models.Model(4)
        instance = model.rvs()
        vbayes = models.VBayes(model)
        vbayes.α *= (1 + np.random.randn(model.n) * 0.1)
        vbayes.β *= (1 + np.random.randn(model.n) * 0.1)
        vbayes.μ += np.random.randn() * (sqrt(instance.v) / 3)
        vbayes.σ *= (1 + np.random.randn(model.n) * 0.1)

        obs = instance.observe(3)

        yield ('case_{i}', vbayes, obs)


class TestGradients(unittest.TestCase):
    @parameterized.expand(list(get_params()))
    def test_gradient_invgamma_entropy(self, name, vbayes, obs):
        α, β = vbayes.α, vbayes.β

        g = vbayes.__gradient_invgamma_entropy__()

        # Compute approximate gradients and assert that the relative error to the analytical gradient is small
        ε = 1e-6
        integrand = lambda α, β, v: - invgamma(α, scale=1/β).pdf(v) * invgamma(α, scale=1/β).logpdf(v)
        Gα = quad(lambda v : (integrand(α + ε, β, v) - integrand(α - ε, β, v))/(2 * ε), 0, inf)[0]
        self.assertLess(abs(Gα - g.α)/g.α, 1e-3)
        Gβ = quad(lambda v : (integrand(α, β + ε, v) - integrand(α, β - ε, v))/(2 * ε), 0, inf)[0]
        self.assertLess(abs(Gβ - g.β)/g.β, 1e-3)


    @parameterized.expand(list(get_params()))
    def test_gradient_normal_entropy(self, name, vbayes, obs):
        µ, σ = vbayes.µ, vbayes.σ

        g = vbayes.__gradient_normal_entropy__()

        # Compute approximate gradients and assert that the relative error to the analytical gradient is small
        ε = 1e-6
        integrand = lambda μ, σ, x: - norm(loc=µ, scale=σ).pdf(x) * norm(loc=µ, scale=σ).logpdf(x)
        for i in range(self.n):
            Gμ = quad(lambda x : (integrand(µ[i] + ε, σ[i], x) - integrand(µ[i] - ε, σ[i], x))/(2 * ε), -inf, inf)[0]
            self.assertAlmostEqual(abs(Gμ - g.μ[i])/g.μ[i], 0.0)
            Gσ = quad(lambda x : (integrand(µ[i], σ[i] + ε, x) - integrand(µ[i], σ[i] - ε, x))/(2 * ε), -inf, inf)[0]
            self.assertLess(abs(Gσ - g.σ[i])/g.σ[i], 1e-3)

    @parameterized.expand(list(get_params()))
    def test_gradient_gamma_cross_entropy(self, name, vbayes, obs):
        α, β = vbayes.α, vbayes.β

        g = vbayes.__gradient_gamma_cross_entropy__()

        ε = 1e-6
        integrand = lambda α, β, v: - invgamma(α, scale=1/β).pdf(v) * invgamma(vbayes.model.α, scale=1/vbayes.model.β).logpdf(v)

        Gα = quad(lambda v : (integrand(α + ε, β, v) - integrand(α - ε, β, v))/(2 * ε), 0, inf)[0]
        self.assertLess((Gα - g.α) / abs(g.α), 1e-3)
        Gβ = quad(lambda v : (integrand(α, β + ε, v) - integrand(α, β - ε, v))/(2 * ε), 0, inf)[0]
        self.assertLess((Gβ - g.β) / abs(g.β), 1e-3)

    @parameterized.expand(list(get_params()))
    def test_gradient_normal_cross_entropy(self, name, vbayes, obs):
        α, β = vbayes.α, vbayes.β
        μ, σ = vbayes.μ, vbayes.σ

        g = vbayes.__gradient_normal_cross_entropy__()

        ε = 1e-6
        integrand = lambda α, β, μ, σ, v, z: -invgamma(α, scale=1/β).pdf(v) * norm(loc=0, scale=σ).pdf(z) * norm(loc=µ, scale=sqrt(v)).logpdf(z)

        Gα, Gβ = 0, 0
        for i in range(self.n):
            Gα += dblquad(lambda v, z: (integrand(α + ε, β, μ[i], σ[i], v, z) - integrand(α - ε, β, μ[i], σ[i], v, z))/(2 * ε), 0, inf, -inf, inf)[0]
            Gβ += dblquad(lambda v, z: (integrand(α, β + ε, μ[i], σ[i], v, z) - integrand(α, β - ε, μ[i], σ[i], v, z))/(2 * ε), 0, inf, -inf, inf)[0]
            Gμ = dblquad(lambda v, z: (integrand(α, β, μ[i] + ε, σ[i], v, z) - integrand(α, β, μ[i] - ε, σ[i], v, z))/(2 * ε), 0, inf, -inf, inf)[0]
            Gσ = dblquad(lambda v, z: (integrand(α, β, μ[i], σ[i] + ε, v, z) - integrand(α, β, μ[i], σ[i] - ε, v, z))/(2 * ε), 0, inf, -inf, inf)[0]
            self.assertLess(abs(Gμ - g.μ[i])/g.μ[i], 1e-3)
            self.assertLess(abs(Gσ - g.σ[i])/g.σ[i], 1e-3)
        self.assertLess(abs(Gα - g.α)/g.α, 1e-3)
        self.assertLess(abs(Gβ - g.β)/g.β, 1e-3)

    @parameterized.expand(list(get_params()))
    def test_gradient__observations(self, name, vbayes, obs):
        μ, σ = vbayes.μ, vbayes.σ
        ε = 1e-6

        g = vbayes.__gradient__observations__(obs)
        Gμ, Gσ = np.zeros(vbayes.model.n), np.zeros(vbayes.model.n)
        for (i, j) in obs:
            """ gradient of  Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ """
            σδ = sqrt(σ[i]**2 + σ[j]**2)
            μδ = μ[i] - μ[j]

            # ∫ Normal(μ, σ), δ) log(1 + e^(-δ)) ~ ∫ Normal(μ, σ), δ) (2 e^(-π x²/16) / π - (1/2) x erfc(√(π) x / 4)) dδ
            true_integrand = lambda δ: - norm(loc=μδ, scale=σδ).pdf(δ) * log(1 + exp(-δ))
            approx_integrand = lambda δ: - norm(loc=μδ, scale=σδ).pdf(δ) * (2 * exp(-π * δ**2 / 16) / π - (1/2) * δ * erfc(sqrt(π) * δ / 4))
            Gμ[i] += quad(lambda δ: (approx_integrand(δ + ε) - approx_integrand(δ - ε))/(2 * ε), -inf, inf)[0]
            Gμ[j] -= quad(lambda δ: (approx_integrand(δ + ε) - approx_integrand(δ - ε))/(2 * ε), -inf, inf)[0]
            Gσ[i] += quad(lambda δ: (approx_integrand(δ + ε) - approx_integrand(δ - ε))/(2 * ε), -inf, inf)[0] * (σ[i]/σδ)
            Gσ[j] += quad(lambda δ: (approx_integrand(δ + ε) - approx_integrand(δ - ε))/(2 * ε), -inf, inf)[0] * (σ[j]/σδ)

            approx_integral = quad(approx_integrand, -inf, inf)[0]
            true_integral = quad(true_integrand, -inf, inf)[0]
            self.assertLess(abs(approx_integral - true_integral)/true_integral, 1e-3) # dubious but we'll see

        for i in range(vbayes.model.n):
            self.assertLess(abs(Gμ[i] - g.μ[i])/g.μ[i], 1e-3)
            self.assertLess(abs(Gσ[i] - g.σ[i])/g.σ[i], 1e-3)
