
# class TestGradients(unittest.TestCase):

#     def setUp(self):
#         np.random.seed(42)




#     @parameterized.expand(params)
#     def test_gradient_invgamma_entropy(self, name, vbayes, obs):
#         α, β = vbayes.α, vbayes.β

#         g = vbayes.__gradient_invgamma_entropy__()

#         # Compute approximate gradients and assert that the relative error to the analytical gradient is small
#         ε = 1e-6
#         integrand = lambda α, β, v: - invgamma(α, scale=1/β).pdf(v) * invgamma(α, scale=1/β).logpdf(v)
#         Gα = quad(lambda v : (integrand(α + ε, β, v) - integrand(α - ε, β, v))/(2 * ε), 0, inf)[0]
#         self.assertLess(abs(Gα - g.α)/g.α, 1e-3)
#         Gβ = quad(lambda v : (integrand(α, β + ε, v) - integrand(α, β - ε, v))/(2 * ε), 0, inf)[0]
#         self.assertLess(abs(Gβ - g.β)/g.β, 1e-3)
#         # Test the integral itself
#         val = quad(lambda v : integrand(α,β,v), 0, inf)[0]
#         self.assertLess(abs(val - g.value)/g.value, 1e-3)


#     @parameterized.expand(params)
#     def test_gradient_normal_entropy(self, name, vbayes, obs):
#         µ, σ = vbayes.µ, vbayes.σ

#         g = vbayes.__gradient_normal_entropy__()

#         # Compute approximate gradients and assert that the relative error to the analytical gradient is small
#         ε = 1e-6
#         integrand = lambda μ, σ, x: - norm(loc=µ, scale=σ).pdf(x) * norm(loc=µ, scale=σ).logpdf(x)
#         val = 0
#         for i in range(vbayes.model.n):
#             Gμ = quad(lambda x : (integrand(µ[i] + ε, σ[i], x) - integrand(µ[i] - ε, σ[i], x))/(2 * ε), -inf, inf)[0]
#             self.assertLess(abs(Gμ), 1e-10) # both of these ought to be 0
#             self.assertLess(abs(g.μ[i]), 1e-10)
#             Gσ = quad(lambda x : (integrand(µ[i], σ[i] + ε, x) - integrand(µ[i], σ[i] - ε, x))/(2 * ε), -inf, inf)[0]
#             self.assertLess(abs(Gσ - g.σ[i])/g.σ[i], 1e-3)
#             val += quad(lambda x: integrand(μ[i],σ[i],x), -inf, inf)[0]
#         self.assertLess(abs(val - g.value)/g.value, 1e-3)

#     @parameterized.expand(params)
#     def test_gradient_gamma_cross_entropy(self, name, vbayes, obs):
#         α, β = vbayes.α, vbayes.β

#         g = vbayes.__gradient_gamma_cross_entropy__()

#         ε = 1e-6
#         integrand = lambda α, β, v: - invgamma(α, scale=1/β).pdf(v) * invgamma(vbayes.model.α, scale=1/vbayes.model.β).logpdf(v)

#         Gα = quad(lambda v : (integrand(α + ε, β, v) - integrand(α - ε, β, v))/(2 * ε), 0, inf)[0]
#         self.assertLess((Gα - g.α) / abs(g.α), 1e-3)
#         Gβ = quad(lambda v : (integrand(α, β + ε, v) - integrand(α, β - ε, v))/(2 * ε), 0, inf)[0]
#         self.assertLess((Gβ - g.β) / abs(g.β), 1e-3)
#         # Test the integral itself
#         val = quad(lambda v: integrand(α,β,v), 0, inf)[0]
#         self.assertLess(abs(val - g.value)/g.value, 1e-3)

#     @parameterized.expand(params)
#     def test_gradient_normal_cross_entropy(self, name, vbayes, obs):
#         α, β = vbayes.α, vbayes.β
#         μ, σ = vbayes.μ, vbayes.σ

#         g = vbayes.__gradient_normal_cross_entropy__()

#         ε = 1e-6
#         # integrand = lambda α, β, μ, σ, v, z: -invgamma(α, scale=1/β).pdf(v) * norm(loc=μ, scale=σ).pdf(z) * norm(loc=0, scale=sqrt(v)).logpdf(z)
#         # dlbquad too slow, use the analytical solution

#         integral = lambda α, β, μ, σ : 1/2 * (log(2 * π / β) + α * β * (μ**2 + σ**2) - polygamma(0, α))

#         Gα, Gβ = 0, 0
#         for i in range(vbayes.n):

#             #Gα += dblquad(lambda v, z: (integrand(α + ε, β, μ[i], σ[i], v, z) - integrand(α - ε, β, μ[i], σ[i], v, z))/(2 * ε), -inf, inf, 0, inf)[0]
#             #Gβ += dblquad(lambda v, z: (integrand(α, β + ε, μ[i], σ[i], v, z) - integrand(α, β - ε, μ[i], σ[i], v, z))/(2 * ε), -inf, inf, 0, inf)[0]
#             #Gμ = dblquad(lambda v, z: (integrand(α, β, μ[i] + ε, σ[i], v, z) - integrand(α, β, μ[i] - ε, σ[i], v, z))/(2 * ε), -inf, inf, 0, inf)[0]
#             #Gσ = dblquad(lambda v, z: (integrand(α, β, μ[i], σ[i] + ε, v, z) - integrand(α, β, μ[i], σ[i] - ε, v, z))/(2 * ε), -inf, inf, 0, inf)[0]

#             Gα += (integral(α + ε, β, μ[i], σ[i]) - integral(α - ε, β, μ[i], σ[i]))/(2 * ε)
#             Gβ += (integral(α, β + ε, μ[i], σ[i]) - integral(α, β - ε, μ[i], σ[i]))/(2 * ε)
#             Gμ = (integral(α, β, μ[i] + ε, σ[i]) - integral(α, β, μ[i] - ε, σ[i]))/(2 * ε)
#             Gσ = (integral(α, β, μ[i], σ[i] + ε) - integral(α, β, μ[i], σ[i] - ε))/(2 * ε)

#             self.assertLess(abs(Gμ - g.μ[i])/g.μ[i], 1e-3)
#             self.assertLess(abs(Gσ - g.σ[i])/g.σ[i], 1e-3)
#         self.assertLess(abs(Gα - g.α)/g.α, 1e-3)
#         self.assertLess(abs(Gβ - g.β)/g.β, 1e-3)
#         # Test the integral itself
#         val = integral(α, β, μ, σ).sum()
#         self.assertLess(abs(val - g.value)/g.value, 1e-3)




#     @parameterized.expand(list(get_params()))
#     def test_gradient_observations(self, name, vbayes, obs):
#         μ, σ = vbayes.μ, vbayes.σ
#         ε = 1e-6

#         g = vbayes.__gradient_observations__(obs)
#         Gμ, Gσ = np.zeros(vbayes.model.n), np.zeros(vbayes.model.n)
#         val = 0
#         for (i, j) in obs:
#             """ gradient of  Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ """
#             σδ = sqrt(σ[i]**2 + σ[j]**2)
#             μδ = μ[i] - μ[j]

#             # - ∫ Normal(μ, σ), δ) log(1 + e^(-δ)) ~ - ∫ Normal(μ, σ), δ) (2 e^(-π x²/16) / π - (1/2) x erfc(√(π) x / 4)) dδ
#             true_integrand = lambda μδ, σδ, δ: norm(loc=μδ, scale=σδ).pdf(δ) * log_expit(δ)
#             approx_integrand = lambda μδ, σδ, δ: - norm(loc=μδ, scale=σδ).pdf(δ) * (2 * exp(-π * δ**2 / 16) / π - (1/2) * δ * erfc(sqrt(π) * δ / 4))

#             δμ = quad(lambda δ: (approx_integrand(μδ + ε, σδ, δ) - approx_integrand(μδ - ε, σδ, δ))/(2 * ε), -inf, inf)[0]
#             δσ = quad(lambda δ: (approx_integrand(μδ, σδ + ε, δ) - approx_integrand(μδ, σδ - ε, δ))/(2 * ε), -inf, inf)[0]

#             Gμ[i] += δμ
#             Gμ[j] -= δμ
#             Gσ[i] += δσ * σ[i] / σδ
#             Gσ[j] += δσ * σ[j] / σδ

#             val += quad(lambda δ: approx_integrand(μδ, σδ, δ), -inf, inf)[0]
#             # true_integral = quad(lambda δ: true_integrand(μδ, σδ, δ), -inf, inf)[0]
#             # self.assertLess(abs(approx_integral - true_integral)/true_integral, 1e-3) # dubious but we'll see

#         for i in range(vbayes.model.n):
#             if abs(g.μ[i]) > 1e-12:
#                 self.assertLess(abs(Gμ[i] - g.μ[i])/g.μ[i], 1e-3)
#             else:
#                 self.assertLess(abs(Gμ[i]), 1e-6)

#             if abs(g.σ[i]) > 1e-12:
#                 self.assertLess(abs(Gσ[i] - g.σ[i])/g.σ[i], 1e-3)
#             else:
#                 self.assertLess(abs(Gσ[i]), 1e-6)

#         # Test the integral itself
#         self.assertLess(abs(val - g.value)/g.value, 1e-3)
