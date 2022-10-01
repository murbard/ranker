from scipy.stats import invgamma, norm
from scipy.special import gammaln, gamma, polygamma, erf, expit, log_expit
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π
import numpy as np
import copy

class Model():

    def __init__(self, n, α=1.2, β=0.5):
        self.n = n
        self.α = α
        self.β = β

        # Consider the binomial distribution over 10 stars, obtained by flipping a coin
        # ten times and summing the number of heads. It has a standard deviation of
        # √(5/2) ~ 1.5811. The difference between two adjacent start ratings is thus
        # √(2/5) ~ 0.6325.

        # Based on this intuition, we call a difference of √(2/5) standard deviation
        # a "star" of difference.

        # Assuming this is an intuitive notion of a "star", what are odds
        # should an additional star confer to an item in a heads up match?

        # If those comparisons are hard to judge and very subjective, maybe only 55% of the time?
        # If they're very clear, perhaps over 95% of the time?

        # Using a logistic rule:
        # 55% would correspond to a standard deviation of √(5/2) Log(11/9) ~ 0.317
        # 99% would correspond to a standard deviation of √(5/2) Log[19] ~ 4.656

        # None of this is very rigorous, but it gives use some sense of the scale
        # of standard deviations we're dealing with. Clearly 0.0001 and 100 are silly.

        # These roughlt correspond to the 1% and 99% tails of the InverseGamma(1.2, 0.5)
        # distribution for the variance.
        # chance of beat between adjacent stars



    def entropy(self):
        return invgamma.entropy(self.α, scale=1/self.β) + norm.entropy(scale=sqrt(self.v)) * self.n

    def rvs(self):
        return {'v': invgamma.rvs(self.α, scale=1/self.β), 'z': norm.rvs(loc=0, scale=sqrt(self.v), size=self.n)}


    def logP(self, v, z, obs):
        return invgamma(self.α, scale=1/self.β).logpdf(v) + norm(loc=0, scale=sqrt(v)).logpdf(z).sum() + log_expit([z[o[0]] - z[o[1]] for o in obs]).sum()


class VBayes():

    def __init__(self, model):
        self.model = model
        self.µ = zeros(model.n)

        # Given that the variance is picked from an inverse gamma distribution, the
        # posterior should be a student t distribution with 2 a degrees of freedom
        # and scaling factor √(b/a). We minimize the KL divergence between a normal
        # and that distribution by picking a standard deviation approximately equal to
        # √(b/a) (1 + 3 / (6 a + 2 a²))
        self.σ = ones(model.n) * sqrt(model.β  / model.α) * (1 + 3 / (6 * model.α + 2 * model.α**2))

        # Parameters of the distribution reprensting the posterior of model.v
        self.α = model.α
        self.β = model.β


    def logQ(self, z, v):
        return invgamma(self.α, scale=1/self.β).logpdf(v) + norm(loc=self.µ, scale=self.σ).logpdf(z).sum()

    def rvs(self):
        return {'v': invgamma.rvs(self.α, scale=1/self.β), 'z': norm.rvs(loc=self.µ, scale=self.σ)}

    def gradient(self, obs):
        # Compute the gradient of the paramters (µ, σ, α, β) with respect to the KL divergence
        # of the approximate posterior with the true posterior.

        # sum Q(Z) log Q(Z) / P(Z|X) = sum Q(Z) log Q(Z) - sum Q(Z) log P(Z|X)
        # start with the first one, the entropy

        α, β = self.α, self.β
        µ, σ = self.µ, self.σ

        # H = - invgamma(α, scale=1/β).entropy() - norm(loc=µ, scale=sqrt(σ)).entropy().sum()

        dKL_dα = (1 + α) * polygamma(1, α) - 1
        dKL_dβ = 1 / β


        dKL_dµ = zeros(self.model.n) # µ does not contribute to the entropy
        dKL_dσ = - 1 / σ


        # ∫ InvGamma(α, β, v) log( InvGamma(α', β', v)) dv
        dKL_dα -= (1 + self.model.α) * polygamma(1, α) - β / self.model.β
        dKL_dβ -= - α / self.model.β + (1 + self.model.α) / β

        # ∫∫ InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz
        dKL_dμ -= - α * β * μ
        dKL_dσ -= - α * β * σ
        dKL_dα -= - β / 2 * (μ**2 +  σ**2).sum() + 1/2 * polygamma(1, α)
        dKL_dβ -= - α / 2 * (μ**2 +  σ**2).sum() + 1/(2 * β)

        # Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ

        # let σ = √(σi²+σj²) and μ = μi - μj

        # dμ / dμi = 1
        # dμ / dμj = -1
        # dσ / dσi = σi / √(σi^2 + σj^2)
        # dσ / dσj = σj / √(σi^2 + σj^2)


        # This can be approximated by the approximation to the logistic-normal integral

        # by approximating the logistic sigmoid with the erf function, and integrating that
        # to get an approximation of the cumulative function log(1 + e^(-x))

        # log(1+ e^(-x)) ~ 2 e^(-π x²/16) / π - (1/2) x erfc(√(π) x / 4)

        # ∫ Normal(μ, σ), δ) log(1 + e^(-δ)) ~ ∫ Normal(μ, σ), δ) (2 e^(-π x²/16) / π - (1/2) x erfc(√(π) x / 4)) dδ
        # after some heroic integration we get that the right term is
        # e^(-π μ²/(16 + 2 π σ²)) √(8 + π σ²) / (√2 π) - 1/2 μ erfc(√(π) μ / √(16 + 2 π σ²))

        # let h = √((16 + 2 π σ²)/π)
        # the derivative wrt μ is:  - 1/2 erfc(μ / h)
        # the derivative wrt σ is:  exp(-(μ/h)²) σ / (√π h)


        for (i, j) in obs:

            σδ = sqrt(σ[i]**2 + σ[j]**2)
            μδ = μ[i] - μ[j]
            h = sqrt(16/π + 2 * σδ**2)

            dKL_dμδ = - 1 / 2 * (1 - erf(μδ / h))
            dKL_dσδ = exp(-(μδ/h)**2) * σδ / (sqrt(π) * h)

            dKL_dμ[i] -= -dKL_dμδ
            dKL_dμ[j] -= dKL_dμδ
            dKL_dσ[i] -= dKL_dσδ * σ[i] / σδ
            dKL_dσ[j] -= dKL_dσδ * σ[j] / σδ

        return np.concatenate([dKL_dμ, dKL_dσ, [dKL_dα, dKL_dβ]])


def test():
    m = Model(n=3)
    v = VBayes(m)

    v.α = 1.3
    v.β = 2.6
    v.µ = np.array([1.5,2.1,3.1])
    v.σ = np.array([2.5,0.3,1.2])

    ε = 1e-10

    vα = copy.copy(v) ; v.α += ε
    vβ = copy.copy(v) ; v.β += ε
    vµ0 = copy.copy(v) ; v.µ[0] += ε
    vµ1 = copy.copy(v) ; v.µ[1] += ε
    vµ2 = copy.copy(v) ; v.µ[2] += ε
    vσ0 = copy.copy(v) ; v.σ[0] += ε
    vσ1 = copy.copy(v); v.σ[1] += ε
    vσ2 = copy.copy(v) ; v.σ[2] += ε


    KL = 0 ; KLα = 0 ; KLβ = 0 ; KLµ0 = 0 ; KLµ1 = 0 ; KLµ2 = 0 ; KLσ0 = 0 ; KLσ1 = 0 ; KLσ2 = 0

    for i in range(0, 1000):
        draw = v.rvs()
        logp = m.logP(**draw, obs=[(0,1), (1,2), (0,1)])
        KL += v.logQ(**draw) - logp
        KLα += vα.logQ(**draw) - logp
        KLβ += vβ.logQ(**draw) - logp
        KLµ0 += vµ0.logQ(**draw) - logp
        KLµ1 += vµ1.logQ(**draw) - logp
        KLµ2 += vµ2.logQ(**draw) - logp
        KLσ0 += vσ0.logQ(**draw) - logp
        KLσ1 += vσ1.logQ(**draw) - logp
        KLσ2 += vσ2.logQ(**draw) - logp

    # Average
    KL /= 1000
    KLα /= 1000
    KLβ /= 1000
    KLµ0 /= 1000
    KLµ1 /= 1000
    KLµ2 /= 1000
    KLσ0 /= 1000
    KLσ1 /= 1000
    KLσ2 /= 1000

    # Estimate gradients
    dKL_dα = (KLα - KL) / ε
    dKL_dβ = (KLβ - KL) / ε
    dKL_dµ = np.array([KLµ0 - KL, KLµ1 - KL, KLµ2 - KL]) / ε
    dKL_dσ = np.array([KLσ0 - KL, KLσ1 - KL, KLσ2 - KL]) / ε

    # Analytical gradients
    dKL = v.gradient(obs=[(0,1), (1,2), (0,1)])

    # Compare


    print("dKL_dµ", dKL_dµ, dKL[0:3])
    print("dKL_dσ", dKL_dσ, dKL[3:6])
    print("dKL_dα", dKL_dα, dKL[6])
    print("dKL_dβ", dKL_dβ, dKL[7])


if __name__ == '__main__':
    test()