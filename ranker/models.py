from scipy.stats import invgamma, norm
from scipy.special import gammaln, gamma, polygamma, erf, erfc, expit, log_expit
from scipy.integrate import quad
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π, inf
import numpy as np
import copy



def almost_log_expit(x):
    return - 2 * exp(- π * x**2 / 16) / π + x / 2 * erfc(sqrt(π) * x / 4)

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
        v = invgamma.rvs(self.α, scale=1/self.β)
        return Instance(v, norm.rvs(loc=0, scale=sqrt(v), size=self.n))

    def logP(self, instance, obs):
        return (
            invgamma(self.α, scale=1/self.β).logpdf(instance.v) + norm(loc=0, scale=sqrt(instance.v)).logpdf(instance.z).sum() +
            # log_expit([instance.z[o[0]] - instance.z[o[1]] for o in obs]).sum()
            almost_log_expit(np.array([instance.z[o[0]] - instance.z[o[1]] for o in obs])).sum())

class Instance():

    def __init__(self, v, z):
        self.v = v
        self.z = z
        self.n = len(z)

    def __repr__(self) -> str:
        return f"Instance(v={self.v}, z={self.z})"

    def observe(self, n_obs):
        obs = []
        for _ in range(n_obs):
            i = np.random.randint(self.n)
            while True:
                j = np.random.randint(self.n)
                if i != j:
                    break
            if np.random.rand() < expit(self.z[i] - self.z[j]):
                obs.append((i, j))
            else:
                obs.append((j, i))
        return obs

class Gradient():

    def __init__(self, α = None, β = None, μ = None, σ = None, value = None):
        self.α = α
        self.β = β
        self.μ = μ
        self.σ = σ
        self.value = value

    def to_vector(self):
        return np.concatenate([[self.α, self.β], self.μ, self.σ])

    # addition and subtraction
    def __add__(self, other):
        def add_monad(x, y):
            if x is None and y is None:
                return None
            elif x is None:
                return y
            elif y is None:
                return x
            else:
                return x + y

        return Gradient(
            α = add_monad(self.α, other.α),
            β = add_monad(self.β, other.β),
            μ = add_monad(self.μ, other.μ),
            σ = add_monad(self.σ, other.σ),
            value = add_monad(self.value, other.value)
        )

    def __sub__(self, other):
        def sub_monad(x, y):
            if x is None and y is None:
                return None
            elif x is None:
                return -y
            elif y is None:
                return x
            else:
                return x - y
        return Gradient(
            α = sub_monad(self.α, other.α),
            β = sub_monad(self.β, other.β),
            μ = sub_monad(self.μ, other.μ),
            σ = sub_monad(self.σ, other.σ),
            value = sub_monad(self.value, other.value)
        )

    def __repr__(self) -> str:
        return f"Gradient(α={self.α}, β={self.β}, μ={self.μ}, σ={self.σ}, value={self.value})"

    def __str__(self) -> str:
        return self.__repr__()


class VBayes():

    def __init__(self, model):
        self.model = model
        self.µ = zeros(model.n)
        self.n = model.n

        # Given that the variance is picked from an inverse gamma distribution, the
        # posterior should be a student t distribution with 2 a degrees of freedom
        # and scaling factor √(b/a). We minimize the KL divergence between a normal
        # and that distribution by picking a standard deviation approximately equal to
        # √(b/a) (1 + 3 / (6 a + 2 a²))
        self.σ = ones(model.n) * sqrt(model.β  / model.α) * (1 + 3 / (6 * model.α + 2 * model.α**2))

        # Parameters of the distribution reprensting the posterior of model.v
        self.α = model.α
        self.β = model.β


    def logQ(self, instance):
        return invgamma(self.α, scale=1/self.β).logpdf(instance.v) + norm(loc=self.µ, scale=self.σ).logpdf(instance.z).sum()

    def rvs(self):
        return Instance(invgamma.rvs(self.α, scale=1/self.β), norm.rvs(loc=self.µ, scale=self.σ))


    def __gradient_invgamma_entropy__(self):
        """" The gradient of the entropy of an inverse gamma distribution with respect to its parameters. """
        α, β = self.α, self.β
        gα = 1 - (1 + α) * polygamma(1, α)
        gβ = - 1 / β
        return Gradient(α = gα, β = gβ, value = invgamma.entropy(α, scale=1/β))

    def __gradient_normal_entropy__(self):
        """ The gradient of the entropy of a normal distribution with respect to its parameters. """
        σ = self.σ
        gµ = zeros(self.model.n) # µ does not contribute to the entropy
        gσ = 1 / σ
        return Gradient(µ = gµ, σ = gσ, value = norm.entropy(scale=σ).sum())

    def __gradient_gamma_cross_entropy__(self):
        """ gradient of ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv"""
        α, β = self.α, self.β
        gα = β / self.model.β - (1 + self.model.α) * polygamma(1, α)
        gβ  = α / self.model.β - (1 + self.model.α) / β
        value = α * β / self.model.β + self.model.α * log(self.model.β) + gammaln(self.model.α) - (1 + self.model.α) * (log(β) + polygamma(0, α))
        return Gradient(α = gα, β = gβ, value = value)

    def __gradient_normal_cross_entropy__(self):
        """ gradient of ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz """
        α, β = self.α, self.β
        µ, σ = self.µ, self.σ
        n = self.model.n
        gμ = α * β * μ
        gσ = α * β * σ
        gα = 1/2 * (β * (μ**2 + σ**2).sum() - n * polygamma(1, α))
        gβ = 1/2 * (α * (μ**2 + σ**2).sum() - n / β)
        value = 1/2 * (α * β * (μ**2 + σ**2).sum() + self.model.n * (log(2 * π / β) - polygamma(0, α)))
        return Gradient(α = gα, β = gβ, μ = gμ, σ = gσ, value = value)

    def __gradient_observations__(self, obs):
        """ gradient of  Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ """
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

        µ, σ = self.µ, self.σ

        gμ = zeros(self.model.n)
        gσ = zeros(self.model.n)
        value = 0
        for (i, j) in obs:

            σδ = sqrt(σ[i]**2 + σ[j]**2)
            μδ = μ[i] - μ[j]
            h = sqrt(16/π + 2 * σδ**2)

            value -= exp(-(μδ/h)**2) *  h / (2 * sqrt(π)) - 1/2 * μδ * (1 - erf(μδ/h))

            gμδ = - 1 / 2 * (1 - erf(μδ / h))
            gσδ = - exp(-(μδ/h)**2) * σδ / (sqrt(π) * h)

            gμ[i] += -gμδ
            gμ[j] += gμδ
            gσ[i] += gσδ * σ[i] / σδ
            gσ[j] += gσδ * σ[j] / σδ

        return Gradient(μ = gμ, σ = gσ, value = value)

    def gradient(self, obs):
        # Compute the gradient of the paramters (µ, σ, α, β) with respect to the KL divergence
        # of the approximate posterior with the true posterior.

        # sum Q(Z) log Q(Z) / P(Z|X) = sum Q(Z) log Q(Z) - sum Q(Z) log P(Z|X)
        # start with the first one, the entropy

        α, β = self.α, self.β
        µ, σ = self.µ, self.σ

        g = Gradient()

        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))
        g -= self.__gradient_invgamma_entropy__()


        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))
        g -= self.__gradient_normal_entropy__()

        # ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv
        g += self.__gradient_gamma_cross_entropy__()

        # ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz
        g += self.__gradient_normal_cross_entropy__()

        # Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ
        g -= self.__gradient_observations__(obs)

        if np.any(np.isnan(g.to_vector())):
            raise ValueError("NaN in gradient")
        return g

    def fit(self, obs, niter=100000, lr=0.01):

        for i in range(niter):
            gradient = self.gradient(obs)
            self.µ -= lr * gradient.µ
            self.σ -= lr * gradient.σ
            self.α -= lr * gradient.α
            self.β -= lr * gradient.β

            # to check gradients
            #new_val = self.gradient(obs).value
            #print((new_val - gradient.value), -lr * (gradient.α**2 + gradient.β**2 + (gradient.σ**2 + gradient.µ**2).sum()))

            if i % 1000 == 0:
                print(i, gradient.value)



            # ensure that the parameters are valid
            self.σ = np.maximum(self.σ, 1e-6)
            self.α = np.maximum(self.α, 1e-6)
            self.β = np.maximum(self.β, 1e-6)

            if i % 10 == 0:
                print(self, gradient.value, "KL~", self.KL(obs))



    def v95(self):
        return sqrt(invgamma(self.α, scale=1/self.β).interval(0.95))

    def KL(self, obs, max_iter=2000, accuracy_goal=1e-6):
        # Compute the KL divergence between the approximate posterior and the true posterior using monte carlo sampling

        cum_sum, cum_sum2 = 0, 0
        for i in range(max_iter):
            instance = self.rvs()
            lq = self.logQ(instance)
            lp = self.model.logP(instance, obs)
            sample = (lq - lp)
            cum_sum += sample
            cum_sum2 += sample**2
            mean = cum_sum / (i + 1)
            var = cum_sum2 / (i + 1) - mean**2

            # 95% interval
            interval = 1.96 * sqrt(var / (i + 1))
            if np.abs(interval / mean) < accuracy_goal and i > 100:
                return (mean - interval, mean + interval)
        return (mean - interval, mean + interval)

    def __str__(self):
        return f"Q(α={self.α}, β={self.β}, µ={self.µ}, σ={self.σ}, v~{self.v95()})"


def test():
    m = Model(n=5)
    v = VBayes(m)

    instance = m.rvs()
    obs = instance.observe(100)
    print(obs)

    print(instance)
    print(v)
    v.fit(obs)
    print(v)


if __name__ == '__main__':
    test()
