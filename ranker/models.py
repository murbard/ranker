from scipy.stats import invgamma, norm
from scipy.special import gammaln, gamma, polygamma, erf, erfc, expit, log_expit
from scipy.integrate import quad
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π, inf
import numpy as np
import copy
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("error")



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
        return f"Instance(√v={np.sqrt(self.v)}, z={self.z})"

    def observe(self, n_obs):
        obs = np.zeros((self.n, self.n))
        for _ in range(n_obs):
            i = np.random.randint(self.n)
            while True:
                j = np.random.randint(self.n)
                if i != j:
                    break
            if np.random.rand() < expit(self.z[i] - self.z[j]):
                obs[i,j] += 1
            else:
                obs[j, i] += 1
        return obs

class GradientHessian():

    def __init__(self, n):
        self.val = 0
        self.n = n
        self.g = zeros(2*n+2)
        self.h = zeros((2*n+2, 2*n+2))
        # format: μ, σ, α, β

    def __add__(self, other):
        self.val += other.val
        self.g += other.g
        self.h += other.h
        return self

    def __sub__(self, other):
        self.val -= other.val
        self.g -= other.g
        self.h -= other.h
        return self

    def gα(self):
        return self.g[2 * self.n: 2 * self.n+1]

    def gβ(self):
        return self.g[2 * self.n + 1 : 2 * self.n + 2]

    def gμ(self):
        return self.g[:self.n]

    def gσ(self):
        return self.g[self.n:2*self.n]

    def hαα(self):
        return self.h[2 * self.n : 2 * self.n + 1, 2 * self.n : 2 * self.n + 1]

    def hαβ(self):
        return self.h[2 * self.n : 2 * self.n + 1, 2 * self.n + 1 : 2 * self.n + 2]

    def hββ(self):
        return self.h[2 * self.n + 1 : 2 * self.n + 2, 2 * self.n + 1 : 2 * self.n + 2]

    def hμμ(self):
        return self.h[:self.n, :self.n]

    def hμσ(self):
        return self.h[:self.n, self.n:2*self.n]

    def hσσ(self):
        return self.h[self.n:2*self.n, self.n:2*self.n]

    def hμα(self):
        return self.h[:self.n, 2*self.n : 2*self.n + 1]

    def hσα(self):
        return self.h[self.n:2*self.n, 2*self.n : 2*self.n + 1]

    def hμβ(self):
        return self.h[:self.n, 2*self.n+1 : 2*self.n + 2]

    def hσβ(self):
        return self.h[self.n:2*self.n, 2*self.n+1 : 2*self.n + 2]



class VBayes():

    def __init__(self, model):
        self.model = model

        self.params = zeros(2*model.n+2)
        self.n = model.n

        # Given that the variance is picked from an inverse gamma distribution, the
        # posterior should be a student t distribution with 2 a degrees of freedom
        # and scaling factor √(b/a). We minimize the KL divergence between a normal
        # and that distribution by picking a standard deviation approximately equal to
        # √(b/a) (1 + 3 / (6 a + 2 a²))
        self.params[self.n : 2 * self.n] = ones(model.n) * sqrt(model.β  / model.α) * (1 + 3 / (6 * model.α + 2 * model.α**2))

        # Parameters of the distribution reprensting the posterior of model.v
        self.params[2 * self.n] = model.α
        self.params[2 * self.n + 1] = model.β

    def μ(self):
        return self.params[:self.n]

    def σ(self):
        return self.params[self.n:2*self.n]

    def α(self):
        return self.params[2*self.n : 2*self.n + 1]

    def β(self):
        return self.params[2*self.n+1 : 2*self.n + 2]

    def logQ(self, instance):
        return invgamma(self.α()[0], scale=1/self.β()[0]).logpdf(instance.v) + norm(loc=self.µ(), scale=self.σ()).logpdf(instance.z).sum()

    def rvs(self):
        return Instance(invgamma.rvs(self.α()[0], scale=1/self.β()[0]), norm.rvs(loc=self.µ(), scale=self.σ()))


    def __gradient_invgamma_entropy__(self):
        """" The gradient of the entropy of an inverse gamma distribution with respect to its parameters. """
        α, β = self.α()[0], self.β()[0]
        gα = 1 - (1 + α) * polygamma(1, α)
        gβ = - 1 / β

        gh = GradientHessian(self.n)
        gh.val = invgamma.entropy(α, scale=1/β)
        gh.gα()[0] = gα
        gh.gβ()[0] = gβ # negative but we subtract that so this pushes beta *down*
        gh.hαα()[0] = -polygamma(1, α) - ((1 + α) * polygamma(2, α))
        gh.hββ()[0] = 1/β**2

        #return Gradient(α = gα, β = gβ, value = invgamma.entropy(α, scale=1/β))
        return gh

    def __gradient_normal_entropy__(self):
        """ The gradient of the entropy of a normal distribution with respect to its parameters. """
        σ = self.σ()
        #  gµ = zeros(self.model.n) # µ does not contribute to the entropy

        gh = GradientHessian(self.n)
        gh.val = norm.entropy(scale=σ).sum()
        gh.gσ()[:] = 1 / σ # we subtract the entropy gradient, so when we minimize this pushes σ to be bigger
        gh.hσσ()[:] = np.diag(-1/σ**2)

        return gh
        #return Gradient(µ = gµ, σ = gσ, value = norm.entropy(scale=σ).sum())

    def __gradient_gamma_cross_entropy__(self):
        """ gradient of ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv"""

        # we try to mimimize this one, this means this pushes α and β towards the hyperparameter's α and β
        α, β = self.α()[0], self.β()[0]
        gα = β / self.model.β - (1 + self.model.α) * polygamma(1, α)
        gβ  = α / self.model.β - (1 + self.model.α) / β
        value = α * β / self.model.β + self.model.α * log(self.model.β) + gammaln(self.model.α) - (1 + self.model.α) * (log(β) + polygamma(0, α))

        gαα = -(1 + self.model.α) * polygamma(2, α)
        gββ = (1 + self.model.α) / β**2
        gαβ = 1 / self.model.β

        gh = GradientHessian(self.n)
        gh.val = value

        gh.gα()[0] = gα
        gh.gβ()[0] = gβ
        gh.hαα()[0] = gαα
        gh.hββ()[0] = gββ
        gh.hαβ()[0] = gαβ

        return gh

        #return Gradient(α = gα, β = gβ, value = value)

    def __gradient_normal_cross_entropy__(self):
        """ gradient of ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz """
        α, β = self.α()[0], self.β()[0]
        µ, σ = self.µ(), self.σ()
        n = self.model.n
        gμ = α * β * μ
        gσ = α * β * σ # this pushes σ to be smaller
        gα = 1/2 * (β * (μ**2 + σ**2).sum() - n * polygamma(1, α))
        gβ = 1/2 * (α * (μ**2 + σ**2).sum() - n / β)
        value = 1/2 * (α * β * (μ**2 + σ**2).sum() + n * (log(2 * π / β) - polygamma(0, α)))

        gh = GradientHessian(self.n)
        gh.val = value
        gh.gα()[0] = gα
        gh.gβ()[0] = gβ
        gh.gμ()[:] = gμ
        gh.gσ()[:] = gσ


        hαα = - 1 / 2 * n * polygamma(2, α)
        hαβ = 1/2 * (μ**2 + σ**2).sum()
        hββ = n / (2 * β**2)
        hμα = β * μ
        hμβ = α * μ
        hσα = β * σ
        hσβ = α * σ
        hμμ = α * β * np.eye(n)
        hσσ = α * β * np.eye(n)

        gh.hαα()[0] = hαα
        gh.hαβ()[0] = hαβ
        gh.hββ()[0] = hββ
        gh.hμα()[:,0] = hμα
        gh.hμβ()[:,0] = hμβ
        gh.hσα()[:,0] = hσα
        gh.hσβ()[:,0] = hσβ
        gh.hμμ()[:] = hμμ
        gh.hσσ()[:] = hσσ

        return gh

        # return Gradient(α = gα, β = gβ, μ = gμ, σ = gσ, value = value)

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

        # note: if f(a,b) = g(c(a,b)) then
        # d²f/dadb = d²g/dc² (dc/da)(dc/db) + dg/dc d²c/(da db)

        µ, σ = self.µ(), self.σ()

        gh = GradientHessian(self.n)
        for i in range(0, self.n):
            for j in range(0, self.n):

                if obs[i, j] == 0:
                    continue
                else:
                    count = obs[i, j]


                σδ = sqrt(σ[i]**2 + σ[j]**2)
                μδ = μ[i] - μ[j]
                h = sqrt(16/π + 2 * σδ**2)

                gh.val -= count * (exp(-(μδ/h)**2) *  h / (2 * sqrt(π)) - 1/2 * μδ * (1 - erf(μδ/h)))

                gμδ = - 1 / 2 * (1 - erf(μδ / h)) # negative
                gσδ = - exp(-(μδ/h)**2) * σδ / (sqrt(π) * h)  # negative, but this is subtracted so, so this makes σ smaller

                hμμδ = - exp(-(μδ/h)**2) / (sqrt(π) * h)
                hμσδ = 2 * μδ * σδ * exp(-(μδ/h)**2) / (sqrt(π) * h**3)
                hσσδ = - exp(-(μδ/h)**2) * (256 + 4 * π * σδ**2 * (8 + μδ**2)) / (π**(5/2) * h**5)

                gh.gμ()[i] += count * (-gμδ)
                gh.gμ()[j] += count * gμδ

                gh.gσ()[i] += count * (gσδ * σ[i] / σδ)
                gh.gσ()[j] += count * (gσδ * σ[j] / σδ)

                gh.hμμ()[i,i] += count * hμμδ
                gh.hμμ()[j,j] += count * hμμδ
                gh.hμμ()[min(i,j), max(i,j)] += count * (-hμμδ)

                gh.hσσ()[i,i] += count * (hσσδ * (σ[i]/σδ)**2 + gσδ * σ[j]**2/σδ**3)
                gh.hσσ()[min(i,j), max(i,j)] += count * (hσσδ * (σ[i]*σ[j] / σδ**2) - gσδ * σ[i]*σ[j] / σδ**3)
                gh.hσσ()[j,j] += count * (hσσδ * (σ[j]/σδ)**2 + gσδ * σ[i]**2/σδ**3)

                gh.hμσ()[i,i] += count * hμσδ * σ[i] / σδ
                gh.hμσ()[i,j] += count * hμσδ * σ[j] / σδ
                gh.hμσ()[j,i] += count *  (-hμσδ * σ[i] / σδ)
                gh.hμσ()[j,j] += count *  (-hμσδ * σ[j] / σδ)


        # return Gradient(μ = gμ, σ = gσ, value = value)

        return gh

    def gradient(self, obs):
        # Compute the gradient of the paramters (µ, σ, α, β) with respect to the KL divergence
        # of the approximate posterior with the true posterior.

        # sum Q(Z) log Q(Z) / P(Z|X) = sum Q(Z) log Q(Z) - sum Q(Z) log P(Z|X)
        # start with the first one, the entropy

        gh = GradientHessian(self.n)

        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))
        gh -= self.__gradient_invgamma_entropy__()


        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))
        gh -= self.__gradient_normal_entropy__()

        # ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv
        gh += self.__gradient_gamma_cross_entropy__()

        # ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz
        gh += self.__gradient_normal_cross_entropy__()

        # Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ
        gh -= self.__gradient_observations__(obs)

        if np.any(np.isnan(gh.g)):
            raise ValueError("NaN in gradient")
        if np.any(np.isnan(gh.h)):
            raise ValueError("NaN in hessian")
        return gh

    def fit(self, obs, niter=100000, tol=1e-4, verbose=True):
        last_val = None
        for i in range(niter):
            gh = self.gradient(obs)
            if last_val and np.abs(last_val - gh.val)/np.abs(gh.val) < tol:
                break
            last_val = gh.val

            # BEGIN TEST
            # gh1 = self.__gradient_observations__(obs)
            # ε = np.random.randn(2 * self.n + 2) * 1e-5
            # ε[1:] = 0.0 # let's see if mu works

            # self.params += ε
            # gh2 = self.__gradient_observations__(obs)
            # self.params -= ε

            # first_order = gh1.g.dot(ε)
            # second_order = ε.dot(gh1.h - 0.5 * np.diag(np.diag(gh1.h))).dot(ε)
            # actual = gh2.val - gh1.val
            # if np.abs(actual) > 1e-20 and np.abs((actual - first_order)) > 1e-20:
            #     error = np.abs(actual - first_order) / np.abs(actual)
            #     print(f"first order relative error : {error}")
            #     second_order_error = np.abs((actual - first_order) - second_order) / np.abs(actual - first_order)
            #     print(f"second order relative error : {second_order_error}")

            # END TEST


            # Newton step

            # check the condition number of gh.h


            G = gh.g
            H = gh.h

            H[self.n:,self.n:] *= self.params[self.n:].reshape(-1,1).dot(self.params[self.n:].reshape(1,-1))
            G[self.n:] *= self.params[self.n:]

            # don't touch α and β
            #gh.g = gh.g[:-2]
            #gh.h = gh.h[:-2,:-2]

            # newton update for the log of positive parameters

            λ = 1e-8

            while True:
                try:
                    diff =  linalg.solve(gh.h + np.diag(np.diag(gh.h) * (1 + λ)), gh.g, assume_a='sym')

                    old_params = self.params.copy()
                    self.params[:self.n] -= diff[:self.n]
                    factor = exp(-diff[self.n:])
                    # factor[:-2] = np.clip(factor[:-2], 0.999, 1.001)
                    self.params[self.n:] *= factor

                    if self.gradient(obs).val < gh.val: # if it's better stop
                        break

                    else:
                        self.params = old_params
                        λ *= 10 # otherwise increase the conditioning

                except RuntimeWarning as r:
                    λ *= 10



            # Sanity clamp α, β, σ to be positive
            #self.params[self.n:] = np.clip(self.params[self.n:], 1e-6, inf)

            # Todo remove, prevent a and b from collpasing to see if that causes the ill conditioning



            #self.params[-1] = np.clip(self.params[-1], 0.01, inf)
            #self.params[-2] = np.clip(self.params[-1], 0.01, inf)



            # to check gradients
            #new_val = self.gradient(obs).value
            #print((new_val - gradient.value), -lr * (gradient.α**2 + gradient.β**2 + (gradient.σ**2 + gradient.µ**2).sum()))



            # ensure that the parameters are valid
            #self.σ = np.maximum(self.σ, 1e-6)
            #self.α = np.maximum(self.α, 1e-6)
            #self.β = np.maximum(self.β, 1e-6)

            if verbose:
                print(gh.val) # , "KL~", self.KL(obs))



    def v95(self):
        return sqrt(invgamma(self.α()[0], scale=1/self.β()[0]).interval(0.95))

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
        return f"Q(α={self.α()[0]}, β={self.β()[0]}, µ={self.µ()}, σ={self.σ()}, √v~{self.v95()})"


def test():
    m = Model(n=50)
    v = VBayes(m)

    instance = m.rvs()
    obs = instance.observe(50*50 * 20)
    print(v)



    # ε = 1e-8
    # np.core.arrayprint._line_width = 200

    # max_so_far = 0.0
    # for p,func in enumerate([lambda v, _: v.__gradient_invgamma_entropy__(),
    # lambda  v, _: v.__gradient_normal_entropy__(),
    # lambda  v, _: v.__gradient_gamma_cross_entropy__(),
    # lambda  v, _:  v.__gradient_normal_cross_entropy__(),
    # lambda  v, obs:  v.__gradient_observations__(obs)]):
    #     gh = func(v, obs)
    #     for i in range(0, 2 * v.n + 2):
    #         v.params[i] += ε
    #         gh2 = func(v, obs)
    #         v.params[i] -= ε
    #         mx = np.max(np.abs(((gh2.g - gh.g) / ε)[i:] -  gh.h[i,i:]))
    #         if mx > max_so_far:
    #             max_so_far = mx
    #             best = (p, i, np.argmax(np.abs(((gh2.g - gh.g) / ε)[i:] -  gh.h[i,i:])))
    #             if mx > 1e-6:
    #                 print('plouf')

    #         print(i, ((gh2.g - gh.g) / ε)[i:] -  gh.h[i,i:],'\n\n')
    # print(max_so_far, best)

    v.fit(obs, verbose=True)

    class Memoizer():

        def __init__(self, vbayes):
            self.gh = None
            self.vbayes = vbayes
            self.n = vbayes.n

        def adjust_params(self, params):
            params[self.n:] = np.exp(params[self.n:])
            return

        def adjust_gh(self, gh, params):
            self.gh.h[self.n:,self.n:] *= params[self.n:].reshape(-1,1).dot(params[self.n:].reshape(1,-1))
            self.gh.g[self.n:] *= params[self.n:]

        def f(self, params):
            self.adjust_params(params)
            if np.any(params != self.vbayes.params) or self.gh is None:
                self.vbayes.params = params
                self.gh = self.vbayes.gradient(obs)
                self.adjust_gh(self.gh, params)

            return self.gh.val

        def grad(self, params):
            self.adjust_params(params)
            if np.any(params != self.vbayes.params):
                self.vbayes.params = params
                self.gh = self.vbayes.gradient(obs)
                self.adjust_gh(self.gh, params)
            return self.gh.g

        def hess(self, params):
            self.adjust_params(params)
            if np.any(params != self.vbayes.params):
                self.vbayes.params = params
                self.gh = self.vbayes.gradient(obs)
                self.adjust_gh(self.gh, params)
            return self.gh.h

    memo = Memoizer(v)
    params = v.params.copy()
    params[v.n:] = np.log(params[v.n:])
    #minimize(memo.f, params, jac=memo.grad, hess=memo.hess, method='trust-ncg', options={'maxiter': 100000, 'disp': True})




    print(v, instance)
    from matplotlib import pyplot as plt
    plt.scatter(v.μ(), instance.z)
    plt.show()


if __name__ == '__main__':
    test()
