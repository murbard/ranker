from scipy.stats import invgamma, norm
from scipy.special import gammaln, gamma, polygamma, erf, erfc, expit
from scipy.integrate import quad
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π, inf
import numpy as np
import copy
from scipy import linalg
from scipy.optimize import minimize
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("error")
from line_profiler import LineProfiler

try:
    from scipy.special import log_expit
except ImportError:
    def log_expit(x):
        return  -log(1 + exp(-np.abs(x))) + min(x, 0)


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

    def match(self, i, j):
        return np.random.rand() < expit(self.z[i] - self.z[j])

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

    def entropy(self):
        return invgamma(self.α(), scale=1/self.β()).entropy() + norm(loc=self.μ(), scale=self.σ()).entropy().sum()

    def rvs(self):
        return Instance(invgamma.rvs(self.α()[0], scale=1/self.β()[0]), norm.rvs(loc=self.µ(), scale=self.σ()))


    def __minus_invgamma_entropy__(self, compute_gradient = True, compute_hessian = True, out = None):
        """" The gradient of the entropy of an inverse gamma distribution with respect to its parameters. """

        α, β = self.α()[0], self.β()[0]

        gh = out if out is not None else GradientHessian(self.n)

        gh.val -= invgamma.entropy(α, scale=1/β)

        if compute_gradient:
            gh.gα()[0] += (1 + α) * polygamma(1, α) - 1
            gh.gβ()[0] += 1 / β

        if compute_hessian:
            gh.hαα()[0] += polygamma(1, α) + ((1 + α) * polygamma(2, α))
            gh.hββ()[0] += - 1 / β**2

        return gh

    def __minus_normal_entropy__(self, compute_gradient = True, compute_hessian = True, out = None):
        """ The gradient of the entropy of a normal distribution with respect to its parameters. """
        σ = self.σ()

        gh = out if out is not None else GradientfHessian(self.n)

        gh.val -= norm.entropy(scale=σ).sum()
        if compute_gradient:
            gh.gσ()[:] += - 1 / σ # we subtract the entropy gradient, so when we minimize this pushes σ to be bigger
        if compute_hessian:
            gh.hσσ()[:] += np.diag(1 / σ**2)

        return gh

    def __gamma_cross_entropy__(self, compute_gradient = True, compute_hessian = True, out = None):
        """ gradient of ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv"""

        # we try to mimimize this one, this means this pushes α and β towards the hyperparameter's α and β
        α, β = self.α()[0], self.β()[0]

        gh = out if out is not None else GradientHessian(self.n)

        gh.val += α * β / self.model.β + self.model.α * log(self.model.β) + gammaln(self.model.α) - (1 + self.model.α) * (log(β) + polygamma(0, α))

        if compute_gradient:
            gh.gα()[0] += β / self.model.β - (1 + self.model.α) * polygamma(1, α)
            gh.gβ()[0] += α / self.model.β - (1 + self.model.α) / β

        if compute_hessian:
            gh.hαα()[0] += -(1 + self.model.α) * polygamma(2, α)
            gh.hββ()[0] += (1 + self.model.α) / β**2
            gh.hαβ()[0] += 1 / self.model.β

        return gh

    def __normal_cross_entropy__(self, compute_gradient = False, compute_hessian = False, out = None):
        """ gradient of ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz """
        α, β = self.α()[0], self.β()[0]
        µ, σ = self.µ(), self.σ()
        n = self.model.n

        gh = out if out is not None else GradientHessian(self.n)

        gh.val += 1/2 * (α * β * (μ**2 + σ**2).sum() + n * (log(2 * π / β) - polygamma(0, α)))

        if compute_gradient:
            gh.gμ()[:] += α * β * μ
            gh.gσ()[:] += α * β * σ # this pushes σ to be smaller
            gh.gα()[0] += 1/2 * (β * (μ**2 + σ**2).sum() - n * polygamma(1, α))
            gh.gβ()[0] += 1/2 * (α * (μ**2 + σ**2).sum() - n / β)

        if compute_hessian:
            gh.hαα()[0] += - 1 / 2 * n * polygamma(2, α)
            gh.hαβ()[0] += 1/2 * (μ**2 + σ**2).sum()
            gh.hββ()[0] += n / (2 * β**2)
            gh.hμα()[:,0] += β * μ
            gh.hμβ()[:,0] += α * μ
            gh.hσα()[:,0] += β * σ
            gh.hσβ()[:,0] += α * σ
            gh.hμμ()[:] += α * β * np.eye(n)
            gh.hσσ()[:] += α * β * np.eye(n)

        return gh

    def __minus_observations__(self, obs, compute_gradient = True, compute_hessian = True, dobs = None, out = None):
        """ Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ """
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

        if compute_hessian and not compute_gradient:
            raise "Must compute the gradient if the hessian is being computed"

        if dobs is not None and not compute_gradient:
            raise "Must compute the gradient if dobs is being computer"


        µ, σ = self.µ(), self.σ()

        gh = out if out is not None else GradientHessian(self.n)

        for i in range(0, self.n):
            for j in range(0, self.n):
                if obs[i, j] == 0 and dobs is None:
                    continue
                else:
                    count = obs[i, j]

                σδ = sqrt(σ[i]**2 + σ[j]**2)
                μδ = μ[i] - μ[j]
                h = sqrt(16/π + 2 * σδ**2)

                gh.val += count * (exp(-(μδ/h)**2) *  h / (2 * sqrt(π)) - 1/2 * μδ * (1 - erf(μδ/h)))

                if compute_gradient:

                    gμδ = - 1 / 2 * (1 - erf(μδ / h)) # negative
                    gσδ = - exp(-(μδ/h)**2) * σδ / (sqrt(π) * h)  # negative, but this is subtracted so, so this makes σ smaller

                    gh.gμ()[i] -= count * (-gμδ)
                    gh.gμ()[j] -= count * gμδ

                    gh.gσ()[i] -= count * (gσδ * σ[i] / σδ)
                    gh.gσ()[j] -= count * (gσδ * σ[j] / σδ)

                    if dobs is not None:
                            dobs[i,j, 0] = - gσδ  / σδ
                            dobs[i, j, 1] = - gμδ

                    if compute_hessian:

                        hμμδ = - exp(-(μδ/h)**2) / (sqrt(π) * h)
                        hμσδ = 2 * μδ * σδ * exp(-(μδ/h)**2) / (sqrt(π) * h**3)
                        hσσδ = - exp(-(μδ/h)**2) * (256 + 4 * π * σδ**2 * (8 + μδ**2)) / (π**(5/2) * h**5)

                        gh.hμμ()[i,i] -= count * hμμδ
                        gh.hμμ()[j,j] -= count * hμμδ
                        gh.hμμ()[min(i,j), max(i,j)] -= count * (-hμμδ)

                        gh.hσσ()[i,i] -= count * (hσσδ * (σ[i]/σδ)**2 + gσδ * σ[j]**2/σδ**3)
                        gh.hσσ()[min(i,j), max(i,j)] -= count * (hσσδ * (σ[i]*σ[j] / σδ**2) - gσδ * σ[i]*σ[j] / σδ**3)
                        gh.hσσ()[j,j] -= count * (hσσδ * (σ[j]/σδ)**2 + gσδ * σ[i]**2/σδ**3)

                        gh.hμσ()[i,i] -= count * hμσδ * σ[i] / σδ
                        gh.hμσ()[i,j] -= count * hμσδ * σ[j] / σδ
                        gh.hμσ()[j,i] -= count *  (-hμσδ * σ[i] / σδ)
                        gh.hμσ()[j,j] -= count *  (-hμσδ * σ[j] / σδ)

        return gh

    def eval(self, obs, compute_gradient = False, compute_hessian = False, dobs = None):
        # Compute the gradient of the paramters (µ, σ, α, β) with respect to the KL divergence
        # of the approximate posterior with the true posterior.

        # sum Q(Z) log Q(Z) / P(Z|X) = sum Q(Z) log Q(Z) - sum Q(Z) log P(Z|X)
        # start with the first one, the entropy

        gh = GradientHessian(self.n)

        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))
        self.__minus_invgamma_entropy__(compute_gradient, compute_hessian, out=gh)


        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))
        self.__minus_normal_entropy__(compute_gradient, compute_hessian, out=gh)

        # ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv
        self.__gamma_cross_entropy__(compute_gradient, compute_hessian, out=gh)

        # ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz
        self.__normal_cross_entropy__(compute_gradient, compute_hessian, out=gh)

        # Σ_ij  o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ
        self.__minus_observations__(obs, compute_gradient, compute_hessian, dobs=dobs, out=gh)

        if np.any(np.isnan(gh.g)):
            raise ValueError("NaN in gradient")
        if np.any(np.isnan(gh.h)):
            raise ValueError("NaN in hessian")
        return gh

    def fit(self, obs, niter=100000, tol=1e-6, verbose=True, λ0 = 1e-4):
        last_val = None
        for _ in range(niter):
            gh = self.eval(obs, compute_gradient=True, compute_hessian=True)
            if last_val and np.abs(last_val - gh.val)/np.abs(gh.val) < tol and λ == λ0: # don't stop if we're still dampening the hessian
                break
            last_val = gh.val


            # newton update for the log of positive parameters
            gh.h[self.n:,self.n:] *= self.params[self.n:].reshape(-1,1).dot(self.params[self.n:].reshape(1,-1))
            gh.g[self.n:] *= self.params[self.n:]
            gh.h[self.n:,self.n:] += np.diag(gh.g[self.n:])

            λ = λ0

            if np.any(np.diag(gh.h) < 0):
                raise "negative diagonal"

            while True:
                try:
                    diff =  linalg.solve(gh.h + np.diag(np.diag(gh.h) * (1 + λ)), gh.g, assume_a='sym')

                    old_params = self.params.copy()
                    self.params[:self.n] -= diff[:self.n]
                    factor = exp(-diff[self.n:])
                    self.params[self.n:] *= factor

                    if self.eval(obs).val < gh.val: # if it's better stop
                        if verbose and λ > λ0:
                            print(f"λ = {λ}")
                        break

                    else:
                        self.params = old_params
                        λ *= 10 # otherwise increase the conditioning

                except RuntimeWarning as r:
                    λ *= 10

            if verbose:
                print(gh.val, (self.σ()**2).mean() ) # , "KL~", self.KL(obs))


    def prob_win(self, i, j):
        # Compute the probability that player i beats player j
        μ = self.µ()[i] - self.µ()[j]
        σ = sqrt(self.σ()[i]**2 + self.σ()[j]**2)
        return 1/2 * (1 + erf(sqrt(π) * μ / (4 * sqrt(1 + π * σ**2 / 8))))


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
    m = Model(n=10)
    v = VBayes(m)

    instance = m.rvs()
    obs = instance.observe(400)

    squares =  np.sum(v.σ()**2)
    for _ in range(200):

        v.fit(obs, verbose=False)
        dobs = np.zeros((m.n, m.n, 2))
        gh = v.eval(obs,compute_gradient=True,compute_hessian=True, dobs=dobs)
        plt.errorbar(x=instance.z,y=v.μ(),yerr=1.96 * v.σ(),fmt='.')
        plt.show()
        factor = -linalg.solve(gh.h, np.concatenate([np.zeros(m.n), v.σ(), np.zeros(2)]), assume_a='sym')
        best_gain = 0.0
        best_pair = None
        for i in range(m.n):
            for j in range(i + 1 , m.n):
                ifwin = (factor[m.n + i] * v.σ()[i] + factor[m.n + j] * dobs[i,j,0] * v.σ()[j]) * dobs[i,j,0] + (factor[i] - factor[j]) * dobs[i,j,1]  * dobs[i,j,1]
                iflose = (factor[m.n +j] * v.σ()[j] + factor[m.n + i] * dobs[j,i,0] * v.σ()[j]) * dobs[j,i,0] + (factor[j] - factor[i]) * dobs[j,i,1]  * dobs[j,i,1]
                p = v.prob_win(i, j)

                gain = p * ifwin * 0.5 + (1 - p) * iflose
                if gain > 0:
                    print(f"What's up, gain = {gain} > 0??")
                if gain < best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        print(best_gain, best_pair, np.sum(v.σ()**2), (np.sum(v.σ()**2) - squares))
        squares = np.sum(v.σ()**2)
        if instance.match(best_pair[0], best_pair[1]):
            obs[best_pair[0], best_pair[1]] += 1
        else:
            obs[best_pair[1], best_pair[0]] += 1


    plt.errorbar(x=instance.z,y=v.μ(),yerr=1.96 * v.σ(),fmt='.')

    plt.show()




if __name__ == '__main__':
    # lp = LineProfiler()
    # lp.add_function(VBayes.eval)
    # lp.add_function(VBayes.fit)
    # lp.add_function(VBayes.__minus_invgamma_entropy__)
    # lp.add_function(VBayes.__minus_normal_entropy__)
    # lp.add_function(VBayes.__gamma_cross_entropy__)
    # lp.add_function(VBayes.__normal_cross_entropy__)
    # lp.add_function(VBayes.__minus_observations__)
    # lp_wrapper = lp(test)
    # lp_wrapper()
    # lp.print_stats()
    test()
