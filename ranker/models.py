from operator import pos
from scipy.stats import invgamma, norm
from scipy.special import gammaln, gamma, polygamma, erf, erfc, expit
from scipy.integrate import quad
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg
from numpy import log, sqrt, exp, linspace, array, zeros, ones, arange, pi as π, inf, sign
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

κstats = []

rng = np.random.default_rng(12345)

def almost_log_expit(x):
    return - 2 * exp(- π * x**2 / 16) / π + x / 2 * erfc(sqrt(π) * x / 4)


# TODO:
# Use a sparse Matrix to represent the Hessian
# use scipy sparse cg to solve the linear system, scipy.sparse.linalg.cg
# set up a benchmark to compare my homebrew solution vs directly using scipy.minimize

class Model():

    def __init__(self, n, α=1.2, β=2.0):
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
        v = invgamma.rvs(self.α, scale=1/self.β, random_state=rng)
        return Instance(v, norm.rvs(loc=0, scale=sqrt(v), size=self.n, random_state=rng))

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
        ps = np.empty(self.n * (self.n - 1) // 2)
        ps.fill(1.0 / len(ps))
        counts = rng.multinomial(n_obs, ps)
        k = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                p = expit(self.z[i] - self.z[j])
                obs[i, j] = rng.binomial(counts[k], p)
                obs[j, i] = counts[k] - obs[i, j]
                k += 1
        return coo_matrix(obs)


    def match(self, i, j):
        return rng.rand() < expit(self.z[i] - self.z[j])


class GradientHessian():

    def __init__(self, n):
        self.val = 0
        self.n = n
        self.g = zeros(2*n+2)                
        self.h_diag = zeros(2*n+2) # the diagolanl
        self.h_αβ = zeros((2,2)) # the off diagonal terms between α and β
        self.h_μσαβ = zeros((2*n,2)) # the terms between μ σ on the one hand and α and β on the other
        self.h_μσ = zeros(n) # a diagonal matrix of the terms between μ and σ
        self.h_obs = coo_matrix((2*n, 2*n)) # the off diagonal terms coming from observations, relating
        self.h = LinearOperator((2*n+2, 2*n+2), matvec=self.matvec)

    def matvec(self, x):
        res =  self.h_diag * x
        res[-2:] += (self.h_αβ + self.h_αβ.T).dot(x[-2:])
        res[:-2] += self.h_μσαβ.dot(x[-2:])
        res[-2:] += self.h_μσαβ.T.dot(x[:-2])
        res[:-2] += self.h_obs.dot(x[:-2]) + self.h_obs.T.dot(x[:-2])
        return res

    def gα(self):
        return self.g[2 * self.n: 2 * self.n+1]

    def gβ(self):
        return self.g[2 * self.n + 1 : 2 * self.n + 2]

    def gμ(self):
        return self.g[:self.n]

    def gσ(self):
        return self.g[self.n:2*self.n]

    def nans_in_grad(self):
        return np.any(np.isnan(self.g))

    def nans_in_hess(self):
        return np.any(np.isnan(self.h_diag)) or np.any(np.isnan(self.h_αβ)) or np.any(np.isnan(self.h_μσαβ)) or np.any(np.isnan(self.h_μσ)) or np.any(np.isnan(self.h_obs.data))


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


        # idea though: assume all variances are 0, take logits of obs, and do a least square fit on μ
        # if obs is large this gives us a very good starting point for μ

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
        return Instance(invgamma.rvs(self.α()[0], scale=1/self.β()[0], random_state=rng), norm.rvs(loc=self.µ(), scale=self.σ(), random_state=rng))


    def eval(self, obs, compute_gradient = False, compute_hessian = False, dobs = None):
        # Compute the gradient of the paramters (µ, σ, α, β) with respect to the KL divergence
        # of the approximate posterior with the true posterior.

        # sum Q(Z) log Q(Z) / P(Z|X) = sum Q(Z) log Q(Z) - sum Q(Z) log P(Z|X)
        # start with the first one, the entropy

        gh = GradientHessian(self.n)

        # ∫ - InvGamma(α, β, v) log(InvGamma(α, β, v))

        α, β = self.α()[0], self.β()[0]
        σ, μ = self.σ(), self.μ()
        n = self.model.n

        gh.val -= invgamma.entropy(α, scale=1/β)

        if compute_gradient:
            gh.gα()[0] += (1 + α) * polygamma(1, α) - 1
            gh.gβ()[0] += 1 / β

        if compute_hessian:
            gh.h_diag[2*n] += (1 + α) * polygamma(2, α)
            gh.h_diag[2*n+1] += -1 / β**2


        # ∫ - Normal(µ, σ², z) log(Normal(µ, σ², z))

        gh.val -= norm.entropy(scale=σ).sum()
        if compute_gradient:
            gh.gσ()[:] += - 1 / σ # we subtract the entropy gradient, so when we minimize this pushes σ to be bigger
        if compute_hessian:
            gh.h_diag[n:2*n] += 1 / σ**2

        # ∫ - InvGamma(α, β, v) log( InvGamma(α', β', v)) dv

        gh.val += α * β / self.model.β + self.model.α * log(self.model.β) + gammaln(self.model.α) - (1 + self.model.α) * (log(β) + polygamma(0, α))

        if compute_gradient:
            gh.gα()[0] += β / self.model.β - (1 + self.model.α) * polygamma(1, α)
            gh.gβ()[0] += α / self.model.β - (1 + self.model.α) / β

        if compute_hessian:
            gh.h_diag[2*n] += -(1 + self.model.α) * polygamma(2, α)
            gh.h_diag[2*n+1] += (1 + self.model.α) / β**2
            gh.h_αβ[0, 1] += 1 / self.model.β


        # ∫∫ - InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dv dz

        gh.val += 1/2 * (α * β * (μ**2 + σ**2).sum() + n * (log(2 * π / β) - polygamma(0, α)))

        if compute_gradient:
            gh.gμ()[:] += α * β * μ
            gh.gσ()[:] += α * β * σ # this pushes σ to be smaller
            gh.gα()[0] += 1/2 * (β * (μ**2 + σ**2).sum() - n * polygamma(1, α))
            gh.gβ()[0] += 1/2 * (α * (μ**2 + σ**2).sum() - n / β)

        if compute_hessian:

            gh.h_diag[2*n] += - 1 / 2 * n * polygamma(2, α)
            gh.h_αβ[0, 1] += 1/2 * (μ**2 + σ**2).sum()
            gh.h_diag[2*n+1] += n / (2 * β**2)

            gh.h_μσαβ[:n, 0] += β * μ
            gh.h_μσαβ[:n, 1] += α * μ
            gh.h_μσαβ[n:, 0] += β * σ
            gh.h_μσαβ[n:, 1] += α * σ
            gh.h_diag[:2*n] += α * β


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

        # note: if f(a,b) = g(c(a,b)) then
        # d²f/dadb = d²g/dc² (dc/da)(dc/db) + dg/dc d²c/(da db)

        if compute_hessian and not compute_gradient:
            raise "Must compute the gradient if the hessian is being computed"

        if dobs is not None and not compute_gradient:
            raise "Must compute the gradient if dobs is being computer"

        l = 6 * len(obs.data)
        gh.h_obs = coo_matrix((np.empty(l), (zeros(l,dtype=np.int32), zeros(l,dtype=np.int32))),shape=(2*n,2*n))

        k = 0
        for (count, i, j) in zip(obs.data,obs. row, obs.col):
        
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
                        dobs[i, j, 0] = - gσδ  / σδ
                        dobs[i, j, 1] = gμδ

                if compute_hessian:

                    hμμδ = - exp(-(μδ/h)**2) / (sqrt(π) * h)
                    hμσδ = 2 * μδ * σδ * exp(-(μδ/h)**2) / (sqrt(π) * h**3)
                    hσσδ = - exp(-(μδ/h)**2) * (256 + 4 * π * σδ**2 * (8 + μδ**2)) / (π**(5/2) * h**5)                      

                    # Diagonal terms, hμiμi, hμjμj, hσiσi, hσjσj
                    gh.h_diag[i] -= count * hμμδ
                    gh.h_diag[j] -= count * hμμδ

                    gh.h_diag[self.n + i] -= count * (hσσδ * (σ[i]/σδ)**2 + gσδ * σ[j]**2/σδ**3)
                    gh.h_diag[self.n + j] -= count * (hσσδ * (σ[j]/σδ)**2 + gσδ * σ[i]**2/σδ**3)

                    # hμiμj
                    gh.h_obs.data[k] = count * hμσδ
                    gh.h_obs.col[k] = min(i,j)
                    gh.h_obs.row[k] = max(i,j)             
                    k += 1
                                        
                    # hσiσj
                    gh.h_obs.data[k] = - count * (hσσδ * (σ[i]*σ[j] / σδ**2) - gσδ * σ[i]*σ[j] / σδ**3)
                    gh.h_obs.col[k] =  n + min(i,j)
                    gh.h_obs.row[k] =  n + max(i,j)
                    k += 1
                    
                    # hμiσi, 
                    gh.h_obs.data[k] = - count * hμσδ * σ[i] / σδ
                    gh.h_obs.col[k] = i
                    gh.h_obs.row[k] = n + i
                    k += 1
                    
                    # hμiσj
                    gh.h_obs.data[k] = - count * hμσδ * σ[j] / σδ
                    gh.h_obs.col[k] = i
                    gh.h_obs.row[k] = n + j
                    k += 1
                    
                    # hμjσi
                    gh.h_obs.data[k] = count * hμσδ * σ[i] / σδ
                    gh.h_obs.col[k] = j
                    gh.h_obs.row[k] = n + i
                    k += 1

                    # hμjσj
                    gh.h_obs.data[k] = count * hμσδ * σ[j] / σδ                
                    gh.h_obs.col[k] = j
                    gh.h_obs.row[k] = n + j
                    k += 1

        if gh.nans_in_grad():        
            raise ValueError("NaN in gradient")
        if gh.nans_in_hess():        
            raise ValueError("NaN in hessian")            
        return gh



    def fit(self, obs, niter=100000, tol=1e-8, verbose=True, λ0 = 1e-10):

        global κstats
        last_val = None
        for _ in range(niter):
            λ = λ0
            gh = self.eval(obs, compute_gradient=True, compute_hessian=True)
            if last_val and np.abs(last_val - gh.val)/np.abs(gh.val) < tol and λ == λ0 or λ > 10000: # don't stop if we're still dampening the hessian, but do stop
                # if it's gone too far
                break
            last_val = gh.val

            # newton update for the log of positive parameters
            # H = np.array([gh.h.dot(np.eye(2*self.n+2, 2*self.n+2)[i,:]) for i in range(2*self.n+2)])
            # gh.h[self.n:,self.n:] *= self.params[self.n:].reshape(-1,1).dot(self.params[self.n:].reshape(1,-1))
            gh.h_diag[self.n:] *= self.params[self.n:]**2
            gh.h_αβ *= np.outer(self.params[-2:], self.params[-2:])
            gh.h_μσ *= self.params[self.n:-2]
            gh.h_μσαβ[self.n:] *= np.outer(self.params[self.n:-2], self.params[-2:])

            for (k, (i, j)) in enumerate(zip(gh.h_obs.row, gh.h_obs.col)):
                if i >= self.n and j >= self.n:
                    gh.h_obs.data[k] *= self.params[i] * self.params[j]
            
            gh.g[self.n:] *= self.params[self.n:]            

            gh.h_diag[self.n:] += gh.g[self.n:]


            if np.any(gh.h_diag < 0):
                raise RuntimeError("negative diagonal")

            # condition by the diagonal
            sd = 1 / sqrt(gh.h_diag)
            
            gh.h_diag *= sd * sd

            gh.h_αβ *= np.outer(sd[-2:],sd[-2:])
            gh.h_μσ *= sd[self.n:-2]
            gh.h_μσαβ *= np.outer(sd[:-2], sd[-2:])

            for (k, (i, j)) in enumerate(zip(gh.h_obs.row, gh.h_obs.col)):                
                gh.h_obs.data[k] *= sd[i] * sd[j]
                        
            gh.g *= sd

            while True:
                try:
                    # κ = np.linalg.cond(gh.h + np.diag(np.diag(gh.h) * (1 + λ)))
                    # κstats.append(κ)

                    # diff =  sd * linalg.solve(gh.h + np.diag(np.diag(gh.h) * (1 + λ)), gh.g, assume_a='sym')
                    # solve with sparse.linalg.cg
                    diff = sd * cg(gh.h, gh.g, tol=1e-8, atol=1e-12)[0]

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
                        #if verbose:
                        #    print(f"λ->{λ} because overshoot")

                except RuntimeWarning as r:
                    λ *= 10
                    #if verbose:
                    #    print(f"λ->{λ} because warning")

            if verbose:
                print(gh.val, (self.σ()**2).mean() ) # , "KL~", self.KL(obs))

        # print some stats regarding κ, such as mean, standard deviation, and various percentiles
        # κstats = np.array(κstats)
        # print("κ stats:")
        # print("mean:", κstats.mean())
        # print("std:", κstats.std())
        # print("min:", κstats.min())
        # print("max:", κstats.max())
        # print("25%:", np.percentile(κstats, 25))
        # print("50%:", np.percentile(κstats, 50))
        # print("75%:", np.percentile(κstats, 75))
        # print("90%:", np.percentile(κstats, 90))
        # print("95%:", np.percentile(κstats, 95))
        # print("99%:", np.percentile(κstats, 99))
        # print("99.9%:", np.percentile(κstats, 99.9))
        return


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

    def __repr__(self):
        return self.__str__()

    def expected_inversions(self, gradient=None):
        μ = self.μ()
        σ = self.σ()


        # Compute the expected number of inversions in the approximate posterior
        inversions = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                σδ = sqrt(σ[i]**2 + σ[j]**2)
                μδ = μ[i] - μ[j]
                z = abs(μδ) / σδ

                # inverse normal cdf of z
                inversions += 0.5 * erfc(z / sqrt(2))
                if gradient is not None:
                    dinv_dz = - exp(-z**2 / 2) / sqrt(2 * π)
                    dinv_dμi = dinv_dz * sign(μδ) / σδ
                    dinv_dμj = - dinv_dμi
                    dinv_dσi = - dinv_dz * σ[i] / (σδ**3)
                    dinv_dσj = - dinv_dz * σ[j] / (σδ**3)
                    gradient[i] += dinv_dμi
                    gradient[j] += dinv_dμj
                    gradient[self.n + i] += dinv_dσi
                    gradient[self.n + j] += dinv_dσj

        return inversions

    def actual_inversions(self, instance):
        μ = self.μ()
        inversions = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if (μ[i] - μ[j]) * (instance.z[i] - instance.z[j]) <= 0:
                    inversions += 1
        return inversions


    def best_pair(self, obs, target='expected_inversions', verify_top=0, possible_pairs=None):

        self.fit(verbose=False, obs=obs)
        dobs = np.zeros((self.n, self.n, 2))
        gh = self.eval(obs,compute_gradient=True,compute_hessian=True, dobs=dobs)

        targets = ['expected_inversions', 'sum_of_variances']
        match target:
            case 'sum_of_variances':
                df_dparam = np.concatenate([np.zeros(self.n), 2 * self.σ(), np.zeros(2)])
            case 'expected_inversions':
                df_dparam = np.zeros(2 * self.n + 2)
                _ = self.expected_inversions(gradient=df_dparam)
            case _:
                raise ValueError(f"target must be one of {targets}")

        factor = -linalg.solve(gh.h, df_dparam, assume_a='sym')

        gains = []
        if possible_pairs is None:
            possible_pairs = [(i, j) for i in range(self.n) for j in range(i+1, self.n)]

        for i,j in possible_pairs:
            ifwin  = (factor[self.n + i] * self.σ()[i] + factor[self.n + j] * self.σ()[j]) * dobs[i,j,0] + (factor[i] - factor[j]) * dobs[i,j,1]
            iflose = (factor[self.n + j] * self.σ()[j] + factor[self.n + i] * self.σ()[i]) * dobs[j,i,0] + (factor[j] - factor[i]) * dobs[j,i,1]
            p = self.prob_win(i, j)

            gain = p * ifwin + (1 - p) * iflose
            gains.append(np.array([gain, i, j]))

        gains = np.array(gains)
        gains = gains[np.lexsort(gains.T[::-1])]

        einv = self.expected_inversions()
        real_gains = np.zeros(gains.shape)
        k = 0
        for i, j in possible_pairs:
            print(k/len(possible_pairs))
            # i, j = round(gains[k,1]), round(gains[k,2])
            old_params = self.params.copy()
            obs[i, j] += 1
            self.fit(verbose=False, obs=obs, tol=1e-6)
            win = self.expected_inversions()
            obs[i, j] -= 1
            self.params = old_params.copy()
            obs[j, i] += 1
            self.fit(verbose=False, obs=obs, tol=1e-6)
            lose = self.expected_inversions()
            obs[j, i] -= 1
            self.params = old_params.copy()
            p = self.prob_win(i, j)
            real_gains[k,0] = p * win + (1 - p) * lose - einv
            real_gains[k,1] = i
            real_gains[k,2] = j
            k = k + 1

        gains = gains[np.lexsort(gains.T[::-1])]
        return gains


def test():
    m = Model(n=50)
    v = VBayes(m)

    instance = m.rvs()
    obs = instance.observe(10)



    plt.axis([-5, 5, -5, 5])

    for _ in range(2000):
        (i,j) = v.best_pair(obs, verify_top=10)
        print(i,j, v.expected_inversions(), v.actual_inversions(instance))
        if instance.match(i,j):
            obs[i,j] += 1
        else:
            obs[j,i] += 1
        plt.clf()
        plt.errorbar(x=instance.z,y=v.μ(),yerr=1.96 * v.σ(),fmt='.')
        plt.pause(0.01)

    plt.show()

test()




if __name__ == '__main__':
    
    κstats = []
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
