import pymc as pm
import numpy as np
import scipy
import math

coords = {
}

n = 4
matches =  [(0,3),(1,3),(2,3),(1,2)] * 50

with pm.Model(coords=coords) as model:
    # There are n items
    # sigma is a hyperparameter
    # Each item has a score drawn from a distribution with mean 0 and variance sigma
    # The probability that an item beats another item in a contest is given by the logistic
    # function applied to their scores

    #matches = pm.ConstantData("matches", coords["matches"])

    # Standard deviation of the scores
    sigma = pm.Gamma('sigma', 0.1, 0.1)

    # The scores themselves
    scores = pm.Normal('scores', 0, 1, shape=n)

    for i, (winner, loser) in enumerate(matches):
        # logistic function with scores[winner] and scores[loser]
        # The probability that the winner beats the loser
        prob = pm.invlogit(sigma * (scores[winner] - scores[loser]))
        # The winner wins
        pm.Bernoulli('win_%d' % i, p=prob, observed=True)

    # we observe a set of contests as pairs of indices, the first index is the winner
    # the second index is the loser


    # Run inference to determine scores
    trace = pm.sample(2500, tune=1000, cores=4)
    print(trace)



# Variational bayes

# Use torch for automatic differentiation
import torch


class VariationalBayes:

    def init_integral(self):
        z = np.linspace(-100, 100, 10000)
        for a in range(0, 10000):
            for o in range(-5000, 5000):
                alpha = a / 1000
                other = o / 1000

                normal = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
                expit =  scipy.special.expit( alpha * (z - other) )

                I = normal * expit

                # derivative integrand  by alpha and other
                dI = normal * expit * (1 - expit)
                dI_dalpha =  dI * (z - other)
                dI_dother =  dI * (-alpha)
                dI_dother_dalpha = dI * (alpha * (z - other) * (2 * expit - 1) - 1)

                self.I[a,o] = np.trapz(I)
                self.dI_dalpha[a,o] = np.trapz(dI_dalpha)
                self.dI_dother[a,o] = np.trapz(dI_dother)
                self.dI_dother_dalpha[a,o] = np.trapz(dI_dother_dalpha)


    def __init__(self):
        self.init_integral()


    def interpolate(self, u, v, f):
        # f is a 2 x 2 x 2 x 2 tensor, such that for (a,b,c,d) in {0,1}^4,
        # f[a,b,c,d] is the (ath, bth) derivative of f with respect to its first
        # two variables at point c and d

        # Interpolate the value
        A = np.array([(1-u)**2, u**2]).reshape(1,1,2,1)
        B = np.array([(1-v)**2, v**2]).reshape(1,1,1,2)
        C = np.array([[1 + 2 * u, 3 - 2 * u], [u, u - 1]]).reshape(1,2,2,1)
        D = np.array([[1 + 2 * v, 3 - 2 * v], [v, v - 1]]).reshape(2,1,1,2)
        res = ((A * B * C * D) * f).sum()

        # Interpolate the derivative with respect to u
        A_u = np.array([1-u, u]).reshape(1,1,2,1)
        C_u = np.array([[- 6 * u, 6 * (1 - u)], [1 - 3 * u, 3 * u - 2]]).reshape(1,2,2,1)
        res_u = ((A_u * B * C_u * D) * f).sum()

        # Interpolate the derivative with respect to v
        B_v = np.array([(1-v), v]).reshape(1,1,1,2)
        D_v = np.array([[- 6 * v, 6 * (1 - v)], [1 - 3 * v, 3 * v - 2]]).reshape(2,1,1,2)
        res_v = ((A * B_v * C * D_v) * f).sum()

        return (res, res_u, res_v)



    def special_integral(self, mu, sigma, alpha, other):
        # Integrate NormalPDF(mu, sigma, x) * log(1/(1+exp(-alpha * (x - other)))) from -inf to inf
        # We can reduce this to the case mu=0 and sigma=1
        # special_integral(0, 1, alpha * sigma, (other - mu) / sigma)

        mu, sigma, alpha, other = 0, 1, alpha * sigma, (other - mu) / sigma
        a = math.floor(alpha * 1000)
        o = math.floor(other * 1000)

        f = np.zeros((2,2,2,2))
        for i in range(0,1):
            for j in range(0,1):
                f[0,0][i,j] = self.I[a+i, o+j]
                f[1,0][i,j] = self.dI_dalpha[a+i, o+j]
                f[0,1][i,j] = self.dI_dother[a+i, o+j]
                f[1,1][i,j] = self.dI_dother_dalpha[a+i, o+j]

        return self.interpolate(alpha - a / 1000, other - o / 1000, f)














"""
The score we try to minimize
"""
def score(mu, sigma, observations):
    # Start with the entropy of the approximate distribution

    s = torch.log(sigma).sum()

    # Subtract the likelihood of the observations

    for (winner, loser) in observations:
       s -= special_integral(mu[winner], sigma[winner], alpha, mu[loser])
       s -= special_integral(mu[loser], sigma[loser], alpha, mu[winner])
