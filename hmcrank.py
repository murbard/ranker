from ast import literal_eval
import pymc4
from optparse import AmbiguousOptionError
import numpy as np
import scipy
import scipy.stats
import torch

def softplus(x) :
    # always take the exp of a negative number for numerical
    # stability
    return np.log(1 + np.exp(-np.abs(x))) + np.clip(x, 0, None)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class Model(object):

    def __init__(self):
        self.comparisons = []

    def ll(self, ambiguity, values):
        # return log likelihood and ll gradient wrt parameters
        ll_ = 0
        lla_ = 0

        # gaussian prior with variance 1 (without loss of generality)
        ll += -0.5 * (values**2).sum()

        # the derivative of a gaussian prior simply shrinks everything linearily to the center
        llv_ = - values

        for (i, j) in self.comparisons:
            lo, hi = values[i], values[j]

            ll_ + softplus((hi - lo) * ambiguity)
            s = logistic((hi - lo) * ambiguity)
            g = ambiguity * s
            llv_[i] -= -g
            llv_[j] += g
            lla_ += (hi - lo) * s

        return (ll_, (lla_, llv_))




    # Take a half step following the hamiltonian monte carlo method
    def half_step(self, ambiguity, values, epsilon):
        # return a new values and new ambiguity
        # epsilon is the step size
        # values is the current values
        # ambiguity is the current ambiguity

        # compute the log likelihood and its gradient

        pv = np.zeros(len(self.values))
        pa = 0.0

        for _ in range(0, L):

            values += epsilon * pv
            ambiguity += epsilon * pa

            ll, (lla, llv) = self.ll(ambiguity, values)

            pv += 0.5 * epsilon * llv
            pa += 0.5 * epsilon * lla






