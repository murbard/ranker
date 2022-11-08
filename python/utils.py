from numpy.linalg import norm
import numpy as np

def condest(lf, maxiter=100, random_state=None):
    if random_state is None:
        rng = np.random.RandomState()
    else:
        rng = random_state
    v = rng.standard_normal(lf.shape[1])
    v = v / norm(v)
    for _ in range(maxiter):
        w = lf.dot(v)
        emax = norm(w)
        w /= emax
        if norm(v - w) < 1e-8:
            break
        v = w    
    v = rng.standard_normal(lf.shape[1])
    v = v / norm(v)
    for _ in range(maxiter):
        w = cg(lf, v, tol=1e-8, atol=1e-10)[0]
        emin = norm(w)
        w /= emin
        if norm(v - w) < 1e-8:
            break
        v = w
    return emax * emin
