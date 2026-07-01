#!/usr/bin/env python3
r"""
Fit a k-term correction to the error-function approximation of  log(1+e^{-x}).

The correction has the form

        corr(z) = sum_{i=1}^k  c_i (1 + a_i z^2) exp(-a_i z^2)

so that  approx(z) + corr(z)  approximates  exact(z) = log(1+e^{-x}), where approx is the
Crooks error-function approximation.  Every term integrates against a Gaussian in closed form,

    \int N(x; mu, s2) c_i (1 + a_i x^2) e^{-a_i x^2} dx
        = c_i / sqrt(D_i) * exp(-a_i mu^2 / D_i) * (1 + a_i mu^2/D_i^2 + a_i s2/D_i),
        D_i = 1 + 2 a_i s2,

so the corrected approximation keeps the closed-form logistic-normal integral / gradient / Hessian.

How the fit works.  The residual r(z) = exact(z) - approx(z) is even, non-negative, peaks at z=0
(at log2 - 2/pi) and decays.  We fit corr to r.  With the widths a_i fixed the coefficients c_i are
LINEAR, so the minimax (L-infinity) fit of c_i is a linear program; only the k widths are nonlinear.
So we (1) globally search the widths with differential evolution wrapping an inner LP for c_i, then
(2) jointly refine (a_i, c_i) by gradient descent on a smooth high-p L_p surrogate of the
non-smooth max-abs loss (annealing p upward toward L-infinity).

Objectives (--mode):
  abs   : minimize the worst ABSOLUTE error  max|r - corr|                       (default)
  mono  : same, but CONSTRAINED so the error improves (never worsens) at EVERY point,
          i.e. 0 <= corr(z) <= 2 r(z).  Solved exactly by the LP; no gradient polish.
  rel   : minimize the worst RELATIVE error  max|r - corr|/exact  on |z| <= zrel

Usage:
  python fit_correction.py -k 5
  python fit_correction.py -k 5 --mode mono
  python fit_correction.py -k 7 --seed 3
"""
import argparse
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution
from scipy.special import erfc


# ---- the function and the existing erf approximation ----
# NB: exact()/approx() intentionally duplicate models.py's log_expit / almost_log_expit so this
# fitting tool stays standalone (importing models.py pulls in line_profiler etc.).  If the Crooks
# constants in almost_log_expit ever change, update approx() here to match.
def exact(z):
    z = np.asarray(z, float)
    return np.log1p(np.exp(-np.abs(z))) + np.clip(-z, 0.0, None)       # stable log(1+e^-z)

def approx(z):
    z = np.asarray(z, float)
    return 2*np.exp(-np.pi*z**2/16)/np.pi - 0.5*z*erfc(np.sqrt(np.pi)*z/4)

def phi(z, a):
    return (1.0 + a*z**2)*np.exp(-a*z**2)                              # one correction-term shape

def corr(z, a, c):
    z = np.asarray(z, float)
    return sum(ci*phi(z, ai) for ai, ci in zip(a, c))

def worst_abs(a, c, zlim=25.0, n=40001):
    """True worst absolute error of (approx+corr) on a fine grid (for ranking / reporting)."""
    z = np.linspace(0.0, zlim, n)                                     # residual is even -> z>=0
    return float(np.max(np.abs((exact(z) - approx(z)) - corr(z, a, c))))


def inner_lp(a, Z, R, mode, EX, zrel):
    """Given widths a (k,), solve for c (k,) minimizing the chosen minimax objective on grid Z.
    Returns (c, t), t = achieved worst (weighted) error; (None, inf) if infeasible/degenerate."""
    a = np.asarray(a, float)
    if np.any(a <= 0) or np.any(~np.isfinite(a)):
        return None, np.inf
    k = len(a); ng = len(Z)
    P = np.stack([phi(Z, ai) for ai in a], axis=1)                    # (ng, k)
    if mode == 'rel':                                                 # band scaled by exact(z)
        w = np.minimum(np.where(Z <= zrel, EX, np.inf), 1e6)
    else:
        w = np.ones(ng)
    # variables x = [c (k), t]; minimize t s.t. |R - Pc| <= t*w   (+ monotone band if requested)
    rows = [np.hstack([-P, -w[:, None]]), np.hstack([P, -w[:, None]])]
    rhs = [-R, R]
    if mode == 'mono':                                               # 0 <= Pc <= 2R everywhere
        rows += [np.hstack([-P, np.zeros((ng, 1))]), np.hstack([P, np.zeros((ng, 1))])]
        rhs += [np.zeros(ng), 2*R]
    cobj = np.zeros(k+1); cobj[-1] = 1.0
    res = linprog(cobj, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                  bounds=[(None, None)]*k + [(0, None)], method='highs')
    if not res.success:
        return None, np.inf
    return res.x[:k], res.x[-1]


def grad_polish(a, c, zmax=18.0, ngrid=12000, p_schedule=(8, 16, 32, 64, 128, 256)):
    """Joint gradient refinement of (a, c) on a smooth high-p L_p surrogate of the (non-smooth)
    max-abs objective, annealing p upward.  ||e||_p -> ||e||_inf as p->inf and IS differentiable.
    Analytic gradients:  d corr/d c_i = (1+a_i z^2)e^{-a_i z^2},  d corr/d a_i = -c_i a_i z^4 e^{-a_i z^2}."""
    a = np.asarray(a, float); c = np.asarray(c, float); k = len(a)
    Z = np.linspace(0.0, zmax, ngrid); R = exact(Z) - approx(Z)
    Z2 = Z[:, None]**2; Z4 = Z[:, None]**4
    def loss_grad(x, p):
        aa = np.exp(x[:k]); cc = x[k:]
        E = np.exp(-aa*Z2)                                            # (n,k); reused below
        P = (1.0 + aa*Z2)*E                                          # phi(Z, a_i) per term
        e = P @ cc - R; ae = np.abs(e); m = ae.max() + 1e-300; r = ae/m
        Sp = np.mean(r**p); Lp = m*Sp**(1.0/p)                        # smooth approx of max|e|
        dLde = (Sp**(1.0/p - 1.0)*r**(p - 1.0)/len(Z))*np.sign(e)
        gc = P.T @ dLde
        dcorr_da = -cc*aa*Z4*E                                       # (n,k): d corr/d a_i
        ga = (dcorr_da.T @ dLde)*aa                                   # chain rule for log(a)
        return Lp, np.concatenate([ga, gc])
    x = np.concatenate([np.log(a), c])
    for p in p_schedule:
        x = minimize(lambda x: loss_grad(x, p), x, jac=True, method='L-BFGS-B',
                     options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}).x
    aa = np.exp(x[:k]); cc = x[k:]; idx = np.argsort(aa)
    return aa[idx], cc[idx]


def fit(k, mode='abs', zmax=18.0, ngrid=3000, zrel=10.0, seed=0,
        verbose=True, warm=None, polish='grad'):
    """Fit k correction terms.  Returns (a, c, score) where score is the worst error in the
    mode's own objective (absolute for abs/mono, relative for rel)."""
    # even residual -> fit on z>=0; denser near 0 where the residual lives, plus a sparse tail
    # out to ~26 so the mono band / fit cover the whole domain that metrics() checks (|z|<=25).
    Z = np.unique(np.concatenate([np.linspace(0, 4, ngrid//2), np.linspace(4, zmax, ngrid//2),
                                  np.linspace(zmax, 26.0, 200)]))
    R = exact(Z) - approx(Z); EX = exact(Z)
    # (1) global search over the widths (sorted to kill the permutation symmetry); inner LP for c.
    objfun = lambda la: inner_lp(np.exp(np.sort(la)), Z, R, mode, EX, zrel)[1]
    de = differential_evolution(objfun, [(np.log(1e-3), np.log(3.0))]*k, seed=seed,
                                maxiter=120, popsize=18, tol=1e-12, mutation=(0.3, 1.2),
                                recombination=0.85, polish=False, init='sobol', updating='deferred')
    if verbose:
        print(f"  differential-evolution best (grid LP): {de.fun:.3e}", flush=True)
    width_cands = [np.sort(np.exp(de.x))]
    if warm is not None:
        width_cands.append(np.sort(np.asarray(warm, float)))
    # rank candidates by the mode's OWN objective; gradient polish (which targets ABSOLUTE error)
    # only applies in 'abs' mode -- in 'rel'/'mono' the LP already optimizes the right thing.
    def score(a, c):
        m = metrics(a, c)
        return m['worst_rel_after'] if mode == 'rel' else m['worst_abs_after']
    # (2) for each candidate set of widths: LP for c; in abs mode also try the gradient polish,
    #     but keep the LP solution as a competing fallback so a diverging polish can't make it worse.
    best = None
    for a0 in width_cands:
        c0, _ = inner_lp(a0, Z, R, mode, EX, zrel)
        if c0 is None:
            continue
        trials = [(a0, c0)]
        if polish == 'grad' and mode == 'abs':
            trials.append(grad_polish(a0, c0, zmax=zmax))
        for a1, c1 in trials:
            s = score(a1, c1)
            if verbose:
                print(f"  candidate score ({mode}): {s:.3e}", flush=True)
            if best is None or s < best[0]:
                best = (s, a1, c1)
    if best is None:
        raise RuntimeError("all width candidates gave an infeasible LP; try a different --seed or --zmax")
    return best[1], best[2], best[0]


def metrics(a, c, zlim=25.0, n=40001, rel_dom=12.0):
    """Return a dict of before/after error metrics for the corrected approximation."""
    z = np.linspace(-zlim, zlim, n)
    R = exact(z) - approx(z); cr = corr(z, a, c); EX = exact(z)
    err_before, err_after = np.abs(R), np.abs(R - cr)
    dom = np.abs(z) <= rel_dom
    rel_before = np.abs((approx(z) - exact(z))/EX)[dom]
    rel_after = np.abs((approx(z) + cr - exact(z))/EX)[dom]
    return dict(worst_abs_before=err_before.max(), worst_abs_after=err_after.max(),
                worst_rel_before=rel_before.max(), worst_rel_after=rel_after.max(),
                monotone=bool(np.all(err_after <= err_before + 1e-15)))


def report(a, c):
    m = metrics(a, c)
    print("\n  worst ABSOLUTE error:   erf-only = %.3e   corrected = %.3e   (%.0fx)" %
          (m['worst_abs_before'], m['worst_abs_after'],
           m['worst_abs_before']/max(m['worst_abs_after'], 1e-300)))
    print("  worst RELATIVE error (|z|<=12): erf-only = %.3e   corrected = %.3e" %
          (m['worst_rel_before'], m['worst_rel_after']))
    print("  monotonic improvement at every point: %s" % m['monotone'])
    print("\n  a = {" + ", ".join("%.12g" % x for x in a) + "}")
    print("  c = {" + ", ".join("%.12g" % x for x in c) + "}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, default=5, help="number of correction terms")
    ap.add_argument("--mode", choices=['abs', 'mono', 'rel'], default='abs')
    ap.add_argument("--zmax", type=float, default=18.0, help="fit grid extent in z")
    ap.add_argument("--zrel", type=float, default=10.0, help="domain for the relative-error band (rel mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--polish", choices=['grad', 'none'], default='grad',
                    help="gradient refinement of (a,c) on a smooth L_p surrogate after the LP")
    args = ap.parse_args()
    print(f"fitting k={args.k} terms, mode={args.mode} ...")
    a, c, t = fit(args.k, mode=args.mode, zmax=args.zmax, zrel=args.zrel,
                  seed=args.seed, polish=args.polish)
    report(a, c)
