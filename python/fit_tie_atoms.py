#!/usr/bin/env python3
"""
Generate the particle (atom) representation of the tie parameter for the Davidson tie model.

The Davidson model adds one nuisance parameter lambda >= 0:

    P(i wins) = e^{d/2}/(A+lambda),  P(tie) = lambda/(A+lambda),  A = 2 cosh(d/2),

whose variational posterior is a categorical over J fixed atoms {lambda_j}. The prior is placed
on the OBSERVABLE quantity p0 = lambda/(2+lambda) -- the tie probability between equally matched
items -- as p0 ~ Beta(a, b) (default uniform). Atoms sit at the prior's (j-1/2)/J quantiles, so
each atom carries prior mass exactly 1/J: the prior determines the grid, there are no arbitrary
bounds, and the range scales automatically with J.

Each atom's expected log-normalizer needs E_N(mu,sg^2)[ b_lambda(d) ] for the even bump
b_lambda(d) = log(1 + lambda/(2 cosh(d/2))). Atoms are fitted in the shared basis
{ e^{-a_l d^2},  a_l d^2 e^{-a_l d^2} } whose Gaussian integrals are closed-form
(E[(p + q a d^2) e^{-a d^2}] = e^{-a mu^2/D}/sqrt(D) (p + q(a mu^2/D^2 + a s/D)), D = 1+2as),
so the tie model reuses the ranker's correction machinery: the per-pair cost is one premixed
kernel evaluation, independent of J.

Usage:  python python/fit_tie_atoms.py [J] [a] [b] [outfile]
        defaults: J=16, a=1, b=1 (uniform p0), fit_results/tie_atoms_J16.txt

Output format (read by CC/ranker.cc):
    J L
    a_1 ... a_L                                   shared widths
    lam_j pi_j p_1 q_1 ... p_L q_L                one line per atom
"""
import sys
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

J = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
PB = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
OUT = sys.argv[4] if len(sys.argv) > 4 else f"fit_results/tie_atoms_J{J}.txt"

# --- atoms at prior quantiles of p0 = lambda/(2+lambda);  each has prior mass 1/J ---
p0 = beta_dist.ppf((np.arange(J) + 0.5) / J, PA, PB)
lams = 2 * p0 / (1 - p0)

# --- shared basis, sized for the atom range (largest atoms have wide, tall bumps) ---
L = 20
A = np.geomspace(0.003, 1.0, L)
GRID = np.linspace(0, 26, 5201)

def b(lam, d):
    return np.log1p(lam / (2 * np.cosh(d / 2)))

d2 = GRID[:, None] ** 2
E = np.exp(-A * d2)
P = np.concatenate([E, A * d2 * E], axis=1)        # (grid, 2L)

M = np.empty((2 * L, J))
sups = []
for j, lam in enumerate(lams):
    y = b(lam, GRID)
    coef, *_ = np.linalg.lstsq(P, y, rcond=None)
    M[:, j] = coef
    sups.append(np.max(np.abs(P @ coef - y)))

# --- validate the closed-form Gaussian integrals against quadrature ---
# NB: closed() intentionally duplicates models.bump_eval's order-0 branch so this script stays
# standalone; if the basis parameterization changes there (or in CC/ranker.cc), update here too,
# or the validation below silently checks the wrong closed form.
def closed(mu, sig, pq):
    p, q = pq[:L], pq[L:]
    D = 1 + 2 * A * sig ** 2
    return np.sum(np.exp(-A * mu ** 2 / D) / np.sqrt(D) * (p + q * (A * mu ** 2 / D ** 2 + A * sig ** 2 / D)))

ierr = 0.0
for j in range(0, J, max(1, J // 5)):
    for mu in (0.0, 1.5, 4.0):
        for sig in (0.4, 1.2, 2.5):
            tgt = quad(lambda d: np.exp(-(d - mu) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi) / sig * b(lams[j], d),
                       mu - 10 * sig, mu + 10 * sig, limit=200)[0]
            ierr = max(ierr, abs(closed(mu, sig, M[:, j]) - tgt))

with open(OUT, "w") as f:
    f.write(f"{J} {L}\n")
    f.write(" ".join(f"{a:.17g}" for a in A) + "\n")
    for j, lam in enumerate(lams):
        pq = " ".join(f"{M[l, j]:.17g} {M[L + l, j]:.17g}" for l in range(L))
        f.write(f"{lam:.17g} {1.0 / J:.17g} {pq}\n")

print(f"J={J} atoms from Beta({PA:g},{PB:g}) prior on p0: lambda in [{lams[0]:.4g}, {lams[-1]:.4g}]")
print(f"basis L={L}: sup fit error {max(sups):.2e}, integrated error vs quadrature {ierr:.2e}")
print(f"wrote {OUT}")
