#!/usr/bin/env python3
"""
Regenerate the figures in the Ranker paper (ranker.tex).

Usage (run from the repository root):
    python doc/ranker.py [approx] [correction] [mcmc] [active]

With no arguments it regenerates all four figures. It imports python/models.py and reads
fit_results/sweep.json. Each target writes one PDF into doc/figures/:

    approx      -> approx.pdf          error-function approximation of the softplus log(1+e^-z)
    correction  -> correction_k5.pdf   the k-term correction: error across z (k=5), and the
                                       worst error vs the number of terms k
    mcmc        -> vb_vs_mcmc.pdf       VB posterior vs a reference Metropolis sampler (n=12)
    active      -> active_selection.pdf active pair-selection strategies; shells out to the C++
                                       implementation CC/ranker (builds it if absent). SLOW at the
                                       paper's runs=800 (~1h); pass a smaller count for a preview.
"""
import sys, os, json, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
from scipy.special import erfc, gammaln

sys.path.insert(0, "python")
import models

OUT = "doc/figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
ALPHA_H, BETA_H = 1.2, 2.0   # inverse-gamma prior on the variance (beta is a rate; scale = 1/beta)


# ---------------------------------------------------------------------------
# approx.pdf: accuracy of the error-function approximation of the softplus.
# ---------------------------------------------------------------------------
def fig_approx():
    z = np.linspace(-8, 8, 1000)
    exact = np.log1p(np.exp(-np.abs(z))) + np.clip(-z, 0, None)               # stable log(1+e^-z)
    approx = 2*np.exp(-np.pi*z**2/16)/np.pi - z/2*erfc(np.sqrt(np.pi)*z/4)    # Crooks erf approximation
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(z, exact, label=r"$\log(1+e^{-z})$", lw=2)
    a1.plot(z, approx, "--", label="erf approximation", lw=2)
    a1.set_xlabel("z"); a1.legend(); a1.set_title("softplus vs. approximation")
    a2.plot(z, approx - exact, color="C3")
    a2.axhline(2/np.pi - np.log(2), ls=":", color="gray")                    # min error = -(log2 - 2/pi)
    a2.set_xlabel("z"); a2.set_title("approximation error")
    fig.tight_layout(); fig.savefig(f"{OUT}/approx.pdf"); plt.close(fig)
    print("  max abs err =", np.max(np.abs(approx-exact)))


# ---------------------------------------------------------------------------
# correction_k5.pdf: the k-term correction to the approximation.
# ---------------------------------------------------------------------------
def fig_correction():
    with open("fit_results/sweep.json") as f:
        d = json.load(f)
    def exact(z):  return np.log1p(np.exp(-np.abs(z))) + np.clip(-z, 0, None)
    def approx(z): return 2*np.exp(-np.pi*z**2/16)/np.pi - z/2*erfc(np.sqrt(np.pi)*z/4)
    def corr(z, a, c): return sum(ci*(1+ai*z**2)*np.exp(-ai*z**2) for ai, ci in zip(a, c))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    # left: error across z for the k=5 correction actually shipped in models.py
    z = np.linspace(-12, 12, 6001)
    a5, c5 = list(models.CORR_A), list(models.CORR_C)
    axL.semilogy(z, np.abs(approx(z)-exact(z)), color='0.6', lw=1.4, label='erf approximation')
    axL.semilogy(z, np.clip(np.abs(approx(z)+corr(z, a5, c5)-exact(z)), 1e-12, None),
                 color='C3', lw=1.4, label='with $k=5$ correction')
    axL.set_xlabel('$z$'); axL.set_ylabel(r'$|\,\widehat f(z)+\varepsilon(z)-\log(1+e^{-z})\,|$')
    axL.set_ylim(1e-9, 2e-1); axL.legend(loc='lower center', fontsize=8); axL.grid(alpha=0.3, which='both')
    axL.set_title('error across $z$ ($k=5$)')
    # right: worst error vs number of terms k (k=0 is the plain erf approximation)
    ks = [0] + list(range(1, 8))
    we = [np.log(2)-2/np.pi] + [d[str(k)]['worst_abs'] for k in range(1, 8)]
    axR.semilogy(ks, we, 'o-', color='C0', lw=1.5, ms=5)
    axR.set_xlabel('number of correction terms $k$'); axR.set_ylabel('worst absolute error')
    axR.set_xticks(ks); axR.grid(alpha=0.3, which='both'); axR.set_title('convergence in $k$')
    fig.tight_layout(); fig.savefig(f"{OUT}/correction_k5.pdf"); plt.close(fig)
    print("  worst error vs k:", [f"{w:.2e}" for w in we])


# ---------------------------------------------------------------------------
# Reference posterior via random-walk Metropolis on the EXACT logistic model.
# State: z (n,) and log v.  Validates the VB factorization AND the erf approximation.
# ---------------------------------------------------------------------------
def exact_logpost(z, logv, obs):
    v = np.exp(logv); n = len(z)
    lp = -0.5*n*logv - 0.5*np.sum(z*z)/v                                          # z_i ~ N(0, v)
    lp += ALPHA_H*np.log(BETA_H) - gammaln(ALPHA_H) - (ALPHA_H+1)*logv - BETA_H/v  # v ~ InvGamma
    lp += logv                                                                   # Jacobian for logv
    d = z[obs.row] - z[obs.col]
    lp += np.sum(obs.data * (-np.log1p(np.exp(-np.abs(d))) - np.clip(-d, 0, None)))  # logistic likelihood
    return lp


def metropolis(obs, n, iters=120000, burn=30000, thin=15):
    # component-wise random-walk Metropolis: mixes far better than a block update
    z = np.zeros(n); logv = np.log(BETA_H/ALPHA_H)
    lp = exact_logpost(z, logv, obs)
    sz, sv = 0.6, 0.2
    zs, vs = [], []
    acc = tot = 0
    for t in range(iters):
        for k in range(n):
            zp = z.copy(); zp[k] += rng.normal(0, sz)
            lpp = exact_logpost(zp, logv, obs); tot += 1
            if np.log(rng.random()) < lpp - lp:
                z, lp = zp, lpp; acc += 1
        lvp = logv + rng.normal(0, sv)
        lpp = exact_logpost(z, lvp, obs)
        if np.log(rng.random()) < lpp - lp:
            logv, lp = lvp, lpp
        if t >= burn and (t-burn) % thin == 0:
            zs.append(z.copy()); vs.append(np.exp(logv))
    zs = np.array(zs); vs = np.array(vs)
    print("  MH accept rate (z) =", acc/tot, "n samples =", len(zs))
    return zs, vs


def fig_vb_vs_mcmc():
    n = 12
    m = models.Model(n=n, α=ALPHA_H, β=BETA_H)
    inst = m.rvs()
    obs = inst.observe(n*n*6)
    vb = models.VBayes(m); vb.fit(obs=obs, verbose=False)
    mu, sig = vb.μ().copy(), vb.σ().copy()
    print("  running reference Metropolis ...")
    zs, _ = metropolis(obs, n)
    mc_mu = zs.mean(0)
    # std of the CENTERED scores: the absolute level is unidentified (a global shift the weak prior
    # barely pins), so we compare the identified per-item spread, matching what mean-field VB represents
    mc_sd = (zs - zs.mean(1, keepdims=True)).std(0)
    mu_c = mu - mu.mean(); mc_mu_c = mc_mu - mc_mu.mean()   # only differences are identified

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    a = axes[0]
    lo = min(mu_c.min(), mc_mu_c.min())-0.3; hi = max(mu_c.max(), mc_mu_c.max())+0.3
    a.plot([lo, hi], [lo, hi], "k:", lw=1)
    a.errorbar(mc_mu_c, mu_c, yerr=1.96*sig, fmt="o", ms=4, capsize=2)
    a.set_xlabel("MCMC posterior mean"); a.set_ylabel("VB mean $\\mu_i$"); a.set_title("posterior means")
    a = axes[1]
    lo = 0; hi = max(sig.max(), mc_sd.max())*1.15
    a.plot([lo, hi], [lo, hi], "k:", lw=1)
    a.plot(mc_sd, sig, "o", ms=4)
    a.set_xlabel("MCMC posterior std"); a.set_ylabel("VB std $\\sigma_i$"); a.set_title("posterior std. dev.")
    fig.tight_layout(); fig.savefig(f"{OUT}/vb_vs_mcmc.pdf"); plt.close(fig)
    print("  mean abs diff (means) =", np.abs(mu_c-mc_mu_c).mean(), " (std) =", np.abs(sig-mc_sd).mean())


# ---------------------------------------------------------------------------
# active_selection.pdf: active pair selection, run in the C++ implementation.
# ---------------------------------------------------------------------------
def fig_active(runs=800, steps=1000, n=12, seed=1, out="active_selection.pdf"):
    """Regenerate the active-selection figure from the C++ implementation (CC/ranker), built with
    -O3 if the binary is absent. SLOW at the paper's runs=800 (~1h); pass a smaller `runs` for a
    quicker, noisier preview (write it elsewhere via `out=` to avoid clobbering the paper figure)."""
    exe = "CC/ranker"
    if not os.path.exists(exe):
        print("  building CC/ranker (-O3) ...")
        subprocess.run(["g++", "-O3", "-std=gnu++17", "CC/ranker.cc", "-o", exe, "-pthread"], check=True)
    csv = tempfile.mktemp(suffix=".csv")
    print(f"  running {exe} experiment {n} {steps} {runs} ...")
    subprocess.run([exe, "experiment", str(n), str(steps), str(runs), str(seed), csv], check=True)
    d = np.genfromtxt(csv, delimiter=",", names=True); os.remove(csv)
    keys = [('var', r'min $\mathrm{E}[\sum_i \sigma_i^2]$', 'C0', '-'),
            ('eig', 'max information gain',                 'C4', '-'),
            ('random', 'random',                            '0.45', '--'),
            ('inv', r'min $\mathrm{E}[\#\,\mathrm{inversions}]$', 'C2', '-')]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for k, lab, c, ls in keys:
        ax.plot(d['step'], np.clip(d[k], 0.3, None), color=c, ls=ls, lw=1.3, label=lab)
    ax.set_yscale('log'); ax.set_xlabel("comparisons"); ax.set_ylabel("mean actual inversions")
    ax.yaxis.set_major_locator(FixedLocator([4, 5, 6, 8, 10, 15, 20, 30, 50]))
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(3.5, 70); ax.grid(alpha=0.3, which='both'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/{out}"); plt.close(fig)
    for k, lab, c, ls in keys:
        print(f"  {k:7} final={d[k][-1]:.2f}  AUC={d[k].mean():.2f}")


if __name__ == "__main__":
    only = set(sys.argv[1:]) or {"approx", "correction", "mcmc", "active"}
    if "approx"     in only: print("[approx]");     fig_approx()
    if "correction" in only: print("[correction]"); fig_correction()
    if "mcmc"       in only: print("[mcmc]");        fig_vb_vs_mcmc()
    if "active"     in only: print("[active]");      fig_active()
    print("DONE")
