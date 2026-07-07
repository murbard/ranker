"""
Unit and property-style tests for the ranker: the observation kernel, the Davidson tie model,
its CAVI weight updates, the API guards, and cross-implementation agreement with CC/ranker.

Pure pytest, no extra dependencies. Property-style tests use seeded numpy generators (fixed
sample loops) rather than hypothesis, so failures are deterministic and reproducible.

Run from the repository root:  pytest tests/test_ties.py
"""
import functools
import os
import shutil
import sys
import hashlib
import subprocess
import tempfile
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.stats import invgamma
from scipy.stats import norm as norm_dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import models  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------------------------
# Shared fixture: the verify3 state (mirrored in CC/ranker.cc's verify3 mode)
# ---------------------------------------------------------------------------------------------
N = 6
WINS = [(0, 1, 3), (1, 0, 1), (2, 3, 2), (0, 4, 2), (4, 5, 1), (5, 2, 2), (3, 1, 1), (2, 5, 2), (4, 0, 1)]
TIES = [(0, 1, 2), (2, 3, 1), (1, 4, 2), (3, 5, 1)]
P0 = np.concatenate([[0.30, -0.50, 0.10, 0.80, -0.20, 0.40],
                     [0.50, 0.70, 0.45, 0.90, 0.60, 0.55], [1.5, 2.5]])

# Bit-exact references for the no-ties evaluation of the fixture (val as IEEE hex; SHA-256 of the
# concatenated val/gradient/h_diag/h_obs bytes).  These freeze the kernel arithmetic: any change
# to the erf approximation, the correction, or the summation order trips them and must be a
# conscious re-baselining.  To re-baseline, evaluate the fixture and record
#     float(gh.val).hex()   and   hashlib.sha256(blob).hexdigest()
# where blob = np.concatenate([[gh.val], gh.g, gh.h_diag, gh.h_obs.data]).tobytes().
# The freeze only makes sense on the reference libm/numpy: a canary detects other platforms
# (macOS/musl/FMA differ by ulps) and skips the bit assertions there -- the value-level and
# quadrature tests below still cover the kernel everywhere.
NOTIES_VAL_HEX = "0x1.f804cbfb3ac55p+3"          # 15.75058555...
NOTIES_BLOB_SHA = "eb5a99ee44e186336d947b4d4aec38b36cc39d1ce47d92032ba283174853152f"
REF_LIBM = (float(erf(0.7)).hex() == "0x1.5b08c21171645p-1"
            and float(np.exp(-0.49)).hex() == "0x1.39aa2aaf607f6p-1")
# Value references at print precision (shared with CC/ranker verify3's output)
TIES_VAL = 35.22469640          # fixture eval with the non-uniform w below
FITTED_VAL = 25.49414535        # fixture after fit
FITTED_ELAM = 0.99236733


def wins_coo():
    X = np.zeros((N, N))
    for i, j, c in WINS:
        X[i, j] = c
    return coo_matrix(X)


def ties_coo(entries=TIES):
    T = np.zeros((N, N))
    for i, j, c in entries:
        T[i, j] = c
    return coo_matrix(T)


def fixture_vb(enable_ties=False, nonuniform_w=False):
    vb = models.VBayes(models.Model(n=N, α=1.2, β=2.0))
    vb.params = P0.copy()
    if enable_ties:
        vb.enable_ties()
        if nonuniform_w:
            t = vb.ties
            t.w = np.array([1.0 + j % 3 for j in range(t.J)], dtype=float)
            t.w /= t.w.sum()
            t.premix()
    return vb


def exact_softplus(d):
    """log(1 + e^{-d}), numerically stable."""
    return np.log1p(np.exp(-np.abs(d))) + np.clip(-d, 0, None)


def gauss_quad(f, m, s):
    return quad(lambda d: np.exp(-(d - m) ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi) / s * f(d),
                m - 10 * s - 5, m + 10 * s + 5, limit=300)[0]


# ---------------------------------------------------------------------------------------------
# Kernel regression: bit-exact freeze of the no-ties evaluation
# ---------------------------------------------------------------------------------------------
class TestKernelFreeze:
    def test_noties_eval_value(self):
        """Value-level check that runs on every platform."""
        vb = fixture_vb()
        assert vb.eval(wins_coo()).val == pytest.approx(15.75058555, abs=5e-8)

    @pytest.mark.skipif(not REF_LIBM, reason="non-reference libm: bit-exact freeze not applicable")
    def test_noties_eval_bit_exact(self):
        vb = fixture_vb()
        gh = vb.eval(wins_coo(), compute_gradient=True, compute_hessian=True)
        assert float(gh.val).hex() == NOTIES_VAL_HEX
        blob = np.concatenate([[gh.val], gh.g, gh.h_diag, gh.h_obs.data]).tobytes()
        assert hashlib.sha256(blob).hexdigest() == NOTIES_BLOB_SHA

    def test_ties_eval_value(self):
        vb = fixture_vb(enable_ties=True, nonuniform_w=True)
        assert vb.eval(wins_coo(), tie_obs=ties_coo()).val == pytest.approx(TIES_VAL, abs=5e-8)

    def test_fitted_fixture(self):
        vb = fixture_vb(enable_ties=True)
        vb.fit(wins_coo(), tie_obs=ties_coo(), verbose=False)
        assert vb.eval(wins_coo(), tie_obs=ties_coo()).val == pytest.approx(FITTED_VAL, abs=1e-6)
        # E[lambda] gets a looser tolerance: lambda is weakly identified, so the val stopping rule
        # amplifies along the flat direction (different libm/BLAS shift it more than val)
        assert vb.ties.E_λ() == pytest.approx(FITTED_ELAM, abs=1e-5)
        # certification: the returned weights are CAVI-stationary for the returned scores
        assert vb.cavi(wins_coo(), ties_coo()) < models.W_STAT

    def test_prior_terms_match_independent_quadrature(self):
        """With no observations, val is exactly the entropy + prior cross-entropy part of the
        KL.  Recompute it from scipy primitives alone (no models.py algebra): this is the one
        check that would catch a wrong coefficient in the alpha/beta prior terms, which FD (self-
        consistent), the bit freeze (freezes whatever is computed), and the cross-implementation
        comparison (shared derivation) all miss."""
        vb = fixture_vb()
        val = vb.eval(coo_matrix((N, N))).val
        α, β = 1.5, 2.5
        Mα, Mβ = 1.2, 2.0
        σ, μ = P0[N:2 * N], P0[:N]
        q = invgamma(α, scale=1 / β)
        # - entropies of the variational factors
        indep = -q.entropy() - sum(norm_dist.entropy(scale=s) for s in σ)
        # - E_q[log p(v)] for the InvGamma prior, by quadrature
        p = invgamma(Mα, scale=1 / Mβ)
        indep += quad(lambda v: -q.pdf(v) * p.logpdf(v), 0, np.inf, limit=300)[0]
        # - E_q(v) E_q(z)[ -log N(z; 0, sqrt(v)) ], inner expectation in closed form, v by quadrature
        m2s2 = float(np.sum(μ ** 2 + σ ** 2))
        indep += quad(lambda v: q.pdf(v) * (N / 2 * np.log(2 * np.pi * v) + m2s2 / (2 * v)),
                      0, np.inf, limit=300)[0]
        assert val == pytest.approx(indep, abs=1e-8)


# ---------------------------------------------------------------------------------------------
# Kernel properties (seeded property-style loops)
# ---------------------------------------------------------------------------------------------
class TestKernelProperties:
    def test_F_kernel_matches_quadrature(self):
        """The erf+correction closed form integrates log(1+e^{-d}) to the k=5 accuracy."""
        rng = np.random.default_rng(0)
        for _ in range(25):
            m, s = rng.normal(0, 2), rng.uniform(0.1, 3)
            F = models.F_kernel(m, s, 0)[0]
            assert F == pytest.approx(gauss_quad(exact_softplus, m, s), abs=2e-6)

    def test_F_reflection_identities(self):
        """F(-m) = m + F(m);  Fmu(-m) = -1 - Fmu(m);  Fsigma even.  (Used by the pair loop and
        the acquisition parity fills.)"""
        rng = np.random.default_rng(1)
        for _ in range(50):
            m, s = rng.normal(0, 3), rng.uniform(0.05, 4)
            F, Fm, Fs, *_ = models.F_kernel(m, s, 1)
            F2, Fm2, Fs2, *_ = models.F_kernel(-m, s, 1)
            assert F2 == pytest.approx(m + F, rel=1e-12, abs=1e-12)
            assert Fm2 == pytest.approx(-1 - Fm, rel=1e-12, abs=1e-12)
            assert Fs2 == pytest.approx(Fs, rel=1e-12, abs=1e-12)

    def test_order_contracts(self):
        """order=0: derivatives zero.  order=1: first derivatives equal order-2's, seconds zero
        (including the sigma-sigma slot the chain rule could leak into)."""
        rng = np.random.default_rng(2)
        for _ in range(25):
            m, s = rng.normal(0, 2), rng.uniform(0.1, 3)
            full = models.F_kernel(m, s, 2)
            o1 = models.F_kernel(m, s, 1)
            o0 = models.F_kernel(m, s, 0)
            assert o0[0] == full[0] and o0[1:] == (0.0,) * 5
            assert o1[:3] == full[:3] and o1[3:] == (0.0,) * 3
            c1 = models.correction(m, s, 1)
            assert c1[3:] == (0.0,) * 3          # bump_eval honors the same contract

    def test_atom_bumps_match_quadrature(self):
        """Each tie atom's fitted bump integrates log(1 + lam/(2cosh(d/2))) in closed form."""
        t = models.TieModel()
        rng = np.random.default_rng(3)
        for j in range(0, t.J, 3):
            lam = t.λ[j]
            m, s = rng.normal(0, 1.5), rng.uniform(0.2, 2.5)
            B = models.bump_eval(t.a, t.P[j], t.Q[j], m, s, 0)[0]
            target = gauss_quad(lambda d: np.log1p(lam / (2 * np.cosh(d / 2))), m, s)
            assert B == pytest.approx(target, abs=1e-6)

    def test_davidson_normalizer_identity(self):
        """E[log(2cosh(d/2)+lam)] = mu/2 + F(mu,sig) + E[b_lam] -- the core closed form."""
        t = models.TieModel()
        rng = np.random.default_rng(4)
        for j in (0, t.J // 2, t.J - 1):
            lam = t.λ[j]
            m, s = rng.normal(0, 1.5), rng.uniform(0.3, 2)
            closed = m / 2 + models.F_kernel(m, s, 0)[0] + models.bump_eval(t.a, t.P[j], t.Q[j], m, s, 0)[0]
            target = gauss_quad(lambda d: np.log(2 * np.cosh(d / 2) + lam), m, s)
            assert closed == pytest.approx(target, abs=2e-6)

    def test_basis_phi_consistent_with_bump_eval(self):
        """cavi's basis integrals are exactly what eval integrates (per-atom reconstruction)."""
        t = models.TieModel()
        rng = np.random.default_rng(5)
        for _ in range(20):
            m, s2 = rng.normal(0, 2), rng.uniform(0.05, 6)
            φp, φq = models.basis_phi(t.a, m, s2)
            for j in (0, t.J - 1):
                direct = models.bump_eval(t.a, t.P[j], t.Q[j], m, np.sqrt(s2), 0)[0]
                assert t.P[j] @ φp + t.Q[j] @ φq == pytest.approx(direct, rel=1e-12, abs=1e-14)


# ---------------------------------------------------------------------------------------------
# Gradient / Hessian correctness (finite differences), with and without ties
# ---------------------------------------------------------------------------------------------
class TestDerivatives:
    @pytest.mark.parametrize("with_ties", [False, True])
    def test_gradient_and_hessian_fd(self, with_ties):
        vb = fixture_vb(enable_ties=with_ties, nonuniform_w=with_ties)
        tobs = ties_coo() if with_ties else None
        gh = vb.eval(wins_coo(), compute_gradient=True, compute_hessian=True, tie_obs=tobs)
        H = np.array([gh.hdot(col) for col in np.eye(2 * N + 2)])
        h = 1e-6

        def eval_at(k, sgn):
            vb.params = P0.copy()
            vb.params[k] += sgn * h
            g = vb.eval(wins_coo(), compute_gradient=True, tie_obs=tobs)
            return g.val, g.g.copy()

        for k in range(2 * N + 2):
            vp, gp = eval_at(k, +1)
            vm, gm = eval_at(k, -1)
            vb.params = P0.copy()
            assert (vp - vm) / (2 * h) == pytest.approx(gh.g[k], abs=1e-6)
            np.testing.assert_allclose((gp - gm) / (2 * h), H[k], atol=1e-6)

    def test_gradient_fd_random_regimes(self):
        """FD sweep over random parameter regimes: well-separated means (erf saturation), tiny
        and large sigmas, off-prior alpha/beta -- territory the mild fixture never reaches."""
        rng = np.random.default_rng(21)
        n, h = 5, 1e-6
        for trial in range(10):
            with_ties = trial % 2 == 1
            vb = models.VBayes(models.Model(n=n))
            params = np.concatenate([rng.normal(0, 2.5, n), rng.uniform(0.05, 3, n),
                                     [rng.uniform(0.5, 4), rng.uniform(0.5, 5)]])
            vb.params = params.copy()
            X = rng.integers(0, 4, (n, n))
            np.fill_diagonal(X, 0)
            obs = coo_matrix(X)
            tobs = None
            if with_ties:
                vb.enable_ties()
                tobs = coo_matrix(np.triu(rng.integers(0, 3, (n, n)), 1))
            gh = vb.eval(obs, compute_gradient=True, tie_obs=tobs)
            for k in range(2 * n + 2):
                vb.params = params.copy(); vb.params[k] += h
                fp = vb.eval(obs, tie_obs=tobs).val
                vb.params = params.copy(); vb.params[k] -= h
                fm = vb.eval(obs, tie_obs=tobs).val
                vb.params = params.copy()
                assert (fp - fm) / (2 * h) == pytest.approx(gh.g[k], rel=2e-4, abs=5e-6)

    def test_dobs_matches_per_direction_kernel(self):
        vb = fixture_vb()
        dobs = np.zeros((N, N, 2))
        vb.eval(wins_coo(), compute_gradient=True, compute_hessian=True, dobs=dobs)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                σδ = np.sqrt(vb.σ()[i] ** 2 + vb.σ()[j] ** 2)
                _, Fm, Fs, *_ = models.F_kernel(vb.μ()[i] - vb.μ()[j], σδ, 1)
                assert dobs[i, j, 1] == pytest.approx(Fm, rel=1e-12, abs=1e-12)
                assert dobs[i, j, 0] == pytest.approx(Fs / σδ, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# CAVI properties
# ---------------------------------------------------------------------------------------------
class TestCavi:
    def test_idempotent(self):
        vb = fixture_vb(enable_ties=True)
        vb.cavi(wins_coo(), ties_coo())
        assert vb.cavi(wins_coo(), ties_coo()) == 0.0   # same params => exactly the same weights

    def test_minimizes_val_over_weights(self):
        """cavi's closed-form update is the exact coordinate minimizer of val over w."""
        rng = np.random.default_rng(6)
        vb = fixture_vb(enable_ties=True)
        vb.cavi(wins_coo(), ties_coo())
        base = vb.eval(wins_coo(), tie_obs=ties_coo()).val
        w_star = vb.ties.w.copy()
        for _ in range(20):
            w = w_star * np.exp(rng.normal(0, 0.3, vb.ties.J))
            vb.ties.w = w / w.sum()
            vb.ties.premix()
            assert vb.eval(wins_coo(), tie_obs=ties_coo()).val >= base - 1e-12
        vb.ties.w = w_star
        vb.ties.premix()

    def test_alternation_monotone(self):
        """val (the -ELBO, including the categorical KL) never increases across CAVI rounds."""
        models.rng = np.random.default_rng(7)
        m = models.Model(n=10)
        inst = m.rvs()
        w10, t10 = inst.observe_ties(400, 0.75)
        vb = models.VBayes(m)
        vb.enable_ties()
        pairs = models.tie_pairs(w10, t10, 10)
        for _ in range(4):
            vb.newton(w10, verbose=False, tie_obs=t10, pairs=pairs)
            before = vb.eval(w10, tie_obs=t10, pairs=pairs).val
            vb.cavi(w10, t10, pairs)
            after = vb.eval(w10, tie_obs=t10, pairs=pairs).val
            assert after <= before + 1e-10


# ---------------------------------------------------------------------------------------------
# Input handling: degenerate observations, tie storage orientation, guards
# ---------------------------------------------------------------------------------------------
class TestInputHandling:
    def test_diagonal_entries_ignored(self):
        vb = fixture_vb()
        gh_clean = vb.eval(wins_coo(), compute_gradient=True, compute_hessian=True)
        Xd = wins_coo().toarray()
        Xd[2, 2] = 7          # explicit self-comparison must be a no-op
        gh_diag = vb.eval(coo_matrix(Xd), compute_gradient=True, compute_hessian=True)
        assert gh_diag.val == gh_clean.val
        np.testing.assert_array_equal(gh_diag.g, gh_clean.g)
        for col in np.eye(2 * N + 2):
            np.testing.assert_allclose(gh_diag.hdot(col), gh_clean.hdot(col), atol=1e-14)

    def test_tie_orientation_invariance(self):
        """Tie counts stored in either triangle (or split across both) give identical results."""
        upper = TIES
        lower = [(j, i, c) for i, j, c in TIES]
        mixed = [TIES[0], (TIES[1][1], TIES[1][0], TIES[1][2])] + TIES[2:]
        vals = []
        for entries in (upper, lower, mixed):
            vb = fixture_vb(enable_ties=True, nonuniform_w=True)
            vals.append(vb.eval(wins_coo(), tie_obs=ties_coo(entries)).val)
        assert vals[0] == vals[1] == vals[2]

    def test_guards(self):
        vb = fixture_vb(enable_ties=True)
        with pytest.raises(ValueError):
            vb.eval(wins_coo())                       # ties enabled, tie_obs missing
        with pytest.raises(ValueError):
            vb.fit(wins_coo(), verbose=False)
        with pytest.raises(ValueError):
            vb.cavi(wins_coo())
        with pytest.raises(ValueError):                # dobs with ties is not implemented
            vb.eval(wins_coo(), compute_gradient=True, compute_hessian=True,
                    dobs=np.zeros((N, N, 2)), tie_obs=ties_coo())
        with pytest.raises(NotImplementedError):
            vb.best_pair(wins_coo())
        with pytest.raises(NotImplementedError):
            vb.KL(wins_coo())

    def test_ties_off_warning(self):
        def fit_recording_warnings(vb, tie_obs):
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                # re-arm the module's RuntimeWarning->error promotion: newton's overflow
                # backoff depends on it, and simplefilter("always") just cleared it
                warnings.filterwarnings("error", category=RuntimeWarning)
                vb.fit(wins_coo(), tie_obs=tie_obs, verbose=False)
            return wl

        wl = fit_recording_warnings(fixture_vb(), ties_coo())
        assert any("IGNORED" in str(w.message) for w in wl)
        # ... but a tie matrix of stored explicit zeros must NOT warn (assert the SPECIFIC
        # message's absence: unrelated future DeprecationWarnings must not break this test)
        tz = coo_matrix((np.zeros(3), ([0, 1, 2], [1, 2, 3])), shape=(N, N))
        wl = fit_recording_warnings(fixture_vb(), tz)
        assert not any("IGNORED" in str(w.message) for w in wl)

    def test_empty_tie_matrix_is_valid(self):
        """No ties observed is legitimate data: the posterior should favor small lambda."""
        vb = fixture_vb(enable_ties=True)
        vb.fit(wins_coo(), tie_obs=coo_matrix((N, N)), verbose=False)
        assert vb.ties.E_λ() < 0.7

    def test_truncated_atom_file_raises(self, tmp_path):
        src = open(os.path.join(REPO, "fit_results", "tie_atoms_J16.txt")).read().split("\n")
        bad = tmp_path / "bad_atoms.txt"
        bad.write_text("\n".join(src[:2] + [" ".join(src[2].split()[:5])] + src[3:]))
        with pytest.raises(ValueError, match="bad atom line"):
            models.TieModel(str(bad))


# ---------------------------------------------------------------------------------------------
# End-to-end recovery (seeded)
# ---------------------------------------------------------------------------------------------
class TestEndToEnd:
    def test_lambda_recovery(self):
        models.rng = np.random.default_rng(11)
        m = models.Model(n=12)
        inst = m.rvs()
        w12, t12 = inst.observe_ties(1500, 0.75)
        vb = models.VBayes(m)
        vb.enable_ties()
        vb.fit(w12, tie_obs=t12, verbose=False)
        assert 0.4 < vb.ties.E_λ() < 1.4           # lambda is weakly identified; loose bounds
        assert vb.actual_inversions(inst) <= 12    # ranking is well recovered
        assert vb.cavi(w12, t12) < models.W_STAT   # returned state is certified stationary

    def test_no_ties_fit_unaffected(self):
        models.rng = np.random.default_rng(12)
        m = models.Model(n=10)
        inst = m.rvs()
        obs = inst.observe(400)
        vb = models.VBayes(m)
        vb.fit(obs, verbose=False)
        assert np.isfinite(vb.expected_inversions())
        assert vb.actual_inversions(inst) <= 15


# ---------------------------------------------------------------------------------------------
# Cross-implementation agreement with CC/ranker (skipped when no binary and no compiler)
# ---------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _ranker_binary():
    """The CC/ranker binary to cross-check against, or None if unavailable.
    Lazy (called from inside the test, never at collection time) and staleness-aware: the
    project's own binary is used only if it is newer than ranker.cc; otherwise a fresh test
    binary is built into a scratch path (NOT CC/ranker, so a -O2 test build never shadows the
    user's -O3 benchmark binary)."""
    src = os.path.join(REPO, "CC", "ranker.cc")
    exe = os.path.join(REPO, "CC", "ranker")
    if os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(src):
        return exe
    if not shutil.which("g++"):
        return None
    scratch = os.path.join(tempfile.gettempdir(),
                           f"ranker_test_{hashlib.sha256(open(src, 'rb').read()).hexdigest()[:16]}")
    if not os.path.exists(scratch):
        try:
            subprocess.run(["g++", "-O2", "-std=gnu++17", src, "-o", scratch, "-pthread"],
                           check=True, capture_output=True, timeout=300)
        except Exception:
            return None
    return scratch


class TestCrossImplementation:
    def test_verify3_values_match(self):
        exe = _ranker_binary()
        if exe is None:
            pytest.skip("CC/ranker binary unavailable and g++ absent")
        out = subprocess.run([exe, "verify3"], cwd=REPO, capture_output=True,
                             text=True, timeout=300).stdout
        cpp_noties = float(out.split("no-ties: val=")[1].split()[0])
        cpp_ties = float(out.split("ties   : val=")[1].split()[0])
        cpp_fitted = float(out.split("fitted: val=")[1].split()[0])
        cpp_elam = float(out.split("E[λ]=")[1].split()[0])

        vb = fixture_vb()
        assert vb.eval(wins_coo()).val == pytest.approx(cpp_noties, abs=5e-8)
        vb = fixture_vb(enable_ties=True, nonuniform_w=True)
        assert vb.eval(wins_coo(), tie_obs=ties_coo()).val == pytest.approx(cpp_ties, abs=5e-8)
        vb = fixture_vb(enable_ties=True)
        vb.fit(wins_coo(), tie_obs=ties_coo(), verbose=False)
        assert vb.eval(wins_coo(), tie_obs=ties_coo()).val == pytest.approx(cpp_fitted, abs=1e-6)
        assert vb.ties.E_λ() == pytest.approx(cpp_elam, abs=1e-5)   # weakly identified: loose
