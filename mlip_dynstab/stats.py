"""Interval estimates and cluster-aware tests for the benchmark's headline rates.

Referee 3 (RSC Advances RA-ART-07-2026-006452) made three connected statistical objections:

1. Accuracies are quoted as bare decimals -- 1.000, 0.895, 0.789 -- when they are 19/19, 17/19
   and 15/19 on nineteen systems, and should carry counts and an interval.
2. The AUC of the ensemble guardrail is computed on sixty units that are repeated measurements
   over a much smaller number of systems, with the five model votes already collapsed into each
   unit, so the units are not independent and cannot be treated as such.
3. The Spearman coefficients relating harmonic to finite-temperature accuracy are computed
   across five models, where the reported range is indistinguishable from noise.

This module supplies what those need: Wilson score intervals, and permutation/bootstrap
machinery that resamples whole *systems* rather than units. The clustering unit is the system
throughout -- that is the level at which the physics, the ground-truth label and the error
structure are shared.

Everything here is hand-rolled on numpy, including the normal quantile. scipy *is* importable in
the analysis environment (``finite_t`` uses ``scipy.optimize.brentq`` and
``scipy.linalg.eigh_tridiagonal``), so this is a consistency choice rather than a necessity: it
matches ``scripts/estimator_noise.py``, which is already scipy-free, and it keeps every
statistic in the paper readable in one place without a version-dependent library between the
ledger and the number. The permutation code follows the pattern that script established for the
phi coefficient.

``estimator_noise.py`` deliberately still carries its own copy rather than importing from here.
Its output is deposited in ``results/estimator_noise.json`` and quoted in ESI section S1.2, and
it is byte-reproducible from its own seed (verified: all four temperatures match the deposited
file exactly). Routing it through this module would change its RNG consumption pattern -- it
shares one generator across the temperature ladder, whereas the functions here seed per call --
and so would silently move published p-values and interval endpoints. Two implementations is the
lesser evil; if they are ever unified, re-deposit the JSON and re-propagate the ESI in the same
commit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

__all__ = [
    "norm_ppf", "wilson", "RateCI", "rate_ci", "fmt_rate",
    "auc", "cluster_permutation_auc", "cluster_bootstrap_auc",
    "phi", "cluster_permutation_phi", "cluster_exact_paired",
    "spearman", "cluster_permutation_spearman",
]


# ------------------------------------------------------------- normal ppf ----

# Acklam's rational approximation to the inverse normal CDF (|abs error| < 1.15e-9). Used only
# to turn a confidence level into a z; hard-coding z = 1.96 would work for the 95% intervals the
# paper reports, but the interval helper takes alpha and should honour it.
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


# ------------------------------------------------------------- rate + CI ----

def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) interval because every rate in this paper is on a small
    denominator and several sit at the boundary: Wald gives MatterSim's 19/19 the interval
    [1.000, 1.000], which asserts certainty from nineteen observations. Wilson does not.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = norm_ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class RateCI:
    k: int
    n: int
    p: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return fmt_rate(self)


def rate_ci(k: int, n: int, alpha: float = 0.05) -> RateCI:
    lo, hi = wilson(k, n, alpha)
    return RateCI(k=int(k), n=int(n), p=(k / n if n else float("nan")), lo=lo, hi=hi)


def fmt_rate(r: RateCI, digits: int = 3) -> str:
    """Render as the paper should quote it: count, value, interval."""
    if r.n == 0:
        return "0/0"
    return f"{r.k}/{r.n} = {r.p:.{digits}f} [{r.lo:.{digits}f}, {r.hi:.{digits}f}]"


# ------------------------------------------------------ clustered resampling ----

def _blocks(cluster: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the unique cluster labels and the row indices belonging to each."""
    uniq = pd.unique(pd.Series(cluster))
    return np.asarray(uniq), [np.flatnonzero(cluster == u) for u in uniq]


def auc(score, label) -> float:
    """Rank AUC of ``score`` as a predictor of boolean ``label`` (Mann-Whitney form).

    Kept numerically identical to ``analysis._auc`` -- including the 0.5 credit for ties -- so
    that the interval machinery here brackets exactly the number the manuscript reports, rather
    than a subtly different estimator.
    """
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def cluster_permutation_auc(score, label, cluster, n_perm: int = 10000,
                            seed: int = 0) -> dict:
    """Two-sided permutation p for an AUC, permuting whole clusters.

    The null is that the score carries no information about the label. Shuffling individual
    units would break the within-system correlation that makes the units non-independent in the
    first place and would return a p that is too small -- which is precisely the criticism. So
    the label vector is permuted in whole-system blocks, preserving each system's internal
    error structure and destroying only the system-to-score correspondence.

    Requires equal-sized blocks (here: fifteen systems x four temperatures); raises otherwise,
    because silently falling back to unit-level shuffling would reintroduce the bug.
    """
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    cluster = np.asarray(cluster)
    _, idx = _blocks(cluster)
    sizes = {len(i) for i in idx}
    if len(sizes) != 1:
        raise ValueError(
            f"cluster_permutation_auc needs equal-sized blocks, got sizes {sorted(sizes)}. "
            "Use the bootstrap, or aggregate to one row per cluster."
        )
    obs = auc(score, label)
    rng = np.random.default_rng(seed)
    order = np.concatenate(idx)
    null = np.empty(n_perm, float)
    for i in range(n_perm):
        perm = rng.permutation(len(idx))
        null[i] = auc(score[order], label[np.concatenate([idx[j] for j in perm])])
    null = null[~np.isnan(null)]
    if math.isnan(obs) or null.size == 0:
        p = float("nan")
    else:
        # two-sided about the 0.5 no-information point, +1 correction
        p = float((np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (null.size + 1))
    return {"auc": round(obs, 3), "p_perm_clustered": round(p, 4),
            "n_perm": int(null.size), "n_clusters": len(idx), "n_units": int(score.size)}


def cluster_bootstrap_auc(score, label, cluster, n_boot: int = 10000,
                          seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI for an AUC, resampling whole clusters with replacement."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    cluster = np.asarray(cluster)
    _, idx = _blocks(cluster)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, float)
    for i in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[j] for j in pick])
        boot[i] = auc(score[rows], label[rows])
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
            "n_boot": int(boot.size), "n_clusters": len(idx)}


# ------------------------------------------------------------------- phi ----

def phi(x, y) -> float:
    """Pearson phi on two boolean vectors; nan if either is constant."""
    x = np.asarray(x, bool).astype(int)
    y = np.asarray(y, bool).astype(int)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cluster_permutation_phi(H: np.ndarray, F: np.ndarray, n_perm: int = 10000,
                            seed: int = 0, alpha: float = 0.05) -> dict:
    """Permutation p and cluster-bootstrap CI for phi between two (system x model) matrices.

    Mirrors the logic of ``scripts/estimator_noise.py`` for use by new analyses. It is NOT the
    implementation behind the published phi numbers, and it is not interchangeable with it: that
    script shares one generator across the temperature ladder while this seeds per call, so the
    two agree on phi, which is deterministic, but not necessarily on the last digits of a
    permutation p or a bootstrap endpoint. See the module docstring.
    """
    H = np.asarray(H, bool)
    F = np.asarray(F, bool)
    n_sys = H.shape[0]
    rng = np.random.default_rng(seed)
    obs = phi(H.ravel(), F.ravel())
    null = np.array([phi(H.ravel(), F[rng.permutation(n_sys)].ravel()) for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    p = (float((np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1))
         if not math.isnan(obs) and null.size else float("nan"))
    boot = np.array([phi(H[i].ravel(), F[i].ravel())
                     for i in (rng.integers(0, n_sys, n_sys) for _ in range(n_perm))])
    boot = boot[~np.isnan(boot)]
    ci = ([float(np.percentile(boot, 100 * alpha / 2)),
           float(np.percentile(boot, 100 * (1 - alpha / 2)))] if boot.size
          else [float("nan"), float("nan")])
    return {"phi": round(obs, 4) if not math.isnan(obs) else float("nan"),
            "p_perm_clustered": round(p, 4), "ci95": [round(c, 4) for c in ci],
            "n_systems": int(n_sys), "n_perm": int(null.size)}


# ----------------------------------------------------- clustered paired test ----

def cluster_exact_paired(a_wins, b_wins, cluster) -> dict:
    """Cluster-aware test that method A beats method B on paired units.

    An exact McNemar over the discordant *units* is the natural test, and it is what a paired
    comparison usually gets, but it assumes the units are independent. Here they are not: the
    units are (system, model, temperature) triples and several share a system, which is the
    same objection Referee 3 raised against the guardrail AUC. Applying the objection to our own
    paired test rather than only to the ones the referee named:

    the randomisation is over whole clusters. Under the null that the two methods are
    exchangeable, the labels "A" and "B" can be swapped within any system without changing the
    distribution, so the null distribution of the net discordance is generated by the 2^k sign
    assignments over the k systems. That is enumerated exactly.

    The price is resolution, and it must be reported rather than hidden. With k systems the
    smallest attainable two-sided p is 2 / 2^k: 0.0625 for five systems, 0.5 for two. A
    clustered p at or near that floor means the direction is consistent across every system and
    the design cannot say more, which is a different statement from a large p.
    """
    a_wins = np.asarray(a_wins, bool)
    b_wins = np.asarray(b_wins, bool)
    cluster = np.asarray(cluster)
    _, idx = _blocks(cluster)
    k = len(idx)

    # per-system net discordance, A-favouring minus B-favouring
    net = np.array([int((a_wins[i] & ~b_wins[i]).sum() - (b_wins[i] & ~a_wins[i]).sum())
                    for i in idx], float)
    obs = float(net.sum())

    signs = np.array(np.meshgrid(*[[1, -1]] * k)).T.reshape(-1, k) if k <= 20 else None
    if signs is None:
        raise ValueError("too many clusters to enumerate; use a sampled randomisation")
    null = signs @ net
    p = float((np.sum(np.abs(null) >= abs(obs))) / null.size)
    return {
        "n_clusters": int(k),
        "per_cluster_net": net.astype(int).tolist(),
        "clusters_favouring_a": int((net > 0).sum()),
        "clusters_favouring_b": int((net < 0).sum()),
        "clusters_tied": int((net == 0).sum()),
        "observed_net": int(obs),
        "p_exact_clustered": round(p, 5),
        "finest_attainable_p": round(2.0 / 2 ** k, 5),
        "at_resolution_floor": bool(abs(p - 2.0 / 2 ** k) < 1e-12),
    }


# --------------------------------------------------------------- spearman ----

def spearman(a, b) -> float:
    return float(pd.Series(list(a)).corr(pd.Series(list(b)), method="spearman"))


def cluster_permutation_spearman(a, b, n_perm: int = 10000, seed: int = 0) -> dict:
    """Exact-if-small permutation p for a Spearman rho.

    With five models there are only 5! = 120 distinct pairings, so the p-value is enumerated
    exactly and ``n_perm`` is ignored. That is the point worth reporting: the finest resolution
    this statistic can express is 1/120, so no arrangement of five models can reach p < 0.008,
    and the coefficient cannot carry an inference no matter what it equals.
    """
    from itertools import permutations
    a = list(a)
    b = list(b)
    n = len(a)
    obs = spearman(a, b)
    if n <= 8:
        null = np.array([spearman(a, [b[i] for i in p]) for p in permutations(range(n))])
        exact = True
    else:
        rng = np.random.default_rng(seed)
        null = np.array([spearman(a, [b[i] for i in rng.permutation(n)]) for _ in range(n_perm)])
        exact = False
    null = null[~np.isnan(null)]
    p = (float((np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1))
         if not math.isnan(obs) and null.size else float("nan"))
    return {"rho": round(obs, 3) if not math.isnan(obs) else float("nan"),
            "p_perm": round(p, 4), "n": n, "exact": exact,
            "finest_resolvable_p": round(1.0 / max(null.size, 1), 4)}
