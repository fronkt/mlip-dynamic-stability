"""Estimator-noise treatment for the harmonic layer.

Answers the Dyna-Mat (arXiv:2607.03433) sec. II.4 objection, which attributes a weak
harmonic/finite-T correlation to the harmonic estimator being NOISY -- explicitly "its
sensitivity to the finite displacement used in phonon calculations" -- rather than to a
physical harmonic/finite-T distinction. A referee will therefore claim the CHGNet
reordering is estimator noise.

Four analyses, all zero-compute (they re-read the deposited ledger; nothing is measured):

  A. EMPIRICAL NOISE FLOOR. Harmonic generation 1 vs generation 2 is a genuine paired
     replicate: identical physics settings (disp=0.01 Ang, 2x2x2, 12x12x12) re-measured in
     independently pinned environments (see mlip_dynstab/__init__.py:23-24). 100 paired
     (system, model) measurements of the same estimator. Reported PER MODEL -- the pooled
     spread is an ORB-v2 float32 artifact and understates CHGNet/MatterSim by ~100x.

  B. TOLERANCE-BANDED ACCURACY. Fig. 1 sweeps the decision THRESHOLD, not the estimator,
     and pools all five models. Per-model, CHGNet's harmonic accuracy is 0.789 only for
     tol in [0.05, 0.20]; two of its four errors are marginal false-unstables that the
     manuscript itself calls finite-displacement noise (manuscript.md:250-253). This bands
     the number and tests whether the ORDERING is tolerance-invariant.

  C. CLUSTER-AWARE ASSOCIATION TEST. sec. 4's prose asserts anti-ASSOCIATION, but the
     reported McNemar p tests marginal HOMOGENEITY -- a different hypothesis -- and phi
     carries no p-value or CI anywhere in the codebase. The 75 matched pairs are also
     clustered (5 models x 15 systems) and currently treated as independent. This adds a
     system-clustered permutation null and a system-clustered bootstrap CI for phi.

  D. Reports what is NOT covered: a displacement-amplitude sweep, which is the axis
     Dyna-Mat names by name. That requires compute and is flagged, not faked.

numpy/pandas only -- scipy is deliberately absent from this repo's pinned environments and
this script does not add it. Writes results/estimator_noise.json; does NOT touch
results/ledger.parquet (that ledger is for measurement units keyed on uhash; this is
derived analysis, following the results/convergence_study.parquet precedent).

Usage: python scripts/estimator_noise.py [--n-perm 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from mlip_dynstab import analysis as A

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "results" / "ledger.parquet"
OUT = REPO / "results" / "estimator_noise.json"

# The two models that carry the load-bearing reordering claim.
CLAIM_MODELS = ("chgnet", "mattersim")
TOLS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)


def _generation(df: pd.DataFrame) -> pd.Series:
    """Method generation per row, mirroring analysis.canonical()'s inference exactly.

    canonical() drops superseded rows; we need them, so we reproduce its version logic
    rather than calling it. Keep in sync with analysis.py:18-47.
    """
    inferred = pd.Series(1.0, index=df.index)
    if "ft_n_imag_total" in df.columns:
        inferred[(df["method"] == "softmode") & df["ft_n_imag_total"].notna()] = 3.0
    return pd.to_numeric(df["method_version"], errors="coerce").fillna(inferred)


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    """The scored set: drop borderline systems (analysis.per_model_table's convention)."""
    return df[~df["system"].isin(A.borderline_systems())]


# --------------------------------------------------------------------------- A
def noise_floor(led: pd.DataFrame) -> dict:
    """Paired harmonic v1 vs v2 re-measurement spread, per model."""
    h = led[led["method"] == "harmonic"].copy()
    h["gen"] = _generation(h)
    v1 = h[h["gen"] == 1.0][["system", "model", "min_freq_thz", "dynamically_stable"]]
    v2 = h[h["gen"] == 2.0][["system", "model", "min_freq_thz", "dynamically_stable"]]
    m = v1.merge(v2, on=["system", "model"], suffixes=("_v1", "_v2"))
    m["abs_delta"] = (m["min_freq_thz_v2"] - m["min_freq_thz_v1"]).abs()
    m["call_flip"] = m["dynamically_stable_v1"].astype(bool) != m["dynamically_stable_v2"].astype(bool)

    per_model = {}
    for model, g in m.groupby("model"):
        d = g["abs_delta"].to_numpy(float)
        per_model[model] = {
            "n_paired": int(len(g)),
            "median_abs_delta_thz": float(np.median(d)),
            "mean_abs_delta_thz": float(d.mean()),
            "p95_abs_delta_thz": float(np.percentile(d, 95)),
            "max_abs_delta_thz": float(d.max()),
            "call_flips": int(g["call_flip"].sum()),
            "flipped_units": sorted(g.loc[g["call_flip"], "system"].tolist()),
        }

    pooled = m["abs_delta"].to_numpy(float)
    worst_claim = max(per_model[k]["max_abs_delta_thz"] for k in CLAIM_MODELS if k in per_model)
    return {
        "per_model": per_model,
        "pooled": {
            "n_paired": int(len(m)),
            "median_abs_delta_thz": float(np.median(pooled)),
            "mean_abs_delta_thz": float(pooled.mean()),
            "max_abs_delta_thz": float(pooled.max()),
            "call_flips": int(m["call_flip"].sum()),
            "WARNING": "pooled spread is an ORB-v2 float32 artifact; report per model",
        },
        "claim_models_worst_max_abs_delta_thz": float(worst_claim),
        "claim_models_call_flips": int(
            sum(per_model[k]["call_flips"] for k in CLAIM_MODELS if k in per_model)
        ),
    }


# --------------------------------------------------------------------------- B
def tolerance_bands(led: pd.DataFrame) -> dict:
    """Per-model harmonic accuracy across the imaginary-frequency tolerance, plus an
    explicit test of whether the CHGNet-vs-MatterSim ORDERING is tolerance-invariant."""
    can = A.canonical(led)
    h = _scored(can[can["method"] == "harmonic"])
    gt = h["gt_stable"].astype(bool)

    per_model: dict[str, dict[str, float]] = {}
    for tol in TOLS:
        pred = h["min_freq_thz"] >= -tol
        for model, idx in h.groupby("model").groups.items():
            sub = h.index.isin(idx)
            acc = float((pred[sub] == gt[sub]).mean())
            per_model.setdefault(model, {})[f"{tol:.2f}"] = acc

    # Which models are uniquely worst, at each tolerance?
    worst = {}
    for tol in TOLS:
        k = f"{tol:.2f}"
        accs = {mo: per_model[mo][k] for mo in per_model}
        lo = min(accs.values())
        worst[k] = sorted(mo for mo, v in accs.items() if v == lo)

    # Ordering invariance: is CHGNet strictly below MatterSim harmonically at every tol?
    ordering = {}
    if all(mo in per_model for mo in CLAIM_MODELS):
        lo_m, hi_m = CLAIM_MODELS
        for tol in TOLS:
            k = f"{tol:.2f}"
            ordering[k] = bool(per_model[lo_m][k] < per_model[hi_m][k])

    band_models = {}
    for mo, accs in per_model.items():
        vals = [accs[f"{t:.2f}"] for t in TOLS]
        plateau = [accs[f"{t:.2f}"] for t in TOLS if 0.05 <= t <= 0.20]
        band_models[mo] = {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "plateau_min": float(min(plateau)),
            "plateau_max": float(max(plateau)),
        }

    return {
        "per_model_accuracy": per_model,
        "band": band_models,
        "uniquely_worst_at": worst,
        "chgnet_below_mattersim_at_every_tol": ordering,
        "ordering_invariant": bool(ordering) and all(ordering.values()),
    }


# --------------------------------------------------------------------------- C
def _phi(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson phi on two boolean vectors; nan if either is constant."""
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def association(led: pd.DataFrame, temps=(100.0, 300.0, 600.0, 900.0),
                n_perm: int = 10000, seed: int = 0) -> dict:
    """System-clustered permutation null and bootstrap CI for phi on the matched set.

    The matched set follows analysis.h2_paired_summary: drop borderline systems and any
    system whose name contains "bcc". NOTE that also drops the superionic agi_bcc, which
    is why n=15 systems rather than 16 -- preserved here for exact comparability with the
    published numbers, and flagged in the output.
    """
    can = A.canonical(led)
    bl = A.borderline_systems()

    def matched(d):
        return d[(~d["system"].isin(bl)) & (~d["system"].str.contains("bcc"))]

    h = matched(can[can["method"] == "harmonic"])[["system", "model", "pred_stable", "gt_stable"]]
    h = h.rename(columns={"pred_stable": "hp", "gt_stable": "hg"})

    rng = np.random.default_rng(seed)
    out = {
        "matched_set_note": (
            "systems containing 'bcc' are dropped, which also removes the superionic "
            "agi_bcc; retained for comparability with the published n=15"
        ),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "by_temperature": {},
    }

    for t in temps:
        f = matched(can[(can["method"] == "softmode") & (can["temperature_K"] == t)])
        f = f[["system", "model", "pred_stable", "gt_stable"]].rename(
            columns={"pred_stable": "fp", "gt_stable": "fg"}
        )
        m = h.merge(f, on=["system", "model"])
        if m.empty:
            continue
        m["hok"] = m["hp"].astype(bool) == m["hg"].astype(bool)
        m["fok"] = m["fp"].astype(bool) == m["fg"].astype(bool)

        # (system x model) correctness matrices; clustering is by system (rows).
        piv_h = m.pivot_table(index="system", columns="model", values="hok", aggfunc="first")
        piv_f = m.pivot_table(index="system", columns="model", values="fok", aggfunc="first")
        piv_f = piv_f.reindex(index=piv_h.index, columns=piv_h.columns)
        H = piv_h.to_numpy(bool)
        F = piv_f.to_numpy(bool)
        n_sys, n_mod = H.shape

        phi_obs = _phi(H.ravel(), F.ravel())

        # Permutation null: shuffle whole systems' finite-T rows against the harmonic
        # rows, preserving each system's within-row (across-model) correlation structure.
        null = np.empty(n_perm, float)
        for i in range(n_perm):
            null[i] = _phi(H.ravel(), F[rng.permutation(n_sys)].ravel())
        null = null[~np.isnan(null)]
        if np.isnan(phi_obs) or null.size == 0:
            p_perm = float("nan")
        else:
            # two-sided, +1 correction
            p_perm = float((np.sum(np.abs(null) >= abs(phi_obs)) + 1) / (null.size + 1))

        # Cluster bootstrap CI: resample systems with replacement.
        boot = np.empty(n_perm, float)
        for i in range(n_perm):
            idx = rng.integers(0, n_sys, n_sys)
            boot[i] = _phi(H[idx].ravel(), F[idx].ravel())
        boot = boot[~np.isnan(boot)]
        ci = (
            [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            if boot.size
            else [float("nan"), float("nan")]
        )

        a = int((H & F).sum())
        b = int((H & ~F).sum())
        c = int((~H & F).sum())
        d = int((~H & ~F).sum())

        out["by_temperature"][f"{t:.0f}"] = {
            "n_pairs": int(H.size),
            "n_systems": int(n_sys),
            "n_models": int(n_mod),
            "cells": {"a_both_ok": a, "b_harm_ok_ft_bad": b, "c_harm_bad_ft_ok": c, "d_both_bad": d},
            "phi": phi_obs,
            "phi_perm_p_two_sided": p_perm,
            "phi_cluster_bootstrap_ci95": ci,
            "phi_ci_excludes_zero": bool(
                not any(np.isnan(ci)) and (ci[0] > 0 or ci[1] < 0)
            ),
            "empty_cell_warning": (
                "d cell is empty; phi's sign is driven by a structural zero" if d == 0 else None
            ),
        }
    return out


# --------------------------------------------------------------------------- D
def not_covered() -> dict:
    return {
        "displacement_amplitude_sweep": {
            "status": "NOT COMPUTED - requires compute",
            "why_it_matters": (
                "Dyna-Mat sec. II.4 names finite displacement specifically. Analysis A "
                "probes environment/library nondeterminism at FIXED disp=0.01 Ang, so it "
                "is necessary but not sufficient against that exact objection."
            ),
            "how": (
                "disp is hardcoded at harmonic.py:40 and is NOT in the unit hash "
                "(cli.py:54). Add it to the settings dict and bump METHOD_VERSION, or "
                "has_unit() silently skips every re-run unit."
            ),
            "cost": "95 scored units x 4 displacements ~= 400 units at 0.31-9.70 s each",
            "suggested_grid_ang": [0.005, 0.01, 0.02, 0.03],
        },
        "independence_caveat": (
            "v1 and v2 share the same code path, disp and seedless algorithm, so A is a "
            "LOWER BOUND on estimator noise, not a full characterisation."
        ),
    }


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    led = pd.read_parquet(LEDGER)  # deliberately NOT canonical(): A needs superseded rows
    res = {
        "A_noise_floor": noise_floor(led),
        "B_tolerance_bands": tolerance_bands(led),
        "C_association": association(led, n_perm=args.n_perm, seed=args.seed),
        "D_not_covered": not_covered(),
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    a = res["A_noise_floor"]
    print("\n=== A. Harmonic estimator noise floor (v1 vs v2, identical settings) ===\n")
    print(_md_table(
        [[mo,
          str(v["n_paired"]),
          f"{v['median_abs_delta_thz']:.2e}",
          f"{v['p95_abs_delta_thz']:.2e}",
          f"{v['max_abs_delta_thz']:.5f}",
          str(v["call_flips"])]
         for mo, v in sorted(a["per_model"].items())],
        ["model", "n", "median |d| THz", "p95 |d|", "max |d| THz", "call flips"],
    ))
    print(f"\nclaim-bearing models ({', '.join(CLAIM_MODELS)}): "
          f"worst max |delta| = {a['claim_models_worst_max_abs_delta_thz']:.5f} THz, "
          f"{a['claim_models_call_flips']} call flips")

    b = res["B_tolerance_bands"]
    print("\n=== B. Per-model harmonic accuracy vs tolerance ===\n")
    print(_md_table(
        [[mo] + [f"{b['per_model_accuracy'][mo][f'{t:.2f}']:.3f}" for t in TOLS]
         for mo in sorted(b["per_model_accuracy"])],
        ["model"] + [f"tol {t:.2f}" for t in TOLS],
    ))
    print("\nuniquely-worst model by tolerance:")
    for k, v in b["uniquely_worst_at"].items():
        print(f"  tol {k}: {', '.join(v)}")
    print(f"\nCHGNet strictly below MatterSim at every tolerance: {b['ordering_invariant']}")
    print(f"  per tol: {b['chgnet_below_mattersim_at_every_tol']}")

    print("\n=== C. Association, system-clustered ===\n")
    print(_md_table(
        [[t,
          str(v["n_pairs"]),
          f"{v['phi']:.4f}",
          f"{v['phi_perm_p_two_sided']:.4f}",
          f"[{v['phi_cluster_bootstrap_ci95'][0]:.3f}, {v['phi_cluster_bootstrap_ci95'][1]:.3f}]",
          "yes" if v["phi_ci_excludes_zero"] else "no",
          str(v["cells"]["d_both_bad"])]
         for t, v in res["C_association"]["by_temperature"].items()],
        ["T (K)", "n", "phi", "perm p", "cluster boot CI95", "CI excl 0", "d cell"],
    ))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
