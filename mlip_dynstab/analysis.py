"""Analysis of the results ledger: confusion matrices, false-stable rates, and the H2/H3
tests. All figures regenerate from results/ledger.parquet, so the analysis is reproducible
and independent of when/where the units were computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .ledger import load as load_ledger


# --------------------------------------------------------------- confusion ----

def confusion(df: pd.DataFrame) -> dict[str, int]:
    """TP/TN/FP/FN where 'positive' = predicted dynamically stable.

    false-stable (FP) = model says stable, ground truth unstable — the screening-dangerous
    error this benchmark headlines.
    """
    pred = df["pred_stable"].astype(bool)
    gt = df["gt_stable"].astype(bool)
    tp = int(((pred) & (gt)).sum())
    tn = int(((~pred) & (~gt)).sum())
    fp = int(((pred) & (~gt)).sum())     # false-stable
    fn = int(((~pred) & (gt)).sum())     # false-unstable
    return {"TP": tp, "TN": tn, "FP_false_stable": fp, "FN_false_unstable": fn, "n": len(df)}


def rates(c: dict[str, int]) -> dict[str, float]:
    fp, tn = c["FP_false_stable"], c["TN"]
    fn, tp = c["FN_false_unstable"], c["TP"]
    n = max(c["n"], 1)
    return {
        "accuracy": (tp + tn) / n,
        "false_stable_rate": fp / max(fp + tn, 1),   # P(say stable | truly unstable)
        "false_unstable_rate": fn / max(fn + tp, 1),
        "balanced_acc": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
    }


def borderline_systems() -> set[str]:
    """System ids whose ground-truth label is genuinely ambiguous (e.g. KTaO3, an incipient
    ferroelectric whose DFT soft mode is marginally soft). These are excluded from headline
    scoring so models are not penalised on an ill-defined label; the registry is the single
    source of truth for the flag."""
    try:
        from .systems import load_specs
        return {s.id for s in load_specs() if s.borderline}
    except Exception:
        return set()


def per_model_table(df: pd.DataFrame, method: Optional[str] = None,
                    include_borderline: bool = False) -> pd.DataFrame:
    """Headline table: per-model confusion + rates for a given method (harmonic/tdep/...).

    Borderline-labelled systems are dropped by default (see ``borderline_systems``)."""
    if method:
        df = df[df["method"] == method]
    if not include_borderline:
        df = df[~df["system"].isin(borderline_systems())]
    rows = []
    for model, g in df.groupby("model"):
        c = confusion(g)
        r = rates(c)
        rows.append({"model": model, "method": method or "all", **c, **{k: round(v, 3) for k, v in r.items()}})
    return pd.DataFrame(rows).sort_values("false_stable_rate")


# ---------------------------------------------- finite-T headline (low-T) ----

def low_t_false_stable(df: pd.DataFrame, method: str = "softmode", t_max: float = 300.0,
                       include_borderline: bool = False) -> pd.DataFrame:
    """HEADLINE finite-T metric: per-model false-stable rate restricted to the LOW-temperature
    regime (T <= ``t_max``). Here the single-mode SCHA is reliable, so this cleanly separates
    models that capture soft-mode instabilities from those that miss them (vs the absolute
    transition temperature, which a single-mode treatment underestimates -- see ``predicted_tstar``)."""
    d = df[(df["method"] == method) & (df["temperature_K"] <= t_max)]
    if not include_borderline:
        d = d[~d["system"].isin(borderline_systems())]
    rows = []
    for model, g in d.groupby("model"):
        c = confusion(g); r = rates(c)
        rows.append({"model": model, "t_max": t_max, **c, **{k: round(v, 3) for k, v in r.items()}})
    return pd.DataFrame(rows).sort_values("false_stable_rate")


def sscha_dynamic_stabilization(df: pd.DataFrame, system: str = "zr_bcc") -> pd.DataFrame:
    """Multi-mode SSCHA dynamic-stabilization curve: the free-energy (physical) Hessian minimum
    frequency vs T for each model on ``system`` (default bcc-Zr). >0 => the high-symmetry phase
    is a dynamically (meta)stable free-energy minimum at that T. The cross-model spread at fixed
    T is the finding (it tracks how deep each model's harmonic instability is). Note: martensitic
    bcc->hcp means this dynamic-stabilization T differs from the thermodynamic transition_T_K."""
    d = df[(df["method"] == "sscha") & (df["system"] == system)]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index="temperature_K", columns="model",
                         values="min_eff_freq_thz").round(3)


def predicted_tstar(df: pd.DataFrame, method: str = "softmode") -> pd.DataFrame:
    """Per (system, model) predicted stabilisation temperature T* = lowest ladder T at which the
    cubic phase is called stable, compared to the experimental transition_T_K. A single-mode
    SCHA systematically UNDER-estimates T* for entropy-stabilised transitions, so this is a
    qualitative/ordering check, not a quantitative Tc prediction."""
    d = df[df["method"] == method]
    rows = []
    for (system, model), g in d.groupby(["system", "model"]):
        g = g.sort_values("temperature_K")
        tc = g["transition_T_K"].iloc[0]
        stable_T = g[g["pred_stable"].astype(bool)]["temperature_K"]
        tstar = float(stable_T.min()) if len(stable_T) else float("inf")
        rows.append({"system": system, "model": model, "transition_T_K": tc,
                     "T_star_pred": tstar, "n_T": len(g)})
    return pd.DataFrame(rows).sort_values(["system", "model"])


# ------------------------------------------------- H2: harmonic vs finite-T ----

def h2_harmonic_predictiveness(df: pd.DataFrame) -> pd.DataFrame:
    """For each (system, model), pair the harmonic call with the finite-T call and ask:
    does harmonic correctness predict finite-T correctness? Returns a contingency table and
    the phi correlation. H2 expects LOW correlation (harmonic accuracy not predictive)."""
    harm = df[df["method"] == "harmonic"][["system", "model", "pred_stable", "gt_stable"]]
    harm = harm.rename(columns={"pred_stable": "harm_pred", "gt_stable": "harm_gt"})
    ft = df[df["method"].isin(["softmode", "hiphive", "tdep", "sscha"])][["system", "model", "pred_stable", "gt_stable", "temperature_K"]]
    ft = ft.rename(columns={"pred_stable": "ft_pred", "gt_stable": "ft_gt"})
    m = harm.merge(ft, on=["system", "model"], how="inner")
    if m.empty:
        return m
    m["harm_correct"] = m["harm_pred"] == m["harm_gt"]
    m["ft_correct"] = m["ft_pred"] == m["ft_gt"]
    # phi coefficient between harm_correct and ft_correct
    a = m["harm_correct"].astype(int)
    b = m["ft_correct"].astype(int)
    phi = float(np.corrcoef(a, b)[0, 1]) if a.nunique() > 1 and b.nunique() > 1 else float("nan")
    m.attrs["phi_harm_vs_ft_correct"] = phi
    return m


# ------------------------------------------------- H3: ensemble disagreement ----

def h3_ensemble_guardrail(df: pd.DataFrame, method: str = "tdep") -> pd.DataFrame:
    """Per (system, T): does cross-model disagreement (std of min frequency, and stable-vote
    split) predict that the consensus call is wrong? H3 expects high disagreement to coincide
    with errors. Frequency column differs by layer."""
    d = df[df["method"] == method].copy()
    freq_col = "min_eff_freq_thz" if method in ("softmode", "tdep", "sscha") else "min_freq_thz"
    if d.empty or freq_col not in d:
        return pd.DataFrame()
    rows = []
    for (system, T), g in d.groupby(["system", "temperature_K"]):
        votes = g["pred_stable"].astype(bool)
        gt = bool(g["gt_stable"].iloc[0])
        consensus = votes.mean() >= 0.5
        rows.append({
            "system": system, "T": T, "n_models": len(g),
            "freq_std_thz": float(np.nanstd(g[freq_col])),
            "stable_vote_frac": float(votes.mean()),
            "disagreement": float(min(votes.mean(), 1 - votes.mean())),  # 0=unanimous .5=split
            "consensus_stable": bool(consensus), "gt_stable": gt,
            "consensus_correct": bool(consensus == gt),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- report ----

def summary(ledger_path=None) -> dict:
    df = load_ledger(ledger_path) if ledger_path else load_ledger()
    out = {"n_rows": len(df)}
    if df.empty:
        return out
    out["methods"] = sorted(df["method"].unique())
    out["models"] = sorted(df["model"].unique())
    bl = sorted(borderline_systems())
    out["borderline_excluded"] = bl  # ambiguous-label systems dropped from headline rates
    for meth in out["methods"]:
        out[f"per_model_{meth}"] = per_model_table(df, meth).to_dict("records")
    if "softmode" in out["methods"]:
        out["headline_low_t_false_stable"] = low_t_false_stable(df).to_dict("records")
        out["predicted_tstar"] = predicted_tstar(df).to_dict("records")
    if "sscha" in out["methods"]:
        tab = sscha_dynamic_stabilization(df)
        out["sscha_zr_dynamic_stabilization"] = tab.reset_index().to_dict("records")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2, default=str))
