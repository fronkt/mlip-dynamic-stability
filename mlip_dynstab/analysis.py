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
                       include_borderline: bool = False, exclude_bcc: bool = False) -> pd.DataFrame:
    """HEADLINE finite-T metric: per-model false-stable rate restricted to the LOW-temperature
    regime (T <= ``t_max``). Here the single-mode SCHA is reliable, so this cleanly separates
    models that capture soft-mode instabilities from those that miss them (vs the absolute
    transition temperature, which a single-mode treatment underestimates -- see ``predicted_tstar``).
    Set ``exclude_bcc`` for the clean displacive headline: the bcc thermodynamic-Tc label is the
    wrong reference for dynamic stability (those rows belong to the SSCHA analysis, not here)."""
    d = df[(df["method"] == method) & (df["temperature_K"] <= t_max)]
    if not include_borderline:
        d = d[~d["system"].isin(borderline_systems())]
    if exclude_bcc:
        d = d[~d["system"].str.contains("bcc")]
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


def method_agreement(df: pd.DataFrame, t_max: float | None = None) -> pd.DataFrame:
    """Cross-validate the fast single-mode ``softmode`` route against gold-standard multi-mode
    ``sscha`` on the units where BOTH ran. Returns one row per (system, model, T) with each
    method's minimum effective frequency and whether they agree on the dynamic-stability SIGN
    (stable iff freq >= 0). This is the evidence that the cheap softmode screen tracks the
    expensive SSCHA result, justifying softmode as the headline method."""
    sm = df[df["method"] == "softmode"][["system", "model", "temperature_K", "min_eff_freq_thz"]]
    sc = df[df["method"] == "sscha"][["system", "model", "temperature_K", "min_eff_freq_thz"]]
    m = sm.merge(sc, on=["system", "model", "temperature_K"], suffixes=("_softmode", "_sscha"))
    if t_max is not None:
        m = m[m["temperature_K"] <= t_max]
    if m.empty:
        return m
    m["stable_softmode"] = m["min_eff_freq_thz_softmode"] >= 0
    m["stable_sscha"] = m["min_eff_freq_thz_sscha"] >= 0
    m["agree"] = m["stable_softmode"] == m["stable_sscha"]
    return m.sort_values(["system", "model", "temperature_K"]).reset_index(drop=True)


def method_agreement_summary(df: pd.DataFrame, t_max: float | None = None) -> dict:
    """Headline numbers for the softmode-vs-SSCHA cross-check: sign-agreement fraction, count of
    paired units, and the Pearson/Spearman correlation of the two minimum frequencies."""
    m = method_agreement(df, t_max=t_max)
    if m.empty:
        return {"n_paired": 0}
    a, b = m["min_eff_freq_thz_softmode"], m["min_eff_freq_thz_sscha"]
    fin = np.isfinite(a) & np.isfinite(b)
    out = {"n_paired": int(len(m)),
           "sign_agreement": round(float(m["agree"].mean()), 3),
           "n_disagree": int((~m["agree"]).sum())}
    if fin.sum() >= 3:
        out["pearson_freq"] = round(float(np.corrcoef(a[fin], b[fin])[0, 1]), 3)
        out["spearman_freq"] = round(float(pd.Series(a[fin].values).corr(
            pd.Series(b[fin].values), method="spearman")), 3)
    return out


def harmonic_tolerance_sweep(df: pd.DataFrame,
                             tols=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Referee m5: harmonic false-stable / false-unstable counts as a function of the imaginary
    tolerance (a call is 'stable' iff min_freq >= -tol), summed over the 5 models on the scored
    set. Shows that the default -0.1 THz sits in the basin where finite-displacement noise near Γ
    does not dominate (strict tol=0 floods false-unstables; loose tol inflates false-stables)."""
    h = df[(df["method"] == "harmonic") & (~df["system"].isin(borderline_systems()))]
    if h.empty:
        return pd.DataFrame()
    gt = h["gt_stable"].astype(bool)
    rows = []
    for tol in tols:
        pred = h["min_freq_thz"] >= -tol
        rows.append({"tol_THz": tol,
                     "false_stable": int((pred & ~gt).sum()),
                     "false_unstable": int((~pred & gt).sum()),
                     "n_calls": int(len(h))})
    return pd.DataFrame(rows)


def _family(systems: pd.Series) -> pd.Series:
    return np.where(systems.str.contains("bcc"), "bcc",
                    np.where(systems.str.contains("o2_"), "fluorite", "perovskite"))


def sscha_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """SSCHA reliability by structural family. As implemented (ForcePositiveDefinite init, root2
    auxiliary matrix, include_v4=False) SSCHA is the clean gold-standard for bcc martensitic
    systems but FAILS on deep displacive (ferroelectric perovskite) instabilities: the auxiliary
    matrix collapses to the cubic minimum and the v4=False free-energy Hessian echoes its positive
    curvature -> false-stable; float32 models additionally blow up numerically. This quantifies the
    numerical-blowup mode and the physical range per family. (Diagnostic: on cubic BaTiO3/mace/100K
    the harmonic soft mode is -5.6 THz but SSCHA reports +2.87 THz; include_v4=True ran >18 min on a
    single unit without finishing, so it is not viable at grid scale.)"""
    sc = df[df["method"] == "sscha"].copy()
    if sc.empty:
        return pd.DataFrame()
    sc["family"] = _family(sc["system"])
    rows = []
    for fam, g in sc.groupby("family"):
        f = g["min_eff_freq_thz"]
        rows.append({"family": fam, "n": len(g),
                     "n_numerical_blowup": int((f.abs() > 50).sum()),
                     "freq_min_THz": round(float(f.min()), 2),
                     "freq_max_THz": round(float(f.max()), 2)})
    return pd.DataFrame(rows).sort_values("family")


def displacive_recall(df: pd.DataFrame, t_max: float = 300.0) -> pd.DataFrame:
    """Headline cautionary contrast. On the ferroelectric perovskites (cubic BaTiO3/KNbO3/PbTiO3)
    at T <= t_max -- far below every Tc, so the cubic phase is DEFINITIVELY dynamically unstable --
    the fraction of model units each method correctly calls unstable. The cheap single-mode
    softmode catches the displacive instability; multi-mode SSCHA (v4=False) systematically misses
    it (false-stable). SSCHA numerical blow-ups (|freq|>50 THz) are reported separately, not
    counted as correct."""
    fe = ["batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
    rows = []
    for method in ["softmode", "sscha"]:
        d = df[(df["method"] == method) & (df["system"].isin(fe)) & (df["temperature_K"] <= t_max)]
        d = d[np.isfinite(d["min_eff_freq_thz"])]
        if d.empty:
            continue
        blow = int((d["min_eff_freq_thz"].abs() > 50).sum()) if method == "sscha" else 0
        valid = d[d["min_eff_freq_thz"].abs() <= 50] if method == "sscha" else d
        n = len(valid)
        correct = int((~valid["pred_stable"].astype(bool)).sum())
        rows.append({"method": method, "n_valid": n, "correct_unstable": correct,
                     "recall_unstable": round(correct / n, 3) if n else float("nan"),
                     "n_numerical_blowup": blow})
    return pd.DataFrame(rows)


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


def h2_paired_summary(df: pd.DataFrame, t: float = 100.0) -> dict:
    """H2 done correctly: pair the harmonic and finite-T (softmode at T=``t``) call for each
    (system, model) on the MATCHED non-bcc, non-borderline set, so harmonic and finite-T accuracy
    share the same denominator (the model-level 'inversion' seen with bcc-in-harmonic vs
    bcc-out-of-finite-T is a denominator artifact). Reports the 2x2 concordance of correctness,
    the phi correlation, an exact-binomial McNemar test on the discordant pairs, and the matched
    per-model accuracies with their rank correlation. The honest finding is that harmonic accuracy
    is a WEAK (not negative) predictor of finite-T accuracy, and that the harmonic leaders are not
    the finite-T leaders -- not a clean inversion."""
    from math import comb
    bl = borderline_systems()
    def nonbcc(d):
        return d[(~d["system"].isin(bl)) & (~d["system"].str.contains("bcc"))]
    h = nonbcc(df[df["method"] == "harmonic"])[["system", "model", "pred_stable", "gt_stable"]]
    f = nonbcc(df[(df["method"] == "softmode") & (df["temperature_K"] == t)])[["system", "model", "pred_stable", "gt_stable"]]
    m = h.rename(columns={"pred_stable": "hp", "gt_stable": "hg"}).merge(
        f.rename(columns={"pred_stable": "fp", "gt_stable": "fg"}), on=["system", "model"])
    if m.empty:
        return {}
    hok = (m["hp"] == m["hg"]); fok = (m["fp"] == m["fg"])
    a = int((hok & fok).sum()); b = int((hok & ~fok).sum())
    c = int((~hok & fok).sum()); d = int((~hok & ~fok).sum())
    phi = float(np.corrcoef(hok.astype(int), fok.astype(int))[0, 1]) if hok.nunique() > 1 and fok.nunique() > 1 else float("nan")
    nd = b + c
    mcnemar_p = min(1.0, 2 * sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd) if nd else 1.0
    hacc = {mod: (g["hp"] == g["hg"]).mean() for mod, g in m.groupby("model")}
    facc = {mod: (g["fp"] == g["fg"]).mean() for mod, g in m.groupby("model")}
    mods = sorted(hacc)
    rho = float(pd.Series([hacc[x] for x in mods]).corr(pd.Series([facc[x] for x in mods]), method="spearman"))
    return {
        "n_pairs": int(len(m)),
        "concordance": {"harm_ok_ft_ok": a, "harm_ok_ft_bad": b, "harm_bad_ft_ok": c, "harm_bad_ft_bad": d},
        "phi": round(phi, 3), "mcnemar_exact_p": round(mcnemar_p, 3),
        "matched_per_model": {mod: {"harm_acc": round(hacc[mod], 3), "ft_acc": round(facc[mod], 3)} for mod in mods},
        "spearman_matched_model_acc": round(rho, 3),
    }


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


def _auc(score, label) -> float:
    """Rank AUC of ``score`` as a predictor of the boolean ``label`` (Mann-Whitney form)."""
    score = np.asarray(score, float); label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def h3_guardrail_summary(df: pd.DataFrame, method: str = "softmode",
                         exclude_bcc: bool = True) -> dict:
    """H3 quantified. Does cross-model disagreement flag that the consensus finite-T call is
    wrong? Compares the consensus error rate on split-vote vs unanimous units, and reports the
    rank-AUC of two disagreement signals (the binary stable/unstable vote split, and the raw
    cross-model frequency standard deviation) as predictors of consensus error. bcc is excluded by
    default because its thermodynamic-Tc ground truth is the wrong reference for dynamic stability
    (see sscha_dynamic_stabilization); borderline systems are always excluded."""
    g = h3_ensemble_guardrail(df, method=method)
    if g.empty:
        return {}
    if exclude_bcc:
        g = g[~g["system"].str.contains("bcc")]
    g = g[~g["system"].isin(borderline_systems())]
    if g.empty:
        return {}
    err = ~g["consensus_correct"].astype(bool)
    split, unan = g[g["disagreement"] > 0], g[g["disagreement"] == 0]
    return {
        "n_units": int(len(g)),
        "consensus_error_rate": round(float(err.mean()), 3),
        "split_vote_error_rate": round(float((~split["consensus_correct"]).mean()), 3) if len(split) else float("nan"),
        "unanimous_error_rate": round(float((~unan["consensus_correct"]).mean()), 3) if len(unan) else float("nan"),
        "n_split": int(len(split)), "n_unanimous": int(len(unan)),
        "auc_vote_disagreement": round(_auc(g["disagreement"], err), 3),
        "auc_freq_std": round(_auc(g["freq_std_thz"], err), 3),
    }


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
        out["displacive_recall"] = displacive_recall(df).to_dict("records")
        out["h2_paired"] = h2_paired_summary(df)
        out["h3_guardrail"] = h3_guardrail_summary(df)
    if "sscha" in out["methods"]:
        tab = sscha_dynamic_stabilization(df)
        out["sscha_zr_dynamic_stabilization"] = tab.reset_index().to_dict("records")
        out["sscha_reliability"] = sscha_reliability(df).to_dict("records")
        out["bcc_method_agreement"] = method_agreement_summary(df[df["system"].str.contains("bcc")])
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2, default=str))
