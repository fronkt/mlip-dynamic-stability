"""Re-express every headline rate with counts and an interval, and re-test the clustered ones.

Answers Referee 3's item 3 (RSC Advances RA-ART-07-2026-006452) and Referee 1's item 6. Nothing
here is a new measurement: it re-reads the deposited ledger and reports the same quantities
honestly. Specifically it

  * turns every rate into k/n with a Wilson score interval;
  * states the composition of the n = 60 and n = 75 analysis sets, which the ESI never gave;
  * re-tests both guardrail AUCs by permuting whole systems rather than units, and brackets
    them with a cluster bootstrap;
  * enumerates the exact permutation distribution of the five-model Spearman rho, showing that
    no arrangement of five models can reach conventional significance;
  * recomputes the headline tables with and without ORB-v2 (Referee 3's item 5).

Revision plan S1-S4 (tasks/todo.md Phase 2) adds four blocks, each with and without ORB-v2:

  * ``h2_clustered``  -- the H2 transfer asymmetry at every ladder T, tested by system, with
    per-system counts, a leave-one-system-out table and the shared-screen-error split;
  * ``bcc_agreement`` -- screen-vs-SSCHA agreement on bcc scored on curvature sign AND on the
    stability call, with the harmonically trivial pairs separated;
  * ``orb_split_s3``  -- the four-model guardrail, family recalls, SSCHA blow-ups and failures
    per model, and Referee 3's ORB-v2 premises re-measured (named ``_s3`` because the key
    ``orb_split`` already exists and its value must not change);
  * ``sscha_high_t``  -- SSCHA false-unstables by temperature on the non-bcc systems.

Writes results/stats_hardening.json. Run from the repo root:

    python scripts/stats_hardening.py [--n-perm 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # match the convention in the sibling scripts

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

from mlip_dynstab import analysis as A  # noqa: E402
from mlip_dynstab import stats as S     # noqa: E402

LEDGER = REPO / "results" / "ledger.parquet"
OUT = REPO / "results" / "stats_hardening.json"

FE = ["batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
HALIDE = ["cspbi3_cubic", "cssnbr3_cubic", "cssni3_cubic"]
FLUORITE = ["zro2_cubic", "hfo2_cubic"]
AFD = ["srtio3_cubic"]


def _rate(mask_correct: pd.Series) -> dict:
    k = int(mask_correct.sum())
    n = int(len(mask_correct))
    return S.rate_ci(k, n).as_dict() | {"fmt": S.fmt_rate(S.rate_ci(k, n))}


def per_model_with_ci(df: pd.DataFrame, method: str, t_max: float | None = None,
                      exclude_bcc: bool = False, exclude_models: tuple[str, ...] = ()) -> dict:
    """Per-model accuracy as k/n with a Wilson interval, on the paper's scoring set."""
    d = df[df["method"] == method]
    d = d[~d["system"].isin(A.borderline_systems())]
    if t_max is not None:
        d = d[d["temperature_K"] <= t_max]
    if exclude_bcc:
        d = d[~d["system"].str.contains("bcc")]
    if exclude_models:
        d = d[~d["model"].isin(exclude_models)]
    out = {}
    for model, g in d.groupby("model"):
        correct = g["pred_stable"].astype(bool) == g["gt_stable"].astype(bool)
        c = A.confusion(g)
        fs_k, fs_n = c["FP_false_stable"], c["FP_false_stable"] + c["TN"]
        out[model] = {
            "accuracy": _rate(correct),
            "false_stable": S.rate_ci(fs_k, fs_n).as_dict() | {"fmt": S.fmt_rate(S.rate_ci(fs_k, fs_n))},
            "n_systems": int(g["system"].nunique()),
        }
    return out


def family_recall_with_ci(df: pd.DataFrame, t_max: float = 300.0) -> dict:
    """Per-family recall of the unstable class, with counts and intervals."""
    fams = {"fe_oxide": FE, "halide": HALIDE, "fluorite": FLUORITE, "afd_srtio3": AFD}
    out = {}
    for name, systems in fams.items():
        d = df[(df["method"] == "softmode") & df["system"].isin(systems)
               & (df["temperature_K"] <= t_max)]
        d = d[~d["gt_stable"].astype(bool)]          # genuinely-unstable units only
        if d.empty:
            continue
        caught = ~d["pred_stable"].astype(bool)
        out[name] = _rate(caught) | {"systems": systems}
    return out


def guardrail_clustered(df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """H3 with the clustering respected. This is Referee 3's sharpest point."""
    g = A.h3_ensemble_guardrail(df, method="softmode")
    g = g[~g["system"].str.contains("bcc")]
    g = g[~g["system"].isin(A.borderline_systems())]
    g = g.sort_values(["system", "T"]).reset_index(drop=True)

    err = (~g["consensus_correct"].astype(bool)).to_numpy()
    systems = g["system"].to_numpy()

    composition = {
        "n_units": int(len(g)),
        "n_systems": int(g["system"].nunique()),
        "systems": sorted(g["system"].unique().tolist()),
        "temperature_ladder_K": sorted(g["T"].unique().tolist()),
        "units_per_system": {s: int(k) for s, k in g["system"].value_counts().items()},
        "balanced": bool(g["system"].value_counts().nunique() == 1),
        "note": ("each unit already collapses five model votes, so the five models are NOT "
                 "an independent axis here; the clustering unit is the system"),
    }

    res = {"composition": composition,
           "consensus_error_rate": _rate(pd.Series(~err)) | {
               "reported_as": "fraction CORRECT; the manuscript quotes the error rate, 1 - this"},
           }
    split = g[g["disagreement"] > 0]
    unan = g[g["disagreement"] == 0]
    res["split_vote_error"] = _rate(~split["consensus_correct"].astype(bool))
    res["unanimous_error"] = _rate(~unan["consensus_correct"].astype(bool))

    for label, col in [("vote_disagreement", "disagreement"), ("freq_std", "freq_std_thz")]:
        score = g[col].to_numpy(float)
        entry = S.cluster_permutation_auc(score, err, systems, n_perm=n_perm, seed=seed)
        entry |= S.cluster_bootstrap_auc(score, err, systems, n_boot=n_perm, seed=seed)
        # the naive unit-level p, reported alongside purely to show the size of the error
        rng = np.random.default_rng(seed)
        naive = np.array([S.auc(score, rng.permutation(err)) for _ in range(n_perm)])
        entry["p_perm_naive_unit_level"] = round(
            float((np.sum(np.abs(naive - 0.5) >= abs(entry["auc"] - 0.5)) + 1) / (naive.size + 1)), 4)
        res[f"auc_{label}"] = entry
    return res


def h2_composition_and_spearman(df: pd.DataFrame, temps=(100.0, 300.0, 600.0, 900.0)) -> dict:
    """The n = 75 matched set: state what is in it, and bound what its rho can express."""
    bl = A.borderline_systems()

    def matched(d):
        return d[(~d["system"].isin(bl)) & (~d["system"].str.contains("bcc"))]

    h = matched(df[df["method"] == "harmonic"])[["system", "model", "pred_stable", "gt_stable"]]
    out: dict = {}
    for t in temps:
        f = matched(df[(df["method"] == "softmode") & (df["temperature_K"] == t)])[
            ["system", "model", "pred_stable", "gt_stable"]]
        m = h.rename(columns={"pred_stable": "hp", "gt_stable": "hg"}).merge(
            f.rename(columns={"pred_stable": "fp", "gt_stable": "fg"}), on=["system", "model"])
        if m.empty:
            continue
        hok = (m["hp"] == m["hg"])
        fok = (m["fp"] == m["fg"])
        hacc = {mod: float((g["hp"] == g["hg"]).mean()) for mod, g in m.groupby("model")}
        facc = {mod: float((g["fp"] == g["fg"]).mean()) for mod, g in m.groupby("model")}
        mods = sorted(hacc)
        sp = S.cluster_permutation_spearman([hacc[x] for x in mods], [facc[x] for x in mods])
        out[str(int(t))] = {
            "n_pairs": int(len(m)),
            "n_systems": int(m["system"].nunique()),
            "n_models": int(m["model"].nunique()),
            "systems": sorted(m["system"].unique().tolist()),
            "harmonic_correct": _rate(hok),
            "finite_t_correct": _rate(fok),
            "spearman_over_models": sp,
        }
    if out:
        first = next(iter(out.values()))
        out["composition_note"] = (
            f"n = {first['n_pairs']} is {first['n_systems']} systems x {first['n_models']} "
            "models. The same 15 systems carry the n = 60 guardrail set (15 x 4 temperatures), "
            "so the two analysis sets share their clustering structure. Systems whose id "
            "contains 'bcc' are dropped, which also removes the superionic agi_bcc; this is "
            "retained for comparability with the published numbers and is stated in section 3.2."
        )
        out["spearman_power_note"] = (
            "With five models there are only 5! = 120 distinct pairings, so this statistic has a "
            "resolution floor (reported per temperature as finest_attainable_p) and any rho at "
            "this n is descriptive only. Note also "
            "that the coefficient changes sign across the ladder, which is itself the argument "
            "against reading it."
        )
    return out


def screen_vs_sscha_paired(df: pd.DataFrame, systems: list[str], t_max: float = 300.0,
                           exclude_models: tuple[str, ...] = ()) -> dict:
    """Paired test of the paper's central cautionary contrast.

    The screen's 0.53 and SSCHA's 0.19 are two marginal rates, and their Wilson intervals
    overlap. That overlap is *not* the right inference: the two methods are evaluated on the
    same (system, model, T) units, so the comparison is paired and the marginal intervals
    ignore the pairing. On the units where both methods return a physical number, count the
    discordant pairs and test them exactly.
    """
    from math import comb
    d = df[(df["system"].isin(systems)) & (df["temperature_K"] <= t_max)]
    if exclude_models:
        d = d[~d["model"].isin(exclude_models)]
    keys = ["system", "model", "temperature_K"]
    s = d[d["method"] == "softmode"][keys + ["pred_stable", "min_eff_freq_thz"]]
    q = d[d["method"] == "sscha"][keys + ["pred_stable", "min_eff_freq_thz"]]
    q = q[np.isfinite(q["min_eff_freq_thz"]) & (q["min_eff_freq_thz"].abs() <= 50)]
    m = s.rename(columns={"pred_stable": "s_pred"}).merge(
        q.rename(columns={"pred_stable": "q_pred"}), on=keys, suffixes=("_s", "_q"))
    if m.empty:
        return {}
    # ground truth on this set is "unstable" throughout (T far below every Tc)
    s_ok = ~m["s_pred"].astype(bool)
    q_ok = ~m["q_pred"].astype(bool)
    b = int((s_ok & ~q_ok).sum())      # screen right, SSCHA wrong
    c = int((~s_ok & q_ok).sum())      # SSCHA right, screen wrong
    nd = b + c
    p = min(1.0, 2 * sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd) if nd else 1.0
    clustered = S.cluster_exact_paired(s_ok.to_numpy(), q_ok.to_numpy(), m["system"].to_numpy())
    return {
        "n_paired_units": int(len(m)),
        "n_systems": int(m["system"].nunique()),
        "both_correct": int((s_ok & q_ok).sum()),
        "screen_right_sscha_wrong": b,
        "sscha_right_screen_wrong": c,
        "both_wrong": int((~s_ok & ~q_ok).sum()),
        "mcnemar_exact_p_UNIT_LEVEL": round(p, 7),
        "clustered_by_system": clustered,
        "note": ("Two tests, and the unit-level one is NOT the one to quote. Pairing is the "
                 "right structure -- both methods see the same (system, model, temperature) "
                 "units, so comparing two marginal Wilson intervals would ignore it -- but an "
                 "exact McNemar over discordant units also assumes those units are "
                 "independent, and they are not: they cluster by system. That is the same "
                 "objection Referee 3 raised about the guardrail AUC, and it applies here too. "
                 "The clustered test randomises the method label over whole systems and is "
                 "exact. Its resolution floor is 2/2^k, so with five systems no arrangement "
                 "can reach p < 0.0625, and the honest report is the effect size and its "
                 "consistency across systems rather than a significance claim."),
    }


def orb_split(df: pd.DataFrame) -> dict:
    """Referee 3's item 5: every headline rate with and without ORB-v2."""
    out = {}
    for tag, excl in [("all_models", ()), ("excl_orb_v2", ("orb_v2",))]:
        out[tag] = {
            "harmonic": per_model_with_ci(df, "harmonic", exclude_models=excl),
            "finite_t": per_model_with_ci(df, "softmode", t_max=300.0, exclude_bcc=True,
                                          exclude_models=excl),
        }
        d = df if not excl else df[~df["model"].isin(excl)]
        bcc = [s for s in d["system"].unique() if "bcc" in s and s != "agi_bcc"]
        ma = A.method_agreement_summary(d[d["system"].isin(bcc)])
        out[tag]["bcc_method_agreement"] = {
            k: ma.get(k) for k in
            ("n_paired", "sign_agreement", "pearson_freq", "spearman_freq")
        }
        rec = A.displacive_recall(d)
        out[tag]["displacive_recall"] = {
            r["method"]: S.rate_ci(int(r["correct_unstable"]), int(r["n_valid"])).as_dict()
            | {"fmt": S.fmt_rate(S.rate_ci(int(r["correct_unstable"]), int(r["n_valid"]))),
               "n_numerical_blowup": int(r["n_numerical_blowup"])}
            for _, r in rec.iterrows()
        }
    out["note"] = (
        "ORB-v2 is float32 and non-conservative, which is an architecture confound rather than "
        "a model property of interest. The manuscript reports the sign-agreement split "
        "(0.78 all / 0.83 excluding) but not the rank-correlation split; both are given here. "
        "Note the ex-ORB frequency rank correlation is approximately zero, not merely smaller."
    )
    return out


# ------------------------------------------- revision plan S1-S4 (2026-09-26) ----

TEMPS = (100.0, 300.0, 600.0, 900.0)
MODEL_SETS = (("all_models", ()), ("excl_orb_v2", ("orb_v2",)))
KEYS = ["system", "model", "temperature_K"]
BCC_METALS = ["ti_bcc", "zr_bcc", "hf_bcc"]
CONTROLS = ["si_diamond", "mgo_rocksalt", "nacl_rocksalt", "cu_fcc", "c_diamond", "ceo2_cubic"]
SSCHA_GRID = REPO / "results" / "sscha_units_v1.csv"   # the 208 attempted SSCHA units
BLOWUP_THZ = 50.0                                        # |f| above this = numerical blow-up
AS_REVIEWED = "paper/submissions/rsc-advances-2026-07/manuscript-as-reviewed.md"


def _kn(k: int, n: int) -> dict:
    """k/n with a Wilson interval; an empty denominator gives nulls, never NaN (strict JSON)."""
    if int(n) == 0:
        return {"k": int(k), "n": 0, "p": None, "lo": None, "hi": None, "fmt": "0/0"}
    r = S.rate_ci(int(k), int(n))
    return r.as_dict() | {"fmt": S.fmt_rate(r)}


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar on the discordant counts (same formula as
    analysis.h2_paired_summary). Treats units as independent."""
    from math import comb
    nd = b + c
    return min(1.0, 2 * sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd) if nd else 1.0


def _drop_models(df: pd.DataFrame, excl: tuple[str, ...]) -> pd.DataFrame:
    return df[~df["model"].isin(excl)] if excl else df


def _r(x, nd: int = 3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---- S1: H2 clustered ----

def _h2_pairs(df: pd.DataFrame, t: float) -> pd.DataFrame:
    """The matched set of ``analysis.h2_paired_summary`` at one T: one row per (system, model),
    non-bcc and non-borderline, harmonic call paired with the screen's call at T."""
    bl = A.borderline_systems()

    def matched(d):
        return d[(~d["system"].isin(bl)) & (~d["system"].str.contains("bcc"))]

    cols = ["system", "model", "pred_stable", "gt_stable"]
    h = matched(df[df["method"] == "harmonic"])[cols]
    f = matched(df[(df["method"] == "softmode") & (df["temperature_K"] == t)])[cols]
    m = h.rename(columns={"pred_stable": "hp", "gt_stable": "hg"}).merge(
        f.rename(columns={"pred_stable": "fp", "gt_stable": "fg"}), on=["system", "model"])
    m["harm_ok"] = m["hp"].astype(bool) == m["hg"].astype(bool)
    m["ft_ok"] = m["fp"].astype(bool) == m["fg"].astype(bool)
    return m.sort_values(["system", "model"]).reset_index(drop=True)


def _h2_test(m: pd.DataFrame) -> dict:
    hok = m["harm_ok"].to_numpy(bool)
    fok = m["ft_ok"].to_numpy(bool)
    b = int((hok & ~fok).sum())
    c = int((~hok & fok).sum())
    cl = S.cluster_exact_paired(hok, fok, m["system"].to_numpy())
    # A system with zero net discordance adds nothing to either tail, so the attainable floor is
    # set by the systems that are NOT tied: 2 / 2^(k - tied), coarser than 2 / 2^k.
    k_inf = cl["n_clusters"] - cl["clusters_tied"]
    cl["finest_attainable_p_given_ties"] = round(2.0 / 2 ** k_inf, 5) if k_inf else 1.0
    return {
        "n_pairs": int(len(m)),
        "n_systems": int(m["system"].nunique()),
        "b_harm_right_ft_wrong": b,
        "c_harm_wrong_ft_right": c,
        "mcnemar_exact_p_UNIT_LEVEL": round(_mcnemar_exact_p(b, c), 5),
        "clustered_by_system": cl,
    }


def _shared_screen_errors(df: pd.DataFrame, t: float, min_models: int = 4) -> dict:
    """Matched-set systems the screen mis-calls at T for >= ``min_models`` of the FIVE models.
    Always computed on the five-model grid, so the same systems are removed in both model sets."""
    bl = A.borderline_systems()
    f = df[(df["method"] == "softmode") & (df["temperature_K"] == t)
           & ~df["system"].isin(bl) & ~df["system"].str.contains("bcc")]
    wrong = f["pred_stable"].astype(bool) != f["gt_stable"].astype(bool)
    per = wrong.groupby(f["system"]).agg(["sum", "size"])
    return {s: {"n_models_wrong": int(r["sum"]), "n_models": int(r["size"])}
            for s, r in per.iterrows() if r["sum"] >= min_models}


def h2_clustered(df: pd.DataFrame, temps=TEMPS) -> dict:
    """S1. The H2 transfer asymmetry, tested at the level the data are clustered at."""
    out: dict = {
        "answers": ["R1.6", "R3.1", "R3.3 (H2 McNemar clustering)", "R3.5 (ORB split)"],
        "definition": (
            "Matched set of analysis.h2_paired_summary (non-bcc, non-borderline; 15 systems x "
            "5 models, 15 x 4 without ORB-v2). b = harmonic call right and screen call at T wrong; "
            "c = the reverse. mcnemar_exact_p_UNIT_LEVEL treats the units as independent and is a "
            "companion only. clustered_by_system is stats.cluster_exact_paired(a = harmonic right, "
            "b = finite-T right, cluster = system): exact enumeration of the 2^k per-system sign "
            "flips, so clusters_favouring_a counts systems with more b than c units. A 'shared "
            "screen error' system is one the screen mis-calls at that T for >= 4 of the 5 models "
            "(the screen's own T* error rather than an MLIP-specific one); the set is defined on "
            "all five models and applied to both model sets."),
    }
    for t in temps:
        shared = _shared_screen_errors(df, t)
        entry: dict = {"shared_screen_error_systems": shared}
        for tag, excl in MODEL_SETS:
            m = _h2_pairs(_drop_models(df, excl), t)
            if m.empty:
                continue
            res = _h2_test(m)
            if not excl:     # must be the same numbers the manuscript's function returns
                ref = A.h2_paired_summary(df, t=t)
                assert (ref["concordance"]["harm_ok_ft_bad"], ref["concordance"]["harm_bad_ft_ok"]) \
                    == (res["b_harm_right_ft_wrong"], res["c_harm_wrong_ft_right"]), (t, ref)
                assert abs(ref["mcnemar_exact_p"] - res["mcnemar_exact_p_UNIT_LEVEL"]) < 5e-4
            per_sys = {}
            for s, g in m.groupby("system"):
                b = int((g["harm_ok"] & ~g["ft_ok"]).sum())
                c = int((~g["harm_ok"] & g["ft_ok"]).sum())
                per_sys[s] = {"b": b, "c": c, "net": b - c}
            res["per_system_discordance"] = per_sys
            loso = {}
            for s in sorted(m["system"].unique()):
                r = _h2_test(m[m["system"] != s])
                loso[s] = {"b": r["b_harm_right_ft_wrong"], "c": r["c_harm_wrong_ft_right"],
                           "p_unit": r["mcnemar_exact_p_UNIT_LEVEL"],
                           "p_clustered": r["clustered_by_system"]["p_exact_clustered"]}
            pcl = {s: v["p_clustered"] for s, v in loso.items()}
            res["leave_one_system_out"] = {
                "p_clustered_min": min(pcl.values()),
                "dropped_system_at_min": min(pcl, key=pcl.get),
                "p_clustered_max": max(pcl.values()),
                "drops_with_p_clustered_below_0.05": sorted(s for s, p in pcl.items() if p < 0.05),
                "by_dropped_system": loso,
            }
            disc = m[m["harm_ok"] != m["ft_ok"]]
            in_sh = disc["system"].isin(shared)
            res["discordant_units"] = [
                {"system": r.system, "model": r.model, "type": "b" if r.harm_ok else "c",
                 "shared_screen_error": bool(r.system in shared)}
                for r in disc.itertuples()]
            w = _h2_test(m[~m["system"].isin(shared)])
            res["shared_screen_error_split"] = {
                "b_in_shared_systems": int((disc["harm_ok"] & in_sh).sum()),
                "c_in_shared_systems": int((~disc["harm_ok"] & in_sh).sum()),
                "without_shared_systems": {
                    "n_pairs": w["n_pairs"], "n_systems": w["n_systems"],
                    "b": w["b_harm_right_ft_wrong"], "c": w["c_harm_wrong_ft_right"],
                    "p_unit": w["mcnemar_exact_p_UNIT_LEVEL"],
                    "p_clustered": w["clustered_by_system"]["p_exact_clustered"],
                    "finest_attainable_p_given_ties":
                        w["clustered_by_system"]["finest_attainable_p_given_ties"]},
            }
            entry[tag] = res
        out[str(int(t))] = entry
    out["note"] = (
        "The unit-level McNemar p is not quotable as evidence: the 75 (60) pairs are 15 systems "
        "x 5 (4) models and cluster by system, which ESI Table S7 already states. The clustered "
        "p is the test. The shared-screen-error split is a decomposition, not an independent "
        "test: removing systems selected on finite-T error removes b-type units by "
        "construction, so it shows WHERE the asymmetry sits (in systems where the screen's T* "
        "is wrong for nearly every model), not that it is absent elsewhere.")
    return out


# ---- S2: bcc agreement, curvature sign vs call ----

def bcc_agreement(df: pd.DataFrame) -> dict:
    """S2. Screen-vs-SSCHA agreement on the bcc metals, scored two ways."""
    ss = df[df["method"] == "sscha"]
    sscha_call_is_sign = bool((ss["pred_stable"].astype(bool)
                               == (ss["min_eff_freq_thz"] >= 0)).all())
    harm = df[(df["method"] == "harmonic") & df["system"].isin(BCC_METALS)][
        ["system", "model", "pred_stable", "min_freq_thz"]].rename(
        columns={"pred_stable": "harm_stable", "min_freq_thz": "harm_min_freq_thz"})
    out: dict = {
        "answers": ["R1.1 residue (bcc 'clean gold standard')", "R3.5"],
        "definition": {
            "set": ("analysis.method_agreement on ti/zr/hf_bcc: the (system, model, T) units where "
                    "both the screen and SSCHA ran, i.e. T = 100/300/600 K."),
            "curvature_sign": ("min_eff_freq_thz >= 0 in both methods (analysis.method_agreement "
                               "'agree'). This is the statistic the manuscript calls 'call "
                               "agreement'. For the screen it is the symmetric-point curvature, "
                               "which §2.4 says is not the call."),
            "call": "pred_stable of the softmode row == pred_stable of the sscha row.",
            "trivial": ("pairs whose harmonic layer (canonical harmonic row, tolerance -0.1 THz) "
                        "has NO instability for that (system, model): both methods then agree "
                        "without any thermal stabilisation having been tested."),
            "sscha_call_equals_curvature_sign_on_every_row": sscha_call_is_sign,
        },
    }
    for tag, excl in MODEL_SETS:
        d = _drop_models(df[df["system"].isin(BCC_METALS)], excl)
        m = A.method_agreement(d)
        sm = d[d["method"] == "softmode"][KEYS + ["pred_stable"]].rename(
            columns={"pred_stable": "call_screen"})
        sq = d[d["method"] == "sscha"][KEYS + ["pred_stable"]].rename(
            columns={"pred_stable": "call_sscha"})
        m = m.merge(sm, on=KEYS).merge(sq, on=KEYS).merge(harm, on=["system", "model"], how="left")
        m["call_agree"] = m["call_screen"].astype(bool) == m["call_sscha"].astype(bool)
        m["trivial"] = m["harm_stable"].astype(bool)
        summ = A.method_agreement_summary(d)
        assert summ["n_paired"] == len(m) and abs(summ["sign_agreement"] - m["agree"].mean()) < 5e-4

        def two(x: pd.DataFrame) -> dict:
            return {"curvature_sign": _kn(x["agree"].sum(), len(x)),
                    "call": _kn(x["call_agree"].sum(), len(x))}

        triv_pairs = sorted({f"{s}/{mo}" for s, mo in
                             m.loc[m["trivial"], ["system", "model"]].itertuples(index=False)})
        diff = m[m["agree"] != m["call_agree"]]
        out[tag] = {
            "n_paired": int(len(m)),
            "curvature_sign_agreement": _kn(m["agree"].sum(), len(m)),
            "call_agreement": _kn(m["call_agree"].sum(), len(m)),
            "trivial_split": {"trivial_pairs": triv_pairs,
                              "trivial": two(m[m["trivial"]]),
                              "non_trivial": two(m[~m["trivial"]])},
            "per_model": {mo: two(g) for mo, g in m.groupby("model")},
            "units_where_scores_differ": [
                {"system": r.system, "model": r.model, "T": float(r.temperature_K),
                 "screen_curvature_thz": _r(r.min_eff_freq_thz_softmode),
                 "sscha_freq_thz": _r(r.min_eff_freq_thz_sscha),
                 "screen_call_stable": bool(r.call_screen), "sscha_call_stable": bool(r.call_sscha),
                 "curvature_sign_agree": bool(r.agree), "call_agree": bool(r.call_agree)}
                for r in diff.itertuples()],
            "frequency_correlation_DESCRIPTIVE": {
                "spearman": summ.get("spearman_freq"), "pearson": summ.get("pearson_freq"),
                "n": int(len(m)),
                "note": ("descriptive only: pairs cluster by system and model, and no test is "
                         "attached")},
        }
    return out


# ---- S3: ORB-v2 split of everything else ----

def _tie_info(df: pd.DataFrame) -> dict:
    g = A.h3_ensemble_guardrail(df, method="softmode")
    g = g[~g["system"].str.contains("bcc") & ~g["system"].isin(A.borderline_systems())]
    tied = g[g["stable_vote_frac"] == 0.5]
    return {
        "rule": ("analysis.h3_ensemble_guardrail: consensus_stable = stable_vote_frac >= 0.5, "
                 "so a 2-2 split among four models is called STABLE"),
        "n_models_per_unit": sorted(int(x) for x in g["n_models"].unique()),
        "n_tied_units": int(len(tied)),
        "n_tied_units_in_error": int((~tied["consensus_correct"].astype(bool)).sum()),
        "tied_units": [{"system": r.system, "T": float(r.T), "gt_stable": bool(r.gt_stable),
                        "consensus_correct": bool(r.consensus_correct)}
                       for r in tied.itertuples()],
    }


def _family_recall(d: pd.DataFrame, method: str, systems: list[str],
                   t_max: float = 300.0) -> dict | None:
    """Recall of the unstable class on one family at T <= t_max, displacive_recall's rules:
    finite frequencies only; for SSCHA, blow-ups (|f| > 50 THz) leave the denominator and are
    counted separately."""
    x = d[(d["method"] == method) & d["system"].isin(systems) & (d["temperature_K"] <= t_max)]
    x = x[~x["gt_stable"].astype(bool) & np.isfinite(x["min_eff_freq_thz"])]
    if x.empty:
        return None
    blow = (x["min_eff_freq_thz"].abs() > BLOWUP_THZ) if method == "sscha" else \
        pd.Series(False, index=x.index)
    valid = x[~blow]
    return _kn((~valid["pred_stable"].astype(bool)).sum(), len(valid)) | {
        "n_numerical_blowup": int(blow.sum()),
        "systems_present": sorted(x["system"].unique().tolist())}


def _sscha_blowups_failures(df: pd.DataFrame) -> dict:
    grid = pd.read_csv(SSCHA_GRID)
    ss = df[df["method"] == "sscha"]
    extra = ss[KEYS].merge(grid, on=KEYS, how="left", indicator=True)
    assert (extra["_merge"] == "left_only").sum() == 0, "SSCHA rows outside the attempted grid"
    mg = grid.merge(ss[KEYS + ["min_eff_freq_thz"]], on=KEYS, how="left", indicator=True)
    per_model = {}
    for mo, g in mg.groupby("model"):
        failed = g[g["_merge"] == "left_only"]
        ret = g[g["_merge"] == "both"]
        blow = ret[ret["min_eff_freq_thz"].abs() > BLOWUP_THZ]
        per_model[mo] = {
            "n_grid": int(len(g)), "n_returned": int(len(ret)),
            "n_failed": int(len(failed)),
            "failed_units": [f"{r.system}@{int(r.temperature_K)}K" for r in failed.itertuples()],
            "n_blowup": int(len(blow)),
            "blowup_units": [{"system": r.system, "T": float(r.temperature_K),
                              "min_eff_freq_thz": _r(r.min_eff_freq_thz, 1)}
                             for r in blow.itertuples()],
        }
    totals = {}
    for tag, excl in MODEL_SETS:
        pm = [v for mo, v in per_model.items() if mo not in excl]
        totals[tag] = {k: int(sum(v[k] for v in pm))
                       for k in ("n_grid", "n_returned", "n_failed", "n_blowup")}
    return {
        "definition": (
            f"blow-up: a returned SSCHA unit with |min_eff_freq_thz| > {BLOWUP_THZ:g} THz "
            "(analysis.sscha_reliability / displacive_recall; manuscript §3.3, ESI Table S3). "
            "failed: a unit of the attempted grid (results/sscha_units_v1.csv, 208 units) with "
            "no canonical ledger row -- it died at a cellconstructor symmetry or ensemble "
            "assertion (manuscript §3.3; ESI §S2.3)."),
        "per_model": per_model,
        "totals": totals,
    }


def orb_split_s3(df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """S3. Everything Referee 3's item 5 asks to see with and without ORB-v2 that the existing
    ``orb_split`` block does not already carry."""
    out: dict = {"answers": ["R3.5", "R2.2 (guardrail robustness)"]}

    guard = {}
    for tag, excl in MODEL_SETS:
        d = _drop_models(df, excl)
        gc = guardrail_clustered(d, n_perm, seed)
        comp = gc.pop("composition")
        gc["consensus_error"] = _kn(gc["consensus_error_rate"]["n"] - gc["consensus_error_rate"]["k"],
                                    gc["consensus_error_rate"]["n"])
        gc.pop("consensus_error_rate")      # the old key held the fraction CORRECT; not repeated
        guard[tag] = {"n_units": comp["n_units"], "n_systems": comp["n_systems"],
                      "balanced": comp["balanced"], **gc,
                      "h3_guardrail_summary": A.h3_guardrail_summary(d, method="softmode"),
                      "ties": _tie_info(d)}
    out["guardrail"] = guard | {"note": (
        f"guardrail_clustered on each model set: clustered permutation p and cluster-bootstrap "
        f"CI with n = {n_perm}, seed {seed}; auc_freq_std is item (ii). The all_models entry "
        "reproduces the h3_guardrail block.")}

    fams = {"fe_oxide": FE, "halide": HALIDE, "fluorite": FLUORITE, "afd_srtio3": AFD}
    rec = {}
    for tag, excl in MODEL_SETS:
        d = _drop_models(df, excl)
        scr = {f: _family_recall(d, "softmode", s) for f, s in fams.items()}
        ssc = {f: _family_recall(d, "sscha", s) for f, s in fams.items()}
        dr = A.displacive_recall(d).set_index("method")      # FE must agree with the paper's function
        for meth, v in (("softmode", scr["fe_oxide"]), ("sscha", ssc["fe_oxide"])):
            assert (v["k"], v["n"]) == (int(dr.loc[meth, "correct_unstable"]),
                                        int(dr.loc[meth, "n_valid"])), (tag, meth)
        ctl = d[(d["method"] == "softmode") & d["system"].isin(CONTROLS)]
        rec[tag] = {
            "screen": scr,
            "sscha": ssc,
            "controls_screen_false_unstable": _kn((~ctl["pred_stable"].astype(bool)).sum(),
                                                  len(ctl)) | {"temperatures": "all four"},
            "sscha_reliability_by_family": A.sscha_reliability(d).to_dict("records"),
        }
    out["family_recall"] = rec | {"note": (
        "Recall of the unstable class at T <= 300 K on genuinely unstable units (same rules as "
        "analysis.displacive_recall, which covers FE only and is asserted equal here). "
        "analysis.sscha_reliability gives n, blow-ups and the frequency range per family but no "
        "recall. SSCHA was run on one halide (cssni3_cubic) and on no control; the controls' "
        "entry is the screen's false-unstable count over all four temperatures.")}

    out["sscha_blowups_and_failures"] = _sscha_blowups_failures(df)

    # (v) the ORB-v2 premises Referee 3 quoted, re-measured
    h = df[(df["method"] == "harmonic") & (df["model"] == "orb_v2")
           & (df["system"] == "mgo_rocksalt")].iloc[0]
    fs = per_model_with_ci(df, "softmode", t_max=300.0, exclude_bcc=True)["orb_v2"]["false_stable"]
    v = df[(df["method"] == "softmode") & (df["model"] == "orb_v2")
           & df["system"].isin(BCC_METALS)]
    bcc_tab = {s: {str(int(t)): _r(f) for t, f in zip(g["temperature_K"], g["min_eff_freq_thz"])}
               for s, g in v.sort_values("temperature_K").groupby("system")}
    paired_t = sorted(set(df[(df["method"] == "sscha") & df["system"].isin(BCC_METALS)]
                          ["temperature_K"]) & set(TEMPS))     # 100/300/600 K on bcc

    def _argmin(x: pd.DataFrame) -> dict:
        r = x.loc[x["min_eff_freq_thz"].idxmin()]
        return {"min_eff_freq_thz": _r(r["min_eff_freq_thz"]), "system": r["system"],
                "T": float(r["temperature_K"])}

    bf = out["sscha_blowups_and_failures"]
    out["orb_premises"] = {
        "source_of_old_values": AS_REVIEWED,
        "mgo_harmonic_min_freq_thz": {
            "now": _r(h["min_freq_thz"]), "as_reviewed": -2.8,
            "pred_stable": bool(h["pred_stable"]), "gt_stable": bool(h["gt_stable"])},
        "orb_finite_t_false_stable": {"now": fs, "as_reviewed": 0.562,
                                      "set": "softmode, T <= 300 K, non-bcc, non-borderline"},
        "orb_softmode_bcc_min_eff_freq_thz": {
            "by_system_and_T": bcc_tab,
            "sscha_paired_T": [float(t) for t in paired_t],
            "min_on_sscha_paired_T": _argmin(v[v["temperature_K"].isin(paired_t)]),
            "min_full_ladder": _argmin(v),
            "hf_bcc_min": _r(v[v["system"] == "hf_bcc"]["min_eff_freq_thz"].min()),
            "as_reviewed": "Ti/Hf near -35 THz"},
        "sscha_blowups": {
            "orb_v2": bf["per_model"]["orb_v2"]["n_blowup"],
            "all_models": bf["totals"]["all_models"]["n_blowup"],
            "models_with_blowups": sorted(mo for mo, x in bf["per_model"].items() if x["n_blowup"]),
            "as_reviewed": "six numerical blow-ups, concentrated in the float32 ORB-v2 runs"},
    }
    out["harmonic_tolerance_sweep"] = {
        tag: A.harmonic_tolerance_sweep(_drop_models(df, excl)).to_dict("records")
        for tag, excl in MODEL_SETS}
    return out


# ---- S4: SSCHA high-temperature false-unstables ----

def sscha_high_t(df: pd.DataFrame) -> dict:
    """S4. The anti-stabilisation with temperature that the manuscript does not discuss."""
    grid = pd.read_csv(SSCHA_GRID)
    harm = df[df["method"] == "harmonic"][["system", "model", "min_freq_thz"]]
    scr = df[df["method"] == "softmode"][KEYS + ["ft_decide_harm_thz", "ft_decide_q"]]
    out: dict = {
        "answers": ["R1.2 (OOD hypothesis)"],
        "definition": (
            "Non-bcc SSCHA rows. false-unstable = SSCHA calls unstable (min_eff_freq_thz < 0) "
            "and the label is stable; blow-ups (|f| > 50 THz) are INCLUDED in the counts and "
            "tallied alongside. harmonic_min_freq_thz is the model's harmonic-layer minimum "
            "(T-independent); screen_deciding_mode_harm_thz is the harmonic frequency of the "
            "screen's deciding mode (ft_decide_harm_thz) on the softmode row at the same T."),
    }
    for tag, excl in MODEL_SETS:
        ss = _drop_models(df[(df["method"] == "sscha") & ~df["system"].str.contains("bcc")], excl)
        by_t = {}
        for t in TEMPS:
            x = ss[ss["temperature_K"] == t]
            gs = x[x["gt_stable"].astype(bool)]
            fu = gs[~gs["pred_stable"].astype(bool)]
            by_t[str(int(t))] = _kn(len(fu), len(gs)) | {
                "n_units_at_T": int(len(x)),
                "n_blowup_among_false_unstable": int((fu["min_eff_freq_thz"].abs() > BLOWUP_THZ).sum()),
                "by_system": {s: int(k) for s, k in fu["system"].value_counts().sort_index().items()}}

        st = ss[(ss["system"] == "srtio3_cubic") & ss["gt_stable"].astype(bool)]
        st = st[KEYS + ["transition_T_K", "pred_stable", "min_eff_freq_thz"]].merge(
            harm, on=["system", "model"], how="left").merge(scr, on=KEYS, how="left")
        st = st.sort_values(["temperature_K", "model"])
        tc = float(st["transition_T_K"].iloc[0])
        g_st = _drop_models(grid[(grid["system"] == "srtio3_cubic") & (grid["temperature_K"] > tc)], excl)
        k_fu = int((~st["pred_stable"].astype(bool)).sum())
        srtio3 = {
            "transition_T_K": tc,
            "n_grid_above_tc": int(len(g_st)),
            "n_returned": int(len(st)),
            "false_unstable": _kn(k_fu, len(st)),
            "sscha_freq_range_thz": [_r(st["min_eff_freq_thz"].min()), _r(st["min_eff_freq_thz"].max())],
            "units": [{"model": r.model, "T": float(r.temperature_K),
                       "sscha_min_eff_freq_thz": _r(r.min_eff_freq_thz),
                       "false_unstable": not bool(r.pred_stable),
                       "blow_up": bool(abs(r.min_eff_freq_thz) > BLOWUP_THZ),
                       "harmonic_min_freq_thz": _r(r.min_freq_thz),
                       "screen_deciding_mode_q": r.ft_decide_q,
                       "screen_deciding_mode_harm_thz": _r(r.ft_decide_harm_thz)}
                      for r in st.itertuples()],
        }

        piv = ss.pivot_table(index=["system", "model"], columns="temperature_K",
                             values="min_eff_freq_thz")
        gtp = ss.pivot_table(index=["system", "model"], columns="temperature_K",
                             values="gt_stable", aggfunc="first")
        s100 = piv[piv[100.0] >= 0]
        units = []
        for (s, mo), row in s100.iterrows():
            neg_t = [t for t in TEMPS if t in row.index and np.isfinite(row[t]) and row[t] < 0]
            hi_neg = [t for t in neg_t if t >= 600.0]
            hm = harm[(harm["system"] == s) & (harm["model"] == mo)]["min_freq_thz"]
            sd = scr[(scr["system"] == s) & (scr["model"] == mo)].set_index("temperature_K")
            units.append({
                "system": s, "model": mo,
                "sscha_min_eff_freq_thz": {str(int(t)): _r(row[t]) for t in TEMPS if t in row.index},
                "first_negative_T": neg_t[0] if neg_t else None,
                "negative_by_600_900K": bool(hi_neg),
                "high_T_negative_is_false_unstable": bool(any(bool(gtp.loc[(s, mo), t])
                                                               for t in hi_neg)),
                "harmonic_min_freq_thz": _r(hm.iloc[0]) if len(hm) else None,
                "screen_deciding_mode_harm_thz": {str(int(t)): _r(sd.loc[t, "ft_decide_harm_thz"])
                                                  for t in TEMPS if t in sd.index},
            })
        n_neg = sum(u["negative_by_600_900K"] for u in units)
        out[tag] = {
            "false_unstable_by_T": by_t,
            "srtio3_above_tc": srtio3,
            "stable_at_100K_turn_negative": _kn(n_neg, len(units)) | {
                "n_stable_at_100K": len(units),
                "n_negative_by_600_900K": int(n_neg),
                "n_of_those_false_unstable_at_high_T": int(sum(
                    u["high_T_negative_is_false_unstable"] for u in units if u["negative_by_600_900K"])),
                "units": units},
        }
    out["note"] = (
        "Growth of the SSCHA instability with temperature on the same PES is the signature "
        "Referee 1's out-of-distribution hypothesis (item 2) predicts. For the fluorites the "
        "label is 'unstable' at every T, so their high-T negatives are label-correct but the "
        "trend with T is physically the wrong direction; high_T_negative_is_false_unstable "
        "separates the two cases.")
    return out


def criterion_blindness(df: pd.DataFrame) -> dict:
    """The SSCHA false-stables seen through the screen's two observables on the SAME PES.

    The screen reports two things per unit from one E(Q) map: a stability CALL (does any
    displaced centroid have lower single-mode SCHA free energy than Q0 = 0, a global
    comparison) and a CURVATURE (the free-energy curvature at the symmetric point).

    CAUTION, and the reason the curvature counts below are not evidence: for a single mode with
    an even potential the symmetric-point curvature equals the trial stiffness M*Omega^2, so it
    is positive by construction whenever a bound Gaussian exists; negative values are numerical
    (width-solver fallbacks, fit artefacts). See scripts/curvature_identity_check.py. The
    informative count is `screen_call_unstable`: on how many SSCHA false-stables the screen's
    free-energy comparison finds a lower displaced minimum on the same MLIP energies.
    Descriptive counts; no test.
    """
    keep = ["system", "model", "temperature_K"]
    sm = df[df["method"] == "softmode"][keep + ["min_eff_freq_thz", "pred_stable", "gt_stable"]]
    ss = df[df["method"] == "sscha"][keep + ["min_eff_freq_thz", "pred_stable"]]
    out: dict = {"answers": ["R1.1b (global vs local)", "R1.2", "R1.4 ('methodological trap')"]}
    for tag, excl in MODEL_SETS:
        m = _drop_models(sm.merge(ss, on=keep, suffixes=("_scr", "_ss")), excl)
        m = m[~m["system"].str.contains("bcc")]
        fs = m[m["pred_stable_ss"].astype(bool) & ~m["gt_stable"].astype(bool)]
        curv_pos = fs["min_eff_freq_thz_scr"] > 0
        call_unst = ~fs["pred_stable_scr"].astype(bool)
        lo = m[m["temperature_K"] <= 300.0]
        out[tag] = {
            "n_sscha_false_stable_nonbcc": int(len(fs)),
            "screen_curvature_positive": _kn(int(curv_pos.sum()), len(fs)),
            "screen_call_unstable": _kn(int(call_unst.sum()), len(fs)),
            "curvature_positive_and_call_unstable": _kn(int((curv_pos & call_unst).sum()), len(fs)),
            "by_system": {s: {"n": int(len(g)),
                              "screen_curv_pos": int((g["min_eff_freq_thz_scr"] > 0).sum()),
                              "screen_call_unstable": int((~g["pred_stable_scr"].astype(bool)).sum())}
                          for s, g in fs.groupby("system")},
            "paired_nonbcc_T_le_300": {
                "n": int(len(lo)),
                "sscha_sign_eq_screen_curvature_sign": int(
                    ((lo["min_eff_freq_thz_ss"] >= 0) == (lo["min_eff_freq_thz_scr"] >= 0)).sum()),
                "sscha_call_eq_screen_call": int(
                    (lo["pred_stable_ss"].astype(bool) == lo["pred_stable_scr"].astype(bool)).sum()),
            },
        }
    out["note"] = (
        "Counts over non-bcc (system, model, T) units with both a softmode and an sscha row. "
        "Curvature > 0 means the screen's symmetric-point free-energy curvature is positive.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = A.load_canonical(LEDGER)

    res = {
        "_generated_by": "scripts/stats_hardening.py",
        "_answers": ["RSC Advances R3 item 3 (counts, intervals, clustering)",
                     "RSC Advances R3 item 5 (ORB-v2 split)",
                     "RSC Advances R1 item 6 (statistical language)"],
        "_n_perm": args.n_perm,
        "_seed": args.seed,
        "harmonic_per_model": per_model_with_ci(df, "harmonic"),
        "finite_t_per_model": per_model_with_ci(df, "softmode", t_max=300.0, exclude_bcc=True),
        "family_recall": family_recall_with_ci(df),
        "h3_guardrail": guardrail_clustered(df, args.n_perm, args.seed),
        "h2_matched_set": h2_composition_and_spearman(df),
        # Three system sets on purpose. The FE-perovskite subset is the one the manuscript
        # headlines, and it is also the one that loses significance when ORB-v2 is removed --
        # so the claim has to rest on the combined displacive set, where the fluorites (which
        # are numerically clean and ORB-independent) carry it. Reporting only the favourable
        # slice here would be the exact failure Referee 3's item 5 is guarding against.
        "screen_vs_sscha_paired": {
            f"{label}__{tag}": screen_vs_sscha_paired(df, systems, exclude_models=excl)
            for label, systems in [("fe_oxide", FE), ("fluorite", FLUORITE),
                                   ("displacive_combined", FE + FLUORITE)]
            for tag, excl in [("all_models", ()), ("excl_orb_v2", ("orb_v2",))]
        },
        "orb_split": orb_split(df),
        # revision plan S1-S4 (tasks/todo.md Phase 2)
        "h2_clustered": h2_clustered(df),
        "bcc_agreement": bcc_agreement(df),
        "orb_split_s3": orb_split_s3(df, args.n_perm, args.seed),
        "sscha_high_t": sscha_high_t(df),
        "criterion_blindness": criterion_blindness(df),
    }

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")

    g = res["h3_guardrail"]
    print("\n-- guardrail set --")
    print(f"  {g['composition']['n_units']} units = "
          f"{g['composition']['n_systems']} systems x "
          f"{len(g['composition']['temperature_ladder_K'])} temperatures, "
          f"balanced={g['composition']['balanced']}")
    for k in ("auc_vote_disagreement", "auc_freq_std"):
        e = g[k]
        print(f"  {k:24s} AUC {e['auc']:.3f}  clustered p {e['p_perm_clustered']:.3f}  "
              f"CI [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]  (naive unit p {e['p_perm_naive_unit_level']:.3f})")
    print("\n-- harmonic accuracy --")
    for m, v in sorted(res["harmonic_per_model"].items()):
        print(f"  {m:12s} {v['accuracy']['fmt']}")
    print("\n-- finite-T accuracy (T<=300 K, non-bcc) --")
    for m, v in sorted(res["finite_t_per_model"].items()):
        print(f"  {m:12s} {v['accuracy']['fmt']}")
    print("\n-- screen vs SSCHA, paired (the central contrast) --")
    print(f"   {'set / model subset':30s} {'units':>5s} {'disc.':>7s} {'unit-p':>9s} "
          f"{'sys':>5s} {'clust-p':>8s} {'floor':>7s}")
    for tag, v in res["screen_vs_sscha_paired"].items():
        if not v:
            continue
        c = v["clustered_by_system"]
        print(f"   {tag:30s} {v['n_paired_units']:5d} "
              f"{v['screen_right_sscha_wrong']:3d}/{v['sscha_right_screen_wrong']:<3d} "
              f"{v['mcnemar_exact_p_UNIT_LEVEL']:9.2g} "
              f"{c['clusters_favouring_a']}/{c['n_clusters']:<3d} "
              f"{c['p_exact_clustered']:8.4f} {c['finest_attainable_p']:7.4f}")
    print("   The unit-level p treats correlated units as independent. The clustered column is")
    print("   the one to quote; its resolution floor is 2/2^k, shown alongside.")

    print("\n-- five-model Spearman --")
    for t, v in res["h2_matched_set"].items():
        if isinstance(v, dict) and "spearman_over_models" in v:
            sp = v["spearman_over_models"]
            print(f"  T={t:>3s} K  rho {sp['rho']:+.3f}  exact p {sp['p_perm']:.4f}  "
                  f"(finest attainable {sp['finest_attainable_p']:.4f})")

    print("\n-- S1 H2 transfer asymmetry, clustered by system --")
    print(f"   {'T':>4s} {'set':12s} {'b/c':>6s} {'unit-p':>8s} {'sys a/b':>8s} {'clust-p':>8s} "
          f"{'LOSO p range':>16s} {'ex-shared b/c, clust-p':>24s}")
    for t in ("100", "300", "600", "900"):
        for tag in ("all_models", "excl_orb_v2"):
            e = res["h2_clustered"][t][tag]
            c = e["clustered_by_system"]
            lo = e["leave_one_system_out"]
            w = e["shared_screen_error_split"]["without_shared_systems"]
            print(f"   {t:>4s} {tag:12s} {e['b_harm_right_ft_wrong']:>3d}/{e['c_harm_wrong_ft_right']:<2d} "
                  f"{e['mcnemar_exact_p_UNIT_LEVEL']:8.4f} "
                  f"{c['clusters_favouring_a']:>4d}/{c['clusters_favouring_b']:<3d} "
                  f"{c['p_exact_clustered']:8.4f} "
                  f"[{lo['p_clustered_min']:.3f}, {lo['p_clustered_max']:.3f}]   "
                  f"{w['b']:>3d}/{w['c']:<2d} {w['p_clustered']:.3f}")

    print("\n-- S2 bcc screen vs SSCHA --")
    for tag in ("all_models", "excl_orb_v2"):
        e = res["bcc_agreement"][tag]
        ts = e["trivial_split"]
        print(f"   {tag:12s} curvature sign {e['curvature_sign_agreement']['fmt']}; "
              f"call {e['call_agreement']['fmt']}")
        print(f"   {'':12s} non-trivial: sign {ts['non_trivial']['curvature_sign']['fmt']}; "
              f"call {ts['non_trivial']['call']['fmt']}; "
              f"rho (descriptive) {e['frequency_correlation_DESCRIPTIVE']['spearman']:+.3f}")

    print("\n-- S3 guardrail with / without ORB-v2 --")
    for tag in ("all_models", "excl_orb_v2"):
        e = res["orb_split_s3"]["guardrail"][tag]
        a = e["auc_vote_disagreement"]
        print(f"   {tag:12s} split err {e['split_vote_error']['fmt']}  unan err "
              f"{e['unanimous_error']['fmt']}  AUC {a['auc']:.3f} p {a['p_perm_clustered']:.4f} "
              f"CI [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]  ties {e['ties']['n_tied_units']}")
    bf = res["orb_split_s3"]["sscha_blowups_and_failures"]
    print("   SSCHA blow-ups / failed per model: " + ", ".join(
        f"{m} {v['n_blowup']}/{v['n_failed']}" for m, v in bf["per_model"].items()))

    print("\n-- S4 SSCHA non-bcc false-unstable by T --")
    for tag in ("all_models", "excl_orb_v2"):
        e = res["sscha_high_t"][tag]
        print(f"   {tag:12s} " + "  ".join(f"{t} K {v['k']}/{v['n']}"
                                          for t, v in e["false_unstable_by_T"].items())
              + f"   SrTiO3 above Tc {e['srtio3_above_tc']['false_unstable']['fmt']}"
              + f"   100 K-stable -> negative {e['stable_at_100K_turn_negative']['fmt']}")


if __name__ == "__main__":
    main()
