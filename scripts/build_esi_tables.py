"""Render the ESI's supplementary tables from the deposited ledger.

Referee 3 (RSC Advances RA-ART-07-2026-006452), item 4: the ESI promised Tables S1-S4 but
carried only the names of the functions that generate them, so "none of the results are
checkable". The functions existed and ran; the tables were simply never rendered into the
document. This script renders them, and adds the tables the rest of the revision needs:
S4-S11 and S13 from the ledger and results/stats_hardening.json, S14 (screen sensitivity,
Referee 1.5) from results/screen_sensitivity.json, and S15-S17 (H2 clustered ladder, bcc
agreement and criterion blindness, SSCHA high-T false-unstables and failures) from
results/stats_hardening.json, and S18 (the pre-registered force-level ensemble test,
Referee 2.2) from results/revision/force_spread/summary.json, and S19-S20 (PBE along the
screen's coordinates, Referee 1.1; MLIP-versus-PBE errors on SSCHA configurations, Referee 1.2)
from the C3a/C3b tables that ``scripts/dft_reference.py analyze`` writes under
results/revision/dft/, and S21 (the SSCHA seed study with the production recipe, Referee 1.4:
sample sizes, gradient history, stopping criteria, Hessian uncertainty, seeds beyond bcc-Zr; and
the bcc-Zr cell-size re-measurement) from the unit JSONs that scripts/sscha_seed_study.py writes
under results/revision/sscha_seeds/, and S22 (the converged-recipe SSCHA grid beside the production
claims, Referee 1.4) from results/revision/grid_compare.json, which ``scripts/grid_compare.py``
writes from the grid's summary.csv; S22 is left out, with a note on stderr, until that file exists.
Tables S1-S3 and S12 are hand-written in their own sections.

It rewrites the block between the sentinels

    <!-- BEGIN GENERATED TABLES -->
    <!-- END GENERATED TABLES -->

in paper/supplementary.md, so the ESI stays regenerable rather than hand-maintained.

Run from the repo root:

    python scripts/build_esi_tables.py [--check]

``--check`` regenerates into memory and fails if the file on disk is out of date, which is what
a pre-submission verification pass should call.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from mlip_dynstab import analysis as A   # noqa: E402
from mlip_dynstab import stats as S      # noqa: E402
from mlip_dynstab.systems import load_specs  # noqa: E402

LEDGER = REPO / "results" / "ledger.parquet"
STATS = REPO / "results" / "stats_hardening.json"
SENS = REPO / "results" / "screen_sensitivity.json"
CURV = REPO / "results" / "curvature_identity_check.json"
FSPREAD = REPO / "results" / "revision" / "force_spread" / "summary.json"
C3A_UNITS = REPO / "results" / "revision" / "dft" / "c3a_unit_calls.csv"
C3A_PATHS = REPO / "results" / "revision" / "dft" / "c3a_paths.csv"
C3B_UNITS = REPO / "results" / "revision" / "dft" / "c3b_units.csv"
SEED_DIR = REPO / "results" / "revision" / "sscha_seeds"
GRID_SUMMARY = REPO / "results" / "revision" / "sscha_converged_grid" / "summary.csv"
GRID_COMPARE = REPO / "results" / "revision" / "grid_compare.json"
ESI =REPO / "paper" / "supplementary.md"

BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"

PRETTY = {"chgnet": "CHGNet", "mace_mp0": "MACE-MP-0", "mattersim": "MatterSim",
          "orb_v2": "ORB-v2", "sevennet0": "SevenNet-0"}
FE = ["batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
FLUORITE = ["zro2_cubic", "hfo2_cubic"]
BCC = ["ti_bcc", "zr_bcc", "hf_bcc"]
ELEMENT = {"ti_bcc": "Ti", "zr_bcc": "Zr", "hf_bcc": "Hf"}
CONTROL_SYSTEMS = [s.id for s in load_specs() if s.klass == "control"]
MINUS = "−"


def _sci(x: float) -> str:
    """1.2e-07 -> '1.2 × 10⁻⁷' for prose."""
    m, ex = f"{x:.0e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(ex)).translate(sup)}"


def md(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ci(k: int, n: int) -> str:
    r = S.rate_ci(int(k), int(n))
    return f"{r.k}/{r.n} = {r.p:.3f} [{r.lo:.3f}, {r.hi:.3f}]" if n else "--"


def ci2(r: dict) -> str:
    """Compact k/n [lo, hi] from a stored rate dict (two-decimal Wilson bounds)."""
    if not r.get("n"):
        return f"{r.get('k', 0)}/0"
    return f"{r['k']}/{r['n']} [{r['lo']:.2f}, {r['hi']:.2f}]"


def pfmt(p: float) -> str:
    """A p-value at the precision the text quotes: three decimals at or above 0.01, one
    significant figure below."""
    return f"{p:.3f}" if p >= 0.01 else f"{p:.1g}"


def signed(x: float, nd: int = 2) -> str:
    """Signed number with a typographic minus, for captions and prose."""
    return f"{x:+.{nd}f}".replace("-", MINUS)


# ------------------------------------------------------------------ tables ----

def table_s4(df: pd.DataFrame) -> str:
    t = A.per_model_table(df, "harmonic")
    rows = []
    for _, r in t.sort_values("model").iterrows():
        n = int(r["n"])
        correct = int(r["TP"]) + int(r["TN"])
        rows.append([
            PRETTY.get(r["model"], r["model"]), int(r["TP"]), int(r["TN"]),
            int(r["FP_false_stable"]), int(r["FN_false_unstable"]),
            ci(correct, n),
            ci(int(r["FP_false_stable"]), int(r["FP_false_stable"]) + int(r["TN"])),
        ])
    return (
        "**Table S4** Per-model harmonic confusion matrices over the 19 scored systems "
        "(KTaO₃ excluded as borderline). Positive = predicted dynamically stable, so a "
        "false-stable is the screening-dangerous error. Rates carry Wilson score intervals. "
        "`analysis.per_model_table(df, \"harmonic\")`.\n\n"
        + md(rows, ["Model", "TP", "TN", "False-stable", "False-unstable",
                    "Accuracy [95% CI]", "False-stable rate [95% CI]"])
    )


def table_s5(df: pd.DataFrame) -> str:
    rows = []
    for t_max, tag in [(300.0, "T ≤ 300 K"), (900.0, "full ladder")]:
        for excl_bcc in (True, False):
            t = A.low_t_false_stable(df, t_max=t_max, exclude_bcc=excl_bcc)
            for _, r in t.sort_values("model").iterrows():
                fs, tn = int(r["FP_false_stable"]), int(r["TN"])
                rows.append([
                    tag, "excluded" if excl_bcc else "included",
                    PRETTY.get(r["model"], r["model"]),
                    ci(fs, fs + tn),
                    ci(int(r["TP"]) + tn, int(r["n"])),
                ])
    return (
        "**Table S5** Per-model finite-temperature (soft-mode screen) false-stable and accuracy "
        "rates, at the T ≤ 300 K restriction used for the headline table and over the full "
        "100/300/600/900 K ladder, each with and without the bcc metals. bcc is excluded from "
        "the headline because its thermodynamic-T_c label is the wrong reference for dynamic "
        "stability (§3.3). `analysis.low_t_false_stable(df, t_max=..., exclude_bcc=...)`.\n\n"
        + md(rows, ["Temperature set", "bcc", "Model", "False-stable rate [95% CI]",
                    "Accuracy [95% CI]"])
    )


def table_s6(df: pd.DataFrame) -> str:
    t = A.predicted_tstar(df)
    rows = []
    for _, r in t.sort_values(["system", "model"]).iterrows():
        tc = r.get("transition_T_K")
        ts = r.get("T_star_pred")
        rows.append([
            r["system"], PRETTY.get(r["model"], r["model"]),
            "--" if pd.isna(tc) else f"{tc:.0f}",
            "never stable" if (ts is None or not np.isfinite(ts)) else f"{ts:.0f}",
        ])

    # The bcc rows do not share one direction of error, so the caption counts them rather than
    # characterising them. T* = 100 K is the bottom of the ladder (a censored value), and on the
    # (system, model) pairs with no harmonic instability it is reached trivially.
    bb = t[t["system"].isin(BCC)].copy()
    ts = pd.to_numeric(bb["T_star_pred"], errors="coerce")
    fin = np.isfinite(ts)
    n_below = int((fin & (ts < bb["transition_T_K"])).sum())
    n_above = int((fin & (ts >= bb["transition_T_K"])).sum())
    never = bb[~fin]
    by_model: dict[str, list[str]] = {}
    for _, r in never.sort_values(["model", "system"]).iterrows():
        by_model.setdefault(PRETTY.get(r["model"], r["model"]), []).append(ELEMENT[r["system"]])
    never_txt = "; ".join(f"{m} on {', '.join(v)}" for m, v in by_model.items())
    harm = df[(df["method"] == "harmonic") & df["system"].isin(BCC)]
    trivial = harm[harm["pred_stable"].astype(bool)]
    triv_txt = " and ".join(sorted({PRETTY.get(m, m) for m in trivial["model"]}))
    triv_sys = " and ".join(ELEMENT[s] for s in BCC if s in set(trivial["system"]))
    above_txt = (f", above it in {n_above}," if n_above else ",")
    # A bcc row that is never stable on the ladder is censored, not an over-estimate, whenever
    # the ladder stops below every bcc transition.
    ladder_max = float(df[df["method"] == "softmode"]["temperature_K"].max())
    tc_min = float(bb["transition_T_K"].min())
    censor_txt = (
        f"; the ladder ends at {ladder_max:.0f} K, below every bcc transition ({tc_min:.0f} K "
        "and above), so those rows are censored and do not say on which side of T_c the "
        "screen's T* lies" if ladder_max < tc_min else "")
    bcc_txt = (
        f"For the bcc metals, of the {len(bb)} (system, model) rows the screen places T* below "
        f"the experimental transition in {n_below}{above_txt} and never calls the bcc phase "
        f"stable on the ladder in the other {len(never)} ({never_txt}){censor_txt}. Of the rows "
        f"below the transition, the {len(trivial)} for {triv_txt} on {triv_sys} have no "
        "harmonic instability at all, so their T* = 100 K is the bottom of the ladder rather "
        "than a measured stabilisation. "
    )
    return (
        "**Table S6** Predicted stabilisation temperature T* (the lowest ladder temperature at "
        "which the high-symmetry phase is called stable) against the experimental transition "
        "temperature, per system and model. T* is a **diagnostic, not a prediction**: a "
        "single-mode treatment is not expected to reproduce an absolute T_c, and the screen "
        "fails to order PbTiO₃ against the other perovskite anchors (§3.2). "
        + bcc_txt
        + "A T* of 100 K means only that the phase is called stable at the lowest ladder "
        "temperature. `analysis.predicted_tstar(df)`.\n\n"
        + md(rows, ["System", "Model", "Experimental T_c (K)", "Predicted T* (K)"])
    )


def table_s7(df: pd.DataFrame) -> str:
    g = A.h3_ensemble_guardrail(df, method="softmode")
    g = g[~g["system"].str.contains("bcc")]
    g = g[~g["system"].isin(A.borderline_systems())]
    rows = []
    for _, r in g.sort_values(["system", "T"]).iterrows():
        rows.append([
            r["system"], f"{r['T']:.0f}", int(r["n_models"]),
            f"{r['freq_std_thz']:.3f}", f"{r['stable_vote_frac']:.2f}",
            "split" if r["disagreement"] > 0 else "unanimous",
            "stable" if r["consensus_stable"] else "unstable",
            "stable" if r["gt_stable"] else "unstable",
            "yes" if r["consensus_correct"] else "**no**",
        ])
    return (
        "**Table S8** The ensemble-disagreement guardrail set in full: every (system, "
        "temperature) unit behind the H3 analysis, with the cross-model frequency spread, the "
        "stable-vote fraction, and whether the majority consensus was correct. This is the "
        "n = 60 set. `analysis.h3_ensemble_guardrail(df, \"softmode\")`.\n\n"
        + md(rows, ["System", "T (K)", "n models", "Freq. std (THz)", "Stable-vote frac.",
                    "Vote", "Consensus", "Label", "Consensus correct"])
    )


def table_s8_composition(st: dict) -> str:
    g = st["h3_guardrail"]["composition"]
    h = st["h2_matched_set"]["100"]
    rows = [
        ["n = 60 (H3 guardrail, §3.4)", g["n_units"], g["n_systems"],
         "4 (100/300/600/900 K)", "collapsed into each unit",
         "yes" if g["balanced"] else "no"],
        ["n = 75 (H2 matched set, §3.2)", h["n_pairs"], h["n_systems"],
         "1 per analysis (each T analysed separately)", f"{h['n_models']} (an explicit axis)",
         "yes"],
    ]
    syslist = ", ".join(f"`{s}`" for s in g["systems"])
    return (
        "**Table S7** Composition of the two analysis sets. Both rest on the **same 15 "
        "systems**, so they share their clustering "
        "structure: the 60 guardrail units are 15 systems × 4 temperatures, and the 75 matched "
        "pairs are 15 systems × 5 models. In the guardrail set the five model votes are already "
        "collapsed into each unit, so the models are not an independent axis there and the "
        "clustering unit is the system. Systems whose identifier contains `bcc` are dropped, "
        "which also removes the superionic `agi_bcc` (§3.2), so each set has 15 systems "
        "rather than 16.\n\n"
        + md(rows, ["Analysis set", "n units", "n systems", "Temperature axis", "Model axis",
                    "Balanced"])
        + f"\n\nThe 15 systems are: {syslist}."
    )


def table_s9_orb(df: pd.DataFrame, st: dict, sens: dict) -> str:
    """Every pooled rate split by ORB-v2 (Referee 3, item 5).

    Per-model rates (Tables S4, S5) cannot change when another model is removed, so only the
    pooled quantities are split here. Values come from stats_hardening.json (orb_split_s3,
    bcc_agreement) and, for the pooled finite-T accuracy, from the production row of
    screen_sensitivity.json. The MgO value is read from the ledger because the JSON stores it
    pre-rounded to three decimals (-1.065), which would round again to the wrong second decimal.
    """
    mgo = float(df[(df["method"] == "harmonic") & (df["model"] == "orb_v2")
                   & (df["system"] == "mgo_rocksalt")]["min_freq_thz"].iloc[0])
    s3 = st["orb_split_s3"]
    ba = st["bcc_agreement"]
    tags = ("all_models", "excl_orb_v2")
    prod = next(s for s in sens["settings"] if s["setting"]["key"] == "production")
    sens_tag = {"all_models": "all_models", "excl_orb_v2": "exORB"}
    rows: list[list[str]] = []

    def row(label: str, f) -> None:
        rows.append([label] + [f(t) for t in tags])

    # harmonic layer: the tolerance sweep, pooled over models
    sweep = {t: {r["tol_THz"]: r for r in s3["harmonic_tolerance_sweep"][t]} for t in tags}
    for tol in sorted(sweep["all_models"]):
        row(f"Harmonic, pooled false-stable / false-unstable at tolerance {tol:g} THz",
            lambda t, tol=tol: (f"{sweep[t][tol]['false_stable']} / "
                                f"{sweep[t][tol]['false_unstable']} of {sweep[t][tol]['n_calls']}"))
    # finite-T screen
    row("Screen, pooled scored accuracy, T ≤ 300 K",
        lambda t: prod[sens_tag[t]]["accuracy"]["fmt"])
    row("Screen, pooled false-stable rate, T ≤ 300 K",
        lambda t: prod[sens_tag[t]]["false_stable"]["fmt"])
    fam_label = {"fe_oxide": "ferroelectric oxides", "halide": "halide perovskites",
                 "fluorite": "fluorites", "afd_srtio3": "SrTiO₃"}
    for meth, mlabel in (("screen", "Screen"), ("sscha", "SSCHA")):
        for fam, flabel in fam_label.items():
            if meth == "sscha" and fam == "halide":
                flabel = "halide perovskites (SSCHA ran on CsSnI₃ only)"
            row(f"{mlabel} recall of the unstable class, {flabel}, T ≤ 300 K",
                lambda t, meth=meth, fam=fam: s3["family_recall"][t][meth][fam]["fmt"])
        if meth == "screen":
            row("Screen false-unstable on the six controls, all four temperatures",
                lambda t: s3["family_recall"][t]["controls_screen_false_unstable"]["fmt"])
    bf = s3["sscha_blowups_and_failures"]["totals"]
    row(r"SSCHA numerical blow-ups (\|f\| > 50 THz), of returned units",
        lambda t: f"{bf[t]['n_blowup']} of {bf[t]['n_returned']}")
    row("SSCHA failed units (no number returned), of the attempted grid",
        lambda t: f"{bf[t]['n_failed']} of {bf[t]['n_grid']}")
    # The curvature-sign agreement is not tabulated: for a single even mode the screen's
    # symmetric-point curvature equals its self-consistent trial stiffness, so it is positive by
    # construction and its bcc negatives are numerical (scripts/curvature_identity_check.py).
    row("bcc screen-vs-SSCHA stability-call agreement",
        lambda t: ba[t]["call_agreement"]["fmt"])
    row("bcc frequency Spearman ρ (descriptive, no test)",
        lambda t: f"{ba[t]['frequency_correlation_DESCRIPTIVE']['spearman']:+.3f}".replace("-", MINUS))
    # guardrail
    g = s3["guardrail"]
    row("Guardrail: consensus error on split-vote units",
        lambda t: g[t]["split_vote_error"]["fmt"])
    row("Guardrail: consensus error on unanimous units",
        lambda t: g[t]["unanimous_error"]["fmt"])

    def auc(t: str, key: str) -> str:
        a = g[t][key]
        return (f"{a['auc']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}], "
                f"clustered p = {a['p_perm_clustered']:.4f}")

    row("Guardrail: vote-split AUC [cluster-bootstrap 95% CI], clustered permutation p",
        lambda t: auc(t, "auc_vote_disagreement"))
    row("Guardrail: frequency-spread AUC [cluster-bootstrap 95% CI], clustered permutation p",
        lambda t: auc(t, "auc_freq_std"))
    row("Guardrail: 2–2 tied units (broken to 'stable'), of which wrong",
        lambda t: f"{g[t]['ties']['n_tied_units']}, {g[t]['ties']['n_tied_units_in_error']} wrong")

    pr = s3["orb_premises"]
    bccmin = pr["orb_softmode_bcc_min_eff_freq_thz"]
    per = s3["sscha_blowups_and_failures"]["per_model"]
    n_orb_fail, n_fail = per["orb_v2"]["n_failed"], bf["all_models"]["n_failed"]
    orb_fail_sys = sorted({u.split("@")[0] for u in per["orb_v2"]["failed_units"]})
    ga, gx = g["all_models"]["auc_vote_disagreement"], g["excl_orb_v2"]["auc_vote_disagreement"]
    ties = g["excl_orb_v2"]["ties"]
    # stats.cluster_bootstrap_auc drops draws whose AUC is undefined (one outcome class only)
    nb = sorted({g[t][k]["n_boot"] for t in tags for k in ("auc_vote_disagreement", "auc_freq_std")})
    n_boot_txt = f"{nb[0]:,}" if len(nb) == 1 else f"{nb[0]:,} to {nb[-1]:,}"
    caption = (
        "**Table S9** Every pooled rate that ORB-v2 could plausibly drive, with and without it. "
        "ORB-v2 is the only one of the five models whose forces are predicted directly rather "
        "than as the gradient of an energy (non-conservative). As run, CHGNet, SevenNet-0, "
        "MatterSim and ORB-v2 all return float32 forces and only MACE-MP-0 returns float64, so "
        "precision does not single ORB-v2 out, and the design cannot separate its architecture "
        "from anything else about it. For reference: ORB-v2 is the only model that calls MgO "
        f"harmonically unstable ({signed(mgo)} THz); its most "
        "negative bcc screen curvatures are "
        f"{signed(pr['orb_softmode_bcc_min_eff_freq_thz']['min_on_sscha_paired_T']['min_eff_freq_thz'], 1)} THz "
        f"(Ti, {pr['orb_softmode_bcc_min_eff_freq_thz']['min_on_sscha_paired_T']['T']:.0f} K) and "
        f"{signed(bccmin['min_full_ladder']['min_eff_freq_thz'], 1)} THz "
        f"(Ti, {bccmin['min_full_ladder']['T']:.0f} K), both on fitted polynomials whose quadratic "
        "term is positive (fit artefacts of the kind described in §S1.4), with Hf at "
        f"{signed(bccmin['hf_bcc_min'])} THz, the softest commensurate harmonic value of a model "
        "with no screened imaginary mode there; "
        f"and it accounts for {pr['sscha_blowups']['orb_v2']} of the {pr['sscha_blowups']['all_models']} "
        f"SSCHA blow-ups and {n_orb_fail} of the {n_fail} failed SSCHA units "
        f"(all on {', '.join(orb_fail_sys)}). Per-model rates (Tables S4 and S5) cannot change when "
        "another model is removed, so only pooled quantities are split here. The guardrail is "
        "where removing ORB-v2 changes the reading: without it the vote-split AUC falls from "
        f"{ga['auc']:.3f} to {gx['auc']:.3f} "
        f"and its cluster-bootstrap interval spans 0.5, so the guardrail is suggestive and not "
        "robust to removing ORB-v2. The frequency spread is the cross-model standard deviation "
        "of the screen's effective frequency, a potential-energy-surface proxy that is only "
        "partly force-derived (the mode patterns come from force constants; E(Q) is energy-only); "
        "it is not a force-level ensemble uncertainty; the force-level spread, pre-registered and "
        "given with and without ORB-v2, is in Table S18. "
        "Four-model consensus uses the same rule as five-model "
        "consensus (stable when at least half the votes are stable), so a 2–2 split is called "
        f"stable; this decides {ties['n_tied_units']} units, {ties['n_tied_units_in_error']} of them "
        "wrong. Removing ORB-v2 leaves the bcc stability-call agreement at "
        f"{ba['excl_orb_v2']['call_agreement']['p']:.2f}. Permutation p-values and "
        f"bootstrap intervals resample whole systems ({ga['n_perm']:,} permutations and "
        f"{ga['n_perm']:,} bootstrap draws, seed 0; the {n_boot_txt} bootstrap draws that "
        "contain both outcomes are used), so each p carries Monte Carlo error. The H2 "
        "ladder without ORB-v2 is in Table S15, the paired screen-versus-SSCHA "
        "tests in Table S10, the bcc agreement by model in Table S16, and the SSCHA "
        "high-temperature false-unstables, blow-ups and failures by model in Table S17. "
        "The SSCHA rows here are the production recipe, which did not converge (§S2.1); the "
        "converged-recipe values, with and without ORB-v2, are in Table S22. "
        "`scripts/stats_hardening.py` (`orb_split_s3`, `bcc_agreement`); pooled finite-T "
        "accuracy from `scripts/screen_sensitivity.py`.\n\n"
    )
    return caption + md(rows, ["Quantity", "All five models", "Excluding ORB-v2"])


def table_s10_paired(st: dict) -> str:
    rows = []
    entries = [("production", key, v) for key, v in st["screen_vs_sscha_paired"].items()]
    conv_txt = ""
    if GRID_COMPARE.exists():
        gcj = json.loads(GRID_COMPARE.read_text(encoding="utf-8"))
        cv5 = gcj["variants"]["converged_only"]["converged_matched"].get("v") or {}
        for tag, sets in cv5.items():
            for setname, v in sets.items():
                entries.append(("converged", f"{setname}__{tag}", v))
        cc = (cv5.get("all_models") or {}).get("displacive_combined")
        if cc:
            cnets = ", ".join(("0" if x == 0 else f"{x:+d}").replace("-", MINUS)
                              for x in cc["clustered_by_system"]["per_cluster_net"])
            conv_txt = (
                " The *converged* rows repeat the test with the converged-recipe SSCHA values of "
                "Table S22 on the units where that recipe converged (`results/revision/"
                "grid_compare.json`, block v): there "
                f"{cc['clustered_by_system']['clusters_favouring_a']} of "
                f"{cc['clustered_by_system']['n_clusters']} systems favour the screen (per-system "
                f"net discordances {cnets}), the fluorites show no discordance at all, and the "
                f"clustered p is {cc['clustered_by_system']['p_exact_clustered']:g}.")
    for recipe, key, v in entries:
        if not v:
            continue
        setname, tag = key.rsplit("__", 1)
        c = v["clustered_by_system"]
        p_unit = v["mcnemar_exact_p_UNIT_LEVEL"]
        rows.append([
            recipe, setname.replace("_", " "), tag.replace("_", " "),
            v["n_paired_units"],
            f'{v["screen_right_sscha_wrong"]}/{v["sscha_right_screen_wrong"]}',
            f"{p_unit:.2g}" if p_unit >= 1e-6 else f"{p_unit:.0e}",
            f'{c["clusters_favouring_a"]}/{c["n_clusters"]}',
            f'{c["p_exact_clustered"]:.4f}',
            f'{c["finest_attainable_p"]:.4f}',
        ])
    comb = st["screen_vs_sscha_paired"]["displacive_combined__all_models"]["clustered_by_system"]
    nets = ", ".join(f"{x:+d}".replace("-", MINUS) for x in comb["per_cluster_net"])
    return (
        "**Table S10** The central screen-versus-SSCHA contrast, tested as the paired "
        "comparison it is. Both methods see the same (system, model, temperature) units, so "
        "comparing their two marginal Wilson intervals would ignore the pairing. Two p-values "
        "are given and **the unit-level one should not be quoted**: an exact McNemar over "
        "discordant units also assumes those units are independent, and they cluster by system. "
        "The clustered column randomises the method label over whole systems and is exact at "
        "this size. Its resolution floor, 2/2^k, is given alongside, because with five "
        "displacive systems no arrangement of the data can reach p < 0.0625 and with two "
        "fluorite systems none can go below 0.5. The reportable content of this table is the "
        f"size and the consistency of the effect ({comb['clusters_favouring_a']} of "
        f"{comb['n_clusters']} systems favour the screen, with per-system net discordances "
        f"{nets}, in the order BaTiO₃, KNbO₃, PbTiO₃, ZrO₂, HfO₂ where all five are present) "
        "rather than a significance claim. The *production* rows use the production-recipe SSCHA, "
        "which did not converge (§S2.1)." + conv_txt + " Why SSCHA loses these units is "
        "examined separately, on the same MLIP energies, in Table S16 (lower part), Table S22 and "
        "§3.3. `scripts/stats_hardening.py`, `scripts/grid_compare.py`.\n\n"
        + md(rows, ["SSCHA recipe", "System set", "Model set", "n paired", "Discordant (screen/SSCHA)",
                    "Unit-level p (do not quote)", "Systems favouring screen",
                    "Clustered p", "Floor"])
    )


def table_s11_sscha_diag(df: pd.DataFrame) -> str:
    """SSCHA numerical-quality diagnostics.

    Deliberately *not* a table of the sampling configuration. Those settings are identical for
    all 201 units (256 configurations per population, a cap of 8 populations, a 512-configuration
    dedicated Hessian ensemble, 2560 samples in total), so tabulating them per family and model
    would produce fifteen identical rows and look like evidence while carrying none. They are
    stated once, in prose, in the caption.

    What the ledger *does* retain per unit is the six lowest free-energy-Hessian frequencies.
    Care is needed reading them. The three translational acoustic modes are exactly zero for an
    exact calculation, but they are only *among the six lowest* when fewer than six other modes
    lie below them. On a deeply unstable spectrum they are not in the recorded window at all, so
    "the three smallest of the recorded six" is not a way to find them, and a residual computed
    that way would measure nothing. The two quantities below are the ones the recorded window
    can actually support.
    """
    TINY = 1e-3  # THz; an acoustic zero recorded to double precision is many orders below this
    d = df[df["method"] == "sscha"].copy()
    d = d[d["ft_lowest6_thz"].notna()]
    d["family"] = np.where(d["system"].str.contains("bcc"), "bcc",
                  np.where(d["system"].isin(FLUORITE), "fluorite", "perovskite"))

    def n_zeros(v) -> int:
        return int((np.abs(np.asarray(v, float)) < TINY).sum())

    def zero_residual(v) -> float:
        """Largest magnitude among the modes that are recognisably acoustic zeros; nan if the
        expected three are not present in the recorded window."""
        a = np.abs(np.asarray(v, float))
        z = a[a < TINY]
        return float(z.max()) if z.size == 3 else float("nan")

    d["n_zeros"] = d["ft_lowest6_thz"].map(n_zeros)
    d["resid"] = d["ft_lowest6_thz"].map(zero_residual)
    rows = []
    sw: dict[str, dict[str, tuple[int, int]]] = {}
    for (fam, model), g in d.groupby(["family", "model"]):
        r = g["resid"].dropna()
        n_ok = int((g["n_zeros"] == 3).sum())
        n_swamped = int((g["n_zeros"] == 0).sum())
        sw.setdefault(fam, {})[model] = (n_swamped, len(g))
        rows.append([
            fam, PRETTY.get(model, model), len(g),
            f"{n_ok}/{len(g)}",
            f"{r.max():.1e}" if len(r) else "--",
            n_swamped,
            f"{g['wall_s'].min():.0f}–{g['wall_s'].max():.0f}",
        ])

    # swamping fractions, read from the rows above rather than typed
    def frac_txt(m: str, fam: str) -> str:
        k, n = sw[fam][m]
        return f"{k}/{n} = {k / n:.2f} for {PRETTY.get(m, m)}"

    pv = sw["perovskite"]
    lo_m = min(pv, key=lambda m: pv[m][0] / pv[m][1])
    hi_m = max(pv, key=lambda m: pv[m][0] / pv[m][1])
    every = "present for every architecture" if all(k > 0 for k, _ in pv.values()) else "absent for some architectures"
    fl = sw["fluorite"]
    fl_nz = sorted((m for m in fl if fl[m][0] > 0), key=lambda m: -fl[m][0] / fl[m][1])
    fl_zero = [m for m in fl if fl[m][0] == 0]
    zero_n = {fl[m][1] for m in fl_zero}
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    fl_txt = (" and ".join(f"{fl[m][0]}/{fl[m][1]} for {PRETTY.get(m, m)}" for m in fl_nz)
              + (f" against 0/{zero_n.pop()} for the other {words.get(len(fl_zero), len(fl_zero))}"
                 if fl_zero and len(zero_n) == 1 else ""))
    swamp_txt = (
        f"on the perovskites it runs from {frac_txt(lo_m, 'perovskite')} to "
        f"{frac_txt(hi_m, 'perovskite')}, so it is {every} but is not uniform across them. "
        + (f"On the fluorites it is **not** architecture-neutral, being {fl_txt}."
           if fl_nz and fl_zero else "")
    )
    return (
        "**Table S11** SSCHA numerical quality, aggregated per family and model (not per unit). "
        f"The SSCHA settings are identical for all {len(d)} returned units and are therefore "
        "stated once here rather than tabulated. The starting dynamical matrix is built from "
        "phonopy full force constants at a 0.03 Å finite displacement, made positive definite "
        "(`ForcePositiveDefinite`) and symmetrised. The auxiliary dynamical matrix is relaxed in "
        "the root2 representation with `min_step_dyn = 0.5`; the convergence threshold is "
        "`meaningful_factor = 1e-4`. The minimiser's steps are capped at `max_ka = 20`, a cap "
        "that python-sscha 1.6.1 applies to the step count accumulated over all populations, "
        "not to each population (§S2.1); populations hold 256 configurations and at most 8 are "
        "drawn (2048 configurations). "
        "The free-energy Hessian is then evaluated at bubble level (`include_v4 = False`) on a "
        "dedicated 512-configuration ensemble at the final auxiliary matrix. The harness did "
        "not record how many populations each unit used or whether it met the convergence "
        "threshold before the population cap, so 2560 configurations is the configured maximum "
        "per unit, not a count. The SSCHA here inherits the MLIP potential-energy surface; "
        "nothing in this table tests that surface.\n\n"
        "Two quantities are tabulated from the six lowest recorded Hessian frequencies. "
        "*Acoustic zeros resolved* counts units in which all three translational zeros "
        f"(|ω| < {TINY:g} THz) appear within that window, and the residual column gives the "
        "largest of their magnitudes over those units. This is a check that the Hessian was "
        "symmetrised correctly; it is not a bound on the stochastic noise of the soft mode "
        "(§S2.4 gives the only seed spread measured). *Swamped* counts the opposite case: units "
        "with **no** "
        "recorded mode near zero, meaning at least six modes lie below the acoustic branches. "
        "Swamping is not itself an error: a deeply unstable phase genuinely has many imaginary "
        "modes, and the reported minimum frequency excludes the acoustic branches from the full "
        "spectrum rather than from this window. It does separate the families sharply. Read "
        "swamping as a fraction rather than a count, because the per-model denominators differ: "
        + swamp_txt + "\n\n"
        "**What the production harness did not retain:** the per-iteration free-energy gradient "
        "history, and a per-unit uncertainty on the Hessian eigenvalues. The only uncertainty "
        "probe in the production data is the independent-seed study of §S2.4 on "
        "bcc-Zr/MACE-MP-0 at 100 K, where the harmonic layer finds no bcc instability. The "
        "revision re-ran four units that do carry an instability, four seeds each, with both "
        "recorded (Table S21): no seed converged, the seed spread of the lowest Hessian "
        "frequency is 0.007 to 0.075 THz on BaTiO₃, ZrO₂ and bcc Zr and 2.0 THz on SrTiO₃ at "
        "600 K, and every seed gives the same call. The converged-recipe grid records all of "
        "these for every unit it ran (Table S22).\n\n"
        + md(rows, ["Family", "Model", "n units", "Acoustic zeros resolved",
                    "Max zero residual (THz)", "Swamped", "Wall time (s)"])
    )


def table_s13_disp_sweep(raw: pd.DataFrame) -> str:
    """Displacement-amplitude sensitivity of the harmonic layer.

    Takes the RAW ledger, not the canonical view: sweep rows live under the method name
    `harmonic_dispsweep` precisely so they cannot reach any harmonic rate, and the production
    0.01 A arm is the existing `harmonic` rows.
    """
    sw = raw[raw["method"] == "harmonic_dispsweep"]
    if sw.empty:
        return ""
    prod = A.canonical(raw)
    prod = prod[prod["method"] == "harmonic"][
        ["system", "model", "min_freq_thz", "pred_stable", "gt_stable"]]
    m = sw.merge(prod, on=["system", "model"], suffixes=("", "_prod"))
    m["delta"] = m["min_freq_thz"] - m["min_freq_thz_prod"]
    m["flip"] = m["pred_stable"].astype(bool) != m["pred_stable_prod"].astype(bool)
    bl = A.borderline_systems()

    rows = []
    for (model, d), g in m.groupby(["model", "disp_ang"]):
        sc = g[~g["system"].isin(bl)]
        k = int((sc["pred_stable"].astype(bool) == sc["gt_stable_prod"].astype(bool)).sum())
        rows.append([PRETTY.get(model, model), f"{d:g}", len(g),
                     f"{g['delta'].abs().median():.4f}", f"{g['delta'].abs().max():.4f}",
                     int(g["flip"].sum()), ci(k, len(sc))])
    # the production arm, for reference
    for model, g in prod[prod["model"].isin(m["model"].unique())].groupby("model"):
        sc = g[~g["system"].isin(bl)]
        k = int((sc["pred_stable"].astype(bool) == sc["gt_stable"].astype(bool)).sum())
        rows.append([PRETTY.get(model, model), "0.01 (production)", len(g), "--", "--", 0,
                     ci(k, len(sc))])
    rows.sort(key=lambda r: (r[0], r[1]))

    flips = m[m["flip"]][["system", "model", "disp_ang", "min_freq_thz_prod", "min_freq_thz"]]

    def flip_lines(model: str) -> str:
        f = flips[flips["model"] == model]
        return "; ".join(
            f"{r.system} at {r.disp_ang:g} Å ({r.min_freq_thz_prod:+.3f} → {r.min_freq_thz:+.3f} THz)"
            for r in f.itertuples())

    def acc_range(model: str) -> str:
        ks = []
        for _, g in m[m["model"] == model].groupby("disp_ang"):
            sc = g[~g["system"].isin(bl)]
            ks.append((int((sc["pred_stable"].astype(bool) == sc["gt_stable_prod"].astype(bool)).sum()), len(sc)))
        p = prod[(prod["model"] == model) & ~prod["system"].isin(bl)]
        ks.append((int((p["pred_stable"].astype(bool) == p["gt_stable"].astype(bool)).sum()), len(p)))
        lo, hi = min(ks), max(ks)
        return f"{lo[0]}/{lo[1]}" if lo == hi else f"{lo[0]}/{lo[1]} to {hi[0]}/{hi[1]}"

    models = sorted(m["model"].unique())
    invariant = [mm for mm in models if not flips["model"].eq(mm).any()]
    ctrl_only = [mm for mm in models if mm not in invariant
                 and set(flips.loc[flips["model"] == mm, "system"]) <= set(CONTROL_SYSTEMS)]
    other = [mm for mm in models if mm not in invariant and mm not in ctrl_only]
    names = lambda ms: ", ".join(PRETTY.get(x, x) for x in ms)

    text = (
        "**Table S13** Sensitivity of the harmonic layer to the finite-displacement amplitude "
        "(0.005, 0.02 and 0.03 Å against the production 0.01 Å), all five models: an axis "
        "that ESI §S1.2's v1/v2 replicate cannot probe. Deviations are against the same model's "
        "production 0.01 Å row.\n\n"
        + md(rows, ["Model", "Amplitude (Å)", "n", r"Median \|Δ\| (THz)", r"Max \|Δ\| (THz)",
                    "Call flips", "Harmonic accuracy [95% CI]"])
        + f"\n\n**{names(invariant)} change no stability call at any amplitude**; their harmonic "
        "accuracies are " + ", ".join(f"{PRETTY.get(x, x)} {acc_range(x)}" for x in invariant)
        + " at every amplitude. ")
    for x in ctrl_only:
        text += (f"{PRETTY.get(x, x)} changes calls only on harmonically stable controls "
                 f"({flip_lines(x)}), so its harmonic accuracy runs from {acc_range(x)} across "
                 "amplitudes; with its tolerance dependence (§3.1), it is not a robust number "
                 "under either knob. ")
    for x in other:
        text += (f"{PRETTY.get(x, x)} changes calls on test systems as well as controls "
                 f"({flip_lines(x)}); its accuracy runs {acc_range(x)}. Its harmonic calls are "
                 "amplitude-dependent. Of the five models it is the only one whose forces are "
                 "predicted directly rather than as gradients of an energy (non-conservative), a "
                 "plausible cause that this design cannot isolate; CHGNet, SevenNet-0 and "
                 "MatterSim also return float32 forces as run and do not show it, so precision "
                 "alone does not explain it. Its harmonic calls should be read with that caveat. ")
    return text + "`scripts/run_disp_sweep.py`."


# ------------------------------------------------------- revision tables S14-S17 ----

SENS_AXIS = {"baseline": "Production", "a_fit_window": "(a) Fit window",
             "b_sampling_range": "(b) Sampling range",
             "c_cell_normalisation": "(c) Frozen-cell normalisation",
             "d_constants": "(d) Solver constants"}


def _sens_label(s: dict) -> str:
    import re
    lab = s["label"]
    if s["key"] == "production":
        return ("production: window 5× / floor 60 meV, Q ≤ 0.45 Å, minimal cell, threshold "
                "1.5 steps, box 0.6 Å")
    lab = lab.replace(" (supplementary)", "").replace(" (prod.)", "").replace("<=", "≤")
    lab = re.sub(r"(\d)x\b", "\\1×", lab)
    lab = re.sub(r"\bA\b", "Å", lab)
    return lab.replace("FC supercell", "force-constant supercell").replace(" pts", " points")


def table_s14_screen_sensitivity(sens: dict) -> str:
    """Referee 1.5: sensitivity of the single-mode screen, from scripts/screen_sensitivity.py.

    Compact: the production row once, then every non-production setting. Each cell carries
    all five models, then ORB-v2 excluded. The notes below the table are generated from the
    JSON's diagnostics block so no number in them is typed by hand.
    """
    sets = sens["settings"]
    by_key = {s["setting"]["key"]: s for s in sets}

    def two(s: dict, f) -> str:
        return f"{f(s['all_models'], 'all')} · {f(s['exORB'], 'exORB')}"

    def uf(s: dict, lvl: str) -> str:
        u = s["unit_flips"]
        a, x = u[lvl], u["exORB" if lvl == "all" else "exORB_le300K"]
        return f"{a['k']}/{a['n']} · {x['k']}/{x['n']}"

    rows = []
    for s in sets:
        st_ = s["setting"]
        if st_["is_production"] and st_["key"] != "production":
            continue
        rows.append([
            SENS_AXIS.get(st_["axis"], st_["axis"]), _sens_label(st_),
            uf(s, "le300K"), uf(s, "all"),
            two(s, lambda m, _: ci2(m["fe_recall"])),
            two(s, lambda m, _: m["srtio3_gate_passing"]["fmt"]),
            two(s, lambda m, _: ci2(m["accuracy"])),
            two(s, lambda m, _: f"{ci2(m['accuracy_all_T'])} ({m['false_unstable_all_T']['k']}/"
                                f"{m['false_unstable_all_T']['n']})"),
        ])

    diag = sens["diagnostics"]
    win = [s for s in sets if s["setting"]["axis"] == "a_fit_window" and not s["setting"]["is_production"]]
    trn = [s for s in sets if s["setting"]["axis"] == "b_sampling_range" and not s["setting"]["is_production"]]
    kept = diag["kept_set_identity"]["floor_60meV"]
    n_shallow = diag["kept_set_identity"]["modes_shallower_than_7p5meV"]

    def unit_systems(s: dict) -> str:
        cnt: dict[str, int] = {}
        for u in s["unit_flips"]["all"]["units"]:
            sysname = u.split("/")[0] if isinstance(u, str) else u["system"]
            cnt[sysname] = cnt.get(sysname, 0) + 1
        return ", ".join(f"{k} {v}" for k, v in sorted(cnt.items()))

    t25, t40 = by_key["trunc_0.25"], by_key["trunc_0.40"]
    fb = diag["brentq_fallback"]
    cells = [("per formula unit", by_key["cell_per_fu"]), ("the production minimal cell", by_key["production"]),
             ("a doubled minimal cell", by_key["cell_doubled"]),
             ("the common force-constant supercell", by_key["cell_common_fc"])]
    x8 = by_key["cell_x8"]
    stab = diag["le300K_scored_stable_units_with_modes"]
    ext = sens["scaling_checks"]["mlip_extensivity"]
    prod = by_key["production"]["all_models"]
    bl = sens["baseline"]

    def all_or(r: dict) -> str:
        return f"all {r['n']}" if r["k"] == r["n"] else f"{r['k']} of {r['n']}"

    baseline_txt = (f"The production baseline reproduces {all_or(bl['unit_calls_match'])} unit calls "
                    f"and {all_or(bl['mode_T_calls_match'])} mode-temperature calls in the ledger. ")

    def seq(f) -> str:
        vals = [f(s["all_models"]) for _, s in cells]
        return ", ".join(vals[:-1]) + " and " + vals[-1]

    notes = (
        f"**Fit window.** None of the {len(win)} non-production window settings (multiplier 3×, "
        "5× or 8× the well depth, floor 30, 60 or 120 meV) changes a unit call at any "
        "temperature; mode-level condensation calls change on at most "
        f"{max(s['mode_flips']['le300K']['k'] for s in win)}/756 mode-temperature evaluations at "
        "T ≤ 300 K. The multiplier on its own is a weak probe: at the production 60 meV floor "
        f"the kept point set is identical at 3×, 5× and 8× for {kept['k']}/{kept['n']} modes "
        f"({100 * kept['k'] / kept['n']:.0f}%), including all {n_shallow} modes shallower than "
        "7.5 meV, which is why the floor is varied as well.\n\n"
        f"**Sampling range.** Truncating each sampled E(Q) at 0.25 to 0.40 Å changes at most "
        f"{max(s['unit_flips']['le300K']['k'] for s in trn)}/200 unit calls at T ≤ 300 K. Over "
        f"all four temperatures truncation at 0.25 Å changes {t25['unit_flips']['all']['k']}/400 "
        f"({unit_systems(t25)}) and at 0.40 Å {t40['unit_flips']['all']['k']}/400 "
        f"({unit_systems(t40)}). In the production maps "
        f"{diag['unbracketed']['Q<=0.45']['fmt']} wells are still descending at the largest "
        f"sampled Q (0.45 Å), {diag['unit_Q0_beyond_sampled_range_0p45A']['fmt']} unit order "
        f"parameters lie beyond it, and {diag['ledger_unit_Q0_at_box_edge']['fmt']} sit at the "
        "0.6 Å edge of the centroid scan; doubling the scan to 1.2 Å changes no call.\n\n"
        f"**Solver constants.** Every non-zero centroid is at least "
        f"{diag['mode_Q0_distribution']['smallest_nonzero_A']:g} Å, so condensation thresholds "
        "from 0.0025 to 0.015 Å (0.5 to 3 scan steps) cannot change a call. The bracketed width "
        f"solve falls back to the grid-nearest σ² on {fb['k']}/{fb['n']} centroid evaluations "
        f"({100 * fb['share']:.1f}%), and never at a deciding minimum "
        f"({fb['at_argmin']['k']}/{fb['at_argmin']['n']}).\n\n"
        "**Frozen-cell normalisation is outcome-determining.** Rescaling the cell over which M "
        "and V(Q) are summed by a factor n maps (a, b, c, M) to n(a, b, c, M), which is the "
        "minimal-cell problem at mass n²M and temperature T/n: a larger cell is a heavier, "
        "effectively colder and more classical mode, and condenses more readily. (The MLIP "
        "energies are extensive in the cell as assumed: for the SrTiO₃/MACE-MP-0 R mode the "
        f"doubled-cell energy and mass are {ext['E0_ratio']:.6f} and {ext['M_ratio']:.6f} times "
        "the minimal-cell values.) Taking the cell as "
        + ", ".join(n for n, _ in cells[:-1]) + " and " + cells[-1][0]
        + f", FE recall runs {seq(lambda m: m['fe_recall']['fmt'].split(' =')[0])} and the "
        f"SrTiO₃ gate passes for {seq(lambda m: m['srtio3_gate_passing']['fmt'])} models. "
        "Accuracy cannot choose among the conventions. At T ≤ 300 K it is "
        f"{seq(lambda m: m['accuracy']['fmt'].split(' =')[0])}, but only {stab['k']}/{stab['n']} "
        "of the truly stable units in that set carry a screened imaginary mode, so the column is "
        "nearly blind to over-condensation. Over all four temperatures the extra condensation of "
        "larger cells shows up as false-unstables "
        f"({seq(lambda m: m['false_unstable_all_T']['fmt'].split(' =')[0])}), and the 8× cell "
        f"moves from above production at T ≤ 300 K ({x8['all_models']['accuracy']['fmt'].split(' =')[0]} "
        f"against {prod['accuracy']['fmt'].split(' =')[0]}) to below it over all T "
        f"({x8['all_models']['accuracy_all_T']['fmt'].split(' =')[0]} against "
        f"{prod['accuracy_all_T']['fmt'].split(' =')[0]}). Which temperatures are scored decides "
        "the ranking, so these data do not identify the physically right normalisation. The "
        "production convention is the minimal cell in which the mode is a single commensurate "
        "distortion (§S1.3), and the screen's T* and FE recall are conditional on it."
    )
    return (
        "**Table S14** Sensitivity of the single-mode soft-mode screen to its fit window, "
        "sampling range, frozen-cell normalisation and solver constants. Every row re-solves "
        "the cached E(Q) maps with the production solver; no MLIP is re-evaluated. Each cell "
        "gives all five models, then ORB-v2 excluded (all · ex-ORB). Unit flips count "
        "(system, model, T) stability calls that differ from production. FE recall is the "
        "recall of the unstable class on BaTiO₃, KNbO₃ and PbTiO₃ at T ≤ 300 K. The SrTiO₃ gate "
        "counts models that condense at 100 K and are stable at 300, 600 and 900 K. Accuracy is "
        "on the scored finite-T set (T ≤ 300 K, bcc and KTaO₃ excluded); the all-T column scores "
        "the same systems over 100 to 900 K, with the false-unstable count in parentheses. "
        "Intervals are unit-level Wilson and do not account for clustering by system. "
        + baseline_txt
        + "`scripts/screen_sensitivity.py` → `results/screen_sensitivity.json`.\n\n"
        + md(rows, ["Axis", "Setting", "Unit flips, T ≤ 300 K", "Unit flips, all T",
                    "FE recall [95% CI]", "SrTiO₃ gate", "Accuracy, T ≤ 300 K [95% CI]",
                    "Accuracy, all T [95% CI] (false-unstable)"])
        + "\n\n" + notes
    )


def table_s15_h2_clustered(st: dict) -> str:
    """The H2 transfer asymmetry at every ladder temperature, clustered by system."""
    h = st["h2_clustered"]
    rows = []
    loso_lines = []
    for T in ("100", "300", "600", "900"):
        t = h[T]
        shared = t["shared_screen_error_systems"]
        shared_txt = ", ".join(f"{s} ({v['n_models_wrong']}/{v['n_models']})"
                               for s, v in shared.items()) or "none"
        for tag, label in (("all_models", "all five"), ("excl_orb_v2", "excluding ORB-v2")):
            x = t[tag]
            c = x["clustered_by_system"]
            sp = x["shared_screen_error_split"]
            w = sp["without_shared_systems"]
            rows.append([
                T, label, f"{x['b_harm_right_ft_wrong']} v {x['c_harm_wrong_ft_right']}",
                pfmt(x["mcnemar_exact_p_UNIT_LEVEL"]), pfmt(c["p_exact_clustered"]),
                f"{c['clusters_favouring_a']} / {c['clusters_favouring_b']} / {c['clusters_tied']}",
                shared_txt,
                f"{sp['b_in_shared_systems']} v {sp['c_in_shared_systems']}",
                f"{w['b']} v {w['c']} ({w['n_systems']} systems)",
            ])
            lo = x.get("leave_one_system_out")
            if lo:
                below = lo.get("drops_with_p_clustered_below_0.05") or []
                loso_lines.append((int(T), label, lo["p_clustered_min"], lo["p_clustered_max"],
                                   below))
    all_lo = [r for r in loso_lines if r[1] == "all five"]
    lowest = min(all_lo, key=lambda r: r[2])
    breaks = [r for r in loso_lines if r[4]]
    loso_txt = (
        "Leave-one-system-out: with all five models, dropping any single system never brings "
        f"the clustered p below 0.05 at any temperature (lowest {pfmt(lowest[2])}, at "
        f"{lowest[0]} K)")
    if breaks:
        loso_txt += "; " + "; ".join(
            f"{r[1]} at {r[0]} K, dropping {' or '.join(r[4] if isinstance(r[4][0], str) else [d['system'] for d in r[4]])} "
            f"gives {pfmt(r[2])}" for r in breaks)
    loso_txt += (". Where one system's removal is enough to move p across 0.05, the value "
                 "cannot carry a significance claim either way." if breaks else ".")
    return (
        "**Table S15** The H2 transfer asymmetry at every ladder temperature, with and without "
        "ORB-v2. b counts matched (system, model) pairs whose harmonic call is right and whose "
        "screen call at T is wrong; c counts the reverse. The matched set is 15 non-bcc, "
        "non-borderline systems × 5 models (75 pairs; 60 without ORB-v2). **The system-clustered "
        "p is the test**: exact enumeration of the 2^k per-system sign flips "
        "(`stats.cluster_exact_paired`), with the systems favouring b, favouring c and tied "
        "given alongside. The unit-level McNemar p treats the pairs as independent, which they "
        "are not (Table S7), and is given only as a labelled companion. The last three columns "
        "are a descriptive decomposition, not a test. A shared-screen-error system is one the "
        "screen mis-calls at that temperature for at least four of the five models; the set is "
        "defined on all five models and applied to both model sets. Because it is selected on "
        "the finite-temperature outcome, removing it removes b-type pairs by construction. The "
        "columns show where the asymmetry sits, in systems the screen mis-calls for nearly "
        "every model. They do not show that it is absent elsewhere, and they do not say whether "
        "an error shared by four or five models comes from the single-mode approximation or "
        "from a feature of the surface the models share (§S1.3). "
        "`scripts/stats_hardening.py` (`h2_clustered`).\n\n"
        + md(rows, ["T (K)", "Model set", "b v c", "Unit-level McNemar p (companion only)",
                    "System-clustered exact p", "Systems b > c / c > b / tied",
                    "Shared-screen-error systems (models mis-called)", "b v c inside them",
                    "b v c without them"])
        + "\n\n" + loso_txt
    )


def table_s16_bcc_agreement(st: dict) -> str:
    """bcc agreement scored on the call, plus the same-PES comparison behind the SSCHA
    false-stables on the displacive systems (criterion_blindness).

    The screen's symmetric-point curvature is not scored against SSCHA. For a single even mode
    it equals the self-consistent trial stiffness at Q0 = 0 (scripts/curvature_identity_check.py),
    so it is positive wherever the width equation is solved and cannot register condensation;
    its negative bcc values are numerical and are accounted for in the caption from
    results/curvature_identity_check.json."""
    ba = st["bcc_agreement"]
    a, x = ba["all_models"], ba["excl_orb_v2"]
    triv_pairs = a["trivial_split"]["trivial_pairs"]
    triv_names = sorted({PRETTY.get(p.split("/")[1], p) for p in triv_pairs})
    triv_sys = [ELEMENT[s] for s in BCC if any(p.startswith(s) for p in triv_pairs)]
    rows = [
        ["all pairs, all five models", a["call_agreement"]["fmt"]],
        ["all pairs, excluding ORB-v2", x["call_agreement"]["fmt"]],
        [f"trivial pairs ({' and '.join(triv_names)} on {' and '.join(triv_sys)})",
         a["trivial_split"]["trivial"]["call"]["fmt"]],
        ["non-trivial pairs, all five models", a["trivial_split"]["non_trivial"]["call"]["fmt"]],
        ["non-trivial pairs, excluding ORB-v2", x["trivial_split"]["non_trivial"]["call"]["fmt"]],
    ]
    for m in sorted(a["per_model"]):
        rows.append([PRETTY.get(m, m), a["per_model"][m]["call"]["fmt"]])
    fc, fx = a["frequency_correlation_DESCRIPTIVE"], x["frequency_correlation_DESCRIPTIVE"]

    # account for every negative bcc screen curvature on the 45 paired units
    neg_txt = ""
    if CURV.exists():
        cc = json.loads(CURV.read_text(encoding="utf-8"))
        units = cc["bcc_sscha_paired_units_with_negative_screen_curvature"]
        groups: dict[str, list[str]] = {}
        for u in units:
            groups.setdefault(u["cause"], []).append(
                f"{ELEMENT.get(u['system'], u['system'])}/{PRETTY.get(u['model'], u['model'])} "
                f"{u['T']:.0f} K")
        label = {
            "width-solver fallback at a stencil point":
                "the width solver fell back to its nearest grid value at a point of the "
                "finite-difference stencil",
            "root found at every stencil point (see a, b, c)":
                "the fitted polynomial has a positive quadratic term and a well only between "
                "sample points, a fit artefact of the kind described in §S1.4",
            "no screened imaginary mode: ledger shows the softest commensurate harmonic value":
                "the model has no screened imaginary mode and the value recorded is the softest "
                "commensurate harmonic frequency, not a curvature",
        }
        parts = [f"{len(v)} where {label.get(k, k)} ({', '.join(v)})" for k, v in groups.items()]
        pr = cc["positive_omega_eff_vs_trial_frequency_rel_diff"]
        neg_txt = (
            "The screen's symmetric-point curvature is not scored against SSCHA. For a single "
            "mode with an even potential it equals the self-consistent trial stiffness at "
            "Q₀ = 0 (§S1.3), so it is positive wherever the width equation has a root and cannot "
            "register condensation; over all "
            f"{cc['n_mode_T_evaluations']} mode-temperature evaluations the positive values agree "
            f"with the trial frequency to a median relative difference of {_sci(pr['median'])} "
            f"(95th percentile {pr['p95']:.1%}). Of the {a['call_agreement']['n']} paired bcc "
            f"units, {len(units)} carry a negative screen value, and none of them is a physical "
            "curvature: " + "; ".join(parts) + ". `scripts/curvature_identity_check.py`. "
        )

    cb = st["criterion_blindness"]
    ca, cx = cb["all_models"], cb["excl_orb_v2"]
    crows = [
        ["non-bcc units SSCHA calls stable against an unstable label",
         str(ca["n_sscha_false_stable_nonbcc"]), str(cx["n_sscha_false_stable_nonbcc"])],
        ["screen's free-energy comparison calls the phase unstable",
         ca["screen_call_unstable"]["fmt"], cx["screen_call_unstable"]["fmt"]],
        ["screen's symmetric-point curvature positive (by construction where the width "
         "equation is solved; §S1.3)", ca["screen_curvature_positive"]["fmt"],
         cx["screen_curvature_positive"]["fmt"]],
    ]
    for s in sorted(ca["by_system"]):
        v, w = ca["by_system"][s], cx["by_system"].get(s, {"n": 0, "screen_curv_pos": 0,
                                                            "screen_call_unstable": 0})
        crows.append([f"  of which {s} (n; call unstable)",
                      f"{v['n']}; {v['screen_call_unstable']}",
                      f"{w['n']}; {w['screen_call_unstable']}"])
    pa, px = ca["paired_nonbcc_T_le_300"], cx["paired_nonbcc_T_le_300"]
    crows.append(["all paired non-bcc units at T ≤ 300 K: SSCHA call = screen call",
                  f"{pa['sscha_call_eq_screen_call']}/{pa['n']}",
                  f"{px['sscha_call_eq_screen_call']}/{px['n']}"])

    return (
        "**Table S16** Screen-versus-SSCHA agreement on the bcc metals, scored on the stability "
        "call, and the same-energy comparison behind the SSCHA false-stables on the displacive "
        "systems, with the production-recipe SSCHA (2×2×2, not converged, §S2.1; the same "
        "comparisons with the converged recipe are in Table S22). For SSCHA the call is the sign "
        "of the lowest free-energy-Hessian frequency; for the screen it is the variational argmin "
        "of §2.4. Both methods run on the same MLIP "
        "potential-energy surface, so agreement between them is a consistency check and says "
        "nothing about agreement with first principles. *Trivial* pairs are (system, model) "
        "combinations whose harmonic layer has no instability, so both methods agree without "
        "any thermal stabilisation having been tested. bcc pairs are Ti, Zr and Hf at 100, 300 "
        "and 600 K. " + neg_txt
        + "`scripts/stats_hardening.py` (`bcc_agreement`, `criterion_blindness`).\n\n"
        + md(rows, ["Subset", "Stability-call agreement [95% CI]"])
        + "\n\nThe frequency magnitudes are given descriptively only, because the pairs cluster "
        f"by system and model and no test is attached: Spearman ρ = {signed(fc['spearman'], 3)} "
        f"(n = {fc['n']}), {signed(fx['spearman'], 3)} without ORB-v2 (n = {fx['n']}).\n\n"
        "Lower part: on every non-bcc unit where SSCHA calls the phase stable against an "
        "unstable label, the screen's call on the same MLIP energies. Where the screen calls the "
        "phase unstable, its free energy has a displaced minimum below the symmetric point, "
        "while the symmetric point of the single-mode problem keeps a positive curvature, so a "
        "criterion read at the symmetric reference would report stable. That shows the local "
        "and the global question have different answers on these energies. It does not show "
        "that SSCHA's positive Hessian has the same origin; the converged recipe of Table S22 "
        "keeps the false-stables on BaTiO₃ and KNbO₃ and removes those on the fluorites (§3.3).\n\n"
        + md(crows, ["Quantity", "All five models", "Excluding ORB-v2"])
    )


def table_s17_sscha_high_t(st: dict) -> str:
    """SSCHA false-unstables growing with temperature, the SrTiO3 above-Tc units, and the
    blow-ups and failures per model."""
    sh = st["sscha_high_t"]
    rows = []
    for T in ("100", "300", "600", "900"):
        for tag, label in (("all_models", "all five"), ("excl_orb_v2", "excluding ORB-v2")):
            r = sh[tag]["false_unstable_by_T"][T]
            bysys = ", ".join(f"{s} {k}" for s, k in sorted(r["by_system"].items())) or "--"
            frac = r["fmt"] if r["n"] else "0/0 (no stable label at this T)"
            rows.append([T, label, r["n_units_at_T"], frac, r["n_blowup_among_false_unstable"],
                         bysys])

    so = sh["all_models"]["srtio3_above_tc"]
    srows = []
    for u in so["units"]:
        q = tuple(float(c) for c in u["screen_deciding_mode_q"].split(","))
        srows.append([PRETTY.get(u["model"], u["model"]), f"{u['T']:.0f}",
                      signed(u["sscha_min_eff_freq_thz"], 1), "yes" if u["blow_up"] else "no",
                      signed(u["harmonic_min_freq_thz"], 2),
                      "(" + ", ".join(f"{c:g}" for c in q) + ")",
                      signed(u["screen_deciding_mode_harm_thz"], 2)])
    sox = sh["excl_orb_v2"]["srtio3_above_tc"]
    below_harm = all(u["sscha_min_eff_freq_thz"] < u["harmonic_min_freq_thz"] for u in so["units"])
    # A listed screen mode with a non-negative harmonic frequency is the placeholder the screen
    # records when a model has no imaginary commensurate mode at all; say so under the table.
    no_imag = {}
    for u in so["units"]:
        if u["screen_deciding_mode_harm_thz"] >= 0:
            no_imag[PRETTY.get(u["model"], u["model"])] = u["screen_deciding_mode_harm_thz"]
    no_imag_txt = "".join(
        f" {m} has no imaginary commensurate mode in SrTiO₃, so the screen condenses nothing and "
        f"the mode listed for it is its softest commensurate mode, which is harmonically stable "
        f"({signed(f, 2)} THz)."
        for m, f in sorted(no_imag.items()))

    tn = sh["all_models"]["stable_at_100K_turn_negative"]
    tnx = sh["excl_orb_v2"]["stable_at_100K_turn_negative"]
    neg = [u for u in tn["units"] if u["negative_by_600_900K"]]
    groups: dict[str, list[str]] = {}
    for u in neg:
        groups.setdefault(u["system"], []).append(PRETTY.get(u["model"], u["model"]))
    neg_txt = "; ".join(f"{s} ({', '.join(v)})" for s, v in sorted(groups.items()))
    label_right = sorted({u["system"] for u in neg if not u["high_T_negative_is_false_unstable"]})

    bf = st["orb_split_s3"]["sscha_blowups_and_failures"]
    prow = []
    for m in sorted(bf["per_model"]):
        p = bf["per_model"][m]
        fails = ", ".join(u.replace("@", " ").replace("K", " K") for u in p["failed_units"]) or "--"
        blows = ", ".join(f"{b['system']} {b['T']:.0f} K ({signed(b['min_eff_freq_thz'], 1)} THz)"
                          for b in p["blowup_units"]) or "--"
        prow.append([PRETTY.get(m, m), p["n_grid"], p["n_returned"], p["n_failed"], fails,
                     p["n_blowup"], blows])
    for tag, label in (("all_models", "total, all five"), ("excl_orb_v2", "total, excluding ORB-v2")):
        t = bf["totals"][tag]
        prow.append([f"*{label}*", t["n_grid"], t["n_returned"], t["n_failed"], "", t["n_blowup"], ""])

    return (
        "**Table S17** Where SSCHA, with the production recipe (not converged, §S2.1), calls a "
        "phase unstable that its label calls stable, and where it fails numerically. Upper part: non-bcc SSCHA false-unstables by temperature, over "
        "the units whose label is stable at that temperature; numerical blow-ups "
        "(|f| > 50 THz) are included in the counts and also tallied separately. The count grows "
        "with temperature partly because more labels are stable at high temperature; the direct "
        "evidence is the per-unit trend below the table, in which units SSCHA calls stable at "
        "100 K turn negative by 600 to 900 K. An instability that grows with thermal amplitude "
        "on a fixed potential-energy surface is the opposite of entropy stabilisation. It is what "
        "an MLIP extrapolating on large-amplitude thermal configurations would produce; an "
        "instability of the stochastic sampling itself, and a free-energy Hessian evaluated at an "
        "auxiliary matrix that has not reached the SCHA minimum, are the other candidates, and "
        "these data do not separate them. PBE forces on twelve configurations of the SrTiO₃ "
        "MACE-MP-0 600 K ensemble (Table S20) show the MLIPs extrapolating there (relative force "
        "error 0.19 against 0.11 near equilibrium, energy errors to 44 meV per atom), without "
        "showing that this rather than the sampling drives the runaway. The seed study shows the "
        "SrTiO₃ 600 K relaxation had not converged on any of four seeds (Table S21). With the "
        "converged recipe these counts are 0/4, 3/12 and 3/18 at 300, 600 and 900 K (all "
        "CsSnI₃), every SrTiO₃ unit above its transition is stable, and no unit turns negative "
        "with temperature (Table S22): the runaway is the production relaxation, not the MLIP. "
        "`scripts/stats_hardening.py` (`sscha_high_t`, `orb_split_s3`).\n\n"
        + md(rows, ["T (K)", "Model set", "Non-bcc units returned",
                    "False-unstable / stable-labelled [95% CI]", "Of which blow-ups", "By system"])
        + f"\n\nOf the {tn['n_stable_at_100K']} non-bcc units SSCHA calls stable at 100 K, "
        f"{tn['fmt']} turn negative by 600 to 900 K (without ORB-v2 {tnx['fmt']}): {neg_txt}. "
        f"For {tn['n_of_those_false_unstable_at_high_T']} of the {tn['n_negative_by_600_900K']} "
        "the label is stable at that temperature, so the negative value is a false-unstable; "
        f"the rest are {' and '.join(label_right)} units, labelled unstable at every temperature, "
        "where the sign is right but the trend with temperature runs the wrong way.\n\n"
        f"SrTiO₃ above its {so['transition_T_K']:.0f} K transition: SSCHA calls every returned "
        f"unit unstable ({so['false_unstable']['fmt']}; without ORB-v2 {sox['false_unstable']['fmt']}), "
        f"at {signed(so['sscha_freq_range_thz'][1], 1)} to {signed(so['sscha_freq_range_thz'][0], 0)} THz "
        f"({signed(sox['sscha_freq_range_thz'][1], 1)} to {signed(sox['sscha_freq_range_thz'][0], 1)} THz "
        "without ORB-v2)"
        + (", each more negative than the same model's harmonic minimum. " if below_harm else ". ")
        + "The production recipe never calls SrTiO₃ stable at any temperature; the converged "
        "recipe calls it stable in every converged unit, including 100 K, below the transition "
        "(Table S22)." + no_imag_txt + "\n\n"
        + md(srows, ["Model", "T (K)", "SSCHA min frequency (THz)", "Blow-up",
                     "Harmonic min frequency (THz)", "Screen deciding mode q",
                     "Its harmonic frequency (THz)"])
        + "\n\nBlow-ups and failed units by model. A failed unit is one of the "
        f"{bf['totals']['all_models']['n_grid']}-unit attempted grid that returned no number: it "
        "stopped at a cellconstructor symmetry or ensemble assertion. The blow-ups spread over "
        "four models and the failures over three. As run, only MACE-MP-0 returns float64 "
        "forces, and it has neither; it is also a different architecture, and one model is no "
        "basis for attributing the failures to numerical precision.\n\n"
        + md(prow, ["Model", "Grid units", "Returned", "Failed", "Failed units", "Blow-ups",
                    "Blow-up units"])
    )


def table_s18_force_spread() -> str:
    """The pre-registered force-level ensemble test (Referee 2.2; scripts/force_spread.py).

    Every score the pre-registration names is listed, primary first, whatever it shows; the
    numbers are read from results/revision/force_spread/summary.json, whose block hash ties them
    to the pre-registration text at the head of the script."""
    if not FSPREAD.exists():
        return ""
    fs = json.loads(FSPREAD.read_text(encoding="utf-8"))
    cl, pm = fs["consensus_level"], fs["per_model_level"]

    def ci_txt(v: dict) -> str:
        return f"{v['auc']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"

    def p_txt(v: dict) -> str:
        p = v.get("p_perm_clustered")
        return "--" if p is None else f"{p:.3f}"

    spec = [
        ("primary", cl["primary_S_5model"], "force spread S, five models", "consensus wrong"),
        ("(a)", cl["a_S_ex_orb_4model"], "S, four models (no ORB-v2)", "four-model consensus wrong"),
        ("(b)", cl["b_S_mace_committee"], "S, MACE-MP-0 small/medium/large", "consensus wrong"),
        ("(c)", cl["c_S_mattersim_committee"], "S, MatterSim 1M/5M", "consensus wrong"),
        ("(d)", cl["d_Snorm_5model"], "normalised S, five models", "consensus wrong"),
        ("(d)", cl["d_Snorm_ex_orb_4model"], "normalised S, four models", "four-model consensus wrong"),
        ("(e)", pm["e_loo_pooled_5model"], "leave-one-out deviation, pooled over five models",
         "that model's call wrong"),
        ("(e)", pm["e_loo_pooled_ex_orb_4model"], "leave-one-out deviation, pooled over four models",
         "that model's call wrong"),
    ]
    for m in ("mace_mp0", "chgnet", "orb_v2", "sevennet0", "mattersim"):
        for tag, lab in (("5model", "mean of the other four"), ("ex_orb", "mean of the other three, ORB-v2 excluded")):
            k = f"e_loo_{m}_{tag}"
            if k in pm:
                spec.append(("(e)", pm[k], f"leave-one-out deviation of {PRETTY[m]} from the "
                             f"{lab}", f"{PRETTY[m]}'s call wrong"))
    spec += [
        ("(f)", pm["f_committee_mace_mp0_own_call"], "S, MACE-MP-0 committee", "MACE-MP-0's call wrong"),
        ("(f)", pm["f_committee_mattersim_own_call"], "S, MatterSim committee", "MatterSim's call wrong"),
        ("(g)", cl["g_S_5model_no_overlap"], "primary, units with no atomic overlap in any "
         "configuration", "consensus wrong"),
    ]
    rows = [[tag, score, label, f"{v['n_units']} ({v['n_wrong']})", ci_txt(v), p_txt(v)]
            for tag, v, score, label in spec]
    sec = [v for tag, v, _, _ in spec if tag != "primary"]
    above = [f"{tag} {score}" for tag, v, score, _ in spec if tag != "primary" and v["ci_lo"] > 0.5]
    pri, g = cl["primary_S_5model"], cl["g_S_5model_no_overlap"]
    per_t = fs["per_temperature_primary__descriptive"]
    per_t_txt = ", ".join(f"{T} K {v['auc']:.2f} ({v['n_wrong']} wrong)" for T, v in per_t.items())
    return (
        "**Table S18** Force-level ensemble spread as a predictor of unreliable finite-temperature "
        "calls, the test pre-registered at the head of `scripts/force_spread.py` (block hash "
        f"{fs['preregistration']['block_sha256'][:8]}, recorded in the output). Units are the "
        "60 consensus (system, temperature) units of Table S8, or the 300 per-model units behind "
        "them. For each unit, 16 thermally displaced configurations are drawn from the quantum "
        "harmonic distribution of the five models' mean force constants in the 2×2×2 supercell of "
        "the unrelaxed reference cell, the same configurations for every model. S is the median "
        "over configurations of the root-mean-square over atoms and components of the standard "
        "deviation of the forces across the models in the set. AUCs are for predicting the "
        "outcome in the Label column; intervals are cluster-bootstrap 95% intervals over systems and "
        "p is the system-clustered permutation p (10,000 draws each, seed 0); for (g) the blocks are "
        "unequal and only the interval is defined. The rule fixed in advance: the spread is "
        "reported as flagging untrustworthy calls only if the primary interval lies entirely above "
        f"0.5 and the four-model estimate (a) is also above 0.5. Verdict: "
        f"{fs['verdict'].split(':')[0]}, because the primary interval spans 0.5. "
        f"Of the {len(sec)} secondary scores, {len(above)} have a lower bound above 0.5"
        f": {'; '.join(above)}. None is adjusted for multiple comparisons, all use the same "
        "configurations, and the overlap restriction (g) was pre-registered for the primary only, "
        f"so it was not applied to them. Restricted by (g), {g['n_units']} units remain and hold "
        f"{g['n_wrong']} of the {pri['n_wrong']} consensus errors: the soft halide and "
        "ferroelectric spectra that carry most errors are the ones whose thermal draws bring "
        "A-site cations into the anions, where every model is extrapolating. The configurations "
        "sit near the unrelaxed reference cell rather than each model's relaxed cell, and the "
        "committees differ in size and training run, so they are family committees and not a "
        "deep ensemble of one model. Per-temperature primary AUCs (15 units each, descriptive "
        f"only): {per_t_txt}. `scripts/force_spread.py` → "
        "`results/revision/force_spread/summary.json`.\n\n"
        + md(rows, ["Pre-registered score", "Spread", "Label", "Units (wrong)",
                    "AUC [95% CI]", "Clustered p"])
    )


# ------------------------------------------- revision tables S19-S20 (PBE, C3a/C3b) ----

C3A_ORDER = ["batio3_cubic", "knbo3_cubic", "srtio3_cubic", "cssnbr3_cubic", "zr_bcc",
             "zro2_cubic"]
EDGE_Q_A = 0.449      # the scan's largest amplitude is 0.45 A; a minimum at or beyond this is at the edge
SHORT = {"batio3_cubic": "BaTiO₃", "knbo3_cubic": "KNbO₃", "srtio3_cubic": "SrTiO₃",
         "cssnbr3_cubic": "CsSnBr₃", "zr_bcc": "bcc-Zr", "zro2_cubic": "ZrO₂"}
SYSNAME = SHORT | {"ti_bcc": "bcc-Ti", "hf_bcc": "bcc-Hf", "hfo2_cubic": "HfO₂",
                   "pbtio3_cubic": "PbTiO₃", "cssni3_cubic": "CsSnI₃"}


def c3a_ladder_units() -> pd.DataFrame:
    """The C3a unit calls on the ladder: the rows the ledger has a call for (100/300/600/900 K;
    the 50 K rows have none), each scored against the label for the MLIP's own energies on the
    PBE paths (``mlip_same_paths_stable``) and for PBE (``pbe_backed_stable``)."""
    u = pd.read_csv(C3A_UNITS)
    u = u[u["mlip_pred_stable_ledger"].notna()].copy()
    gt = u["gt_stable"].astype(bool)
    mlip, pbe = u["mlip_same_paths_stable"].astype(bool), u["pbe_backed_stable"].astype(bool)
    u["mlip_ok"], u["pbe_ok"] = mlip == gt, pbe == gt
    u["corrected"] = ~u["mlip_ok"] & u["pbe_ok"]
    u["newly_wrong"] = u["mlip_ok"] & ~u["pbe_ok"]
    return u


def _nz(x: float) -> float:
    """-0.0 -> 0.0, so that a zero well depth or ratio prints without a sign."""
    return float(x) + 0.0


def table_s19_c3a_pbe() -> str:
    """Referee 1.1: PBE along the screen's own soft-mode coordinates.

    Upper part: the unit calls, MLIP on the same paths against PBE, per system and in total with
    and without ORB-v2. Lower part: the well depth of each path's own model against PBE. Every
    number is read from results/revision/dft/c3a_unit_calls.csv and c3a_paths.csv, which
    ``scripts/dft_reference.py analyze`` writes."""
    if not (C3A_UNITS.exists() and C3A_PATHS.exists()):
        return ""
    u = c3a_ladder_units()

    def row(label: str, g: pd.DataFrame) -> list:
        return [label, len(g), int(g["mlip_ok"].sum()), int(g["pbe_ok"].sum()),
                int(g["corrected"].sum()), int(g["newly_wrong"].sum())]

    rows = [row(s, u[u["system"] == s]) for s in C3A_ORDER]
    rows += [row("*total, all five models*", u),
             row("*total, excluding ORB-v2*", u[u["model"] != "orb_v2"])]
    tot = rows[-2]

    # lower part: one row per PBE path, own-model curve against the PBE curve
    p = pd.read_csv(C3A_PATHS)
    own = p[p["curve"] == p["path_model"]]
    pbe = p[p["curve"] == "pbe"]
    m = own.merge(pbe, on="stem", suffixes=("_own", "_pbe"))
    m["pbe_well"] = m["depth_meV_pbe"] > 1e-6
    m["ratio"] = np.where(m["pbe_well"], m["depth_meV_own"].abs() / m["depth_meV_pbe"].where(
        m["pbe_well"], 1.0), np.nan)
    m["edge"] = m["pbe_well"] & (m["Q_min_A_pbe"] >= EDGE_Q_A)
    m["order"] = m["system_own"].map({s: i for i, s in enumerate(C3A_ORDER)})
    m = m.sort_values(["order", "stem"])
    drows = []
    for r in m.itertuples():
        has_well = r.depth_meV_own > 0
        drows.append([r.stem, r.role_own, f"{_nz(r.depth_meV_own):.1f}",
                      f"{_nz(r.depth_meV_pbe):.1f}",
                      f"{_nz(r.ratio):.2f}" if r.pbe_well else "no PBE well",
                      f"{r.Q_min_A_own:.3f}" if has_well else "no well",
                      f"{r.Q_min_A_pbe:.3f}" if r.pbe_well else "no well",
                      "yes" if r.edge else "no"])
    fe = m[m["system_own"].isin(["batio3_cubic", "knbo3_cubic"]) & (m["role_own"] == "decide")]
    fe4 = fe[fe["path_model_own"] != "orb_v2"]
    edge_sys = sorted({SHORT[s] for s in m.loc[m["edge"], "system_own"]})
    n_decide, n_ref = int((m["role_own"] == "decide").sum()), int((m["role_own"] == "ref").sum())
    n_mode = int((m["role_own"] == "mode").sum())
    n_pbe_nowell = int((~m["pbe_well"]).sum())
    nowell_roles = sorted(set(m.loc[~m["pbe_well"], "role_own"]))
    # coverage of the screened paths (summary.json lists every selected path, with or without PBE)
    ps = pd.DataFrame([{k: e.get(k) for k in ("stem", "system", "path_model", "role", "status",
                                              "cache_check_pass", "n_atoms")}
                       for e in json.loads((C3A_PATHS.parent / "summary.json").read_text(
                           encoding="utf-8"))["c3a"]["paths"]])
    n_screened_paths, n_done = len(ps), int((ps["status"] == "complete").sum())
    pend = ps[ps["status"] == "pending"]
    big = pend[pend["n_atoms"] > 12]
    n_copy = len(pend) - len(big)
    dr_off = {(r.system, r.path_model) for r in ps[ps["role"].isin(["decide", "ref"])
                                                   & (ps["cache_check_pass"] == False)].itertuples()}  # noqa: E712

    # caption facts, read from the data
    corr = u[u["corrected"]]
    zr_ref = corr[(corr["system"] == "zr_bcc") & (corr["kind"] == "reference_coordinate")]
    fe_corr = corr[corr["system"].isin(["batio3_cubic", "knbo3_cubic"])]
    fe_T = ", ".join(f"{t:.0f}" for t in sorted(set(fe_corr["T"])))
    fe_models = ", ".join(PRETTY[x] for x in sorted(set(fe_corr["model"])))
    nw = u[u["newly_wrong"]]
    nw_txt = "; ".join(f"{SHORT[r.system]}/{PRETTY[r.model]} at {r.T:.0f} K" for r in nw.itertuples())
    same_ledger = int((u["mlip_pred_stable_ledger"].astype(bool) == u["mlip_same_paths_stable"].astype(bool)).sum())
    n_off_cache = int((~u["all_paths_match_cache"]).sum())
    n_off_dr = int(sum((r.system, r.model) in dr_off for r in u.itertuples()))
    n_decide_pbe = int(u["deciding_mode_has_pbe"].sum())
    resid = u[~u["pbe_ok"]].groupby("system").size()
    resid_txt = ", ".join(f"{SHORT[s]} {resid[s]}" for s in C3A_ORDER if s in resid)
    n_nowell = int((m["depth_meV_own"] <= 0).sum())
    npb = u["n_paths_pbe"]
    return (
        f"**Table S19** PBE along the screen's own soft-mode coordinates. Each of the {tot[1]} ladder units (6 systems × 5 models × 100, 300, 600 and 900 K) "
        "is called twice by the screen's rule (unstable if any computed path condenses, §2.4), on "
        "the same structures: from the model's own energies (*MLIP, same paths*) and from "
        "Quantum ESPRESSO PBE single-point energies (*PBE-backed*). Both are scored against the "
        "finite-temperature label used throughout (stable iff T is at or above the experimental "
        "transition temperature). *Corrected by PBE* counts units whose MLIP call is wrong and "
        "whose PBE-backed call is right; *newly wrong* counts the reverse. **PBE is evaluated at "
        "each MLIP's own relaxed lattice, along that MLIP's coordinate**, so it is a different "
        f"PBE potential for each model, not one reference curve per system. PBE covers "
        f"{n_done} of the {n_screened_paths} screened paths: the deciding and reference paths and "
        "every other screened mode whose modulated cell has at most 12 atoms; of the "
        f"{len(pend)} without PBE, {n_copy} are symmetry copies of a computed path and {len(big)} "
        f"lie in cells of {int(big['n_atoms'].min())} to {int(big['n_atoms'].max())} atoms, so a "
        "PBE-backed *stable* means stable on the computed modes only. PBE covers "
        f"{int(npb.min())} to {int(npb.max())} paths per unit, out of the "
        f"{int(u['n_screened'].min())} to {int(u['n_screened'].max())} "
        f"modes the screen maps; the mode that decided the ledger's call has a PBE path in "
        f"{n_decide_pbe} of {len(u)} units. The MLIP column is the model's own call on those "
        f"same paths and equals the ledger's call in {same_ledger} of {len(u)} units. In "
        f"{n_off_cache} units ({n_off_dr} counting only deciding and reference paths) a path's "
        "regenerated E(Q) map did not reproduce the cached map the ledger was computed from (a substitute direction inside a degenerate eigenspace; "
        "`all_paths_match_cache` in the deposited table), so those calls rest partly on a "
        f"substitute coordinate. Of the {tot[4]} corrections, {len(zr_ref)} are bcc-Zr units on a "
        "reference coordinate and "
        f"{len(fe_corr)} are BaTiO₃ and KNbO₃ units, all at {fe_T} K and all for {fe_models}. "
        "**The bcc-Zr paths of every model except MatterSim are MatterSim's deciding coordinate, "
        "q = (1/3, 2/3, 0)** (role `ref`: a coordinate borrowed from MatterSim, not one of the "
        f"model's own), and ORB-v2's SrTiO₃ path is MatterSim's pattern at q = (1/2, 1/2, 1/2); "
        f"{n_ref} of the {len(m)} paths below are of this kind, and on "
        f"{n_nowell} of them the model's own curve has no well at all. The "
        f"{len(nw)} newly wrong units are {nw_txt}; the {int((~u['pbe_ok']).sum())} units that "
        f"stay wrong under PBE are {resid_txt}. Counts are descriptive: the units cluster by system "
        "(6 systems) and no test is attached. The label is a transition-temperature rule, so a "
        "PBE call that disagrees with it is not necessarily a PBE error. "
        "`scripts/dft_reference.py analyze` → "
        "`results/revision/dft/c3a_unit_calls.csv`.\n\n"
        + md(rows, ["System", "n units", "MLIP, same paths: correct", "PBE-backed: correct",
                    "Corrected by PBE", "Newly wrong"])
        + f"\n\nLower part: the well depth of each of the {len(m)} PBE paths ({n_decide} deciding, "
        f"{n_ref} reference, {n_mode} other screened modes), the path model's own E(Q) against PBE on the same {int(p['n_Q_dft'].max())} "
        "structures. Depth is −min E(Q) over the sampled amplitudes, zero when none is below "
        "E(0), and Q_min is the sampled amplitude at that minimum, so both are limited to the "
        f"{int(p['n_Q_dft'].max())}-point scan to 0.45 Å. *Ratio* is own depth over PBE depth. "
        f"The BaTiO₃ and KNbO₃ deciding paths of the four models other than ORB-v2 "
        f"({len(fe4)} paths: " + ", ".join(PRETTY[x] for x in sorted(set(fe4['path_model_own'])))
        + f") have own-model depths {fe4['ratio'].min():.3f} to {fe4['ratio'].max():.3f} of PBE "
        f"(median {fe4['ratio'].median():.3f}); with ORB-v2's {len(fe) - len(fe4)} paths the range is "
        f"{fe['ratio'].min():.3f} to {fe['ratio'].max():.3f}. **The PBE minimum is at the scan "
        f"edge (Q ≥ {EDGE_Q_A} Å) on {int(m['edge'].sum())} of {len(m)} paths, all "
        f"{' and '.join(edge_sys)} deciding paths**; there the sampled depth is a lower bound on "
        "the PBE depth and the ratio is not a well-depth comparison. On "
        f"{n_pbe_nowell} paths, all of role {' and '.join(nowell_roles)}, PBE has no well at all "
        "(no sampled energy below E(0)) where the model has one; the ratio is not defined "
        "there.\n\n"
        + md(drows, ["Path", "Role", "Own depth (meV)", "PBE depth (meV)", "Ratio own/PBE",
                     "Own Q_min (Å)", "PBE Q_min (Å)", "PBE minimum at scan edge"])
    )


def table_s20_c3b() -> str:
    """Referee 1.2: MLIP-versus-PBE forces and energies on SSCHA-sampled
    configurations, against a near-equilibrium baseline, from results/revision/dft/c3b_units.csv.

    One row per (system, owner model, T). The owner model is the one whose SSCHA ensemble
    produced the configurations; the five models are all scored on the same configurations. The
    five-model ranges are given with and without ORB-v2, as in Tables S9 and S15."""
    if not C3B_UNITS.exists():
        return ""
    c = pd.read_csv(C3B_UNITS)
    ss, base = c[c["set"] == "sscha"], c[c["set"] == "baseline"]
    order = {s: i for i, s in enumerate(C3A_ORDER)}

    def rng(x: pd.Series, nd: int) -> str:
        return f"{x.min():.{nd}f} to {x.max():.{nd}f}"

    def two(g: pd.DataFrame, col: str, nd: int) -> str:
        """Range over the five models; the range without ORB-v2 in brackets where it differs."""
        a, x = rng(g[col], nd), rng(g[g["eval_model"] != "orb_v2"][col], nd)
        return a if a == x else f"{a} ({x} without ORB-v2)"

    rows, facts = [], {}
    keys = ss[["system", "owner_model", "T"]].drop_duplicates()
    keys = keys.assign(o=keys["system"].map(order)).sort_values(["o", "T"]).drop(columns="o")
    for k in keys.itertuples(index=False):
        s = ss[(ss["system"] == k.system) & (ss["owner_model"] == k.owner_model) & (ss["T"] == k.T)]
        b = base[(base["system"] == k.system) & (base["owner_model"] == k.owner_model)]
        so, bo = s[s["is_owner"]].iloc[0], b[b["is_owner"]].iloc[0]
        ratio = so["rel_force_rmse"] / so["rel_force_rmse_baseline"]
        facts[k.system] = dict(so=so, bo=bo, ratio=ratio)
        rows.append([
            k.system, PRETTY[k.owner_model], f"{k.T:.0f}",
            f"{int(bo['n_configs'])} + {int(so['n_configs'])}",
            f"{bo['u_rms_mean_A']:.3f} / {so['u_rms_mean_A']:.3f}",
            f"{bo['rms_f_pbe']:.3f} / {so['rms_f_pbe']:.3f}",
            f"{bo['force_rmse']:.3f} / {so['force_rmse']:.3f}",
            f"{so['rel_force_rmse_baseline']:.3f}", f"{so['rel_force_rmse']:.3f}", f"{ratio:.2f}",
            f"{two(b, 'rel_force_rmse', 3)}; {two(s, 'rel_force_rmse', 3)}",
            f"{bo['e_err_rms_meV_per_atom']:.2f} / {bo['e_err_maxabs_meV_per_atom']:.2f}; "
            f"{so['e_err_rms_meV_per_atom']:.2f} / {so['e_err_maxabs_meV_per_atom']:.2f}",
            two(s, "e_err_rms_meV_per_atom", 1),
        ])
    sr, zr = facts["srtio3_cubic"], facts["zr_bcc"]
    others = [v["ratio"] for k, v in facts.items() if k != "srtio3_cubic"]
    nb = sorted({int(x) for x in base["n_configs"]})
    ns = sorted({int(x) for x in ss["n_configs"]})
    return (
        "**Table S20** MLIP errors on SSCHA-sampled configurations, scored against PBE. The "
        "configurations are drawn from the production-recipe seed-study SSCHA ensembles of Table S21 "
        "(`scripts/sscha_seed_study.py`), one temperature per system; the baseline is "
        "near-equilibrium rattled configurations of the same supercell. Per set there are "
        f"{' or '.join(str(x) for x in nb)} baseline and {' or '.join(str(x) for x in ns)} SSCHA "
        "configurations, so every figure rests on very few configurations and carries no "
        "uncertainty. The owner model is the one whose SSCHA ensemble produced the "
        "configurations; all five models are scored on the same configurations. Relative force "
        "RMSE is the RMSE over all Cartesian components divided by the RMS PBE force component; "
        "*Ratio* is the owner's relative force RMSE on the SSCHA configurations over its own "
        "baseline value. u_rms is the mean over configurations of the rms atomic displacement "
        "from the ideal supercell. Energy errors are per atom, taken relative to the undisplaced "
        "supercell in each code separately, and given as RMS / maximum absolute value over "
        "configurations; the baseline and SSCHA values are separated by a semicolon where two "
        "are given. Ranges run over the five evaluated models, with the range without ORB-v2 in "
        f"brackets where it differs. **The owner's relative force error is {min(others):.2f} to {max(others):.2f} "
        f"times its baseline for BaTiO₃, ZrO₂ and bcc-Zr, but {sr['ratio']:.2f} times for SrTiO₃ "
        f"at {sr['so']['T']:.0f} K** ({sr['so']['rel_force_rmse_baseline']:.3f} to "
        f"{sr['so']['rel_force_rmse']:.3f}), where the configurations are displaced by "
        f"{sr['so']['u_rms_mean_A']:.3f} Å rms and the owner's energy error is "
        f"{sr['so']['e_err_rms_meV_per_atom']:.1f} meV/atom rms and "
        f"{sr['so']['e_err_maxabs_meV_per_atom']:.1f} meV/atom at its worst; its absolute force "
        f"RMSE grows {sr['so']['force_rmse'] / sr['bo']['force_rmse']:.0f}-fold while the PBE "
        f"forces themselves grow {sr['so']['rms_f_pbe'] / sr['bo']['rms_f_pbe']:.0f}-fold. "
        f"**bcc-Zr's relative error has a PBE-force denominator of about "
        f"{zr['bo']['rms_f_pbe']:.2f} eV/Å on the baseline and is uninformative**: it goes from "
        f"{zr['so']['rel_force_rmse_baseline']:.3f} to {zr['so']['rel_force_rmse']:.3f} on the "
        f"SSCHA configurations although the owner's absolute force RMSE grows "
        f"{zr['so']['force_rmse'] / zr['bo']['force_rmse']:.1f}-fold, because the PBE force RMS "
        f"(the denominator) grows {zr['so']['rms_f_pbe'] / zr['bo']['rms_f_pbe']:.1f}-fold. "
        "`scripts/dft_reference.py analyze` → `results/revision/dft/c3b_units.csv`.\n\n"
        + md(rows, ["System", "Owner model", "T (K)", "Configs (baseline + SSCHA)",
                    "u_rms (Å), baseline / SSCHA", "PBE force RMS (eV/Å), baseline / SSCHA",
                    "Owner force RMSE (eV/Å), baseline / SSCHA",
                    "Owner relative force RMSE, baseline", "Owner relative force RMSE, SSCHA",
                    "Ratio",
                    "Relative force RMSE over five models, baseline; SSCHA",
                    "Owner energy error RMS / max (meV/atom), baseline; SSCHA",
                    "Energy error RMS over five models, SSCHA (meV/atom)"])
    )


# ------------------------------------ revision table S21 (SSCHA seed study, C1 and C5) ----

# C1: four seeds on each of these units (the order of PRESETS["revision"] in
# scripts/sscha_seed_study.py). C5 is every 3x3x3 unit JSON in the same directory (seed 0).
S21_C1 = [("batio3_cubic", "mace_mp0", 100.0), ("zro2_cubic", "mace_mp0", 100.0),
          ("zr_bcc", "mattersim", 50.0), ("srtio3_cubic", "mace_mp0", 600.0)]
# The converged-recipe grid (C1c) is a later table; its number is set here and nowhere else.
CONVERGED_TABLE = "Table S22"
STOP_LABEL = {"converged": "converged", "max_ka_cumulative": "max_ka (cumulative)",
              "kong_liu": "Kong-Liu", "other": "other"}
NEAR_START_THZ = 0.5      # a Hessian is "near its start" when the largest |Hessian - start| is below this
BOOT_NA_THZ = 1e-6        # a bootstrap SD below this carries no information to divide by


def _sig(x: float, nd: int = 2) -> str:
    """x to nd significant figures; 10^e form outside 1e-3 <= |x| < 1e2 (typographic signs)."""
    x = float(x)
    if x == 0.0:
        return "0"
    m, e = f"{abs(x):.{nd - 1}e}".split("e")
    e = int(e)
    sign = MINUS if x < 0 else ""
    if -3 <= e < 2:
        return sign + f"{abs(x):.{max(nd - 1 - e, 0)}f}"
    return f"{sign}{m} × 10{str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def _span(vals, fmt=str) -> str:
    """'lo to hi', or one number when they are equal."""
    lo, hi = min(vals), max(vals)
    return fmt(lo) if lo == hi else f"{fmt(lo)} to {fmt(hi)}"


def _and(items) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _seed_docs() -> dict:
    """The real (not dry-run) unit JSONs of the seed study, keyed (system, model, T, supercell)."""
    docs = {}
    for p in sorted(SEED_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("schema") != "sscha_seed_study/v1" or d.get("dry_run"):
            continue
        u = d["unit"]
        docs[(u["system"], u["model"], float(u["T"]), tuple(int(x) for x in u["supercell"]))] = d
    return docs


def _ok_seeds(d: dict) -> list:
    """[(seed, record)] for the seeds that finished, in seed order."""
    return [(int(k), d["seeds"][k]) for k in sorted(d["seeds"], key=int)
            if d["seeds"][k].get("status") == "ok"]


def _run_facts(s: dict) -> dict:
    """The numbers one finished seed contributes to Table S21, all read from its record.

    Refuses (rather than printing a caption the data no longer support) if the recorded gradient
    error is not the library's constant placeholder of 0.433 per primitive-cell atom."""
    r, sc = s["relax"], s["stopping_criteria"]
    st, pops, mf = r["steps"], r["populations"], sc["meaningful_factor"]
    errs = {round(x["gc_err"], 9) for x in st}
    nat = s["dyn_meta"]["nat_prim"]
    if len(errs) != 1 or abs(next(iter(errs)) - nat * float(np.sqrt(3)) / 4) > 1e-8:
        raise SystemExit("Table S21: the recorded gradient error is no longer the constant "
                         "0.433-per-atom placeholder; rewrite the caption")
    reasons: dict = {}
    for p in pops:
        reasons[p["stop_reason"]] = reasons.get(p["stop_reason"], 0) + 1
    err, gw = st[-1]["gc_err"], st[-1]
    return dict(
        n_pop=r["n_populations"], moved=r["n_populations_moving_dyn"], n_cfg=sc["n_configs_per_population"],
        steps=r["n_steps_total"], kept=r["n_steps_kept_total"], cap=sc["max_ka"],
        converged=bool(r["converged"]),
        stop="; ".join(f"{STOP_LABEL.get(k, k)} ×{n}" for k, n in reasons.items()),
        gc0=st[0]["gc"], gc1=st[-1]["gc"], err=err, ratio=st[-1]["gc"] / (err * mf),
        thr=err * mf, gw_ok=bool(gw["gw"] <= max(gw["gw_err"] * mf, sc["abs_conv_thr"])),
        rel_max=max([p["rel_change_vs_prev_end"] for p in pops if p["pop"] >= 2], default=None),
        harm=s["harmonic_pre_fpd"]["min_nonac_thz"], start=s["start_post_fpd"]["min_nonac_thz"],
        aux=s["final_aux"]["min_nonac_thz"], hess=s["hessian"]["min_nonac_thz"],
        boot=s["bootstrap"]["std_thz"], boot_B=s["bootstrap"]["B"], n_hess=s["bootstrap"]["n_configs"])


def table_s21_sscha_seeds() -> str:
    """Referee 1.4 (plan items C1 and C5): what the SSCHA numbers never carried.

    Sample sizes, populations, steps against the cap, stopping reason, gradient history,
    Hessian uncertainty, and the four-seed test on units beyond bcc-Zr (BaTiO3, ZrO2, SrTiO3 at
    600 K, bcc-Zr/MatterSim at 50 K), with the 3x3x3 re-measurement of bcc-Zr beneath. The
    production recipe, re-run once per seed: no C1 seed converges, and the caption says so. Every
    number is read from results/revision/sscha_seeds/*.json (scripts/sscha_seed_study.py
    --preset revision); the converged-recipe values belong to a later table."""
    if not SEED_DIR.exists():
        return ""
    docs = _seed_docs()
    c1 = [docs.get((s, m, T, (2, 2, 2))) for s, m, T in S21_C1]
    c5 = [d for k, d in sorted(docs.items(), key=lambda kv: (kv[0][1], kv[0][2])) if k[3] == (3, 3, 3)]
    if None in c1 or not c5:
        raise SystemExit("Table S21: results/revision/sscha_seeds/ lacks a C1 or a C5 unit JSON")
    recipe = c1[0]["recipe"]
    if any(d["recipe"] != recipe for d in c1 + c5):
        raise SystemExit("Table S21: the unit JSONs were computed with different recipes")

    def cell(d: dict) -> str:
        return "×".join(str(x) for x in d["unit"]["supercell"])

    def pops_txt(f: dict) -> str:
        return f"{f['n_pop']} × {f['n_cfg']} ({f['moved']})"

    def steps_txt(f: dict) -> str:
        return f"{f['steps']} ({f['kept']}) / {f['cap']}"

    # ---- upper part: one row per (unit, seed)
    facts = {d["unit_tag"]: [(seed, _run_facts(s)) for seed, s in _ok_seeds(d)] for d in c1 + c5}
    seeds_of = {d["unit_tag"]: dict(_ok_seeds(d)) for d in c1 + c5}
    up = []
    for d in c1:
        u = d["unit"]
        for seed, f in facts[d["unit_tag"]]:
            lc = seeds_of[d["unit_tag"]][seed].get("ledger_comparison") or {}
            ledger = (f"{signed(lc['ledger_thz'], 3)}; Δ {lc['abs_diff_thz']:.3f} "
                      f"({'reproduced' if lc['reproduces_ledger'] else 'not reproduced'})"
                      if lc.get("ledger_thz") is not None else "--")
            up.append([SHORT[u["system"]], PRETTY[u["model"]], f"{u['T']:.0f}", cell(d), seed,
                       pops_txt(f), steps_txt(f), "yes" if f["converged"] else "no", f["stop"],
                       f"{_sig(f['gc0'])} → {_sig(f['gc1'])}", f"{f['err']:.3f}", _sig(f["ratio"]),
                       signed(f["start"], 3), signed(f["hess"], 3), _sig(f["boot"]), ledger])

    # ---- middle part: one row per C1 unit
    mid, near, far, eq_aux, sames, repro, notrepro = [], [], [], [], [], [], []
    for d in c1:
        u, tag = d["unit"], d["unit_tag"]
        fs = [f for _, f in facts[tag]]
        h, st, ax, hm = (np.array([f[k] for f in fs]) for k in ("hess", "start", "aux", "harm"))
        bs = np.array([f["boot"] for f in fs])
        tol, n = d["recipe"]["imag_tol_thz"], len(fs)
        n_stable = int((h >= tol).sum())
        sd, med = float(h.std(ddof=1)), float(np.median(bs))
        dstart = float(np.abs(h - st).max())
        name = SHORT[u["system"]]
        (near if dstart < NEAR_START_THZ else far).append(
            (name, u["T"], dstart, float(st.mean()), float(h.min()), float(h.max())))
        if float(np.abs(h - ax).max()) < 1e-9:
            eq_aux.append((name, u["model"], u["T"], float(np.abs(h - ax).max()), bs, sd))
        sames.append((name, n_stable in (0, n)))
        lc = seeds_of[tag][0].get("ledger_comparison") or {}
        if lc.get("ledger_thz") is not None:
            (repro if lc["reproduces_ledger"] else notrepro).append(
                (name, u["T"], lc["this_thz"], lc["ledger_thz"], float(h.min()), float(h.max())))
        mid.append([name, PRETTY[u["model"]], f"{u['T']:.0f}", cell(d),
                    signed(float(hm.mean()), 3), signed(float(st.mean()), 3),
                    f"{signed(float(ax.min()), 3)} to {signed(float(ax.max()), 3)}",
                    f"{signed(float(h.min()), 3)} to {signed(float(h.max()), 3)}",
                    signed(float(h.mean()), 3), _sig(sd), _sig(med),
                    _sig(sd / med) if med > BOOT_NA_THZ else "n/a", _sig(dstart),
                    f"{sum(f['converged'] for f in fs)} of {n}",
                    (f"stable, {n} of {n} seeds" if n_stable == n else
                     f"unstable, {n} of {n} seeds" if n_stable == 0 else
                     f"mixed, {n_stable} of {n} stable")])

    # ---- lower part: C5, one row per (model, T)
    low, c5f = [], []
    for d in c5:
        u = d["unit"]
        f = facts[d["unit_tag"]][0][1]
        two = next((o for o in d["ledger"]["other_supercells"] if o["supercell"] == [2, 2, 2]), None)
        t2 = two["min_eff_freq_thz"] if two else None
        c5f.append((u, f, t2))
        low.append([PRETTY[u["model"]], f"{u['T']:.0f}", signed(t2, 3) if two else "--",
                    signed(f["hess"], 3), _sig(f["boot"]),
                    signed(f["hess"] - t2, 3) if two else "--",
                    signed(f["start"], 3), signed(f["harm"], 3), pops_txt(f), steps_txt(f),
                    "yes" if f["converged"] else "no", f["stop"],
                    f"{_sig(f['gc0'])} → {_sig(f['gc1'])}", _sig(f["ratio"])])

    # ---- caption facts, read from the data
    f1 = [f for tag in (d["unit_tag"] for d in c1) for _, f in facts[tag]]
    f5 = [f for u, f, _ in c5f]
    unconv = [f for f in f1 + f5 if not f["converged"]]
    n_seeds = len(facts[c1[0]["unit_tag"]])
    conv5 = [f"{PRETTY[u['model']]} {u['T']:.0f} K" for u, f, _ in c5f if f["converged"]]
    B, nh = f1[0]["boot_B"], f1[0]["n_hess"]
    gfall = [f["gc0"] / f["gc1"] for f in f1]

    near_txt = (f"{_sig(max(x[2] for x in near))} THz or less for "
                + _and(x[0] for x in near))
    far_txt = "; ".join(
        f"for {x[0]} at {x[1]:.0f} K it is {_sig(x[2])} THz (start {signed(x[3], 2)}, Hessian "
        f"{signed(x[5], 1)} to {signed(x[4], 1)}), so there the finite-temperature correction, "
        "not the starting matrix, sets the lowest frequency" for x in far)
    eq_txt = "".join(
        f"For {x[0]}/{PRETTY[x[1]]} at {x[2]:.0f} K the lowest Hessian frequency is the final auxiliary "
        f"matrix's own in every seed ({'identical in the recorded digits' if x[3] == 0 else 'largest difference ' + _sig(x[3], 1) + ' THz'}) "
        "and the bootstrap SD is "
        f"{_span(list(x[4]), _sig)}, so the bootstrap says nothing there and the {_sig(x[5])} THz "
        "seed spread comes entirely from the relaxation. " for x in eq_aux)
    repro_txt = (
        "*Ledger* is the deposited canonical SSCHA value for the unit, compared for seed 0 only "
        f"(tolerance {seeds_of[c1[0]['unit_tag']][0]['ledger_comparison']['tol_thz']:g} THz): "
        "seed 0 reproduces it for "
        + _and(x[0] for x in repro)
        + "".join(f", but not for {x[0]} at {x[1]:.0f} K, where seed 0 gives {signed(x[2], 2)} THz "
                  f"against {signed(x[3], 2)} and the ledger value "
                  + ("lies inside" if x[4] <= x[3] <= x[5] else "lies outside")
                  + f" the four-seed range ({signed(x[4], 2)} to {signed(x[5], 2)})" for x in notrepro)
        + ".")
    ma = [(u, f, t2) for u, f, t2 in c5f if u["model"] == "mace_mp0" and f["converged"]]
    ms = [(u, f, t2) for u, f, t2 in c5f if u["model"] == "mattersim"]
    ma_txt = ""
    if ma:
        harm_is_start = all(abs(f["harm"] - f["start"]) < 1e-6 for _, f, _ in ma)
        ma_T = _and(f"{u['T']:.0f}" for u, _, _ in ma) + " K"
        ma_txt = (
            f"The {len(ma)} runs that the library's test calls converged are MACE-MP-0 at {ma_T}: "
            + ("its harmonic matrix is already positive there, so the start is the harmonic matrix, and "
               if harm_is_start else "")
            + f"one population ({_and(str(f['steps']) for _, f, _ in ma)} steps) takes the gradient from "
            f"{_and(_sig(f['gc0']) for _, f, _ in ma)} to {_and(_sig(f['gc1']) for _, f, _ in ma)}, "
            f"below the threshold of {_sig(ma[0][1]['thr'])}. ")
    ms_txt = ""
    if ms:
        flips = [(u["T"], t2, f["hess"]) for u, f, t2 in ms if t2 is not None and (t2 >= 0) != (f["hess"] >= 0)]
        ms_txt = (
            f"The {len(ms)} MatterSim runs end at the cumulative cap as the seed-study units do (final gradient "
            f"{_span([f['ratio'] for _, f, _ in ms], _sig)} times its threshold; the Hessian lies "
            f"{_span([abs(f['hess'] - f['aux']) for _, f, _ in ms], _sig)} THz from the final auxiliary "
            "matrix). "
            + "".join(f"The MatterSim value changes sign with cell size at {T:.0f} K "
                      f"({signed(t2, 2)} THz in 2×2×2, {signed(h3, 2)} in 3×3×3). " for T, t2, h3 in flips))
    diffs = {m: [f["hess"] - t2 for u, f, t2 in c5f if u["model"] == m and t2 is not None]
             for m in ("mace_mp0", "mattersim")}

    caption = (
        "**Table S21** SSCHA with the production recipe, re-run once per seed "
        "(`scripts/sscha_seed_study.py --preset revision`). "
        f"**This is the production recipe, and no seed converged: {sum(f['converged'] for f in f1)} of "
        f"{len(f1)} runs ({n_seeds} seeds on each of {len(c1)} units) satisfy the minimiser's own "
        f"stopping test.** Converged values are in the converged-recipe grid ({CONVERGED_TABLE}), not here. "
        f"Recipe: {recipe['n_configs']} configurations per population, at most {recipe['max_pop']} "
        f"populations, `max_ka` = {recipe['max_ka']}, `meaningful_factor` = {recipe['meaningful_factor']:g}, "
        f"then a separate Hessian ensemble of {nh} configurations ({nh // 2} antithetic pairs) at the final "
        "auxiliary matrix, without the fourth-order term. Seed 0 replays the production computation; the "
        "other seeds change only the random seed. *Populations × configs (moved)* is the populations drawn, "
        "the configurations in each and, in brackets, the populations whose kept steps changed the auxiliary "
        "matrix; *Steps taken (kept) / cap* counts the minimiser's steps over all populations, those not "
        "discarded when a population stops unconverged, and `max_ka`. "
        f"**`max_ka` is a cumulative cap in python-sscha 1.6.1** (SchaMinimizer.py:1370 compares it with "
        f"the step history accumulated over populations). In all {len(unconv)} unconverged runs (these "
        f"{len(f1)} and the {len(f5) - len(conv5)} MatterSim runs of the lower part) the cap ended "
        f"population 1 after {_span([f['kept'] for f in unconv])} kept steps and each of populations 2 to "
        f"{_span([f['n_pop'] for f in unconv])} took one step and discarded it: the auxiliary matrix changes "
        f"by less than {_sig(max(f['rel_max'] for f in unconv), 1)} (relative) in populations 2 and later, so only "
        "the first population moved it. "
        "**The recorded gradient error is the library's placeholder.** python-sscha passes a constant in "
        "place of the stochastic error of the gradient (Ensemble.py:2657): the recorded value is the same "
        "at every step of every run of a unit and equals 0.433 per primitive-cell atom, so *Recorded gc "
        "error* is not a measurement. The convergence threshold is that value times `meaningful_factor`, "
        f"and *Final gc ÷ threshold* is how far the last gradient sits above it: {_span([f['ratio'] for f in f1], _sig)} "
        f"on the seed-study units, while the gradient falls by a factor of only {_span(gfall, lambda v: f'{v:.1f}')} "
        "from the first step to the last. "
        + ("The structure gradient is at or below its threshold at the last step of every run, so the "
           "dynamical-matrix gradient alone is unconverged. " if all(f["gw_ok"] for f in f1 + f5) else "")
        + f"**The Hessian lies close to the start on {len(near)} of {len(c1)} units.** The start is the "
        "matrix after ForcePositiveDefinite (FPD), where the relaxation begins; the largest difference "
        f"between it and the lowest Hessian frequency is {near_txt}"
        + (f"; {far_txt}" if far else "") + ". "
        f"*Bootstrap SD* is the standard deviation of the lowest non-acoustic Hessian frequency over {B} "
        "resamples of the Hessian ensemble's antithetic pairs at the fixed final matrix. It measures the "
        "finite size of that ensemble only, whereas the seed spread (middle part) also contains the "
        "unconverged relaxation. " + eq_txt + repro_txt
        + " `results/revision/sscha_seeds/*.json`.")

    all_same = all(ok for _, ok in sames)
    return (
        caption + "\n\n"
        + md(up, ["System", "Model", "T (K)", "Cell", "Seed", "Populations × configs (moved)",
                  "Steps taken (kept) / cap", "Converged", "Stop reason", "gc, first → final step",
                  "Recorded gc error (placeholder)", "Final gc ÷ threshold",
                  "Start, after FPD (THz)", "Hessian lowest (THz)", "Bootstrap SD (THz)",
                  "Ledger (THz), seed 0"])
        + "\n\nMiddle part: the seed spread per unit. *SD* is the sample standard deviation (n − 1) of "
        "the lowest Hessian frequency over the seeds and *Seed SD ÷ bootstrap SD* its ratio to the "
        f"median bootstrap SD (n/a where that is below {_sig(BOOT_NA_THZ, 1)} THz). *Call* is the sign rule "
        "used throughout: stable if the lowest non-acoustic frequency is at or above "
        f"{signed(c1[0]['recipe']['imag_tol_thz'], 1)} THz. "
        + ("All seeds give the same call on every unit. " if all_same else
           "The seeds disagree on the call for "
           + ", ".join(n for n, ok in sames if not ok) + ". ")
        + "None converged.\n\n"
        + md(mid, ["System", "Model", "T (K)", "Cell", "Harmonic, before FPD (THz)",
                   "Start, after FPD (THz)", "Final auxiliary matrix, min to max (THz)",
                   "Hessian, min to max (THz)", "Hessian mean (THz)", "SD over seeds (THz)",
                   "Median bootstrap SD (THz)", "Seed SD ÷ bootstrap SD", "Largest Hessian − start difference (THz)",
                   "Seeds converged", "Call"])
        + "\n\nLower part: bcc-Zr cell size. The 3×3×3 cell was run once (seed 0) with the "
        "same production recipe in the current environments; the 2×2×2 value is the deposited canonical "
        "ledger row, not a re-run. The 3×3×3 minus 2×2×2 difference is "
        + " and ".join(f"{_span(v, lambda t: signed(t, 2))} THz for {PRETTY[m]}"
                       for m, v in diffs.items() if v)
        + ". " + ma_txt + ms_txt
        + f"These are the production recipe's values and are not converged; {CONVERGED_TABLE} has the "
        "converged ones.\n\n"
        + md(low, ["Model", "T (K)", "2×2×2, ledger (THz)", "3×3×3 Hessian (THz)", "Bootstrap SD (THz)",
                   "3×3×3 − 2×2×2 (THz)", "Start, after FPD (THz)", "Harmonic, before FPD (THz)",
                   "Populations × configs (moved)", "Steps taken (kept) / cap", "Converged",
                   "Stop reason", "gc, first → final step", "Final gc ÷ threshold"])
    )


# ------------------------- revision table S22 (converged-recipe SSCHA grid, plan item C1c) ----

def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _grid_compare():
    """scripts/grid_compare.py as a module (imported here, not at the top, so that the build does
    not depend on it until the grid's outputs exist)."""
    sd = str(REPO / "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import grid_compare
    return grid_compare


def table_s22_converged_grid(compare_path: Path = GRID_COMPARE, summary_path: Path = GRID_SUMMARY,
                             allow_dry_run: bool = False) -> str:
    """Referee 1.4: SSCHA with the converged recipe on the units behind the §3.3
    claims, beside the production numbers on the same units.

    Every number is read from results/revision/grid_compare.json, which ``scripts/grid_compare.py``
    writes from the grid's summary.csv (``scripts/sscha_seed_study.py --preset grid --summarize``)
    and the converged-mode JSONs of SrTiO3/MACE-MP-0 at 600 and 900 K. Left out, with a note on
    stderr, while that file does not exist. Refuses (rather than render stale or mocked numbers)
    a file made from a --dry-run grid, from another summary.csv or another ledger than the ones on
    disk, or whose production numbers no longer reproduce results/stats_hardening.json.
    ``allow_dry_run`` is for testing the plumbing only."""
    if not compare_path.exists():
        print(f"Table S22 skipped: {_rel(compare_path)} does not exist (run scripts/sscha_seed_study.py "
              "--preset grid, then --preset grid --summarize, then scripts/grid_compare.py)",
              file=sys.stderr)
        return ""
    cmp = json.loads(compare_path.read_text(encoding="utf-8"))
    m = cmp["meta"]
    if m["dry_run"] and not allow_dry_run:
        raise SystemExit(f"Table S22: {_rel(compare_path)} was made from a --dry-run grid (mocked "
                         "SSCHA values); rerun scripts/grid_compare.py on the real grid")
    if not summary_path.exists():
        raise SystemExit(f"Table S22: {_rel(compare_path)} exists but {_rel(summary_path)} does not")
    if _sha256(summary_path) != m["summary_csv"]["sha256"]:
        raise SystemExit(f"Table S22: {_rel(summary_path)} changed since scripts/grid_compare.py "
                         "read it; rerun scripts/grid_compare.py")
    if _sha256(LEDGER) != m["ledger"]["sha256"]:
        raise SystemExit("Table S22: results/ledger.parquet changed since scripts/grid_compare.py "
                         "read it; rerun scripts/grid_compare.py")
    for name in ("production_reproduces_stats_hardening", "grid_vs_ledger"):
        if not m[name]["all_ok"]:
            raise SystemExit(f"Table S22: grid_compare.json records a failed check ({name}); "
                             "rerun scripts/grid_compare.py and read its output")
    gc = _grid_compare()
    rs = cmp["run_status"]
    cv = cmp["variants"]["converged_only"]
    cov, fo = cv["coverage"], cv["fs57_outcomes"]
    R = m["recipe"]
    docs = _seed_docs() if SEED_DIR.exists() else {}
    P = next(iter(docs.values()))["recipe"] if docs else None

    # ---- tables
    def pretty_col(rows, col):
        for r in rows:
            r[col] = PRETTY.get(r[col], r[col])
        return rows

    h_claim, claim = gc.claim_table(cmp, "converged_only", typographic=True)
    h_cov, covrows = gc.coverage_table(cmp, "converged_only", typographic=True)
    h_stat, stat = gc.status_table(cmp)
    h_chg, chg = gc.changed_table(cmp, typographic=True)
    stat = [r for r in pretty_col([list(r) for r in stat], 0)]
    chg = pretty_col([list(r) for r in chg], 1)

    # the SrTiO3 units above T_c that have a converged value, beside production
    so_c = (cv["converged_all"]["iv"].get("all_models") or {}).get("srtio3_above_tc")
    so_p = cmp["production_full"]["iv"]["all_models"]["srtio3_above_tc"]
    so_rows, so_txt = [], ""
    if so_c:
        prod = {(u["model"], u["T"]): u for u in so_p["units"]}
        for u in sorted(so_c["units"], key=lambda u: (u["T"], u["model"])):
            p = prod.get((u["model"], u["T"]))
            so_rows.append([PRETTY.get(u["model"], u["model"]), f"{u['T']:.0f}",
                            signed(p["sscha_min_eff_freq_thz"], 2) if p else "--",
                            signed(u["sscha_min_eff_freq_thz"], 2),
                            "unstable" if p and p["false_unstable"] else "stable" if p else "--",
                            "unstable" if u["false_unstable"] else "stable"])
        so_txt = (
            f"SrTiO₃ above its {so_c['transition_T_K']:.0f} K transition (the units of Table S17's "
            "second part that the grid or the converged mode evaluated; the label is stable at "
            f"every one): the converged recipe calls {so_c['false_unstable']['short']} of them "
            "unstable.\n\n"
            + md(so_rows, ["Model", "T (K)", "Production SSCHA minimum (THz)",
                           "Converged Hessian minimum (THz)", "Production call", "Converged call"]))

    # ---- caption facts, read from the data
    n, ok, conv = rs["n_planned_units"], rs["n_ok"], rs["n_converged"]
    stops = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(rs["stop_reasons_of_ok"].items()))
    used = [e for e in m["c1c_extra_units"] if e["used"]]
    by_unit: dict = {}
    for e in used:
        by_unit.setdefault((e["system"], e["model"]), []).append(f"{e['T_K']:.0f}")
    gsum = pd.read_csv(summary_path)
    gsum = gsum[gsum["start"] == "A"]
    t_nb = sorted({int(t) for t in gsum.loc[gsum["family"] != "bcc", "T_K"]})
    t_bcc = sorted({int(t) for t in gsum.loc[gsum["family"] == "bcc", "T_K"]})
    extra_txt = (
        f"The grid covers the non-bcc systems at {_and(str(t) for t in t_nb)} K in the 2×2×2 cell "
        f"and the bcc metals at {_and(str(t) for t in t_bcc)} K in the 3×3×3 cell"
        + ("; " + _and(f"{SHORT.get(s, s)} with {PRETTY.get(mo, mo)} at {' and '.join(Ts)} K"
                       for (s, mo), Ts in by_unit.items())
           + " come from the converged mode (`results/revision/sscha_converged/`) with the same "
             "recipe" if used else ""))
    # fresh-ensemble gradient check and bootstrap resolution of the converged units
    okc = gsum[(gsum["status"] == "ok") & (gsum["converged"].astype(str) == "True")]
    fresh_bad = []
    for t in okc["unit_tag"]:
        d = json.loads((summary_path.parent / f"{t}_startA.json").read_text(encoding="utf-8"))
        if not d["run"]["fresh_gradient_check"]["consistent_with_minimum"]:
            r = okc[okc["unit_tag"] == t].iloc[0]
            fresh_bad.append(f"{SYSNAME.get(r.system, r.system)}/{PRETTY.get(r.model, r.model)} "
                             f"{r.T_K:.0f} K")
    fs_conv = okc[(okc["family"] != "bcc") & (okc["gt_stable"].astype(str) == "False")
                  & (okc["stable_call"].astype(str) == "True")]
    weak = fs_conv[fs_conv["hessian_min_thz"] < 2 * fs_conv["boot_sd_thz"]]
    weak_txt = _and(f"{SYSNAME.get(r.system, r.system)}/{PRETTY.get(r.model, r.model)} at "
                    f"{r.T_K:.0f} K ({signed(r.hessian_min_thz, 2)} ± {r.boot_sd_thz:.2f} THz)"
                    for r in weak.itertuples())
    fresh_txt = (
        f"**Fresh-ensemble check.** At the final matrix of each converged unit the gradient was "
        f"recomputed on an independent ensemble of {R['n_hessian']} configurations; it is consistent "
        f"with a minimum on {len(okc) - len(fresh_bad)} of the {len(okc)} converged units and not on "
        f"{len(fresh_bad)} ({_and(fresh_bad)}), so *converged* means the library's stopping test, "
        f"not an independent verification. None of the {len(fs_conv)} converged false-stables is "
        "among them"
        + (f"; among the false-stables, {weak_txt} lie{'s' if len(weak) == 1 else ''} within two "
           "bootstrap standard deviations of zero" if len(weak) else "")
        + ". ")
    miss_T = cov["i"]["not_in_grid_by_T"]
    miss_txt = (", ".join(f"{k} at {t} K" for t, k in miss_T.items()) if miss_T else "")
    still = fo["outcomes"].get("still_false_stable", 0)
    now = fo["outcomes"].get("now_unstable", 0)
    rest = cov["i"]["in_grid_plan"] - still - now
    rest_states = ", ".join(f"{gc.STATE_WORDS.get(k, k)} {v}" for k, v in sorted(fo["outcomes"].items())
                            if k not in ("still_false_stable", "now_unstable", "not_in_grid"))
    n_prod_failed = rs["failed_in_production"]
    wall = len(rs["wall_cap_timeouts"])

    prod_txt = ""
    if P:
        prod_txt = (f" Production: {P['n_configs']} configurations per population, at most "
                    f"{P['max_pop']} populations, `max_ka` = {P['max_ka']} cumulative, "
                    f"`meaningful_factor` = {P['meaningful_factor']:g} against the placeholder error, "
                    f"a Hessian ensemble of {P['n_hessian']}.")
    caption = (
        "**Table S22** SSCHA with the converged recipe on the units behind the §3.3 claims, beside "
        "the production numbers on the same units ("
        "`scripts/sscha_seed_study.py --preset grid`, compared with the production ledger by "
        "`scripts/grid_compare.py`). "
        f"**Recipe:** {R['n_configs']} configurations per population, at most {R['max_pop']} "
        f"populations and at most {R['max_steps_per_pop']} minimiser steps per population (a cap per "
        f"population, where the production `max_ka` is cumulative over populations, Table S21), "
        f"`meaningful_factor` = {R['meaningful_factor']:g} applied to the real stochastic error of "
        "the gradient (the library's serial estimate, in place of the constant placeholder of "
        f"Table S21; it is set to zero wherever KL/N < {R['converge_min_kl_ratio']:g}, so convergence "
        f"cannot be declared at a statistically invalid point), Kong-Liu ratio {R['kong_liu_ratio']:g}, "
        f"then a Hessian ensemble of {R['n_hessian']} configurations with "
        f"{' or '.join(str(x) for x in m['n_boot'])} bootstrap resamples, and a "
        f"{m['unit_timeout_s']:g} s wall cap on each relaxation." + prod_txt + " "
        "**Start A only** (the production ForcePositiveDefinite start): start dependence is the "
        "six-unit start-A against start-B study (§3.5), not repeated across the grid. A call is the "
        f"sign rule used throughout: stable iff the lowest free-energy-Hessian frequency is at or "
        f"above {signed(m['imag_tol_thz'], 1)} THz. "
        f"**Run status:** {n} planned units, {ok} finished, {conv} of them "
        + (f"({conv / ok:.0%} of those finished) " if ok else "")
        + "met the library's stopping test"
        + (f" (stop reasons: {stops})" if ok else "") + f"; {wall} ended at the wall cap, "
        f"{len(rs['failed'])} failed, {len(rs['not_finished'])} are not finished, "
        f"{len(rs['blowups'])} returned a blow-up (|f| > 50 THz)"
        + (f"; production returned no value for {n_prod_failed} of the {n} units" if n_prod_failed else "")
        + ". Only runs that met the stopping test enter the claim table; the others are listed "
        "under *Planned, not evaluated* in the coverage table. "
        "**Coverage.** "
        + (f"Of the {cov['i']['n']} units where SSCHA calls a phase stable against an unstable label "
           f"in Table S16, {cov['i']['in_grid_plan']} are in the grid"
           + (f" and {cov['i']['n'] - cov['i']['in_grid_plan']} ({miss_txt}) are not" if miss_txt else "")
           + f"; of the {cov['i']['in_grid_plan']} in the grid, {still} are still called stable by the "
           f"converged recipe (the screen calls {fo['still_false_stable_screen_unstable']['short']} of "
           f"them unstable) and {now} are called unstable"
           + (f", and {rest} have no converged value ({rest_states})" if rest else "") + ". ")
        + extra_txt + ". " + fresh_txt +
        "**The bcc cell is 3×3×3 in the grid and 2×2×2 in production**, so on bcc a change of call "
        "mixes the cell change with the convergence (Table S21, lower part, measures the cell change "
        "at the production recipe), and the bcc agreement below compares the screen with a "
        "different cell's SSCHA. "
        "**How the columns are made.** Each production claim is recomputed with the function that "
        "produced it (`stats_hardening.criterion_blindness` for claim (i), `analysis.displacive_recall` "
        "for (ii), `stats_hardening.bcc_agreement` for (iii), `stats_hardening.sscha_high_t` for (iv); "
        "Tables S16 and S17 and §3.3), so the unit sets, the ORB-v2 split and the blow-up rules are "
        "the production ones: SSCHA blow-ups leave the denominator in (ii), where they are counted "
        "beside, and stay in (i), (iii) and (iv). The first column is the production number on its "
        "full set, recomputed from the ledger and checked against `results/stats_hardening.json`; "
        "the second is the production number on exactly the units the grid evaluated; the third is "
        "the converged value on those same units; the fourth is the converged value on every unit "
        f"evaluated, which includes units that production never returned ({n_prod_failed} in the "
        "plan). A count of 0/0 means no unit of that kind was evaluated. "
        "`results/revision/grid_compare.json`.")

    ch = cmp["changed_calls"]
    a, c = ch["all_changed"], ch["converged_no_blowup_changed"]
    nb = ch["bcc_3x3x3_vs_2x2x2"]["n"]
    chg_txt = (
        f"Calls that differ from production, with the old and new Hessian minimum: {a['n']} of "
        f"the {ch['n_compared_ok']} finished units that have a production value change their call "
        f"(stable → unstable {a['S_to_U']}, unstable → stable {a['U_to_S']}, whatever stopped the "
        f"relaxation); of the {ch['n_compared_converged']} that met the stopping test, {c['n']} "
        f"change without a blow-up on either side ({ch['same_cell']['n']} in the production cell and "
        f"{nb} bcc unit{'s' if nb != 1 else ''} where the cell changes as well), and on the non-bcc "
        f"units {ch['nonbcc_corrected_vs_label']} of those change from wrong to right against the "
        f"label and {ch['nonbcc_worsened_vs_label']} from right to wrong. *n/s*: the bcc label is the "
        "thermodynamic one and is not scored against a dynamical call (§3.3).")
    return (
        caption + "\n\n"
        + md(claim, h_claim)
        + "\n\nCoverage of the production units behind each claim: how many are in the grid, how many "
        "have a converged value, which are not in the grid (by temperature) and which were planned "
        "but have no converged value.\n\n"
        + md(covrows, h_cov)
        + "\n\nRun status by model.\n\n"
        + md(stat, h_stat)
        + ("\n\n" + so_txt if so_txt else "")
        + "\n\n" + chg_txt + "\n\n"
        + (md(chg, h_chg) if chg else "No call changes.")
    )


# ------------------------------- revision table S23 (checks on the persistent PBE errors) ----

DFT_CHECKS = REPO / "results" / "revision" / "dft_checks"


def table_s23_dft_checks() -> str:
    """Referee 1.1: checks on the errors that persist on PBE (k-point and cutoff convergence,
    PBEsol on the same structures, the PBE lattice). Every number is read from the outputs of
    ``scripts/dft_checks.py analyze-checks`` under results/revision/dft_checks/."""
    sp = DFT_CHECKS / "summary.json"
    if not sp.exists():
        print("Table S23 skipped: results/revision/dft_checks/summary.json does not exist",
              file=sys.stderr)
        return ""
    chk = json.loads(sp.read_text(encoding="utf-8"))
    order = ["batio3_cubic", "knbo3_cubic", "cssnbr3_cubic"]
    names = {"batio3_cubic": "BaTiO₃", "knbo3_cubic": "KNbO₃", "cssnbr3_cubic": "CsSnBr₃"}

    def pct(x: float) -> str:
        v = 100 * x
        return ("−" if v < 0 else "+") + f"{abs(v):.1f} %" if abs(v) >= 0.05 else "−0.0 %" if v < 0 else "+0.0 %"

    def qtxt(stem: str) -> str:
        q, b = stem.split("_q")[1].split("-b")
        parts = [x.replace("d", "/") for x in q.split("-")]
        return f"q = ({', '.join(parts)}), band {b}"

    # upper part: convergence
    conv = {c["system"]: c for c in chk["conv"]["paths"]}
    crow, n_T = [], None
    for sname in order:
        c = conv[sname]
        v = c["variants"]
        n_T = len(chk["conv"]["acceptance"].split("[")[1].split("]")[0].split(","))
        changed = sum(int(v[k]["n_T_call_changed"]) for k in ("k", "e", "ke"))
        edge_all = all(v[k]["min_at_edge"] for k in ("prod", "k", "e", "ke"))
        failed = c.get("failed_variants") or []
        crow.append([names[sname], qtxt(c["stem"]), f"{v['prod']['depth_meV']:.1f}",
                     pct(v["k"]["depth_rel_change"]), pct(v["e"]["depth_rel_change"]),
                     pct(v["ke"]["depth_rel_change"]), str(changed),
                     "yes, all variants" if edge_all else ("yes" if v["prod"]["min_at_edge"] else "no"),
                     "met" if c["converged"] else f"missed (depth; {', '.join(failed)})"])
    # middle part: PBEsol
    xu = pd.read_csv(DFT_CHECKS / "xc_unit_calls.csv")
    xu = xu[xu["in_ladder"].astype(bool) & xu["unit_complete"].astype(bool)]
    xp = pd.read_csv(DFT_CHECKS / "xc_paths.csv")
    xrow = []
    tot = [0, 0, 0, 0, 0]
    for sname in order:
        g = xu[xu["system"] == sname]
        gt = g["gt_stable"].astype(bool)
        pbe_ok = g["pbe_backed_stable"].astype(bool) == gt
        sol_ok = g["pbesol_backed_stable"].astype(bool) == gt
        fixed, new = int((~pbe_ok & sol_ok).sum()), int((pbe_ok & ~sol_ok).sum())
        pp = xp[xp["system"] == sname]
        r = pp["depth_ratio_pbesol_over_pbe"]
        edge = bool(pp["pbesol_min_at_edge"].astype(bool).any())
        fixed_who = ""
        if 0 < fixed <= 2:
            fx = g[~pbe_ok & sol_ok]
            fixed_who = " (" + "; ".join(f"{PRETTY[x.model]}, {x.T:.0f} K" for x in fx.itertuples()) + ")"
        qsol = sorted({round(float(x), 3) for x in pp["pbesol_Q_min_A"]})
        xrow.append([names[sname], len(g), int(pbe_ok.sum()), int(sol_ok.sum()),
                     f"{fixed}{fixed_who}", new, f"{r.min():.2f}–{r.max():.2f}",
                     "yes" if edge else ("no" if not pp["pbe_min_at_edge"].astype(bool).any()
                                         else f"no ({'/'.join(f'{q:.2f}' for q in qsol)} Å; PBE at the 0.45 Å edge)")])
        for i, x in enumerate((len(g), int(pbe_ok.sum()), int(sol_ok.sum()), fixed, new)):
            tot[i] += x
    xrow.append(["*total*", *[str(x) for x in tot], "", ""])
    xc = chk["xc"]
    left = xu[(xu["pbesol_backed_stable"].astype(bool) != xu["gt_stable"].astype(bool))]
    left_txt = _and(f"{names[x.system]}/{PRETTY[x.model]} at {x.T:.0f} K" for x in left.itertuples())
    # lower part: PBE lattice
    pl = chk["pbe_lattice"]["systems"]
    lrow = []
    for sname in order:
        L = pl[sname]["lattice"]
        a = list(L["mlip_a_A"].values())
        d = L["mlip_minus_pbe_pct"]
        big = max(d, key=d.get)
        lrow.append([names[sname], f"{L['a_pbe_A']:.4f}", f"{min(a):.4f}–{max(a):.4f}",
                     f"+{min(d.values()):.2f} to +{max(d.values()):.2f} (largest {PRETTY[big]})"])
    phases = {(v["phase_B"], v["phase_C"]) for v in pl.values()}
    bc_txt = ("The PBE force constants and E(Q) profiles at the PBE lattice (phases B and C of "
              "the check) were not computed, so whether the lattice or the eigenvector moves a "
              "call is not tested." if phases == {("pending", "pending")} else
              "Phases B and C: see `results/revision/dft_checks/pl_phonons.csv`.")
    pmax = max(abs(v["lattice"]["final_scf_pressure_kbar"]) for v in pl.values())
    jobs = pd.read_csv(DFT_CHECKS / "qe_jobs.csv")
    jobs = jobs[jobs["job"].str.match(r"(cv|xs|pl)_") & (jobs["status"] == "ok")]
    n_jobs = len(jobs)
    return (
        "**Table S23** Checks on the errors that persist on PBE (`scripts/dft_checks.py "
        "analyze-checks` → `results/revision/dft_checks/`"
        + (f"; {n_jobs} calculations" if n_jobs else "") + "). "
        "Upper part: k-point and cutoff convergence of the MACE-MP-0 deciding path of each system, "
        "as the relative change of the well depth (−min E(Q) over the ten sampled amplitudes, per "
        "modulated cell) from the production settings (k-spacing 0.25 Å⁻¹ with 2π included, SSSP "
        "cutoffs) to a k-spacing of 0.15 Å⁻¹ (*k*), cutoffs ×1.3 with the density cutoff kept at "
        "eight times the wavefunction cutoff (*ecut*), and both. The criterion, fixed before any "
        f"variant ran, is {chk['conv']['acceptance']}; the verdict is **{chk['conv']['verdict']}**. "
        "It fails on CsSnBr₃ only, by the depth, and no call changes anywhere; *Calls changed* is "
        f"over {n_T} temperatures × 3 variants. Middle part: the screen re-solved with PBEsol on "
        "the same structures, over the ladder units of each system (5 models × 100, 300, 600, "
        f"900 K), along the {xc['n_paths']} deciding paths; **PBEsol here is input_dft = 'pbesol' "
        "on the SSSP 1.3 PBE pseudopotentials at each MLIP's lattice, not the SSSP PBEsol set and "
        f"not PBEsol's own lattice**. PBEsol corrects {xc['pbe_errors_fixed_by_pbesol']} of the "
        f"{xc['pbe_errors']} PBE errors and introduces {xc['new_errors_from_pbesol']}; the "
        f"{len(left)} left wrong are {left_txt}. Lower part: the cubic lattice relaxed in PBE "
        f"(vc-relax, production settings, final pressure within {pmax:.2f} kbar) against the MLIP "
        "lattices the curves of Table S19 use. " + bc_txt + " Counts are descriptive (three "
        "systems).\n\n"
        + md(crow, ["System", "Path (MACE-MP-0, deciding)", "Production depth (meV)", "*k*",
                    "*ecut*", "*k* + *ecut*", f"Calls changed (of {3 * n_T})",
                    "PBE minimum at scan edge", "Criterion"])
        + "\n\n"
        + md(xrow, ["System", "Units", "Correct, PBE", "Correct, PBEsol", "PBE errors corrected",
                    "New errors", "Depth PBEsol/PBE", "PBEsol minimum at scan edge"])
        + "\n\n"
        + md(lrow, ["System", "PBE a (Å)", "MLIP a (Å), five models", "MLIP − PBE (%)"])
    )


def build() -> str:
    raw = pd.read_parquet(LEDGER)
    df = A.canonical(raw)
    if not STATS.exists():
        raise SystemExit("results/stats_hardening.json missing - run scripts/stats_hardening.py first")
    if not SENS.exists():
        raise SystemExit("results/screen_sensitivity.json missing - run scripts/screen_sensitivity.py first")
    st = json.loads(STATS.read_text(encoding="utf-8"))
    sens = json.loads(SENS.read_text(encoding="utf-8"))
    blocks = [
        table_s4(df),
        table_s5(df),
        table_s6(df),
        table_s8_composition(st),   # numbered Table S7 in the text; see caption
        table_s7(df),               # numbered Table S8
        table_s9_orb(df, st, sens),
        table_s10_paired(st),
        table_s11_sscha_diag(df),
        table_s13_disp_sweep(raw),
        table_s14_screen_sensitivity(sens),
        table_s15_h2_clustered(st),
        table_s16_bcc_agreement(st),
        table_s17_sscha_high_t(st),
        table_s18_force_spread(),
        table_s19_c3a_pbe(),
        table_s20_c3b(),
        table_s21_sscha_seeds(),
        table_s22_converged_grid(),
        table_s23_dft_checks(),
    ]
    blocks = [b for b in blocks if b]
    return "\n\n".join(blocks)


# a Python replacement field left in the output: {name}, {name[1]}, {name.attr}, {name:.2f}, {name!r}
PLACEHOLDER = re.compile(r"\{[A-Za-z_]\w*(?:\[[^\]{}]*\]|\.[A-Za-z_]\w*)*(?:![rsa])?(?::[^{}]*)?\}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the ESI on disk is out of date instead of rewriting it")
    args = ap.parse_args()

    body = build()
    # Comparing the generated text with the file cannot catch a string that lost its f prefix
    # (Table S19 caption, eeb10aa): the bug is regenerated identically. Fail on any unrendered
    # Python format field in the generated tables instead.
    leftover = sorted(set(PLACEHOLDER.findall(body)))
    if leftover:
        raise SystemExit("ESI generated tables contain unrendered format fields "
                         f"(missing f prefix?): {', '.join(leftover)}")
    text = ESI.read_text(encoding="utf-8")
    i, j = text.index(BEGIN), text.index(END)
    new = text[:i] + BEGIN + "\n\n" + body + "\n\n" + text[j:]

    if args.check:
        if new != text:
            raise SystemExit("ESI generated tables are STALE - run scripts/build_esi_tables.py")
        print("ESI generated tables are up to date")
        return

    ESI.write_text(new, encoding="utf-8")
    n_tables = body.count("**Table S")
    print(f"wrote {ESI.relative_to(REPO)}: {n_tables} generated tables, {len(body.splitlines())} lines")


if __name__ == "__main__":
    main()
