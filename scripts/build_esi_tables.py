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
results/revision/dft/. Tables S1-S3 and S12 are hand-written in their own sections.

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
        "`scripts/stats_hardening.py` (`orb_split_s3`, `bcc_agreement`); pooled finite-T "
        "accuracy from `scripts/screen_sensitivity.py`.\n\n"
    )
    return caption + md(rows, ["Quantity", "All five models", "Excluding ORB-v2"])


def table_s10_paired(st: dict) -> str:
    rows = []
    for key, v in st["screen_vs_sscha_paired"].items():
        if not v:
            continue
        setname, tag = key.rsplit("__", 1)
        c = v["clustered_by_system"]
        p_unit = v["mcnemar_exact_p_UNIT_LEVEL"]
        rows.append([
            setname.replace("_", " "), tag.replace("_", " "),
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
        f"{nets}) rather than a significance claim. Why SSCHA loses these units is examined "
        "separately, on the same MLIP energies, in Table S16 (lower part) and §3.3: on most of "
        "them the screen's free-energy comparison finds a displaced minimum below the symmetric "
        "point, which a criterion read at the symmetric reference does not see. "
        "`scripts/stats_hardening.py`.\n\n"
        + md(rows, ["System set", "Model set", "n paired", "Discordant (screen/SSCHA)",
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
        "**What the harness did not retain**, and what would therefore need a re-run to supply: "
        "the per-iteration free-energy gradient history, and a per-unit uncertainty on the "
        "Hessian eigenvalues. The only uncertainty probe in the production data is the "
        "independent-seed study of §S2.4, which covers one unit, bcc-Zr/MACE-MP-0 at 100 K, "
        "on which the harmonic layer finds no bcc instability. "
        "<!-- PENDING-C1: four-seed SSCHA with recorded gradient/error history, population count "
        "and convergence flag for BaTiO3/MACE-MP-0 100 K, ZrO2/MACE-MP-0 100 K, Zr/MatterSim "
        "50 K and SrTiO3/MACE-MP-0 600 K; the seed spread of the lowest Hessian eigenvalue is the "
        "Hessian uncertainty; results go in a new ESI table referenced here and in §S2.4 -->\n\n"
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
        "systems. For SSCHA the call is the sign of the lowest free-energy-Hessian frequency; for "
        "the screen it is the variational argmin of §2.4. Both methods run on the same MLIP "
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
        "that SSCHA's positive Hessian has the same origin (§3.3).\n\n"
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
        "**Table S17** Where SSCHA calls a phase unstable that its label calls stable, and where "
        "it fails numerically. Upper part: non-bcc SSCHA false-unstables by temperature, over "
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
        "showing that this rather than the sampling drives the runaway. "
        "<!-- PENDING-C1: the SrTiO3/MACE-MP-0 "
        "600 K seed study records whether the relaxation reached the SCHA minimum (the third "
        "candidate); one clause here, whichever way it falls --> "
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
        + "SSCHA never calls SrTiO₃ stable at any temperature, so a missing zone-boundary "
        "q-point cannot be what produces a false-stable here." + no_imag_txt + "\n\n"
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
    """Referee 1.1 (plan item C3a): PBE along the screen's own soft-mode coordinates.

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
    m["ratio"] = m["depth_meV_own"].abs() / m["depth_meV_pbe"]
    m["edge"] = m["Q_min_A_pbe"] >= EDGE_Q_A
    m["order"] = m["system_own"].map({s: i for i, s in enumerate(C3A_ORDER)})
    m = m.sort_values(["order", "stem"])
    drows = []
    for r in m.itertuples():
        has_well = r.depth_meV_own > 0
        drows.append([r.stem, r.role_own, f"{_nz(r.depth_meV_own):.1f}", f"{r.depth_meV_pbe:.1f}",
                      f"{_nz(r.ratio):.2f}",
                      f"{r.Q_min_A_own:.3f}" if has_well else "no well", f"{r.Q_min_A_pbe:.3f}",
                      "yes" if r.edge else "no"])
    fe = m[m["system_own"].isin(["batio3_cubic", "knbo3_cubic"]) & (m["role_own"] == "decide")]
    fe4 = fe[fe["path_model_own"] != "orb_v2"]
    edge_sys = sorted({SHORT[s] for s in m.loc[m["edge"], "system_own"]})
    n_decide, n_ref = int((m["role_own"] == "decide").sum()), int((m["role_own"] == "ref").sum())

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
    n_decide_pbe = int(u["deciding_mode_has_pbe"].sum())
    resid = u[~u["pbe_ok"]].groupby("system").size()
    resid_txt = ", ".join(f"{SHORT[s]} {resid[s]}" for s in C3A_ORDER if s in resid)
    n_nowell = int((m["depth_meV_own"] <= 0).sum())
    npb = u["n_paths_pbe"]
    return (
        "**Table S19** PBE along the screen's own soft-mode coordinates (Referee 1.1, plan item "
        f"C3a). Each of the {tot[1]} ladder units (6 systems × 5 models × 100, 300, 600 and 900 K) "
        "is called twice by the screen's rule (unstable if any computed path condenses, §2.4), on "
        "the same structures: from the model's own energies (*MLIP, same paths*) and from "
        "Quantum ESPRESSO PBE single-point energies (*PBE-backed*). Both are scored against the "
        "finite-temperature label used throughout (stable iff T is at or above the experimental "
        "transition temperature). *Corrected by PBE* counts units whose MLIP call is wrong and "
        "whose PBE-backed call is right; *newly wrong* counts the reverse. **PBE is evaluated at "
        "each MLIP's own relaxed lattice, along that MLIP's coordinate**, so it is a different "
        f"PBE potential for each model, not one reference curve per system. PBE covers "
        f"{int(npb.min())} to {int(npb.max())} paths per unit, out of the "
        f"{int(u['n_screened'].min())} to {int(u['n_screened'].max())} "
        f"modes the screen maps; the mode that decided the ledger's call has a PBE path in "
        f"{n_decide_pbe} of {len(u)} units. The MLIP column is the model's own call on those "
        f"same paths and equals the ledger's call in {same_ledger} of {len(u)} units. In "
        f"{n_off_cache} units a path's regenerated E(Q) map did not reproduce the cached map the "
        "ledger was computed from (a substitute direction inside a degenerate eigenspace; "
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
        f"{n_ref} reference), the path model's own E(Q) against PBE on the same {int(p['n_Q_dft'].max())} "
        "structures. Depth is −min E(Q) over the sampled amplitudes, zero when none is below "
        "E(0), and Q_min is the sampled amplitude at that minimum, so both are limited to the "
        f"{int(p['n_Q_dft'].max())}-point scan to 0.45 Å. *Ratio* is own depth over PBE depth. "
        f"The BaTiO₃ and KNbO₃ deciding paths of the four models other than ORB-v2 "
        f"({len(fe4)} paths: " + ", ".join(PRETTY[x] for x in sorted(set(fe4['path_model_own'])))
        + f") have own-model depths {fe4['ratio'].min():.3f} to {fe4['ratio'].max():.3f} of PBE "
        f"(median {fe4['ratio'].median():.3f}); with ORB-v2's {len(fe) - len(fe4)} paths the range is "
        f"{fe['ratio'].min():.3f} to {fe['ratio'].max():.3f}. **The PBE minimum is at the scan "
        f"edge (Q ≥ {EDGE_Q_A} Å) on {int(m['edge'].sum())} of {len(m)} paths, all "
        f"{' and '.join(edge_sys)}**; there the sampled depth is a lower bound on the PBE depth "
        "and the ratio is not a well-depth comparison.\n\n"
        + md(drows, ["Path", "Role", "Own depth (meV)", "PBE depth (meV)", "Ratio own/PBE",
                     "Own Q_min (Å)", "PBE Q_min (Å)", "PBE minimum at scan edge"])
    )


def table_s20_c3b() -> str:
    """Referee 1.2 (plan item C3b): MLIP-versus-PBE forces and energies on SSCHA-sampled
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
        "**Table S20** MLIP errors on SSCHA-sampled configurations, scored against PBE (Referee "
        "1.2, plan item C3b). The configurations are drawn from the C1 SSCHA ensembles "
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
    ]
    blocks = [b for b in blocks if b]
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the ESI on disk is out of date instead of rewriting it")
    args = ap.parse_args()

    body = build()
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
