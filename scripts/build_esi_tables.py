"""Render the ESI's supplementary tables from the deposited ledger.

Referee 3 (RSC Advances RA-ART-07-2026-006452), item 4: the ESI promised Tables S1-S4 but
carried only the names of the functions that generate them, so "none of the results are
checkable". The functions existed and ran; the tables were simply never rendered into the
document. This script renders them, and adds the tables the rest of the revision needs.

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

LEDGER = REPO / "results" / "ledger.parquet"
STATS = REPO / "results" / "stats_hardening.json"
ESI = REPO / "paper" / "supplementary.md"

BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"

PRETTY = {"chgnet": "CHGNet", "mace_mp0": "MACE-MP-0", "mattersim": "MatterSim",
          "orb_v2": "ORB-v2", "sevennet0": "SevenNet-0"}
FE = ["batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
FLUORITE = ["zro2_cubic", "hfo2_cubic"]


def md(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ci(k: int, n: int) -> str:
    r = S.rate_ci(int(k), int(n))
    return f"{r.k}/{r.n} = {r.p:.3f} [{r.lo:.3f}, {r.hi:.3f}]" if n else "--"


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
    return (
        "**Table S6** Predicted stabilisation temperature T* (the lowest ladder temperature at "
        "which the high-symmetry phase is called stable) against the experimental transition "
        "temperature, per system and model. T* is a **diagnostic, not a prediction**: a "
        "single-mode treatment is not expected to reproduce an absolute T_c, the screen "
        "systematically under-estimates T* for the entropy-stabilised bcc metals, and it fails "
        "to order PbTiO₃ against the other perovskite anchors (§3.2). `analysis.predicted_tstar(df)`.\n\n"
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
                    "Vote", "Consensus", "Ground truth", "Consensus correct"])
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
        "**Table S7** Composition of the two analysis sets, which the previous version of this "
        "ESI did not state. Both rest on the **same 15 systems**, so they share their clustering "
        "structure: the 60 guardrail units are 15 systems × 4 temperatures, and the 75 matched "
        "pairs are 15 systems × 5 models. In the guardrail set the five model votes are already "
        "collapsed into each unit, so the models are not an independent axis there and the "
        "clustering unit is the system. Systems whose identifier contains `bcc` are dropped, "
        "which also removes the superionic `agi_bcc` — retained deliberately for comparability "
        "with the published numbers and stated in §3.2.\n\n"
        + md(rows, ["Analysis set", "n units", "n systems", "Temperature axis", "Model axis",
                    "Balanced"])
        + f"\n\nThe 15 systems are: {syslist}."
    )


def table_s9_orb(st: dict) -> str:
    o = st["orb_split"]
    rows = []
    for tag, label in [("all_models", "all five models"), ("excl_orb_v2", "excluding ORB-v2")]:
        b = o[tag]["bcc_method_agreement"]
        rows.append(["bcc screen-vs-SSCHA sign agreement", label,
                     f"{b['sign_agreement']:.3f} over {b['n_paired']} pairs"])
        rows.append(["bcc screen-vs-SSCHA Spearman ρ (magnitudes)", label,
                     f"{b['spearman_freq']:+.3f}"])
        for meth, v in o[tag]["displacive_recall"].items():
            rows.append([f"FE-perovskite recall, {meth}", label, v["fmt"]])
    return (
        "**Table S9** Every rate that ORB-v2 could plausibly drive, reported with and without "
        "it. ORB-v2 is float32 with non-conservative forces, which is an architecture confound "
        "for finite-difference force constants rather than a model property of interest, and it "
        "accounts for the MgO false-unstable, the ≈ −35 THz Ti/Hf outliers and most of the SSCHA "
        "blow-ups. Two entries deserve attention: the frequency rank correlation on bcc is "
        "≈ 0.00 once ORB-v2 is removed, not merely smaller, which is why call agreement rather "
        "than magnitude correlation is the cross-validation statistic we report; and the "
        "FE-perovskite recall contrast narrows. See Table S10 for the paired tests.\n\n"
        + md(rows, ["Quantity", "Model set", "Value"])
    )


def table_s10_paired(st: dict) -> str:
    rows = []
    for key, v in st["screen_vs_sscha_paired"].items():
        if not v:
            continue
        setname, tag = key.rsplit("__", 1)
        rows.append([
            setname.replace("_", " "), tag.replace("_", " "),
            v["n_paired_units"], v["screen_right_sscha_wrong"],
            v["sscha_right_screen_wrong"], f"{v['mcnemar_exact_p']:.2g}",
        ])
    return (
        "**Table S10** The central screen-versus-SSCHA contrast tested **as a paired "
        "comparison**, which is what the design supports: both methods are evaluated on the same "
        "(system, model, temperature) units, so comparing their two marginal Wilson intervals "
        "would ignore the pairing. Counts are of discordant pairs; the test is an exact "
        "two-sided McNemar over them. The result to read honestly is the middle pair of rows: on "
        "the ferroelectric oxides **alone**, the contrast is significant with ORB-v2 included "
        "and not significant without it. The claim therefore rests on the combined displacive "
        "set, where the cubic fluorites — numerically clean, zero blow-ups, and essentially "
        "unaffected by ORB-v2 — carry it. `scripts/stats_hardening.py`.\n\n"
        + md(rows, ["System set", "Model set", "n paired", "Screen right, SSCHA wrong",
                    "SSCHA right, screen wrong", "Exact p"])
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
    for (fam, model), g in d.groupby(["family", "model"]):
        r = g["resid"].dropna()
        n_ok = int((g["n_zeros"] == 3).sum())
        n_swamped = int((g["n_zeros"] == 0).sum())
        rows.append([
            fam, PRETTY.get(model, model), len(g),
            f"{n_ok}/{len(g)}",
            f"{r.max():.1e}" if len(r) else "--",
            n_swamped,
            f"{g['wall_s'].min():.0f}–{g['wall_s'].max():.0f}",
        ])
    return (
        "**Table S11** SSCHA numerical quality per family and model. The sampling configuration "
        "is identical for all 201 units and is therefore stated once here rather than tabulated: "
        "256 configurations per population, a cap of 8 populations, a dedicated "
        "512-configuration ensemble for the free-energy Hessian, 2560 samples in total. The "
        "stopping criterion is `python-sscha`'s automatic stochastic relaxation under the "
        "per-population step cap `minim.max_ka = 20` (§S2.1), with the Hessian evaluated on a "
        "fresh ensemble at the converged auxiliary matrix.\n\n"
        "Two quantities are tabulated from the six lowest recorded Hessian frequencies. "
        "*Acoustic zeros resolved* counts units in which all three translational zeros "
        f"(|ω| < {TINY:g} THz) appear within that window, and the residual column gives the "
        "largest of their magnitudes over those units, which bounds the numerical noise on a "
        "quantity known analytically to be zero. *Swamped* counts the opposite case: units with "
        "**no** recorded mode near zero, meaning at least six modes lie below the acoustic "
        "branches. Swamping is not itself an error — a deeply unstable phase genuinely has many "
        "imaginary modes, and the reported minimum frequency excludes the acoustic branches from "
        "the full spectrum rather than from this window — but it separates the families "
        "sharply, and it marks the units on which the free-energy Hessian is furthest from the "
        "regime its bubble truncation is valid in (§3.3).\n\n"
        "**What the harness did not retain**, and what would therefore need a re-run to supply: "
        "the per-iteration free-energy gradient history, and a per-unit uncertainty on the "
        "Hessian eigenvalues. The uncertainty probe that does exist is the independent-seed "
        "study of §S2.4, which this revision extends beyond bcc-Zr at Referee 1's request.\n\n"
        + md(rows, ["Family", "Model", "n units", "Acoustic zeros resolved",
                    "Max zero residual (THz)", "Swamped", "Wall time (s)"])
    )


def build() -> str:
    df = A.canonical(pd.read_parquet(LEDGER))
    if not STATS.exists():
        raise SystemExit("results/stats_hardening.json missing - run scripts/stats_hardening.py first")
    st = json.loads(STATS.read_text(encoding="utf-8"))
    blocks = [
        table_s4(df),
        table_s5(df),
        table_s6(df),
        table_s8_composition(st),   # numbered Table S7 in the text; see caption
        table_s7(df),               # numbered Table S8
        table_s9_orb(st),
        table_s10_paired(st),
        table_s11_sscha_diag(df),
    ]
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
