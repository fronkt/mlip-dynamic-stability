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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = A.canonical(pd.read_parquet(LEDGER))

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


if __name__ == "__main__":
    main()
