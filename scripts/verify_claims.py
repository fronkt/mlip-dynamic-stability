"""Assert every headline number in the manuscript against the ledger.

Run this after any re-run, before rebuilding the DOCX. It exists because the manuscript and the
data drifted apart once already: the deposited ledger was produced by a q-search the code no
longer contained, and nothing checked. Each assertion names the section it defends.

Current generations: harmonic v2 (pinned envs), softmode v4 (multi-mode, curvature observable),
sscha v3 (phonopy initialiser, pinned envs; 201 of 208 units complete, 7 die at cellconstructor
assertions on the deepest wells -- that failure count is itself asserted below).

Usage:  python scripts/verify_claims.py        # exit 0 if every claim holds
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from mlip_dynstab import analysis as A

CONTROLS = ["si_diamond", "mgo_rocksalt", "nacl_rocksalt", "cu_fcc", "c_diamond", "ceo2_cubic"]
FE_OXIDE = ["batio3_cubic", "pbtio3_cubic", "knbo3_cubic"]
HALIDE = ["cspbi3_cubic", "cssni3_cubic", "cssnbr3_cubic"]
FLUORITE = ["zro2_cubic", "hfo2_cubic"]
BCC = ["ti_bcc", "zr_bcc", "hf_bcc"]

_fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        _fails.append(msg)


def recall(v: pd.DataFrame, systems, t_max=300.0):
    x = v[v.system.isin(systems) & (v.temperature_K <= t_max)]
    u = x[~x.gt_stable.astype(bool)]
    return (float((~u.dynamically_stable.astype(bool)).mean()) if len(u) else float("nan")), len(u)


def main() -> int:
    d = A.canonical(pd.read_parquet("results/ledger.parquet"))
    v = d[d.method == "softmode"]
    ss = d[d.method == "sscha"]
    h = d[d.method == "harmonic"]

    check(len(v) == 400 and len(h) == 100 and len(ss) == 201,
          f"[data] generations: 400 softmode / 100 harmonic / 201 sscha "
          f"(got {len(v)}/{len(h)}/{len(ss)})")

    # ---- 3.1 harmonic (v2, pinned envs)
    t31 = A.per_model_table(d, "harmonic").set_index("model")
    check(t31.loc["mattersim", "accuracy"] == 1.0 and t31.loc["sevennet0", "accuracy"] == 1.0,
          "[3.1] MatterSim and SevenNet-0 harmonically perfect")
    check(abs(t31.loc["orb_v2", "accuracy"] - 0.895) < 1e-3
          and abs(t31.loc["orb_v2", "false_stable_rate"] - 0.077) < 1e-3,
          "[3.1] ORB-v2 accuracy 0.895, false-stable rate 0.077")
    check(abs(t31.loc["mace_mp0", "false_stable_rate"] - 0.154) < 1e-3
          and abs(t31.loc["chgnet", "false_stable_rate"] - 0.154) < 1e-3,
          "[3.1] MACE-MP-0 and CHGNet false-stable rate 0.154 (bcc Zr/Hf)")
    sw = A.harmonic_tolerance_sweep(d).set_index("tol_THz")
    check(int(sw.loc[0.10, "false_stable"]) == 5 and int(sw.loc[0.10, "false_unstable"]) == 3
          and int(sw.loc[0.30, "false_stable"]) == 7,
          "[3.1] tolerance sweep 5FP/3FN at 0.1, 7FP at 0.3")

    # ---- 3.2 screen (softmode v4)
    c = v[v.system.isin(CONTROLS)]
    check(len(c) == 120 and bool(c.dynamically_stable.astype(bool).all())
          and bool((c.ft_n_imag_total == 0).all()),
          "[3.2] all 120 control units stable with zero imaginary modes")
    for name, sysl, exp in [("FE oxide", FE_OXIDE, 0.53), ("halide", HALIDE, 0.92),
                            ("fluorite", FLUORITE, 1.00), ("AFD SrTiO3", ["srtio3_cubic"], 0.60)]:
        r, n = recall(v, sysl)
        check(abs(r - exp) < 0.006, f"[3.2] {name} screen recall {r:.2f} == {exp} (n={n})")
    t32 = A.low_t_false_stable(d, exclude_bcc=True).set_index("model")
    check(t32.loc["sevennet0", "accuracy"] == 0.900 and t32.loc["orb_v2", "accuracy"] == 0.800,
          "[3.2] finite-T table: SevenNet-0 leads 0.900, ORB-v2 last 0.800")
    for t, phi, p in ((100.0, 0.149, 0.727), (300.0, -0.129, 0.007)):
        r = A.h2_paired_summary(d, t=t)
        check(abs(r["phi"] - phi) < 0.005 and abs(r["mcnemar_exact_p"] - p) < 0.005,
              f"[3.2] H2 at {int(t)} K: phi={r['phi']:.3f}, McNemar p={r['mcnemar_exact_p']:.3f}")

    # ---- 3.3 SSCHA (v3, pinned envs)
    dr = A.displacive_recall(d).set_index("method")
    check(abs(dr.loc["softmode", "recall_unstable"] - 0.533) < 0.005
          and abs(dr.loc["sscha", "recall_unstable"] - 0.185) < 0.005
          and int(dr.loc["sscha", "n_valid"]) == 27,
          f"[3.3] FE recall: screen {dr.loc['softmode','recall_unstable']:.3f} "
          f"vs SSCHA {dr.loc['sscha','recall_unstable']:.3f} (n_valid 27)")
    check(208 - len(ss) == 7, "[3.3] exactly 7 SSCHA units failed at assertions")
    bcc_ss = ss[ss.system.isin(BCC)]
    check(len(bcc_ss) == 75 and bool((bcc_ss.min_eff_freq_thz > 0).all()),
          "[3.3] bcc: 75/75 clean, all positive (stabilised by <=50 K)")
    ma = A.method_agreement_summary(d[d.system.isin(BCC)])
    check(ma["n_paired"] == 45 and abs(ma["sign_agreement"] - 0.778) < 0.005,
          f"[3.3] bcc call agreement {ma['sign_agreement']:.3f} over 45 pairs")
    ma2 = A.method_agreement_summary(d[d.system.isin(BCC) & (d.model != "orb_v2")])
    check(abs(ma2["sign_agreement"] - 0.833) < 0.005,
          f"[3.3] bcc call agreement excl ORB {ma2['sign_agreement']:.3f}")
    fl = ss[ss.system.isin(FLUORITE)]
    fl100 = fl[fl.temperature_K == 100]
    check(bool((fl100.min_eff_freq_thz > 0).all()) and len(fl100) == 10,
          "[3.3] fluorites: all 10 units false-stable at 100 K")
    check(int((fl.min_eff_freq_thz.abs() > 50).sum()) == 0,
          "[3.3] fluorites: zero numerical blow-ups")

    # ---- 3.4 guardrail
    h3 = A.h3_guardrail_summary(d, method="softmode")
    check(h3["n_units"] == 60 and abs(h3["auc_vote_disagreement"] - 0.762) < 0.005
          and abs(h3["auc_freq_std"] - 0.361) < 0.005,
          f"[3.4] H3 AUC vote {h3['auc_vote_disagreement']:.3f} vs freq {h3['auc_freq_std']:.3f}")

    # ---- 2.4 gate (curvature observable + argmin call)
    s = v[(v.system == "srtio3_cubic") & (v.model == "mace_mp0")]
    r100 = s[s.temperature_K == 100].iloc[0]
    r300 = s[s.temperature_K == 300].iloc[0]
    check(not bool(r100.dynamically_stable) and bool(r300.dynamically_stable),
          "[2.4] SrTiO3/MACE: unstable at 100 K, stable at 300 K (brackets Tc=105 K)")
    check(abs(float(r100.min_eff_freq_thz) - 0.92) < 0.02
          and abs(float(r300.min_eff_freq_thz) - 1.43) < 0.02,
          f"[2.4] gate curvature +{float(r100.min_eff_freq_thz):.2f} -> "
          f"+{float(r300.min_eff_freq_thz):.2f} THz (positive while condensing at 100 K)")
    check(int(r100.ft_n_curv_blind) == 1,
          "[2.4] the 100 K condensation is curvature-blind (n_curv_blind = 1)")
    s100 = v[(v.system == "srtio3_cubic") & (v.temperature_K == 100)]
    check(int((~s100.dynamically_stable.astype(bool)).sum()) == 3,
          "[2.4] 3 of 5 models condense the SrTiO3 tilt at 100 K")
    cap = v[v.ft_n_imag_screened < v.ft_n_imag_total]
    check(len(cap) == 20 and not bool(cap.dynamically_stable.astype(bool).any()),
          f"[2.4] mode cap binds on {len(cap)} rows, none called stable")

    # ---- 3.5 reproducibility
    zr = ss[(ss.system == "zr_bcc") & (ss.model == "mace_mp0")]
    check(bool((zr.min_eff_freq_thz.round(2) == 1.80).all()) or
          bool(((zr.min_eff_freq_thz - 1.80).abs() < 0.02).all()),
          "[3.5] zr/MACE SSCHA +1.80 THz across the ladder (matches v1 seed study +1.798)")

    print()
    if _fails:
        print(f"{len(_fails)} CLAIM(S) FAILED -- the manuscript and the ledger disagree:")
        for f in _fails:
            print("  -", f)
        return 1
    print("all manuscript claims verified against the ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
