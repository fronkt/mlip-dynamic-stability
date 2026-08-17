"""Assert every headline number in the manuscript against the ledger.

Run this after any re-run, before rebuilding the DOCX. It exists because the manuscript and the
data drifted apart once already: the deposited ledger was produced by a q-search the code no
longer contained, and nothing checked. Each assertion names the section it defends.

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

    check(len(v) == 400, f"[data] 400 multi-mode softmode units (got {len(v)})")

    c = v[v.system.isin(CONTROLS)]
    check(len(c) == 120 and bool(c.dynamically_stable.astype(bool).all()),
          "[3.2] all 120 control units called stable")
    check(bool((c.ft_n_imag_total == 0).all()),
          "[3.2] controls carry zero imaginary commensurate modes")

    for name, sysl, exp, sec in [("FE oxide", FE_OXIDE, 0.53, "3.2/3.3"),
                                 ("halide", HALIDE, 0.92, "3.2"),
                                 ("fluorite", FLUORITE, 1.00, "3.2/3.3"),
                                 ("AFD SrTiO3", ["srtio3_cubic"], 0.60, "3.2")]:
        r, n = recall(v, sysl)
        check(abs(r - exp) < 0.006, f"[{sec}] {name} recall {r:.2f} == {exp} (n={n})")

    dr = A.displacive_recall(d).set_index("method")["recall_unstable"]
    check(abs(dr["softmode"] - 0.533) < 0.005 and abs(dr["sscha"] - 0.300) < 0.005,
          f"[3.3] screen {dr['softmode']:.3f} vs SSCHA {dr['sscha']:.3f}")

    bcc = d[d.system.isin(["ti_bcc", "zr_bcc", "hf_bcc"])]
    ma = A.method_agreement_summary(bcc)
    check(ma["n_paired"] == 45 and abs(ma["spearman_freq"] - 0.743) < 0.005
          and abs(ma["sign_agreement"] - 0.622) < 0.005,
          f"[3.3] bcc agreement rho={ma['spearman_freq']:.3f} sign={ma['sign_agreement']:.3f} n=45")

    t = A.low_t_false_stable(d, exclude_bcc=True).set_index("model")
    check(t.loc["sevennet0", "accuracy"] == 0.900 and t.loc["orb_v2", "accuracy"] == 0.800,
          "[3.2] table: SevenNet-0 leads at 0.900, ORB-v2 last at 0.800")

    h3 = A.h3_guardrail_summary(d, method="softmode")
    check(h3["n_units"] == 60 and abs(h3["auc_vote_disagreement"] - 0.762) < 0.005
          and abs(h3["auc_freq_std"] - 0.547) < 0.005,
          f"[3.4] H3 AUC vote {h3['auc_vote_disagreement']:.3f} vs freq {h3['auc_freq_std']:.3f}")

    cap = v[v.ft_n_imag_screened < v.ft_n_imag_total]
    check(len(cap) == 20 and not bool(cap.dynamically_stable.astype(bool).any())
          and int(cap.ft_n_condensed.min()) >= 1,
          f"[2.4] mode cap binds on {len(cap)} rows, none of them called stable")

    s = v[(v.system == "srtio3_cubic") & (v.model == "mace_mp0")]
    f100 = float(s[s.temperature_K == 100].min_eff_freq_thz.iloc[0])
    f300 = float(s[s.temperature_K == 300].min_eff_freq_thz.iloc[0])
    check(abs(f100 + 2.68) < 0.02 and abs(f300 - 1.43) < 0.02,
          f"[2.4] SrTiO3 gate {f100:.2f} -> {f300:.2f} THz across Tc")
    s100 = v[(v.system == "srtio3_cubic") & (v.temperature_K == 100)]
    check(int((~s100.dynamically_stable.astype(bool)).sum()) == 3,
          "[2.4] 3 of 5 models condense the SrTiO3 tilt at 100 K")

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
