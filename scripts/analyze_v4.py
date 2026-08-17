"""Headline numbers for the multi-mode (v3+) soft-mode screen.

The screen changed from "map E(Q) for the globally softest commensurate mode" to "map E(Q) for
EVERY distinct imaginary commensurate mode, and call the phase unstable if ANY of them
condenses". That is what dynamical stability means, and it matters: in SrTiO3 the Gamma
ferroelectric mode is deeper than the R-point antiferrodistortive tilt, yet the Gamma mode is
quantum-suppressed and never condenses while the R tilt drives the 105 K transition.

Rows produced by the multi-mode screen are identified by a non-null ``ft_n_imag_total``.

Usage:  python scripts/analyze_v4.py [path/to/ledger.parquet]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FE_OXIDE = ["batio3_cubic", "pbtio3_cubic", "knbo3_cubic"]
AFD = ["srtio3_cubic"]
HALIDE = ["cspbi3_cubic", "cssni3_cubic", "cssnbr3_cubic"]
FLUORITE = ["zro2_cubic", "hfo2_cubic"]
BCC = ["ti_bcc", "zr_bcc", "hf_bcc"]
CONTROL = ["si_diamond", "mgo_rocksalt", "nacl_rocksalt", "cu_fcc", "c_diamond", "ceo2_cubic"]


def _unstable_frac(x: pd.DataFrame) -> float:
    return float((~x.dynamically_stable.astype(bool)).mean()) if len(x) else float("nan")


def main(path: str) -> int:
    d = pd.read_parquet(path)
    sm = d[d.method == "softmode"].copy()
    if "ft_n_imag_total" not in sm.columns:
        print("no multi-mode rows in this ledger")
        return 1
    v_new = sm[sm.ft_n_imag_total.notna()].copy()
    v_old = sm[sm.ft_n_imag_total.isna()].copy()
    # the v1 grid may contain more than one algorithm generation per unit; keep the first
    v_old = v_old.groupby(["system", "model", "temperature_K"], as_index=False).first()

    print(f"multi-mode rows: {len(v_new)}   legacy rows: {len(v_old)}")
    print()

    print("=== recall on genuinely-unstable families (truth = UNSTABLE), T <= 300 K ===")
    print(f"{'family':<22}{'legacy':>10}{'multi-mode':>14}{'n':>6}")
    fams = [("FE oxide perovskite", FE_OXIDE), ("AFD (SrTiO3)", AFD),
            ("halide perovskite", HALIDE), ("cubic fluorite", FLUORITE)]
    for name, sysl in fams:
        a = v_old[v_old.system.isin(sysl) & (v_old.temperature_K <= 300)]
        b = v_new[v_new.system.isin(sysl) & (v_new.temperature_K <= 300)]
        print(f"{name:<22}{_unstable_frac(a):>10.2f}{_unstable_frac(b):>14.2f}{len(b):>6}")

    print()
    print("=== controls (truth = STABLE): fraction correctly called stable ===")
    for name, v in (("legacy", v_old), ("multi-mode", v_new)):
        c = v[v.system.isin(CONTROL)]
        acc = float(c.dynamically_stable.astype(bool).mean()) if len(c) else float("nan")
        print(f"  {name:<12} {acc:.3f}   (n={len(c)})")

    print()
    print("=== per-model, all genuinely-unstable non-bcc systems, T <= 300 K ===")
    uns = FE_OXIDE + AFD + HALIDE + FLUORITE
    rows = []
    for m, g in v_new[v_new.system.isin(uns) & (v_new.temperature_K <= 300)].groupby("model"):
        ctl = v_new[(v_new.model == m) & v_new.system.isin(CONTROL)]
        rows.append({"model": m, "n": len(g), "recall": _unstable_frac(g),
                     "control_acc": float(ctl.dynamically_stable.astype(bool).mean())})
    print(pd.DataFrame(rows).sort_values("recall", ascending=False).round(3).to_string(index=False))

    print()
    print("=== screen vs SSCHA on the FE-oxide set, T <= 300 K ===")
    ss = d[(d.method == "sscha") & d.system.isin(FE_OXIDE) & (d.temperature_K <= 300)]
    scr = v_new[v_new.system.isin(FE_OXIDE) & (v_new.temperature_K <= 300)]
    print(f"  multi-mode screen recall : {_unstable_frac(scr):.2f}  (n={len(scr)})")
    print(f"  SSCHA recall             : {_unstable_frac(ss):.2f}  (n={len(ss)})")

    print()
    print("=== mode census: distinct imaginary commensurate modes per system ===")
    cen = (v_new[v_new.temperature_K == 100]
           .groupby("system")
           .agg(n_imag=("ft_n_imag_total", "mean"),
                n_screened=("ft_n_imag_screened", "mean"),
                n_condensed=("ft_n_condensed", "mean")).round(1))
    cen["capped"] = (cen.n_imag > cen.n_screened)
    print(cen.to_string())
    if cen.capped.any():
        print(f"  !! cap still binding for: {sorted(cen[cen.capped].index)}")
    else:
        print("  no system is truncated by the mode cap")

    print()
    print("=== SrTiO3 validation gate: which mode condenses, and at what T ===")
    s = v_new[v_new.system == "srtio3_cubic"].sort_values(["model", "temperature_K"])
    for _, r in s.iterrows():
        print(f"  {r.model:<10} T={int(r.temperature_K):<4} stable={str(r.dynamically_stable):<5} "
              f"decide_q={r.ft_decide_q}  n_cond={int(r.ft_n_condensed)}")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "results" / "ledger.parquet")
    sys.exit(main(p))
