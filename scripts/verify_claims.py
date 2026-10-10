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

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import comb

import numpy as np
import pandas as pd

from mlip_dynstab import analysis as A
from mlip_dynstab import stats as S

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


def h2_test(d: pd.DataFrame, t: float, drop_systems=()):
    """H2 transfer asymmetry on the matched set at T: (b, c, unit McNemar p, clustered p, info)."""
    bl = A.borderline_systems()

    def keep(x):
        return x[~x.system.isin(bl) & ~x.system.str.contains("bcc") & ~x.system.isin(drop_systems)]

    hh = keep(d[d.method == "harmonic"])[["system", "model", "pred_stable", "gt_stable"]]
    ff = keep(d[(d.method == "softmode") & (d.temperature_K == t)])[
        ["system", "model", "pred_stable", "gt_stable"]]
    mm = hh.merge(ff, on=["system", "model"], suffixes=("_h", "_f")).sort_values(["system", "model"])
    hok = (mm.pred_stable_h.astype(bool) == mm.gt_stable_h.astype(bool)).to_numpy()
    fok = (mm.pred_stable_f.astype(bool) == mm.gt_stable_f.astype(bool)).to_numpy()
    b, c = int((hok & ~fok).sum()), int((~hok & fok).sum())
    nd = b + c
    p_unit = min(1.0, 2 * sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd) if nd else 1.0
    cl = S.cluster_exact_paired(hok, fok, mm.system.to_numpy())
    return b, c, p_unit, cl["p_exact_clustered"], cl


def text_vs_data(ms_txt: str) -> None:
    """Headline numbers as printed in the manuscript against a recomputation from raw data.

    The checks above mostly pin pipeline outputs (grid_compare.json, summary.csv). This layer
    re-derives the headline numbers from the per-unit SSCHA JSONs, the yaml labels (stable iff
    T >= T_c), the ledger's softmode v4 calls and the DFT unit tables, without the project's
    analysis code, renders each one as the manuscript prints it, and requires that exact string
    in the section that states it. Text drift (a number edited in the prose but not in the data,
    or the reverse) therefore fails here even when the pipeline products are unchanged.
    """
    import glob
    import itertools
    import math
    import yaml

    def norm(s):
        return " ".join(s.split())

    def sect(start, end):
        return norm(ms_txt.split(start, 1)[1].split(end, 1)[0])

    abst, s26 = sect("## Abstract", "## 1."), sect("### 2.6", "## 3.")
    s32, s33 = sect("### 3.2", "### 3.3"), sect("### 3.3", "### 3.4")
    s4 = sect("## 4.", "## 5.")

    def wil(k, n, z=1.959963984540054):
        p, den = k / n, 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return f"[{c - h:.2f}, {c + h:.2f}]"

    def signflip_p(nets):
        obs = abs(sum(nets))
        hits = [abs(sum(s * x for s, x in zip(sg, nets))) >= obs - 1e-12
                for sg in itertools.product((1, -1), repeat=len(nets))]
        return sum(hits) / len(hits)

    def has(where, name, s):
        check(s in where, f"[text] {name}: '{s}'")

    tc = {s_["id"]: s_["transition_T_K"]
          for s_ in yaml.safe_load(open("configs/curated_systems.yaml", encoding="utf-8"))["systems"]}

    def lab(system, T):
        return tc[system] is not None and T >= tc[system]

    led = pd.read_parquet("results/ledger.parquet")
    sm = led[(led.method == "softmode") & (led.method_version == 4)]
    scr = {(r_.system, r_.model, float(r_.temperature_K)): bool(r_.pred_stable) for r_ in sm.itertuples()}
    rows = []
    for f_ in sorted(glob.glob("results/revision/sscha_converged_grid/*_startA.json")):
        g_ = json.load(open(f_, encoding="utf-8"))
        u_, run = g_["unit"], g_.get("run") or {}
        ok_ = run.get("status") == "ok"
        conv = ok_ and bool((run.get("relax") or {}).get("converged"))
        hz = run["hessian"]["min_nonac_thz"] if ok_ else None
        T = float(u_["T"])
        rows.append(dict(tag=os.path.basename(f_)[:-len("_startA.json")], system=u_["system"], model=u_["model"], T=T, conv=conv,
                         stable=(hz is not None and hz >= -0.1), lab=lab(u_["system"], T),
                         scr=scr.get((u_["system"], u_["model"], T))))
    g = pd.DataFrame(rows)
    check(len(g) == 178, f"[text] grid: {len(g)} start-A unit JSONs read")
    # E3 (pre-registered replicates, tasks/preregistration-repeats-2026-10-03.md): a unit whose call
    # differs between start A, start B and seed 10 is UNRESOLVED and leaves numerator and
    # denominator; recomputed here from the per-replicate values, not from the stored verdict.
    e3r = pd.read_csv("results/revision/e3_replicates/summary.csv")

    def _rcall(r_, p_):
        return None if r_[f"{p_}_status"] != "ok" else bool(float(r_[f"{p_}_hessian_min_thz"]) >= -0.1)
    unres = set()
    for r_ in e3r.to_dict("records"):
        calls = [x for x in (_rcall(r_, "A"), _rcall(r_, "B"), _rcall(r_, "A10")) if x is not None]
        if len(set(calls)) > 1:
            unres.add(r_["unit_tag"])
    check(unres == set(e3r.loc[e3r.verdict == "unresolved", "unit_tag"]) and len(unres) == 8,
          f"[E3] 8 unresolved units, recomputed from the replicate values ({sorted(unres)})")
    g["unres"] = g.tag.isin(unres)
    cA = g[g.conv & ~g.system.str.endswith("_bcc")]
    c = cA[~cA.unres]
    noorb = c.model != "orb_v2"

    # (i) converged false-stables among label-unstable non-bcc units, and the screen on them
    luA = cA[~cA.lab]
    fsA = luA[luA.stable]
    has(s33, "3.3 FS start A", f"alone {len(fsA)}/{len(luA)}, of which the replicates of Section 3.5 leave "
        f"{int(fsA.unres.sum())} unresolved")
    lu = c[~c.lab]
    fs = lu[lu.stable]
    k, n = len(fs), len(lu)
    has(abst, "abstract FS", f"calls {k} of {n} label-unstable non-bcc units stable (seven unresolved)")
    check(int(fsA.unres.sum()) == 7, "[text] seven start-A false-stables are unresolved")
    has(s33, "3.3 FS", f"{k}/{n} = {k / n:.2f} {wil(k, n)}")
    kx, nx = int((fs.model != "orb_v2").sum()), int((lu.model != "orb_v2").sum())
    has(s33, "3.3 FS ex-ORB", f"{kx}/{nx} without ORB-v2")
    ox = ["srtio3_cubic", "batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
    lu_ox, fs_ox = lu[lu.system.isin(ox)], fs[fs.system.isin(ox)]
    check(len(fs_ox) == k, f"[text] every converged false-stable is an oxide perovskite ({len(fs_ox)}/{k})")
    has(s33, "3.3 oxide-perovskite denominator",
        f"{len(fs_ox)} of the {len(lu_ox)} resolved label-unstable oxide-perovskite units")
    lu_fe, fs_fe = lu[lu.system.isin(FE_OXIDE)], fs[fs.system.isin(FE_OXIDE)]
    has(s33, "3.3 ferroelectric part", f"{len(fs_fe)} of {len(lu_fe)} on the three ferroelectrics")
    bysys = fs.groupby("system").size().to_dict()
    has(s33, "3.3 FS by system", f"BaTiO₃ ({bysys['batio3_cubic']}), KNbO₃ ({bysys['knbo3_cubic']}), "
        f"PbTiO₃ at 300 and 600 K ({bysys['pbtio3_cubic']}) and SrTiO₃ at 100 K ({bysys['srtio3_cubic']})")
    # audit 2026-10-09 M5: the count without the SrTiO3 100 K units (2x2x2, no size test, Section 3.5)
    no_sto = ~((lu.system == "srtio3_cubic") & (lu["T"] == 100.0))
    k_ns, n_ns = int((no_sto & lu.stable).sum()), int(no_sto.sum())
    check(k - k_ns == 4 and set(fs[~fs.index.isin(lu[no_sto].index)].system) == {"srtio3_cubic"}
          and set(lu[no_sto & lu.stable].system) <= set(FE_OXIDE),
          f"[text] M5: dropping SrTiO3 100 K removes 4 false-stables ({k_ns}/{n_ns} left, all ferroelectric)")
    has(s33, "3.3 FS without SrTiO3 100 K", f"without them the count is {k_ns}/{n_ns} = {k_ns / n_ns:.2f} "
        f"{wil(k_ns, n_ns)}, all on the three ferroelectrics")
    ks = int((fs.scr == False).sum())  # noqa: E712
    ksA = int((fsA.scr == False).sum())  # noqa: E712
    has(abst, "abstract screen on FS", f"cannot exclude, in {ks} of them")
    has(s33, "3.3 screen on FS", f"{ks}/{k} = {ks / k:.2f} {wil(ks, k)}")
    has(s33, "3.3 screen on FS ex-ORB",
        f"{int(((fs.scr == False) & (fs.model != 'orb_v2')).sum())}/{kx} = "  # noqa: E712
        f"{((fs.scr == False) & (fs.model != 'orb_v2')).sum() / kx:.2f} "  # noqa: E712
        f"{wil(int(((fs.scr == False) & (fs.model != 'orb_v2')).sum()), kx)} without ORB-v2")  # noqa: E712
    has(s33, "3.3 screen on FS start A", f"{ksA}/{len(fsA)} from start A alone")

    # (ii) ferroelectric recall at T <= 300 K on the units where SSCHA converged
    fe = c[c.system.isin(FE_OXIDE) & (c["T"] <= 300) & ~c.lab]
    ks_, kc_, nf = int((fe.scr == False).sum()), int((~fe.stable).sum()), len(fe)  # noqa: E712
    feA = cA[cA.system.isin(FE_OXIDE) & (cA["T"] <= 300) & ~cA.lab]
    has(abst, "abstract screen recall", f"recovers {ks_}/{nf} {wil(ks_, nf)}")
    has(abst, "abstract SSCHA recall", f"against {kc_}/{nf} {wil(kc_, nf)} for converged SSCHA")
    has(s33, "3.3 screen recall", f"{ks_}/{nf} = {ks_ / nf:.2f} {wil(ks_, nf)}")
    has(s33, "3.3 SSCHA recall", f"{kc_}/{nf} = {kc_ / nf:.2f} {wil(kc_, nf)}")
    fx = fe[fe.model != "orb_v2"]
    kfx, kcx = int((fx.scr == False).sum()), int((~fx.stable).sum())  # noqa: E712
    has(s33, "3.3 recall ex-ORB", f"{kcx}/{len(fx)} = {kcx / len(fx):.2f} {wil(kcx, len(fx))} without ORB-v2), "
        f"against {ks_}/{nf}")
    has(s33, "3.3 screen recall ex-ORB", f"({kfx}/{len(fx)} = {kfx / len(fx):.2f} {wil(kfx, len(fx))} without ORB-v2)")
    has(s33, "3.3 recall start A", f"the counts were {int((~feA.stable).sum())}/{len(feA)} and "
        f"{int((feA.scr == False).sum())}/{len(feA)}")  # noqa: E712
    check(int(feA.unres.sum()) == 4, "[text] four FE T<=300 K units unresolved")

    # (v) paired contrast, displacive set at T <= 300 K, exact sign flip over systems
    def _paired(cc):
        dp_ = cc[cc.system.isin(FE_OXIDE + FLUORITE) & (cc["T"] <= 300) & cc.scr.notna()]
        sr_, cr_ = dp_.scr.astype(bool) == dp_.lab, dp_.stable == dp_.lab
        return dp_, dp_[sr_ & ~cr_], dp_[~sr_ & cr_]
    dp, b_, c_ = _paired(c)
    dpA, bA, cA_ = _paired(cA)
    nets = {s_: int((b_.system == s_).sum()) - int((c_.system == s_).sum()) for s_ in FE_OXIDE + FLUORITE}
    p_all, p_fe = signflip_p(list(nets.values())), signflip_p([nets[s_] for s_ in FE_OXIDE])
    has(s33, "3.3 paired counts", f"wrong on {len(b_)} units and SSCHA is right where the screen is "
        f"wrong on {len(c_)} ({int((b_.model != 'orb_v2').sum())} against "
        f"{int((c_.model != 'orb_v2').sum())} on {int((dp.model != 'orb_v2').sum())} units without")
    has(s33, "3.3 paired start A", f"{len(bA)} against {len(cA_)} on {len(dpA)} units from the positive-definite start")
    has(s33, "3.3 paired p", f"is exact at this size, gives p = {p_all:g}")
    has(s33, "3.3 PbTiO3 net", f"PbTiO₃ favours SSCHA by one ({int((b_.system == 'pbtio3_cubic').sum())} against "
        f"{int((c_.system == 'pbtio3_cubic').sum())})")
    has(abst, "abstract paired p (ferroelectric units)", f"(paired, system-clustered p = {p_fe:g})")
    check(len(dp) == 40 and len(dpA) == 44 and nets["pbtio3_cubic"] == -1 and int(noorb.sum()) > 0,
          f"[text] paired displacive set: {len(dp)} units (start A {len(dpA)}), nets {nets}")
    has(s4, "4 paired", f"wrong on {len(b_)} units against {len(c_)}")
    # bcc call agreement, converged 3x3x3, resolved
    bb = g[g.conv & g.system.str.endswith("_bcc") & g.scr.notna() & ~g.unres]
    has(s33, "3.3 bcc agreement", f"{int((bb.stable == bb.scr.astype(bool)).sum())}/{len(bb)} = "
        f"{(bb.stable == bb.scr.astype(bool)).mean():.2f} {wil(int((bb.stable == bb.scr.astype(bool)).sum()), len(bb))}")

    # E1: screen on PBE against the MLIPs along the same coordinates (c3a_unit_calls.csv)
    cu = pd.read_csv("results/revision/dft/c3a_unit_calls.csv")
    cu = cu[cu["T"].isin([100.0, 300.0, 600.0, 900.0])]          # the ladder (the file adds 50 K)
    cl = cu.apply(lambda r_: lab(r_.system, float(r_["T"])), axis=1)
    n_pbe = int((cu.pbe_backed_stable.astype(bool) == cl).sum())
    n_ml = int((cu.mlip_same_paths_stable.astype(bool) == cl).sum())
    has(s32, "3.2 E1 agreement", f"agreement with the labels from {n_ml} to {n_pbe}")
    has(s4, "4 E1 agreement", f"in {n_pbe} of {len(cu)} units against {n_ml} with the MLIPs along the same coordinates")
    ps = json.load(open("results/revision/dft/summary.json", encoding="utf-8"))["c3a"]["paths"]
    n_done = sum(e["status"] == "complete" for e in ps)
    has(s26, "2.6 E1 coverage", f"({n_done} of the {len(ps)} screened paths")
    cp = pd.read_csv("results/revision/dft/c3a_paths.csv")
    dm = (cp[cp.curve == cp.path_model].merge(cp[cp.curve == "pbe"], on="stem", suffixes=("", "_p")))
    dm = dm[dm.system.isin(["batio3_cubic", "knbo3_cubic"]) & (dm.path_model != "orb_v2")]
    ratio = dm.depth_meV / dm.depth_meV_p
    has(s32, "3.2 depth ratio, 12 deciding paths", f"(median {ratio[dm.role == 'decide'].median():.2f})."
        " That is a softening relative to PBE at fixed geometry")
    has(s32, "3.2 depth ratio, all 26 paths", f"are {ratio.min():.2f}–{ratio.max():.2f} of PBE's "
        f"(median {ratio.median():.2f})")

    # E2: PBEsol on CsSnBr3 (xc_unit_calls.csv)
    words = {7: "seven", 8: "eight", 9: "nine"}
    xu = pd.read_csv("results/revision/dft_checks/xc_unit_calls.csv")
    xu = xu[xu.in_ladder.astype(bool) & xu.unit_complete.astype(bool) & (xu.system == "cssnbr3_cubic")]
    xl = xu.apply(lambda r_: lab(r_.system, float(r_["T"])), axis=1)
    pe = xu.pbe_backed_stable.astype(bool) != xl
    me = xu.mlip_pred_stable_ledger.astype(bool) != xl
    fx = xu.pbesol_backed_stable.astype(bool) == xl
    k1, n1 = int((pe & fx).sum()), int(pe.sum())
    k2, n2 = int((pe & me & fx).sum()), int((pe & me).sum())
    has(s32, "3.2 PBEsol, persistent", f"{words[k2]} of the {words[n2]} persistent CsSnBr₃ errors")
    has(s32, "3.2 PBEsol, all PBE errors", f"{k1} of all {n1} CsSnBr₃ PBE errors")


def main() -> int:
    sys.stdout.reconfigure(errors="backslashreplace")  # manuscript strings carry subscripts
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
          f"[3.3] bcc curvature-sign agreement {ma['sign_agreement']:.3f} over 45 pairs "
          f"(not the call agreement; see S2 below)")
    # [3.2] The matched-set harmonic column. The finite-T column scores 15 systems; quoting the
    # 19-system harmonic accuracy beside it produced the (withdrawn) "CHGNet worst harmonically,
    # second best at finite T" illustration. On matched systems CHGNet is 0.867 in BOTH layers.
    bl = A.borderline_systems()
    mh = d[(d.method == "harmonic") & (~d.system.isin(bl)) & (~d.system.str.contains("bcc"))]
    macc = {m: (g.pred_stable.astype(bool) == g.gt_stable.astype(bool)).mean()
            for m, g in mh.groupby("model")}
    check(all(abs(macc[m] - 1.0) < 1e-9 for m in ("mace_mp0", "mattersim", "sevennet0"))
          and abs(macc["chgnet"] - 13 / 15) < 1e-9 and abs(macc["orb_v2"] - 13 / 15) < 1e-9,
          f"[3.2] matched-set harmonic: 3 models at 1.000, CHGNet and ORB-v2 at "
          f"{macc['chgnet']:.3f} (13/15)")

    # [3.2] Displacement-amplitude sweep, all five models (0.005/0.02/0.03 A vs production
    # 0.01 A). Pins the reported outcomes per model: MACE-MP-0, MatterSim and SevenNet-0 flip
    # no call; CHGNet flips only controls and its accuracy spans 14/19-17/19; ORB-v2 flips
    # test systems too.
    raw = pd.read_parquet("results/ledger.parquet")
    sw = raw[raw.method == "harmonic_dispsweep"]
    if len(sw):
        pr = d[(d.method == "harmonic") & (d.model.isin(sw.model.unique()))][
            ["system", "model", "pred_stable", "gt_stable"]]
        mm = sw.merge(pr, on=["system", "model"], suffixes=("", "_prod"))
        flipped = mm[mm.pred_stable.astype(bool) != mm.pred_stable_prod.astype(bool)]
        fl = {m: sorted(set(flipped.system[flipped.model == m])) for m in sorted(sw.model.unique())}
        check(sorted(sw.model.unique()) == sorted(["chgnet", "mace_mp0", "mattersim", "orb_v2",
                                                   "sevennet0"])
              and not fl["mace_mp0"] and not fl["mattersim"] and not fl["sevennet0"],
              "[3.2] disp sweep: all five models swept; MACE-MP-0, MatterSim, SevenNet-0 flip no call")
        check(set(fl["chgnet"]) <= set(CONTROLS),
              f"[3.2] disp sweep: CHGNet flips only controls ({fl['chgnet']})")
        check(any(s not in CONTROLS for s in fl["orb_v2"]),
              f"[3.2] disp sweep: ORB-v2 flips test systems too ({fl['orb_v2']})")
        bl2 = A.borderline_systems()
        accs = {}
        ch = mm[(mm.model == "chgnet") & (~mm.system.isin(bl2))]
        for dd, g in ch.groupby("disp_ang"):
            accs[float(dd)] = (g.pred_stable.astype(bool) == g.gt_stable_prod.astype(bool)).mean()
        check(abs(min(accs.values()) - 14 / 19) < 1e-9 and abs(max(accs.values()) - 17 / 19) < 1e-9,
              f"[3.2] disp sweep: CHGNet accuracy spans {min(accs.values()):.3f}-{max(accs.values()):.3f}")

    # [3.5] within-cell control: the same 2x2x2 cell that SSCHA calls stable is a cell in
    # which the harmonic calculation sees the fluorite instability. Pins the argument that the
    # fluorite false-stable is the truncation and not finite size (R3 item 2).
    FLUOR = ["zro2_cubic", "hfo2_cubic"]
    fh = d[(d.method == "harmonic") & d.system.isin(FLUOR)]
    fq = d[(d.method == "sscha") & d.system.isin(FLUOR) & (d.temperature_K == 100)]
    check(len(fh) == 10 and (~fh.pred_stable.astype(bool)).all()
          and abs(fh.min_freq_thz.max() - (-3.788)) < 0.01
          and abs(fh.min_freq_thz.min() - (-10.595)) < 0.01,
          f"[3.5] harmonic calls all 10 fluorite units unstable in the 2x2x2 cell "
          f"({fh.min_freq_thz.min():.1f} to {fh.min_freq_thz.max():.1f} THz)")
    check(len(fq) == 10 and (fq.pred_stable.astype(bool)).all()
          and abs(fq.min_eff_freq_thz.min() - 1.903) < 0.01
          and abs(fq.min_eff_freq_thz.max() - 3.331) < 0.01,
          f"[3.5] production recipe: SSCHA calls all 10 fluorite units stable in the SAME cell "
          f"(+{fq.min_eff_freq_thz.min():.1f} to +{fq.min_eff_freq_thz.max():.1f} THz)")

    ma2 = A.method_agreement_summary(d[d.system.isin(BCC) & (d.model != "orb_v2")])
    check(abs(ma2["sign_agreement"] - 0.833) < 0.005,
          f"[3.3] bcc curvature-sign agreement excl ORB {ma2['sign_agreement']:.3f}")
    fl = ss[ss.system.isin(FLUORITE)]
    fl100 = fl[fl.temperature_K == 100]
    check(bool((fl100.min_eff_freq_thz > 0).all()) and len(fl100) == 10,
          "[3.3] production recipe: fluorites, all 10 units false-stable at 100 K")
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

    # ==== Revision plan S1-S4 (results/stats_hardening.json blocks h2_clustered, bcc_agreement,
    # orb_split_s3, sscha_high_t). Recomputed here from the ledger, not read from the JSON.
    exo = d[d.model != "orb_v2"]

    # ---- S1: H2 transfer asymmetry clustered by system (R1.6, R3.1, R3.3, R3.5)
    for t, (b0, c0, pu0, pc0), (b1, c1, pu1, pc1) in (
            (100.0, (5, 3, 0.727, 0.844), (3, 2, 1.000, 1.000)),
            (300.0, (17, 4, 0.007, 0.152), (14, 2, 0.004, 0.156)),
            (600.0, (23, 4, 0.0003, 0.066), (20, 2, 0.0001, 0.055)),
            (900.0, (11, 4, 0.118, 0.414), (10, 2, 0.039, 0.250))):
        r0, r1 = h2_test(d, t), h2_test(exo, t)
        check(r0[:2] == (b0, c0) and abs(r0[2] - pu0) < 5e-4 and abs(r0[3] - pc0) < 5e-4
              and r1[:2] == (b1, c1) and abs(r1[2] - pu1) < 5e-4 and abs(r1[3] - pc1) < 5e-4,
              f"[3.2] H2 {int(t)} K: {r0[0]} v {r0[1]}, unit p {r0[2]:.4f}, system-clustered p "
              f"{r0[3]:.3f}; ex-ORB {r1[0]} v {r1[1]}, unit p {r1[2]:.4f}, clustered {r1[3]:.3f}")
    cl300 = h2_test(d, 300.0)[4]
    check(cl300["clusters_favouring_a"] == 5 and cl300["clusters_favouring_b"] == 4,
          "[3.2] H2 300 K: 5 systems favour harmonic-right/finite-T-wrong, 4 the reverse")
    v300 = v[(v.temperature_K == 300) & ~v.system.isin(A.borderline_systems())
             & ~v.system.str.contains("bcc")]
    nwrong = (v300.pred_stable.astype(bool) != v300.gt_stable.astype(bool)).groupby(v300.system).sum()
    shared = sorted(nwrong[nwrong >= 4].index)
    rs = h2_test(d, 300.0, drop_systems=shared)
    check(shared == ["batio3_cubic", "cssnbr3_cubic", "knbo3_cubic"] and rs[:2] == (4, 4),
          f"[3.2] H2 300 K: shared screen errors ({', '.join(shared)}: >=4/5 models "
          f"mis-called) carry 13 of the 17; without them {rs[0]} v {rs[1]}")
    loso = {s: h2_test(d, 600.0, drop_systems=(s,))[3] for s in
            sorted(set(v300.system))}
    loso_x = {s: h2_test(exo, 600.0, drop_systems=(s,))[3] for s in sorted(set(v300.system))}
    check(abs(min(loso.values()) - 0.0586) < 5e-4 and min(loso.values()) > 0.05
          and sorted(s for s, p in loso_x.items() if p < 0.05) == ["ceo2_cubic", "nacl_rocksalt"]
          and abs(min(loso_x.values()) - 0.0469) < 5e-4,
          f"[3.2] H2 600 K leave-one-system-out: clustered p >= {min(loso.values()):.4f} (all "
          f"models); ex-ORB drops of CeO2 or NaCl give {min(loso_x.values()):.4f}")

    # ---- S2: bcc agreement scored on curvature sign vs on the call (R1.1 residue, R3.5)
    def bcc_scores(frame):
        sm = frame[(frame.method == "softmode") & frame.system.isin(BCC)]
        sq = frame[(frame.method == "sscha") & frame.system.isin(BCC)]
        mm = sm.merge(sq, on=["system", "model", "temperature_K"], suffixes=("_s", "_q"))
        curv = ((mm.min_eff_freq_thz_s >= 0) == (mm.min_eff_freq_thz_q >= 0)).to_numpy()
        call = (mm.pred_stable_s.astype(bool) == mm.pred_stable_q.astype(bool)).to_numpy()
        return mm, curv, call

    mm, curv, call = bcc_scores(d)
    mmx, curvx, callx = bcc_scores(exo)
    check(len(mm) == 45 and curv.sum() == 35 and call.sum() == 31
          and len(mmx) == 36 and curvx.sum() == 30 and callx.sum() == 25,
          f"[3.3] bcc screen-vs-SSCHA: curvature-sign agreement {curv.sum()}/45, stability-call "
          f"agreement {call.sum()}/45; ex-ORB {curvx.sum()}/36 and {callx.sum()}/36")
    hb = h[h.system.isin(BCC) & h.pred_stable.astype(bool)]
    trivial = set(zip(hb.system, hb.model))
    triv = np.array([(s, m) in trivial for s, m in zip(mm.system, mm.model)])
    ms = (mm.model == "mattersim").to_numpy()
    check(trivial == {("zr_bcc", "mace_mp0"), ("zr_bcc", "chgnet"), ("hf_bcc", "mace_mp0"),
                      ("hf_bcc", "chgnet")}
          and triv.sum() == 12 and curv[triv].all() and curv[~triv].sum() == 23
          and curv[ms].sum() == 3 and call[ms].sum() == 0 and int((curv != call).sum()) == 10,
          f"[3.3] bcc: 12 trivial pairs (no harmonic instability: MACE/CHGNet on Zr/Hf); "
          f"non-trivial curvature-sign {curv[~triv].sum()}/{(~triv).sum()}; MatterSim "
          f"{curv[ms].sum()}/9 sign, {call[ms].sum()}/9 call; {int((curv != call).sum())} units differ")
    mx = A.method_agreement_summary(d[d.system.isin(BCC) & (d.model != "orb_v2")])
    check(abs(ma["spearman_freq"] - 0.113) < 5e-4 and abs(mx["spearman_freq"] + 0.003) < 5e-4,
          f"[3.3] bcc frequency Spearman (descriptive) {ma['spearman_freq']:+.3f}, "
          f"ex-ORB {mx['spearman_freq']:+.3f}")

    # ---- S3: the ORB-v2 split of everything else (R3.5, R2.2)
    h3x = A.h3_guardrail_summary(exo, method="softmode")
    gx = A.h3_ensemble_guardrail(exo, method="softmode")
    gx = gx[~gx.system.str.contains("bcc") & ~gx.system.isin(A.borderline_systems())]
    gx = gx.sort_values(["system", "T"]).reset_index(drop=True)
    tied = gx[gx.stable_vote_frac == 0.5]
    check(h3x["n_units"] == 60 and h3x["n_split"] == 8 and abs(h3x["split_vote_error_rate"] - 0.5) < 1e-9
          and h3x["n_unanimous"] == 52 and round(h3x["unanimous_error_rate"] * 52) == 8
          and abs(h3x["auc_vote_disagreement"] - 0.628) < 5e-4
          and abs(h3x["auc_freq_std"] - 0.299) < 5e-4
          and len(tied) == 3 and bool(tied.consensus_stable.all()),
          f"[3.4] guardrail ex-ORB: split 4/8 vs unanimous 8/52 wrong, vote AUC "
          f"{h3x['auc_vote_disagreement']:.3f}, freq-std AUC {h3x['auc_freq_std']:.3f}, "
          f"{len(tied)} 2-2 ties broken to 'stable'")
    errx = (~gx.consensus_correct.astype(bool)).to_numpy()
    px = S.cluster_permutation_auc(gx.disagreement.to_numpy(float), errx, gx.system.to_numpy(),
                                   n_perm=10000, seed=0)
    bx = S.cluster_bootstrap_auc(gx.disagreement.to_numpy(float), errx, gx.system.to_numpy(),
                                 n_boot=10000, seed=0)
    check(abs(px["p_perm_clustered"] - 0.0469) < 5e-4 and abs(bx["ci_lo"] - 0.438) < 5e-4
          and abs(bx["ci_hi"] - 0.839) < 5e-4,
          f"[3.4] guardrail ex-ORB: clustered p {px['p_perm_clustered']:.4f}, cluster-bootstrap "
          f"CI [{bx['ci_lo']:.3f}, {bx['ci_hi']:.3f}] spans 0.5 (10000, seed 0)")
    vx = v[v.model != "orb_v2"]
    fam_x = {n: recall(vx, s) for n, s in (("FE", FE_OXIDE), ("halide", HALIDE),
                                            ("fluorite", FLUORITE), ("SrTiO3", ["srtio3_cubic"]))}
    cx = vx[vx.system.isin(CONTROLS)]
    check(fam_x["FE"] == (0.5, 24) and fam_x["halide"] == (1.0, 20)
          and fam_x["fluorite"] == (1.0, 16) and fam_x["SrTiO3"] == (0.75, 4)
          and len(cx) == 96 and bool(cx.dynamically_stable.astype(bool).all()),
          "[3.2] ex-ORB screen recalls: FE 12/24, halide 20/20, fluorite 16/16, SrTiO3 3/4; "
          "controls 0/96 false-unstable")
    grid = pd.read_csv("results/sscha_units_v1.csv")
    kk = ["system", "model", "temperature_K"]
    gm = grid.merge(ss[kk + ["min_eff_freq_thz"]], on=kk, how="left", indicator=True)
    failed = gm[gm._merge == "left_only"].groupby("model").size().to_dict()
    blow = ss[ss.min_eff_freq_thz.abs() > 50].groupby("model").size().to_dict()
    check(len(grid) == 208 and failed == {"mattersim": 2, "orb_v2": 4, "sevennet0": 1}
          and blow == {"chgnet": 1, "mattersim": 3, "orb_v2": 3, "sevennet0": 1},
          f"[3.3] SSCHA per model: blow-ups {blow} (8, ORB-v2 3 of 8); failed units {failed} (7)")
    mgo = h[(h.model == "orb_v2") & (h.system == "mgo_rocksalt")].iloc[0]
    check(abs(float(mgo.min_freq_thz) + 1.065) < 5e-3 and not bool(mgo.pred_stable),
          f"[3.1] ORB-v2 MgO harmonic minimum {float(mgo.min_freq_thz):.2f} THz (as reviewed: -2.8)")
    check(int(t32.loc["orb_v2", "FP_false_stable"]) == 5 and int(t32.loc["orb_v2", "TN"]) == 11,
          "[3.2] ORB-v2 finite-T false-stable 5/16 = 0.312 (as reviewed: 0.562)")
    ob = v[(v.model == "orb_v2") & v.system.isin(BCC)].set_index(["system", "temperature_K"])
    check(abs(ob.loc[("ti_bcc", 100.0), "min_eff_freq_thz"] + 6.205) < 5e-3
          and abs(ob.loc[("ti_bcc", 900.0), "min_eff_freq_thz"] + 8.887) < 5e-3
          and abs(ob.loc["hf_bcc", "min_eff_freq_thz"].min() + 0.092) < 5e-3
          and float(ob.min_eff_freq_thz.min()) > -10,
          "[3.3] ORB-v2 bcc screen minima: Ti -6.20 (100 K) / -8.89 (900 K), Hf -0.09; "
          "no -35 THz outlier")
    swx = A.harmonic_tolerance_sweep(d[d.model != "orb_v2"]).set_index("tol_THz")
    check(int(swx.loc[0.10, "false_stable"]) == 4 and int(swx.loc[0.10, "false_unstable"]) == 2
          and int(swx.loc[0.10, "n_calls"]) == 76,
          "[3.1] tolerance sweep ex-ORB: 4 FP / 2 FN of 76 calls at 0.1 THz")

    # ---- S4: SSCHA false-unstables growing with temperature (R1.2)
    nb = ss[~ss.system.str.contains("bcc")]
    nbx = nb[nb.model != "orb_v2"]

    def fu_by_t(x):
        return [int((~x[(x.temperature_K == t) & x.gt_stable.astype(bool)].pred_stable.astype(bool)).sum())
                for t in (100.0, 300.0, 600.0, 900.0)]

    check(fu_by_t(nb) == [0, 5, 10, 15] and fu_by_t(nbx) == [0, 4, 8, 12],
          f"[3.3] production recipe: SSCHA non-bcc false-unstables by T {fu_by_t(nb)} (ex-ORB {fu_by_t(nbx)})")
    st = nb[(nb.system == "srtio3_cubic") & nb.gt_stable.astype(bool)]
    stx = st[st.model != "orb_v2"]
    check(len(st) == 14 and not st.pred_stable.astype(bool).any()
          and abs(st.min_eff_freq_thz.min() + 913.736) < 5e-3
          and abs(st.min_eff_freq_thz.max() + 3.392) < 5e-3
          and len(stx) == 11 and not stx.pred_stable.astype(bool).any(),
          f"[3.3] production recipe: SrTiO3 above Tc, SSCHA false-unstable in all {len(st)} units "
          f"({st.min_eff_freq_thz.max():.1f} to {st.min_eff_freq_thz.min():.0f} THz); ex-ORB 11/11")

    def reversals(x):
        p = x.pivot_table(index=["system", "model"], columns="temperature_K",
                          values="min_eff_freq_thz")
        s100 = p[p[100.0] >= 0]
        return len(s100), int(((s100[600.0] < 0) | (s100[900.0] < 0)).sum())

    check(reversals(nb) == (23, 14) and reversals(nbx) == (19, 10),
          f"[3.3] production recipe: SSCHA {reversals(nb)[1]} of {reversals(nb)[0]} non-bcc units stable at 100 K "
          f"turn negative by 600-900 K (ex-ORB {reversals(nbx)[1]} of {reversals(nbx)[0]})")

    # [3.3] On the non-bcc units SSCHA calls stable against an unstable label, the screen's
    # free-energy comparison finds a lower displaced minimum on 46/57 (the informative count).
    # The 52/57 'curvature positive' count is pinned only as a true number: the symmetric-point
    # curvature is positive by construction (scripts/curvature_identity_check.py), so it is not
    # evidence of anything. Recomputed from the ledger.
    kk = ["system", "model", "temperature_K"]
    cb = d[d.method == "softmode"][kk + ["min_eff_freq_thz", "pred_stable", "gt_stable"]].merge(
        d[d.method == "sscha"][kk + ["min_eff_freq_thz", "pred_stable"]], on=kk,
        suffixes=("_scr", "_ss"))
    cb = cb[~cb.system.str.contains("bcc")]
    cfs = cb[cb.pred_stable_ss.astype(bool) & ~cb.gt_stable.astype(bool)]
    cpos = int((cfs.min_eff_freq_thz_scr > 0).sum())
    cblind = int(((cfs.min_eff_freq_thz_scr > 0) & ~cfs.pred_stable_scr.astype(bool)).sum())
    ccall = int((~cfs.pred_stable_scr.astype(bool)).sum())
    check(len(cfs) == 57 and ccall == 46 and cpos == 52 and cblind == 41,
          f"[3.3] production recipe: SSCHA false-stables, screen call finds a displaced minimum on {ccall}/{len(cfs)} "
          f"(curvature positive by construction on {cpos}; {cblind} both)")

    # ==== Revision C3a / C3b (R1.1, R1.2): PBE along the screen's coordinates and PBE errors on
    # SSCHA configurations. Recomputed from the committed tables that `scripts/dft_reference.py
    # analyze` writes (not read from summary.json). The ladder is the 120 rows the ledger has a
    # call for; the 50 K rows have none. Counts are descriptive over six systems.
    dft = "results/revision/dft/"
    c3u = pd.read_csv(dft + "c3a_unit_calls.csv")
    c3u = c3u[c3u.mlip_pred_stable_ledger.notna()].copy()
    c3_gt = c3u.gt_stable.astype(bool)
    c3_mlip, c3_pbe = c3u.mlip_same_paths_stable.astype(bool), c3u.pbe_backed_stable.astype(bool)
    c3u["m_ok"], c3u["p_ok"] = c3_mlip == c3_gt, c3_pbe == c3_gt
    c3u["fixed"] = ~c3u.m_ok & c3u.p_ok          # MLIP wrong on the same paths, PBE right
    c3u["newly"] = c3u.m_ok & ~c3u.p_ok         # MLIP right, PBE wrong
    c3x = c3u[c3u.model != "orb_v2"]
    c3_tot = tuple(int(x) for x in (len(c3u), c3u.m_ok.sum(), c3u.p_ok.sum(), c3u.fixed.sum(),
                                    c3u.newly.sum()))
    check(c3_tot == (120, 81, 101, 23, 3),
          f"[R1.1] C3a: MLIP on the same paths {c3_tot[1]}/{c3_tot[0]} correct, PBE-backed "
          f"{c3_tot[2]}/{c3_tot[0]}; {c3_tot[3]} corrected by PBE, {c3_tot[4]} newly wrong")
    c3_totx = (len(c3x), int(c3x.m_ok.sum()), int(c3x.p_ok.sum()))
    check(c3_totx == (96, 61, 82),
          f"[R1.1] C3a ex-ORB-v2: MLIP {c3_totx[1]}/{c3_totx[0]}, PBE {c3_totx[2]}/{c3_totx[0]} "
          f"({int(c3x.fixed.sum())} corrected, {int(c3x.newly.sum())} newly wrong)")
    c3_per = c3u.groupby("system")[["m_ok", "p_ok"]].sum()
    c3_exp = {"batio3_cubic": (16, 19), "cssnbr3_cubic": (9, 11), "knbo3_cubic": (11, 15),
              "srtio3_cubic": (18, 20), "zr_bcc": (7, 16), "zro2_cubic": (20, 20)}
    check(all((int(c3_per.loc[k, "m_ok"]), int(c3_per.loc[k, "p_ok"])) == e
              for k, e in c3_exp.items()) and (c3u.groupby("system").size() == 20).all(),
          "[R1.1] C3a per system (MLIP -> PBE of 20): "
          + ", ".join(f"{k} {int(c3_per.loc[k, 'm_ok'])}->{int(c3_per.loc[k, 'p_ok'])}"
                      for k in c3_exp))
    c3_zr = c3u[(c3u.system == "zr_bcc") & c3u.fixed]
    check(len(c3_zr) == 10 and bool((c3_zr.kind == "reference_coordinate").all()),
          f"[R1.1] C3a: all {len(c3_zr)} bcc-Zr corrections are on MatterSim's reference "
          f"coordinate (models {sorted(set(c3_zr.model))})")
    c3_fe = c3u[c3u.system.isin(["batio3_cubic", "knbo3_cubic"])]
    c3_fe8 = c3_fe[(c3_fe["T"] == 300) & c3_fe.mlip_same_paths_stable.astype(bool)
                   & ~c3_fe.pbe_backed_stable.astype(bool) & ~c3_fe.gt_stable.astype(bool)]
    check(len(c3_fe8) == 8 and int(c3_fe.fixed.sum()) == 8
          and sorted(set(c3_fe8.model)) == ["chgnet", "mace_mp0", "mattersim", "sevennet0"]
          and c3_fe8.groupby("system").size().eq(4).all(),
          f"[R1.1] C3a: {len(c3_fe8)} BaTiO3+KNbO3 units at 300 K (CHGNet, MACE, MatterSim, "
          f"SevenNet) are MLIP-stable, PBE-unstable, label-unstable; no other BaTiO3/KNbO3 unit "
          f"is corrected ({int(c3_fe.fixed.sum())} in all)")

    c3p = pd.read_csv(dft + "c3a_paths.csv")
    c3_own, c3_pbe_c = c3p[c3p.curve == c3p.path_model], c3p[c3p.curve == "pbe"]
    c3_dm = c3_own.merge(c3_pbe_c, on="stem", suffixes=("_own", "_pbe"))
    c3_dm["ratio"] = c3_dm.depth_meV_own / c3_dm.depth_meV_pbe
    c3_dr = c3_dm[c3_dm.role_own.isin(["decide", "ref"])]          # the 38 deciding/reference paths
    c3_fe = c3_dm[c3_dm.system_own.isin(["batio3_cubic", "knbo3_cubic"])
                  & (c3_dm.path_model_own != "orb_v2")]
    c3_r = c3_fe[c3_fe.role_own == "decide"].ratio
    c3_models = sorted(set(c3_fe[c3_fe.role_own == "decide"].path_model_own))
    check(len(c3_dm) == 93 and len(c3_dr) == 38 and len(c3_r) == 12
          and c3_models == ["chgnet", "mace_mp0", "mattersim", "sevennet0"]
          and abs(c3_r.min() - 0.315) < 5e-4 and abs(c3_r.max() - 0.763) < 5e-4
          and abs(c3_r.median() - 0.525) < 1e-3,
          f"[R1.1] C3a: BaTiO3+KNbO3 own-path well depth is {c3_r.min():.3f}-{c3_r.max():.3f} of "
          f"PBE (median {c3_r.median():.3f}) over {len(c3_r)} deciding paths, 4 models")
    check(len(c3_fe) == 26 and abs(c3_fe.ratio.min() - 0.315) < 5e-4
          and abs(c3_fe.ratio.max() - 0.828) < 5e-4 and abs(c3_fe.ratio.median() - 0.525) < 1e-3,
          f"[3.2/S5.1] C3a E1: over all {len(c3_fe)} PBE-covered BaTiO3+KNbO3 paths of the 4 models "
          f"(12 deciding + 14 other modes) own depth is {c3_fe.ratio.min():.3f}-"
          f"{c3_fe.ratio.max():.3f} of PBE (median {c3_fe.ratio.median():.3f})")
    c3_edge = c3_pbe_c[c3_pbe_c.Q_min_A >= 0.449]
    c3_cs = c3_pbe_c[c3_pbe_c.system == "cssnbr3_cubic"]
    c3_cs_dr = c3_cs[c3_cs.role.isin(["decide", "ref"])]
    c3_cs_x = c3_cs[c3_cs.role == "mode"]
    check(len(c3_edge) == 8 and set(c3_edge.system) == {"cssnbr3_cubic"}
          and set(c3_edge.role) == {"decide"} and len(c3_cs_dr) == 8
          and len(c3_cs_x) == 11 and bool((c3_cs_x.depth_meV <= 1e-6).all()),
          f"[R1.1] C3a: {len(c3_edge)} PBE minima at the 0.45 A scan edge, all CsSnBr3 deciding "
          f"paths (= every CsSnBr3 deciding path); the {len(c3_cs_x)} other CsSnBr3 modes with PBE "
          f"have no PBE well")
    c3_cs_ml = c3_dm[(c3_dm.system_own == "cssnbr3_cubic") & (c3_dm.role_own == "mode")]
    check(len(c3_cs_ml) == 11 and bool((c3_cs_ml.depth_meV_own > 1.0).all())
          and sorted(set(c3_cs_ml.path_model_own)) == ["chgnet", "mattersim"]
          and abs(c3_cs_ml.depth_meV_own.min() - 1.4) < 0.05
          and abs(c3_cs_ml.depth_meV_own.max() - 13.8) < 0.05,
          f"[S5.1] C3a E1: CHGNet/MatterSim CsSnBr3 wells absent in PBE on {len(c3_cs_ml)} "
          f"non-deciding modes ({c3_cs_ml.depth_meV_own.min():.1f}-"
          f"{c3_cs_ml.depth_meV_own.max():.1f} meV); no unit call changes")
    # E1 coverage: c3a_paths.csv lists only paths with curves; the pending ones are in summary.json
    c3_ps = pd.DataFrame([{k: e.get(k) for k in ("stem", "system", "path_model", "role",
                                                 "n_atoms", "status", "cache_check_pass")}
                          for e in json.load(open(dft + "summary.json"))["c3a"]["paths"]])
    c3_sel = json.load(open(dft + "qe/c3a_selection.json"))
    c3_skip = {d_["stem"]: d_["why"] for d_ in c3_sel["skipped"]}
    c3_pend = c3_ps[c3_ps.status == "pending"]
    c3_copy = [s_ for s_ in c3_pend.stem if "symmetry copy" in c3_skip.get(s_, "")]
    c3_big = c3_pend[c3_pend.stem.map(lambda s_: "atoms > cap 12" in c3_skip.get(s_, ""))]
    c3_src_ok = all(c3_ps.set_index("stem").loc[c3_skip[s_].split("copy of ")[1], "status"]
                    == "complete" for s_ in c3_copy)
    c3_ndone = int((c3_ps.status == "complete").sum())
    check(len(c3_ps) == 139 and c3_ndone == 93 and len(c3_pend) == 46 and len(c3_copy) == 8
          and c3_src_ok and len(c3_big) == 38 and int(c3_big.n_atoms.min()) == 18
          and int(c3_big.n_atoms.max()) == 216,
          f"[2.6/S5.1] C3a E1: PBE on {c3_ndone}/{len(c3_ps)} screened paths; of the {len(c3_pend)} "
          f"without, {len(c3_copy)} are symmetry copies of a path with PBE and {len(c3_big)} have a "
          f"modulated cell above the 12-atom cap (18-216 atoms)")
    c3_off = int((~c3u.all_paths_match_cache.astype(bool)).sum())
    check(int(c3u.n_paths_pbe.min()) == 1 and int(c3u.n_paths_pbe.max()) == 11
          and int(c3u.deciding_mode_has_pbe.sum()) == 100 and c3_off == 40,
          f"[S5.1] C3a E1: 1-{int(c3u.n_paths_pbe.max())} PBE paths per unit; deciding mode has "
          f"PBE in {int(c3u.deciding_mode_has_pbe.sum())}/120; {c3_off} units with an off-cache path")
    c3_dr_off = (c3_ps[c3_ps.role.isin(["decide", "ref"]) & (c3_ps.cache_check_pass == False)]  # noqa: E712
                 [["system", "path_model"]].drop_duplicates())
    c3_dr_units = int(sum((r_.system, r_.model) in set(map(tuple, c3_dr_off.values))
                          for r_ in c3u.itertuples()))
    check(c3_dr_units == 28,
          f"[S5.1] C3a: {c3_dr_units} ladder units have an off-cache deciding/reference path")
    c3_unc = c3_big.groupby(["system", "path_model"]).size()
    c3_ps_st = c3u[c3u.pbe_backed_stable.astype(bool).values
                   & c3u.set_index(["system", "model"]).index.isin(c3_unc.index)]
    check(len(c3_ps_st) == 19 and int((~c3_ps_st.p_ok).sum()) == 3,
          f"[S5.1] C3a E1: {len(c3_ps_st)} PBE-stable ladder units have a mode above the cap "
          f"without PBE ({int((~c3_ps_st.p_ok).sum())} of them PBE-wrong)")
    qj = pd.read_csv(dft + "qe_jobs.csv") if os.path.exists(dft + "qe_jobs.csv") else None
    if qj is not None:
        jc = qj.iloc[:, 0].astype(str)
        n_c3 = int(jc.str.match(r"(a|ax|b)_").sum())
        check(n_c3 == 981 and int(jc.str.match(r"a_").sum()) == 380
              and int(jc.str.match(r"ax_").sum()) == 533 and int(jc.str.match(r"b_").sum()) == 68,
              f"[2.6/S5] C3a/C3b: {n_c3} PBE calculations (380 deciding/reference, 533 other "
              "modes, 68 SSCHA samples)")

    # ==== E2: checks on the persistent PBE errors (scripts/dft_checks.py analyze-checks; Table S23)
    chk = json.load(open("results/revision/dft_checks/summary.json"))
    cvd = {p_["system"]: p_ for p_ in chk["conv"]["paths"]}
    cv_rel = {s_: {v_: cvd[s_]["variants"][v_]["depth_rel_change"] for v_ in ("k", "e", "ke")}
              for s_ in cvd}
    check(chk["conv"]["verdict"] == "NOT converged"
          and [s_ for s_ in cvd if not cvd[s_]["converged"]] == ["cssnbr3_cubic"]
          and cvd["cssnbr3_cubic"]["failed_variants"] == ["k", "ke"]
          and abs(cv_rel["cssnbr3_cubic"]["k"] + 0.068) < 1e-3
          and all(cvd[s_]["variants"][v_]["calls_identical"] for s_ in cvd for v_ in ("k", "e", "ke"))
          and max(abs(x) for s_ in ("batio3_cubic", "knbo3_cubic") for x in cv_rel[s_].values()) < 0.02
          and cvd["cssnbr3_cubic"]["min_at_edge"],
          "[3.2/S5.3] E2 conv: NOT converged by the pre-set criterion, on one path only (CsSnBr3/MACE "
          f"R path, k 0.15: depth {cv_rel['cssnbr3_cubic']['k']:+.1%}, minimum at the scan edge); "
          "BaTiO3/KNbO3 within 2 %; every call identical at every T for all 9 variants")
    xc = chk["xc"]
    check((xc["pbe_errors"], xc["pbe_errors_fixed_by_pbesol"], xc["new_errors_from_pbesol"],
           xc["pbe_errors_also_wrong_in_pbesol"], xc["n_unit_rows_ladder_complete"])
          == (15, 9, 0, 6, 60)
          and xc["by_system"] == {"batio3_cubic": {"pbe_errors": 1, "pbesol_errors": 1},
                                  "cssnbr3_cubic": {"pbe_errors": 9, "pbesol_errors": 1},
                                  "knbo3_cubic": {"pbe_errors": 5, "pbesol_errors": 4}},
          "[3.2/S5.3] E2 xc: PBEsol (PBE pseudopotentials, MLIP geometry) fixes 9 of 15 PBE errors "
          "over 60 units (8 CsSnBr3, KNbO3/SevenNet-0 600 K), 0 new; left: BaTiO3/ORB-v2 300 K, "
          "KNbO3 600 K x4, CsSnBr3/ORB-v2 300 K")
    xu = pd.read_csv("results/revision/dft_checks/xc_unit_calls.csv")
    xu = xu[xu.in_ladder.astype(bool) & xu.unit_complete.astype(bool)
            & (xu.system == "cssnbr3_cubic")]
    xgt = xu.gt_stable.astype(bool)
    xpe, xso = xu.pbe_backed_stable.astype(bool) != xgt, xu.pbesol_backed_stable.astype(bool) != xgt
    x600 = xu["T"] == 600
    check(int(xpe.sum()) == 9 and int((xpe & ~xso).sum()) == 8 and int((xpe & x600).sum()) == 4
          and int((xpe & x600 & xso).sum()) == 0
          and int((xpe & (xu["T"] == 300)).sum()) == 5 and int((xpe & (xu["T"] == 300) & ~xso).sum()) == 4,
          "[Section 4/R1.6] E2: CsSnBr3 PBE errors 9; PBEsol corrects 8, every one at 600 K (4) and 4 of "
          "the 5 at 300 K")
    xcp = pd.read_csv("results/revision/dft_checks/xc_paths.csv")
    xcs, xcf = xcp[xcp.system == "cssnbr3_cubic"], xcp[xcp.system != "cssnbr3_cubic"]
    check(len(xcp) == 23 and abs(xcs.depth_ratio_pbesol_over_pbe.min() - 0.317) < 5e-4
          and abs(xcs.depth_ratio_pbesol_over_pbe.max() - 0.422) < 5e-4
          and bool(xcs.pbe_min_at_edge.all()) and not bool(xcs.pbesol_min_at_edge.any())
          and abs(xcf.depth_ratio_pbesol_over_pbe.min() - 1.027) < 5e-4
          and abs(xcf.depth_ratio_pbesol_over_pbe.max() - 1.304) < 5e-4,
          f"[3.2/S5.3] E2 xc: PBEsol/PBE depth {xcs.depth_ratio_pbesol_over_pbe.min():.2f}-"
          f"{xcs.depth_ratio_pbesol_over_pbe.max():.2f} on the 8 CsSnBr3 paths (minimum moves "
          f"inside the scan), {xcf.depth_ratio_pbesol_over_pbe.min():.2f}-"
          f"{xcf.depth_ratio_pbesol_over_pbe.max():.2f} on the 15 BaTiO3/KNbO3 paths")
    pl = chk["pbe_lattice"]["systems"]
    off = {s_: pl[s_]["lattice"]["mlip_minus_pbe_pct"] for s_ in pl}
    check(all(pl[s_]["phase_A"] == "done" and pl[s_]["phase_B"] == "done"
              and set(pl[s_]["phase_C"].values()) == {"complete"} for s_ in pl)
          and abs(min(min(v_.values()) for v_ in off.values()) - 0.119) < 1e-3
          and abs(max(max(v_.values()) for v_ in off.values()) - 0.797) < 1e-3
          and (round(min(off["batio3_cubic"].values()), 2), round(max(off["batio3_cubic"].values()), 2)) == (0.26, 0.31)
          and (round(min(off["knbo3_cubic"].values()), 2), round(max(off["knbo3_cubic"].values()), 2)) == (0.74, 0.80),
          "[Section 3.2/Section 4/S5.3] E2 pbe-lattice phases A, B, C done; MLIP lattices "
          + ", ".join(f"{s_.split('_')[0]} +{min(v_.values()):.2f} to +{max(v_.values()):.2f} %"
                      for s_, v_ in off.items()) + " above PBE")
    # PL (phases B/C): PBE at its own lattice along its own band at MACE-MP-0's deciding q
    plc = pd.read_csv("results/revision/dft_checks/pl_calls.csv").set_index(["system", "T"])
    plp = pd.read_csv("results/revision/dft_checks/pl_profile.csv").set_index(["system", "curve"])
    dep = lambda s_, c_: round(float(plp.loc[(s_, c_), "depth_meV"]), 1)  # noqa: E731
    check((dep("batio3_cubic", "pbe_lattice_deciding"), dep("knbo3_cubic", "pbe_lattice_deciding"),
           dep("cssnbr3_cubic", "pbe_lattice_deciding"), dep("batio3_cubic", "pbe_mlip_lattice"),
           dep("knbo3_cubic", "pbe_mlip_lattice"), dep("cssnbr3_cubic", "pbe_mlip_lattice"),
           dep("batio3_cubic", "pbe_lattice_softest"), dep("knbo3_cubic", "pbe_lattice_softest"))
          == (48.7, 30.0, 131.2, 73.5, 87.8, 127.2, 21.8, 32.2),
          "[3.2/S5.3] PL depths: PBE lattice 48.7 / 30.0 / 131.2 meV v MACE-MP-0 lattice 73.5 / 87.8 / "
          "127.2 meV; Gamma 21.8 / 32.2 meV")
    for s_ in ("batio3_cubic", "knbo3_cubic"):
        r3 = plc.loc[(s_, 300.0)]
        check(not r3.gt_stable and bool(r3.pbe_lattice_deciding_stable) and bool(r3.pbe_lattice_softest_stable)
              and not bool(r3.pbe_mlip_lattice_stable) and bool(r3.mlip_ledger_stable)
              and not bool(plc.loc[(s_, 100.0)].pbe_lattice_deciding_stable),
              f"[abstract/3.2/4/S5.3] PL {s_} 300 K: PBE at its own lattice calls stable (wrong, both "
              "profiled modes), PBE at MACE-MP-0's lattice unstable (right); 100 K deciding mode condenses")
    check(bool(plc.loc[("batio3_cubic", 100.0)].pbe_lattice_softest_stable)
          and not bool(plc.loc[("knbo3_cubic", 100.0)].pbe_lattice_softest_stable),
          "[S5.3] PL 100 K: the Gamma mode alone would not condense in BaTiO3, does in KNbO3")
    check(all(bool(plc.loc[(s_, T_)].pbe_lattice_deciding_error) and bool(plc.loc[(s_, T_)].pbe_mlip_lattice_error)
              for s_, T_ in (("knbo3_cubic", 600.0), ("cssnbr3_cubic", 300.0), ("cssnbr3_cubic", 600.0))),
          "[S5.3] PL: KNbO3 600 K and CsSnBr3 300/600 K mis-called at both lattices")
    im = {s_: [round(m_["freq_thz"], 2) for m_ in pl[s_]["profiles"]["deciding"]["imaginary_modes"]] for s_ in pl}
    check(im == {"batio3_cubic": [-7.44, -6.63, -5.95], "knbo3_cubic": [-6.97, -5.47, -4.17],
                 "cssnbr3_cubic": [-0.7, -0.69]}
          and max(pl[s_]["gamma_acoustic_max_abs_thz"] for s_ in pl) < 2e-7,
          f"[S5.3] PL PBE imaginary modes at the PBE lattice {im}; acoustic sum < 2e-7 THz")
    cp_ = pd.read_csv("results/revision/dft/c3a_paths.csv")
    md_ = {s_: float(cp_[(cp_.stem == pl[s_]["mlip_path_stem"]) & (cp_.curve == "mace_mp0")].depth_meV.iloc[0])
           for s_ in ("batio3_cubic", "knbo3_cubic")}
    check((round(md_["batio3_cubic"], 1), round(md_["knbo3_cubic"], 1)) == (42.6, 27.9),
          "[3.2/S5.3] MACE-MP-0's own wells at its own lattice on the same paths: 42.6 / 27.9 meV")
    # audit 2026-10-09 M3: own-equilibrium ratio MACE-MP-0 / PBE within 15 % on both systems
    rat_ = {s_: md_[s_] / float(plp.loc[(s_, "pbe_lattice_deciding"), "depth_meV"]) for s_ in md_}
    check(all(0.85 <= r_ <= 1.0 for r_ in rat_.values()),
          f"[3.2 M3] own-lattice MACE-MP-0/PBE well ratio within 15 % ({ {k: round(v, 2) for k, v in rat_.items()} })")
    # audit 2026-10-09 M8: Hessian bootstrap on the ten BaTiO3/KNbO3 100 K converged false-stables
    bs_ = {}
    for s_ in ("batio3_cubic", "knbo3_cubic"):
        for m_ in ("chgnet", "mace_mp0", "mattersim", "orb_v2", "sevennet0"):
            r_ = json.load(open(f"results/revision/sscha_converged_grid/{s_}_{m_}_100K_sc222_startA.json",
                                encoding="utf-8"))["run"]["bootstrap"]
            bs_[(s_, m_)] = r_
    cb_ = bs_[("batio3_cubic", "chgnet")]
    check(all(b_["B"] == 10 and b_["n_ok"] == 10 for b_ in bs_.values())
          and round(cb_["frac_unstable"], 2) == 0.6 and round(cb_["mean_thz"], 2) == -0.33
          and round(cb_["full_ensemble_min_nonac_thz"], 2) == 1.65 and round(cb_["std_thz"], 2) == 1.50
          and all(b_["frac_unstable"] == 0 for k_, b_ in bs_.items() if k_ != ("batio3_cubic", "chgnet")),
          "[3.3 M8] bootstrap B = 10 on all ten units; BaTiO3/CHGNet 6/10 resamples unstable (mean -0.33, "
          "full +1.65 +- 1.50 THz); the other nine 0/10")
    # audit 2026-10-09 M1: at the PBE lattice the X mode (q = (1/2,0,0)) is imaginary and was NOT
    # profiled; the two profiled modes are the deciding M band and the softest (Gamma) mode
    for s_ in ("batio3_cubic", "knbo3_cubic"):
        prof_q = {tuple(sorted(pl[s_]["profiles"][k_]["q"])) for k_ in ("deciding", "softest")}
        xm = [m_ for m_ in pl[s_]["profiles"]["deciding"]["imaginary_modes"] if sorted(m_["q"]) == [0.0, 0.0, 0.5]]
        fr = sorted(m_["freq_thz"] for m_ in pl[s_]["profiles"]["deciding"]["imaginary_modes"])
        check(prof_q == {(0.0, 0.0, 0.0), (0.0, 0.5, 0.5)} and len(xm) == 1 and fr[0] < xm[0]["freq_thz"] < fr[2],
              f"[3.2/4/5 M1] {s_}: X mode ({xm[0]['freq_thz']:.2f} THz) imaginary at the PBE lattice, not "
              "profiled, between the two profiled modes in frequency")
    qc = pd.read_csv("results/revision/dft_checks/qe_jobs.csv")
    n_chk = int((qc.job.str.match(r"(cv|xs|pl)_") & (qc.status == "ok")).sum())
    check(n_chk == 382 and len(qc) == 382, f"[2.6/S5.3] E2: {n_chk} check calculations (cv_, xs_, pl_), all ok")
    gl = {}
    for m_ in ("chgnet", "mace_mp0", "mattersim", "orb_v2", "sevennet0"):
        for s_ in ("batio3_cubic", "knbo3_cubic", "srtio3_cubic", "cssnbr3_cubic"):
            fp = f"{dft}geom/checks/{s_}_{m_}.json"
            if os.path.exists(fp):
                cell = np.array(json.load(open(fp))["relaxed_prim_cell"], dtype=float)
                gl.setdefault(s_, []).append(abs(np.linalg.det(cell)) ** (1 / 3))
    spread = {s_: 100 * (max(v_) - min(v_)) / min(v_) for s_, v_ in gl.items() if len(v_) == 5}
    check(len(spread) == 4 and max(spread[s_] for s_ in spread if s_ != "cssnbr3_cubic") < 0.065
          and abs(spread["cssnbr3_cubic"] - 0.45) < 0.01,
          "[S5.1] MLIP lattice spread across models: oxide perovskites "
          + ", ".join(f"{s_.split('_')[0]} {v_:.3f} %" for s_, v_ in spread.items()) + " (CsSnBr3 0.45 %)")

    c3b = pd.read_csv(dft + "c3b_units.csv")
    c3_own3 = c3b[c3b.is_owner & (c3b["set"] == "sscha")].set_index("system")
    c3_base = c3b[c3b.is_owner & (c3b["set"] == "baseline")].set_index("system")
    c3_exp3 = {"batio3_cubic": (0.100, 0.104), "zro2_cubic": (0.225, 0.175),
               "zr_bcc": (0.596, 0.439), "srtio3_cubic": (0.113, 0.190)}
    check(all(abs(c3_own3.loc[k, "rel_force_rmse_baseline"] - b0) < 5e-4
              and abs(c3_base.loc[k, "rel_force_rmse"]
                      - c3_own3.loc[k, "rel_force_rmse_baseline"]) < 1e-9
              and abs(c3_own3.loc[k, "rel_force_rmse"] - s0) < 5e-4
              for k, (b0, s0) in c3_exp3.items())
          and set(c3b[c3b["set"] == "baseline"].n_configs) == {4}
          and set(c3b[c3b["set"] == "sscha"].n_configs) == {12},
          "[R1.2] C3b owner relative force RMSE baseline -> SSCHA: "
          + ", ".join(f"{k} {c3_own3.loc[k, 'rel_force_rmse_baseline']:.3f}->"
                      f"{c3_own3.loc[k, 'rel_force_rmse']:.3f}" for k in c3_exp3)
          + " (4 baseline + 12 SSCHA configs per set)")
    c3_sr = c3_own3.loc["srtio3_cubic"]
    c3_ratio = c3_sr.rel_force_rmse / c3_sr.rel_force_rmse_baseline
    check(abs(c3_sr.rel_force_rmse - 0.190) < 5e-4 and abs(c3_ratio - 1.67) < 5e-3
          and abs(c3_sr.u_rms_mean_A - 0.337) < 5e-4
          and abs(c3_sr.e_err_rms_meV_per_atom - 15.1) < 0.05
          and abs(c3_sr.e_err_maxabs_meV_per_atom - 44.3) < 0.05,
          f"[R1.2] C3b SrTiO3/MACE 600 K: owner relative force RMSE {c3_sr.rel_force_rmse:.3f} = "
          f"{c3_ratio:.2f}x baseline, u_rms {c3_sr.u_rms_mean_A:.3f} A, energy error RMS "
          f"{c3_sr.e_err_rms_meV_per_atom:.1f} / max {c3_sr.e_err_maxabs_meV_per_atom:.1f} meV/atom")

    # ==== Revision C1 / C5 (R1.4): the SSCHA seed study with the PRODUCTION recipe (Table S21).
    # Recomputed from the unit JSONs that scripts/sscha_seed_study.py wrote under
    # results/revision/sscha_seeds/ (not from their stored 'summary' blocks), and the ledger values
    # the JSONs carry are cross-checked against the deposited ledger. None of these runs is a
    # converged SSCHA; the converged-recipe numbers (C1c) are pinned where that table lands.
    seed_dir = "results/revision/sscha_seeds/"

    def seed_unit(tag):
        with open(seed_dir + tag + ".json", encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc, [doc["seeds"][k] for k in sorted(doc["seeds"], key=int)]

    def hess_of(runs, key="hessian"):
        return np.array([s[key]["min_nonac_thz"] for s in runs])

    def ledger_row(system, model, temp):
        x = ss[(ss.system == system) & (ss.model == model) & (ss.temperature_K == temp)]
        return float(x.min_eff_freq_thz.iloc[0]) if len(x) == 1 else None

    d_ba, s_ba = seed_unit("batio3_cubic_mace_mp0_100K_sc222")
    d_zo, s_zo = seed_unit("zro2_cubic_mace_mp0_100K_sc222")
    d_zb, s_zb = seed_unit("zr_bcc_mattersim_50K_sc222")
    d_sr, s_sr = seed_unit("srtio3_cubic_mace_mp0_600K_sc222")
    c1_runs = s_ba + s_zo + s_zb + s_sr
    c1_shape = [(s["relax"]["n_populations"], s["relax"]["n_populations_moving_dyn"],
                 s["relax"]["n_steps_total"], s["relax"]["n_steps_kept_total"]) for s in c1_runs]
    check(len(c1_runs) == 16 and all(s["status"] == "ok" for s in c1_runs)
          and not any(s["relax"]["converged"] for s in c1_runs)
          and c1_shape == [(8, 1, 27, 19)] * 16
          and all(p["stop_reason"] == "max_ka_cumulative" for s in c1_runs
                  for p in s["relax"]["populations"])
          and all(s["stopping_criteria"]["max_ka"] == 20
                  and s["stopping_criteria"]["n_configs_per_population"] == 256 for s in c1_runs),
          "[R1.4] seed study: 0 of 16 production-recipe runs converged (4 seeds x 4 units); each "
          "draws 8 populations of 256, takes 27 steps, keeps 19, and only population 1 moves the "
          "matrix; every population ends at the cumulative max_ka = 20")
    placeholder = all(
        len({round(x["gc_err"], 9) for x in s["relax"]["steps"]}) == 1
        and abs(s["relax"]["steps"][-1]["gc_err"] - s["dyn_meta"]["nat_prim"] * np.sqrt(3) / 4) < 1e-8
        for s in c1_runs)
    ratio = [s["relax"]["steps"][-1]["gc"] / (s["relax"]["steps"][-1]["gc_err"] * 1e-4) for s in c1_runs]
    check(placeholder and min(ratio) > 700 and max(ratio) < 1.1e4,
          f"[R1.4] the recorded gradient error is a constant 0.433 per primitive-cell atom (the "
          f"library placeholder); the last gradient is {min(ratio):.0f}-{max(ratio):.0f} times the "
          f"convergence threshold on all 16 runs")

    # BaTiO3 / MACE 100 K: the Hessian is the FPD start (the Table S2 false-stable).
    h_ba, st_ba = hess_of(s_ba), hess_of(s_ba, "start_post_fpd")
    check(abs(h_ba.min() - 2.867) < 5e-4 and abs(h_ba.max() - 2.881) < 5e-4
          and abs(h_ba.mean() - 2.874) < 5e-4 and np.ptp(st_ba) < 1e-9 and abs(st_ba[0] - 2.881) < 1e-3
          and np.abs(h_ba - st_ba).max() < 0.02 and bool((h_ba >= -0.1).all())
          and all(abs(s["harmonic_pre_fpd"]["min_nonac_thz"] + 6.299) < 1e-3 for s in s_ba),
          f"[R1.4] BaTiO3/MACE 100 K: Hessian {h_ba.min():+.3f} to {h_ba.max():+.3f} THz (mean "
          f"{h_ba.mean():+.3f}) against the FPD start {st_ba[0]:+.3f} (harmonic -6.299); all four "
          f"seeds stable")
    # SrTiO3 / MACE 600 K: the opposite case, the Hessian is nowhere near the start.
    h_sr, st_sr = hess_of(s_sr), hess_of(s_sr, "start_post_fpd")
    led_sr = d_sr["ledger"]["min_eff_freq_thz"]
    check(abs(h_sr.min() + 20.264) < 5e-3 and abs(h_sr.max() + 15.774) < 5e-3
          and bool((h_sr < -0.1).all()) and np.ptp(st_sr) < 1e-9 and abs(st_sr[0] - 0.904) < 1e-3
          and abs(h_sr[0] + 18.714) < 5e-3 and abs(led_sr + 20.229) < 5e-3
          and h_sr.min() <= led_sr <= h_sr.max()
          and np.abs(h_sr - st_sr).min() > 16.0,
          f"[R1.4] SrTiO3/MACE 600 K: seeds {h_sr.min():+.2f} to {h_sr.max():+.2f} THz, all unstable, "
          f"{np.abs(h_sr - st_sr).min():.0f}+ THz from the FPD start {st_sr[0]:+.2f}; seed 0 "
          f"{h_sr[0]:+.2f} against ledger {led_sr:+.2f} (inside the seed range)")
    h_zo, st_zo = hess_of(s_zo), hess_of(s_zo, "start_post_fpd")
    h_zb, ax_zb = hess_of(s_zb), hess_of(s_zb, "final_aux")
    check(abs(h_zo.min() - 3.069) < 5e-4 and abs(h_zo.max() - 3.087) < 5e-4
          and abs(st_zo[0] - 2.962) < 1e-3 and bool((h_zo >= -0.1).all())
          and abs(h_zb.min() - 1.636) < 5e-4 and abs(h_zb.max() - 1.786) < 5e-4
          and float(np.abs(h_zb - ax_zb).max()) < 1e-9
          and all(s["bootstrap"]["std_thz"] < 1e-12 for s in s_zb)
          and abs(float(h_zb.std(ddof=1)) - 0.0754) < 5e-4,
          f"[R1.4] ZrO2/MACE 100 K Hessian {h_zo.min():+.3f} to {h_zo.max():+.3f} (start "
          f"{st_zo[0]:+.3f}); bcc-Zr/MatterSim 50 K {h_zb.min():+.3f} to {h_zb.max():+.3f} (seed SD "
          f"{float(h_zb.std(ddof=1)):.4f}), equal to the final auxiliary matrix's with bootstrap SD "
          f"< 1e-12 (the bootstrap is uninformative)")
    # Seed 0 against the deposited ledger: reproduced for three units, not for SrTiO3 600 K.
    repro = [bool(d["seeds"]["0"]["ledger_comparison"]["reproduces_ledger"])
             for d in (d_ba, d_zo, d_zb, d_sr)]
    led_xc = [ledger_row("batio3_cubic", "mace_mp0", 100.0), ledger_row("zro2_cubic", "mace_mp0", 100.0),
              ledger_row("zr_bcc", "mattersim", 50.0), ledger_row("srtio3_cubic", "mace_mp0", 600.0)]
    check(repro == [True, True, True, False]
          and all(x is not None and abs(x - d["ledger"]["min_eff_freq_thz"]) < 1e-9
                  for x, d in zip(led_xc, (d_ba, d_zo, d_zb, d_sr))),
          f"[R1.4] seed 0 reproduces the deposited ledger on BaTiO3, ZrO2 and bcc-Zr (50 K) but not "
          f"SrTiO3 600 K (ledger values {', '.join(f'{x:+.3f}' for x in led_xc)} match the parquet)")

    # C5: bcc-Zr cell size, 3x3x3 re-measured (seed 0) against the canonical 2x2x2 ledger rows.
    c5 = {(m, t): seed_unit(f"zr_bcc_{m}_{t}K_sc333") for m in ("mace_mp0", "mattersim") for t in (100, 300)}
    c5_h = {k: v[1][0]["hessian"]["min_nonac_thz"] for k, v in c5.items()}
    c5_l = {k: ledger_row("zr_bcc", k[0], float(k[1])) for k in c5}
    check(all(len(v[1]) == 1 and v[1][0]["status"] == "ok" for v in c5.values())
          and abs(c5_h[("mace_mp0", 100)] - 1.562) < 1e-3 and abs(c5_h[("mace_mp0", 300)] - 1.547) < 1e-3
          and abs(c5_h[("mattersim", 100)] - 1.320) < 1e-3
          and abs(c5_h[("mattersim", 300)] + 2.106) < 1e-3
          and abs(c5_l[("mace_mp0", 100)] - 1.800) < 1e-3 and abs(c5_l[("mace_mp0", 300)] - 1.797) < 1e-3
          and abs(c5_l[("mattersim", 100)] - 1.846) < 1e-3 and abs(c5_l[("mattersim", 300)] - 1.953) < 1e-3
          and all(c5_h[k] < c5_l[k] for k in c5),
          "[R1.4] C5 bcc-Zr 3x3x3 against 2x2x2 ledger: MACE-MP-0 " + ", ".join(
              f"{t} K {c5_l[('mace_mp0', t)]:+.3f} -> {c5_h[('mace_mp0', t)]:+.3f}" for t in (100, 300))
          + "; MatterSim " + ", ".join(
              f"{t} K {c5_l[('mattersim', t)]:+.3f} -> {c5_h[('mattersim', t)]:+.3f}" for t in (100, 300))
          + " THz (production recipe)")
    c5_conv = {k: bool(v[1][0]["relax"]["converged"]) for k, v in c5.items()}
    check(c5_conv == {("mace_mp0", 100): True, ("mace_mp0", 300): True,
                      ("mattersim", 100): False, ("mattersim", 300): False}
          and all(c5[("mace_mp0", t)][1][0]["relax"]["n_populations"] == 1 for t in (100, 300))
          and all((c5[("mattersim", t)][1][0]["relax"]["n_populations"],
                   c5[("mattersim", t)][1][0]["relax"]["n_populations_moving_dyn"],
                   c5[("mattersim", t)][1][0]["relax"]["n_steps_kept_total"]) == (8, 1, 19)
                  for t in (100, 300)),
          "[R1.4] C5: the two MACE-MP-0 3x3x3 runs satisfy the library's test in one population "
          "(harmonic matrix already positive); the two MatterSim 3x3x3 runs end at the cumulative cap "
          "(8 populations, 19 kept steps) like the C1 units")


    # ==== converged-recipe SSCHA (Table S22, Sections 2.5, 3.3, 3.5): results/revision/grid_compare.json
    # (scripts/grid_compare.py) and the grid's summary.csv / unit JSONs, recomputed where possible.
    import hashlib
    gcj = json.load(open("results/revision/grid_compare.json", encoding="utf-8"))
    gsum = pd.read_csv("results/revision/sscha_converged_grid/summary.csv")
    gsum = gsum[gsum.start == "A"]
    check(hashlib.sha256(open("results/revision/sscha_converged_grid/summary.csv", "rb").read())
          .hexdigest() == gcj["meta"]["summary_csv"]["sha256"]
          and gcj["meta"]["grid_vs_ledger"]["all_ok"]
          and gcj["meta"]["production_reproduces_stats_hardening"]["all_ok"],
          "[S22] grid_compare.json was built from the summary.csv on disk; both self-checks pass")
    rs = gcj["run_status"]
    check((rs["n_planned_units"], rs["n_ok"], rs["n_converged"], len(rs["failed"]),
           len(rs["blowups"]), len(rs["wall_cap_timeouts"])) == (178, 171, 159, 7, 4, 1)
          and (gsum.status == "running").sum() == 0,
          "[2.5/S22] grid: 178 planned, 171 ok, 159 converged, 7 failed, 4 blow-ups, 1 wall cap")
    check(rs["by_model"]["orb_v2"]["converged"] == 18 and rs["by_model"]["orb_v2"]["blowups"] == 4
          and not any(b_["converged"] for b_ in rs["blowups"]),
          "[3.3/S22] ORB-v2: 18/33 converged; all 4 blow-ups are unconverged ORB-v2 units")
    cvg = gcj["variants"]["converged_only"]
    check(cvg["fs57_outcomes"]["outcomes"]
          == {"still_false_stable": 21, "ok_not_converged": 4, "now_unstable": 32},
          "[3.3] of the 57 production false-stables: 21 survive, 32 turn unstable, 4 unconverged")
    fsu = cvg["fs57_units"]
    flu = [u_ for u_ in fsu if u_["system"] in FLUORITE]
    check(sorted({u_["system"] for u_ in fsu if u_["outcome"] == "still_false_stable"})
          == ["batio3_cubic", "knbo3_cubic"]
          and sum(u_["outcome"] == "now_unstable" for u_ in flu) == 29
          and not any(u_["outcome"] == "still_false_stable" for u_ in flu),
          "[3.3] surviving false-stables are BaTiO3/KNbO3 only; no fluorite false-stable survives")
    fsx = [u_ for u_ in fsu if u_["model"] != "orb_v2"]
    check(sum(u_["outcome"] == "still_false_stable" for u_ in fsx) == 18 and len(fsx) == 50,
          "[3.3] without ORB-v2: 18 of 50 production false-stables survive")
    ci = cvg["converged_matched"]["i"]
    check(ci["all_models"]["sscha_false_stable_of_label_unstable"]["short"] == "34/84"
          and ci["excl_orb_v2"]["sscha_false_stable_of_label_unstable"]["short"] == "30/74"
          and ci["all_models"]["screen_call_unstable_on_those"]["short"] == "17/34"
          and ci["excl_orb_v2"]["screen_call_unstable_on_those"]["short"] == "14/30",
          "[abstract/3.3] converged false-stable 34/84 (30/74); screen unstable on 17/34 (14/30)")
    gok = gsum[(gsum.status == "ok") & (gsum.converged == True)]  # noqa: E712
    gnb = gok[gok.family != "bcc"]
    cfs2 = gnb[(gnb.gt_stable == False) & (gnb.stable_call == True)]  # noqa: E712
    check(cfs2.groupby("system").size().to_dict()
          == {"batio3_cubic": 8, "knbo3_cubic": 15, "pbtio3_cubic": 7, "srtio3_cubic": 4}
          and set(cfs2[cfs2.system == "pbtio3_cubic"].T_K) == {300.0, 600.0}
          and set(cfs2[cfs2.system == "srtio3_cubic"].T_K) == {100.0},
          "[3.3] converged false-stables: BaTiO3 8, KNbO3 15, PbTiO3 7 (300/600 K), SrTiO3 4 (100 K)")
    b100 = gnb[gnb.system.isin(["batio3_cubic", "knbo3_cubic"]) & (gnb.T_K == 100)]
    check(len(b100) == 10 and bool(b100.stable_call.astype(bool).all())
          and not bool(b100.screen_stable_call.astype(bool).any()),
          "[3.3/Section 4] all ten BaTiO3/KNbO3 100 K units: converged SSCHA stable, screen unstable")
    cii = cvg["converged_matched"]["ii"]
    check(cii["all_models"]["sscha"]["recall"]["short"] == "4/26"
          and cii["all_models"]["softmode"]["recall"]["short"] == "15/26"
          and cii["excl_orb_v2"]["sscha"]["recall"]["short"] == "4/23"
          and cii["excl_orb_v2"]["softmode"]["recall"]["short"] == "12/23"
          and cii["all_models"]["sscha"]["n_numerical_blowup"] == 0,
          "[abstract/3.3/Fig5] FE recall T<=300 K, converged units: SSCHA 4/26, screen 15/26")
    fe_hit = gnb[gnb.system.isin(FE_OXIDE) & (gnb.T_K <= 300) & (gnb.gt_stable == False)  # noqa: E712
                 & (gnb.stable_call == False)]  # noqa: E712
    check(set(fe_hit.system) == {"pbtio3_cubic"} and set(fe_hit.T_K) == {100.0} and len(fe_hit) == 4
          and fe_hit.hessian_min_thz.min() > -0.4,
          "[3.3] the 4 converged FE recalls are all PbTiO3 at 100 K, within 0.4 THz of zero")
    civ = cvg["converged_matched"]["iv"]["all_models"]["false_unstable_by_T"]
    civx = cvg["converged_matched"]["iv"]["excl_orb_v2"]["false_unstable_by_T"]
    check([civ[t]["short"] for t in ("300", "600", "900")] == ["0/4", "3/12", "3/18"]
          and [civx[t]["short"] for t in ("300", "600", "900")] == ["0/4", "3/11", "3/16"]
          and set(civ["600"]["by_system"]) == set(civ["900"]["by_system"]) == {"cssni3_cubic"},
          "[3.3/S17] converged high-T false-unstables 0/4, 3/12, 3/18 (ex-ORB 0/4, 3/11, 3/16), "
          "all CsSnI3")
    cs_fu = gnb[(gnb.system == "cssni3_cubic") & (gnb.gt_stable == True)]  # noqa: E712
    check(len(cs_fu) == 6 and not cs_fu.stable_call.astype(bool).any()
          and abs(cs_fu.hessian_min_thz.max() + 0.32) < 0.005
          and abs(cs_fu.hessian_min_thz.min() + 1.09) < 0.005,
          f"[3.3] converged CsSnI3 false-unstables {cs_fu.hessian_min_thz.max():.2f} to "
          f"{cs_fu.hessian_min_thz.min():.2f} THz")
    cs_lo = gnb[(gnb.system == "cssni3_cubic") & (gnb.T_K <= 300)]
    check(len(cs_lo) == 8 and not cs_lo.stable_call.astype(bool).any(),
          "[3.3] converged SSCHA calls CsSnI3 unstable in all 8 units at T <= 300 K")
    sto = gnb[gnb.system == "srtio3_cubic"]
    sto_hi = sto[sto.T_K > 105]
    check(bool(sto.stable_call.astype(bool).all()) and len(sto) == 15
          and set(sto[sto.T_K == 100].model) == {"chgnet", "mace_mp0", "mattersim", "sevennet0"}
          and abs(sto[sto.T_K == 100].hessian_min_thz.min() - 1.06) < 0.005
          and abs(sto[sto.T_K == 100].hessian_min_thz.max() - 1.20) < 0.005
          and len(sto_hi) == 11 and abs(sto_hi.hessian_min_thz.min() - 1.71) < 0.005
          and abs(sto_hi.hessian_min_thz.max() - 2.76) < 0.005,
          "[3.3/3.5/R3.2] converged SSCHA calls SrTiO3 stable in all 15 converged units: 100 K "
          "+1.06 to +1.20 THz, above 105 K +1.71 to +2.76 THz (0/11 false-unstable)")
    s100 = gnb[(gnb.T_K == 100) & (gnb.stable_call == True)]  # noqa: E712
    turn = [(r_.system, r_.model) for r_ in s100.itertuples()
            if (~gnb[(gnb.system == r_.system) & (gnb.model == r_.model)
                     & (gnb.T_K > 100)].stable_call.astype(bool)).any()]
    check(len(s100) == 14 and not turn,
          "[3.3] none of the 14 non-bcc units converged SSCHA calls stable at 100 K turns negative")
    fl_c = gnb[gnb.system.isin(FLUORITE)]
    fl_all = gsum[gsum.system.isin(FLUORITE) & (gsum.status == "ok")]
    check(len(fl_c) == 38 and not fl_c.stable_call.astype(bool).any() and len(fl_all) == 40
          and not fl_all.stable_call.astype(bool).any()
          and len(fl_c[fl_c.T_K <= 300]) == 18,
          "[3.3] converged SSCHA calls every converged fluorite unit unstable (38; the 2 unconverged "
          "ORB-v2 100 K units are HfO2 -49.4 THz and a ZrO2 blow-up; 18 at T <= 300 K)")
    ciii = cvg["converged_matched"]["iii"]
    check(ciii["all_models"]["call_agreement"]["short"] == "33/41"
          and ciii["excl_orb_v2"]["call_agreement"]["short"] == "32/36"
          and ciii["all_models"]["per_model_call"]["mattersim"]["short"] == "7/9",
          "[3.3] bcc call agreement, converged 3x3x3: 33/41 (32/36); MatterSim 7/9")
    pv = cvg["converged_matched"]["v"]
    pc, pcx = pv["all_models"]["displacive_combined"], pv["excl_orb_v2"]["displacive_combined"]
    check((pc["n_paired_units"], pc["screen_right_sscha_wrong"], pc["sscha_right_screen_wrong"])
          == (44, 13, 2)
          and (pcx["n_paired_units"], pcx["screen_right_sscha_wrong"], pcx["sscha_right_screen_wrong"])
          == (39, 10, 2)
          and pc["clustered_by_system"]["p_exact_clustered"] == 0.5
          and pcx["clustered_by_system"]["p_exact_clustered"] == 0.5
          and pc["clustered_by_system"]["per_cluster_net"] == [5, 6, 0, 0, 0],
          "[3.3/Section 4/Section 5/S10] converged paired contrast T<=300 K: 13 v 2 on 44 units (10 v 2 on 39), "
          "system-clustered p = 0.5; nets +5 BaTiO3, +6 KNbO3, 0 PbTiO3, 0, 0")
    pp = gcj["production_full"]["v"]["all_models"]["displacive_combined"]
    check((pp["n_paired_units"], pp["screen_right_sscha_wrong"], pp["sscha_right_screen_wrong"],
           pp["clustered_by_system"]["p_exact_clustered"]) == (47, 33, 3, 0.125),
          "[3.3] production paired contrast reproduced: 33 v 3 on 47 units, p = 0.125")
    fresh_bad = [t_ for t_ in gok.unit_tag if not json.load(open(
        f"results/revision/sscha_converged_grid/{t_}_startA.json", encoding="utf-8"))
        ["run"]["fresh_gradient_check"]["consistent_with_minimum"]]
    check(len(fresh_bad) == 7 and not set(fresh_bad) & set(cfs2.unit_tag),
          "[2.5/Section 4/S22] 7 of 159 converged units fail the fresh-ensemble check; no false-stable "
          "among them")

    def ab(tag, d_="sscha_converged"):
        return json.load(open(f"results/revision/{d_}/{tag}_AB.json", encoding="utf-8"))
    ba, ba1 = ab("batio3_cubic_mace_mp0_100K_sc222"), ab("batio3_cubic_mace_mp0_100K_sc222",
                                                           "sscha_converged_b1")
    zo, st1 = ab("zro2_cubic_mace_mp0_100K_sc222"), ab("srtio3_cubic_mace_mp0_100K_sc222")
    z50, z2, z3 = (ab("zr_bcc_mattersim_50K_sc222"), ab("zr_bcc_mattersim_300K_sc222"),
                   ab("zr_bcc_mattersim_300K_sc333"))
    st6 = ab("srtio3_cubic_mace_mp0_600K_sc222")
    r2 = lambda x: round(x, 2)  # noqa: E731
    check(r2(ba["A"]["hessian_min_thz"]) == 2.03 and ba["B"]["status"] == "failed"
          and ba1["B"]["converged"] and r2(ba1["B"]["hessian_min_thz"]) == 1.85
          and abs(ba1["B"]["start_min_thz"] - 1.0) < 1e-6,
          "[3.3/3.5] C1c BaTiO3/MACE 100 K: A +2.03, B(0.3) failed, B(1.0) +1.85, all converged")
    check(zo["A"]["converged"] and zo["B"]["converged"]
          and abs(zo["A"]["hessian_min_thz"] + 22.37) < 0.01
          and abs(zo["B"]["hessian_min_thz"] + 25.97) < 0.01,
          "[3.3] C1c ZrO2/MACE 100 K: A -22.4, B -26.0 (unstable, as labelled)")
    check((r2(st1["A"]["hessian_min_thz"]), r2(st1["B"]["hessian_min_thz"])) == (1.13, 1.02)
          and st1["verdict"] == "converged but START-DEPENDENT" and st1["same_stability_call"],
          "[3.5/R3.2] C1c SrTiO3/MACE 100 K: +1.13 / +1.02, same call, flagged start-dependent")
    check(r2(st6["A"]["hessian_min_thz"]) == 2.44
          and r2(float(gsum[gsum.unit_tag == "srtio3_cubic_mace_mp0_600K_sc222"]
                       .hessian_min_thz.iloc[0])) == 2.41,
          "[3.3/3.5] SrTiO3/MACE 600 K converged: +2.44 (converged-mode run) / +2.41 (grid)")
    st9 = ab("srtio3_cubic_mace_mp0_900K_sc222")
    check(r2(st9["A"]["hessian_min_thz"]) == 2.71
          and r2(float(gsum[gsum.unit_tag == "srtio3_cubic_mace_mp0_900K_sc222"]
                       .hessian_min_thz.iloc[0])) == 2.76,
          "[R1.2 letter] SrTiO3/MACE 900 K converged: +2.71 (converged-mode run) / +2.76 (grid)")
    chc = gcj["changed_calls"]
    check((chc["n_compared_converged"], chc["converged_no_blowup_changed"]["n"],
           chc["nonbcc_corrected_vs_label"], chc["nonbcc_worsened_vs_label"]) == (159, 72, 51, 12),
          "[R1.4 letter] converged recipe changes 72 of the 154 converged calls with a non-blow-up "
          "production value; non-bcc 51 wrong->right, 12 right->wrong")
    check((r2(z50["A"]["hessian_min_thz"]), r2(z50["B"]["hessian_min_thz"])) == (0.41, 0.41)
          and (r2(z2["A"]["hessian_min_thz"]), r2(z2["B"]["hessian_min_thz"])) == (0.92, 0.93)
          and (r2(z3["A"]["hessian_min_thz"]), r2(z3["B"]["hessian_min_thz"])) == (-0.92, -0.95)
          and z2["both_fresh_consistent"] is False,
          "[3.5] C5 converged Zr/MatterSim: 50 K +0.41; 300 K 2x2x2 +0.92/+0.93 -> 3x3x3 -0.92/-0.95")
    v4 = json.load(open("results/revision/sscha_seeds/batio3_cubic_mace_mp0_100K_sc222.json",
                        encoding="utf-8"))["v4"]
    check(v4["status"] == "finished" and abs(v4["v4_min_nonac_thz"] - 2.8784) < 5e-4
          and abs(v4["v4_false_parent_thz"] - 2.8783) < 5e-4 and 9800 < v4["wall_s"] < 9900,
          "[3.3/S2.2] C1b include_v4=True: +2.878 v +2.878 THz bubble, finished in under 9,900 s")
    s444 = {}
    for t_ in ("hfo2_cubic_mace_mp0_600K", "srtio3_cubic_chgnet_100K", "srtio3_cubic_mace_mp0_100K",
               "srtio3_cubic_mattersim_100K", "srtio3_cubic_sevennet0_100K"):
        with open(f"results/revision/sscha_converged_sc444/{t_}_sc444_startA.json",
                  encoding="utf-8") as fh:
            s444[t_] = json.load(fh)["run"]
    h444 = s444["hfo2_cubic_mace_mp0_600K"]
    check(h444["relax"]["converged"] is True and h444["relax"]["n_populations"] == 7
          and r2(h444["hessian"]["min_nonac_thz"]) == -3.22
          and h444["fresh_gradient_check"]["consistent_with_minimum"] is True
          and "bootstrap" not in h444["hessian"],
          "[3.5/S2.4/R3.2] 4x4x4 HfO2/MACE-MP-0 600 K: relaxation converged (7 pops), Hessian -3.22 THz, no bootstrap")
    sto = {k: v for k, v in s444.items() if k.startswith("srtio3")}
    check(all(not (v.get("hessian") or {}).get("min_nonac_thz") for v in sto.values())
          and [sto[k]["relax"]["converged"] for k in sorted(sto)] == [True, False, False, False]
          and [sto[k]["relax"]["n_populations"] for k in sorted(sto)] == [5, 14, 14, 10],
          "[3.5/S2.4/R3.2] 4x4x4 SrTiO3 100 K: no Hessian on any of 4 (CHGNet relax converged in 5; 14/14/10 not)")
    tim = gsum[(gsum.system == "ti_bcc") & (gsum.model == "mace_mp0") & (gsum.converged == True)]  # noqa: E712
    check([r2(x) for x in tim.sort_values("T_K").hessian_min_thz] == [1.57, 1.66, 1.71],
          "[3.3] Ti/MACE-MP-0 converged 3x3x3 hardens +1.57 -> +1.71 THz (100 -> 600 K)")
    with open("paper/manuscript.md", encoding="utf-8") as fh:
        ms_txt = fh.read()
    text_vs_data(ms_txt)
    # ==== E3: pre-registered replicates of the converged grid (Table S24, Sections 3.5, S2.4, R1.4)
    e3 = pd.read_csv("results/revision/e3_replicates/summary.csv")
    e3j = json.load(open("results/revision/e3_replicates/summary.json", encoding="utf-8"))
    sel = pd.read_csv("scripts/box/as_run/e3_units_2026-10-08.tsv", sep="\t")
    sel_tags = set(sel["unit_tag"]) if "unit_tag" in sel.columns else set(sel.iloc[:, 0])
    check(len(e3) == 65 and set(e3.unit_tag) == sel_tags and (e3.why == "disagree").sum() == 53
          and (e3.why == "random").sum() == 12,
          "[E3] 65 units replicated = the registered selection (53 disagreeing + 12 random)")
    check((e3.A_status == "ok").all() and (e3.A10_status == "ok").all()
          and e3.B_status.value_counts().to_dict() == {"ok": 51, "failed": 13, "missing": 1},
          "[E3/Section 3.5] seed 10 finished 65/65; start B 51 ok, 13 failed, 1 wrote no result")

    def _berr(t_, d_):
        p_ = f"results/revision/{d_}/{t_}_startB.json"
        if not os.path.exists(p_):
            return None
        r_ = json.load(open(p_, encoding="utf-8")).get("run") or {}
        return "ok" if r_.get("status") == "ok" else str(r_.get("error"))
    ret = e3[e3.b_retry_1thz.astype(bool)]
    first = {t_: _berr(t_, "sscha_converged_grid_startB") for t_ in e3.unit_tag}
    second = {t_: _berr(t_, "sscha_converged_grid_startB1") for t_ in ret.unit_tag}
    check(len(ret) == 19 and all("dynamical matrix is complex" in first[t_] for t_ in ret.unit_tag)
          and sum(v_ == "ok" for v_ in second.values()) == 12
          and sum("dynamical matrix is complex" in (v_ or "") for v_ in second.values()) == 7,
          "[E3/Section 3.5] 19 start-B units hit the complex-dynamical-matrix assertion at 0.3 THz; at 1.0 THz 12 "
          "finished and 7 failed again")
    oth = [first[t_] for t_ in e3.unit_tag if t_ not in second and first[t_] not in (None, "ok")]
    check(len(oth) == 6 and sum("same type" in x_ for x_ in oth) == 5 and sum("q star" in x_ for x_ in oth) == 1,
          "[E3/S2.4] 6 other start-B failures (5 'same type' structure checks, 1 q-star mismatch), not retried")
    al = e3j["all"]
    check((al["n_complete"], al["n_same_call_all_three"], len(al["unresolved"]), len(al["incomplete"]))
          == (51, 43, 8, 14) and abs(al["median_range_thz"] - 0.22) < 0.005,
          "[E3/Section 3.5] 51 complete, 43 same call, 8 unresolved, 14 incomplete; median range 0.22 THz")
    xe = e3[(e3.model != "orb_v2") & (e3.n_ok == 3)]
    check(len(xe) == 39 and int((xe.verdict == "agree").sum()) == 32,
          "[E3/Section 3.5] without ORB-v2: same call 32/39")
    ur = e3[e3.verdict == "unresolved"]
    ur7 = ur[ur.unit_tag != "ti_bcc_orb_v2_600K_sc333"]
    check(len(ur7) == 7 and set(ur7.model) == {"chgnet", "mattersim"}
          and set(ur7.system) == {"pbtio3_cubic", "knbo3_cubic"} and set(ur7.T_K) == {300.0, 600.0}
          and (ur7.B_converged.astype(str) == "False").all() and set(ur7.B_stop_reason) == {"wall_cap", "max_pop"}
          and round(ur7.B_hessian_min_thz.max()) == -210 and round(ur7.B_hessian_min_thz.min()) == -6230
          and ur7.A_call_stable.all() and ur7.A10_call_stable.all()
          and (ur7.A_converged.astype(str) == "True").all() and (ur7.A10_converged.astype(str) == "True").all(),
          "[E3/Section 3.5] 7 unresolved PbTiO3/KNbO3 CHGNet/MatterSim units: start B a non-converged blow-up "
          "(-209 to -6230 THz, wall/population cap); start A and seed 10 converged and stable")
    ti = e3.set_index("unit_tag").loc["ti_bcc_orb_v2_600K_sc333"]
    check(round(ti.A_hessian_min_thz, 2) == 2.53 and round(ti.B_hessian_min_thz, 1) == -26.4
          and round(ti.A10_hessian_min_thz, 2) == -3.73 and str(ti.B_converged) == "False"
          and str(ti.A10_converged) == "False",
          "[E3/Section 3.5] Ti/ORB-v2 600 K: +2.53 / -26.4 / -3.73 THz, the last two not converged")
    both = e3[(e3.A_status == "ok") & (e3.A10_status == "ok")]
    check(int((both.A_call_stable == both.A10_call_stable).sum()) == 64
          and e3j["converged_in_all_three"]["n_units"] == 31
          and e3j["converged_in_all_three"]["n_same_call_all_three"] == 31,
          "[E3/Section 3.5, not pre-registered] seed 10 = start A call on 64/65; 31/31 agree where all three converged")
    b10 = e3[e3.system.isin(["batio3_cubic", "knbo3_cubic"]) & (e3.T_K == 100)]
    bok = b10[b10.B_status == "ok"]
    bch = b10[(b10.model == "chgnet") & (b10.system == "batio3_cubic")]
    check(len(b10) == 10 and b10.A10_call_stable.all() and (b10.A10_converged.astype(str) == "True").all()
          and len(bok) == 7 and bok.B_call_stable.all() and int((bok.B_converged.astype(str) == "True").sum()) == 4
          and round(float(bch.A10_hessian_min_thz.iloc[0]), 2) == 1.56,
          "[Section 3.3] BaTiO3/KNbO3 100 K: seed 10 stable+converged on 10/10; start B stable on 7 (4 converged); "
          "BaTiO3/CHGNet seed 10 +1.56 THz")

    # ==== E5: pre-registered fine-tuning trial (Sections 4, S6, Tables S25-S26, R2.1)
    f30 = json.load(open("results/revision/finetune_mace30/summary.json", encoding="utf-8"))
    f6 = json.load(open("results/revision/finetune/summary.json", encoding="utf-8"))
    reps = ("seed0", "seed1", "seed2")

    def _cell(base, v):
        r_ = [v["replicates"][s_]["pred_stable"] for s_ in reps]
        return "changed" if all(x != base for x in r_) else "unchanged" if all(x == base for x in r_) else "unresolved"
    m30 = f30["P2"]["models"]["mace_mp0"]
    ch = f6["P2"]["models"]["chgnet"]
    m6 = f6["P2"]["models"]["mace_mp0"]
    for nm, blk, bk in (("MACE-MP-0 30 ep", m30, lambda v: v["base_call_ledger"]),
                        ("CHGNet", ch, lambda v: v["base_rerun"]["pred_stable"]),
                        ("MACE-MP-0 6 ep", m6, lambda v: v["base_rerun"]["pred_stable"])):
        for c_ in ("batio3_cubic@300", "knbo3_cubic@300"):
            check(_cell(bk(blk[c_]), blk[c_]) == "unchanged" and blk[c_]["status"] == "refuted"
                  and not any(blk[c_]["replicates"][s_]["pred_stable"] == blk[c_]["label_stable"] for s_ in reps),
                  f"[E5] {nm} {c_}: mis-call not corrected in any replicate (refuted)")
        for c_ in ("knbo3_cubic@600", "cssnbr3_cubic@300"):
            check(_cell(bk(blk[c_]), blk[c_]) == "unchanged" and blk[c_]["status"] == "supported",
                  f"[E5] {nm} {c_}: mis-call persists 3/3 (supported)")
    check(_cell(m30["cssnbr3_cubic@600"]["base_call_ledger"], m30["cssnbr3_cubic@600"]) == "unchanged"
          and m30["cssnbr3_cubic@600"]["status"] == "supported"
          and _cell(ch["cssnbr3_cubic@600"]["base_rerun"]["pred_stable"], ch["cssnbr3_cubic@600"]) == "unresolved",
          "[E5] CsSnBr3 600 K: persists for MACE-MP-0 (30 ep), unresolved for CHGNet")
    for nm, blk, b_ in (("30 ep", m30, m30["cssnbr3_cubic@900"]["base_call_ledger"]),
                        ("6 ep", m6, m6["cssnbr3_cubic@900"]["base_rerun"]["pred_stable"])):
        v_ = blk["cssnbr3_cubic@900"]
        check(b_ and v_["label_stable"] and _cell(b_, v_) == "changed",
              f"[E5] MACE-MP-0 {nm}: CsSnBr3 900 K correct at base, wrong in 3/3 replicates (not predicted)")
    ncor = [sum(m30[k_]["replicates"][s_]["pred_stable"] == m30[k_]["label_stable"] for k_ in m30) for s_ in reps]
    check(ncor == [6, 6, 6] and sum(m30[k_]["base_call_ledger"] == m30[k_]["label_stable"] for k_ in m30) == 7
          and len(m30) == 12,
          "[E5/S6] MACE-MP-0 30 ep: correct calls of 12 base 7, replicates 6/6/6")
    p1 = f30["P1"]["mace_mp0"]["primary_own_deciding"]
    p1c = f6["P1"]["chgnet"]["primary_own_deciding"]
    check(p1["verdict"] == "supported" and round(p1["median_over_paths_and_replicates"], 2) == 0.99
          and round(p1["base_median_ratio"], 2) == 0.45
          and all(0.99 <= round(x, 2) <= 1.01 for x in p1["replicate_median_ratio"].values())
          and p1c["verdict"] == "unresolved" and round(p1c["median_over_paths_and_replicates"], 2) == 1.04
          and round(p1c["replicate_median_ratio"]["seed0"], 2) == 1.33,
          "[E5/Section 4] P1: MACE-MP-0 30 ep 0.45 -> 0.99 (supported); CHGNet pooled 1.04, seed 0 1.33 (unresolved)")
    p3 = f30["P3"]
    check(p3["status"] == "supported" and p3["called_stable"] and p3["relax_converged"]
          and round(p3["min_nonac_thz"], 2) == 1.73 and not p3["label_stable"],
          "[E5/Section 4/Section 5] P3: converged SSCHA on fine-tuned MACE-MP-0 calls BaTiO3 100 K stable, +1.73 THz")
    s2b = f6["S2"]["mace_mp0"]["controls"]
    s2r = f30["S2"]["mace_mp0"]["controls"]
    check(all(s2b[k_]["base"]["pred_stable"] and all(s2r[k_]["replicates"][s_]["pred_stable"] for s_ in reps)
              for k_ in s2r) and len(s2r) == 6
          and (f6["S2"]["chgnet"]["n_unchanged"], f6["S2"]["chgnet"]["n_unresolved"]) == (2, 4),
          "[E5] S2: MACE-MP-0 30 ep keeps 6/6 control calls; CHGNet 2 unchanged, 4 unresolved")
    s1 = f30["S1"]["mace_mp0"]["c3a"]
    fold = [s1["base"][s_]["window_force_rmse_eV_A"] / s1[r_][s_]["window_force_rmse_eV_A"]
            for s_ in ("batio3_cubic", "knbo3_cubic") for r_ in reps]
    s1c = f6["S1"]["chgnet"]["c3a"]
    foldc = [s1c["base"][s_]["window_force_rmse_eV_A"] / s1c[r_][s_]["window_force_rmse_eV_A"]
             for s_ in ("batio3_cubic", "knbo3_cubic") for r_ in reps]
    check(round(min(fold)) == 6 and round(max(fold)) == 15 and round(min(foldc), 1) == 1.3
          and round(max(foldc), 1) == 6.4,
          f"[S6] S1 well-window force RMSE falls {min(fold):.1f}-{max(fold):.1f}x (MACE-MP-0 30 ep), "
          f"{min(foldc):.1f}-{max(foldc):.1f}x (CHGNet)")
    nc = f30["P1"]["mace_mp0"]["negative_control_cssnbr3_own_deciding"]
    check(round(nc["median_over_paths_and_replicates"], 2) == 1.48 and round(nc["base_median_ratio"], 2) == 0.92,
          "[S6] CsSnBr3 well depth overshoots PBE after fine-tuning: 0.92 -> 1.48")
    import re as _re
    for s_ in reps:
        lg = open(f"results/revision/finetune_mace30/models/mace_mp0/{s_}/logs/ft_mace_mp0_{s_}_run-{s_[-1]}.log",
                  encoding="utf-8").read()
        ep = sorted({int(x) for x in _re.findall(r"INFO: Epoch (\d+):", lg)})
        sel_ = _re.findall(r"Loading checkpoint: .*epoch-(\d+)\.pt", lg)
        check(ep == list(range(30)) and bool(sel_) and int(sel_[-1]) == {"seed0": 12, "seed1": 11, "seed2": 13}[s_],
              f"[S6/D4] MACE-MP-0 {s_}: 30 epochs trained, checkpoint epoch {sel_[-1] if sel_ else None} selected")
    rb6 = open("results/revision/finetune/train/box_sequence.sh", encoding="utf-8").read()
    rb30 = open("results/revision/finetune_mace30/train/box_sequence.sh", encoding="utf-8").read()
    check(rb6.count("--max_num_epochs=6") == 3 and rb30.count("--max_num_epochs=30") == 3,
          "[Section 4/S6/D4] first MACE-MP-0 run 6 epochs (disclosed); re-run 30 epochs")
    # post hoc lattice check (scripts/finetune_lattice_diag.py)
    ld = json.load(open("results/revision/finetune_mace30/lattice_diag.json", encoding="utf-8"))
    apbe = {"batio3_cubic": 4.02374737, "knbo3_cubic": 4.026883539}
    rows_ = [(s_, r_, v_) for s_, vv in ld["systems"].items() for r_, v_ in vv.items() if r_ != "base"]
    d_pbe = [100 * (v_["a_relaxed_A"] / apbe[s_] - 1) for s_, r_, v_ in rows_]
    d_base = [100 * (v_["a_relaxed_A"] / ld["systems"][s_]["base"]["a_relaxed_A"] - 1) for s_, r_, v_ in rows_]
    check(len(rows_) == 6 and max(abs(x) for x in d_pbe) < 0.3 and all(-0.53 < x < -0.46 for x in d_base)
          and all(v_["own_lattice"]["pred_stable"] and not v_["base_lattice"]["pred_stable"]
                  and v_["own_lattice_reproduces_deposited_call"] for s_, r_, v_ in rows_)
          and round(min(v_["own_lattice"]["well_depth_meV"] for _, _, v_ in rows_)) == 28
          and round(max(v_["own_lattice"]["well_depth_meV"] for _, _, v_ in rows_)) == 34
          and round(min(v_["base_lattice"]["well_depth_meV"] for _, _, v_ in rows_)) == 49
          and round(max(v_["base_lattice"]["well_depth_meV"] for _, _, v_ in rows_)) == 64
          and all(-2.8 <= round(v_["pressure_at_base_lattice_GPa"], 1) <= -2.6 for _, _, v_ in rows_),
          "[Section 4/S6 post hoc] 30-ep MACE-MP-0 relaxes ~0.5 % below base, within 0.3 % of PBE; 300 K stable "
          "at own lattice (6/6, reproduces P2), unstable at base lattice (6/6); wells 28-34 v 49-64 meV")
    wsha = {}
    for l_ in open("results/revision/finetune_mace30/models/WEIGHTS.sha256", encoding="utf-8").read().splitlines():
        h_, p_ = l_.split(None, 1)
        wsha[p_.strip()] = h_
    check(all(ld["weights_sha256"][s_] == wsha[f"./mace_mp0/{s_}/ft_mace_mp0_{s_}.model"] for s_ in reps),
          "[S6] lattice check ran on the archived 30-epoch weights (sha256 match)")
    s4n = " ".join(ms_txt.split("## 4.", 1)[1].split("## 5.", 1)[0].split())
    for frag in ("rose from 0.45 to 0.99 for MACE-MP-0", "pooled median is 1.04 but one replicate's is 1.33",
                 "about 0.5 % smaller than the base model, to within 0.3 % of the PBE lattice",
                 "(+1.73 THz)", "trained 6 epochs instead of the registered 30"):
        check(frag in s4n, f"[text Section 4 E5] '{frag}'")
    # audit 2026-10-09: the new numbers in the text, each pinned to data above (M1, M3, M8)
    s32n = " ".join(ms_txt.split("### 3.2", 1)[1].split("### 3.3", 1)[0].split())
    s33n = " ".join(ms_txt.split("### 3.3", 1)[1].split("### 3.4", 1)[0].split())
    xs_ = {s_: next(m_["freq_thz"] for m_ in pl[s_]["profiles"]["deciding"]["imaginary_modes"]
                    if sorted(m_["q"]) == [0.0, 0.0, 0.5]) for s_ in ("batio3_cubic", "knbo3_cubic")}
    for frag in (f"at X ({xs_['batio3_cubic']:.2f} THz in BaTiO₃, {xs_['knbo3_cubic']:.2f} THz in KNbO₃".replace("-", "−"),
                 f"({md_['batio3_cubic']:.1f} against {dep('batio3_cubic', 'pbe_lattice_deciding'):.1f} meV and "
                 f"{md_['knbo3_cubic']:.1f} against {dep('knbo3_cubic', 'pbe_lattice_deciding'):.1f} meV)"):
        check(frag in s32n, f"[text Section 3.2 audit] '{frag}'")
    frag = (f"B = {cb_['B']} resamples of the Hessian ensemble, has a mean of "
            f"{cb_['mean_thz']:.2f} THz, and {round(cb_['frac_unstable'] * cb_['B'])} of the {cb_['B']} "
            "resamples are unstable").replace("-0.", "−0.")
    check(frag in s33n, f"[text Section 3.3 audit M8] '{frag}'")
    abs_txt = ms_txt.split("## Abstract", 1)[1].split("## 1.", 1)[0]
    n_words = len(abs_txt.split())
    check(n_words <= 350, f"[abstract] {n_words} words (350: the RSC Advances portal limit, read 2026-10-10)")

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
