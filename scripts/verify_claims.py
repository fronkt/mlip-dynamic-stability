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
          f"[3.5] SSCHA calls all 10 fluorite units stable in the SAME cell "
          f"(+{fq.min_eff_freq_thz.min():.1f} to +{fq.min_eff_freq_thz.max():.1f} THz)")

    ma2 = A.method_agreement_summary(d[d.system.isin(BCC) & (d.model != "orb_v2")])
    check(abs(ma2["sign_agreement"] - 0.833) < 0.005,
          f"[3.3] bcc curvature-sign agreement excl ORB {ma2['sign_agreement']:.3f}")
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
          f"[3.3] SSCHA non-bcc false-unstables by T {fu_by_t(nb)} (ex-ORB {fu_by_t(nbx)})")
    st = nb[(nb.system == "srtio3_cubic") & nb.gt_stable.astype(bool)]
    stx = st[st.model != "orb_v2"]
    check(len(st) == 14 and not st.pred_stable.astype(bool).any()
          and abs(st.min_eff_freq_thz.min() + 913.736) < 5e-3
          and abs(st.min_eff_freq_thz.max() + 3.392) < 5e-3
          and len(stx) == 11 and not stx.pred_stable.astype(bool).any(),
          f"[3.3] SrTiO3 above Tc: SSCHA false-unstable in all {len(st)} units "
          f"({st.min_eff_freq_thz.max():.1f} to {st.min_eff_freq_thz.min():.0f} THz); ex-ORB 11/11")

    def reversals(x):
        p = x.pivot_table(index=["system", "model"], columns="temperature_K",
                          values="min_eff_freq_thz")
        s100 = p[p[100.0] >= 0]
        return len(s100), int(((s100[600.0] < 0) | (s100[900.0] < 0)).sum())

    check(reversals(nb) == (23, 14) and reversals(nbx) == (19, 10),
          f"[3.3] SSCHA: {reversals(nb)[1]} of {reversals(nb)[0]} non-bcc units stable at 100 K "
          f"turn negative by 600-900 K (ex-ORB {reversals(nbx)[1]} of {reversals(nbx)[0]})")

    # [3.3] Local vs global criterion on the SAME PES: on the non-bcc units SSCHA calls stable
    # against an unstable label, the screen's own symmetric-point curvature (the single-mode
    # analogue of the SSCHA Hessian) is positive on 52/57, and on 41/57 it is positive while the
    # screen's free-energy comparison calls the phase unstable. Recomputed from the ledger.
    kk = ["system", "model", "temperature_K"]
    cb = d[d.method == "softmode"][kk + ["min_eff_freq_thz", "pred_stable", "gt_stable"]].merge(
        d[d.method == "sscha"][kk + ["min_eff_freq_thz", "pred_stable"]], on=kk,
        suffixes=("_scr", "_ss"))
    cb = cb[~cb.system.str.contains("bcc")]
    cfs = cb[cb.pred_stable_ss.astype(bool) & ~cb.gt_stable.astype(bool)]
    cpos = int((cfs.min_eff_freq_thz_scr > 0).sum())
    cblind = int(((cfs.min_eff_freq_thz_scr > 0) & ~cfs.pred_stable_scr.astype(bool)).sum())
    check(len(cfs) == 57 and cpos == 52 and cblind == 41,
          f"[3.3] SSCHA false-stables: screen curvature positive on {cpos}/{len(cfs)}, "
          f"curvature-positive-but-condensed on {cblind}/{len(cfs)}")

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
