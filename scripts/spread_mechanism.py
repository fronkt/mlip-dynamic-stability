"""Why force-level committee spread stops flagging wrong finite-T calls once the overlapped-atom
configurations are removed (RSC Advances R2: the referee asks to tone down the ensemble
"guardrail").

POST-HOC. Written 2026-10-21, after scripts/force_spread.py's pre-registered result was known
(primary AUC 0.681, clustered CI [0.416, 0.908], verdict "not shown"; variant (g) on the 43
overlap-free units 0.314). This script explains that result. It does not re-score the primary,
promote a secondary, or test a new guardrail, and every number in it is descriptive.

Hypothesis under test (tested here, not assumed): committee spread across the five foundation
MLIPs (CHGNet, MACE-MP-0, MatterSim, SevenNet-0, ORB-v2) is an extrapolation detector that
fires when interatomic distances fall into the overlap region, while in the moderately
anharmonic region that decides stability calls the models share a common error (similar PBE
training data) that their variance cannot see.

Three layers, all from committed data; no model and no DFT code is run here.

1. Configuration level. The 960 thermally displaced configurations of force_spread.py (15
   systems x 4 temperatures x 16 draws). Per-configuration spread computed exactly as that
   script's primary statistic (sqrt of the mean over atoms and Cartesian components of the
   population variance across the five production models), and the per-unit median of it is
   checked against force_spread._spread and units_consensus.csv. Binned by the pre-registered
   pair-resolved contact ratio r (closest pair / shortest reference distance of the same
   species pair); r < 0.75 is the variant (g) overlap rule.
2. Unit level. The 60 consensus units of force_spread.py, split by variant (g) into the 43
   overlap-free units and the 17 overlap units. Spread against "consensus call wrong", and
   every wrong unit listed with each model's own call.
3. PBE set. C3b of scripts/dft_reference.py: 64 configurations x 5 models, each with PBE
   forces. Per configuration, with Fbar the across-model mean force and everything averaged
   over atoms and Cartesian components,
       mean_m |F_m - F_PBE|^2  =  |Fbar - F_PBE|^2  +  mean_m |F_m - Fbar|^2
          (total MSE)             (shared part)        (spread part)
   is an exact identity, and the spread part is the square of the force_spread statistic. The
   shared fraction is therefore the share of the committee's squared force error that its own
   spread cannot see. Two kinds of ensemble per system, labelled from the geometry files
   themselves (dft_reference.stage_c3b_inputs):
     baseline  the undisplaced ideal supercell rattled by ase.Atoms.rattle (iid Gaussian,
               stdev 0.02 A per Cartesian component, seeds 20260926 + k, k = 0..3);
               near-equilibrium, i.e. the harmonic region.
     sscha     12 configurations drawn from the owner model's own C1 SSCHA ensemble at T
               (scripts/sscha_seed_study.py, 3 per seed x 4 seeds): the thermally sampled,
               anharmonic region the finite-T calls are made in.
   The same r < 0.75 rule, against the unit's undisplaced reference supercell, marks overlap.

Reference values for an uninformative committee (5 models, independent equal-variance errors,
no shared bias): shared fraction 1/M = 0.20, and RMS(Fbar - F_PBE) / spread = 1/sqrt(M - 1)
= 0.50. They are written to the output beside the measured values.

Usage:  python scripts/spread_mechanism.py
Output: results/revision/r2/spread_mechanism.json (never touches results/ledger.*)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

import dft_reference as D                             # noqa: E402
import force_spread as FS                             # noqa: E402
from mlip_dynstab import stats as S                   # noqa: E402

SCRIPT = "scripts/spread_mechanism.py"
FS_DIR = REPO / "results" / "revision" / "force_spread"
DFT_DIR = REPO / "results" / "revision" / "dft"
OUT = REPO / "results" / "revision" / "r2" / "spread_mechanism.json"

MODELS = FS.PRODUCTION                    # the five production models only
assert tuple(sorted(MODELS)) == tuple(sorted(D.MODELS))
MIN_RATIO = FS.MIN_DIST_RATIO             # 0.75, the pre-registered variant (g) rule
N_BOOT = 10000
N_PERM = 10000
SEED = 0
POS_TOL = 1e-3                            # dft_reference analyze default (--pos-tol)
# pair-ratio bins, lower bound inclusive: lo <= r < hi
BINS = (("r >= 0.85", 0.85, np.inf), ("0.75 <= r < 0.85", 0.75, 0.85),
        ("0.65 <= r < 0.75", 0.65, 0.75), ("0.50 <= r < 0.65", 0.50, 0.65),
        ("r < 0.50", -np.inf, 0.50))

# values from the reconnaissance pass that motivated this script; compared, never asserted
RECON = {
    "config_bin_median": {"r >= 0.85": 0.058, "0.75 <= r < 0.85": 0.063,
                          "0.65 <= r < 0.75": 0.105, "0.50 <= r < 0.65": 0.29,
                          "r < 0.50": 2.11},
    "config_rho_spread_amplitude_nonoverlap": (-0.06, 763),
    "unit_rho": {"all": 0.25, "clean": -0.19, "overlap": 0.55},
    "unit_auc": {"clean": 0.314, "overlap": 0.819},
    "pbe_shared_fraction": {"baseline": (0.32, 0.47), "sscha_ex_srtio3": (0.49, 0.60),
                            "srtio3_sscha": 0.88},
    "pbe_rho_pooled": 0.918, "pbe_rho_pooled_nonoverlap": 0.894,
    "pbe_rho_within": {"batio3_cubic": 0.23, "zro2_cubic": 0.62, "srtio3_cubic": 0.87,
                       "zr_bcc": 0.19},
    "pbe_error_over_spread": {"nonoverlap": 0.99, "overlap": 2.72},
}

_READ: set[Path] = set()


def _track(p: Path) -> Path:
    _READ.add(Path(p).resolve())
    return p


# ------------------------------------------------------------------ stats ----

def spearman(a, b) -> float:
    """Spearman rho as the Pearson correlation of average ranks (pandas rank; no scipy).

    stats.spearman goes through pandas Series.corr(method="spearman"), which imports scipy.
    """
    a = pd.Series(np.asarray(a, float))
    b = pd.Series(np.asarray(b, float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    ra = a[ok].rank(method="average").to_numpy()
    rb = b[ok].rank(method="average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_perm(a, b, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Two-sided permutation p for rho (+1 correction), permuting single rows. Used only
    inside one system, where the rows are that system's configurations."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = spearman(a, b)
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = (ra - ra.mean()) / np.linalg.norm(ra - ra.mean())
    rng = np.random.default_rng(seed)
    perm = np.stack([rb[rng.permutation(rb.size)] for _ in range(n_perm)])
    perm = perm - perm.mean(axis=1, keepdims=True)
    null = perm @ ra / np.linalg.norm(perm, axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))
    return {"rho": _r(obs, 3), "p_perm": _r(p, 4), "n": int(a.size), "n_perm": n_perm}


def cluster_boot(stat, cluster, n_boot: int = N_BOOT, seed: int = SEED,
                 alpha: float = 0.05) -> dict:
    """Percentile CI of any row statistic, resampling whole clusters (systems) with
    replacement: the resampling of stats.cluster_bootstrap_auc with the statistic swapped."""
    _, idx = S._blocks(np.asarray(cluster))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        vals[i] = stat(np.concatenate([idx[j] for j in pick]))
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return {"ci_lo": None, "ci_hi": None, "n_boot": 0, "n_clusters": len(idx)}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_lo": _r(lo, 3), "ci_hi": _r(hi, 3), "n_boot": int(vals.size),
            "n_clusters": len(idx)}


def rho_block(x, y, cluster) -> dict:
    x, y, cluster = np.asarray(x, float), np.asarray(y, float), np.asarray(cluster)
    out = {"rho": _r(spearman(x, y), 3), "n": int(x.size),
           "n_systems": int(len(set(cluster.tolist())))}
    out.update(cluster_boot(lambda r: spearman(x[r], y[r]), cluster))
    return out


def _r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return round(x, n) if np.isfinite(x) else None


def _q(v) -> dict:
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": _r(q50), "q25": _r(q25), "q75": _r(q75),
            "iqr": _r(q75 - q25), "min": _r(v.min()), "max": _r(v.max())}


# ------------------------------------------------- 1. configuration level ----

def load_force_spread():
    units = pd.read_csv(_track(FS_DIR / "units_consensus.csv"))
    per = pd.read_csv(_track(FS_DIR / "units_per_model.csv"))
    summary = json.loads(_track(FS_DIR / "summary.json").read_text(encoding="utf-8"))
    units = units.sort_values(["system", "T"]).reset_index(drop=True)
    systems = sorted(units["system"].unique())
    configs = {}
    for s in systems:
        p = _track(FS_DIR / "configs" / f"configs_{s}.npz")
        with np.load(p) as z:
            configs[s] = {"disp": z["disp"], "pair_ratio": z["pair_ratio"],
                          "temps": [float(t) for t in z["temps_K"]],
                          "meta": json.loads(str(z["meta"]))}
    forces = {}
    for m in MODELS:
        p = _track(FS_DIR / "forces" / f"forces_{m}.npz")
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            if meta["tag"] != m:
                raise SystemExit(f"{p.name}: tag {meta['tag']} is not {m}")
            forces[m] = {}
            for s in systems:
                if meta["systems"][s]["config_hash"] != configs[s]["meta"]["config_hash"]:
                    raise SystemExit(f"{p.name}: {s} evaluated on other configurations")
                forces[m][s] = z[f"{s}__F"]
    return units, per, summary, systems, configs, forces


def config_table(systems, configs, forces) -> pd.DataFrame:
    rows = []
    for s in systems:
        F = np.stack([forces[m][s] for m in MODELS])          # (M, nT, nC, N, 3)
        with np.errstate(invalid="ignore"):
            # force_spread._spread, per configuration (population std across models)
            spread = np.sqrt(np.mean(F.std(axis=0) ** 2, axis=(2, 3)))       # (nT, nC)
        finite = np.isfinite(F).all(axis=(0, 3, 4))
        u = configs[s]["disp"]
        u_rms = np.sqrt((u ** 2).sum(axis=3).mean(axis=2))                # (nT, nC), A
        for ti, T in enumerate(configs[s]["temps"]):
            for c in range(u.shape[1]):
                rows.append({"system": s, "T": T, "cfg": c,
                             "spread": float(spread[ti, c]),
                             "finite_all_models": bool(finite[ti, c]),
                             "u_rms_A": float(u_rms[ti, c]),
                             "pair_ratio": float(configs[s]["pair_ratio"][ti, c])})
    return pd.DataFrame(rows)


def check_config_level(cfg, units, forces, configs) -> dict:
    """The per-configuration spread must reproduce force_spread's unit statistic exactly."""
    worst_fs, worst_csv, worst_ratio = 0.0, 0.0, 0.0
    for _, u in units.iterrows():
        s, T = u["system"], float(u["T"])
        ti = configs[s]["temps"].index(T)
        g = cfg[(cfg["system"] == s) & (cfg["T"] == T)]
        F = np.stack([forces[m][s][ti] for m in MODELS])
        mine, _ = FS._median_ok(g["spread"].to_numpy(), F)
        theirs = FS._spread(forces, MODELS, s, ti)[0]
        worst_fs = max(worst_fs, abs(mine - theirs))
        worst_csv = max(worst_csv, abs(mine - float(u["S_5model"])))
        worst_ratio = max(worst_ratio, abs(g["pair_ratio"].min() - float(u["min_pair_ratio"])))
    ok = worst_fs < 1e-12 and worst_csv < 1e-9 and worst_ratio < 1e-9
    if not ok:
        raise SystemExit(f"spread does not reproduce force_spread: {worst_fs}, {worst_csv}, "
                         f"{worst_ratio}")
    return {"unit_median_vs_force_spread__spread_max_abs_diff": worst_fs,
            "unit_median_vs_units_consensus_csv_max_abs_diff": worst_csv,
            "unit_min_pair_ratio_vs_csv_max_abs_diff": worst_ratio,
            "n_configs_nonfinite_any_model": int((~cfg["finite_all_models"]).sum()),
            "passed": ok}


def config_level(cfg) -> dict:
    cfg = cfg[cfg["finite_all_models"]]
    bins = []
    for name, lo, hi in BINS:
        g = cfg[(cfg["pair_ratio"] >= lo) & (cfg["pair_ratio"] < hi)]
        b = {"bin": name, "lo_inclusive": None if not np.isfinite(lo) else lo,
             "hi_exclusive": None if not np.isfinite(hi) else hi,
             "n_systems": int(g["system"].nunique()),
             "systems": sorted(g["system"].unique()),
             "spread_eV_A": _q(g["spread"]),
             "u_rms_A_median": _r(g["u_rms_A"].median()) if len(g) else None,
             "recon_median": RECON["config_bin_median"][name]}
        bins.append(b)
    clean = cfg[cfg["pair_ratio"] >= MIN_RATIO]
    over = cfg[cfg["pair_ratio"] < MIN_RATIO]
    per_sys = {}
    for s, g in clean.groupby("system"):
        per_sys[s] = {"n": int(len(g)), "rho_spread_vs_u_rms": _r(spearman(g["spread"],
                                                                         g["u_rms_A"]), 3),
                      "median_spread_eV_A": _r(g["spread"].median())}
    rhos = np.array([v["rho_spread_vs_u_rms"] for v in per_sys.values()
                     if v["rho_spread_vs_u_rms"] is not None], float)
    per_sys_all = {s: _r(spearman(g["spread"], g["pair_ratio"]), 3)
                   for s, g in cfg.groupby("system")}
    pr = np.array([v for v in per_sys_all.values() if v is not None], float)
    return {
        "n_configs": int(len(cfg)), "n_overlap_free": int(len(clean)),
        "n_overlap": int(len(over)),
        "overlap_rule": f"pair-resolved contact ratio < {MIN_RATIO} (force_spread variant (g))",
        "spread_by_pair_ratio_bin": bins,
        "overlap_free_vs_overlap_spread_eV_A": {"overlap_free": _q(clean["spread"]),
                                                "overlap": _q(over["spread"])},
        "rho_spread_vs_u_rms__overlap_free_pooled": {
            **rho_block(clean["spread"], clean["u_rms_A"], clean["system"]),
            "recon": RECON["config_rho_spread_amplitude_nonoverlap"][0],
            "note": "pooled over 15 systems, so it mixes between-system spread levels (C "
                    "diamond and Si sit at 0.1-0.27 eV/A at small amplitude) with the "
                    "within-system amplitude dependence; the per-system rhos separate them"},
        "rho_spread_vs_u_rms__overlap_free_within_system": {
            "per_system": per_sys, "median_rho": _r(np.median(rhos), 3),
            "n_systems_rho_positive": int((rhos > 0).sum()), "n_systems": int(rhos.size)},
        "rho_spread_vs_u_rms__all_configs_pooled": rho_block(cfg["spread"], cfg["u_rms_A"],
                                                             cfg["system"]),
        "rho_spread_vs_pair_ratio__all_configs_pooled": rho_block(
            cfg["spread"], cfg["pair_ratio"], cfg["system"]),
        "rho_spread_vs_pair_ratio__all_configs_within_system": {
            "per_system": per_sys_all, "median_rho": _r(np.median(pr), 3),
            "n_systems_rho_negative": int((pr < 0).sum()), "n_systems": int(pr.size)},
    }


# ----------------------------------------------------------- 2. unit level ----

def unit_level(units, per, summary) -> dict:
    units = units.copy()
    units["clean"] = units["min_pair_ratio"].astype(float) >= MIN_RATIO
    calls = {(r.system, r.model, float(r.T)): bool(r.pred_stable) for r in per.itertuples()}
    out = {"label": "consensus_wrong (force_spread.py / analysis.h3_ensemble_guardrail)",
           "score": "S_5model: median over the 16 configurations of the per-configuration "
                    "spread, five production models",
           "subsets": {}}
    for name, d in (("all", units), ("clean", units[units["clean"]]),
                    ("overlap", units[~units["clean"]])):
        d = d.reset_index(drop=True)
        sc, lab, cl = (d["S_5model"].to_numpy(float), d["consensus_wrong"].to_numpy(bool),
                       d["system"].to_numpy())
        blk = {"n_units": int(len(d)), "n_wrong": int(lab.sum()),
               "n_systems": int(d["system"].nunique()),
               "spearman_spread_vs_wrong": rho_block(sc, lab.astype(float), cl),
               "recon_rho": RECON["unit_rho"][name]}
        auc = {"auc": _r(S.auc(sc, lab), 3)}
        try:
            auc.update(S.cluster_permutation_auc(sc, lab, cl, n_perm=N_PERM, seed=SEED))
        except ValueError as e:
            auc["p_perm_clustered"] = None
            auc["perm_note"] = str(e)
        auc.update(S.cluster_bootstrap_auc(sc, lab, cl, n_boot=N_BOOT, seed=SEED))
        auc["recon"] = RECON["unit_auc"].get(name)
        blk["auc_spread_for_wrong"] = auc
        blk["median_spread_wrong"] = _r(np.median(sc[lab])) if lab.any() else None
        blk["median_spread_right"] = _r(np.median(sc[~lab])) if (~lab).any() else None
        # descriptive only: force_spread variant (d), and the AUC over (wrong, right) pairs
        # taken inside one system, which removes each system's own spread level
        blk["auc_normalised_spread_Snorm_5model"] = _r(
            S.auc(d["Snorm_5model"].to_numpy(float), lab), 3)
        hit = n_pairs = 0.0
        for _, g in d.groupby("system"):
            w, rr = g.loc[g["consensus_wrong"], "S_5model"].to_numpy(), \
                g.loc[~g["consensus_wrong"], "S_5model"].to_numpy()
            diff = w[:, None] - rr[None, :]
            hit += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            n_pairs += diff.size
        blk["within_system_pair_auc"] = {"auc": _r(hit / n_pairs, 3) if n_pairs else None,
                                         "n_pairs": int(n_pairs)}
        errs = []
        for _, u in d[d["consensus_wrong"]].iterrows():
            mc = {m: ("stable" if calls[(u["system"], m, float(u["T"]))] else "unstable")
                  for m in MODELS}
            agree = sum((v == "stable") == bool(u["consensus_stable"]) for v in mc.values())
            agree4 = sum((v == "stable") == bool(u["consensus_stable"])
                         for m, v in mc.items() if m != "orb_v2")
            errs.append({
                "system": u["system"], "T": float(u["T"]),
                "kind": "false-stable" if u["consensus_stable"] else "false-unstable",
                "S_5model_eV_A": _r(u["S_5model"]),
                "spread_percentile_in_subset": _r(100 * float((sc < u["S_5model"]).mean()
                                                              + 0.5 * (sc == u["S_5model"])
                                                              .mean()), 1),
                "min_pair_ratio": _r(u["min_pair_ratio"], 3),
                "min_pair_ratio_pair": u["min_pair_ratio_pair"],
                "rms_disp_A": _r(u["rms_disp_ang"]),
                "vote_disagreement": _r(u["vote_disagreement"], 3),
                "per_model_call": mc, "n_models_agreeing_with_consensus": int(agree),
                "unanimous": bool(agree == len(MODELS)),
                "dissenting_models": [m for m, v in mc.items()
                                      if (v == "stable") != bool(u["consensus_stable"])],
                "unanimous_ex_orb": bool(agree4 == len(MODELS) - 1)})
        blk["errors"] = errs
        blk["n_errors_unanimous"] = int(sum(e["unanimous"] for e in errs))
        blk["n_errors_unanimous_ex_orb"] = int(sum(e["unanimous_ex_orb"] for e in errs))
        if name == "clean":
            top = d.sort_values("S_5model", ascending=False).head(8)
            blk["highest_spread_units"] = [
                {"system": r["system"], "T": float(r["T"]), "S_5model_eV_A": _r(r["S_5model"]),
                 "consensus_wrong": bool(r["consensus_wrong"])} for _, r in top.iterrows()]
            g = summary["consensus_level"]["g_S_5model_no_overlap"]
            blk["reproduces_force_spread_variant_g"] = bool(
                g["auc"] == auc["auc"] and g["ci_lo"] == auc["ci_lo"]
                and g["ci_hi"] == auc["ci_hi"] and g["n_units"] == len(d))
        if name == "all":
            p = summary["primary"]
            blk["reproduces_force_spread_primary"] = bool(
                p["auc"] == auc["auc"] and p["ci_lo"] == auc["ci_lo"]
                and p["ci_hi"] == auc["ci_hi"]
                and p["p_perm_clustered"] == auc.get("p_perm_clustered"))
        out["subsets"][name] = blk
    return out


# ------------------------------------------------------------- 3. PBE set ----

def _pbe_jobs() -> dict:
    """(geom_file relative to results/revision/dft, frame) -> job directory, from job.json."""
    jobs = {}
    for jj in sorted((DFT_DIR / "qe").glob("b_*/job.json")):
        m = json.loads(_track(jj).read_text(encoding="utf-8"))
        for mem in m["members"]:
            if mem.get("study") != "c3b":
                continue
            if mem.get("reduction"):
                raise SystemExit(f"{jj}: reduced-cell C3b job; not handled here")
            key = (mem["geom_file"], int(mem["frame"]))
            if key in jobs:
                raise SystemExit(f"two jobs serve {key}")
            jobs[key] = jj.parent
    return jobs


def _pbe_forces(job_dir: Path, frame) -> np.ndarray:
    po, pin, ran = job_dir / "pw.out", job_dir / "pw.in", job_dir / "pw.in.ran"
    r = D.parse_pw_out(_track(po))
    if not (r["job_done"] and r["converged"]) or "parse_error" in r:
        raise SystemExit(f"{po}: not a finished, converged, parsable pw.x run")
    if _track(ran).read_bytes() != _track(pin).read_bytes():
        raise SystemExit(f"{job_dir.name}: pw.out was produced by another pw.in")
    mis = D._pos_mismatch(r["atoms"], frame)
    if mis > POS_TOL:
        raise SystemExit(f"{job_dir.name}: pw.out structure differs from its frame by {mis}")
    return np.asarray(r["forces"], float)


def _mlip_forces(gf: Path) -> dict:
    sha = D._sha256(gf)
    out = {}
    for m in MODELS:
        p = _track(DFT_DIR / "mlip" / m / gf.parent.name / f"{gf.stem}.json")
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["geom_sha256"] != sha:
            raise SystemExit(f"{p}: MLIP numbers are for another version of {gf.name}")
        out[m] = [np.asarray(f, float) for f in d["forces_eV_A"]]
    return out


def decompose(F: np.ndarray, Fp: np.ndarray) -> dict:
    """Exact shared/spread split of the committee's mean squared force error (per component)."""
    err = F - Fp[None]
    mse_m = np.mean(err ** 2, axis=(1, 2))
    total = float(mse_m.mean())
    Fbar = F.mean(axis=0)
    shared = float(np.mean((Fbar - Fp) ** 2))
    spread_var = float(np.mean(F.var(axis=0)))
    if abs(total - shared - spread_var) > 1e-9 * max(total, 1e-30):
        raise RuntimeError("bias-variance identity failed")
    e = err.reshape(len(F), -1)
    nrm = np.linalg.norm(e, axis=1)
    cos = (e @ e.T) / np.outer(nrm, nrm)
    iu = np.triu_indices(len(F), 1)
    return {"total_mse": total, "shared_mse": shared, "spread_var": spread_var,
            "spread": float(np.sqrt(spread_var)),
            "ens_mean_rmse": float(np.sqrt(shared)),
            "mean_model_rmse": float(np.mean(np.sqrt(mse_m))),
            "shared_fraction": shared / total,
            "mean_pairwise_error_cosine": float(cos[iu].mean())}


def pbe_table() -> tuple[pd.DataFrame, dict]:
    jobs = _pbe_jobs()
    gdir = DFT_DIR / "geom_c3b"
    refs, rows, desc = {}, [], {}
    for gf in sorted(gdir.glob("*_ref.extxyz")):
        frames = D._read_frames(_track(gf))
        a0 = frames[0]
        refs[a0.info["system"]] = (a0, FS.Contacts(a0.numbers, a0.cell.array,
                                                   a0.get_positions()))
    files = sorted(gdir.glob("*_ref.extxyz")) + sorted(
        f for f in gdir.glob("*.extxyz") if not f.name.endswith(("_ref.extxyz", ".tmp.extxyz")))
    for gf in files:
        frames = D._read_frames(_track(gf))
        ml = _mlip_forces(gf)
        grel = D._rel(gf, DFT_DIR)
        system = frames[0].info["system"]
        ref, contacts = refs[system]
        for k, a in enumerate(frames):
            role = a.info["role"]
            if role == "ref":
                continue                      # the undisplaced energy reference, not scored
            if (a.numbers != ref.numbers).any() or not np.allclose(a.cell.array, ref.cell.array,
                                                                   atol=1e-6):
                raise SystemExit(f"{gf.name}#{k}: atoms or cell differ from the reference")
            Fp = _pbe_forces(jobs[(grel, k)], a)
            F = np.stack([ml[m][k] for m in MODELS])
            dmin, ratio, pair = contacts(a.get_positions())
            setname = "baseline" if role == "baseline" else "sscha"
            row = {"system": system, "owner_model": a.info["owner_model"],
                   "set": setname, "T": float(a.info["T"]) if setname == "sscha" else None,
                   "seed": int(a.info["seed"]) if setname == "sscha" else -1,
                   "index": int(a.info["index"]), "job": jobs[(grel, k)].name,
                   "n_atoms": len(a), "u_rms_A": float(a.info["u_rms_A"]),
                   "min_dist_A": dmin, "pair_ratio": ratio, "pair": pair,
                   "rms_f_pbe": float(np.sqrt(np.mean(Fp ** 2)))}
            row.update(decompose(F, Fp))
            exo = [i for i, m in enumerate(MODELS) if m != "orb_v2"]
            d4 = decompose(F[exo], Fp)
            row.update({"total_mse_ex_orb": d4["total_mse"], "shared_mse_ex_orb": d4["shared_mse"],
                        "spread_var_ex_orb": d4["spread_var"]})
            row["_per_model_rmse"] = {m: float(np.sqrt(np.mean((F[i] - Fp) ** 2)))
                                      for i, m in enumerate(MODELS)}
            rows.append(row)
            key = (system, setname)
            dd = desc.setdefault(key, {"system": system, "set": setname,
                                       "owner_model": a.info["owner_model"],
                                       "geom_file": grel, "n_configs": 0})
            dd["n_configs"] += 1
            if setname == "baseline":
                dd.update(kind="baseline (rattled about the undisplaced ideal supercell)",
                          rattle_stdev_A=float(a.info["rattle_stdev_A"]))
                dd.setdefault("rattle_seeds", []).append(int(a.info["rattle_seed"]))
            else:
                dd.update(kind="SSCHA-sampled (owner model's C1 ensemble at T)",
                          T=float(a.info["T"]))
                dd.setdefault("src_files", set()).add(str(a.info["src_file"]))
    for dd in desc.values():
        if "src_files" in dd:
            dd["src_files"] = sorted(dd["src_files"])
    return pd.DataFrame(rows), desc


def check_pbe_against_c3b(t: pd.DataFrame) -> dict:
    """Per-(configuration, model) force RMSE and RMS PBE force must equal dft_reference's."""
    c = pd.read_csv(_track(DFT_DIR / "c3b_configs.csv"))
    c["seed"] = c["seed"].fillna(-1).astype(int)
    c = c.set_index(["system", "set", "seed", "index", "eval_model"])
    if not c.index.is_unique or len(c) != len(t) * len(MODELS):
        raise SystemExit(f"c3b_configs.csv: {len(c)} rows for {len(t)} configs")
    d_rmse = d_pbe = 0.0
    for _, r in t.iterrows():
        for m, v in r["_per_model_rmse"].items():
            x = c.loc[(r["system"], r["set"], r["seed"], r["index"], m)]
            d_rmse = max(d_rmse, abs(v - x["force_rmse"]))
            d_pbe = max(d_pbe, abs(r["rms_f_pbe"] - x["rms_f_pbe"]))
    ok = d_rmse < 1e-9 and d_pbe < 1e-9
    if not ok:
        raise SystemExit(f"PBE forces do not reproduce c3b_configs.csv: {d_rmse}, {d_pbe}")
    return {"force_rmse_max_abs_diff": d_rmse, "rms_f_pbe_max_abs_diff": d_pbe,
            "n_rows": int(len(c)), "passed": ok}


def pbe_level(t: pd.DataFrame, desc: dict) -> dict:
    M = len(MODELS)
    t = t.copy()
    t["overlap"] = t["pair_ratio"] < MIN_RATIO
    t["ratio_err_spread"] = t["ens_mean_rmse"] / t["spread"]

    def pooled(g, suffix=""):
        tot, sh, sv = (g[f"total_mse{suffix}"].sum(), g[f"shared_mse{suffix}"].sum(),
                       g[f"spread_var{suffix}"].sum())
        return {"shared_fraction_pooled": _r(sh / tot), "spread_rms_eV_A": _r(np.sqrt(sv / len(g))),
                "ens_mean_rmse_eV_A": _r(np.sqrt(sh / len(g))),
                "error_over_spread_pooled": _r(np.sqrt(sh / sv), 3)}

    groups = []
    for (s, st), g in t.groupby(["system", "set"], sort=True):
        b = {**{k: v for k, v in desc[(s, st)].items()}, "n_overlap": int(g["overlap"].sum()),
             "u_rms_A_median": _r(g["u_rms_A"].median()),
             "min_pair_ratio": _r(g["pair_ratio"].min(), 3),
             "rms_f_pbe_median": _r(g["rms_f_pbe"].median()),
             **pooled(g),
             "shared_fraction_per_config_median": _r(g["shared_fraction"].median()),
             "mean_pairwise_error_cosine_median": _r(g["mean_pairwise_error_cosine"].median(), 3),
             "mean_model_rmse_median_eV_A": _r(g["mean_model_rmse"].median()),
             "shared_fraction_pooled_ex_orb": _r(g["shared_mse_ex_orb"].sum()
                                                 / g["total_mse_ex_orb"].sum())}
        groups.append(b)
    for b in groups:
        b.setdefault("rattle_seeds", None)

    by_set = {st: pooled(g) for st, g in t.groupby("set")}
    within = {}
    for s, g in t.groupby("system"):
        within[s] = {"ens_mean_rmse": spearman_perm(g["spread"], g["ens_mean_rmse"]),
                     "mean_model_rmse": _r(spearman(g["spread"], g["mean_model_rmse"]), 3),
                     "sscha_only_ens_mean_rmse": _r(spearman(g.loc[g["set"] == "sscha", "spread"],
                                                             g.loc[g["set"] == "sscha",
                                                                   "ens_mean_rmse"]), 3),
                     "recon": RECON["pbe_rho_within"].get(s)}
    nov, ov = t[~t["overlap"]], t[t["overlap"]]

    def ratio_block(g):
        if g.empty:
            return {"n": 0}
        return {"n": int(len(g)), "systems_sets": sorted({f"{a}/{b}" for a, b in
                                                          zip(g["system"], g["set"])}),
                "per_config_median": _r(g["ratio_err_spread"].median(), 3),
                "per_config_q25_q75": [_r(g["ratio_err_spread"].quantile(.25), 3),
                                       _r(g["ratio_err_spread"].quantile(.75), 3)],
                "pooled": _r(np.sqrt(g["shared_mse"].sum() / g["spread_var"].sum()), 3),
                "shared_fraction_pooled": _r(g["shared_mse"].sum() / g["total_mse"].sum()),
                "spread_median_eV_A": _r(g["spread"].median()),
                "ens_mean_rmse_median_eV_A": _r(g["ens_mean_rmse"].median())}

    cfg_rows = [{k: (_r(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                 if not k.startswith("_") and not k.endswith("_ex_orb")}
                | {"per_model_rmse": {m: _r(x, 5) for m, x in r["_per_model_rmse"].items()}}
                for r in t.to_dict("records")]
    return {
        "n_configs": int(len(t)), "n_models": M,
        "uninformative_committee_reference": {
            "shared_fraction": _r(1 / M, 3), "error_over_spread": _r(1 / np.sqrt(M - 1), 3),
            "assumption": "independent, equal-variance, zero-mean model errors"},
        "definitions": {
            "spread": "sqrt(mean over atoms, components of the population variance across "
                      "models) = force_spread's per-configuration statistic",
            "ens_mean_rmse": "RMS over atoms, components of (Fbar - F_PBE)",
            "mean_model_rmse": "mean over models of each model's force RMSE vs PBE",
            "shared_fraction": "|Fbar - F_PBE|^2 / mean_m |F_m - F_PBE|^2 (per component)",
            "error_over_spread": "ens_mean_rmse / spread; pooled = sqrt(sum shared / sum "
                                 "spread_var)",
            "overlap": f"pair-resolved contact ratio < {MIN_RATIO} against the unit's "
                       "undisplaced reference supercell (_ref.extxyz frame 0)"},
        "per_ensemble": groups,
        "pooled_by_set": by_set,
        "spearman_spread_vs_ens_mean_rmse": {
            "pooled": {**rho_block(t["spread"], t["ens_mean_rmse"], t["system"]),
                       "recon": RECON["pbe_rho_pooled"],
                       "note": "4 systems, so the clustered interval is coarse"},
            "pooled_overlap_free": {**rho_block(nov["spread"], nov["ens_mean_rmse"],
                                                nov["system"]),
                                    "recon": RECON["pbe_rho_pooled_nonoverlap"]},
            "pooled_vs_mean_model_rmse": _r(spearman(t["spread"], t["mean_model_rmse"]), 3),
            "within_system": within},
        "error_over_spread": {"overlap_free": ratio_block(nov), "overlap": ratio_block(ov),
                              "recon": RECON["pbe_error_over_spread"]},
        "configs": cfg_rows,
    }


# ------------------------------------------------- recon comparison ----

def _agrees(recon, computed) -> bool | None:
    """Equal at the precision the recon value was quoted to (ranges: both ends)."""
    if computed is None:
        return None
    if isinstance(recon, (tuple, list)):
        return all(_agrees(a, b) for a, b in zip(recon, computed))
    d = len(repr(float(recon)).split(".")[1].rstrip("0")) if "." in repr(float(recon)) else 0
    return bool(abs(float(computed) - float(recon)) <= 0.5 * 10 ** -d + 1e-9)


def recon_comparison(l1, l2, l3) -> list:
    rows = []

    def add(q, recon, computed, note=None):
        rows.append({"quantity": q, "recon": recon, "computed": computed,
                     "agrees_at_recon_precision": _agrees(recon, computed),
                     **({"note": note} if note else {})})

    for b in l1["spread_by_pair_ratio_bin"]:
        add(f"config spread median, {b['bin']}", b["recon_median"], b["spread_eV_A"]["median"])
    r = l1["rho_spread_vs_u_rms__overlap_free_pooled"]
    add("config rho(spread, u_rms), overlap-free pooled", -0.06, r["rho"],
        f"n = {r['n']} (recon 763); within-system median rho "
        f"{l1['rho_spread_vs_u_rms__overlap_free_within_system']['median_rho']}")
    for k in ("all", "clean", "overlap"):
        add(f"unit rho(S_5model, wrong), {k}", RECON["unit_rho"][k],
            l2["subsets"][k]["spearman_spread_vs_wrong"]["rho"])
    for k in ("clean", "overlap"):
        add(f"unit AUC, {k}", RECON["unit_auc"][k], l2["subsets"][k]["auc_spread_for_wrong"]["auc"])
    ce = l2["subsets"]["clean"]
    add("clean-set errors unanimous (all five models)", 4, ce["n_errors_unanimous"],
        "recon said all four clean errors were unanimous; the per-model ledger calls give "
        f"{ce['n_errors_unanimous']} of 4 ({ce['n_errors_unanimous_ex_orb']} of 4 without "
        "ORB-v2)")
    g = {(x["system"], x["set"]): x for x in l3["per_ensemble"]}
    base = [x for (s, st), x in g.items() if st == "baseline"]
    ss = [x for (s, st), x in g.items() if st == "sscha" and s != "srtio3_cubic"]
    for lab, grp, rec in (("baseline", base, RECON["pbe_shared_fraction"]["baseline"]),
                          ("SSCHA except SrTiO3", ss,
                           RECON["pbe_shared_fraction"]["sscha_ex_srtio3"])):
        med = [x["shared_fraction_per_config_median"] for x in grp]
        pool = [x["shared_fraction_pooled"] for x in grp]
        add(f"PBE shared fraction range, {lab}, per-config median", rec,
            [_r(min(med), 2), _r(max(med), 2)])
        add(f"PBE shared fraction range, {lab}, pooled", rec,
            [_r(min(pool), 2), _r(max(pool), 2)])
    st = g[("srtio3_cubic", "sscha")]
    add("PBE shared fraction SrTiO3 600 K SSCHA, per-config median", 0.88,
        st["shared_fraction_per_config_median"])
    add("PBE shared fraction SrTiO3 600 K SSCHA, pooled", 0.88, st["shared_fraction_pooled"])
    sp = l3["spearman_spread_vs_ens_mean_rmse"]
    add("PBE rho(spread, ens-mean error), pooled", 0.918, sp["pooled"]["rho"])
    add("PBE rho, pooled overlap-free", 0.894, sp["pooled_overlap_free"]["rho"])
    for s, v in sp["within_system"].items():
        add(f"PBE rho within {s}, all 16 configs", v["recon"], v["ens_mean_rmse"]["rho"])
        add(f"PBE rho within {s}, 12 SSCHA configs only", v["recon"],
            v["sscha_only_ens_mean_rmse"])
    eo = l3["error_over_spread"]
    for k, rk in (("overlap_free", "nonoverlap"), ("overlap", "overlap")):
        add(f"PBE error/spread, {k}, per-config median", RECON["pbe_error_over_spread"][rk],
            eo[k]["per_config_median"])
        add(f"PBE error/spread, {k}, pooled", RECON["pbe_error_over_spread"][rk],
            eo[k]["pooled"])
    return rows


# ----------------------------------------------------------- provenance ----

def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _head_blobs() -> dict:
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "HEAD", "--",
                          "results/revision/force_spread", "results/revision/dft"],
                         capture_output=True, text=True, timeout=60).stdout
    blobs = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        blobs[path] = meta.split()[2]
    return blobs


def input_records() -> list:
    """Git blob hash of every file read. A text file (no NUL in its first 8000 bytes, git's
    own test) is hashed CRLF->LF normalised unless its raw bytes already match HEAD, so an
    autocrlf checkout of the committed file hashes to the committed blob."""
    head = _head_blobs()
    recs = []
    for p in sorted(_READ):
        rel = p.relative_to(REPO).as_posix()
        raw = p.read_bytes()
        text = b"\0" not in raw[:8000]
        b_raw = _git_blob(raw)
        b_lf = _git_blob(raw.replace(b"\r\n", b"\n")) if text else b_raw
        blob = b_raw if head.get(rel) == b_raw else b_lf
        recs.append({"path": rel, "git_blob": blob, "crlf_normalised": blob != b_raw,
                     "matches_HEAD": (head.get(rel) == blob) if rel in head else None})
    return recs


# ------------------------------------------------------------------ main ----

def main() -> int:
    units, per, summary, systems, configs, forces = load_force_spread()
    cfg = config_table(systems, configs, forces)
    chk1 = check_config_level(cfg, units, forces, configs)
    lvl1 = config_level(cfg)
    lvl2 = unit_level(units, per, summary)
    t, desc = pbe_table()
    chk3 = check_pbe_against_c3b(t)
    lvl3 = pbe_level(t, desc)
    inputs = input_records()

    script_bytes = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    doc = {
        "_provenance": {
            "script": SCRIPT, "script_git_blob_lf": _git_blob(script_bytes), **FS._git(),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "packages": {p: FS._ver(p) for p in ("numpy", "pandas", "ase")},
            "settings": {"n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED,
                         "min_pair_ratio": MIN_RATIO, "pos_tol_A": POS_TOL,
                         "models": list(MODELS)}},
        "status": "POST-HOC, descriptive; explains force_spread.py's pre-registered result "
                  "and does not replace it",
        "hypothesis_tested": (
            "Committee spread across the five foundation MLIPs is an extrapolation detector "
            "that fires when interatomic distances fall into the overlap region, while in the "
            "moderately anharmonic region that decides stability calls the models share a "
            "common error that their variance cannot see."),
        "checks": {"config_spread_reproduces_force_spread": chk1,
                   "pbe_forces_reproduce_c3b_configs_csv": chk3,
                   "inputs_all_match_HEAD": all(r["matches_HEAD"] is True for r in inputs)},
        "config_level": lvl1,
        "unit_level": lvl2,
        "pbe_set": lvl3,
        "recon_comparison": recon_comparison(lvl1, lvl2, lvl3),
        "notes": [
            "Configuration and unit levels use force_spread.py's configurations, drawn near "
            "the unrelaxed registry cell from the five-model mean FCs; the PBE set uses C3b's "
            "configurations (rattled baseline and owner-model SSCHA ensembles), so the two "
            "sets of configurations are not the same and the PBE set does not contain the "
            "wrong consensus units themselves.",
            "Force RMS over all components is dominated by the stiff modes; a stability call "
            "depends on the soft-mode curvature and well depth, which C3a (PBE along the "
            "screen's own coordinates) addresses. The decomposition says what the spread can "
            "see of the force error, not of the call.",
            "Spearman is the Pearson correlation of pandas average ranks (no scipy). Clustered "
            "intervals resample whole systems (stats._blocks, 10000 resamples, seed 0).",
        ],
        "inputs": inputs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FS._write_json(OUT, doc)

    # ---- console summary
    print("[1] spread by pair-ratio bin (median [q25, q75], n):")
    for b in lvl1["spread_by_pair_ratio_bin"]:
        q = b["spread_eV_A"]
        print(f"    {b['bin']:18s} {q.get('median')} [{q.get('q25')}, {q.get('q75')}] "
              f"n={q['n']}  (recon {b['recon_median']})")
    r = lvl1["rho_spread_vs_u_rms__overlap_free_pooled"]
    print(f"    rho(spread, u_rms) overlap-free pooled {r['rho']} n={r['n']} "
          f"CI [{r['ci_lo']}, {r['ci_hi']}]; within-system median "
          f"{lvl1['rho_spread_vs_u_rms__overlap_free_within_system']['median_rho']}")
    print("[2] unit level:")
    for k, b in lvl2["subsets"].items():
        a = b["auc_spread_for_wrong"]
        print(f"    {k:8s} n={b['n_units']} wrong={b['n_wrong']} rho "
              f"{b['spearman_spread_vs_wrong']['rho']}  AUC {a['auc']} "
              f"CI [{a['ci_lo']}, {a['ci_hi']}]  Snorm AUC "
              f"{b['auc_normalised_spread_Snorm_5model']}  within-system pair AUC "
              f"{b['within_system_pair_auc']}")
        for e in b["errors"]:
            print(f"        {e['system']:14s} {e['T']:5.0f} {e['kind']:14s} S={e['S_5model_eV_A']} "
                  f"r={e['min_pair_ratio']} agree={e['n_models_agreeing_with_consensus']}/5")
    print("[3] PBE set:")
    for g in lvl3["per_ensemble"]:
        print(f"    {g['system']:13s} {g['set']:8s} shared {g['shared_fraction_pooled']} "
              f"(ex-ORB {g['shared_fraction_pooled_ex_orb']})  err/spread "
              f"{g['error_over_spread_pooled']}  n_overlap {g['n_overlap']}")
    sp = lvl3["spearman_spread_vs_ens_mean_rmse"]
    print(f"    rho pooled {sp['pooled']['rho']}  overlap-free {sp['pooled_overlap_free']['rho']}"
          f"  within {[(s, v['ens_mean_rmse']['rho']) for s, v in sp['within_system'].items()]}")
    eo = lvl3["error_over_spread"]
    print(f"    err/spread overlap-free {eo['overlap_free']} ")
    print(f"    err/spread overlap      {eo['overlap']}")
    print("[recon] disagreements at recon precision:")
    for x in doc["recon_comparison"]:
        if not x["agrees_at_recon_precision"]:
            print(f"    {x['quantity']}: recon {x['recon']} computed {x['computed']}")
    print(f"checks: {chk1['passed']}, {chk3['passed']}, inputs match HEAD: "
          f"{doc['checks']['inputs_all_match_HEAD']} ({len(inputs)} files)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
