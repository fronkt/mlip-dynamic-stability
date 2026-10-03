"""summary.json: the evaluation files mapped onto the pre-registered outcomes P1, P2, P3, S1, S2.

All replicates are listed.  Verdict rules (pre-registration, 'Outcomes' and 'Analysis'):

  * a screen call counts as CHANGED only if all three replicates differ from the base call, and as
    UNCHANGED only if all three equal it; otherwise it is UNRESOLVED;
  * P2 predictions: BaTiO3 and KNbO3 at 300 K corrected (supported iff changed, refuted iff
    unchanged); CsSnBr3 at 300 and 600 K and KNbO3 at 600 K persist (supported iff unchanged,
    refuted iff changed).  A prediction about a mis-call that the base model did not make is
    reported as 'n/a (base call already correct)'.
  * P1: per base model, the median over (deciding paths of BaTiO3 and KNbO3) x (replicates) of the
    ratio of fine-tuned to PBE well depth.  Supported iff that median AND each replicate's median
    over paths lie in [0.8, 1.2]; refuted iff all lie outside; otherwise unresolved.
  * P3: supported iff the converged SSCHA (fine-tuned MACE-MP-0, seed 0) calls BaTiO3 100 K STABLE
    (the label is unstable); refuted iff it calls it unstable; unresolved if the run did not finish.
  * S2: harmonic calls of the six controls are 'unchanged' iff all replicates equal the base call.
Nothing is dropped: a missing evaluation file is listed under "missing", never silently skipped.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import common as C
from .evalu import DEPTH_FLOOR_MEV

P1_BAND = (0.8, 1.2)
P1_SYSTEMS = ("batio3_cubic", "knbo3_cubic")
EXPECT_CORRECTED = (("batio3_cubic", 300.0), ("knbo3_cubic", 300.0))
EXPECT_PERSIST = (("cssnbr3_cubic", 300.0), ("cssnbr3_cubic", 600.0), ("knbo3_cubic", 600.0))
TAGS = ("seed0", "seed1", "seed2")


def _load(out: Path):
    ev, missing = {}, []
    for part in ("c3a", "c3b", "p2", "s2"):
        for model in C.BASE_MODELS:
            for tag in ("base", *TAGS):
                p = out / "eval" / f"{part}_{model}_{tag}.json"
                if p.exists():
                    ev[(part, model, tag)] = C.jload(p)
                else:
                    missing.append(p.name)
    return ev, missing


def _verdict_band(vals_all, vals_by_rep, lo, hi):
    if not vals_all:
        return "pending"
    inside = lambda x: lo <= x <= hi                                  # noqa: E731
    med = float(np.median(vals_all))
    reps = [float(np.median(v)) for v in vals_by_rep if v]
    if len(reps) < len(TAGS):
        return "pending (replicates missing)"
    if inside(med) and all(inside(r) for r in reps):
        return "supported"
    if not inside(med) and not any(inside(r) for r in reps):
        return "refuted"
    return "unresolved"


# ------------------------------------------------------------------------------- P1 ----

def p1(ev):
    out = {}
    for model in C.BASE_MODELS:
        base = ev.get(("c3a", model, "base"))
        reps = {t: ev.get(("c3a", model, t)) for t in TAGS}
        if base is None:
            out[model] = {"status": "pending (base evaluation missing)"}
            continue
        sets = {
            "primary_own_deciding": lambda r: r["path_model"] == model and r["role"] == "decide"
            and r["system"] in P1_SYSTEMS,
            "own_all_modes": lambda r: r["path_model"] == model and r["system"] in P1_SYSTEMS,
            "all_models_paths": lambda r: r["system"] in P1_SYSTEMS,
            "negative_control_cssnbr3_own_deciding": lambda r: r["path_model"] == model
            and r["role"] == "decide" and r["system"] == "cssnbr3_cubic",
        }
        blk = {}
        for name, sel in sets.items():
            stems = [s for s, r in base["paths"].items() if sel(r) and r["depth_ratio"] is not None]
            rows = []
            for s in stems:
                row = {"stem": s, "system": base["paths"][s]["system"],
                       "path_model": base["paths"][s]["path_model"], "role": base["paths"][s]["role"],
                       "pbe_depth_meV": base["paths"][s]["pbe_depth_meV"],
                       "base_ratio": base["paths"][s]["depth_ratio"], "replicates": {}}
                for t, r in reps.items():
                    if r is not None and s in r["paths"]:
                        row["replicates"][t] = r["paths"][s]["depth_ratio"]
                rows.append(row)
            by_rep = [[row["replicates"][t] for row in rows if row["replicates"].get(t) is not None]
                      for t in TAGS]
            allv = [v for rep in by_rep for v in rep]
            blk[name] = {"n_paths": len(rows), "paths": rows,
                         "base_median_ratio": float(np.median([r["base_ratio"] for r in rows])) if rows else None,
                         "replicate_median_ratio": {t: (float(np.median(v)) if v else None)
                                                    for t, v in zip(TAGS, by_rep)},
                         "median_over_paths_and_replicates": float(np.median(allv)) if allv else None,
                         "band": list(P1_BAND)}
            if name == "primary_own_deciding":
                blk[name]["verdict"] = _verdict_band(allv, by_rep, *P1_BAND)
        out[model] = blk
    return out


# ------------------------------------------------------------------------------- P2 ----

def _ledger_calls():
    """Base (production) softmode calls, read-only from the canonical ledger."""
    from mlip_dynstab.analysis import load_canonical
    df = load_canonical()
    d = df[(df.method == "softmode") & df.system.isin(C.TEST_SYSTEMS) & df.model.isin(C.BASE_MODELS)]
    return {(r.system, r.model, float(r.temperature_K)): bool(r.pred_stable) for r in d.itertuples()}


def _status(base_call, rep_calls, expect):
    got = [c for c in rep_calls if c is not None]
    if len(got) < len(TAGS):
        return "pending (replicates missing)"
    changed = all(c != base_call for c in got)
    unchanged = all(c == base_call for c in got)
    if expect == "corrected":
        return "supported" if changed else ("refuted" if unchanged else "unresolved")
    if expect == "persist":
        return "supported" if unchanged else ("refuted" if changed else "unresolved")
    return "changed" if changed else ("unchanged" if unchanged else "unresolved")


def p2(ev):
    try:
        ledger = _ledger_calls()
    except Exception as exc:                                              # noqa: BLE001
        ledger, note = {}, f"ledger unavailable: {type(exc).__name__}: {exc}"
    else:
        note = "base calls from results/ledger.parquet (read-only, canonical softmode rows)"
    out = {"base_call_source": note, "models": {}}
    for model in C.BASE_MODELS:
        blk = {}
        rerun = ev.get(("p2", model, "base"))
        for system in C.TEST_SYSTEMS:
            for T in C.LADDER_T:
                key = (system, model, T)
                bc = ledger.get(key)
                gt = None
                reps = {}
                for t in TAGS:
                    e = ev.get(("p2", model, t))
                    if e is None:
                        reps[t] = None
                        continue
                    r = next((x for x in e["rows"] if x["system"] == system and float(x["T"]) == T), None)
                    reps[t] = None if r is None else {"pred_stable": r["pred_stable"],
                                                      "min_eff_freq_thz": r["min_eff_freq_thz"],
                                                      "correct": r["correct"]}
                    gt = r["gt_stable"] if r is not None else gt
                rb = None
                if rerun is not None:
                    r0 = next((x for x in rerun["rows"] if x["system"] == system and float(x["T"]) == T), None)
                    if r0 is not None:
                        rb = {"pred_stable": r0["pred_stable"], "min_eff_freq_thz": r0["min_eff_freq_thz"]}
                        gt = r0["gt_stable"]
                expect = ("corrected" if (system, T) in EXPECT_CORRECTED
                          else "persist" if (system, T) in EXPECT_PERSIST else None)
                calls = [None if reps[t] is None else reps[t]["pred_stable"] for t in TAGS]
                base_wrong = None if (bc is None or gt is None) else (bc != gt)
                if expect and base_wrong is False:
                    status = "n/a (base call already correct)"
                elif bc is None:
                    status = "pending (base call missing)"
                else:
                    status = _status(bc, calls, expect)
                blk[f"{system}@{T:g}"] = {
                    "system": system, "T": T, "label_stable": gt, "base_call_ledger": bc,
                    "base_rerun": rb, "base_rerun_agrees_with_ledger": (None if rb is None or bc is None
                                                                        else rb["pred_stable"] == bc),
                    "base_correct": None if base_wrong is None else (not base_wrong),
                    "replicates": reps, "prediction": expect, "status": status,
                    "n_replicates_correct": sum(1 for t in TAGS if reps[t] is not None and reps[t]["correct"])}
        out["models"][model] = blk
    return out


# ------------------------------------------------------------------------------- P3 ----

def p3(out: Path):
    p = out / "sscha_converged_grid" / "batio3_cubic_mace_mp0_100K_sc222_startA.json"
    if not p.exists():
        return {"status": "pending", "expected_file": C.rel(p),
                "command": "python scripts/finetune_trial.py evaluate --part p3-cmd",
                "prediction": "still called stable (the label is unstable)"}
    d = C.jload(p)
    run = d.get("run") or {}
    hs = run.get("hessian") or {}
    rec = {"file": C.rel(p), "run_status": run.get("status"), "model_version": (d.get("provenance_last_run") or {}).get("model_version"),
           "min_nonac_thz": hs.get("min_nonac_thz"), "called_stable": hs.get("dynamically_stable"),
           "relax_converged": (run.get("relax") or {}).get("converged"),
           "label_stable": False, "prediction": "called stable"}
    if run.get("status") != "ok" or hs.get("dynamically_stable") is None:
        rec["status"] = "unresolved (run not ok)"
    else:
        rec["status"] = "supported" if hs["dynamically_stable"] else "refuted"
    return rec


# ------------------------------------------------------------------------------- S1 ----

def s1(ev):
    out = {}
    for model in C.BASE_MODELS:
        blk = {"c3a": {}, "c3b": {}}
        for tag in ("base", *TAGS):
            e = ev.get(("c3a", model, tag))
            if e is not None:
                per_sys = {}
                for system in C.TEST_SYSTEMS:
                    rs = [r for r in e["paths"].values() if r["system"] == system]
                    if not rs:
                        continue
                    n = sum(r["n_pts"] for r in rs)
                    nf = sum(r["n_force_components"] for r in rs)
                    nw = sum(r["n_pts_window"] for r in rs)
                    nfw = sum(r["n_force_components_window"] for r in rs)
                    per_sys[system] = {
                        "n_paths": len(rs), "n_points": n, "n_points_window": nw,
                        "note": "all points include the steep repulsive wall at large Q; the "
                                "window is the well region the screen fits",
                        "energy_rmse_meV_per_cell": float(np.sqrt(sum(r["e_sumsq_meV2"] for r in rs) / n)),
                        "energy_rmse_meV_per_atom": float(np.sqrt(sum(r["e_sumsq_per_atom_meV2"] for r in rs) / n)),
                        "force_rmse_eV_A": float(np.sqrt(sum(r["f_sumsq"] for r in rs) / nf)),
                        "force_rel_rmse": float(np.sqrt(sum(r["f_sumsq"] for r in rs)
                                                        / sum(r["f_sumsq_pbe"] for r in rs))),
                        "window_energy_rmse_meV_per_cell": float(np.sqrt(sum(r["e_sumsq_window_meV2"] for r in rs) / nw)),
                        "window_energy_rmse_meV_per_atom": float(np.sqrt(sum(r["e_sumsq_window_per_atom_meV2"] for r in rs) / nw)),
                        "window_force_rmse_eV_A": float(np.sqrt(sum(r["f_sumsq_window"] for r in rs) / nfw)),
                        "window_force_rel_rmse": float(np.sqrt(sum(r["f_sumsq_window"] for r in rs)
                                                               / sum(r["f_sumsq_pbe_window"] for r in rs)))}
                blk["c3a"][tag] = per_sys
            e = ev.get(("c3b", model, tag))
            if e is not None:
                per = {}
                for role in ("sscha", "baseline"):
                    rs = [r for r in e["configs"] if r["role"] == role]
                    if not rs:
                        continue
                    ee = [r["e_err_meV_atom"] for r in rs if "e_err_meV_atom" in r]
                    per[role] = {"n_configs": len(rs),
                                 "force_rmse_eV_A": float(np.sqrt(sum(r["f_sumsq"] for r in rs)
                                                                  / sum(r["n_components"] for r in rs))),
                                 "force_rel_rmse": float(np.sqrt(sum(r["f_sumsq"] for r in rs)
                                                                 / sum(r["f_sumsq_pbe"] for r in rs))),
                                 "energy_err_rms_meV_atom": float(np.sqrt(np.mean(np.square(ee)))) if ee else None,
                                 "energy_err_mean_meV_atom": float(np.mean(ee)) if ee else None,
                                 "systems": sorted({r["system"] for r in rs})}
                blk["c3b"][tag] = per
        out[model] = blk
    return out


# ------------------------------------------------------------------------------- S2 ----

def s2(ev):
    out = {}
    for model in C.BASE_MODELS:
        base = ev.get(("s2", model, "base"))
        rows = {}
        for system in C.CONTROLS:
            rec = {"label_stable": None, "base": None, "replicates": {}}
            if base is not None:
                r = next((x for x in base["rows"] if x["system"] == system), None)
                if r:
                    rec["base"] = {"min_freq_thz": r["min_freq_thz"], "pred_stable": r["pred_stable"]}
                    rec["label_stable"] = r["label_stable"]
            for t in TAGS:
                e = ev.get(("s2", model, t))
                if e is None:
                    rec["replicates"][t] = None
                    continue
                r = next((x for x in e["rows"] if x["system"] == system), None)
                rec["replicates"][t] = None if r is None else {"min_freq_thz": r["min_freq_thz"],
                                                               "pred_stable": r["pred_stable"]}
            calls = [None if v is None else v["pred_stable"] for v in rec["replicates"].values()]
            rec["status"] = ("pending" if rec["base"] is None or any(c is None for c in calls)
                             else _status(rec["base"]["pred_stable"], calls, None))
            rows[system] = rec
        done = [r for r in rows.values() if r["status"] in ("changed", "unchanged", "unresolved")]
        out[model] = {"controls": rows,
                      "n_unchanged": sum(r["status"] == "unchanged" for r in done),
                      "n_changed": sum(r["status"] == "changed" for r in done),
                      "n_unresolved": sum(r["status"] == "unresolved" for r in done),
                      "prediction": "all six unchanged"}
    return out


def base_reproduction(ev, dft_root: Path):
    """Base-model depths recomputed here against results/revision/dft/c3a_paths.csv."""
    p = dft_root / "c3a_paths.csv"
    out = {}
    if not p.exists():
        return out
    rows = {}
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[(r["stem"], r["curve"])] = r
    for model in C.BASE_MODELS:
        e = ev.get(("c3a", model, "base"))
        if e is None:
            continue
        d = []
        for stem, r in e["paths"].items():
            ref = rows.get((stem, model))
            if ref and ref.get("depth_meV"):
                d.append(abs(r["metrics"]["depth_meV"] - float(ref["depth_meV"])))
        out[model] = {"n_compared": len(d), "max_abs_depth_diff_meV": float(max(d)) if d else None}
    return out


def build(out: Path, dft_root: Path) -> int:
    ev, missing = _load(out)
    summary = {
        "preregistration": "tasks/preregistration-finetune-2026-10-03.md (commit 033b3d8)",
        "provenance": C.provenance("summary"),
        "available": sorted(f"{p}_{m}_{t}" for (p, m, t) in ev), "missing": missing,
        "rules": __doc__,
        "P1": p1(ev), "P2": p2(ev), "P3": p3(out), "S1": s1(ev), "S2": s2(ev),
        "base_reproduction_vs_c3a_paths_csv": base_reproduction(ev, dft_root),
        "dataset": {m: (C.jload(out / "dataset" / m / "dataset.json") if (out / "dataset" / m / "dataset.json").exists() else None)
                    for m in C.BASE_MODELS},
        "configs_manifest": C.rel(out / "manifest.json") if (out / "manifest.json").exists() else None,
    }
    C.jdump(out / "summary.json", summary)
    print(f"[summary] wrote {C.rel(out / 'summary.json')}; {len(ev)} evaluation files, {len(missing)} missing")
    for model in C.BASE_MODELS:
        v = summary["P1"].get(model, {}).get("primary_own_deciding", {})
        print(f"   P1 {model}: median {v.get('median_over_paths_and_replicates')} -> {v.get('verdict', 'pending')}")
        st = [b["status"] for b in summary["P2"]["models"][model].values() if b["prediction"]]
        print(f"   P2 {model}: " + ", ".join(f"{k}={st.count(k)}" for k in sorted(set(st))))
    print(f"   P3: {summary['P3'].get('status')}")
    return 0
