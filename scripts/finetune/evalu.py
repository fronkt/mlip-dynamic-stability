"""Stage `evaluate`: P1 (held-out C3a wells), S1 (held-out force/energy errors), P2 (full screen
pipeline re-run), S2 (harmonic calls of the six stable controls), P3 (SSCHA command) and the
summary that maps them onto the pre-registered outcomes.

Per model env and checkpoint tag (base, seed0, seed1, seed2):

    python scripts/finetune_trial.py evaluate --model mace_mp0 --tag seed0 --part all --device cuda
    (parts: c3a, c3b, p2, s2)  -> results/revision/finetune/eval/<part>_<model>_<tag>.json

then anywhere (no MLIP needed):

    python scripts/finetune_trial.py evaluate --part summary   -> results/revision/finetune/summary.json
    python scripts/finetune_trial.py evaluate --part p3-cmd    -> the SSCHA command (P3)

A checkpoint is loaded through mlip_dynstab.calculators with MLIP_DYNSTAB_CKPT_<MODEL> set (the
same hook the SSCHA run uses), so both paths load the model the same way; `base` loads the
foundation model exactly as production does.  The held-out sets (every PBE C3a path point of the
three test systems, every C3b configuration) are read here and nowhere in training.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import common as C
from .pbe import PBEData

DEPTH_FLOOR_MEV = 1.0          # ratios are formed only where the PBE well is at least this deep


# ------------------------------------------------------------------ model handling ----

def ckpt_path(out: Path, model: str, tag: str) -> Path | None:
    if tag == "base":
        return None
    k = int(tag.replace("seed", ""))
    if model == "mace_mp0":
        return out / "models" / "mace_mp0" / f"seed{k}" / f"ft_mace_mp0_seed{k}.model"
    return out / "models" / "chgnet" / f"seed{k}" / "best.pth.tar"


def load_handle(model: str, tag: str, out: Path, device: str, ckpt: Path | None = None):
    """CalcHandle for the foundation model (tag base) or a fine-tuned checkpoint (the env-var
    hook of mlip_dynstab.calculators; no handle cache, so tags cannot bleed into each other)."""
    from mlip_dynstab import calculators as K
    var = K._CKPT_ENV + model.upper()
    os.environ.pop(var, None)
    if tag != "base":
        p = Path(ckpt) if ckpt else ckpt_path(out, model, tag)
        if not p.exists():
            raise SystemExit(f"checkpoint {p} does not exist (train first)")
        os.environ[var] = str(p.resolve())
    try:
        return K._load_calculator(model, device)
    finally:
        os.environ.pop(var, None)


def eval_frames(calc, frames):
    from ase import Atoms
    E, F = [], []
    for a in frames:
        b = Atoms(symbols=a.get_chemical_symbols(), positions=a.get_positions(), cell=a.get_cell(),
                  pbc=True)
        b.calc = calc
        E.append(float(b.get_potential_energy()))
        F.append(np.asarray(b.get_forces(), float))
    return np.array(E), F


# --------------------------------------------------------------- curve metrics (copies) ----

def fit_window(dE):
    dE = np.asarray(dE, float)
    emin = float(dE.min())
    return dE <= emin + max(0.06, 5.0 * abs(min(emin, 0.0)))


def omega_thz(a, m_eff):
    from mlip_dynstab.finite_t import _W_TO_OMEGA2
    om2 = 2.0 * a * _W_TO_OMEGA2 / m_eff
    return float(np.sign(om2) * np.sqrt(abs(om2)) / (2 * np.pi) / 1e12)


def curve_metrics(Qs, dE, m_eff):
    """dft_reference._curve_metrics: depth, minimum, the screen's double-well fit."""
    from mlip_dynstab.finite_t import _fit_double_well
    Qs, dE = np.asarray(Qs, float), np.asarray(dE, float)
    a, b, c = _fit_double_well(Qs, dE)
    i = int(np.argmin(dE))
    return {"depth_meV": float(-min(dE.min(), 0.0) * 1000), "Q_min_A": float(Qs[i]),
            "a": float(a), "b": float(b), "c": float(c),
            "bounded": bool(c > 0 or (b > 0 and c >= 0)),
            "omega_harm_thz": omega_thz(a, m_eff)}


def screen_call(task):
    """One single-mode screen solve (module level so a process pool can run it)."""
    from mlip_dynstab.finite_t import _solve_scha, _sym_curvature_freq
    a, b, c, m_eff, T = task
    eff, Q0, st = _solve_scha(a, b, c, m_eff, T)
    curv = _sym_curvature_freq(a, b, c, m_eff, T)
    return {"stable": bool(st), "Q0_A": float(Q0), "eff_thz": float(eff),
            "curv_thz": None if curv is None else float(curv)}


def solve_all(tasks, workers):
    if workers <= 1 or len(tasks) < 8:
        return [screen_call(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(screen_call, tasks, chunksize=4))


# ----------------------------------------------------------------------- part c3a ----

def pbe_reference(pbe: PBEData, paths, out: Path, workers: int):
    """PBE curve metrics and screen calls per path, cached in eval/c3a_pbe.json (stem -> block)."""
    p = out / "eval" / "c3a_pbe.json"
    cache = C.jload(p) if p.exists() else {"paths": {}}
    todo = [q for q in paths if q["stem"] not in cache["paths"]
            or cache["paths"][q["stem"]].get("n_Q_dft") != q["n_Q_dft"]]
    tasks, keys = [], []
    for q in todo:
        dEp = q["E"] - q["E"][0]
        mt = curve_metrics(q["Q"], dEp, q["m_eff"])
        cache["paths"][q["stem"]] = {"n_Q_dft": q["n_Q_dft"], "Q": q["Q"].tolist(),
                                     "dE_meV": (dEp * 1000).tolist(), "metrics": mt, "calls": {}}
        for T in C.SCREEN_T:
            tasks.append((mt["a"], mt["b"], mt["c"], q["m_eff"], T))
            keys.append((q["stem"], T))
    for (stem, T), r in zip(keys, solve_all(tasks, workers)):
        cache["paths"][stem]["calls"][f"{T:g}"] = r
    if todo:
        C.jdump(p, cache)
    return cache["paths"]


def part_c3a(args, handle, model, tag, out, pbe):
    paths = [q for q in pbe.c3a_paths() if q["status"] in ("complete", "partial")]
    if args.max_paths:
        paths = paths[:args.max_paths]
    ref = pbe_reference(pbe, paths, out, args.workers)
    res, tasks, keys = {}, [], []
    t0 = time.time()
    for n, q in enumerate(paths):
        frames = [pbe.frames(q["geom_file"])[i] for i in q["idx"]]
        E, F = eval_frames(handle.calc, frames)
        dE = E - E[0]
        dEp = q["E"] - q["E"][0]
        win = fit_window(dEp)
        mt = curve_metrics(q["Q"], dE, q["m_eff"])
        pm = ref[q["stem"]]["metrics"]
        dF = np.concatenate([(f - fp).ravel() for f, fp in zip(F, q["F"])])
        fp = np.concatenate([fp.ravel() for fp in q["F"]])
        diff = (dE - dEp) * 1000
        # the well window (the points the screen fits, dft_reference._fit_window): the steep
        # repulsive wall at large Q dominates an all-points RMSE, so both are kept
        dFw = np.concatenate([(f - g).ravel() for f, g, w in zip(F, q["F"], win) if w])
        fpw = np.concatenate([g.ravel() for g, w in zip(q["F"], win) if w])
        res[q["stem"]] = {
            "n_pts_window": int(win.sum()), "e_sumsq_window_meV2": float(np.sum(diff[win] ** 2)),
            "e_sumsq_window_per_atom_meV2": float(np.sum((diff[win] / q["n_atoms"]) ** 2)),
            "n_force_components_window": int(dFw.size), "f_sumsq_window": float(np.sum(dFw ** 2)),
            "f_sumsq_pbe_window": float(np.sum(fpw ** 2)),
            "system": q["system"], "path_model": q["path_model"], "role": q["role"],
            "n_atoms": q["n_atoms"], "n_Q_dft": q["n_Q_dft"], "m_eff": q["m_eff"],
            "decide_T": q["decide_T"], "metrics": mt, "pbe_depth_meV": pm["depth_meV"],
            "depth_ratio": (mt["depth_meV"] / pm["depth_meV"]
                            if pm["depth_meV"] >= DEPTH_FLOOR_MEV else None),
            "dE_meV": (dE * 1000).tolist(),
            "n_pts": int(len(diff)), "e_sumsq_meV2": float(np.sum(diff ** 2)),
            "e_sumsq_per_atom_meV2": float(np.sum((diff / q["n_atoms"]) ** 2)),
            "e_rmse_meV": float(np.sqrt(np.mean(diff ** 2))),
            "e_rmse_window_meV": float(np.sqrt(np.mean(diff[win] ** 2))),
            "e_max_abs_meV": float(np.abs(diff).max()),
            "f_rmse_eV_A": float(np.sqrt(np.mean(dF ** 2))),
            "f_rms_pbe_eV_A": float(np.sqrt(np.mean(fp ** 2))), "n_force_components": int(dF.size),
            "f_sumsq": float(np.sum(dF ** 2)), "f_sumsq_pbe": float(np.sum(fp ** 2)),
            "calls": {}}
        for T in C.SCREEN_T:
            tasks.append((mt["a"], mt["b"], mt["c"], q["m_eff"], T))
            keys.append((q["stem"], T))
        if (n + 1) % 10 == 0:
            print(f"[c3a] {model}/{tag}: {n + 1}/{len(paths)} paths ({time.time() - t0:.0f} s)", flush=True)
    for (stem, T), r in zip(keys, solve_all(tasks, args.workers)):
        res[stem]["calls"][f"{T:g}"] = r
    return {"part": "c3a", "model": model, "tag": tag, "model_version": handle.version,
            "n_paths": len(res), "paths": res, "flags": pbe.flags,
            "provenance": C.provenance("evaluate:c3a", args)}


# ----------------------------------------------------------------------- part c3b ----

def part_c3b(args, handle, model, tag, out, pbe):
    """Force and energy errors on the C3b configurations of the test systems (every config the
    PBE queue finished).  Energy error per atom is of E(config) - E(undisplaced), each in its own
    code, exactly as dft_reference.analyze defines it."""
    sets = pbe.c3b_sets()
    ref_ml = {}
    rows = []
    cache = {}
    for rel, items in sets.items():
        frames = [a for a, _, _, _ in items]
        cache[rel] = eval_frames(handle.calc, frames)
    for rel, items in sets.items():
        if items[0][1] == "ref":
            ref_ml[(items[0][2]["system"], items[0][2]["owner_model"])] = (rel, cache[rel][0][0])
    for rel, items in sets.items():
        E, F = cache[rel]
        system, owner = items[0][2]["system"], items[0][2]["owner_model"]
        refrel, e_ml_ref = ref_ml.get((system, owner), (None, None))
        ref_dft = None
        ref_set = None
        if refrel is not None:
            ref_items = sets[refrel]
            ref_dft = ref_items[0][3]
            ref_set = pbe.settings(refrel, 0)
        for k, (a, role, info, d) in enumerate(items):
            if d is None or role == "ref":
                continue
            dF = F[k] - d["forces"]
            row = {"file": rel, "system": system, "owner_model": owner, "role": role,
                   "T": info.get("T"), "seed": info.get("seed"), "index": info.get("index"),
                   "n_atoms": len(a), "u_rms_A": info.get("u_rms_A"),
                   "f_sumsq": float(np.sum(dF ** 2)), "f_sumsq_pbe": float(np.sum(d["forces"] ** 2)),
                   "n_components": int(dF.size), "f_rmse": float(np.sqrt(np.mean(dF ** 2)))}
            if ref_dft is not None and pbe.settings(rel, k) == ref_set and e_ml_ref is not None:
                dEp = d["energy_eV"] - ref_dft["energy_eV"]
                dEm = E[k] - e_ml_ref
                row["e_err_meV_atom"] = float((dEm - dEp) / len(a) * 1000)
                row["dE_pbe_meV_atom"] = float(dEp / len(a) * 1000)
            rows.append(row)
    return {"part": "c3b", "model": model, "tag": tag, "model_version": handle.version,
            "configs": rows, "flags": pbe.flags, "provenance": C.provenance("evaluate:c3b", args)}


# ------------------------------------------------------------------------ part p2 ----

def part_p2(args, handle, model, tag, out):
    """The production harmonic -> soft-mode screen (cli.run_unit's softmode branch, with the
    calculator swapped) for the three systems at the ladder temperatures.  The E(Q) maps are
    T-independent and cached per (system, model, tag) under <out>/p2cache."""
    from mlip_dynstab import SOFTMODE_EQMAP_VERSION
    from mlip_dynstab.cli import _finite_t_gt
    from mlip_dynstab.finite_t import compute_finite_t_softmode
    from mlip_dynstab.systems import build_atoms, get_spec
    rows = []
    for system in args.systems:
        spec = get_spec(system)
        atoms = build_atoms(spec)
        cache = out / "p2cache" / (f"softmode_v{SOFTMODE_EQMAP_VERSION}m{C.MAX_MODES}_{system}"
                                   f"_{model}_{tag}_sc222.json")
        for T in C.LADDER_T:
            t0 = time.time()
            res = compute_finite_t_softmode(atoms, handle.calc, T, supercell=C.SUPERCELL,
                                            max_modes=C.MAX_MODES, cache_path=str(cache))
            row = res.as_row()
            gt = bool(_finite_t_gt(spec, T))
            row.update({"system": system, "model": model, "tag": tag, "T": float(T),
                        "gt_stable": gt, "pred_stable": bool(res.dynamically_stable),
                        "correct": bool(res.dynamically_stable) == gt,
                        "wall_s": round(time.time() - t0, 1)})
            rows.append(row)
            print(f"[p2] {system} {model}/{tag} T={T:g}: pred_stable={row['pred_stable']} gt={gt} "
                  f"min_eff {row['min_eff_freq_thz']:+.3f} THz ({row['wall_s']} s)", flush=True)
    return {"part": "p2", "model": model, "tag": tag, "model_version": handle.version,
            "rows": rows, "provenance": C.provenance("evaluate:p2", args)}


# ------------------------------------------------------------------------ part s2 ----

def part_s2(args, handle, model, tag, out):
    """Harmonic calls (cli.run_unit's harmonic branch) of the six stable controls."""
    from mlip_dynstab.harmonic import compute_harmonic
    from mlip_dynstab.systems import build_atoms, get_spec
    rows = []
    for system in C.CONTROLS:
        spec = get_spec(system)
        t0 = time.time()
        res = compute_harmonic(build_atoms(spec), handle.calc, supercell=C.SUPERCELL, disp=C.FC_DISP_ANG)
        row = res.as_row()
        row.update({"system": system, "model": model, "tag": tag, "pred_stable": bool(res.dynamically_stable),
                    "label_stable": bool(spec.harmonic_stable), "wall_s": round(time.time() - t0, 1)})
        rows.append(row)
        print(f"[s2] {system} {model}/{tag}: min_freq {row['min_freq_thz']:+.4f} THz stable={row['pred_stable']}",
              flush=True)
    return {"part": "s2", "model": model, "tag": tag, "model_version": handle.version,
            "rows": rows, "provenance": C.provenance("evaluate:s2", args)}


# ------------------------------------------------------------------------- P3 ----

def p3_commands(out: Path) -> str:
    rel = lambda p: C.rel(p)                                              # noqa: E731
    ck = rel(ckpt_path(out, "mace_mp0", "seed0"))
    od = rel(out / "sscha_converged_grid")
    L = ["# P3: converged SSCHA (the grid recipe, start A) on BaTiO3 100 K with the seed-0 fine-tuned MACE-MP-0.",
         "# scripts/sscha_seed_study.py is NOT edited: it loads its calculator through",
         "# mlip_dynstab.calculators.get_calculator, which honours MLIP_DYNSTAB_CKPT_MACE_MP0 (unset in production).",
         "# Writes only under --out-dir (never the production grid directory, never the ledger).",
         "cd /root/mlip-dynamic-stability",
         "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4",
         f"MLIP_DYNSTAB_CKPT_MACE_MP0=$(pwd)/{ck} \\",
         "  /root/env-mace/bin/python -u scripts/sscha_seed_study.py --preset grid --model mace_mp0 \\",
         f"  --system batio3_cubic --T 100 --start A --device cuda --out-dir {od} \\",
         "  2>&1 | tee /root/logs/ft_p3_sscha.log",
         "# estimated 0.5 h on a free GPU (cap 2.2 h: the relaxation cap is 7200 s)",
         f"# result: {od}/batio3_cubic_mace_mp0_100K_sc222_startA.json  -> run.hessian.dynamically_stable",
         f"# summarise: python scripts/finetune_trial.py evaluate --part summary"]
    return "\n".join(L)


# ----------------------------------------------------------------------- stage ----

def stage_evaluate(args) -> int:
    out = Path(args.out)
    if args.part == "summary":
        from . import summary
        return summary.build(out, Path(args.dft_root))
    if args.part == "p3-cmd":
        text = p3_commands(Path(C.rel(out)))
        p = out / "train" / "box_p3.sh"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/bash\n" + text + "\n", encoding="utf-8", newline="\n")
        print(text)
        return 0
    if args.model is None or args.tag is None:
        raise SystemExit("--model and --tag are required (base | seed0 | seed1 | seed2)")
    handle = load_handle(args.model, args.tag, out, args.device, args.ckpt)
    print(f"[evaluate] {args.model}/{args.tag}: {handle.version}")
    parts = ["c3a", "c3b", "p2", "s2"] if args.part == "all" else [args.part]
    pbe = None
    for part in parts:
        dest = out / "eval" / f"{part}_{args.model}_{args.tag}.json"
        if dest.exists() and not args.force:
            print(f"[skip] {dest.name}")
            continue
        if part in ("c3a", "c3b") and pbe is None:
            pbe = PBEData(Path(args.dft_root), args.systems)
            print(f"[evaluate] PBE: {pbe.n_ok} finished jobs; flags {len(pbe.flags)}")
        if part == "c3a":
            res = part_c3a(args, handle, args.model, args.tag, out, pbe)
        elif part == "c3b":
            res = part_c3b(args, handle, args.model, args.tag, out, pbe)
        elif part == "p2":
            res = part_p2(args, handle, args.model, args.tag, out)
        else:
            res = part_s2(args, handle, args.model, args.tag, out)
        C.jdump(dest, res)
        print(f"[evaluate] wrote {C.rel(dest)}")
    return 0
