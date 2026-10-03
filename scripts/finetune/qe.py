"""Stage `qe-inputs`: pw.x jobs for the training configurations, in the C3b job format.

pw_input() below is a faithful copy of scripts/dft_reference.py:pw_input (copied rather than
imported because that file is being extended concurrently); selftest.py regenerates an existing
C3b pw.in from its geometry and requires byte equality (apart from the one comment line naming
the writing script), so any drift between the two writers is caught.

Settings = those of the C3b jobs (read from their job.json): SSSP 1.3.0 PBE efficiency cutoffs
(max over the species present, per element from scripts/finetune/sssp_cutoffs.json), Gamma-centred
automatic k-mesh with n_i = max(1, ceil(|b_i| / 0.25 A^-1)) (|b| includes 2 pi), Marzari-
Vanderbilt smearing 0.005 Ry, conv_thr = 1e-10 x nat Ry, mixing_beta 0.4, tprnfor, no stress.
Job names: ft_<system>_<model>_c<NN> (NN = index in the configuration file) and
ft_<system>_<model>_ref (the undisplaced 2x2x2 cell of that base model's relaxed lattice).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import common as C

SCRIPT = "scripts/finetune_trial.py"
KSPACING_CONVENTION = ("n_i = max(1, ceil(|b_i| / spacing)) with b = 2*pi*inv(cell).T, i.e. |b_i| "
                       "includes 2*pi (the VASP KSPACING convention); Gamma-centred, unshifted")


def load_table(sssp_json=None) -> dict:
    """Per-element pseudopotential/cutoff table: the harvested one, cross-checked against the
    box's SSSP json when given."""
    raw = C.jload(C.FT_DIR / "sssp_cutoffs.json")
    table = raw["elements"]
    info = {"json": raw["sssp_json"], "sha256": raw["sssp_sha256"], "table": table,
            "checked_against": None}
    if sssp_json:
        box = C.jload(sssp_json)
        for el, d in table.items():
            b = box.get(el)
            if b is None:
                raise SystemExit(f"{sssp_json} has no entry for {el}")
            wfc = b.get("cutoff_wfc", b.get("cutoff"))
            rho = b.get("cutoff_rho")
            if rho is None and "dual" in b:
                rho = float(wfc) * float(b["dual"])
            got = (b["filename"], float(wfc), float(rho), b.get("md5"))
            want = (d["filename"], d["ecutwfc"], d["ecutrho"], d["md5"])
            if got != want:
                raise SystemExit(f"SSSP mismatch for {el}: json {got} vs harvested {want}")
        info["checked_against"] = {"path": str(sssp_json), "sha256": C.sha256_file(sssp_json)}
        info["sha256"] = info["checked_against"]["sha256"]
    return info


def kmesh(cell, spacing):
    rec = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, float)).T
    return [max(1, int(math.ceil(np.linalg.norm(b) / spacing - 1e-9))) for b in rec]


def nk_tr_upper(mesh):
    n = int(np.prod(mesh))
    self_inv = int(np.prod([2 if m % 2 == 0 else 1 for m in mesh]))
    return (n + self_inv) // 2


def nk_irr(atoms, mesh):
    try:
        import spglib
        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        mapping, _ = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0],
                                                   is_time_reversal=True, symprec=1e-5)
        return int(len(np.unique(mapping)))
    except Exception:
        return None


def _fortran(x):
    return f"{x:.3e}".replace("e", "d")


def structure_sha(atoms):
    frac = np.round(np.mod(np.round(atoms.get_scaled_positions(wrap=False), 6), 1.0), 6) + 0.0
    blob = json.dumps([atoms.get_chemical_symbols(), np.round(atoms.cell.array, 6).tolist(),
                       frac.tolist()], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def pw_input(atoms, job, sssp, spacing=0.25, degauss=0.005, script=SCRIPT):
    """(pw.in text, settings dict), insulators only (all three test systems are)."""
    from ase.data import atomic_masses, atomic_numbers
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    missing = [s for s in species if s not in sssp["table"]]
    if missing:
        raise SystemExit(f"no SSSP entry for {missing}")
    ecutwfc = max(sssp["table"][s]["ecutwfc"] for s in species)
    ecutrho = max(sssp["table"][s]["ecutrho"] for s in species)
    mesh = kmesh(atoms.cell.array, spacing)
    nat = len(atoms)
    conv = 1e-10 * nat
    L = ["&CONTROL",
         f"  ! job {job}, written by {script}",
         "  ! pseudo_dir is taken from $ESPRESSO_PSEUDO (exported by scripts/box/qe_queue.sh)",
         "  calculation = 'scf'", "  prefix = 'pwscf'", "  outdir = './scratch'",
         "  disk_io = 'none'", "  tprnfor = .true.", "  tstress = .false.",
         "  verbosity = 'low'", "/",
         "&SYSTEM", "  ibrav = 0", f"  nat = {nat}", f"  ntyp = {len(species)}",
         f"  ecutwfc = {ecutwfc:.1f}", f"  ecutrho = {ecutrho:.1f}",
         "  occupations = 'smearing'", "  smearing = 'mv'", f"  degauss = {degauss}", "/",
         "&ELECTRONS", f"  conv_thr = {_fortran(conv)}", "  mixing_beta = 0.4",
         "  electron_maxstep = 200", "/", "ATOMIC_SPECIES"]
    L += [f"{s} {atomic_masses[atomic_numbers[s]]:.4f} {sssp['table'][s]['filename']}"
          for s in species]
    L += ["CELL_PARAMETERS angstrom"]
    L += ["  " + " ".join(f"{x:.10f}" for x in v) for v in atoms.cell.array]
    L += ["ATOMIC_POSITIONS angstrom"]
    L += [f"{s} " + " ".join(f"{x:.10f}" for x in p)
          for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
    L += ["K_POINTS automatic", f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0", ""]
    settings = {"is_metal": False, "smearing": "mv", "degauss_Ry": degauss,
                "kspacing_inv_A": spacing, "kspacing_convention": KSPACING_CONVENTION,
                "kmesh": mesh, "nk_full": int(np.prod(mesh)), "nk_tr_upper": nk_tr_upper(mesh),
                "nk_irr_spglib": nk_irr(atoms, mesh),
                "ecutwfc_Ry": ecutwfc, "ecutrho_Ry": ecutrho, "conv_thr_Ry": conv,
                "mixing_beta": 0.4, "disk_io": "none", "functional": "PBE (from the SSSP PBE set)",
                "pseudos": {s: sssp["table"][s] for s in species},
                "sssp_json": sssp["json"], "sssp_sha256": sssp["sha256"]}
    return "\n".join(L), settings


# ------------------------------------------------------------------------------ cost ----

def _measured_per_k(dft_root: Path) -> dict:
    """{(system, n_atoms): median core-h per irreducible k-point} from the finished jobs of
    results/revision/dft/qe_jobs.csv (written by dft_reference.py analyze)."""
    p = dft_root / "qe_jobs.csv"
    rows = {}
    if not p.exists():
        return rows
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                ch, nk, nat = float(r["core_h"]), float(r["nk_irr"]), int(float(r["n_atoms"]))
            except (KeyError, ValueError, TypeError):
                continue
            if r.get("status") != "ok" or nk <= 0 or ch <= 0:
                continue
            name = r["job"]
            for s in C.TEST_SYSTEMS + ("srtio3_cubic",):
                if f"_{s}_" in name or name.endswith(f"_{s}"):
                    rows.setdefault((s, nat), []).append(ch / nk)
    return {k: float(np.median(v)) for k, v in rows.items()}


def estimate_core_h(system: str, nat: int, nk: int, per_k: dict) -> tuple[float, str]:
    """Modelled core-hours of one job from MEASURED C3a/C3b costs (not an assumed c0): direct
    (same system, same atom count) where there is one, else BaTiO3's measured per-k cost at
    this size scaled by this system's per-k ratio to BaTiO3 at the largest size measured for both."""
    if (system, nat) in per_k:
        return nk * per_k[(system, nat)], "measured same system/size"
    ref = "batio3_cubic"
    common = sorted(n for (s, n) in per_k if s == system and (ref, n) in per_k)
    if (ref, nat) in per_k and common:
        ratio = per_k[(system, common[-1])] / per_k[(ref, common[-1])]
        return nk * per_k[(ref, nat)] * ratio, f"BaTiO3 {nat}-atom x ({system}/BaTiO3 at {common[-1]} atoms = {ratio:.2f})"
    return float("nan"), "no measured analogue"


# ------------------------------------------------------------------------------ stage ----

def _write_job(qe_root: Path, job: str, atoms, member: dict, sssp, args, prov) -> str:
    """'new' | 'same' | 'refused'.  Never overwrites a pw.in under a finished pw.out unless
    --force, and then moves the old outputs aside (their JOB DONE would make the queue skip)."""
    text, settings = pw_input(atoms, job, sssp, args.kspacing_insulator, args.degauss_insulator)
    d = qe_root / job
    pw = d / "pw.in"
    status = "new"
    if pw.exists():
        if pw.read_text(encoding="utf-8") == text:
            return "same"
        if (d / "pw.out").exists() and not args.force:
            return "refused"
        stamp = C.now_utc().replace(":", "")
        for f in ("pw.out", "pw.err", "pw.in.ran"):
            if (d / f).exists():
                os.replace(d / f, d / f"{f}.superseded-{stamp}")
        status = "rewritten"
    d.mkdir(parents=True, exist_ok=True)
    with open(pw, "w", encoding="utf-8", newline="\n") as fh:           # LF: pw.x runs on Linux
        fh.write(text)
    meta = {"job": job, "system": member["system"], "structure_sha": structure_sha(atoms),
            "n_atoms": len(atoms), "species": sorted(set(atoms.get_chemical_symbols())),
            "members": [member], "qe": settings,
            "pw_in_sha256": hashlib.sha256(text.encode()).hexdigest(), "provenance": prov}
    C.jdump(d / "job.json", meta)
    return status


def stage_qe_inputs(args) -> int:
    from . import gen
    out, qe_root = Path(args.out), Path(args.qe_root)
    sssp = load_table(args.sssp_json)
    manifest = C.jload(out / "manifest.json")
    prov = C.provenance("qe-inputs", args)
    per_k = _measured_per_k(C.DFT_ROOT)
    counts = {"new": 0, "same": 0, "rewritten": 0, "refused": 0}
    plan, jobs_index = [], {}
    held_cache = {}
    for system in args.systems:
        for model in args.models:
            key = f"{system}__{model}"
            ent = manifest["entries"].get(key)
            if ent is None:
                raise SystemExit(f"no configurations for {key}: run `configs` first")
            frames = C.read_frames(out / ent["configs_file"])
            if gen.config_hash(frames) != ent["config_hash"]:
                raise SystemExit(f"{ent['configs_file']} no longer matches the manifest")
            if system not in held_cache:
                held_cache[system] = gen.HeldOut(system)
            chk = gen.recheck_guard(frames, system, held_cache[system])
            if chk["n_violations"]:
                raise SystemExit(f"{key}: {chk['n_violations']} configurations are within "
                                 f"{C.LEAK_RMS_A} A of a held-out geometry now; regenerate")
            ref = C.read_frames(out / ent["ref_file"])[0]
            todo = [(f"ft_{system}_{model}_c{i:02d}", a, i, "train_pool") for i, a in enumerate(frames)]
            todo.append((f"ft_{system}_{model}_ref", ref, -1, "ref"))
            for job, a, i, role in todo:
                member = {"study": "ft", "geom_file": "../finetune/" + ent["configs_file"]
                          if role != "ref" else "../finetune/" + ent["ref_file"],
                          "frame": max(i, 0), "role": role, "system": system, "base_model": model,
                          "T": float(a.info.get("T", 0.0)), "draw": int(a.info.get("draw", -1))}
                if args.no_write:
                    text, settings = pw_input(a, job, sssp, args.kspacing_insulator,
                                              args.degauss_insulator)
                    st = "counted"
                else:
                    st = _write_job(qe_root, job, a, member, sssp, args, prov)
                    counts[st] += 1
                    settings = C.jload(qe_root / job / "job.json")["qe"]
                nk = settings["nk_irr_spglib"] or settings["nk_tr_upper"]
                core_h, how = estimate_core_h(system, len(a), nk, per_k)
                plan.append({"job": job, "system": system, "model": model, "role": role,
                             "n_atoms": len(a), "kmesh": settings["kmesh"], "nk": nk,
                             "ecutwfc": settings["ecutwfc_Ry"], "ecutrho": settings["ecutrho_Ry"],
                             "core_h_est": core_h, "estimate_basis": how, "status": st})
    n_jobs = len(plan)
    tot = float(np.nansum([p["core_h_est"] for p in plan]))
    print(f"[qe-inputs] {n_jobs} jobs ({sum(p['role'] == 'train_pool' for p in plan)} configurations "
          f"+ {sum(p['role'] == 'ref' for p in plan)} undisplaced references); {counts}")
    by = {}
    for p in plan:
        g = by.setdefault((p["system"], p["model"]), [0, 0.0, p["kmesh"], p["ecutwfc"], p["ecutrho"], set()])
        g[0] += 1
        g[1] += p["core_h_est"]
        g[5].add(p["estimate_basis"])
    for (s, m), (n, h, km, ew, er, how) in sorted(by.items()):
        print(f"   {s:14s} {m:9s} {n:3d} jobs  k {'x'.join(map(str, km))}  ecut {ew:g}/{er:g} Ry  "
              f"~{h:6.0f} core-h   ({'; '.join(sorted(how))})")
    print(f"[qe-inputs] modelled total {tot:.0f} core-h (from measured C3a/C3b per-k costs); at 40 "
          f"busy cores (-r 8 -n 5, as C3b) ~{tot / 40:.1f} h wall, at 56 ~{tot / 56:.1f} h")
    C.jdump(out / "qe_plan.json", {"provenance": prov, "n_jobs": n_jobs, "core_h_total": tot,
                                   "counts": counts, "jobs": plan,
                                   "cost_basis": "median core-h per irreducible k-point of the "
                                                 "finished C3a/C3b jobs in results/revision/dft/qe_jobs.csv"})
    return 0
