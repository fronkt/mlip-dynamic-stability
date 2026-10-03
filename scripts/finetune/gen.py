"""Stage `configs`: phonon-rattled 2x2x2 training configurations and the leakage guard.

The sampler is the force-spread study's (scripts/force_spread.py, C2) imported unchanged:
normal modes of the supercell dynamical matrix at Gamma with the translations removed
exactly, |omega| for imaginary modes with C2's 0.5 THz floor, the quantum harmonic mode
variance, Cartesian displacements u = M^(-1/2) E q.  What differs from C2, and why:

  * the force constants come from ONE model (the base model being fine-tuned), not the mean of
    five, because the pre-registration asks for "the base model's harmonic force constants";
  * the geometry is that model's own relaxed cubic cell (cli.run_unit / compute_harmonic
    relax first), not the unrelaxed registry cell, because every held-out C3a path and C3b
    configuration sits on a model-relaxed lattice and the training cells must live where the
    test cells do;
  * 13/13/14 draws at 100/300/600 K with a per-(system, model, T) RNG stream, and a leakage
    guard that redraws (continuing the same stream) any configuration within 0.05 A RMS of a
    held-out geometry.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from . import common as C

_FS = None


def force_spread():
    """scripts/force_spread.py as a module (single source of truth for the sampler)."""
    global _FS
    if _FS is None:
        p = C.SCRIPTS / "force_spread.py"
        spec = importlib.util.spec_from_file_location("force_spread_c2", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FS = mod
    return _FS


# --------------------------------------------------------------------- force constants ----

def relaxed_fc(system: str, model: str, device: str):
    """Relax the primitive cell with the base model (production relaxation), then symmetrised
    2x2x2 finite-displacement force constants at the production amplitude.  The residual force
    of the undisplaced supercell is subtracted as in C2 (a no-op at a relaxed equivariant cell)."""
    import ase
    from phonopy import Phonopy
    from mlip_dynstab.calculators import get_calculator
    from mlip_dynstab.harmonic import _ase_to_phonopy_atoms, _relax
    from mlip_dynstab.systems import build_atoms, get_spec

    handle = get_calculator(model, device=device)
    t0 = time.time()
    prim = build_atoms(get_spec(system))
    prim.calc = handle.calc
    prim = _relax(prim, fmax=C.RELAX_FMAX)
    ph = Phonopy(_ase_to_phonopy_atoms(prim), supercell_matrix=np.diag(C.SUPERCELL),
                 primitive_matrix="auto")
    ph.generate_displacements(distance=C.FC_DISP_ANG)
    sc = ph.supercell

    def forces(symbols, positions, cell):
        a = ase.Atoms(symbols, positions=positions, cell=cell, pbc=True)
        a.calc = handle.calc
        return np.asarray(a.get_forces(), dtype=np.float64)

    f0 = forces(sc.symbols, sc.positions, sc.cell)
    fs = [forces(s.symbols, s.positions, s.cell) - f0 for s in ph.supercells_with_displacements]
    ph.forces = np.array(fs)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    fc = np.asarray(ph.force_constants, dtype=np.float64)
    nat = len(sc.numbers)
    if fc.shape != (nat, nat, 3, 3):
        raise RuntimeError(f"expected full force constants {(nat, nat, 3, 3)}, got {fc.shape}")
    return {"fc": fc, "positions": np.asarray(sc.positions, float),
            "cell": np.asarray(sc.cell, float), "numbers": np.asarray(sc.numbers, int),
            "masses": np.asarray(sc.masses, float),
            "prim_cell": np.asarray(prim.cell.array, float),
            "model_version": handle.version, "n_disp": len(fs),
            "residual_force_rms": float(np.sqrt(np.mean(f0 ** 2))),
            "wall_s": round(time.time() - t0, 1)}


def load_or_make_fc(system: str, model: str, out: Path, device: str, force=False):
    p = out / "fc" / f"fc_{system}__{model}.npz"
    if p.exists() and not force:
        with np.load(p) as z:
            d = {k: z[k] for k in z.files if k != "meta"}
            d["meta"] = json.loads(str(z["meta"]))
        d["from_cache"] = True
        return d, p
    d = relaxed_fc(system, model, device)
    meta = {k: d[k] for k in ("model_version", "n_disp", "residual_force_rms", "wall_s")}
    meta["utc"] = C.now_utc()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, fc=d["fc"], positions=d["positions"], cell=d["cell"],
                            numbers=d["numbers"], masses=d["masses"], prim_cell=d["prim_cell"],
                            meta=np.array(json.dumps(meta)))
    os.replace(tmp, p)
    d["meta"] = meta
    d["from_cache"] = False
    return d, p


# ------------------------------------------------------------------- held-out geometries ----

def _tile_path(frames, sites):
    """(n_frames, 40, 3) normalised displacement fields of one C3a path file, tiled onto the
    2x2x2 ideal sites; None if the modulated cell is not a sublattice of the 2x2x2 cell.

    Frame 0 of every path file is Q = 0, the ideal structure, so the field of frame k is
    pos_k - pos_0 exactly (no site search).  Each ideal 2x2x2 site is matched to the atom of the
    path cell whose ideal position differs from it by a lattice vector of the path cell."""
    a0 = frames[0]
    a = C.lattice_constant(a0)
    x0 = a0.get_positions() / a
    L = np.asarray(a0.cell.array, float) / a
    Linv = np.linalg.inv(L)
    s_num, s_x = sites
    z = a0.get_atomic_numbers()
    # the ideal frame may sit at an origin shift of the prototype: try the shift that puts the
    # first A-site (species of site 0) on the origin
    shift = x0[int(np.where(z == s_num[0])[0][0])] - s_x[0]
    idx = np.empty(len(s_num), int)
    for s in range(len(s_num)):
        cand = np.where(z == s_num[s])[0]
        coef = (s_x[s] + shift - x0[cand]) @ Linv
        err = np.abs(coef - np.round(coef)).max(axis=1)
        k = int(np.argmin(err))
        if err[k] > 1e-3:
            return None
        idx[s] = cand[k]
    p0 = a0.get_positions()[idx]
    return np.stack([(fr.get_positions()[idx] - p0) / a for fr in frames]), a


class HeldOut:
    """Every held-out geometry of one system, as normalised displacement fields.

    C3a: all frames of every path file geom/<system>_*.extxyz (every model's paths, deciding,
    reference and extra modes alike: whichever is a test point of the fine-tuned model).
    C3b: every frame of geom_c3b/<system>_*.extxyz (SSCHA configurations, the undisplaced
    reference and the rattled baselines).  Matching on SYSTEM rather than system-and-model
    is the stricter reading of the pre-registration."""

    def __init__(self, system: str, dft_root: Path = C.DFT_ROOT):
        self.system = system
        self.sites = C.ideal_sites(system)
        F, A, SRC, self.skipped = [], [], [], []
        n_files = 0
        for kind, d in (("c3a", dft_root / "geom"), ("c3b", dft_root / "geom_c3b")):
            for f in sorted(d.glob(f"{system}_*.extxyz")):
                if f.name.endswith(".tmp.extxyz"):
                    continue
                n_files += 1
                frames = C.read_frames(f)
                if kind == "c3a":
                    r = _tile_path(frames, self.sites)
                    if r is None:
                        self.skipped.append({"file": f.name, "why": "cell not a sublattice of 2x2x2"})
                        continue
                    fld, a = r
                    F.append(fld)
                    A += [a] * len(frames)
                    SRC += [(f.name, k, kind) for k in range(len(frames))]
                else:
                    for k, fr in enumerate(frames):
                        fld = C.field_from_sites(fr.get_atomic_numbers(), fr.get_positions(),
                                                 fr.cell.array, self.sites)
                        if fld is None:
                            self.skipped.append({"file": f"{f.name}#{k}", "why": "no site map"})
                            continue
                        F.append(fld[None])
                        A.append(C.lattice_constant(fr))
                        SRC.append((f.name, k, kind))
        self.n_files = n_files
        self.F = np.concatenate(F, axis=0) if F else np.zeros((0, len(self.sites[0]), 3))
        self.a = np.asarray(A, float)
        self.src = SRC

    def nearest(self, field, a_ang: float):
        """(distance in A, src tuple, is_cross_lattice) of the closest held-out frame."""
        d = C.rms_distance(field[None], self.F, a_ang)
        i = int(np.argmin(d))
        return float(d[i]), self.src[i], bool(abs(self.a[i] - a_ang) / a_ang > 1e-4)

    def summary(self) -> dict:
        return {"n_files": self.n_files, "n_frames": int(len(self.F)),
                "n_c3a_frames": int(sum(s[2] == "c3a" for s in self.src)),
                "n_c3b_frames": int(sum(s[2] == "c3b" for s in self.src)),
                "skipped": self.skipped}


# ------------------------------------------------------------------------------ drawing ----

def draw_configs(system: str, model: str, fcd: dict, held: HeldOut, max_tries: int = 100000,
                 min_pair_ratio: float = 0.0):
    """All 40 accepted draws of one (system, model) plus the full rejection record.

    A draw is redrawn (the same per-T stream continues) only if it lies within LEAK_RMS_A of a
    held-out geometry.  ``min_pair_ratio`` > 0 adds an OPTIONAL second redraw rule, off by
    default because the pre-registration does not contain it: a draw whose closest pair of
    atoms is nearer than that fraction of the reference distance of the same species pair is
    redrawn too (force_spread.py's 0.75 variant (g) threshold).  Its effect is reported in the
    manifest whether on or off, so the choice can be made on numbers."""
    from ase import Atoms
    FS = force_spread()
    masses, numbers, cell, ref = fcd["masses"], fcd["numbers"], fcd["cell"], fcd["positions"]
    lam, evecs = FS.normal_modes(fcd["fc"], masses)
    nu = FS.signed_thz(lam)
    nu_used = np.maximum(np.abs(nu), FS.FREQ_FLOOR_THZ)
    contacts = FS.Contacts(numbers, cell, ref)
    a_ang = C.lattice_constant(_Atoms(numbers, cell))
    frames, per_t, rej_all = [], {}, []
    for T, n in C.N_PER_T.items():
        rng = C.rng_for(system, model, T)
        acc, rej = [], []
        tries = n_unassignable = n_contact_rej = 0
        while len(acc) < n:
            tries += 1
            if tries > max_tries or (tries >= 3000 and not acc):
                raise RuntimeError(
                    f"{system}/{model} {T:g} K: {len(acc)} of {n} accepted after {tries - 1} draws "
                    f"({n_contact_rej} rejected on contact at ratio {min_pair_ratio}): the redraw "
                    "rule is infeasible for this spectrum; lower --min-pair-ratio or drop it")
            u = FS.sample_displacements(evecs, nu_used, masses, T, 1, rng)[0]
            com = np.abs(np.einsum("i,ia->a", masses, u)).max()
            if com > 1e-8 * masses.sum():
                raise RuntimeError(f"{system}/{model} {T} K: net displacement {com}")
            pos = ref + u
            dmin, ratio, pair = contacts(pos)
            fld = C.field_from_sites(numbers, pos, cell, held.sites)
            if fld is None:
                # an atom is > 0.3 a from its own site: not a perturbation of the prototype, so
                # no held-out geometry (all within ~0.1 a of the ideal) can be near it
                dist, src, cross = float("inf"), ("none", -1, "none"), False
                n_unassignable += 1
            else:
                dist, src, cross = held.nearest(fld, a_ang)
            rec = {"T": T, "try": tries, "nearest_heldout_A": dist, "nearest_src": list(src),
                   "cross_lattice": cross, "min_pair_ratio": ratio}
            if dist < C.LEAK_RMS_A:
                rec["reason"] = "leak"
                rej.append(rec)
                continue
            if min_pair_ratio > 0 and ratio < min_pair_ratio:
                rec["reason"] = "contact"
                n_contact_rej += 1
                rej.append(rec)
                continue
            rec.update({"index_in_T": len(acc), "rms_disp_A": float(np.sqrt((u ** 2).sum(1).mean())),
                        "max_disp_A": float(np.linalg.norm(u, axis=1).max()),
                        "min_dist_A": dmin, "min_pair_ratio": ratio, "min_pair": pair,
                        "site_map_ok": fld is not None})
            acc.append((u, pos, rec))
        per_t[f"{T:g}"] = {
            "n": n, "n_draws": tries, "n_rejected": len(rej),
            "n_rejected_leak": len(rej) - n_contact_rej, "n_rejected_contact": n_contact_rej,
            "n_unassignable_draws": n_unassignable,
            "n_accepted_unassignable": int(sum(not r["site_map_ok"] for _, _, r in acc)),
            "min_pair_ratio_rule": min_pair_ratio,
            "rng": [C.CONFIG_SEED, "crc32(system)", "crc32(model)", int(round(T))],
            "rms_disp_A_mean": float(np.mean([r["rms_disp_A"] for _, _, r in acc])),
            "rms_disp_A_max": float(np.max([r["rms_disp_A"] for _, _, r in acc])),
            "min_pair_ratio_min": float(np.min([r["min_pair_ratio"] for _, _, r in acc])),
            "min_pair_ratio_median": float(np.median([r["min_pair_ratio"] for _, _, r in acc])),
            "n_pair_ratio_below_0.75": int(sum(r["min_pair_ratio"] < FS.MIN_DIST_RATIO
                                               for _, _, r in acc)),
            "min_dist_A_min": float(np.min([r["min_dist_A"] for _, _, r in acc])),
            "nearest_heldout_A_min": float(np.min([min(r["nearest_heldout_A"], 99.0)
                                                   for _, _, r in acc]))}
        rej_all += rej
        for u, pos, rec in acc:
            at = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
            at.info.update({"system": system, "base_model": model, "role": "train_pool",
                            "T": float(T), "draw": int(rec["index_in_T"]),
                            "n_tries": int(per_t[f"{T:g}"]["n_draws"]),
                            "rms_disp_A": rec["rms_disp_A"],
                            "min_pair_ratio": rec["min_pair_ratio"],
                            "nearest_heldout_A": min(rec["nearest_heldout_A"], 99.0),
                            "site_map_ok": int(rec["site_map_ok"]),
                            "nearest_heldout_src": f"{rec['nearest_src'][0]}#{rec['nearest_src'][1]}"})
            at.arrays["u_disp"] = np.asarray(u, float)
            frames.append(at)
    ref_atoms = Atoms(numbers=numbers, positions=ref, cell=cell, pbc=True)
    ref_atoms.info.update({"system": system, "base_model": model, "role": "ref", "T": 0.0})
    spectrum = {"min_freq_thz": float(nu.min()), "max_freq_thz": float(nu.max()),
                "n_modes": int(len(nu)), "n_imaginary": int(np.sum(nu < 0)),
                "n_floored": int(np.sum(np.abs(nu) < FS.FREQ_FLOOR_THZ)),
                "floor_thz": FS.FREQ_FLOOR_THZ}
    return frames, ref_atoms, per_t, rej_all, spectrum, contacts.ref_by_pair


class _Atoms:
    def __init__(self, numbers, cell):
        self.numbers = numbers
        self.cell = type("Cell", (), {"array": np.asarray(cell, float)})()

    def __len__(self):
        return len(self.numbers)


def config_hash(frames) -> str:
    """Hash of the positions and cell as READ BACK from the extxyz (rounded to 1e-6 A, well
    above the 1e-8 text precision), so the value is the same at write time and at every
    later read."""
    return C.sha256_arrays(*[np.round(f.get_positions(), 6) for f in frames],
                           np.round(frames[0].cell.array, 6))


def recheck_guard(frames, system: str, held: HeldOut) -> dict:
    """Re-measure the nearest held-out distance of already-written configurations (qe-inputs and
    dataset call this, so a held-out set that grew since `configs` cannot leak unnoticed)."""
    worst = []
    for fr in frames:
        fld = C.field_from_sites(fr.get_atomic_numbers(), fr.get_positions(), fr.cell.array,
                                 held.sites)
        worst.append(float("inf") if fld is None
                     else held.nearest(fld, C.lattice_constant(fr))[0])
    worst = np.asarray(worst)
    return {"n": int(len(worst)), "min_nearest_A": float(worst.min()),
            "n_violations": int((worst < C.LEAK_RMS_A).sum()), "radius_A": C.LEAK_RMS_A,
            "n_heldout_frames": int(len(held.F))}


def _one_process_per_model(args, stage: str) -> int | None:
    """MACE sets torch's default dtype to float64 at import of its calculator, which breaks a
    CHGNet loaded in the same process (float != double in the composition model).  The pinned
    box envs hold one backend each; locally both are installed, so several models are run as
    one child process per model."""
    import subprocess
    import sys
    if len(args.models) <= 1:
        return None
    rc = 0
    for m in args.models:
        cmd = [sys.executable, str(C.SCRIPTS / "finetune_trial.py"), stage,
               "--out", str(args.out), "--systems", *args.systems, "--models", m,
               "--device", args.device] + (["--force"] if args.force else [])
        if getattr(args, "min_pair_ratio", 0.0):
            cmd += ["--min-pair-ratio", str(args.min_pair_ratio)]
        print(f"[{stage}] child process: {' '.join(cmd[2:])}", flush=True)
        rc = max(rc, subprocess.run(cmd).returncode)
    return rc


def stage_configs(args) -> int:
    rc = _one_process_per_model(args, "configs")
    if rc is not None:
        return rc
    out = Path(args.out)
    FS = force_spread()
    selftest = FS.selftest_sampler()
    print(f"[configs] C2 sampler selftest passed: {selftest}")
    manifest_p = out / "manifest.json"
    manifest = C.jload(manifest_p) if manifest_p.exists() else {"entries": {}}
    fs_sha = C.sha256_file(C.SCRIPTS / "force_spread.py")
    manifest.setdefault("design", {
        "n_per_T": {f"{k:g}": v for k, v in C.N_PER_T.items()}, "supercell": list(C.SUPERCELL),
        "leak_rms_A": C.LEAK_RMS_A, "config_seed": C.CONFIG_SEED,
        "rms_definition": "sqrt(mean over atoms of |d|^2), d = difference of the two displacement "
                          "fields from the ideal Pm-3m sites with its mean over atoms removed; "
                          "fields compared in units of the lattice constant a and converted "
                          "with the training configuration's own a (identical cells: exact "
                          "Cartesian comparison; different relaxed lattices: fractional)"})
    held_cache = {}
    n_bad = 0
    for system in args.systems:
        for model in args.models:
            key = f"{system}__{model}"
            cfg_p = out / "configs" / f"{system}_{model}.extxyz"
            ref_p = out / "configs" / f"{system}_{model}_ref.extxyz"
            ent = manifest["entries"].get(key)
            if cfg_p.exists() and ent and not args.force:
                fr = C.read_frames(cfg_p)
                if config_hash(fr) != ent["config_hash"]:
                    raise SystemExit(f"{cfg_p} no longer matches its manifest hash; delete both "
                                     "(and everything downstream) to regenerate")
                print(f"[skip] {key}: {len(fr)} configurations already frozen")
                continue
            if system not in held_cache:
                held_cache[system] = HeldOut(system)
                print(f"[configs] held-out {system}: {held_cache[system].summary()}")
            held = held_cache[system]
            fcd, fc_p = load_or_make_fc(system, model, out, args.device, force=args.force)
            print(f"[configs] {key}: FCs {'cached' if fcd['from_cache'] else 'computed'} "
                  f"({fcd['meta']['model_version']})")
            frames, ref_atoms, per_t, rej, spec, refpair = draw_configs(
                system, model, fcd, held, min_pair_ratio=args.min_pair_ratio)
            for t, v in per_t.items():
                print(f"   T={t:>4s} K  draws {v['n_draws']:3d} ({v['n_rejected_leak']} leak, "
                      f"{v['n_rejected_contact']} contact rejected, {v['n_unassignable_draws']} "
                      f"unassignable)  "
                      f"rms {v['rms_disp_A_mean']:.3f} A  min pair ratio {v['min_pair_ratio_min']:.2f} "
                      f"(median {v['min_pair_ratio_median']:.2f}, {v['n_pair_ratio_below_0.75']} < 0.75)")
            C.write_frames(cfg_p, frames)
            C.write_frames(ref_p, [ref_atoms])
            frames = C.read_frames(cfg_p)         # what every later stage will see
            chk = recheck_guard(frames, system, held)
            if chk["n_violations"]:
                n_bad += 1
                print(f"[configs] !!! {key}: {chk['n_violations']} guard violations after writing")
            a_lat = C.lattice_constant(frames[0])
            manifest["entries"][key] = {
                "system": system, "model": model, "model_version": fcd["meta"]["model_version"],
                "relaxed_lattice_a_A": a_lat, "n_atoms": len(frames[0]),
                "fc_file": C.rel(fc_p, out), "fc_sha256": C.sha256_file(fc_p),
                "fc_residual_force_rms": fcd["meta"]["residual_force_rms"],
                "spectrum": spec, "ref_pair_min_A": refpair,
                "force_spread_sha256": fs_sha, "sampler_selftest_passed": bool(selftest["passed"]),
                "min_pair_ratio_rule": args.min_pair_ratio,
                "per_T": per_t, "n_configs": len(frames),
                "configs_file": C.rel(cfg_p, out), "configs_sha256": C.sha256_file(cfg_p),
                "ref_file": C.rel(ref_p, out), "config_hash": config_hash(frames),
                "guard": {**chk, "heldout": held.summary(),
                          "n_rejected_total": len(rej),
                          "rejected_nearest_A": sorted(r["nearest_heldout_A"] for r in rej),
                          "rejected_src_counts": _count([f"{r['nearest_src'][2]}:{r['nearest_src'][0]}"
                                                         for r in rej]),
                          "rejected_cross_lattice": int(sum(r["cross_lattice"] for r in rej)),
                          # held-out geometries on this configuration's own lattice are compared
                          # exactly (identical cells); the others in fractional coordinates
                          "heldout_frames_same_lattice": int(np.sum(np.abs(held.a - a_lat) / a_lat < 1e-4)),
                          "heldout_frames_other_lattice": int(np.sum(np.abs(held.a - a_lat) / a_lat >= 1e-4)),
                          "heldout_lattice_A_range": [float(held.a.min()), float(held.a.max())],
                          "accepted_nearest_A_min": float(min(f.info["nearest_heldout_A"]
                                                              for f in frames)),
                          "accepted_nearest_A_median": float(np.median(
                              [f.info["nearest_heldout_A"] for f in frames]))},
                "provenance": C.provenance("configs", args)}
            manifest["provenance_last"] = C.provenance("configs", args)
            C.jdump(manifest_p, manifest)
            g = manifest["entries"][key]["guard"]
            print(f"[configs] {key}: wrote {len(frames)} + ref; guard rejected "
                  f"{g['n_rejected_total']}, nearest accepted {g['accepted_nearest_A_min']:.3f} A, "
                  f"a = {a_lat:.4f} A, hash {manifest['entries'][key]['config_hash'][:12]}")
    return 4 if n_bad else 0


def _count(items):
    out = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out
