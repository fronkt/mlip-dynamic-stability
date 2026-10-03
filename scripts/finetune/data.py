"""Stage `dataset`: pw.x outputs -> training / validation extxyz for one base model.

Run in the base model's env (it evaluates the base model on every training configuration to
fix the per-system energy reference, below).

Energy handling (the 'per-element reference' question).  PBE total energies from QE/SSSP sit
~10^3 eV per cell away from the scale the foundation models were trained on (VASP/MPtrj), and
that offset is composition-dependent.  Both tools expect it to be absorbed by the atomic
reference, and both document it:
  * MACE:   --E0s="estimated" (the documented fine-tuning command): a least-squares fit of
            per-element E0 corrections that make the FOUNDATION model's predictions match the
            training energies on average;
  * CHGNet: Trainer.train(train_composition_model=True), or AtomRef.fit(), for "alternative DFT
            codes (r2SCAN, Gaussian, QE) with incompatible elemental energies".
With three fixed compositions the only identifiable part of any such fit is one constant per
system, and Adam at the documented lr cannot move CHGNet's AtomRef by 10^3 eV in a handful of
epochs.  So the offset is applied here, once, identically for both tools: for each system,

    c_sys = mean over its TRAINING configurations of ( E_PBE - E_base ),   target = E_PBE - c_sys,

with E_base the total energy of the base model on the same geometry.  This is exactly the
composition-constant a least-squares E0 estimate would find, computed on training data only
(validation and held-out data never enter c_sys).  A per-cell constant carries no information
about E(Q) or forces, so it cannot move any well depth; it only puts the targets on the energy
scale the pretrained network already predicts.  MACE then needs no further correction (its
--E0s="estimated" runs anyway, as documented, and finds ~0) and CHGNet's composition model is
left frozen.  Energies in the files are TOTAL energies per cell in eV; the raw PBE energy is kept
in `energy_qe_eV`.  QE '!' energy is the smeared free energy (forces consistent); no stress
(tstress is off in the C3b format), so stress is not a training target for either tool.

Split: random 90/10 by configuration over the PLANNED pool (all systems pooled), fixed seed,
fixed BEFORE any job finishes, so a failed SCF removes a configuration without reshuffling the
rest.  The held-out sets (C3a path points, C3b configurations) are never read here.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

from . import common as C
from . import gen


def parse_pw_out(path) -> dict:
    """Same contract as dft_reference.parse_pw_out (copied): energy in eV (the '!' free energy),
    forces in eV/A, convergence flags."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    r = {"job_done": "JOB DONE" in txt,
         "converged": ("convergence has been achieved" in txt
                       and "convergence NOT achieved" not in txt)}
    m = re.findall(r"convergence has been achieved in\s+(\d+)\s+iterations", txt)
    r["n_scf"] = int(m[-1]) if m else None
    if r["job_done"] and r["converged"]:
        from ase.io import read
        try:
            a = read(path, format="espresso-out", index=-1)
            r["energy_eV"] = float(a.get_potential_energy())
            r["forces"] = np.asarray(a.get_forces(), float)
            r["atoms"] = a
        except Exception as exc:                                  # noqa: BLE001
            r["parse_error"] = f"{type(exc).__name__}: {exc}"
    return r


def pos_mismatch(parsed, ref) -> float:
    if parsed.get_chemical_symbols() != ref.get_chemical_symbols():
        return float("inf")
    d = parsed.get_scaled_positions(wrap=False) - ref.get_scaled_positions(wrap=False)
    d -= np.round(d)
    cell_d = float(np.abs(parsed.cell.array - ref.cell.array).max())
    return max(float(np.linalg.norm(d @ ref.cell.array, axis=1).max()), cell_d)


def planned_split(model: str, systems, n_per_system: int):
    """(ids, is_val) over the planned pool; deterministic and independent of which jobs ran."""
    ids = [(s, i) for s in systems for i in range(n_per_system)]
    rng = np.random.default_rng([C.SPLIT_SEED, zlib.crc32(model.encode())])
    perm = rng.permutation(len(ids))
    n_val = int(round(C.VAL_FRACTION * len(ids)))
    val = set(int(k) for k in perm[:n_val])
    return ids, [k in val for k in range(len(ids))]


def collect_labels(out: Path, qe_root: Path, model: str, system: str, ent: dict, allow_partial: bool,
                   mock: bool = False, calc=None):
    """Labelled frames of one (system, model): list of (index, atoms, E_pbe, F_pbe | None)."""
    frames = C.read_frames(out / ent["configs_file"])
    res, problems = [], []
    for i, a in enumerate(frames):
        job = f"ft_{system}_{model}_c{i:02d}"
        if mock:
            b = a.copy()
            b.calc = calc
            f = b.get_forces()
            e = b.get_potential_energy()
            rng = np.random.default_rng([7, i])
            res.append((i, a, float(e) - 1000.0 + 0.3 * rng.normal(), 0.9 * f + 0.02 * rng.normal(size=f.shape)))
            continue
        d = qe_root / job
        po = d / "pw.out"
        if not po.exists():
            problems.append((job, "no pw.out"))
            continue
        r = parse_pw_out(po)
        ran, pin = d / "pw.in.ran", d / "pw.in"
        if not r["job_done"]:
            problems.append((job, "incomplete"))
        elif not ran.exists() or ran.read_bytes() != pin.read_bytes():
            problems.append((job, "pw.out was not produced by the current pw.in"))
        elif not r["converged"]:
            problems.append((job, "SCF not converged"))
        elif "parse_error" in r:
            problems.append((job, r["parse_error"]))
        else:
            mis = pos_mismatch(r["atoms"], a)
            if mis > 1e-5:
                problems.append((job, f"structure differs from its configuration by {mis:.2e} A"))
            else:
                res.append((i, a, r["energy_eV"], r["forces"]))
    return res, problems


def stage_dataset(args) -> int:
    from mlip_dynstab.calculators import get_calculator
    out, model = Path(args.out), args.model
    mock = bool(getattr(args, "mock", False))
    if mock and Path(args.out).resolve() == C.OUT_DEFAULT.resolve():
        raise SystemExit("--mock never writes into results/revision/finetune; give --out <scratch>")
    manifest = C.jload(out / "manifest.json")
    handle = get_calculator(model, device=args.device)
    ids, is_val = planned_split(model, args.systems, C.N_CONFIGS)
    labelled, problems, labels = {}, [], {}
    for system in args.systems:
        ent = manifest["entries"][f"{system}__{model}"]
        res, prob = collect_labels(out, Path(args.qe_root), model, system, ent, args.allow_partial,
                                   mock=mock, calc=handle.calc)
        problems += prob
        for i, a, e, f in res:
            labels[(system, i)] = (a, e, f)
    if problems and not (args.allow_partial or mock):
        for job, why in problems[:20]:
            print(f"[dataset] MISSING {job}: {why}")
        raise SystemExit(f"{len(problems)} of {len(ids)} jobs are not usable; run the queue to "
                         "completion (or --allow-partial for a smoke test)")
    # base-model energies and forces on every labelled configuration
    base = {}
    for key, (a, e, f) in labels.items():
        b = a.copy()
        b.calc = handle.calc
        base[key] = (float(b.get_potential_energy()), np.asarray(b.get_forces(), float))
    # The undisplaced reference cell of each system: PBE and base-model energy and the PBE residual
    # force, kept for diagnostics only.  It is NOT a training or validation configuration: it
    # coincides with the Q = 0 frame of every held-out C3a path of that model (the leakage rule).
    ref_cells = {}
    for system in args.systems:
        ent = manifest["entries"][f"{system}__{model}"]
        ra = C.read_frames(out / ent["ref_file"])[0]
        rec = {"n_atoms": len(ra), "pbe": None}
        po = Path(args.qe_root) / f"ft_{system}_{model}_ref" / "pw.out"
        if po.exists() and not mock:
            r = parse_pw_out(po)
            if r["job_done"] and r["converged"] and "energy_eV" in r:
                b = ra.copy()
                b.calc = handle.calc
                rec["pbe"] = {"energy_qe_eV": r["energy_eV"],
                              "max_abs_force_eV_A": float(np.abs(r["forces"]).max()),
                              "base_energy_eV": float(b.get_potential_energy()),
                              "base_minus_pbe_offset_eV": float(b.get_potential_energy()) - r["energy_eV"]}
        ref_cells[system] = rec
    # per-system offset from TRAINING configurations only
    offsets, resid = {}, {}
    idx_of = {key: k for k, key in enumerate(ids)}
    for system in args.systems:
        tr = [key for key in labels if key[0] == system and not is_val[idx_of[key]]]
        if not tr:
            raise SystemExit(f"{system}: no labelled training configurations")
        d = np.array([labels[k][1] - base[k][0] for k in tr])
        offsets[system] = float(d.mean())
        resid[system] = {"n_train": len(tr), "offset_eV": float(d.mean()),
                         "std_of_residual_eV": float(d.std()),
                         "std_per_atom_meV": float(d.std() / len(labels[tr[0]][0]) * 1000)}
    from ase import Atoms
    train, valid, raw = [], [], []
    for key in sorted(labels, key=lambda k: idx_of[k]):
        a, e, f = labels[key]
        system, i = key
        b = Atoms(numbers=a.get_atomic_numbers(), positions=a.get_positions(), cell=a.cell.array,
                  pbc=True)
        b.info.update({"system": system, "base_model": model, "config_id": f"{system}_c{i:02d}",
                       "T": float(a.info.get("T", 0.0)), "draw": int(a.info.get("draw", -1)),
                       "energy_qe_eV": float(e), "offset_eV": offsets[system],
                       "split": "valid" if is_val[idx_of[key]] else "train",
                       "REF_energy": float(e) - offsets[system], "mock": int(mock)})
        b.arrays["REF_forces"] = np.asarray(f, float)
        (valid if is_val[idx_of[key]] else train).append(b)
        raw.append(b)

    def rmse(keys):
        """Base model against the (offset) targets; the offset is fitted on train only, so the
        valid number is an honest starting error."""
        if not keys:
            return {"n": 0}
        fe = np.concatenate([(base[k][1] - labels[k][2]).ravel() for k in keys])
        ee = np.array([(base[k][0] - (labels[k][1] - offsets[k[0]])) / len(labels[k][0]) * 1000
                       for k in keys])
        return {"n": len(keys), "force_rmse_eV_A": float(np.sqrt(np.mean(fe ** 2))),
                "energy_rmse_meV_atom": float(np.sqrt(np.mean(ee ** 2)))}
    tr_keys = [k for k in labels if not is_val[idx_of[k]]]
    va_keys = [k for k in labels if is_val[idx_of[k]]]
    ddir = out / "dataset" / model
    C.write_frames(ddir / "train.extxyz", train)
    C.write_frames(ddir / "valid.extxyz", valid)
    C.write_frames(ddir / "labelled_all.extxyz", raw)
    info = {"model": model, "mock": mock, "provenance": C.provenance("dataset", args),
            "model_version": handle.version, "n_planned": len(ids),
            "n_labelled": len(labels), "n_train": len(train), "n_valid": len(valid),
            "unusable_jobs": [{"job": j, "why": w} for j, w in problems],
            "split": {"seed": [C.SPLIT_SEED, "crc32(model)"], "val_fraction": C.VAL_FRACTION,
                      "valid_ids": [f"{s}_c{i:02d}" for (s, i), v in zip(ids, is_val) if v],
                      "per_system_valid": {s: int(sum(v for (s2, _), v in zip(ids, is_val) if s2 == s))
                                           for s in args.systems}},
            "energy_reference": {"rule": "target = E_PBE - mean_train(E_PBE - E_base) per system",
                                 "offsets": resid},
            "base_vs_target": {"train": rmse(tr_keys), "valid": rmse(va_keys)},
            "reference_cells_not_in_training": ref_cells,
            "files": {n: {"sha256": C.sha256_file(ddir / n)} for n in
                      ("train.extxyz", "valid.extxyz", "labelled_all.extxyz")}}
    C.jdump(ddir / "dataset.json", info)
    print(f"[dataset] {model}: {len(train)} train + {len(valid)} valid of {len(ids)} planned "
          f"({len(problems)} unusable){' [MOCK LABELS]' if mock else ''}")
    for s, r in resid.items():
        print(f"   {s}: offset {r['offset_eV']:.3f} eV, residual std {r['std_per_atom_meV']:.2f} meV/atom (train)")
    print(f"   base model vs targets: train {info['base_vs_target']['train']}, valid {info['base_vs_target']['valid']}")
    return 0
