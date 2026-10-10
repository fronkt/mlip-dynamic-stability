"""POST HOC (not pre-registered) diagnostic for the E5 fine-tuning trial (Referee 2).

Question: the fine-tuned MACE-MP-0 models reproduce PBE's well depths along the held-out C3a
coordinates (P1), which sit at the BASE model's relaxed lattice, yet the full screen (P2), which
relaxes the cell with the model under test, keeps the BaTiO3 and KNbO3 300 K mis-calls. Does the
fine-tuned model's own relaxed lattice explain the difference?

For each system and model (base MACE-MP-0 medium and the three fine-tuned replicates) this
relaxes the cubic cell (the production relaxation, ``harmonic._relax``), records the lattice
constant and the pressure each model reports at the base model's lattice, and re-runs the
production 300 K screen (2x2x2, up to 24 modes) without further relaxation at (a) the model's own
relaxed lattice and (b) the base model's relaxed lattice. It runs on CPU (float64) with the
archived weights; check them against the run's models/WEIGHTS.sha256 first. The screen at (a)
should reproduce the deposited P2 300 K call; the output records whether it does.

    python scripts/finetune_lattice_diag.py --weights <dir holding mace_mp0/seed{0,1,2}/ft_mace_mp0_seed{s}.model> \
        --run results/revision/finetune_mace30 [--systems batio3_cubic,knbo3_cubic]

Writes <run>/lattice_diag.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run", default="results/revision/finetune_mace30")
    ap.add_argument("--systems", default="batio3_cubic,knbo3_cubic")
    ap.add_argument("--T", type=float, default=300.0)
    args = ap.parse_args()

    from mace.calculators import MACECalculator, mace_mp
    from mlip_dynstab.finite_t import compute_finite_t_softmode
    from mlip_dynstab.harmonic import _relax
    from mlip_dynstab.systems import build_atoms, get_spec

    run = REPO / args.run
    sha = {}
    for line in (run / "models" / "WEIGHTS.sha256").read_text().splitlines():
        h, p = line.split(None, 1)
        sha[p.strip().lstrip("./")] = h
    calcs = {"base": mace_mp(model="medium", device="cpu", default_dtype="float64")}
    weights = {}
    for s in (0, 1, 2):
        rel = f"mace_mp0/seed{s}/ft_mace_mp0_seed{s}.model"
        p = Path(args.weights) / rel
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != sha.get(rel):
            raise SystemExit(f"{p}: sha256 {h} does not match {run}/models/WEIGHTS.sha256")
        weights[f"seed{s}"] = h
        calcs[f"seed{s}"] = MACECalculator(model_paths=str(p), device="cpu", default_dtype="float64")

    p2 = json.loads((run / "summary.json").read_text(encoding="utf-8"))["P2"]["models"]["mace_mp0"]
    out = {"note": "POST HOC, not pre-registered (scripts/finetune_lattice_diag.py)",
           "run": args.run, "T_K": args.T, "weights_sha256": weights, "systems": {}}
    tmp = Path(tempfile.mkdtemp(prefix="ftlat_"))
    for system in args.systems.split(","):
        a0 = build_atoms(get_spec(system))
        relaxed, rec = {}, {}
        for tag, c in calcs.items():
            at = a0.copy()
            at.calc = c
            at = _relax(at, fmax=1e-3)
            relaxed[tag] = at
            rec[tag] = {"a_relaxed_A": float(np.cbrt(at.get_volume()))}
            print(system, tag, "a = %.4f A" % rec[tag]["a_relaxed_A"], flush=True)
        base_at = relaxed["base"]
        for tag, c in calcs.items():
            x = base_at.copy()
            x.calc = c
            rec[tag]["pressure_at_base_lattice_GPa"] = float(-x.get_stress(voigt=True)[:3].mean() * 160.21766)
            for label, src in (("own_lattice", relaxed[tag]), ("base_lattice", base_at)):
                y = src.copy()
                y.calc = c
                res = compute_finite_t_softmode(y, c, args.T, supercell=(2, 2, 2), max_modes=24, relax=False,
                                                cache_path=str(tmp / f"{system}_{tag}_{label}.json"))
                r = res.as_row()
                rec[tag][label] = {"pred_stable": bool(res.dynamically_stable),
                                   "min_eff_thz": float(r["min_eff_freq_thz"]),
                                   "decide_q": r.get("ft_decide_q"),
                                   "well_depth_meV": r.get("ft_well_depth_meV")}
                print(system, tag, label, rec[tag][label], flush=True)
            if tag != "base":
                dep = p2[f"{system}@{int(args.T)}"]["replicates"][tag]
                rec[tag]["own_lattice_reproduces_deposited_call"] = (
                    rec[tag]["own_lattice"]["pred_stable"] == bool(dep["pred_stable"]))
        out["systems"][system] = rec
    (run / "lattice_diag.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", run / "lattice_diag.json")


if __name__ == "__main__":
    main()
