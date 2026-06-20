"""Resumable per-unit runner.

A *unit* is one (system, model, method, temperature) tuple. The CLI:
  1. resolves the system structure and the model calculator,
  2. computes the requested observable (harmonic or finite-T),
  3. records a hashed row in the ledger (skipping if already present).

Run inside the target model's env (so `get_calculator` can import that backend). Example:
    python -m mlip_dynstab.cli --system srtio3_cubic --model mace_mp0 --method tdep --T 300
    python -m mlip_dynstab.cli --system srtio3_cubic --model mace_mp0 --method harmonic
"""
from __future__ import annotations

import argparse
import sys
import time

from . import ledger
from .calculators import get_calculator
from .systems import get_spec, build_atoms


def run_unit(system: str, model: str, method: str, temperature_K: float = 0.0,
             device: str = "cuda", supercell=(2, 2, 2), force: bool = False,
             ledger_path=None) -> dict:
    spec = get_spec(system)
    settings = {"supercell": list(supercell)}

    # We need the model version for the hash, so load the calculator first.
    handle = get_calculator(model, device=device)
    uhash = ledger.unit_hash(system, model, handle.version, temperature_K, method, settings)
    lp = ledger_path or ledger._DEFAULT
    if not force and ledger.has_unit(uhash, lp):
        print(f"[skip] {uhash} already in ledger ({system}/{model}/{method}/T={temperature_K})")
        return {"uhash": uhash, "skipped": True}

    atoms = build_atoms(spec)
    t0 = time.time()
    base = dict(uhash=uhash, system=system, klass=spec.klass, model=model,
                model_version=handle.version, method=method, temperature_K=temperature_K,
                gt_harmonic_stable=spec.harmonic_stable, gt_finite_T_stable=spec.finite_T_stable,
                transition_T_K=spec.transition_T_K)

    if method == "harmonic":
        from .harmonic import compute_harmonic
        res = compute_harmonic(atoms, handle.calc, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = spec.harmonic_stable
    elif method == "tdep":
        from .finite_t import compute_finite_t_tdep
        res = compute_finite_t_tdep(atoms, handle.calc, temperature_K, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = spec.finite_T_stable
    elif method == "md_distort":
        from .finite_t import md_symmetry_breaking
        res = md_symmetry_breaking(atoms, handle.calc, temperature_K, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = spec.finite_T_stable
    elif method == "sscha":
        from .finite_t import compute_finite_t_sscha
        res = compute_finite_t_sscha(atoms, handle.calc, temperature_K, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = spec.finite_T_stable
    else:
        raise ValueError(f"unknown method '{method}'")

    base["wall_s"] = round(time.time() - t0, 2)
    base["pred_stable"] = bool(base["dynamically_stable"])
    base["false_stable"] = bool(base["pred_stable"] and not base["gt_stable"])
    base["false_unstable"] = bool((not base["pred_stable"]) and base["gt_stable"])
    ledger.record(base, lp)
    print(f"[done] {uhash} {system}/{model}/{method}/T={temperature_K} "
          f"min_freq={base.get('min_freq_thz', base.get('min_eff_freq_thz')):.3f} THz "
          f"pred_stable={base['pred_stable']} gt={base['gt_stable']} ({base['wall_s']}s)")
    return base


def main(argv=None):
    p = argparse.ArgumentParser(description="Run one MLIP dynamic-stability unit.")
    p.add_argument("--system", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="harmonic",
                   choices=["harmonic", "tdep", "md_distort", "sscha"])
    p.add_argument("--T", type=float, default=0.0, dest="temperature_K")
    p.add_argument("--device", default="cuda")
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--force", action="store_true", help="recompute even if in ledger")
    args = p.parse_args(argv)
    run_unit(args.system, args.model, args.method, args.temperature_K, args.device,
             tuple(args.supercell), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
