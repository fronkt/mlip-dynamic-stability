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
from . import METHOD_VERSION, SOFTMODE_EQMAP_VERSION
from .calculators import get_calculator
from .systems import get_spec, build_atoms


def _finite_t_gt(spec, temperature_K: float) -> bool:
    """Temperature-resolved finite-T dynamical-stability ground truth.

    A system with a known transition temperature is the high-symmetry phase (stable) only at
    or above it; below it the soft mode has condensed (unstable). Systems with no transition
    (controls, quantum paraelectrics) take their constant ``finite_T_stable`` label.
    """
    Tc = spec.transition_T_K
    if Tc is None:
        return spec.finite_T_stable
    return float(temperature_K) >= float(Tc)


PRODUCTION_DISP_ANG = 0.01   # the displacement every deposited harmonic row was measured at


def run_unit(system: str, model: str, method: str, temperature_K: float = 0.0,
             device: str = "cuda", supercell=(2, 2, 2), force: bool = False,
             ledger_path=None, max_modes: int = 24,
             disp: float = PRODUCTION_DISP_ANG) -> dict:
    spec = get_spec(system)
    # Finite-T (hiPhive) needs a larger supercell so the pair cutoff can exceed nearest
    # neighbors while staying < L/2; harmonic finite-displacement is fine at 2x2x2.
    if method in ("hiphive", "rattled") and tuple(supercell) == (2, 2, 2):
        supercell = (3, 3, 3)
    # softmode on bcc metals: the omega-phase instability sits at q=2/3<111>, which is only
    # exactly commensurate with a 6x6x6 force-constant cell (which also captures N=1/2). bcc is
    # a 1-atom primitive with cubic symmetry, so 6x6x6 still needs only ~1 displacement.
    if method == "softmode" and spec.klass == "bcc-metal" and tuple(supercell) == (2, 2, 2):
        supercell = (6, 6, 6)
    # The unit hash must change whenever the ALGORITHM changes, not just its inputs. Carrying
    # only the supercell here is what let the v1 softmode rows survive the FC-commensurate
    # q-search fix (e592e86): the hash was unchanged, so `has_unit` skipped every stale unit as
    # "already present" and the deposited ledger kept data the deposited code cannot reproduce.
    settings = {"supercell": list(supercell), "mv": METHOD_VERSION.get(method, 1)}
    # Any parameter that can change the RESULT belongs in the hash, not just in the code. The
    # mode cap bounds how many imaginary modes the screen examines, so a unit computed under a
    # tighter cap is not the same measurement as one computed under a looser one.
    if method == "softmode":
        settings["max_modes"] = int(max_modes)

    # Displacement amplitude. Two hazards here, and the fix has to dodge both.
    #
    # (1) `disp` was not in the hash at all, so a re-run at a different amplitude would be
    #     skipped by has_unit() as "already present" -- the same failure that let the stale
    #     softmode rows survive the q-search fix.
    # (2) Adding it unconditionally would change the hash of every EXISTING harmonic row,
    #     detaching the deposited ledger from the code that produced it.
    #
    # So the production amplitude is left out of the hash, exactly as before, and only a
    # departure from it is recorded. A swept unit is also retagged to its own method name
    # rather than sharing "harmonic": sweep rows carry the same method_version as production,
    # so if they shared the method they would pass straight through canonical() and inflate
    # the denominator of every harmonic rate in the paper.
    swept = abs(float(disp) - PRODUCTION_DISP_ANG) > 1e-12
    if swept:
        if method != "harmonic":
            raise SystemExit("--disp is only meaningful for --method harmonic")
        method = "harmonic_dispsweep"
        settings["disp"] = float(disp)
        settings["mv"] = METHOD_VERSION.get(method, 1)

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
                model_version=handle.version, method=method,
                method_version=settings["mv"], temperature_K=temperature_K,
                gt_harmonic_stable=spec.harmonic_stable, gt_finite_T_stable=spec.finite_T_stable,
                transition_T_K=spec.transition_T_K)

    if method in ("harmonic", "harmonic_dispsweep"):
        from .harmonic import compute_harmonic
        # identical code path; only the displacement amplitude differs
        res = compute_harmonic(atoms, handle.calc, supercell=supercell, disp=float(disp))
        base.update(res.as_row())
        base["disp_ang"] = float(disp)
        base["gt_stable"] = spec.harmonic_stable
    elif method == "hiphive":
        from .finite_t import compute_finite_t_hiphive
        cache = f"results/cache/md_{method}_{system}_{model}_{int(temperature_K)}_sc{''.join(map(str,supercell))}.extxyz"
        res = compute_finite_t_hiphive(atoms, handle.calc, temperature_K,
                                       supercell=supercell, cache_path=cache)
        base.update(res.as_row())
        base["gt_stable"] = _finite_t_gt(spec, temperature_K)
    elif method == "rattled":
        from .finite_t import compute_finite_t_rattled
        cache = f"results/cache/md_{method}_{system}_{model}_{int(temperature_K)}_sc{''.join(map(str,supercell))}.extxyz"
        res = compute_finite_t_rattled(atoms, handle.calc, temperature_K,
                                       supercell=supercell, cache_path=cache)
        base.update(res.as_row())
        base["gt_stable"] = _finite_t_gt(spec, temperature_K)
    elif method == "softmode":
        from .finite_t import compute_finite_t_softmode
        # The E(Q) double-well map is temperature-independent, so the cache key omits T;
        # every extra temperature then reuses it for a sub-second 1D quantum solve.
        # The cache key carries the E(Q)-MAP version (not METHOD_VERSION): the cache holds the
        # map layer only -- relax, FCs, q-search, modulation, well sampling, fit -- so it stays
        # valid when the solve layer changes (v4 changed only the reported scalar). A map-layer
        # change MUST bump SOFTMODE_EQMAP_VERSION or a stale map is silently reused.
        cache = (f"results/cache/softmode_v{SOFTMODE_EQMAP_VERSION}m{max_modes}_{system}"
                 f"_{model}_sc{''.join(map(str,supercell))}.json")
        res = compute_finite_t_softmode(atoms, handle.calc, temperature_K,
                                        supercell=supercell, max_modes=max_modes,
                                        cache_path=cache)
        base.update(res.as_row())
        base["gt_stable"] = _finite_t_gt(spec, temperature_K)
    elif method == "tdep":
        from .finite_t import compute_finite_t_tdep
        res = compute_finite_t_tdep(atoms, handle.calc, temperature_K, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = _finite_t_gt(spec, temperature_K)
    elif method == "md_distort":
        from .finite_t import md_symmetry_breaking
        res = md_symmetry_breaking(atoms, handle.calc, temperature_K, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = _finite_t_gt(spec, temperature_K)
    elif method == "sscha":
        from .finite_t import compute_finite_t_sscha
        res = compute_finite_t_sscha(atoms, handle.calc, temperature_K, supercell=supercell)
        base.update(res.as_row())
        base["gt_stable"] = _finite_t_gt(spec, temperature_K)
    else:
        raise ValueError(f"unknown method '{method}'")

    base["wall_s"] = round(time.time() - t0, 2)
    base["pred_stable"] = bool(base["dynamically_stable"])
    base["false_stable"] = bool(base["pred_stable"] and not base["gt_stable"])
    base["false_unstable"] = bool((not base["pred_stable"]) and base["gt_stable"])
    ledger.record(base, lp, overwrite=force)
    print(f"[done] {uhash} {system}/{model}/{method}/T={temperature_K} "
          f"min_freq={base.get('min_freq_thz', base.get('min_eff_freq_thz')):.3f} THz "
          f"pred_stable={base['pred_stable']} gt={base['gt_stable']} ({base['wall_s']}s)")
    return base


def main(argv=None):
    p = argparse.ArgumentParser(description="Run one MLIP dynamic-stability unit.")
    p.add_argument("--system", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="harmonic",
                   choices=["harmonic", "hiphive", "rattled", "softmode", "tdep",
                            "md_distort", "sscha"])
    p.add_argument("--T", type=float, default=0.0, dest="temperature_K")
    p.add_argument("--device", default="cuda")
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--disp", type=float, default=PRODUCTION_DISP_ANG,
                   help="finite-displacement amplitude in Angstrom (default 0.01, the "
                        "production value). Any other value is recorded as method "
                        "'harmonic_dispsweep' so it cannot contaminate the harmonic rates.")
    p.add_argument("--force", action="store_true", help="recompute even if in ledger")
    args = p.parse_args(argv)
    run_unit(args.system, args.model, args.method, args.temperature_K, args.device,
             tuple(args.supercell), args.force, disp=args.disp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
