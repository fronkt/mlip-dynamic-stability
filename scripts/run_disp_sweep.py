"""Re-measure the harmonic layer on a grid of finite-displacement amplitudes.

Closes the one gap this project declares against itself. ESI §S1.2 bounds the harmonic
estimator's reproducibility using the v1/v2 paired replicate, but that pair shares a code path,
a seedless algorithm and a displacement amplitude, so it probes environment and library
nondeterminism only. It does not probe sensitivity to the amplitude itself, which is the axis a
reader familiar with kappa_SRME's estimator noise will ask about, and which bears on the
reordering in §3.2. This script supplies that axis.

Safety, which matters more here than speed. A non-production amplitude is recorded under the
method name `harmonic_dispsweep`, never `harmonic` (see `cli.run_unit`). Sweep rows carry the
same method_version as production rows, so had they shared a method name they would have passed
straight through `analysis.canonical()` and silently inflated the denominator of every harmonic
rate in the paper. The production amplitude 0.01 A is deliberately absent from the unit hash, so
every deposited hash is unchanged and the existing rows serve as the sweep's own 0.01 A arm.

Resumable: units already in the ledger are skipped. Run from the repo root.

    python scripts/run_disp_sweep.py --dry-run            # cost it first
    python scripts/run_disp_sweep.py --models chgnet mace_mp0 --device cpu
    python scripts/run_disp_sweep.py --device cuda        # all five, on a GPU box
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd                                    # noqa: E402

from mlip_dynstab import SUPPORTED_MODELS, METHOD_VERSION   # noqa: E402
from mlip_dynstab import ledger                             # noqa: E402
from mlip_dynstab.cli import run_unit, PRODUCTION_DISP_ANG  # noqa: E402
from mlip_dynstab.systems import load_specs                 # noqa: E402
from mlip_dynstab.calculators import get_calculator         # noqa: E402

# The production value is the sweep's own centre point and is already deposited; only the
# departures need running.
SWEEP_DISP_ANG = (0.005, 0.02, 0.03)
DEFAULT_MODELS = ("mace_mp0", "chgnet", "mattersim", "sevennet0", "orb_v2")


def scored_systems() -> list[str]:
    """The 19 systems the paper scores: everything except the borderline label."""
    return sorted(s.id for s in load_specs() if not s.borderline)


def planned_units(systems, models, disps):
    for system in systems:
        for model in models:
            for disp in disps:
                yield system, model, disp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                    choices=list(SUPPORTED_MODELS))
    ap.add_argument("--disps", type=float, nargs="+", default=list(SWEEP_DISP_ANG))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--systems", nargs="+", default=None,
                    help="default: the 19 scored systems")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would run, and how much is already done")
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args()

    systems = args.systems or scored_systems()
    for d in args.disps:
        if abs(d - PRODUCTION_DISP_ANG) < 1e-12:
            raise SystemExit(
                f"{PRODUCTION_DISP_ANG} A is the production amplitude and is already deposited "
                "under method 'harmonic'. Re-running it here would record it as a sweep unit "
                "and double-count it. Drop it from --disps.")

    plan = list(planned_units(systems, args.models, args.disps))
    print(f"{len(systems)} systems x {len(args.models)} models x {len(args.disps)} amplitudes "
          f"= {len(plan)} units")
    print(f"amplitudes: {args.disps} A   (production {PRODUCTION_DISP_ANG} A already deposited)")

    if args.dry_run:
        # Which are already done? Needs the model version, so resolve calculators lazily and
        # only report coverage per model when it can be loaded.
        lp = Path(args.ledger) if args.ledger else ledger._DEFAULT
        df = ledger.load(lp)
        have = set(df["uhash"]) if not df.empty else set()
        done = 0
        for model in args.models:
            try:
                ver = get_calculator(model, device="cpu").version
            except Exception as exc:
                print(f"  {model:12s} calculator unavailable here ({type(exc).__name__})")
                continue
            n = 0
            for system in systems:
                for disp in args.disps:
                    settings = {"supercell": [2, 2, 2],
                                "mv": METHOD_VERSION["harmonic_dispsweep"],
                                "disp": float(disp)}
                    uh = ledger.unit_hash(system, model, ver, 0.0,
                                          "harmonic_dispsweep", settings)
                    n += uh in have
            done += n
            print(f"  {model:12s} {n}/{len(systems)*len(args.disps)} already in ledger")
        print(f"total already done: {done}/{len(plan)}")
        return

    t0 = time.time()
    ok = skipped = failed = 0
    for i, (system, model, disp) in enumerate(plan, 1):
        try:
            out = run_unit(system, model, "harmonic", 0.0, device=args.device,
                           ledger_path=args.ledger, disp=disp)
            if out.get("skipped"):
                skipped += 1
            else:
                ok += 1
        except Exception as exc:
            # Never abort the sweep on one unit; a missing model or an unrelaxable cell should
            # cost that unit and nothing else.
            failed += 1
            print(f"[FAIL] {system}/{model}/disp={disp}: {type(exc).__name__}: {exc}")
        if i % 25 == 0 or i == len(plan):
            print(f"  ... {i}/{len(plan)}  ok={ok} skipped={skipped} failed={failed}  "
                  f"{time.time()-t0:.0f}s")

    print(f"\ndone: {ok} new, {skipped} already present, {failed} failed, "
          f"{time.time()-t0:.0f}s total")
    if failed:
        print("Re-run to retry the failures; completed units are skipped.")


if __name__ == "__main__":
    main()
