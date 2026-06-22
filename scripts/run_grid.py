"""Launch a grid of units for ONE model (run inside that model's env).

Iterates the curated systems and a temperature ladder, calling the resumable unit runner.
Already-recorded units are skipped, so this is safe to re-run after an interruption.

Examples:
    python scripts/run_grid.py --model mace_mp0 --layer harmonic
    python scripts/run_grid.py --model mace_mp0 --layer finite_t --temps 100 300 600 900
"""
from __future__ import annotations

import argparse

from mlip_dynstab.cli import run_unit
from mlip_dynstab.systems import load_specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", choices=["harmonic", "finite_t"], required=True)
    p.add_argument("--method", default=None,
                   help="finite_t method: softmode (default) | rattled | hiphive | md_distort")
    p.add_argument("--temps", type=float, nargs="+", default=[100, 300, 600, 900])
    p.add_argument("--device", default="cuda")
    p.add_argument("--only-class", default=None,
                   help="restrict to one klass (e.g. perovskite, bcc-metal, control)")
    args = p.parse_args()

    specs = load_specs()
    if args.only_class:
        specs = [s for s in specs if s.klass == args.only_class]

    for s in specs:
        try:
            if args.layer == "harmonic":
                run_unit(s.id, args.model, "harmonic", 0.0, args.device)
            else:
                # softmode (1D quantum SCHA) is the unified finite-T method for all classes;
                # superionic agi_bcc also has a soft framework phonon so it fits the same path
                # (the Ag-sublattice disordering is a documented caveat, not separately modelled).
                method = args.method or "softmode"
                for T in args.temps:
                    run_unit(s.id, args.model, method, T, args.device)
        except Exception as e:  # one bad unit must not kill the grid
            print(f"[error] {s.id}/{args.model}/{args.layer}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
