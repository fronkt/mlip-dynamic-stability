"""Re-measure the deposited SSCHA grid in pinned environments (METHOD_VERSION sscha=2).

Re-runs exactly the unit set of the v1 grid -- (system, model, T) tuples extracted from the
deposited ledger into results/sscha_units_v1.csv -- so the v2 measurement is comparable
one-to-one. The v1 rows carried two defects: an acoustic-mask inversion (corrected in-place
from stored spectra, audit D1) and an unrecorded environment. v2 rows are measured, not
derived, and carry method_version=2 in both the unit hash and the row.

Run inside ONE model's env:
    python scripts/run_sscha_v2.py --model mace_mp0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlip_dynstab.cli import run_unit

UNITS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "sscha_units_v1.csv")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(UNITS_CSV, newline="") as fh:
        units = [r for r in csv.DictReader(fh) if r["model"] == args.model]
    print(f"[sscha-v2] {args.model}: {len(units)} units")

    for i, u in enumerate(units):
        try:
            run_unit(u["system"], args.model, "sscha", float(u["temperature_K"]), args.device)
        except Exception as e:  # one bad unit must not kill the grid
            print(f"[error] {u['system']}/{args.model}/sscha/T={u['temperature_K']}: "
                  f"{type(e).__name__}: {e}")
        print(f"[sscha-v2] {args.model} progress {i + 1}/{len(units)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
