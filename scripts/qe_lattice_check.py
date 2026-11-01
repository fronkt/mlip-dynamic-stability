"""Is our PBE lattice converged? Cutoff and k-mesh check of the pbe-lattice phase-A vc-relax.

Second-round referee, comment 2. Our Quantum ESPRESSO PBE lattices (SSSP 1.3 efficiency, phase A of
scripts/dft_checks.py) lie 0.29 % (BaTiO3) and 0.75 % (KNbO3) below the Materials Project's PBE
lattices (VASP), which the five MLIPs reproduce to within 0.06 % (results/revision/r2/
external_lattices.json). At a well-depth sensitivity d ln(depth)/d ln a near 150 that gap alone is a
factor of 1.5 to 3 in well depth, so before the text interprets it, this checks the obvious
numerical cause: an under-converged basis or k-mesh in the vc-relax.

Each variant is phase A's own tracked input (results/revision/dft_checks/qe/pl_<sys>_vcrelax/pw.in)
with exactly one change, so everything else (pseudopotentials, smearing, start cell, thresholds) is
production:
  e1.5, e2.0  ecutwfc and ecutrho x1.5 and x2.0 (the dual stays 8)
  kfine       11x11x11 instead of 7x7x7 (k-spacing 0.15 instead of 0.25 1/A, 2pi included)
It does not test the pseudopotentials themselves (the SSSP precision set was not used); a gap that
survives these variants is therefore a pseudopotential/code difference, and the text says so.

  python scripts/qe_lattice_check.py write             # inputs under results/revision/r2_qe_lattice/qe/
  (box) run each <dir>/pw.in with pw.x                  # scripts/box/as_run/r2_centroid_2026-10-22.sh
  python scripts/qe_lattice_check.py analyze           # -> results/revision/r2_qe_lattice/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results" / "revision" / "dft_checks" / "qe"
OUT = REPO / "results" / "revision" / "r2_qe_lattice"
EXT = REPO / "results" / "revision" / "r2" / "external_lattices.json"
SYSTEMS = ("batio3_cubic", "knbo3_cubic")
VARIANTS = {"e1.5": {"ecut_scale": 1.5}, "e2.0": {"ecut_scale": 2.0}, "kfine": {"kmesh": (11, 11, 11)}}


def _sub_one(text: str, pattern: str, repl: str) -> str:
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != 1:
        raise SystemExit(f"expected exactly one match for {pattern!r}, found {n}")
    return new


def variant_text(base: str, job: str, spec: dict) -> str:
    t = _sub_one(base, r"^(\s*! job )\S+(, written by .*)$", rf"\g<1>{job}\2 (variant of phase A)")
    if "ecut_scale" in spec:
        s = spec["ecut_scale"]
        w = float(re.search(r"^\s*ecutwfc\s*=\s*([\d.]+)", t, flags=re.M).group(1))
        r = float(re.search(r"^\s*ecutrho\s*=\s*([\d.]+)", t, flags=re.M).group(1))
        t = _sub_one(t, r"^(\s*ecutwfc\s*=\s*)[\d.]+", rf"\g<1>{w * s:.1f}")
        t = _sub_one(t, r"^(\s*ecutrho\s*=\s*)[\d.]+", rf"\g<1>{max(8.0 * w * s, r * s):.1f}")
    if "kmesh" in spec:
        k = " ".join(str(x) for x in spec["kmesh"])
        t = _sub_one(t, r"^(K_POINTS automatic\s*\n)\s*\d+\s+\d+\s+\d+\s+(\d+\s+\d+\s+\d+)",
                     rf"\g<1>{k} \2")
    return t


def write(_args) -> int:
    for s in SYSTEMS:
        src = BASE / f"pl_{s}_vcrelax" / "pw.in"
        base = src.read_text(encoding="utf-8")
        for v, spec in VARIANTS.items():
            job = f"r2l_{s}_{v}"
            d = OUT / "qe" / job
            d.mkdir(parents=True, exist_ok=True)
            txt = variant_text(base, job, spec)
            (d / "pw.in").write_text(txt, encoding="utf-8", newline="\n")
            meta = {"job": job, "system": s, "variant": v, "spec": spec,
                    "base_input": src.relative_to(REPO).as_posix(),
                    "base_sha256": hashlib.sha256(base.replace("\r\n", "\n").encode()).hexdigest()}
            (d / "job.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {d.relative_to(REPO).as_posix()}/pw.in")
    return 0


def _a_from(r: dict):
    v = r.get("volume_A3")
    return None if v is None else float(v) ** (1.0 / 3.0)


def analyze(_args) -> int:
    import dft_checks as dc
    ext = json.load(open(EXT, encoding="utf-8"))["materials_project_pbe"]["values_A"]
    res = {"script": "scripts/qe_lattice_check.py", "variants": VARIANTS, "systems": {}}
    for s in SYSTEMS:
        prod = dc.parse_vcrelax(BASE / f"pl_{s}_vcrelax" / "pw.out")
        a0 = _a_from(prod)
        row = {"production": {"a_A": a0, "pressure_kbar": prod.get("final_scf_pressure_kbar"),
                              "bfgs_converged": prod.get("bfgs_converged")},
               "materials_project_pbe_a_A": ext[s]["a_A"], "variants": {}}
        for v in VARIANTS:
            out = OUT / "qe" / f"r2l_{s}_{v}" / "pw.out"
            if not out.exists():
                row["variants"][v] = {"status": "missing"}
                continue
            r = dc.parse_vcrelax(out)
            a = _a_from(r)
            row["variants"][v] = {
                "status": "ok" if (r.get("bfgs_converged") and a) else "not_converged",
                "a_A": a, "pressure_kbar": r.get("final_scf_pressure_kbar"),
                "minus_production_pct": None if (a is None or a0 is None) else 100 * (a / a0 - 1),
                "minus_mp_pct": None if a is None else 100 * (a / ext[s]["a_A"] - 1)}
        row["production_minus_mp_pct"] = None if a0 is None else 100 * (a0 / ext[s]["a_A"] - 1)
        ok = [x for x in row["variants"].values() if x.get("status") == "ok"]
        row["max_abs_change_vs_production_pct"] = (max(abs(x["minus_production_pct"]) for x in ok)
                                                   if ok else None)
        res["systems"][s] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for s, row in res["systems"].items():
        print(f"{s}: production a {row['production']['a_A']:.5f} ({row['production_minus_mp_pct']:+.3f} % vs MP)"
              + "".join(f"; {v} " + (f"{x['a_A']:.5f} ({x['minus_production_pct']:+.3f} %)"
                                     if x.get("status") == "ok" else x.get("status", "?"))
                        for v, x in row["variants"].items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["write", "analyze"])
    a = ap.parse_args(argv)
    return write(a) if a.cmd == "write" else analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
