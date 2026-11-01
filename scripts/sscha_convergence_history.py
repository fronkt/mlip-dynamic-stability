"""Population-by-population history of the production and the converged SSCHA recipes.

Second-round referee, comment 3: show how the gradient evolved across populations for a
representative unit, production against converged, so the step-cap failure reads as a lesson in
validating a stochastic minimiser rather than as a re-run. Zero compute: it re-reads the
per-step histories the runs already recorded.

  production  results/revision/sscha_seeds/<tag>.json, seed 0 (256 configurations per
              population, max_ka = 20 cumulative over populations, meaningful_factor 1e-4
              against python-sscha 1.6.1's constant placeholder gradient error)
  converged   results/revision/sscha_converged_grid/<tag>_startA.json (1000 configurations,
              400 steps per population, meaningful_factor 0.2 against the real serial error,
              convergence refused where KL/N < 0.7)

Units: BaTiO3/MACE-MP-0/100 K (2x2x2 in both) and bcc-Zr/MatterSim/300 K (3x3x3 in both).
The production runs recorded the placeholder error only, so their gradient is given as the raw
norm gc; the converged runs also give gc divided by the real error.

Writes results/revision/r2/sscha_history.json (one row per population and the per-step traces
the ESI figure plots). Usage: python scripts/sscha_convergence_history.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "revision" / "r2" / "sscha_history.json"
UNITS = {"batio3_mace_100K": "batio3_cubic_mace_mp0_100K_sc222",
         "zr_mattersim_300K": "zr_bcc_mattersim_300K_sc333"}
PROD_DIR = REPO / "results" / "revision" / "sscha_seeds"
CONV_DIR = REPO / "results" / "revision" / "sscha_converged_grid"


def _hash(path: Path) -> dict:
    """sha256 and git blob sha1 as git stores the file (CRLF -> LF), like verify_claims."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    return {"path": path.relative_to(REPO).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": blob}


def _pop_rows(pops: list, real_error: bool) -> list:
    rows = []
    for p in pops:
        r = {"pop": p["pop"], "n_steps": p["n_steps"], "n_steps_kept": p["n_steps_kept"],
             "stop_reason": p["stop_reason"], "gc_last": p["gc_last"],
             "kl_ratio_last": p["kl_ratio_last"],
             "aux_min_start_thz": p["start_aux_min_nonac_thz"],
             "aux_min_end_thz": p["end_aux_min_nonac_thz"],
             "rel_change_vs_prev_end": p["rel_change_vs_prev_end"],
             "fe_last_Ry": p["fe_last_Ry"], "fe_err_last_Ry": p["fe_err_last_Ry"]}
        if real_error:
            r.update(gc_err_real_last=p["gc_err_real_last"],
                     gc_over_err_real_last=p["gc_over_err_real_last"],
                     guarded_last=p["guarded_last"], converged_at_end=p["converged_at_end"])
        else:
            r.update(gc_err_placeholder=p["gc_err_last"])
        rows.append(r)
    return rows


def _trace(steps: list, real_error: bool) -> list:
    keys = ["pop", "n_fe_cumulative", "gc", "kl_ratio", "aux_min_nonac_thz"]
    if real_error:
        keys += ["gc_over_err_real", "convergence_guarded"]
    return [{k: s.get(k) for k in keys} for s in steps]


def unit_history(tag: str) -> dict:
    pf, cf = PROD_DIR / f"{tag}.json", CONV_DIR / f"{tag}_startA.json"
    prod = json.load(open(pf, encoding="utf-8"))["seeds"]["0"]
    conv = json.load(open(cf, encoding="utf-8"))["run"]
    pr, cr = prod["relax"], conv["relax"]
    return {
        "sources": [_hash(pf), _hash(cf)],
        "production": {
            "recipe": "256 configs/pop, <=8 pops, max_ka=20 cumulative, meaningful_factor 1e-4 "
                      "x placeholder error",
            "converged": pr["converged"], "n_populations": pr["n_populations"],
            "n_steps_total": pr["n_steps_total"], "n_steps_kept_total": pr["n_steps_kept_total"],
            "placeholder_error": pr["populations"][0]["gc_err_last"],
            "stop_threshold": 1e-4 * pr["populations"][0]["gc_err_last"],
            "start_min_thz": prod["start_post_fpd"]["min_nonac_thz"],
            "hessian_min_thz": prod["hessian"]["min_nonac_thz"],
            "populations": _pop_rows(pr["populations"], real_error=False),
            "steps": _trace(pr["steps"], real_error=False)},
        "converged": {
            "recipe": "1000 configs/pop, <=30 pops, 400 steps/pop, meaningful_factor 0.2 x real "
                      "serial error, refused where KL/N < 0.7",
            "converged": cr["converged"], "stop_reason": cr["stop_reason"],
            "n_populations": cr["n_populations"], "n_steps_total": cr["n_steps_total"],
            "start_min_thz": conv["start"]["min_nonac_thz"],
            "hessian_min_thz": conv["hessian"]["min_nonac_thz"],
            "hessian_boot_sd_thz": conv["bootstrap"]["std_thz"],
            "fresh_check": {k: conv["fresh_gradient_check"].get(k)
                            for k in ("n_configs", "gc", "gc_err", "gc_over_err",
                                      "consistent_with_minimum")},
            "populations": _pop_rows(cr["populations"], real_error=True),
            "steps": _trace(cr["steps"], real_error=True)},
    }


def main() -> int:
    res = {"script": "scripts/sscha_convergence_history.py",
           "units": {k: unit_history(t) | {"unit_tag": t} for k, t in UNITS.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)
    for k, u in res["units"].items():
        p, c = u["production"], u["converged"]
        print(f"{k}: production {p['n_populations']} pops, {p['n_steps_kept_total']}/"
              f"{p['n_steps_total']} steps kept, Hessian {p['hessian_min_thz']:+.3f} "
              f"(start {p['start_min_thz']:+.3f}); converged {c['n_populations']} pops, "
              f"{c['n_steps_total']} steps, Hessian {c['hessian_min_thz']:+.3f} "
              f"+/- {c['hessian_boot_sd_thz']:.3f}")
        for r in c["populations"]:
            print(f"   conv pop {r['pop']}: {r['n_steps']} steps, {r['stop_reason']}, gc/err "
                  f"{r['gc_over_err_real_last']:.3g}, KL {r['kl_ratio_last']:.2f}, guarded "
                  f"{r['guarded_last']}")
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
