"""E3 replicate summary: the quantities tasks/preregistration-repeats-2026-10-03.md reports.

Per unit of scripts/box/as_run/e3_units_2026-10-08.tsv: the lowest Hessian frequency of
  A        the grid's start-A run       results/revision/sscha_converged_grid/<tag>_startA.json
  B        start B at 0.3 THz           results/revision/sscha_converged_grid_startB/<tag>_startB.json
           or, where that failed on the complex-dynamical-matrix assertion, the one retry at
           1.0 THz                      results/revision/sscha_converged_grid_startB1/<tag>_startB.json
  A-seed10 start A, --conv-seed 10      results/revision/sscha_converged_seed10/<tag>_startA.json
their range and standard deviation (sample sd, ddof = 1, over the replicates that are ok), and
whether the three calls agree. The call is the grid's: stable iff Hessian min >= the recipe's
imag_tol_thz (-0.1 THz). Aggregates: the fraction of units whose call is the same in all three,
and the median range. A unit whose calls differ is UNRESOLVED (never settled by majority); a unit
with a replicate that is not ok is reported with that status (no replicate result is dropped).
Also listed per replicate: converged, stop reason, populations, bootstrap sd, |f| > 50 THz blow-up.

    python scripts/box/as_run/e3_summarize.py         # -> results/revision/e3_replicates/
    python scripts/box/as_run/e3_summarize.py --rev-root <scratch copy> --out <scratch>   # test
Runs in any env with numpy (no model, no GPU).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
UNITS = Path(__file__).resolve().parent / "e3_units_2026-10-08.tsv"
REV = REPO / "results" / "revision"
DIRNAMES = {"A": ("sscha_converged_grid", "A"), "B": ("sscha_converged_grid_startB", "B"),
            "B1": ("sscha_converged_grid_startB1", "B"), "A10": ("sscha_converged_seed10", "A")}
BLOWUP_THZ = 50.0
COMPLEX_DYN = "dynamical matrix is complex"


def dig(d, *k):
    for x in k:
        if not isinstance(d, dict):
            return None
        d = d.get(x)
    return d


def replicate(rev: Path, kind: str, tag: str) -> dict:
    sub, start = DIRNAMES[kind]
    p = rev / sub / f"{tag}_start{start}.json"
    if not p.exists():
        return {"file": None, "status": "missing"}
    doc = json.loads(p.read_text(encoding="utf-8"))
    run = doc.get("run") or {}
    st = run.get("status") or "?"
    bmeth = dig(run, "bootstrap", "method")
    bsd = dig(run, "bootstrap", "std_full_estimate_thz" if bmeth == "disjoint_splits" else "std_thz")
    h = dig(run, "hessian", "min_nonac_thz") if st == "ok" else None
    R = doc.get("recipe") or {}
    return {"file": p.relative_to(rev).as_posix(), "status": st, "error": run.get("error"),
            "hessian_min_thz": h, "imag_tol_thz": R.get("imag_tol_thz"),
            "converged": dig(run, "relax", "converged"), "stop_reason": dig(run, "relax", "stop_reason"),
            "n_populations": dig(run, "relax", "n_populations"), "boot_sd_thz": bsd,
            "blowup": None if h is None else bool(abs(h) > BLOWUP_THZ),
            "seed": R.get("seed"), "start_b_thz": R.get("start_b_thz"),
            "global_seed": run.get("global_seed"), "wall_s": run.get("wall_s"),
            "recipe_wo_seed": {k: v for k, v in R.items() if k not in ("seed", "start_b_thz")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev-root", type=Path, default=REV, help="default results/revision")
    ap.add_argument("--out", type=Path, default=None, help="default <rev-root>/e3_replicates")
    a = ap.parse_args()
    rev, out = a.rev_root.resolve(), (a.out or a.rev_root / "e3_replicates")
    units = [ln.rstrip("\n").split("\t") for ln in UNITS.read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    rows, recipe_mismatch = [], []
    for u in units:
        tag, system, model, T, sc, env, wall, why = u
        A, B0, B1, A10 = (replicate(rev, k, tag) for k in ("A", "B", "B1", "A10"))
        retried = B0["status"] == "failed" and COMPLEX_DYN in str(B0.get("error") or "")
        B = B1 if retried else B0
        reps = {"A": A, "B": B, "A10": A10}
        for k, r in reps.items():
            if r["status"] == "ok" and A["status"] == "ok" and r["recipe_wo_seed"] != A["recipe_wo_seed"]:
                recipe_mismatch.append(f"{tag} {k}")
        itol = A.get("imag_tol_thz") if A.get("imag_tol_thz") is not None else -0.1
        hs = {k: r["hessian_min_thz"] for k, r in reps.items() if r["status"] == "ok"}
        calls = {k: bool(h >= itol) for k, h in hs.items()}
        complete = len(hs) == 3
        v = np.array(list(hs.values()), float)
        rng = float(v.max() - v.min()) if len(v) >= 2 else None
        sd = float(v.std(ddof=1)) if len(v) >= 2 else None
        if complete:
            verdict = "agree" if len(set(calls.values())) == 1 else "unresolved"
        else:
            verdict = ("unresolved" if len(set(calls.values())) > 1 else "incomplete")
        row = {"unit_tag": tag, "system": system, "model": model, "T_K": float(T),
               "supercell": sc.replace(" ", "x"), "why": why, "grid_wall_min": float(wall),
               "b_retry_1thz": retried, "b_first_error": B0.get("error") if retried else None,
               "verdict": verdict, "n_ok": len(hs), "range_thz": rng, "sd_thz": sd,
               "imag_tol_thz": itol}
        for k, r in reps.items():
            for f in ("status", "hessian_min_thz", "converged", "stop_reason", "n_populations",
                      "boot_sd_thz", "blowup", "file", "error"):
                row[f"{k}_{f}"] = r.get(f)
            row[f"{k}_call_stable"] = calls.get(k)
        rows.append(row)

    def agg(rs):
        c = [r for r in rs if r["n_ok"] == 3]
        a = [r for r in c if r["verdict"] == "agree"]
        return {"n_units": len(rs), "n_complete": len(c), "n_same_call_all_three": len(a),
                "fraction_same_call": (len(a) / len(c)) if c else None,
                "median_range_thz": float(np.median([r["range_thz"] for r in c])) if c else None,
                "unresolved": [r["unit_tag"] for r in rs if r["verdict"] == "unresolved"],
                "incomplete": [r["unit_tag"] for r in rs if r["verdict"] == "incomplete"]}

    summary = {
        "prereg": "tasks/preregistration-repeats-2026-10-03.md (d816906; Deviation 2026-10-08)",
        "call_rule": "stable iff Hessian min >= imag_tol_thz (grid rule); sd is ddof=1",
        "all": agg(rows),
        "by_selection": {w: agg([r for r in rows if r["why"] == w]) for w in ("disagree", "random")},
        "by_model": {m: agg([r for r in rows if r["model"] == m])
                     for m in sorted({r["model"] for r in rows})},
        "converged_in_all_three": agg([r for r in rows if all(
            r[f"{k}_converged"] for k in ("A", "B", "A10"))]),
        "b_retries_1thz": [r["unit_tag"] for r in rows if r["b_retry_1thz"]],
        "replicate_status_counts": {k: {s: sum(1 for r in rows if r[f"{k}_status"] == s)
                                        for s in sorted({r[f"{k}_status"] for r in rows})}
                                    for k in ("A", "B", "A10")},
        "recipe_mismatch_vs_A": recipe_mismatch,
        "units": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n",
                                      encoding="utf-8", newline="\n")
    cols = [c for c in rows[0] if not c.endswith(("_file", "_error"))] if rows else []
    with open(out / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    a = summary["all"]
    print(f"E3: {a['n_units']} units, {a['n_complete']} with all three replicates ok; same call in "
          f"all three {a['n_same_call_all_three']}/{a['n_complete']}; median range "
          f"{a['median_range_thz']} THz; unresolved {len(a['unresolved'])}, incomplete "
          f"{len(a['incomplete'])}; B retried at 1.0 THz {len(summary['b_retries_1thz'])}")
    print(f"replicate status: {summary['replicate_status_counts']}")
    if recipe_mismatch:
        print(f"WARNING recipe differs from A beyond seed/start_b_thz: {recipe_mismatch}")
    for r in rows:
        if r["verdict"] != "agree":
            print(f"  {r['verdict']:10s} {r['unit_tag']:34s} A {r['A_status']}:{r['A_hessian_min_thz']} "
                  f"B {r['B_status']}:{r['B_hessian_min_thz']} A10 {r['A10_status']}:{r['A10_hessian_min_thz']}")
    print(f"wrote {out / 'summary.json'} and summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
