"""E3 replicate selection (tasks/preregistration-repeats-2026-10-03.md, Deviation 2026-10-08).

Applies the pre-registered rule mechanically to the grid's summary.csv:
  1. every status-ok unit whose converged call disagrees with its comparison
     (label for non-bcc units, label_scored True; the screen's call for bcc units);
  2. a random 10 % (rounded up) of the remaining ok units, drawn with
     numpy.random.default_rng(20261003).choice over the tags sorted alphabetically,
     replace=False (Deviation 2026-10-08; the replace=True list is printed for the record);
  3. units with status other than ok are not replicated.

Writes a TSV (tag system model T supercell env wall_min why) for the E3 runner, sorted by
model, then grid wall time descending (longest first, so the tail of each queue is short).

    python scripts/box/as_run/e3_select.py [--out scripts/box/as_run/e3_units_2026-10-08.tsv] [--check]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SUMMARY = REPO / "results" / "revision" / "sscha_converged_grid" / "summary.csv"
SEED = 20261003


def truthy(x: str) -> bool:
    return str(x).strip() == "True"


def select(replace: bool):
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8", newline="")))
    ok = [r for r in rows if r["status"] == "ok"]
    dis = []
    for r in ok:
        key = "call_matches_label" if truthy(r["label_scored"]) else "call_matches_screen"
        if r[key] == "":
            raise SystemExit(f"{r['unit_tag']}: {key} is empty")
        if not truthy(r[key]):
            dis.append(r)
    dis_tags = {r["unit_tag"] for r in dis}
    rest = sorted(r["unit_tag"] for r in ok if r["unit_tag"] not in dis_tags)
    k = math.ceil(0.10 * len(rest))
    rnd = list(np.random.default_rng(SEED).choice(rest, size=k, replace=replace))
    return rows, ok, dis, rest, [str(t) for t in rnd]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "e3_units_2026-10-08.tsv"))
    ap.add_argument("--check", action="store_true",
                    help="only print the selection; write nothing")
    a = ap.parse_args()
    rows, ok, dis, rest, rnd = select(replace=False)
    _, _, _, _, rnd_true = select(replace=True)
    print(f"grid rows {len(rows)}, ok {len(ok)}, disagreeing {len(dis)}, remaining {len(rest)}, "
          f"random draw {len(rnd)} (distinct {len(set(rnd))})")
    print("replace=False:", ", ".join(rnd))
    print("replace=True (not used, for the record):", ", ".join(rnd_true))
    by_tag = {r["unit_tag"]: r for r in ok}
    sel = [(r["unit_tag"], "disagree") for r in dis] + [(t, "random") for t in rnd]
    if len({t for t, _ in sel}) != len(sel):
        raise SystemExit("duplicate unit in the selection")
    out = []
    for t, why in sel:
        r = by_tag[t]
        out.append((r["model"], -float(r["wall_min"] or 0), t, r, why))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    lines = []
    for model, negw, t, r, why in out:
        sc = r["supercell"].replace("x", " ")
        lines.append("\t".join([t, r["system"], model, f"{float(r['T_K']):g}", sc, r["env"],
                                f"{-negw:.1f}", why]))
    tot = {}
    for model, negw, *_ in out:
        tot.setdefault(model, [0, 0.0])
        tot[model][0] += 1
        tot[model][1] += -negw / 60.0
    for m, (n, h) in sorted(tot.items()):
        print(f"  {m:10s} {n:3d} units, {h:5.1f} unit-h per replicate (grid start-A wall)")
    print(f"  total      {len(out):3d} units, {sum(h for _, h in tot.values()):5.1f} unit-h per "
          "replicate")
    if a.check:
        return 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# tag\tsystem\tmodel\tT\tsupercell\tenv\tgrid_wall_min\twhy\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {a.out} ({len(lines)} units)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
