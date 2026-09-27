"""The H2 transfer asymmetry under each frozen-cell convention of the soft-mode screen.

Referee 3 (round-2 check of the revision) re-solved the screen under the frozen-cell conventions
of ESI Table S14 and found that the headline H2 count, b v c = 17 v 4 at 300 K, belongs to the
production (minimal-cell) convention: the other conventions give 7 v 4 to 24 v 4. This script
makes that check reproducible from the deposited cache and ledger. Nothing is re-evaluated with
an MLIP: every convention re-solves the cached E(Q) maps with the production solver through the
machinery of ``scripts/screen_sensitivity.py`` (whose baseline reproduces the ledger unit by
unit), rebuilds the softmode calls, and recounts the matched harmonic/finite-T pairs exactly as
``stats_hardening.h2_clustered`` does: 15 non-bcc, non-borderline systems x 5 models, b = harmonic
call right and screen call wrong, c = the reverse, with the system-clustered exact test
``stats.cluster_exact_paired``.

    python scripts/h2_by_convention.py            # writes results/h2_by_convention.json

Takes about 15 minutes on a laptop CPU (the probe solves are parallelised over 6 workers).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import screen_sensitivity as S          # noqa: E402

from mlip_dynstab import analysis as A  # noqa: E402
from mlip_dynstab import stats          # noqa: E402

OUT = REPO / "results" / "h2_by_convention.json"
BORDERLINE = {"ktao3_cubic"}


def _scored(d):
    return d[~d["system"].isin(BORDERLINE) & ~d["system"].str.contains("bcc")]


def main() -> None:
    S._install_probe()
    df = A.load_canonical()
    sm = df[df["method"] == "softmode"].copy()
    modes, _ = S.load_modes(sm)
    real = [m for m in modes if not m.placeholder]
    settings = [s for s in S.build_settings() if s.key == "production" or s.key.startswith("cell_")]
    if not any(s.key == "production" for s in settings):
        settings = [s for s in S.build_settings() if s.is_production][:1] + \
                   [s for s in S.build_settings() if s.key.startswith("cell_")]

    problems, keys = {}, set()
    for s in settings:
        for m in real:
            (a, b, c, M), _ = S.mode_problem(m, s)
            problems[(s.key, m.uid)] = (a, b, c, M)
            for T in S.TEMPS:
                keys.add(("probe", a, b, c, M, T, s.q_box, s.nq))
    res = S._run_jobs(sorted(keys, key=lambda k: (k[5], k[1:])), 6)

    h = _scored(df[df["method"] == "harmonic"])[["system", "model", "pred_stable", "gt_stable"]]
    out = {"script": "scripts/h2_by_convention.py",
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "matched_set": "15 non-bcc, non-borderline systems x 5 models",
           "conventions": {}}
    for s in settings:
        mr = {(m.uid, T): res[("probe", *problems[(s.key, m.uid)], T, s.q_box, s.nq)]
              for m in real for T in S.TEMPS}
        units = S.unit_table(modes, mr, s)
        dm = S.modified_ledger(df, units)
        f_all = _scored(dm[dm["method"] == "softmode"])
        rows = {}
        for T in S.TEMPS:
            f = f_all[f_all["temperature_K"] == T]
            m = h.merge(f[["system", "model", "pred_stable", "gt_stable"]], on=["system", "model"],
                        suffixes=("_h", "_f"))
            hok = m["pred_stable_h"] == m["gt_stable_h"]
            fok = m["pred_stable_f"].astype(bool) == m["gt_stable_f"].astype(bool)
            b = (hok & ~fok).to_numpy()
            c = (~hok & fok).to_numpy()
            r = stats.cluster_exact_paired(b, c, m["system"].to_numpy())
            rows[f"{T:.0f}"] = {"n_pairs": int(len(m)), "b": int(b.sum()), "c": int(c.sum()),
                                "p_clustered": r.get("p_exact_clustered"),
                                "systems_favouring_b": r.get("clusters_favouring_a"),
                                "systems_favouring_c": r.get("clusters_favouring_b")}
            print(f"{s.key:24s} T={T:4.0f}  b={rows[f'{T:.0f}']['b']:2d} "
                  f"c={rows[f'{T:.0f}']['c']:2d}  clustered p={r.get('p_exact_clustered')}")
        out["conventions"][s.key] = {"is_production": bool(s.is_production), "by_T": rows}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
