"""Where the centroid scan's free-energy differences come from: the SCHA split per arm.

Second-round referee, comment 1. The SCHA free energy at a centroid is F = F_harm[Phi] + <V - V_Phi>:
the harmonic free energy of the auxiliary matrix Phi plus the averaged anharmonic remainder (which
carries the potential energy gained by displacing the centroid). This splits each arm's
F(arm) - F(Q0 = 0) of the R2 centroid scan into the two parts, from the final auxiliary matrices
(results/revision/r2_centroid/dyn/*_final_aux.dyn.gz, copied from the box work dirs) and the fresh
ensemble free energies of the arm JSONs. The split depends on Phi and is the SCHA's own, not unique.

    python scripts/centroid_decomposition.py      # -> results/revision/r2_centroid/decomposition.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results" / "revision" / "r2_centroid"
ARMS = ("m_scan_Q0.0000", "m_scan_Q0.0400", "m_scan_Q0.0750", "m_scan_Q0.1000", "gamma", "free_Q0.0750")
T_K = 100.0
KB_EV, H_EVS, RY_EV = 8.617333262e-5, 4.135667696e-15, 13.605693123
ACOUSTIC_THZ = 1e-3


def freqs_thz(arm: str):
    raw = gzip.decompress((D / "dyn" / f"{arm}_final_aux.dyn.gz").read_bytes())
    f = [float(m.group(1)) for m in re.finditer(rb"freq \(\s*\d+\)\s*=\s*([-\d.]+) \[THz\]", raw)]
    return np.array(f), hashlib.sha256(raw).hexdigest()


def f_harm_ev(f_thz) -> float:
    e = H_EVS * f_thz[f_thz > ACOUSTIC_THZ] * 1e12
    return float(np.sum(e / 2 + KB_EV * T_K * np.log1p(-np.exp(-e / (KB_EV * T_K)))))


def main() -> int:
    rec = {a: json.loads((D / f"{a}.json").read_text(encoding="utf-8"))["run"] for a in ARMS}
    f0, _ = freqs_thz(ARMS[0])
    fe0 = rec[ARMS[0]]["free_energy"]["fresh_ensemble"]
    out = {"script": "scripts/centroid_decomposition.py", "T_K": T_K, "units": "meV per 40-atom cell",
           "split": "dF = dF_harm(Phi) + d<V - V_Phi>; the SCHA's own split, Phi-dependent",
           "reference_arm": ARMS[0], "arms": {}}
    pos0 = f0[f0 > ACOUSTIC_THZ]
    for a in ARMS[1:]:
        f, sha = freqs_thz(a)
        fe = rec[a]["free_energy"]["fresh_ensemble"]
        df = (fe["fe_Ry"] - fe0["fe_Ry"]) * RY_EV * 1e3
        dfh = (f_harm_ev(f) - f_harm_ev(f0)) * 1e3
        pos = f[f > ACOUSTIC_THZ]
        ratio = np.sort(pos) / np.sort(pos0)
        out["arms"][a] = {
            "converged": bool(rec[a]["relax"]["converged"]),
            "dF_meV": df, "dF_err_meV": float(np.hypot(fe["fe_err_Ry"], fe0["fe_err_Ry"]) * RY_EV * 1e3),
            "dF_harm_aux_meV": dfh, "d_anharmonic_remainder_meV": df - dfh,
            "n_optical_modes": int(len(pos)), "n_modes_stiffer": int(np.sum(ratio > 1)),
            "mean_ln_freq_ratio": float(np.mean(np.log(ratio))),
            "lowest_aux_thz": float(pos.min()), "lowest_aux_ref_thz": float(pos0.min()),
            "dyn_sha256": sha}
        r = out["arms"][a]
        print(f"{a:15s} conv {r['converged']!s:5s} dF {df:+7.1f} = harm(Phi) {dfh:+7.1f} + remainder "
              f"{df - dfh:+7.1f} meV; stiffer {r['n_modes_stiffer']}/{r['n_optical_modes']}, "
              f"mean dln w {r['mean_ln_freq_ratio']:+.3f}")
    (D / "decomposition.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
