"""What the soft-mode screen's symmetric-point curvature can and cannot show.

For a single mode with an even potential V(Q), the SCHA free energy at centroid Q0 is
F(Q0) = min over Omega of the Gibbs-Bogoliubov bound, so dF/dQ0 = <V'>_{Q0,sigma}. At Q0 = 0 the
Gaussian average of the odd V' vanishes for every width, and sigma^2 is even in Q0, so the only
surviving term of the second derivative is F''(0) = <V''>_{0,sigma} = M Omega^2: the curvature of
the screen's free energy at the symmetric point IS the self-consistent trial stiffness at Q0 = 0.
It is therefore positive whenever a bound Gaussian solves the width equation there, whatever the
depth of the well, and it cannot register condensation; only the free-energy comparison over the
centroid (the screen's call) can. (Referee 1, round-2 check of the revision.)

This script checks that identity on every cached E(Q) map of the production screen and accounts
for every negative value the production observable ``finite_t._sym_curvature_freq`` reports:

  * over all non-placeholder modes x 4 ladder temperatures, the relative difference between the
    reported omega_eff and the trial frequency Omega/2pi at Q0 = 0 (``finite_t._scha_branch``);
  * each negative omega_eff is classified: the width solver fell back to its nearest grid value
    at one of the three stencil points (no downward sign change on its grid, so no
    self-consistent root was found), or a root existed at all three points (listed with the
    fitted a, b, c so a fit artefact can be recognised: a > 0 means no small-Q instability);
  * the 45 bcc (system, model, T) units paired with SSCHA in Table S16, where the ledger's
    screen curvature is negative, are attributed to one of those classes or to a model with no
    screened imaginary mode (the ledger then carries the softest commensurate harmonic value).

    python scripts/curvature_identity_check.py      # writes results/curvature_identity_check.json

No MLIP is evaluated; the maps are the deposited ``results/cache/softmode_v3m24_*.json``.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np                           # noqa: E402

from mlip_dynstab import finite_t as ft      # noqa: E402
from mlip_dynstab import analysis as A       # noqa: E402

OUT = REPO / "results" / "curvature_identity_check.json"
TEMPS = (100.0, 300.0, 600.0, 900.0)
DQ = 0.005                                   # the stencil step of _sym_curvature_freq


def has_root(Q0, a, b, c, m, T) -> bool:
    """True when _scha_branch finds a downward sign change of its residual on its own grid,
    i.e. a genuine self-consistent width rather than the grid-nearest fallback."""
    kT = max(ft._KB_EV * T, 1e-12)
    mkg = m * ft._AMU_KG

    def curv(s2):
        return 2 * a + 12 * b * (Q0 ** 2 + s2) + 30 * c * (Q0 ** 4 + 6 * Q0 ** 2 * s2 + 3 * s2 ** 2)

    def sc(s2):
        K = curv(s2)
        om = np.sqrt(K * ft._W_TO_OMEGA2 / m)
        x = ft._HBAR_EVS * om / (2 * kT)
        return (ft._HBAR_JS / (2 * mkg * om)) / np.tanh(x) * 1e20

    g = np.geomspace(1e-6, 2.0, 240)
    g = g[np.array([curv(s) for s in g]) > 1e-8]
    if not g.size:
        return False
    with np.errstate(all="ignore"):
        r = np.array([sc(s) - s for s in g])
    r = r[np.isfinite(r)]
    return bool(len(np.where(np.diff(np.sign(r)) < 0)[0]))


def main() -> None:
    rel, neg_fallback, neg_root = [], [], []
    n_eval = 0
    for f in sorted(glob.glob(str(REPO / "results" / "cache" / "softmode_v3m24_*.json"))):
        unit = Path(f).stem.replace("softmode_v3m24_", "").rsplit("_sc", 1)[0]
        for e in json.load(open(f, encoding="utf-8"))["modes"]:
            if e["band"] < 0:
                continue                      # placeholder: the model has no imaginary mode
            for T in TEMPS:
                n_eval += 1
                cv = ft._sym_curvature_freq(e["a"], e["b"], e["c"], e["m_eff"], T, dQ=DQ)
                F0, _, om, _ = ft._scha_branch(0.0, e["a"], e["b"], e["c"], e["m_eff"], T)
                aux = om / (2 * math.pi) / 1e12
                if cv is not None and cv > 0 and aux > 0:
                    rel.append(abs(cv - aux) / aux)
                if cv is not None and cv < 0:
                    ok = all(has_root(q, e["a"], e["b"], e["c"], e["m_eff"], T) for q in (0, DQ, 2 * DQ))
                    rec = {"unit": unit, "T": T, "harm_thz": e["harm_thz"], "omega_eff_thz": cv,
                           "a": e["a"], "b": e["b"], "c": e["c"]}
                    (neg_root if ok else neg_fallback).append(rec)

    # the bcc units paired with SSCHA (Table S16): ledger screen curvature < 0, by cause
    df = A.load_canonical()
    sm = df[(df["method"] == "softmode") & df["system"].str.contains("bcc")
            & df["temperature_K"].isin([100.0, 300.0, 600.0])
            & ~df["system"].str.startswith("agi")]
    fb = {(r["unit"], r["T"]) for r in neg_fallback}
    rt = {(r["unit"], r["T"]) for r in neg_root}
    bcc = []
    for _, r in sm.iterrows():
        if not (r["min_eff_freq_thz"] < 0):
            continue
        key = (f"{r['system']}_{r['model']}", float(r["temperature_K"]))
        cause = ("width-solver fallback at a stencil point" if key in fb else
                 "root found at every stencil point (see a, b, c)" if key in rt else
                 "no screened imaginary mode: ledger shows the softest commensurate harmonic value")
        bcc.append({"system": r["system"], "model": r["model"], "T": float(r["temperature_K"]),
                    "ledger_min_eff_freq_thz": float(r["min_eff_freq_thz"]), "cause": cause})

    rel = np.array(rel)
    out = {
        "script": "scripts/curvature_identity_check.py",
        "identity": "F''(0) = <V''>_{0,sigma} = M Omega^2 for a single even mode",
        "n_mode_T_evaluations": n_eval,
        "positive_omega_eff_vs_trial_frequency_rel_diff": {
            "n": int(rel.size), "median": float(np.median(rel)),
            "p95": float(np.percentile(rel, 95)), "max": float(rel.max())},
        "n_negative_omega_eff": len(neg_fallback) + len(neg_root),
        "n_negative_with_width_fallback_at_a_stencil_point": len(neg_fallback),
        "negative_with_root_at_every_stencil_point": neg_root,
        "bcc_sscha_paired_units_with_negative_screen_curvature": bcc,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "negative_with_root_at_every_stencil_point"},
                     indent=1)[:4000])
    print("negative with a root at every stencil point:")
    for r in neg_root:
        print("  ", r)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
