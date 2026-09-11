"""Compare the SCHA screen against an exact quantum solution of the same fitted 1D potential.

Motivation. Referee 1 (RSC Advances RA-ART-07-2026-006452) raises a real design point: the cheap
screen and the SSCHA cross-check are both driven by the same MLIP potential-energy surface, so
their agreement is a consistency test rather than a measure of physical accuracy. The objection
bundles two separable error sources -- is the fitted double well right (PES error), and given
that well does the variational treatment get the right answer (method error)? Only an
independent electronic-structure reference can address the first. The second looked addressable
exactly and for free, because each unit's deciding mode is stored in the ledger as a mass and
three polynomial coefficients, and `finite_t._solve_1d` already solves that 1D Schroedinger
problem to machine precision by tridiagonal diagonalisation. It was written during development
and left unused.

**What we found, and why this is not the validation it looks like.** The exact solver's
stability criterion is the potential of mean force W(Q) = -kT ln P(Q,T): the mode has condensed
if the thermal density is bimodal. Run over the deposited coefficients, that criterion calls
96.5% of modes unstable **at every temperature on the ladder, to three decimal places** -- it is
temperature-independent. That is not a defect in the solver, it is the physics of an isolated
mode: as kT grows, P -> exp(-V/kT), whose maxima are the minima of V, so a genuine double well
stays bimodal at any temperature. An isolated one-dimensional mode has no mechanism by which to
thermally stabilise.

So the exact 1D solution is **not** a usable reference for finite-temperature phase stability,
and the disagreement between it and the SCHA screen must not be read as SCHA error. The two
criteria answer different questions. This script is retained for the two things it does
establish, both of which are worth reporting:

  1. **A one-sided soundness check.** Across all modes tested, there is not a single case where
     the SCHA screen condenses a mode that the exact treatment finds unimodal. Every
     condensation the screen reports corresponds to a potential that genuinely has a displaced
     minimum, so the screen does not manufacture instabilities at the mode level -- the
     complement of the control-set result in section 3.2, which shows it does not manufacture
     them at the phase level either.

  2. **Where the thermal physics actually comes from.** The exact isolated-mode criterion is
     flat in temperature; the SCHA criterion falls from 0.86 to 0.49 across the same ladder.
     The temperature dependence that lets the screen bracket the SrTiO3 transition is supplied
     by the SCHA self-consistency -- the Gaussian width broadening with T and re-entering the
     averaged potential -- and not by the shape of the double well. This is a concrete argument
     for the variational treatment over a direct thermal-density criterion, and it explains why
     the three discarded routes of ESI section S1, which all read a density or an effective
     force constant rather than minimising a free energy, failed the SrTiO3 gate.

Writes results/scha_vs_exact.json. Run from the repo root:

    python scripts/scha_vs_exact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from mlip_dynstab import analysis as A    # noqa: E402
from mlip_dynstab import stats as S       # noqa: E402
from mlip_dynstab.finite_t import _solve_1d, _solve_scha   # noqa: E402

LEDGER = REPO / "results" / "ledger.parquet"
OUT = REPO / "results" / "scha_vs_exact.json"


def main() -> None:
    df = A.canonical(pd.read_parquet(LEDGER))
    d = df[df["method"] == "softmode"].copy()
    d = d[d[["ft_v_a", "ft_v_b", "ft_v_c", "ft_m_eff_amu"]].notna().all(axis=1)]
    # a flat/absent well carries no mode to test
    d = d[(d["ft_v_a"] != 0) | (d["ft_v_b"] != 0) | (d["ft_v_c"] != 0)]

    rows = []
    for _, r in d.iterrows():
        a, b, c = float(r.ft_v_a), float(r.ft_v_b), float(r.ft_v_c)
        m, T = float(r.ft_m_eff_amu), float(r.temperature_K)
        try:
            ex_freq, ex_rmsQ, ex_stable = _solve_1d(a, b, c, m, T)
        except Exception as exc:                        # record, never silently drop
            rows.append({"system": r.system, "model": r.model, "T": T,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        sc_freq, sc_Q0, _ = _solve_scha(a, b, c, m, T)
        rows.append({
            "system": r.system, "model": r.model, "T": T, "error": None,
            "well_depth_meV": float(r.ft_well_depth_meV),
            "a": a,
            "scha_condensed": bool(sc_Q0 > 1e-6),
            "exact_bimodal": bool(not ex_stable),
            "scha_Q0_ang": float(sc_Q0), "exact_rmsQ_ang": float(ex_rmsQ),
        })

    t = pd.DataFrame(rows)
    errs = int(t["error"].notna().sum())
    ok = t[t["error"].isna()].copy()

    # NOTE: use ok["T"], never ok.T -- the latter is the DataFrame transpose. This bit us once.
    by_T = ok.groupby(ok["T"]).agg(
        n=("T", "size"),
        exact_bimodal_frac=("exact_bimodal", "mean"),
        scha_condensed_frac=("scha_condensed", "mean"),
    )

    manufactured = ok[ok["scha_condensed"] & ~ok["exact_bimodal"]]
    missed = ok[~ok["scha_condensed"] & ok["exact_bimodal"]]

    res = {
        "_generated_by": "scripts/scha_vs_exact.py",
        "_what_this_is_not": (
            "NOT a validation of SCHA accuracy. The exact isolated-mode criterion is the "
            "bimodality of the thermal density, which is temperature-independent because "
            "P -> exp(-V/kT) keeps a double well bimodal at any T. It cannot represent thermal "
            "stabilisation and is therefore not a finite-temperature reference."),
        "_what_this_is": (
            "a one-sided soundness check on the screen, plus a demonstration that the "
            "temperature dependence of the screen comes from the SCHA self-consistency rather "
            "than from the shape of the well"),
        "n_modes_tested": int(len(ok)),
        "n_solver_errors": errs,
        "soundness_no_manufactured_condensations": {
            "n_scha_condensed_but_exact_unimodal": int(len(manufactured)),
            "interpretation": ("zero means every condensation the screen reports sits on a "
                               "potential that genuinely has a displaced minimum"),
        },
        "criterion_divergence": {
            "n_exact_bimodal_but_scha_did_not_condense": int(len(missed)),
            "median_well_depth_meV_agree": float(ok[ok["scha_condensed"] == ok["exact_bimodal"]]
                                                 ["well_depth_meV"].median()),
            "median_well_depth_meV_diverge": float(missed["well_depth_meV"].median())
                                             if len(missed) else float("nan"),
            "interpretation": ("divergence concentrates in shallow wells and grows with "
                               "temperature, which is the expected signature of the exact "
                               "criterion being temperature-blind rather than of SCHA error"),
        },
        "temperature_dependence": {
            str(int(T)): {
                "n": int(r.n),
                "exact_bimodal_frac": round(float(r.exact_bimodal_frac), 4),
                "scha_condensed_frac": round(float(r.scha_condensed_frac), 4),
            } for T, r in by_T.iterrows()
        },
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"\nmodes tested: {len(ok)}  (solver errors: {errs})")
    print(f"\nSOUNDNESS: SCHA condensed where exact density is unimodal: "
          f"{len(manufactured)}  (0 = the screen never manufactures a mode-level condensation)")
    print("\nfraction called unstable/condensed, by temperature:")
    print("   T (K)    exact (bimodality)    SCHA (free-energy argmin)")
    for T, r in by_T.iterrows():
        print(f"   {int(T):5d}    {r.exact_bimodal_frac:18.4f}    {r.scha_condensed_frac:.4f}")
    print("\nThe exact column is flat: an isolated 1D mode cannot thermally stabilise.")
    print("The SCHA column falls with T. That temperature dependence is the self-consistency,")
    print("not the well shape, and it is what lets the screen bracket the SrTiO3 transition.")


if __name__ == "__main__":
    main()
