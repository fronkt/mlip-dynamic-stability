"""Correct the SSCHA acoustic-mode defect (audit D1) in the deposited ledger.

`compute_finite_t_sscha` identified the three acoustic (translational) modes as the three LOWEST
frequencies rather than the three NEAREST ZERO. cellconstructor returns imaginary modes as negative,
so for an unstable high-symmetry phase the soft mode is more negative than the acoustic zeros and
`np.sort(w)[3:]` discarded the instability while keeping the exact zeros -- reporting a minimum of
order 1e-7 THz and calling the phase stable.

The code is fixed (87a9183). This script repairs the DATA, using `ft_lowest6_thz`, which stored the
six lowest frequencies before the bad mask was applied.

Why the stored six suffice. We need min over (all modes minus the 3 acoustic). The acoustic sit near
zero. If the true minimum is a genuine soft mode, it is negative and far from zero, so it is inside
the stored six and survives removal under either identification -- the recovered minimum is exact.
If instead every mode is near zero, the six lowest already contain the whole relevant window. The
only quantity that cannot be recovered this way is the full spectrum, which the ledger never stored.
This is therefore a DERIVED correction, not a re-measurement; the honest fix is a re-run of the 208
SSCHA units, and this correction is what stands until that is affordable.

Original values are preserved in `*_v1` columns so nothing is destroyed.

Usage:
    python scripts/fix_sscha_acoustic.py            # dry run: report what would change
    python scripts/fix_sscha_acoustic.py --apply    # write it (backs the ledger up first)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path(__file__).resolve().parent.parent / "results" / "ledger.parquet"
IMAG_TOL = -0.1


def corrected_min(lowest6) -> float:
    """Minimum frequency excluding the three modes nearest zero in magnitude."""
    w = np.asarray(list(lowest6), dtype=float)
    if w.size <= 3:
        return float(w.min())
    return float(np.delete(w, np.argsort(np.abs(w))[:3]).min())


def main(apply: bool) -> int:
    df = pd.read_parquet(LEDGER)
    mask = (df["method"] == "sscha") & df["ft_lowest6_thz"].notna()
    if not mask.any():
        print("no SSCHA rows with a stored spectrum; nothing to do")
        return 0

    new_min = df.loc[mask, "ft_lowest6_thz"].apply(corrected_min)
    old_min = df.loc[mask, "ft_sscha_min_freq_thz"].astype(float)
    new_stable = new_min >= IMAG_TOL
    old_stable = df.loc[mask, "dynamically_stable"].astype(bool)

    flipped = new_stable != old_stable
    print(f"SSCHA rows inspected      : {int(mask.sum())}")
    print(f"rows whose call changes   : {int(flipped.sum())}")
    print(f"  stable -> unstable      : {int((old_stable & ~new_stable).sum())}")
    print(f"  unstable -> stable      : {int((~old_stable & new_stable).sum())}")
    if flipped.any():
        show = df.loc[mask][flipped][["system", "model", "temperature_K"]].copy()
        show["was"] = old_min[flipped].round(9).values
        show["now"] = new_min[flipped].round(4).values
        print()
        print(show.to_string(index=False))

    if not apply:
        print("\nDRY RUN -- rerun with --apply to write it.")
        return 0

    backup = LEDGER.with_suffix(".parquet.pre-d1-fix")
    if not backup.exists():
        shutil.copy2(LEDGER, backup)
        print(f"\nbacked up original ledger -> {backup.name}")

    # Preserve the v1 values before overwriting.
    for col, src in (("ft_sscha_min_freq_thz_v1", "ft_sscha_min_freq_thz"),
                     ("dynamically_stable_v1", "dynamically_stable")):
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[mask, col] = df.loc[mask, src]

    df.loc[mask, "ft_sscha_min_freq_thz"] = new_min
    df.loc[mask, "min_eff_freq_thz"] = new_min
    df.loc[mask, "dynamically_stable"] = new_stable
    if "pred_stable" in df.columns:
        df.loc[mask, "pred_stable"] = new_stable
    if "gt_stable" in df.columns:
        gt = df.loc[mask, "gt_stable"].astype(bool)
        if "false_stable" in df.columns:
            df.loc[mask, "false_stable"] = new_stable & ~gt
        if "false_unstable" in df.columns:
            df.loc[mask, "false_unstable"] = ~new_stable & gt

    df.to_parquet(LEDGER, index=False)
    print(f"wrote corrected SSCHA rows to {LEDGER.name} (v1 values kept in *_v1 columns)")

    fe = df[(df.method == "sscha") & df.system.isin(
        ["batio3_cubic", "pbtio3_cubic", "knbo3_cubic"]) & (df.temperature_K <= 300)]
    fl = df[(df.method == "sscha") & df.system.isin(["zro2_cubic", "hfo2_cubic"])]
    bcc = df[(df.method == "sscha") & df.system.isin(["ti_bcc", "zr_bcc", "hf_bcc"])]
    print()
    print(f"FE perovskite recall T<=300 (n={len(fe)}) : "
          f"{int((~fe.dynamically_stable.astype(bool)).sum())}/{len(fe)} = "
          f"{(~fe.dynamically_stable.astype(bool)).mean():.2f}")
    print(f"fluorite recall all T (n={len(fl)})       : "
          f"{(~fl.dynamically_stable.astype(bool)).mean():.2f}")
    print(f"bcc rows changed (n={len(bcc)})           : "
          f"{int((bcc.dynamically_stable.astype(bool) != bcc.dynamically_stable_v1.astype(bool)).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
