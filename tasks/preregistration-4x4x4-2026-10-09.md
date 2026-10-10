# Pre-registration: 4×4×4 converged SSCHA for SrTiO₃ and one fluorite (2026-10-09)

Committed before any 4×4×4 unit has been run (no box is rented at the time of writing). Deviations go
in the "Deviations" section below, dated, never by editing the sections above.

## Purpose

Referee 3.2 asked to "complete the deferred 4×4×4 SSCHA for SrTiO₃ and one fluorite, or remove all
convergence claims for zone-boundary systems". The submitted revision took the second option. This
run attempts the first, as a cell-size test of the converged-SSCHA calls in the 2×2×2 cell, for the
R-point system (SrTiO₃) and the X-point systems (fluorites). 4×4×4 contains Γ, X, M and R, like 2×2×2,
so the two cells are nested and the comparison is a convergence test in the sense of Section 3.5.

## Units (fixed now, applied mechanically to results/revision/sscha_converged_grid/summary.csv)

1. **SrTiO₃, 100 K**: every SrTiO₃ 100 K grid unit that converged (`converged` true). These are the
   converged false-stables whose call Section 3.5 says "may be a finite-size effect": CHGNet,
   MACE-MP-0, MatterSim, SevenNet-0 (ORB-v2 did not converge and is not run).
2. **One fluorite**: the converged ZrO₂/HfO₂ grid unit with the smallest |lowest Hessian frequency|,
   i.e. the fluorite call closest to flipping: hfo2_cubic_mace_mp0_600K (−2.20 THz at 2×2×2).

Five units, supercell 4×4×4 (SrTiO₃ 320 atoms; fluorite primitive cell, 192 atoms).

## Recipe

The grid's converged recipe, unchanged except for the supercell: `--converge --start A --n-boot 10`,
the CONV_DEFAULTS of scripts/sscha_seed_study.py (1000 configurations, ≤30 populations, 2000 Hessian
configurations, ≤400 steps per population, seed as the grid). `--unit-timeout 28800` (8 h; the
grid's 7200 s was set for 40-atom cells) and an outer 9 h wall cap per process. One start (A) only;
there is no budget for the start-B / seed-10 replicates of the 2026-10-03 pre-registration, and
that is stated wherever the results are reported.

## Reported (whatever comes out)

Per unit: status, converged flag, populations, lowest Hessian frequency at 4×4×4 with its
bootstrap SD, the 2×2×2 value beside it, and the call at each cell.

- 4×4×4 converged and call equal to 2×2×2: the 2×2×2 call is stable to this cell-size change.
- 4×4×4 converged and call different: the 2×2×2 call is a finite-size effect for that unit; for
  SrTiO₃ the headline count without those units (23 of 73) becomes the primary count.
- Not converged, failed, or killed at the cap: reported as such; no convergence claim is made for
  that unit and the narrowed wording of the submitted revision stays.

No unit is dropped, re-run with other settings, or replaced after its result is seen. A rerun after
an infrastructure failure (box lost, process killed by the provider) is allowed once, recorded as a
deviation.

## Deviations

(none yet)

## Outcome (2026-10-10, appended after the run; the sections above are unchanged)

Box: vast.ai 55081379 (RTX 3090, 30.7-vCPU cgroup quota, 94 GB), 2026-10-09 19:33 → 2026-10-10 04:34 UTC;
all five processes ran in parallel and all five reached the outer 9 h cap (`timeout` rc 124,
`jobs.tsv`), so every JSON still reads `status: running`. The box self-stopped at 05:34 UTC; its GPU
was re-rented, so the disk was recovered with `vastai copy` to a relay (no unit was re-run).
Results: `results/revision/sscha_converged_sc444/` (raw logs + work dirs: `mlip-rsc-revision-raw/`,
hashes in `logs/RAW_ARCHIVES.sha256`).

| unit | relaxation | 4×4×4 lowest Hessian freq. | 2×2×2 (grid) | reported as |
|---|---|---|---|---|
| hfo2_cubic_mace_mp0_600K | converged, 7 populations (8822 s); fresh-gradient check consistent with a minimum | −3.22 THz (unstable) | −2.20 ± 0.15 THz (unstable) | converged; call unchanged; killed during the bootstrap, so no 4×4×4 SD |
| srtio3_cubic_chgnet_100K | converged, 5 populations (17334 s) | none (killed in the Hessian) | +1.06 | killed at cap; no 4×4×4 call |
| srtio3_cubic_mace_mp0_100K | not converged (unit wall cap in population 14) | none | +1.12 | not converged; no call |
| srtio3_cubic_mattersim_100K | not converged (population 14) | none | +1.20 | not converged; no call |
| srtio3_cubic_sevennet0_100K | not converged (population 10) | none | +1.09 | not converged; no call |

The auxiliary-dyn frequencies at the end of the relaxations (+0.87 to +1.21 THz for SrTiO₃) are
positive by construction (start A) and are not stability calls; they are not used as such.

Per the rules above: the fluorite 2×2×2 call is unchanged at 4×4×4 for the one unit tested (start A
only, no SD). SrTiO₃ has no 4×4×4 call, so the narrowed SrTiO₃ wording of the submitted revision
stays, now with "attempted; did not complete within 9 h per unit" in place of "not run".
