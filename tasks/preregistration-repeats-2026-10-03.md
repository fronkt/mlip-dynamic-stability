# Pre-registration: replicate runs of the converged SSCHA grid (2026-10-03)

Committed while the grid is running and before its results have been read (the box holds them;
nothing has been copied back or summarised). Deviations go in the "Deviations" section below,
dated, never by editing the sections above.

## Purpose

Referee 1.4 asked for SSCHA convergence diagnostics including Hessian uncertainties. The
production recipe's seed study (ESI Table S21) measured the spread of an unconverged recipe. This
measures the uncertainty of the converged recipe that the revised §3.3 will rest on, in two
independent ways: the starting matrix and the random stream.

## Unit selection (fixed now, applied mechanically to the grid's summary.csv)

1. Every grid unit with status ok whose converged call disagrees with its comparison: the label
   for non-bcc units (`label_scored` true), the screen's call for bcc units.
2. A random 10 % (rounded up) of the remaining ok units, drawn with
   `numpy.random.default_rng(20261003).choice` over the unit tags sorted alphabetically.
3. Units with status other than ok are not replicated; they are reported as they are.

## Replicates per selected unit

- **Start B**: `--converge --start B` (imaginary modes set to +0.3 THz instead of |ω|; seed 0 + 1).
  If a unit fails with the complex-dynamical-matrix assertion seen in C1c, it is re-run once with
  `--start-b-thz 1.0` and that is recorded.
- **Seed replicate**: `--converge --start A --conv-seed 10` into its own directory
  (results/revision/sscha_converged_seed10/), same recipe otherwise.

## Reported

Per unit: the lowest Hessian frequency for A (grid), B and A-seed10, their range and standard
deviation, and whether the three calls agree. Aggregates: the fraction of units whose call is the
same in all three, and the median range. A call that differs between replicates is reported as
unresolved in the revised §3.3 counts, not resolved by majority. No replicate result is dropped.

## Deviations

- **2026-10-08, written before any replicate exists** (no start-B or seed-10 run of the grid units
  has been launched; no box is rented). **The random draw is fixed as `replace=False`.** The
  selection rule above names `numpy.random.default_rng(20261003).choice` but not its `replace=`
  argument, and numpy's default (`replace=True`) gives a different 12 units from `replace=False`
  (both readings happen to give 12 distinct units here). "A random 10 % of the remaining units"
  means a sample of distinct units, which is what `replace=False` draws, so that reading is used.
  Applied mechanically to `results/revision/sscha_converged_grid/summary.csv` (f45e77e) by
  `scripts/box/as_run/e3_select.py`: 178 grid units, 171 ok, **53 disagreeing** (non-bcc: call v
  label, `label_scored` true; bcc: call v the screen's call; identical to `call_matches_label` /
  `call_matches_screen`), 118 remaining, ceil(11.8) = **12 drawn**, **65 units** in all; the list is
  also committed as `scripts/box/as_run/e3_units_2026-10-08.tsv`.
  - Disagreeing (53): batio3_cubic_chgnet_100K_sc222, batio3_cubic_mace_mp0_100K_sc222,
    batio3_cubic_mace_mp0_300K_sc222, batio3_cubic_mattersim_100K_sc222,
    batio3_cubic_mattersim_300K_sc222, batio3_cubic_orb_v2_100K_sc222,
    batio3_cubic_sevennet0_100K_sc222, batio3_cubic_sevennet0_300K_sc222,
    cssni3_cubic_chgnet_600K_sc222, cssni3_cubic_chgnet_900K_sc222, cssni3_cubic_mace_mp0_600K_sc222,
    cssni3_cubic_mace_mp0_900K_sc222, cssni3_cubic_mattersim_600K_sc222,
    cssni3_cubic_mattersim_900K_sc222, hf_bcc_orb_v2_300K_sc333, hf_bcc_orb_v2_600K_sc333,
    knbo3_cubic_chgnet_100K_sc222, knbo3_cubic_chgnet_300K_sc222, knbo3_cubic_chgnet_600K_sc222,
    knbo3_cubic_mace_mp0_100K_sc222, knbo3_cubic_mace_mp0_300K_sc222, knbo3_cubic_mace_mp0_600K_sc222,
    knbo3_cubic_mattersim_100K_sc222, knbo3_cubic_mattersim_300K_sc222,
    knbo3_cubic_mattersim_600K_sc222, knbo3_cubic_orb_v2_100K_sc222, knbo3_cubic_orb_v2_300K_sc222,
    knbo3_cubic_orb_v2_600K_sc222, knbo3_cubic_sevennet0_100K_sc222, knbo3_cubic_sevennet0_300K_sc222,
    knbo3_cubic_sevennet0_600K_sc222, pbtio3_cubic_chgnet_300K_sc222, pbtio3_cubic_chgnet_600K_sc222,
    pbtio3_cubic_mace_mp0_300K_sc222, pbtio3_cubic_mace_mp0_600K_sc222,
    pbtio3_cubic_mattersim_300K_sc222, pbtio3_cubic_sevennet0_300K_sc222,
    pbtio3_cubic_sevennet0_600K_sc222, srtio3_cubic_chgnet_100K_sc222, srtio3_cubic_mace_mp0_100K_sc222,
    srtio3_cubic_mattersim_100K_sc222, srtio3_cubic_orb_v2_300K_sc222, srtio3_cubic_orb_v2_600K_sc222,
    srtio3_cubic_orb_v2_900K_sc222, srtio3_cubic_sevennet0_100K_sc222, ti_bcc_chgnet_100K_sc333,
    ti_bcc_chgnet_300K_sc333, ti_bcc_mattersim_600K_sc333, ti_bcc_orb_v2_100K_sc333,
    ti_bcc_orb_v2_600K_sc333, zr_bcc_mattersim_600K_sc333, zr_bcc_orb_v2_100K_sc333,
    zr_bcc_orb_v2_300K_sc333. (Five of them are status ok but not converged: hf_bcc_orb_v2 300/600 K
    and srtio3_cubic_orb_v2 300/600/900 K; the rule says "status ok", so they are included.)
  - Random, `replace=False` (12, USED): batio3_cubic_mace_mp0_900K_sc222, srtio3_cubic_orb_v2_100K_sc222,
    hfo2_cubic_chgnet_900K_sc222, zro2_cubic_sevennet0_100K_sc222, zro2_cubic_mace_mp0_300K_sc222,
    zr_bcc_sevennet0_100K_sc333, zro2_cubic_sevennet0_600K_sc222, srtio3_cubic_mace_mp0_300K_sc222,
    hfo2_cubic_mace_mp0_100K_sc222, hf_bcc_orb_v2_100K_sc333, pbtio3_cubic_mace_mp0_100K_sc222,
    hfo2_cubic_orb_v2_900K_sc222 (in draw order).
  - Random, `replace=True` (12, NOT used, recorded for transparency): hf_bcc_sevennet0_600K_sc333,
    srtio3_cubic_sevennet0_300K_sc222, zro2_cubic_mace_mp0_300K_sc222, srtio3_cubic_chgnet_300K_sc222,
    hfo2_cubic_mace_mp0_300K_sc222, hfo2_cubic_mace_mp0_600K_sc222, zro2_cubic_mattersim_600K_sc222,
    ti_bcc_chgnet_600K_sc333, batio3_cubic_mace_mp0_900K_sc222, zro2_cubic_sevennet0_600K_sc222,
    zr_bcc_sevennet0_100K_sc333, hfo2_cubic_orb_v2_900K_sc222. (The grid fold note of 10-08 had
    recommended this reading; it is not taken, for the reason above.)

  Implementation, consistent with the design and listed for completeness: "A" is the grid's
  existing start-A run (not re-run). Each replicate is one `--converge` process per (unit, start)
  with the grid's settings (`--n-boot 10 --unit-timeout 7200`, the grid supercell: 2x2x2 non-bcc,
  3x3x3 bcc; the recipe is byte-identical to the grid JSONs apart from the seed for A-seed10).
  Start B writes `results/revision/sscha_converged_grid_startB/` (its own directory, so the grid
  directory, its start-A JSONs and its summary.csv are untouched); the 1.0 THz retry after the
  complex-dynamical-matrix assertion writes `results/revision/sscha_converged_grid_startB1/` (the
  recipe stores start_b_thz, so it cannot share a file with the 0.3 THz attempt, as for C1c's
  `_b1/`); A-seed10 writes `results/revision/sscha_converged_seed10/`. Each process has an outer
  3 h wall cap (the grid's longest unit took 134 min); a run killed there stays `running` in its
  JSON and is reported as killed, like any other non-ok replicate. The C1c start-B runs of
  BaTiO3/SrTiO3/ZrO2-MACE 100 K (`results/revision/sscha_converged{,_b1}/`) are not reused.
  Runner: `scripts/box/as_run/gpu_e3_e5_2026-10-08.sh`; reported quantities:
  `scripts/box/as_run/e3_summarize.py`.
