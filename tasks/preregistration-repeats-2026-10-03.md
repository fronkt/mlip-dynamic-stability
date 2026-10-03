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

(none yet)
