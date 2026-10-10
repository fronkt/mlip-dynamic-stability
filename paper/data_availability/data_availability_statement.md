# Data Availability Statement

**Manuscript:** Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation machine-learning interatomic potential for finite-temperature dynamic stability

**Author:** Frank Cai, Purdue University

<!-- Verbatim copy of the manuscript's "Data availability" section (release v2.1.0, Zenodo 10.5281/zenodo.23286095), 2026-10-10. Rebuild: pandoc data_availability_statement.md -o data_availability_statement.docx -->

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability. The version that produced
the numbers in this article is release v2.1.0 (commit 0343264) on the repository's default branch,
archived at Zenodo as https://doi.org/10.5281/zenodo.23286095, a version under the concept DOI
https://doi.org/10.5281/zenodo.20805799, which resolves to the latest version.

All production results regenerate from `results/ledger.parquet` (per-unit hashed, append-only,
resumable) through `mlip_dynstab.analysis.canonical`, which selects the current generation of each
layer; every figure and statistic in this article is computed through it. Figures are made by
`scripts/make_figures.py` and the analysis is in `mlip_dynstab/analysis.py`. The Wilson intervals
and the system-clustered tests are in `mlip_dynstab/stats.py`, run by `scripts/stats_hardening.py`
into `results/stats_hardening.json`. The ESI tables are generated from the ledger and that file by
`scripts/build_esi_tables.py`, whose `--check` option fails if the ESI on disk is out of date, and
`scripts/verify_claims.py` re-derives the headline numbers from the ledger. The sensitivity
analysis of the screen (ESI Table S14) re-solves the cached E(Q) maps with
`scripts/screen_sensitivity.py` into `results/screen_sensitivity.json`, and the
displacement-amplitude sweep (ESI Table S13) is run by `scripts/run_disp_sweep.py`, with its rows
in the ledger. The stage-by-stage SSCHA diagnostic of ESI Table S2 is `scripts/sscha_v4_diag.py`.
The bcc-Zr seed study of Section 3.5 is `scripts/sscha_repro.py`; its per-seed frequencies were printed to
the run log and are not deposited. The finite-size rows are in `results/convergence_study.parquet`,
whose SSCHA rows are the superseded v1 generation (Section 3.5). The SSCHA seed study with convergence
diagnostics, the 3×3×3 bcc-Zr re-measurement and the converged-recipe grid
(`scripts/sscha_seed_study.py`; the grid is compared with the production values by
`scripts/grid_compare.py` into `results/revision/grid_compare.json`), the force-level ensemble test
(`scripts/force_spread.py`), the first-principles reference calculations
(`scripts/dft_reference.py`) and their convergence, functional and lattice checks
(`scripts/dft_checks.py`) write their outputs under `results/revision/`. The identity between
the screen's symmetric-point curvature and its trial stiffness, and the origin of every negative
value of that observable, are checked by `scripts/curvature_identity_check.py` (into
`results/curvature_identity_check.json`), and the H2 count under each frozen-cell convention by
`scripts/h2_by_convention.py` (into `results/h2_by_convention.json`); both re-solve the cached
maps and need no GPU. The pre-registered fine-tuning trial (`scripts/finetune_trial.py`), the replicates
of the converged grid and the 4×4×4 attempt (runners in `scripts/box/as_run/`) also write under
`results/revision/`, and their pre-registrations are in `tasks/`.

The ledger is append-only and keeps each superseded generation beside the current one, so every
correction is auditable. The screen results reported here are the multi-mode grid of Section 2.4
(5 models × 20 systems × 4 temperatures = 400 units, method version 4). The first screen
generation in the ledger, a single-mode one, had two defects. Its Γ-point acoustic mask removed
the three lowest branches rather than the three nearest zero, which for an unstable phase deletes
the ferroelectric soft mode, and it was computed with a **q**-point search that had been replaced
in the code, because the unit hash did not carry the algorithm version and the stale units were
skipped as already present. The unit hash now includes the method version and the mode cap, so a
re-run under a changed algorithm records new rows rather than reusing old ones. The SSCHA results
are a full re-measurement (`scripts/run_sscha_v2.py`, method version 3) of the 208-unit grid in the
pinned environments. The first SSCHA generation took the three acoustic modes to be the lowest
frequencies rather than those smallest in magnitude, which discards the soft mode of an unstable
phase, and was computed in an unrecorded software environment. It is retained together with the
in-place derived correction (`scripts/fix_sscha_acoustic.py`, `*_v1` columns) and the uncorrected
snapshot (`results/ledger.parquet.pre-d1-fix`). The harmonic layer was likewise re-measured in the
pinned environments, and its first generation is kept as the replicate of ESI Section S1.2. Each model was
run in its own pinned environment, with `pip freeze` manifests deposited as
`envs/lock-<model>-2026-08-16.txt`. The cached E(Q) maps of the current screen generation
(`results/cache/softmode_v3m24_*.json`, one per system and model) are included, so the
temperature solves and the sensitivity analysis regenerate without a GPU.
