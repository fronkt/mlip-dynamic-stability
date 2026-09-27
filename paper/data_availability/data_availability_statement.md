# Data Availability Statement

**Manuscript:** Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation machine-learning interatomic potential for finite-temperature dynamic stability

**Author:** Frank Cai, Purdue University

<!-- PENDING-P5: this text mirrors the manuscript's "Data availability" section as it stood on
2026-09-27, before that section was revised. Re-copy it verbatim once the manuscript section is
final (the audit's PKG-h asks it to name the revision's new code and outputs, e.g.
mlip_dynstab/stats.py, scripts/stats_hardening.py, results/stats_hardening.json,
scripts/build_esi_tables.py, scripts/run_disp_sweep.py, scripts/screen_sensitivity.py, and to
point at the new Zenodo version or a pinned commit), then rebuild the .docx with
pandoc data_availability_statement.md -o data_availability_statement.docx
Stale in the current manuscript text, to be fixed there first: "SSCHA root-cause diagnostic"
(the truncation attribution is withdrawn; call sscha_v4_diag.py the Table S2 diagnostic);
"220 cached E(Q) maps" (200 current v3 maps plus 20 superseded v2); the SSCHA rows of
results/convergence_study.parquet are the June v1 generation, not the pinned re-measurement;
the per-seed values of sscha_repro.py are not deposited (the new scripts/sscha_seed_study.py
output is, PENDING-C1); scripts/dft_reference.py and its outputs (PENDING-C3a/C3b) and
scripts/force_spread.py (PENDING-C2) need naming once they land. -->

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability and archived at Zenodo at
https://doi.org/10.5281/zenodo.20805799 (concept DOI, resolving to the latest version). The
production results regenerate from `results/ledger.parquet` (per-unit hashed, resumable) and the
finite-size runs from `results/convergence_study.parquet`; figures via `scripts/make_figures.py`,
analysis in `mlip_dynstab/analysis.py`, the SSCHA root-cause diagnostic in
`scripts/sscha_v4_diag.py`, and the stochastic-reproducibility study (§3.5) via
`scripts/sscha_repro.py` (its per-seed frequencies print to the run log rather than to the
ledger). The SSCHA results reported here are a full re-measurement (`scripts/run_sscha_v2.py`,
method version 3) of the original 208-unit grid in the pinned environments, after two defects were
found in the original analysis path: an acoustic-mode identification inversion (the three
translational modes were selected by lowest frequency rather than smallest magnitude, which for an
unstable phase discards the soft mode itself), and an unrecorded software environment. The
superseded generations are retained in the append-only ledger — `mlip_dynstab.analysis.canonical`
selects the current one — together with the in-place derived correction
(`scripts/fix_sscha_acoustic.py`, `*_v1` columns) and the uncorrected snapshot
(`results/ledger.parquet.pre-d1-fix`), so every stage of the correction is auditable.

The finite-temperature screen results reported here are the multi-mode grid described in §2.4
(5 models × 20 systems × 4 temperatures = 400 units). The ledger is append-only and also retains
the superseded single-mode rows; `mlip_dynstab.analysis.canonical` selects the current generation,
and every figure and statistic in this article is computed through it. The unit hash includes the
method version and the mode cap, so a re-run under a changed algorithm records new rows rather than
silently reusing old ones. Each model was run in its own pinned environment, with `pip freeze`
manifests deposited as `envs/lock-<model>-2026-08-16.txt`; the 220 cached E(Q) maps are included so
the temperature solves regenerate without a GPU.
