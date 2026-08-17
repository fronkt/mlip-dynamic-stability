# Data Availability Statement

**Manuscript:** Finite-temperature dynamic stability separates foundation machine-learning interatomic potentials that harmonic benchmarks rank equally
**Author:** Frank Cai, Purdue University

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability and archived at Zenodo at
https://doi.org/10.5281/zenodo.20805799 (concept DOI, resolving to the latest version). The
production results regenerate from `results/ledger.parquet` (per-unit hashed, resumable) and the
finite-size runs from `results/convergence_study.parquet`; figures via `scripts/make_figures.py`,
analysis in `mlip_dynstab/analysis.py`, the SSCHA root-cause diagnostic in
`scripts/sscha_v4_diag.py`, and the stochastic-reproducibility study (§3.5) via
`scripts/sscha_repro.py` (its per-seed frequencies print to the run log rather than to the
ledger). The SSCHA results reported are a full re-measurement (`scripts/run_sscha_v2.py`, method
version 3) of the original 208-unit grid in pinned environments (`envs/lock-*.txt`); superseded
generations are retained in the append-only ledger, with `mlip_dynstab.analysis.canonical`
selecting the current one, so every stage of the correction is auditable. Every headline number in
the manuscript is asserted against the ledger by `scripts/verify_claims.py`.
