# Data Availability Statement

**Manuscript:** Finite-temperature dynamic stability is a blind spot of foundation machine-learning interatomic potentials
**Author:** Frank Cai, Purdue University

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability and archived at Zenodo at
https://doi.org/10.5281/zenodo.20805824. The production results regenerate from
`results/ledger.parquet` (per-unit hashed, resumable) and the finite-size runs from
`results/convergence_study.parquet`; figures via `scripts/make_figures.py`, analysis in
`mlip_dynstab/analysis.py`, the SSCHA root-cause diagnostic in `scripts/sscha_v4_diag.py`, and the
stochastic-reproducibility study (§3.5) via `scripts/sscha_repro.py` (its per-seed frequencies
print to the run log rather than to the ledger).
