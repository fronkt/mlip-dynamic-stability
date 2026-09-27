# mlip-dynamic-stability

Code, per-unit results ledger and analysis for a finite-temperature dynamic-stability benchmark
of foundation machine-learning interatomic potentials (MLIPs).

**Paper:** F. Cai, *Neither harmonic benchmarks nor a default SSCHA cross-check certifies a
foundation machine-learning interatomic potential for finite-temperature dynamic stability*
(manuscript under review).
**Archive:** Zenodo, https://doi.org/10.5281/zenodo.20805799 (concept DOI; resolves to the
latest version).

## What it tests

Foundation MLIPs are benchmarked mostly against harmonic (0 K) phonons. Several important phases
have imaginary harmonic modes and are still the equilibrium phase above a transition temperature:
cubic oxide and halide perovskites, bcc Ti/Zr/Hf, cubic ZrO₂ and HfO₂. This repository runs five
foundation MLIPs as shipped, without fine-tuning (MACE-MP-0, CHGNet, ORB-v2, SevenNet-0,
MatterSim), on 20 reference systems, six of them harmonically stable controls, at 100–900 K.
The labels come from the published experimental and DFT literature
(`configs/curated_systems.yaml`); KTaO₃ is flagged borderline and is not scored.

## Three layers

1. **Harmonic baseline.** phonopy finite-displacement phonons with each MLIP; a phase is called
   stable if its lowest frequency is above −0.1 THz. Scored against the documented harmonic
   split. (`mlip_dynstab/harmonic.py`)
2. **Soft-mode screen (single-mode quantum SCHA).** Every distinct imaginary commensurate mode of the
   harmonic force constants is frozen into its minimal commensurate cell, its static energy E(Q)
   is mapped with the MLIP and fitted with a sextic, and a single-mode quantum self-consistent
   harmonic free energy is minimised over the order-parameter centroid. The phase is called
   unstable if any mode condenses, which is a free-energy comparison between the symmetric and
   displaced states. The screen also records the free-energy curvature at the symmetric point
   Q = 0, a local quantity that can stay positive while a displaced minimum is lower. The E(Q)
   maps are cached, so each temperature is a sub-second CPU solve.
   (`compute_finite_t_softmode` in `mlip_dynstab/finite_t.py`)
3. **SSCHA cross-check.** python-sscha / cellconstructor with the MLIP as force engine, used in
   its default form: stochastic relaxation of the auxiliary dynamical matrix, then the
   free-energy Hessian at the symmetric reference, at bubble level (`include_v4=False`). It runs
   on the same MLIP energies as the screen, so it is a cross-check on that potential-energy
   surface, not an independent reference. (`compute_finite_t_sscha`)

## What the paper reports

A summary only. The paper and its ESI give each rate as k/n with a 95% interval and state which
tests are clustered by system; `scripts/verify_claims.py` asserts the numbers against the ledger.

- **Harmonic layer.** The models divide on which instabilities they reproduce: MatterSim and
  SevenNet-0 get 19 of 19 scored systems right, MACE-MP-0 and ORB-v2 17, CHGNet 15. MACE-MP-0
  and CHGNet read the bcc Zr and Hf soft modes as stable.
- **Finite temperature.** Units that the harmonic layer gets right and the screen mis-calls at
  finite temperature exist, and in unit counts they outnumber the reverse (17 against 4 at
  300 K, on the matched set of 15 systems), so a harmonic benchmark does not certify
  finite-temperature behaviour. The asymmetry is not significant once units are clustered by
  system (exact clustered p = 0.15 at 300 K), and 13 of the 17 sit in three systems that the
  screen mis-calls for at least four of the five models, so most of it reflects the screen's own single-mode
  approximation rather than errors specific to one model's potential. A PBE-backed screen on
  those systems is the test of that reading.<!-- PENDING-C3a: PBE-backed screen calls on BaTiO3, KNbO3, CsSnBr3 (and SrTiO3, bcc-Zr, ZrO2); replace the last sentence with the result whichever way it falls --> The design does not separate the five
  models at finite temperature, and the paper makes no ranking claim.
- **SSCHA.** In its default form, SSCHA calls deep displacive wells stable where the reference
  label is unstable. On the same MLIP energies, the screen's own curvature at Q = 0 is also
  positive on 52 of 57 of those units, while the screen's free-energy comparison finds the
  displaced minimum. The default SSCHA criterion asks whether the symmetric phase is a local
  free-energy minimum; it cannot certify stability against condensation into a deeper displaced
  minimum, which is what the thermodynamic labels record. Whether MLIP error on the thermally
  sampled configurations also contributes is being checked against PBE.<!-- PENDING-C3b: PBE forces/energies on SSCHA-sampled configurations; replace the last sentence with the result -->
- **Ensemble guardrail.** A split stable/unstable vote among the models is a suggestive flag
  for units where the majority call is wrong (AUC 0.762, cluster-bootstrap 95% interval
  [0.590, 0.934], all five models), not a robust one: without ORB-v2 the AUC is 0.628
  [0.438, 0.839], an interval that spans 0.5.<!-- PENDING-C2: force-level ensemble spread (five-model spread; MACE and MatterSim size committees); add one clause with its clustered AUC -->

## Layout

```
mlip_dynstab/            # the harness; backend-agnostic, one MLIP per environment
  calculators.py         # ASE calculator factory for the five models
  systems.py             # curated system registry (configs/curated_systems.yaml)
  references.py          # public DFT-phonon reference loaders
  harmonic.py            # layer 1: phonopy finite-displacement phonons and stability call
  finite_t.py            # layer 2 screen and layer 3 SSCHA hook; also three finite-T routes
                         #   (TDEP-style fit, hiPhive, rattled MD) that failed the SrTiO3 gate
                         #   and were discarded, kept for the record
  ledger.py              # hashed, append-only parquet ledger, one row per unit
  analysis.py            # rates, confusion matrices, H2/H3 analyses; `canonical` selects
                         #   the current generation of each method
  stats.py               # Wilson intervals; system-clustered permutation, bootstrap, exact tests
  cli.py                 # run one (system, model, method, temperature) unit
configs/curated_systems.yaml
envs/                    # per-model environment setup and pinned pip-freeze locks (lock-*.txt)
scripts/                 # grid runners, statistics, ESI tables, figures, claim checks
results/ledger.parquet   # the per-unit results ledger
results/*.json, *.md     # derived statistics (stats_hardening, screen_sensitivity, estimator_noise)
results/figures/         # paper figures, 600 dpi PNG and TIFF; upload/ holds numbered copies
paper/                   # manuscript and ESI sources, TOC entry
docs/                    # process notes, the original research brief, run logs
tasks/                   # working notes, plans and audits
```

## Reproduce

Every result is one hashed row in `results/ledger.parquet` carrying the structure, supercell,
settings, model name and version, and the computed observables. The unit of work is one
(system, model, method, temperature) tuple, so runs resume after an interruption: a re-run skips
units already in the ledger. The ledger is append-only. Superseded generations stay in it, and
`mlip_dynstab.analysis.canonical` selects the current one; every figure and statistic goes
through it.

From the repository root, with the base requirements plus matplotlib and Pillow installed (no
GPU or MLIP needed):

```
python scripts/verify_claims.py           # assert the paper's numbers against the ledger
python scripts/stats_hardening.py         # counts, intervals, clustered tests -> results/stats_hardening.json
python scripts/build_esi_tables.py        # regenerate the ESI tables (--check: fail if out of date)
python scripts/make_figures.py            # Figs. 1-6
python paper/toc_entry/make_toc_graphic.py
```

Running new units needs one environment per model; see `envs/README.md`.
