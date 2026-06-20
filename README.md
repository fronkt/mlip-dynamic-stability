# mlip-dynamic-stability

**Beyond harmonic: a finite-temperature dynamic-stability stress test of foundation MLIPs.**

Foundation machine-learning interatomic potentials (MLIPs) are now benchmarked for
*harmonic* phonons and 0 K dynamical stability (npj Comput Mater 2025; PhononBench, Dec
2025). But real dynamic stability is a **finite-temperature** property: an important class
of materials — cubic perovskites, bcc refractory metals, fluorites, superionics — has
**imaginary harmonic modes yet is stabilized at finite T by anharmonicity** (and some
harmonically-stable phases soften with T). This project asks whether foundation MLIPs
reproduce that harmonic→finite-T behavior, or whether the well-documented PES-softening
bias makes their stability calls unreliable exactly where the harmonic approximation breaks.

## Research question

Do foundation MLIPs correctly reproduce *finite-temperature* dynamic stability, or does
PES softening make their stability calls unreliable where the harmonic approximation fails?

**Hypotheses**
- **H1** — PES softening inflates harmonic stability (over-reports stable; high *false-stable* rate).
- **H2** — Harmonic error is *not predictive* of finite-T error (the core blind spot).
- **H3** — Inter-model (ensemble) disagreement flags unreliable finite-T calls better than any single model's energetics.

## Design (two staged layers)

- **Layer 1 — harmonic baseline** (fast, credibility anchor): reproduce npj-style harmonic
  dynamical-stability classification on a public DFT-phonon set; extend model coverage.
- **Layer 2 — finite-T core** (novel): SSCHA / TDEP-from-MD effective phonons with each
  MLIP as the calculator, on a curated set with documented harmonic→finite-T transitions.

**Models:** MACE-MP-0, CHGNet, ORB, SevenNet-0, MatterSim (+ optional GRACE / eqV2-OMat24).
**Headline metric:** false-stable rate (the screening-dangerous error).
**Data:** public only (Materials Project / phonondb / JARVIS + published anharmonic refs) — no new DFT.

## Layout

```
mlip_dynstab/        # package: model-agnostic harness
  calculators.py     # lazy ASE-calculator factory, one model per env
  systems.py         # curated system registry (from configs/curated_systems.yaml)
  references.py      # public DFT-phonon reference loaders
  harmonic.py        # phonopy finite-displacement harmonic phonons + stability call
  finite_t.py        # SSCHA + TDEP/MD-distortion finite-T effective phonons
  ledger.py          # hashed parquet results ledger (checkpointable units)
  cli.py             # run one (system, model, T) unit
configs/curated_systems.yaml
envs/                # per-model environment setup (cu128 for RTX 5090)
scripts/             # orchestration
results/             # parquet ledger + figures (gitignored except schema)
tasks/               # todo.md / lessons.md (pause-resume state)
docs/research_brief.md
```

## Reproducibility

Every result is one hashed row in `results/ledger.parquet` carrying the exact structure,
supercell, displacement/MD settings, model name+version, and computed observables. The unit
of work is a single `(system, model, temperature, method)` tuple, so runs are fully
resumable after an interruption: re-running skips tuples already in the ledger.

## Status

Scaffold (P1). See `tasks/todo.md` for the phase checklist and the current checkpoint.
