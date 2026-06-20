# mlip-dynamic-stability — phase checklist & resume state

> Resume protocol: on a new session, read this file + `results/ledger.parquet`. The current
> checkpoint is the lowest unchecked box. Per-unit work (system, model, T, method) is
> idempotent — re-running skips tuples already in the ledger.

## P0 — Stage 1 research brief
- [x] Novelty check vs npj-2025 + PhononBench (finite-T gap confirmed)
- [ ] docs/research_brief.md: lock wedge, curated set, metric defs, references
- [ ] User confirmation of brief (MANDATORY Stage-1 checkpoint)

## P1 — Project + env scaffold  ← CURRENT
- [x] Repo structure
- [x] Model-agnostic calculator loader (lazy imports)
- [x] Curated systems registry (configs/curated_systems.yaml)
- [x] Ledger (hashed parquet)
- [x] Harmonic + finite-T harness skeletons
- [x] Per-model env setup notes (cu128)
- [ ] Smoke-test each MLIP calculator on one structure (needs GPU box)
- [ ] Local git init + (ask before) GitHub remote

## P2 — Layer 1 harmonic harness + validation  (needs GPU)
- [ ] Reference loader (phonondb / Petretto MP phonons / JARVIS)
- [ ] Phonopy finite-displacement pipeline end-to-end on 1 material
- [ ] Pilot ~100 materials × models; reproduce npj-style stability rates
- [ ] CHECKPOINT: first results + harness-validity check

## P3 — Layer 2 finite-T core  (needs GPU)
- [ ] MD-distortion probe working on SrTiO3 cubic (textbook case)
- [ ] SSCHA harness (python-sscha) with MLIP calculator
- [ ] Curated set × models × T; ledger rows
- [ ] CHECKPOINT: finite-T core complete

## P4 — Analysis & figures
- [ ] Confusion matrices + false-stable rates
- [ ] Harmonic-vs-finite-T error decomposition (H2)
- [ ] Ensemble-disagreement calibration (H3)

## P5–P7 — Write / review / finalize (academic-pipeline stages 2–6)
- [ ] Stage 2 WRITE → 2.5 INTEGRITY → 3 REVIEW → 4 REVISE → 4.5 FINAL INTEGRITY
- [ ] Stage 5 FINALIZE (PDF) → Stage 6 PROCESS SUMMARY

## Lessons
See tasks/lessons.md.
