# mlip-dynamic-stability — phase checklist & resume state

> Resume protocol: on a new session, read this file + `results/ledger.parquet`. The current
> checkpoint is the lowest unchecked box. Per-unit work (system, model, T, method) is
> idempotent — re-running skips tuples already in the ledger.

## P0 — Stage 1 research brief
- [x] Novelty check vs npj-2025 + PhononBench (finite-T gap confirmed)
- [x] docs/research_brief.md: lock wedge, curated set, metric defs, references
- [ ] User confirmation of brief (MANDATORY Stage-1 checkpoint)  ← AWAITING USER

## P1 — Project + env scaffold  [DONE except GPU smoke-test]
- [x] Repo structure
- [x] Model-agnostic calculator loader (lazy imports)
- [x] Curated systems registry (configs/curated_systems.yaml) — 20 systems, verified
- [x] Ledger (hashed parquet) — hash + idempotent append verified locally
- [x] Harmonic + finite-T harness skeletons (py_compile clean)
- [x] Per-model env setup notes (cu128)
- [x] Local git init + initial commit (9c2599f)
- [ ] Smoke-test each MLIP calculator on one structure (needs GPU box)
- [x] GitHub remote + push (fronkt/mlip-dynamic-stability @ 9c2599f)

## P2 — Layer 1 harmonic harness + validation  [MACE done]
- [x] Phonopy finite-displacement pipeline end-to-end (Gamma-centered mesh)
- [x] Harness validity: Si stable (+0.4 THz), SrTiO3 imaginary (-1.83 THz), all 6 controls correct
- [x] MACE-MP-0 harmonic grid, 20 systems: acc 0.85, false-stable rate 0.154
      KEY FINDING: false-stable on bcc Zr & Hf (MACE misses beta-Zr/Hf instability,
      catches beta-Ti); false-unstable on KTaO3 (borderline quantum-paraelectric label).
- [~] Other models harmonic grid: per-model venv with --system-site-packages (reuse working
      torch 2.12/cu130 so Blackwell sm_120 kernels are present). CHGNet env built + running.
      Next: ORB, SevenNet, MatterSim (same pattern).
- [ ] Optional: JARVIS reference cross-check (npj-style rates on shared materials)
- CHECKPOINT: MACE harmonic landed + pushed (dd4516a); cross-model in progress

## DECISION NEEDED (finite-T core)
- one-shot TDEP (hiPhive) under-detects soft modes -> needs SSCHA or symmetry-broken sampling.
- Options for next session: (A) implement SSCHA finite-T, (B) rattled-start MD symmetry-
  breaking probe, (C) ship harmonic cross-model paper first, finite-T as follow-up.

## NOTE — ground-truth label review
- [ ] Reclassify KTaO3 as "borderline" (incipient ferroelectric; DFT mode ~0/marginally soft);
      do not score as a clean control. Integrity item for the paper.

## P3 — Layer 2 finite-T core  (needs GPU)  ← METHOD PIVOT
- [x] First attempt: hand-rolled TDEP-lite — FAILED (ill-conditioned, -32..-40 THz). Purged.
- [ ] DECISION: finite-T method = hiPhive (TDEP-style, symmetry-reduced) vs SSCHA (gold std)
- [ ] Implement chosen method; save MD snapshots so re-fits don't re-run MD
- [ ] VERIFY on SrTiO3: soft mode must harden negative->positive across ~105 K transition
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
