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
- [x] Other models harmonic grid — ALL 5 DONE (mace, chgnet, orb_v2, sevennet0, mattersim),
      20 systems each, 0 errors. Per-model venv via `uv venv --system-site-packages` reusing
      /venv/main torch 2.12+cu130; torch constraint file prevents downgrade.
      HEADLINE (KTaO3 excluded, 19 systems): MatterSim & SevenNet acc=1.00 (0 false-stable);
      MACE 0.895, ORB 0.842, CHGNet 0.789 — all three carry 15.4% false-stable rate from
      softening bcc Zr/Hf soft modes to ~0. ORB also uniquely misses SrTiO3 (float32, reads
      -0.0) and falsely calls MgO unstable (-2.8). MatterSim reproduces every soft mode large.
- [ ] Optional: JARVIS reference cross-check (npj-style rates on shared materials)
- CHECKPOINT: 5-model harmonic landed + pushed (32e67b7); KTaO3 integrity fix (2cd4cb6)

## DECISION NEEDED (finite-T core)
- one-shot TDEP (hiPhive) under-detects soft modes -> needs SSCHA or symmetry-broken sampling.
- Options for next session: (A) implement SSCHA finite-T, (B) rattled-start MD symmetry-
  breaking probe, (C) ship harmonic cross-model paper first, finite-T as follow-up.

## NOTE — ground-truth label review
- [x] Reclassified KTaO3 as borderline=true in registry; analysis excludes borderline from
      headline rates (reported as sensitivity probe). 4/5 MLIPs call it imaginary, confirming
      the "stable" label was the questionable one. CeO2/NaCl KEPT as scored controls — they
      are genuinely stable and only CHGNet marginally trips them (-0.24, just past -0.1 tol).

## P3 — Layer 2 finite-T core  (needs GPU)  ← METHOD = 1D QUANTUM SCHA  [VERIFIED]
- [x] First attempt: hand-rolled TDEP-lite — FAILED (ill-conditioned, -32..-40 THz). Purged.
- [x] Second attempt: one-shot hiPhive TDEP-from-MD — under-detects (SrTiO3 +0.8 flat). Kept as
      documented dead-end (compute_finite_t_hiphive).
- [x] Third attempt: rattled-start MD order parameter — FAILED gate (shallow 3.5meV well melts).
- [x] METHOD = `softmode` (compute_finite_t_softmode): relax -> phonopy harmonic FCs -> softest
      COMMENSURATE mode via phonopy modulation (mask Gamma acoustic) -> static E(Q) double well
      -> bounded quartic/sextic fit -> single-mode QUANTUM SCHA free-energy minimisation over the
      order-parameter centroid Q0 (brentq self-consistent sigma). Stable iff cubic (Q0~0) is the
      free-energy global min. E(Q) map is T-independent -> cached json; each T is a CPU solve.
- [x] Wired into cli.py (method "softmode") + run_grid.py default (non-superionic).
- [x] VERIFY on SrTiO3 PASSED (mace_mp0, 2x2x2): harm -2.089 THz; Q0 condensed ~0.14 below Tc,
      melts to 0 at 150 K; eff_freq hardens -2.6 -> +1.8 THz through Tc~100-150 K (expt 105 K).
- [x] Curated set × 5 models × T-ladder (100/300/600/900); 400 softmode ledger rows, 0 errors.
- [x] SOUNDNESS FIX: softmode q-search restricted to FC-commensurate q (den-6-on-2x2x2 was
      interpolating spurious flat-well modes); bcc uses 6x6x6 FCs.
- [x] MULTI-MODE SSCHA added (user direction) for rigorous bcc: python-sscha+cellconstructor,
      ASE-phonons bridge, free-energy Hessian. Validated PbTiO3 (-0.65 THz imaginary). bcc-Zr
      dynamic-stabilization curve, 5 models x 5 T (25 rows): ALL stabilize bcc-Zr by <=50 K,
      margin tracks harmonic depth (mattersim/orb closest to boundary ~0.4 THz @50K, mace
      firmly stable ~1.8). KEY PHYSICS: bcc->hcp is martensitic, so dynamic-stabilization T
      differs from thermodynamic Tc -> bcc gt label (Tc) is the wrong comparison for dynamic
      stability; SSCHA dynamic-stabilization curve is the right bcc deliverable.
- [x] CHECKPOINT: finite-T core complete (softmode primary + SSCHA bcc cross-check).
- [x] SSCHA extended to bcc Ti/Hf (75 rows total, 0 blowups). 3-panel fig_sscha_bcc.
- [x] SSCHA minimizer max_ka cap (fixes 40-atom perovskite step-collapse/timeout; 63c5cc1).
- [x] CROSS-VALIDATION + REFRAME (e241213): SSCHA = bcc gold-standard (tracks softmode rho=0.78);
      FE-perovskite SSCHA fails (false-stable, displacive recall 0.23 vs softmode 0.77; +numerical
      blowups). Root cause diagnosed (ForcePositiveDefinite + low-T narrow Gaussian + v4=False;
      v4=True impractical >18min/unit). User-confirmed REFRAME (option A). analysis: sscha_reliability,
      displacive_recall, family-aware method_agreement. figs: method_agreement (bcc), displacive_recall.
- [ ] Let grid finish (esp untested cubic FLUORITES zro2/hfo2); pull + commit final ledger;
      check if fluorite SSCHA is clean (legit cross-check) or also false-stables; regen figures.

## P4 — Analysis & figures
- [ ] Confusion matrices + false-stable rates
- [ ] Harmonic-vs-finite-T error decomposition (H2)
- [ ] Ensemble-disagreement calibration (H3)

## P5–P7 — Write / review / finalize (academic-pipeline stages 2–6)
- [ ] Stage 2 WRITE → 2.5 INTEGRITY → 3 REVIEW → 4 REVISE → 4.5 FINAL INTEGRITY
- [ ] Stage 5 FINALIZE (PDF) → Stage 6 PROCESS SUMMARY

## Lessons
See tasks/lessons.md.
