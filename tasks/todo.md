# mlip-dynamic-stability — phase checklist & resume state

> Resume protocol: on a new session, read this file + `results/ledger.parquet`. The current
> checkpoint is the lowest unchecked box. Per-unit work (system, model, T, method) is
> idempotent — re-running skips tuples already in the ledger.

## CURRENT STATUS (2026-08-16) — NOT SUBMITTABLE, primary-screen layer must be re-run

**Stage:** pre-submission audit FAILED. See `tasks/audit-2026-08-16.md` for the full record and
`tasks/todo-archive-2026-08-16.md` for the superseded "SUBMISSION-READY" snapshot.

Three defects verified from the ledger and the source (all fixed in code at `87a9183`; the DATA is
not fixed):
- **D1** SSCHA dropped the 3 *lowest* modes as "acoustic" instead of the 3 nearest zero, deleting
  the instability. 13/208 rows, all manufactured false-stables. FE recall 0.23 → **0.30**, fluorite
  0.17 → **0.23**, **bcc unchanged (0/75)** — the §3.3 headline survives, restated.
- **D2** The same inversion at Γ in `_softest_mesh_mode` masked the T1u ferroelectric triplet, so Γ
  could never win the q-search. §3.5's "Γ ferroelectric mode of BaTiO₃" claim is false.
- **D3** **220/340 non-bcc softmode rows carry q-denominators the deposited code cannot produce.**
  The v1 grid was built by the "D6-on-2×2×2" search that `e592e86` fixed as unsound; because
  `settings` carried only the supercell, the unit hash never changed and every stale unit was
  skipped as "already present". Now fixed by folding `METHOD_VERSION` into the hash + cache key.

**Novelty framing is also dead as written.** "Essentially un-benchmarked" / "blind spot" are
falsified by two verified 2025 papers: PCCP 28, 4459 (arXiv:2510.18178, MACE variants on halide
double perovskites) and J. Phys. Chem. C doi:10.1021/acs.jpcc.5c07541 ("Are Foundational Atomistic
Models Reliable for Finite-Temperature MD?", on PbTiO₃ — one of our own systems). Surviving claim:
cross-FAMILY finite-T dynamic-stability classification across five architectures and four chemistry
families with a free-energy criterion and a T-ladder.

**DONE since the audit (all pushed):**
1. ✅ Screen redefined: evaluates EVERY imaginary commensurate mode; phase unstable if ANY
   condenses (`METHOD_VERSION['softmode']=3`, cap 24 folded into the unit hash). The old
   softest-mode criterion inspected the Γ FE mode in SrTiO₃, which is quantum-suppressed, instead
   of the R-point tilt that drives the 105 K transition.
2. ✅ Full re-run: 5 models × 20 systems × 4 T = 400 units, 0 errors, one pinned env per model,
   `pip freeze` locks in `envs/lock-*-2026-08-16.txt`, 220 cached E(Q) maps committed.
3. ✅ SSCHA D1 correction applied to the ledger (`scripts/fix_sscha_acoustic.py`).
4. ✅ Bianco PRB 96, 014111 and Monacelli PRB 112, 014109 read and quoted; §3.3 reframed around the
   bubble truncation. Φ is positive-definite *by construction*, so its positivity was never a
   finding.
5. ✅ Retitled; novelty framing rewritten against the two verified 2025 papers.
6. ✅ All 6 figures regenerated; every dependent number propagated through manuscript,
   supplementary, cover letter, `.zenodo.json`; DOCX rebuilt. `analysis.canonical` added so a
   figure can never mix method generations.

**COMPLETED 2026-08-17 (the five submission blockers, all pushed):**
1. ✅ §2.4 derivation written: Peierls bound → displaced Gaussian trial state → coth width →
   moment identities → self-consistency, each equation matching the code; Table 1 = six-row
   approximation ledger with measured pointers (fit-window sweep in ESI: 0.5–1.3% of 756 calls).
2. ✅ Sign convention fixed FOR REAL: reported frequency = symmetric-point free-energy curvature
   (softmode v4), the honest analog of the SSCHA Hessian; argmin call unchanged; `n_curv_blind`
   records first-order-like condensations the curvature cannot see (SrTiO₃ R-tilt: +0.92 THz
   while condensing at 100 K — the same blindness that afflicts fixed-reference SSCHA).
3. ✅ Harmonic v2 re-measured in pinned envs (gate: MACE/BaTiO₃ = deposited −6.4113 exactly).
   Four models reproduce v1; ORB-v2 0.842→0.895. §3.1+abstract updated.
4. ✅ SSCHA v3 re-measured: 201/208 units, phonopy-built initialiser (ase 3.29 killed the
   cellconstructor ASE bridge; conversion validated to 0.0003 THz), pinned envs. FE recall 0.19
   vs screen 0.53; 7 deepest-well units fail loudly (v1: silent −2×10⁶ blow-ups); bcc 75/75
   clean, call agreement 0.78/0.83; zr/MACE +1.80 reproduces the v1 seed study.
5. ✅ Zenodo v2.0.0 minted: 10.5281/zenodo.21978533; concept DOI 10.5281/zenodo.20805799 (cited
   in the manuscript) resolves to it. verify_claims.py: 27 assertions, all pass.

**REMAINING (non-blocking, future work):**
- One `include_v4=True` run to completion on a BaTiO₃ unit (inference → direct measurement).
- 3×3×3 soft-mode convergence re-run (marked superseded in §3.5); 4×4×4 SrTiO₃ SSCHA.
- SUBMIT to Digital Discovery. For Yau: research video + 500–1500 word acknowledgments with AI
  disclosure still owed (see chat log for the FAQ requirements).

**Process/progress:**
- Data COMPLETE: ledger.parquet = harmonic (100) + softmode (400, 20 sys × 5 models × 4 T) +
  SSCHA (211: bcc 75, fluorite 40, perovskite 93). Convergence/repro runs in
  `results/convergence_study.parquet` (kept out of the production ledger).
- Paper drafted: `paper/manuscript.md`, `paper/supplementary.md`, `paper/review.md` (referee
  report = major revision; all data-grounded items addressed; compute items M5/m6 done).
- 6 figures regenerate from the ledger via `scripts/make_figures.py`.

**Headline findings (final framing):**
1. H1 — harmonic model split: MatterSim/SevenNet acc 1.00; MACE/CHGNet/ORB carry 15%
   false-stable from softening bcc Zr/Hf. (CHGNet's CeO2/NaCl false-unstables are marginal,
   flip at ~0.25 THz tolerance.)
2. H2 — harmonic accuracy is NECESSARY NOT SUFFICIENT for finite-T (the earlier "inversion
   rho=-0.26" was a denominator artifact; matched set is weakly POSITIVE phi=0.11). Harmonic
   leaders (MatterSim/SevenNet) are NOT the finite-T leaders (MACE/CHGNet best on displacive).
3. SSCHA cautionary result (most novel): clean gold-standard for bcc (tracks softmode rho=0.78,
   0 blowups) but systematically FALSE-STABLES displacive instabilities (FE perovskites recall
   0.23 vs softmode 0.77; fluorites also false-stable though numerically clean). Root cause =
   ForcePositiveDefinite + low-T narrow Gaussian + v4=False; v4=True impractical (>18min/unit).
   Scoped to "SSCHA as deployed with MLIP force engine for a fixed reference."
4. H3 guardrail — ensemble vote-split flags consensus error (AUC 0.75, 5.5x enrichment);
   continuous freq-spread does not (AUC 0.52).
- Robustness: SSCHA repro std 0.001 THz (<< margins); bcc call cell-robust; SrTiO3 R-point
  needs even cells (2x2x2<->3x3x3 invalid for it; 4x4x4 = future work).

**NEXT STEPS (priority order):**
1. Stage 2.5 INTEGRITY — read-through verifying every manuscript number vs the ledger.
2. Deep-research DOI pass (referee m8) — per-system anharmonic-stabilization citations (the
   evidentiary backbone of the ground-truth labels) are still placeholders.
3. Stage 5 FINALIZE — render manuscript+SI to DOCX/PDF; Stage 6 process summary.
4. Optional/future: 4x4x4 SrTiO3 SSCHA even-cell convergence; α-AgI symmetry-breaking probe.

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
- [x] Full SSCHA grid incl. cubic FLUORITES (zro2/hfo2): numerically clean (0 blowups) but still
      false-stable at low T -> confirms cautionary result is methodological, not numerical.
- [x] max_ka cap fix for perovskite minimizer step-collapse; SSCHA seed param added.

## P4 — Analysis & figures  [DONE]
- [x] Confusion matrices + false-stable rates (per_model_table, low_t_false_stable ±bcc)
- [x] Harmonic-vs-finite-T decomposition H2 (h2_paired_summary: phi/McNemar, matched denominator)
- [x] Ensemble-disagreement calibration H3 (h3_guardrail_summary: AUC 0.75 vote-split)
- [x] Extras: sscha_reliability, displacive_recall, harmonic_tolerance_sweep, method_agreement
- [x] 6 figures: sscha_bcc, softmode_heat, method_agreement, displacive_recall,
      ensemble_guardrail, tolerance_sweep

## P5–P7 — Write / review / finalize (academic-pipeline stages 2–6)
- [x] Stage 2 WRITE — manuscript.md + supplementary.md
- [x] Stage 3 REVIEW — review.md (internal referee report, major revision)
- [x] Stage 4 REVISE — all data-grounded items + compute items (M5/m6) addressed
- [x] Stage 2.5 / 4.5 INTEGRITY — full number-vs-ledger read-through DONE (2026-06-22). Verified
      ~30 quantitative claims against ledger.parquet + convergence_study.parquet; ~26 exact. Fixed:
      (1) §3.1 ORB false-stable MISATTRIBUTION — ORB's 2 false-stables are Hf(-0.09)+SrTiO3(-0.0),
      NOT "Zr/Hf"; it correctly catches Zr(-0.43). (2) §3.3 method-agreement GARBLE — "Spearman
      rho=0.78 rising to 0.78 excl-ORB" conflated two stats: full-set Spearman=0.78/sign-agree=0.64;
      excl-ORB the SIGN AGREEMENT rises to 0.78 while Spearman DROPS to 0.63. (3) §3.2 Tc prose
      vs registry: BaTiO3 403->393, KNbO3 676->708. (4) seed-repro (+1.798±0.001) is print-to-log
      only, not in any parquet -> softened "all results regenerate from ledger" in manuscript +
      supplementary. Minor: ZrO2 softmode "-7 to -8"->"-6.4 to -8"; HfO2 "exactly"->"same SSCHA
      pattern"; PbTiO3 T* added ORB outlier note. Stale "Stage 2 / [PENDING]" header refreshed.
- [x] Deep-research DOI pass (referee m8) DONE (2026-06-22). 18 web-verified references built into
      manuscript "References": methods [1-5], 5 model papers [6-10], per-system finite-T ground-truth
      backbone [11-18] (SrTiO3 Tadano PRB92; BaTiO3/KNbO3/PbTiO3 Zhong-Vanderbilt-Rabe PRL73;
      CsPbI3 Marronnier ACS Nano12; CsSnI3/Br Chem Mater 2023; bcc Ti/Zr/Hf Petry+Heiming PRB43;
      ZrO2/HfO2 Parlinski PRL78 + CGD 2023; AgI Wood-Marzari PRB76; KTaO3 Ranalli AdvQT6). Controls
      via DFPT DB [3]. Inline cites added in intro/2.1/2.2/2.4/2.5. NO recalled DOIs - every DOI
      confirmed via WebSearch. Also fixed PhononBench count 1.3e5->1.1e5 (actual 108,843). research_
      brief.md sec 8 + curated_systems pointer closed out.  ← NEW INTEGRITY FIX folded in
- [x] Figure 1 conceptual overview schematic (4cf542a) — excalidraw-diagram skill, src in
      paper/fig1_src/; embedded as lead figure + referenced from intro.
- [~] Stage 5 FINALIZE — DOCX DONE (manuscript.docx + supplementary.docx via pandoc, figures
      embedded). PDF-via-Word COM hangs against the user's open Word session; DOCX is the
      deliverable (all target journals accept Word). PDF optional later.
- [x] Journal targeting (academic-paper-reviewer): target = Digital Discovery (RSC); prestige
      reach = npj Comput Mater. Requirements/formatting notes captured in chat + memory.
- [x] Fig 1 schematic REMOVED (user rejected as unprofessional); data figs renumbered Fig. 1–6.
- [x] Digital Discovery formatting DONE (faef42e): RSC superscript citations + RSC reference list,
      author/affiliation block (placeholder), Conflicts, Data availability + Zenodo, ESI; DOCX
      regenerated; .zenodo.json added.
- [x] Author set (sole: Frank Cai, Purdue, ORCID 0009-0003-0041-1459) in manuscript + .zenodo.json.
- [x] Zenodo archive live (v1.0.0): https://doi.org/10.5281/zenodo.20805824, inserted into Data
      availability. Figures inline Fig. 1–6. SUBMISSION-READY for Digital Discovery.
- [ ] Optional: Stage 6 process summary; submit to Digital Discovery; 4×4×4 SrTiO3 SSCHA convergence.

## Lessons
See tasks/lessons.md.
