# Process summary (Stage 6)

A wrap-up of how this project went from a fresh idea to a submission-ready paper, including the
methodological decisions, dead-ends, and integrity catches along the way. Companion to
`tasks/todo.md` (checklist) and `tasks/lessons.md` (corrections).

## At a glance

- **Thesis.** Foundation MLIPs are benchmarked almost exclusively against *harmonic* (0 K)
  phonons, but dynamic stability is a *finite-temperature* property. The wedge is the
  harmonically-unstable / thermally-stabilised regime (cubic perovskites, bcc Ti/Zr/Hf, fluorites,
  α-AgI) where the harmonic-only oracle breaks.
- **Deliverables.** Manuscript + ESI (Digital Discovery / RSC format), six inline figures, a
  reproducible per-unit results ledger, open code, and a Zenodo archive
  (https://doi.org/10.5281/zenodo.20805824).
- **Models.** MACE-MP-0, CHGNet, ORB-v2, SevenNet-0, MatterSim. **Systems.** 20 curated
  references (19 scored; KTaO₃ borderline-excluded).

## Pipeline walkthrough

1. **Research brief (Stage 1).** Scoped the gap against the two state-of-the-art harmonic
   benchmarks (npj "ready for phonons"; PhononBench) and pre-registered three hypotheses
   (H1 softening→false-stable; H2 finite-T blind spot; H3 ensemble guardrail). `docs/research_brief.md`.

2. **Layer 1 — harmonic baseline (credibility anchor).** Finite-displacement phonons (phonopy +
   MLIP). Reproduced the literature split: MatterSim & SevenNet-0 perfect (acc 1.00); MACE, ORB-v2,
   CHGNet carry a 15.4% false-stable rate, localised to softening bcc Zr/Hf toward zero. Validated
   the harness; **0 errors across 5 models × 20 systems**.

3. **Layer 2 — finite-T method development.** Three routes were built and **discarded** after
   failing the SrTiO₃ validation gate: (a) hand-rolled TDEP-lite (ill-conditioned), (b) one-shot
   hiPhive (under-detects), (c) rattled-MD order parameter (shallow well melts). The surviving
   method = **single-mode quantum SCHA "soft-mode free energy"**: relax → softest commensurate mode
   → static E(Q) double well → 1-D self-consistent quantum SCHA free energy; cubic stable ⇔ Q₀≈0 is
   the global minimum. The static E(Q) map is cached and T-independent, so each temperature is a
   sub-second CPU solve. Passed the SrTiO₃ gate (soft mode hardens −2.6 → +1.8 THz through ~100–150 K;
   expt 105 K).

4. **Cross-check — multi-mode SSCHA.** python-sscha + cellconstructor with MLIP forces. A minimizer
   step-cap (`max_ka=20`) was needed to stop perovskite runs thrashing. The major finding emerged
   here (see below).

5. **Analysis & figures.** Confusion matrices / false-stable rates, the H2 paired test, the H3
   guardrail, method-agreement, displacive recall, SSCHA reliability, and a tolerance sweep — all
   in `mlip_dynstab/analysis.py`; six figures via `scripts/make_figures.py`.

6. **Write → review → revise (Stages 2–4).** Drafted manuscript + ESI, ran an internal referee
   report (major revision), and addressed every data-grounded and compute item (multi-anchor T*
   validation, 20-system methods reconciliation, SSCHA deployment-scoping, tolerance sweep,
   finite-size + reproducibility runs).

7. **Integrity pass (Stage 2.5/4.5).** Full manuscript-vs-ledger read-through (see below).

8. **DOI pass.** 18 web-verified references built in: methods/benchmarks, the five model papers,
   and the per-system anharmonic-stabilisation backbone (the evidentiary core a referee asked for).

9. **Finalize (Stages 5–6).** RSC formatting (superscript citations, RSC reference list, inline
   Fig. 1–6, author block, Conflicts, Data availability, Acknowledgements, ESI), DOCX deliverables,
   Zenodo deposit, cover letter, and this summary.

## Key results

- **H1.** PES softening produces a real, *localised* false-stable rate (bcc Zr/Hf), not a uniform
  bias. ORB-v2 (direct, float32) softens broadly in both directions.
- **H2 (reframed honestly).** Harmonic accuracy is *necessary but not sufficient* for finite-T
  accuracy; the harmonic leaders (MatterSim, SevenNet-0) are **not** the finite-T leaders (MACE,
  CHGNet). The soft-mode screen recovers the displacive instability the models otherwise miss
  (recall 0.77).
- **The headline cautionary result.** MLIP-driven SSCHA is a **trap**: clean gold standard for
  martensitic bcc (tracks the screen, ρ = 0.78) but systematically *false-stabilises* deep
  displacive instabilities (recall 0.23 vs the screen's 0.77), with a controlled root-cause
  diagnostic. The cheap screen is the more reliable indicator in the regime that matters most for
  generative CSP.
- **H3.** Discrete inter-model vote disagreement flags bad consensus calls (AUC 0.75); the
  continuous frequency spread does not (AUC 0.52).

## Decisions & dead-ends worth remembering

- Three finite-T routes were purged before the soft-mode method worked — the failures are
  documented in the ESI as guidance for others building MLIP-driven finite-T screens.
- KTaO₃ was reclassified **borderline** (incipient ferroelectric, DFT mode ≈0) and excluded from
  headline rates — an integrity call so models aren't penalised on an ill-defined ground truth.
- The bcc thermodynamic T_c is the **wrong** reference for *dynamic* stability (martensitic,
  strain-coupled); SSCHA dynamic-stabilisation curves are the right bcc deliverable.
- A conceptual "Figure 1" overview schematic was built (Excalidraw) and then **removed** as
  visually unprofessional for a journal figure; the six data figures lead instead.

## Integrity catches (the main quality value-add)

- **H2 "ranking inversion" overturned.** An eye-catching "finite-T inverts the harmonic ranking,
  ρ = −0.26" was a **denominator artifact** (harmonic-with-bcc vs finite-T-without-bcc). On the
  matched set the correlation is weakly *positive* (φ = 0.11, McNemar p = 0.34). H2 was reframed to
  the defensible "necessary not sufficient" form.
- **Stage-2.5 read-through fixes.** Caught an ORB false-stable misattribution (its two are Hf +
  SrTiO₃, not Zr/Hf — it catches Zr), a conflated method-agreement statistic, prose T_c values that
  had drifted off the registry (BaTiO₃ 393, KNbO₃ 708), and a PhononBench structure count
  (1.1×10⁵, not 1.3×10⁵).
- Rule distilled: never compare rates on different system subsets; match the denominator first.
  Numbers in prose must match the config/ledger they are scored against.

## Reproducibility

Every reported number regenerates from `results/ledger.parquet` (per-unit hashed, resumable) and
`results/convergence_study.parquet`; analysis in `mlip_dynstab/analysis.py`, figures via
`scripts/make_figures.py`, the SSCHA root-cause diagnostic in `scripts/sscha_v4_diag.py`. The whole
repository is archived at Zenodo (https://doi.org/10.5281/zenodo.20805824).

## Status & next

- **Submission-ready** for *Digital Discovery* (RSC): best fit + fast (~46 d to first decision).
  Cover letter drafted (`paper/cover_letter.md`) leading with the SSCHA-trap result.
- **Anticipated decision:** Major Revision. The strongest referee risk is the "you ran SSCHA the
  deliberately-wrong way" critique; the §3.3 scoping (we benchmark the *realistic escalation path*,
  validate the screen against experiment not against SSCHA, and show the failure is methodological
  via the clean fluorites) is the prepared defense.
- **Optional future work:** a 4×4×4 (~320-atom) even-cell SrTiO₃ SSCHA convergence test for the
  R-point instability.
