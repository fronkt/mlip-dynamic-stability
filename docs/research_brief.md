# Stage 1 Research Brief — Finite-Temperature Dynamic-Stability Stress Test of Foundation MLIPs

*Status: draft for user confirmation (MANDATORY Stage-1 checkpoint). Bibliography DOIs to be
finalized by the deep-research pass; claims below are stated at the precision the evidence
currently supports and flagged where a citation is still pending.*

## 1. Motivation & gap

Foundation (universal) machine-learning interatomic potentials — MACE-MP, CHGNet, ORB,
SevenNet, MatterSim, M3GNet — are increasingly used as cheap stand-ins for DFT in
high-throughput stability screening, including dynamical-stability filtering of
generative-CSP outputs. Two 2025 works define the current state of the art:

- **"Universal MLIPs are ready for phonons"** (npj Comput Mater, 2025): benchmarks MLIPs
  against ~10⁴ DFT **harmonic** phonon calculations on (largely in-distribution) materials;
  reports dynamical-stability classification (e.g. MACE-MP-0 ≈95% true-positive on stable,
  ≈73% true-negative on unstable; MatterSim most accurate). Notes a systematic PES-softening
  bias (energy/force under-prediction → mode softening → over-estimated stability).
- **PhononBench** (arXiv:2512.21227, Dec 2025): scores ~1.3×10⁵ **generated** structures for
  **harmonic** dynamical stability **using MatterSim as the oracle**; no DFT false-positive/
  false-negative rates, no finite-temperature treatment.

**The gap.** Both are harmonic (0 K). But dynamical stability is physically a
**finite-temperature** property. A materials class of high technological and scientific
importance is **harmonically unstable yet thermally stabilized by anharmonicity**:
cubic perovskites (SrTiO₃, BaTiO₃, halide perovskites), bcc refractory metals (β-Ti/Zr/Hf),
cubic fluorites (ZrO₂/HfO₂), and superionics (α-AgI). Harmonic phonons assign these
imaginary modes; the high-symmetry phase is nonetheless the equilibrium phase above a
transition temperature. Whether foundation MLIPs reproduce this **harmonic→finite-T
stabilization** is essentially un-benchmarked — even though the harmonic-only oracle that
PhononBench and CSP screening pipelines rely on is exactly what this regime breaks.

## 2. Research question & hypotheses

**RQ.** Do foundation MLIPs correctly reproduce finite-temperature dynamic stability —
specifically the harmonic-unstable → thermally-stabilized transition — or does PES softening
make their stability calls unreliable where the harmonic approximation fails?

- **H1 (softening → false-stable).** PES softening inflates harmonic stability: MLIPs
  over-report stable, with a measurable *false-stable* rate on harmonically-unstable
  references (reproduces npj/lit on our model set; the harness-validity anchor).
- **H2 (finite-T blind spot — the contribution).** Harmonic error is *not predictive* of
  finite-T error. On the curated anharmonic set, MLIPs misjudge the finite-T call (wrong
  sign/magnitude of T-dependence) in a way uncorrelated with their harmonic accuracy — so
  harmonic benchmarks (npj, PhononBench) do not certify a model for finite-T screening.
- **H3 (practical guardrail).** Inter-model (ensemble) disagreement on the soft-mode
  frequency flags unreliable finite-T calls better than any single model's self-reported
  energetics — an actionable screening rule.

## 3. Design — two staged layers

**Layer 1 — harmonic baseline (credibility anchor, fast).** Finite-displacement harmonic
phonons (phonopy) with each MLIP on a public DFT-phonon set (phonondb / Petretto MP phonons /
JARVIS). Success = reproduce published harmonic stability rates within tolerance; this
validates the harness and is *not* claimed as novel.

**Layer 2 — finite-T anharmonic stress test (novel core).** For ~20–35 curated references
(see `configs/curated_systems.yaml`), compute finite-T effective phonons with each MLIP and
classify dynamical stability vs the literature ground truth:
- **TDEP-lite** (primary, implemented): NVT MD at T → least-squares effective harmonic
  force constants → min effective frequency. Cheap, robust, per-T.
- **SSCHA** (cross-check on a subset): the gold-standard anharmonic free-energy-Hessian
  criterion (incl. quantum effects); confirms the TDEP-lite calls on textbook cases.
- **MD symmetry-breaking probe**: for diffusive/superionic systems where effective-FC
  fitting is ill-posed.

**Models.** MACE-MP-0, CHGNet, ORB (v2/v3), SevenNet-0, MatterSim; optional GRACE,
eqV2-OMat24. (Testing MatterSim directly probes PhononBench's own oracle.)

## 4. Metrics

- Harmonic & finite-T **confusion matrices** vs ground truth; **headline = false-stable
  rate** (screening-dangerous error) and false-unstable rate.
- **Harmonic-vs-finite-T error decomposition** (tests H2): correlation between a model's
  harmonic min-frequency error and its finite-T call correctness.
- Soft-mode min-frequency MAE; qualitative transition-temperature accuracy (does the
  predicted stable/unstable boundary bracket the known transition T?).
- **Ensemble-disagreement calibration** (tests H3): does cross-model variance predict error?

## 5. Curated set rationale

Chosen because each has *documented* harmonic-imaginary / finite-T-stable behavior in the
DFT/experimental literature, so the ground-truth label needs no new DFT:
perovskites (displacive/AFD/tilt soft modes), bcc refractory metals (phonon-entropy
stabilization), fluorites (high-T cubic), α-AgI (superionic), plus harmonically-stable
controls (Si, MgO, NaCl, Cu, diamond) and a near-control (KTaO₃, quantum paraelectric) to
probe the stable/soft boundary. Full list with transition temperatures and references in the
config.

## 6. Threats to validity (pre-registered)

- **Finite-displacement noise near Γ** → acoustic-sum-rule correction + a swept imaginary
  tolerance (default −0.1 THz); stability calls reported as a function of tolerance.
- **TDEP-lite vs SSCHA discrepancy** → SSCHA cross-check on ≥3 textbook cases (incl. SrTiO₃)
  before trusting TDEP-lite labels at scale.
- **MLIP-relaxation moving off the soft-mode geometry** → record both at-reference and
  at-MLIP-relaxed geometries; relaxation hiding instability is itself a reportable finding.
- **Supercell-size / MD-length convergence** → convergence sub-study on SrTiO₃ before the
  full grid; settings frozen and hashed into every ledger row.
- **Ground-truth uncertainty** → transition temperatures are approximate; scoring is
  qualitative (correct side of the transition), not a fitted T_c.

## 7. Deliverable & target venue

A benchmark + failure-mode taxonomy + a practical guardrail (H3), with the harmonic layer as
the validation anchor. Target: RSC *Digital Discovery* or *npj Computational Materials*
(evaluation-methodology contribution; matches the venues for the two competitor works).

## 8. Key references (to finalize in bibliography)

1. "Universal MLIPs are ready for phonons", npj Comput Mater (2025) —
   https://www.nature.com/articles/s41524-025-01650-1
2. PhononBench, arXiv:2512.21227 (2025) — https://arxiv.org/abs/2512.21227
3. Petretto et al., "High-throughput DFPT phonons", Sci. Data (2018) — MP phonon reference set.
4. SSCHA — Monacelli et al., J. Phys. Condens. Matter (2021), python-sscha.
5. TDEP — Hellman et al., Phys. Rev. B (2011, 2013).
6. MLIP backends: MACE-MP-0, CHGNet, ORB, SevenNet, MatterSim (cite each model paper).
7. PES-softening in uMLIPs (M3GNet/CHGNet/MACE) — to source during deep-research.
*(Deep-research Stage 1 will verify each DOI and add the anharmonic-stabilization primary
literature per curated system.)*
