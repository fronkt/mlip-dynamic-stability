# Finite-temperature dynamic stability is a blind spot of foundation machine-learning interatomic potentials

*Working draft — Stage 2. Numbers are pulled from `results/ledger.parquet`; cells marked
`[PENDING]` await the in-flight cubic-fluorite SSCHA grid. DOIs to be finalised in the
deep-research pass.*

## Abstract

Foundation (universal) machine-learning interatomic potentials (MLIPs) are now routine cheap
stand-ins for DFT in high-throughput stability screening, yet they are benchmarked almost
exclusively against **harmonic** (0 K) phonons. Dynamic stability is physically a
**finite-temperature** property, and an important class of materials — cubic perovskites, bcc
refractory metals, cubic fluorites — is *harmonically unstable but thermally stabilised by
anharmonicity*. We stress-test five foundation MLIPs (MACE-MP-0, CHGNet, ORB-v2, SevenNet-0,
MatterSim) on this regime using a curated set of references whose finite-T behaviour is
documented in the experimental/DFT literature, so no new DFT ground truth is required. We
introduce a cheap single-mode quantum self-consistent-harmonic (SCHA) "soft-mode free energy"
screen that resolves the harmonic→finite-T stabilisation, and we cross-validate it against the
gold-standard multi-mode stochastic SCHA (SSCHA). Three findings: (i) at the harmonic level the
models split sharply — MatterSim and SevenNet-0 reproduce every documented soft mode (accuracy
1.00), while MACE-MP-0, CHGNet and ORB-v2 soften the bcc Zr/Hf instabilities toward zero and
carry a 15% false-stable rate; (ii) the finite-T behaviour is *not* certified by harmonic
accuracy, and the soft-mode screen recovers the displacive instability of cubic ferroelectric
perovskites that the models otherwise miss; (iii) **multi-mode SSCHA with an MLIP force engine
is itself a sharp methodological trap** — it is a clean gold standard for the martensitic bcc
metals but systematically *false-stabilises* deep displacive (ferroelectric) instabilities and
can diverge numerically, so the cheap soft-mode screen is the more reliable finite-T indicator
in exactly the regime that matters most for generative-CSP screening.

## 1. Introduction

Foundation MLIPs — MACE-MP, CHGNet, ORB, SevenNet, MatterSim, M3GNet — are increasingly used
as DFT surrogates in high-throughput pipelines, including the dynamical-stability filtering of
generative crystal-structure-prediction (CSP) outputs. Two 2025 works define the state of the
art. "Universal MLIPs are ready for phonons" (npj Comput. Mater., 2025) benchmarks MLIPs
against ~10⁴ DFT **harmonic** phonon calculations on largely in-distribution materials and
reports a systematic potential-energy-surface (PES) softening bias (energy/force
under-prediction → mode softening → over-estimated stability). PhononBench (arXiv:2512.21227,
Dec 2025) scores ~1.3×10⁵ generated structures for **harmonic** dynamical stability using
MatterSim itself as the oracle.

Both are harmonic, i.e. 0 K. But dynamic stability is a finite-temperature property, and a
technologically central materials class is **harmonically unstable yet thermally stabilised by
anharmonicity**: cubic perovskites (SrTiO₃, BaTiO₃, halide perovskites), bcc refractory metals
(β-Ti/Zr/Hf), cubic fluorites (ZrO₂/HfO₂). Harmonic phonons assign these imaginary modes, yet
the high-symmetry phase is the equilibrium phase above a transition temperature. Whether
foundation MLIPs reproduce this **harmonic → finite-T stabilisation** is essentially
un-benchmarked — even though the harmonic-only oracle that PhononBench and CSP screening rely on
is exactly what this regime breaks.

**Research question.** Do foundation MLIPs correctly reproduce finite-temperature dynamic
stability — specifically the harmonic-unstable → thermally-stabilised transition — or does PES
softening make their stability calls unreliable where the harmonic approximation fails?

We pre-registered three hypotheses:

- **H1 (softening → false-stable).** PES softening inflates harmonic stability: MLIPs
  over-report stable, with a measurable false-stable rate on harmonically-unstable references.
  This reproduces the npj/literature picture on our model set and anchors harness validity.
- **H2 (finite-T blind spot — the contribution).** Harmonic accuracy is *not predictive* of
  finite-T accuracy; harmonic benchmarks do not certify a model for finite-T screening.
- **H3 (practical guardrail).** Inter-model (ensemble) disagreement flags unreliable calls
  better than any single model's self-reported energetics.

Our contribution is threefold: a finite-T benchmark on the anharmonic regime; a cheap,
validated soft-mode free-energy screen; and a cautionary methodological result that the
"gold-standard" SSCHA is itself unreliable on deep displacive instabilities when driven by an
MLIP, which reframes how such cross-checks should be used.

## 2. Methods

### 2.1 Systems and ground truth

The curated set (`configs/curated_systems.yaml`) comprises references whose
harmonic-imaginary / finite-T-stable behaviour is documented in the literature, so the
ground-truth label needs no new DFT: displacive/antiferrodistortive perovskites (SrTiO₃,
BaTiO₃, PbTiO₃, KNbO₃, CsSnI₃), bcc refractory metals (Ti, Zr, Hf; phonon-entropy
stabilisation), cubic fluorites (ZrO₂, HfO₂), the quantum-paraelectric near-control KTaO₃, and
harmonically-stable controls (Si, MgO, NaCl, Cu, diamond, CeO₂). Transition temperatures are
approximate; scoring is qualitative — the correct side of the transition — not a fitted T_c.
KTaO₃ is flagged borderline (incipient ferroelectric, DFT mode ≈0) and excluded from headline
rates.

### 2.2 Models

MACE-MP-0, CHGNet, ORB-v2, SevenNet-0, MatterSim, run through a backend-agnostic ASE-calculator
harness (one virtual environment per model). Testing MatterSim directly probes PhononBench's
own oracle. ORB-v2 is a direct, float32-only model; we carry this as an explicit caveat
because it manifests in both the harmonic and finite-T results.

### 2.3 Harmonic baseline (credibility anchor)

Finite-displacement harmonic phonons (phonopy, 2×2×2 force-constant supercells) with each MLIP,
classified by the minimum phonon frequency against an imaginary tolerance (default −0.1 THz).
Success criterion: reproduce the published harmonic stability split. This validates the harness
and is not claimed as novel.

### 2.4 Finite-T soft-mode free energy (primary screen)

For each system we relax the structure, compute harmonic force constants, and identify the
softest force-constant-commensurate **q** by exact `run_qpoints` evaluation over the rational
grid the supercell supports (masking the three acoustic modes at Γ; bcc uses 6×6×6 force
constants so the ω-phase q = ⅔⟨111⟩ is commensurate). We freeze that mode into the minimal
commensurate cell via phonopy modulation, map the static double well E(Q) with quadratic Q
sampling, fit the well-plus-barrier window, and minimise a **single-mode quantum SCHA free
energy** over the order-parameter centroid (self-consistent Gaussian width via bracketed root
finding). The high-symmetry phase is stable at T iff Q₀≈0 is the free-energy global minimum.
The E(Q) map is temperature-independent and cached, so each temperature is a sub-second CPU
solve. Earlier finite-T routes (hand-rolled TDEP, one-shot hiPhive, rattled-MD) were
implemented and discarded after failing the SrTiO₃ gate; see SI.

**Validation gate.** On SrTiO₃ the order parameter condenses below the transition, melts by
≈150 K, and the soft mode hardens from −2.6 to +1.8 THz across the transition (experiment
105 K), establishing that the screen resolves the harmonic→finite-T stabilisation.

### 2.5 Multi-mode SSCHA (gold-standard cross-check)

The full stochastic SCHA (python-sscha + cellconstructor) with the MLIP as force engine:
ASE finite-displacement harmonic dynamical matrix → `ForcePositiveDefinite` → stochastic SCHA
relaxation of the auxiliary dynamical matrix at T (root2 representation; per-population step cap
to force ensemble regeneration on noisy large cells) → a dedicated ensemble at the converged
matrix → the free-energy (physical) Hessian. The minimum Hessian frequency excluding the three
acoustic modes is the anharmonic dynamic-stability indicator. Unlike the single-mode screen,
this captures the full multi-mode phonon entropy and is reliable for entropy-stabilised
martensitic transitions.

## 3. Results

### 3.1 Harmonic baseline — the models split (H1)

The five models split sharply on the harmonic set (19 scored systems, KTaO₃ excluded):

| Model | Accuracy | False-stable rate | False-unstable rate |
|---|---|---|---|
| MatterSim | 1.000 | 0.000 | 0.000 |
| SevenNet-0 | 1.000 | 0.000 | 0.000 |
| MACE-MP-0 | 0.895 | 0.154 | 0.000 |
| ORB-v2 | 0.842 | 0.154 | 0.167 |
| CHGNet | 0.789 | 0.154 | 0.333 |

MatterSim and SevenNet-0 reproduce every documented soft mode with large imaginary
frequencies. MACE-MP-0, CHGNet and ORB-v2 each carry two false-stable calls (rate 0.154),
driven by **softening the bcc Zr/Hf soft modes toward zero** — the PES-softening bias of the
literature, localised to specific instabilities. ORB-v2 (direct, float32) softens broadly: it
uniquely reads SrTiO₃ at −0.0 and falsely calls MgO unstable (−2.8 THz). The actionable point
is that the "softening toward zero" is the physics; per-system minimum frequencies, not binary
rates, are the right reporting unit (Fig. `fig_softmode_heat`). This reproduces the published
picture and validates the harness.

### 3.2 Finite-T soft-mode screen — the headline (H2)

On the ferroelectric perovskites at T ≤ 300 K — far below every transition temperature, so the
cubic phase is **definitively dynamically unstable** — the soft-mode screen correctly calls the
cubic phase unstable in **23 of 30** model units (recall 0.77). This is the contribution: the
screen recovers the displacive instability that harmonic accuracy does not predict and that, as
§3.3 shows, even the gold-standard SSCHA misses. [Per-model finite-T false-stable rates over
the full ladder are reported in Table S2; we caution that the bcc entries there compare against
the *thermodynamic* transition temperature, which is the wrong reference for *dynamic*
stability — see §3.3.]

### 3.3 SSCHA — clean for bcc, a trap for perovskites (the cautionary result)

We ran multi-mode SSCHA across both families to cross-validate the screen. The result is
asymmetric and is itself a central finding.

**bcc metals — SSCHA is a clean gold standard, and the screen tracks it.** Across Ti/Zr/Hf ×
5 models × 5 temperatures (75 runs) SSCHA produced **zero** numerical failures, with minimum
free-energy-Hessian frequencies in a physical [0.0, 2.1] THz range. All five models
anharmonically stabilise the bcc phase by ≤50 K (the dynamic-stabilisation temperature, which
is far below the thermodynamic transition because bcc→hcp/ω is a martensitic, strain-coupled
first-order transition — using the thermodynamic T_c to label *dynamic* stability is therefore
the wrong comparison and is what makes the models look "false-stable" on bcc). The margin to
the stability boundary discriminates the models and tracks each one's harmonic-instability
depth: MatterSim and ORB-v2 hug the boundary (~0.4 THz at 50 K), MACE-MP-0 is firmly stable
(~1.8 THz). Critically, the cheap soft-mode screen **tracks SSCHA on bcc** (Spearman ρ = 0.78,
sign agreement 0.64; Fig. `fig_method_agreement`), validating the screen on the family where
the gold standard is trustworthy.

**Ferroelectric perovskites — SSCHA systematically false-stabilises.** On the same FE
perovskites at T ≤ 300 K where the screen achieves 0.77 recall, SSCHA correctly identifies the
instability in only **7 of 30** units (recall 0.23), and the perovskite/fluorite SSCHA runs
include six numerical blow-ups (minimum frequencies down to −2×10⁶ THz, concentrated in the
float32 ORB-v2 runs). The contrast is the cautionary result: the expensive gold standard is
*less* reliable than the cheap screen in exactly the displacive regime that dominates
generative-CSP outputs (Fig. `fig_displacive_recall`).

**Root cause.** A controlled diagnostic on cubic BaTiO₃ (MACE-MP-0, 100 K) isolates the
mechanism. The harmonic soft mode is −5.6 THz (correctly unstable), but `ForcePositiveDefinite`
at initialisation erases it (+2.88 THz); the SCHA auxiliary matrix then stays stuck at +2.89
THz because at low temperature its narrow Gaussian width never samples the anharmonic
double-well; and the free-energy Hessian without the fourth-order term echoes that positive
curvature (+2.87 THz) → false-stable. Enabling the fourth-order term (`include_v4=True`) is the
only route that could recover the instability, but it ran >18 min on a *single* unit without
finishing (≈tens of hours per unit at grid scale) and is numerically viable only in float64.
The physically correct SSCHA treatment of a deep displacive instability is to relax the
*structure* into the distorted phase — which changes the question from "is the cubic phase
dynamically stable" to "what is the ground state," and so cannot serve as the benchmark
criterion. We therefore use SSCHA as the bcc gold standard and the experimental transition
temperature (the SrTiO₃ gate, §2.4) as the validation for the perovskite screen.

*Cubic fluorites (ZrO₂/HfO₂): SSCHA cross-check `[PENDING]` — these have a shallower X-point
instability than the FE perovskites and may fall on the clean side of the SSCHA boundary; the
in-flight grid will resolve this.*

### 3.4 Ensemble disagreement as a guardrail (H3)

`[PENDING — to be computed on the completed grid: does cross-model variance in the soft-mode
frequency predict per-unit error better than any single model's self-reported energetics?]`

## 4. Discussion

The harmonic layer (§3.1) reproduces the literature and confirms H1: PES softening produces a
real false-stable rate, localised to specific instabilities (bcc Zr/Hf) rather than uniform.
The finite-T results confirm H2 in a strong form: not only is harmonic accuracy non-predictive
of finite-T accuracy, but the standard escalation — "if in doubt, run SSCHA" — *fails* on the
displacive regime, because SSCHA driven by an MLIP false-stabilises deep double wells. The
practical recommendation inverts the usual cost/accuracy intuition: for screening
harmonically-unstable displacive candidates, the cheap single-mode soft-mode free energy is the
more reliable indicator, and SSCHA should be reserved for martensitic/entropy-stabilised cases
where its ansatz is sound.

**Limitations.** Ground-truth transition temperatures are approximate and scoring is
qualitative. SSCHA cells are 2×2×2 (finite-size), so dynamic-stabilisation temperatures are
approximate though the cross-model comparison at fixed cell is valid. ORB-v2's float32-only
direct architecture is an outlier in both layers and is flagged throughout rather than
excluded.

## 5. Conclusion

Finite-temperature dynamic stability is a blind spot of foundation MLIPs and of the harmonic
benchmarks used to certify them. A cheap quantum soft-mode free-energy screen recovers the
finite-T stabilisation that harmonic accuracy does not predict, and — unlike MLIP-driven SSCHA
— remains reliable on the deep displacive instabilities that dominate generative-CSP screening.
Harmonic accuracy does not certify a model for finite-T use, and neither does an unexamined
SSCHA cross-check.

## Data and code availability

All results regenerate from `results/ledger.parquet` (per-unit hashed, resumable). Figures via
`scripts/make_figures.py`; analysis in `mlip_dynstab/analysis.py`; the SSCHA root-cause
diagnostic in `scripts/sscha_v4_diag.py`. Repository: github.com/fronkt/mlip-dynamic-stability.

## Key references (to finalise)

1. "Universal MLIPs are ready for phonons", npj Comput. Mater. (2025).
2. PhononBench, arXiv:2512.21227 (2025).
3. Petretto et al., high-throughput DFPT phonons, Sci. Data (2018).
4. Monacelli et al., the SSCHA, J. Phys. Condens. Matter (2021).
5. Hellman et al., TDEP, Phys. Rev. B (2011, 2013).
6. Model papers: MACE-MP-0, CHGNet, ORB, SevenNet-0, MatterSim.
