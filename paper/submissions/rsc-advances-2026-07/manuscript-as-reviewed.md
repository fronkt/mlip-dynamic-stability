# Finite-temperature dynamic stability is a blind spot of foundation machine-learning interatomic potentials

Frank Cai^a^

^a^ Purdue University, West Lafayette, Indiana, USA. E-mail: frankyc11223@gmail.com.
ORCID: 0009-0003-0041-1459

## Abstract

Foundation (universal) machine-learning interatomic potentials (MLIPs) are now used routinely as
cheap stand-ins for DFT in high-throughput stability screening, but they are benchmarked almost
exclusively against harmonic (0 K) phonons. Dynamic stability is physically a finite-temperature
property, and an important class of materials, including cubic perovskites, bcc refractory metals
and cubic fluorites, is harmonically unstable but thermally stabilised by anharmonicity. We test
five foundation MLIPs (MACE-MP-0, CHGNet, ORB-v2, SevenNet-0, MatterSim) on this regime using a
curated set of references whose finite-temperature behaviour is documented in the experimental and
DFT literature, so that no new DFT ground truth is required. We introduce a cheap single-mode
quantum self-consistent-harmonic (SCHA) "soft-mode free energy" screen that resolves the
harmonic-to-finite-temperature stabilisation, and we cross-validate it against the gold-standard
multi-mode stochastic SCHA (SSCHA). We report three findings together with a practical
ensemble-disagreement guardrail. (i) At the harmonic level the models divide: MatterSim and
SevenNet-0 reproduce every documented soft mode (accuracy 1.00), whereas MACE-MP-0, CHGNet and
ORB-v2 soften the bcc Zr/Hf instabilities toward zero and carry a 15% false-stable rate. (ii)
Harmonic accuracy is necessary but not sufficient for finite-temperature accuracy: the two are only
weakly correlated on the matched set, the harmonic leaders (MatterSim, SevenNet-0) are not the
finite-temperature leaders, and the soft-mode screen recovers the displacive instability of cubic
ferroelectric perovskites that the models otherwise miss. (iii) Multi-mode SSCHA with an MLIP force
engine is itself a methodological trap: it is a clean gold standard for the martensitic bcc metals
but systematically false-stabilises deep displacive (ferroelectric) instabilities and can diverge
numerically, so the cheap soft-mode screen is the more reliable finite-temperature indicator for
the displacive instabilities that dominate generative crystal-structure-prediction (CSP) screening.

## 1. Introduction

Foundation MLIPs (MACE-MP, CHGNet, ORB, SevenNet, MatterSim, M3GNet) are increasingly used as DFT
surrogates in high-throughput pipelines, including the dynamical-stability filtering of generative
CSP outputs. Two 2025 works define the state of the art. "Universal MLIPs are ready for
phonons"^1^ benchmarks MLIPs against ~10⁴ DFT harmonic phonon calculations^3^ on largely
in-distribution materials and reports a systematic potential-energy-surface (PES) softening bias,
in which energy and force under-prediction soften modes and over-estimate stability. PhononBench^2^
scores ~1.1×10⁵ generated structures for harmonic dynamical stability using MatterSim itself as the
oracle.

Both are harmonic, that is, 0 K. Dynamic stability is a finite-temperature property, and a
technologically central materials class is harmonically unstable yet thermally stabilised by
anharmonicity: cubic perovskites (SrTiO₃, BaTiO₃, halide perovskites), bcc refractory metals
(β-Ti/Zr/Hf), and cubic fluorites (ZrO₂/HfO₂). Harmonic phonons assign these imaginary modes, yet
the high-symmetry phase is the equilibrium phase above a transition temperature. Whether foundation
MLIPs reproduce this harmonic-to-finite-temperature stabilisation is essentially un-benchmarked,
even though the harmonic-only oracle that PhononBench and CSP screening rely on is what this regime
breaks.

Research question: do foundation MLIPs reproduce finite-temperature dynamic stability,
specifically the harmonic-unstable to thermally-stabilised transition, or does PES softening make
their stability calls unreliable where the harmonic approximation fails?

We pre-registered three hypotheses.

- H1 (softening to false-stable). PES softening inflates harmonic stability, so MLIPs over-report
  stable with a measurable false-stable rate on harmonically-unstable references. This reproduces
  the literature picture on our model set and anchors the validity of the harness.
- H2 (finite-temperature blind spot, the contribution). Harmonic accuracy does not certify a model
  for finite-temperature screening. It is necessary but not sufficient, and the harmonic leaders
  need not be the finite-temperature leaders. (We test, and reject, the stronger pre-registered
  form "harmonic accuracy is uncorrelated with finite-temperature accuracy"; on the matched set the
  two are weakly positively correlated, see §3.2.)
- H3 (practical guardrail). Inter-model (ensemble) disagreement flags unreliable calls better than
  any single model's self-reported energetics.

Our contributions are a finite-temperature benchmark on the anharmonic regime; a cheap, validated
soft-mode free-energy screen; a cautionary result that the gold-standard SSCHA is itself unreliable
on deep displacive instabilities when driven by an MLIP, which changes how such cross-checks should
be used; and a practical ensemble-disagreement guardrail (H3).

## 2. Methods

### 2.1 Systems and ground truth

The curated set (`configs/curated_systems.yaml`, 20 systems) comprises references whose
harmonic-imaginary and finite-temperature-stable behaviour is documented in the literature, so the
ground-truth label needs no new DFT:

- Oxide displacive/antiferrodistortive perovskites (4): SrTiO₃,^11^ BaTiO₃, PbTiO₃, KNbO₃.^12^
- Halide perovskites (3): CsPbI₃,^13^ CsSnBr₃, CsSnI₃.^14^
- bcc refractory metals (3): Ti, Zr, Hf (phonon-entropy stabilisation).^15^
- Cubic fluorites (2): ZrO₂, HfO₂.^16^
- Superionic (1): α-AgI.^17^
- Quantum-paraelectric near-control (1): KTaO₃.^18^
- Harmonically-stable controls (6): Si, MgO, NaCl, Cu, diamond, CeO₂ (0 K harmonic stability
  documented in the DFPT phonon database^3^).

Transition temperatures are approximate, and scoring is qualitative (the correct side of the
transition, not a fitted T_c). KTaO₃ is flagged borderline (incipient ferroelectric, DFT mode ≈0;
four of five MLIPs call it imaginary) and excluded from headline rates, which leaves 19 scored
systems. CeO₂ and NaCl are retained as scored stable controls, since they are genuinely stable and
only CHGNet marginally trips them (§3.1). The soft-mode screen and the harmonic baseline cover all
20 systems; the SSCHA grid covers the perovskites, fluorites and bcc metals. α-AgI is the one
system where the soft-mode screen is on shaky physical ground, because a superionic with a
diffusive Ag sublattice has no single frozen order parameter; its screen call is reported but
flagged, and it is the natural target for a future symmetry-breaking/MD probe.

### 2.2 Models

MACE-MP-0,^6^ CHGNet,^7^ ORB-v2,^8^ SevenNet-0,^9^ and MatterSim^10^ run through a
backend-agnostic ASE-calculator harness, with one virtual environment per model. Testing MatterSim
directly probes PhononBench's own oracle. ORB-v2 is a direct, float32-only model; we carry this as
an explicit caveat because it manifests in both the harmonic and the finite-temperature results.

### 2.3 Harmonic baseline (credibility anchor)

Finite-displacement harmonic phonons (phonopy, 2×2×2 force-constant supercells) with each MLIP,
classified by the minimum phonon frequency against an imaginary tolerance (default −0.1 THz). The
success criterion is to reproduce the published harmonic stability split. This validates the
harness and is not claimed as novel.

Metrics convention (used throughout). A call is false-stable when a model predicts the
high-symmetry phase dynamically stable where the reference is unstable; this is the
screening-dangerous error, because it lets an unphysical structure through a CSP filter. A call is
false-unstable in the opposite case. Recall (unstable) is the fraction of genuinely-unstable
reference units a method correctly flags as unstable. Rates exclude the borderline KTaO₃.

### 2.4 Finite-temperature soft-mode free energy (primary screen)

For each system we relax the structure, compute harmonic force constants, and identify the softest
force-constant-commensurate **q** by exact `run_qpoints` evaluation over the rational grid the
supercell supports (masking the three acoustic modes at Γ; bcc uses 6×6×6 force constants so the
ω-phase q = ⅔⟨111⟩ is commensurate). We freeze that mode into the minimal commensurate cell via
phonopy modulation, map the static double well E(Q) with quadratic Q sampling, fit the
well-plus-barrier window, and minimise a single-mode quantum SCHA free energy over the
order-parameter centroid (self-consistent Gaussian width via bracketed root finding). The
high-symmetry phase is stable at T when Q₀≈0 is the global minimum of the free energy. The E(Q) map
is temperature-independent and cached, so each temperature is a sub-second CPU solve. Three earlier
finite-temperature routes (hand-rolled TDEP,^5^ one-shot hiPhive, and rattled-MD) were implemented
and discarded after they failed the SrTiO₃ gate; see the ESI.

Validation gate. On SrTiO₃ the order parameter condenses below the transition, melts by ≈150 K, and
the soft mode hardens from −2.6 to +1.8 THz across the transition (experiment 105 K), which
establishes that the screen resolves the harmonic-to-finite-temperature stabilisation.

### 2.5 Multi-mode SSCHA (gold-standard cross-check)

The full stochastic SCHA (python-sscha + cellconstructor)^4^ uses the MLIP as force engine: an ASE
finite-displacement harmonic dynamical matrix, then `ForcePositiveDefinite`, then stochastic SCHA
relaxation of the auxiliary dynamical matrix at T (root2 representation, with a per-population step
cap to force ensemble regeneration on noisy large cells), then a dedicated ensemble at the
converged matrix, and finally the free-energy (physical) Hessian. The minimum Hessian frequency
excluding the three acoustic modes is the anharmonic dynamic-stability indicator. Unlike the
single-mode screen, this captures the full multi-mode phonon entropy and is reliable for
entropy-stabilised martensitic transitions.

## 3. Results

### 3.1 Harmonic baseline: the models divide (H1)

The five models divide on the harmonic set (19 scored systems, KTaO₃ excluded):

| Model | Accuracy | False-stable rate | False-unstable rate |
|---|---|---|---|
| MatterSim | 1.000 | 0.000 | 0.000 |
| SevenNet-0 | 1.000 | 0.000 | 0.000 |
| MACE-MP-0 | 0.895 | 0.154 | 0.000 |
| ORB-v2 | 0.842 | 0.154 | 0.167 |
| CHGNet | 0.789 | 0.154 | 0.333 |

MatterSim and SevenNet-0 reproduce every documented soft mode with large imaginary frequencies.
MACE-MP-0 and CHGNet each carry two false-stable calls (rate 0.154) on the bcc Zr and Hf soft
modes, softened toward zero; this is the PES-softening bias of the literature, localised to
specific instabilities. ORB-v2 also carries two false-stable calls (rate 0.154) but on a different
pair: it correctly flags bcc-Zr unstable (−0.43 THz) yet softens Hf (−0.09 THz) and SrTiO₃
(−0.0 THz) just past the −0.1 THz tolerance. ORB-v2 (direct, float32) softens in both directions;
besides those two false-stables it falsely calls MgO unstable (−2.8 THz), its one false-unstable on
the oxide side. CHGNet carries the only non-ORB false-unstables (rate 0.333), CeO₂ (−0.26 THz) and
NaCl (−0.24 THz), both genuinely stable controls tripped just past the −0.1 THz tolerance; both
flip back to stable at a tolerance of ≈0.25 THz, so they are finite-displacement noise rather than
a real instability.

Because every binary call depends on the imaginary tolerance, we sweep it (Fig. 1). A strict
tolerance (0 THz) floods false-unstables (29 calls) as near-Γ finite-displacement noise dominates,
while a loose tolerance (0.3 THz) inflates false-stables (10 calls); the default −0.1 THz sits in
the stable basin between them (6 false-stable, 3 false-unstable), and the split holds across
0.05–0.2 THz. The useful point is that the softening toward zero is the physics: per-system minimum
frequencies, not binary rates, are the right reporting unit (Fig. 2). This reproduces the published
picture and validates the harness.

![**Fig. 1** Harmonic false-stable and false-unstable call counts versus the
imaginary-frequency tolerance (§3.1). The default −0.1 THz sits in the stable basin between the
false-unstable flood at strict tolerance and the false-stable inflation at loose tolerance.](../results/figures/fig_tolerance_sweep.png)

![**Fig. 2** Per-system minimum effective frequency across the five models (§3.1–3.2).
The softening toward zero of the harmonically-unstable systems is the physics; per-system minima,
not binary rates, are the right reporting unit.](../results/figures/fig_softmode_heat.png)

### 3.2 Finite-temperature soft-mode screen (H2)

On the ferroelectric perovskites at T ≤ 300 K, far below every transition temperature, the cubic
phase is definitively dynamically unstable, and the soft-mode screen correctly calls it unstable in
23 of 30 model units (recall 0.77). The screen recovers the displacive instability that harmonic
accuracy does not predict and that, as §3.3 shows, even the gold-standard SSCHA misses.

Multi-anchor validation against experiment. Beyond the SrTiO₃ gate (§2.4), the screen's predicted
stabilisation temperature T* (the lowest ladder T at which the cubic phase is called stable) tracks
the ordering of the experimental transition temperatures across four anchors. SrTiO₃ (T_c ≈ 105 K)
is stabilised earliest, at T* ≈ 100 K for four of five models, while the ferroelectric perovskites
stabilise later: BaTiO₃ (T_c ≈ 393 K) at T* ≈ 600 K, KNbO₃ (708 K) at 300–600 K, and PbTiO₃
(763 K) at 300–600 K (ORB-v2 the low outlier at 100 K). Within the coarse 300 K T-ladder the screen
brackets the correct side of each transition, and the SrTiO₃-below-ferroelectric ordering is
reproduced by every model. As expected for a single-mode treatment it does not predict the absolute
T_c (it over-shoots BaTiO₃ and under-shoots PbTiO₃), so T* is an ordering check rather than a fitted
T_c (full table S3). This multi-anchor agreement, not a single gate, is what licenses using the
screen as the reference against which the §3.3 SSCHA calls are judged.

Per-model false-stable rates on the displacive/anharmonic set (non-bcc, non-borderline, T ≤ 300 K;
bcc excluded because its thermodynamic-T_c label is the wrong reference for dynamic stability,
§3.3):

| Model | Finite-T false-stable rate | Finite-T accuracy | (Harmonic accuracy) |
|---|---|---|---|
| CHGNet | 0.062 | 0.933 | 0.789 |
| MACE-MP-0 | 0.062 | 0.933 | 0.895 |
| SevenNet-0 | 0.188 | 0.867 | 1.000 |
| MatterSim | 0.250 | 0.833 | 1.000 |
| ORB-v2 | 0.562 | 0.700 | 0.842 |

Read against the harmonic ranking (final column), the message is not a clean inversion, and we are
explicit about this because a naive comparison invites one. Comparing harmonic accuracy including
bcc against finite-temperature accuracy excluding bcc produces a spurious anti-correlation
(Spearman ρ ≈ −0.26); this is a denominator artifact, because CHGNet, MACE and ORB-v2 lose most of
their harmonic accuracy on the bcc Zr/Hf softening that is removed from the finite-temperature
column. On the matched non-bcc, non-borderline set the two are positively but imperfectly
correlated: at the unit level (n = 75 system×model pairs) the concordance of correctness gives
φ = 0.11 (McNemar exact p = 0.34, so the harmonic and finite-temperature error sets are not
significantly different), and the matched per-model accuracy rank correlation is ρ = +0.15
(T ≤ 300 K) to +0.65 (T = 100 K).

The defensible H2 statement is therefore the weaker one that the data support: harmonic accuracy is
necessary but not sufficient for finite-temperature accuracy. Seven of the 71 harmonically-correct
(system, model) units are mis-called at finite temperature, against three the other way, and the
matched table shows that the harmonic leaders are not the finite-temperature leaders: MatterSim and
SevenNet-0 are harmonically perfect yet sit behind MACE-MP-0 and CHGNet on finite-temperature
displacive stability. A top harmonic score on the npj/PhononBench benchmarks therefore does not
identify the best finite-temperature screener, but neither does it actively mislead. ORB-v2 is the
one model that degrades sharply from harmonic to finite temperature (0.867 to 0.667–0.70),
consistent with its float32 direct architecture over-softening the PES, though our design cannot
separate the architecture from the precision. The stronger evidence that finite temperature is a
distinct, harder regime comes not from this ranking comparison but from the displacive recall above
and the SSCHA failure in §3.3.

### 3.3 SSCHA: clean for bcc, a trap for perovskites (the cautionary result)

We ran multi-mode SSCHA across both families to cross-validate the screen. The result is asymmetric
and is itself a central finding. We state at the outset that this concerns SSCHA as it is
realistically deployed with an MLIP force engine for a fixed high-symmetry reference, namely
`ForcePositiveDefinite` initialisation, a 2×2×2 cell, the v4=False free-energy Hessian, and
automatic stochastic relaxation, which is the recipe a practitioner escalating from a cheap screen
would actually run. It is not a claim that the SSCHA formalism is wrong; the proper treatment of a
structural instability (relaxing the centroids into the distorted phase, or the expensive v4
Hessian) answers a different question and is impractical at screening scale (see the root cause
below). The finding is that the default escalation path is a trap.

bcc metals: SSCHA is a clean gold standard, and the screen tracks it. Across Ti/Zr/Hf × 5 models ×
5 temperatures (75 runs) SSCHA produced zero numerical failures, with minimum free-energy-Hessian
frequencies in a physical [0.0, 2.1] THz range. All five models anharmonically stabilise the bcc
phase by ≤50 K (the dynamic-stabilisation temperature, which is far below the thermodynamic
transition because bcc-to-hcp/ω is a martensitic, strain-coupled first-order transition; using the
thermodynamic T_c to label dynamic stability is the wrong comparison and is what makes the models
look false-stable on bcc). The margin to the stability boundary discriminates the models and tracks
each model's harmonic-instability depth: MatterSim and ORB-v2 hug the boundary (~0.4 THz at 50 K),
while MACE-MP-0 is firmly stable (~1.8 THz) (Fig. 3). The cheap soft-mode screen tracks SSCHA on
bcc: across the 45 paired units the rank correlation of the two minimum frequencies is Spearman
ρ = 0.78 with 0.64 sign agreement (Fig. 4). The disagreements are concentrated in ORB-v2's float32
softmode outliers (Ti/Hf near −35 THz, where SSCHA is mildly positive); dropping ORB-v2 raises the
sign agreement to 0.78, while the rank correlation softens to 0.63 as the remaining frequency
spread narrows. Either way the screen and the gold standard agree on the family where the gold
standard is trustworthy.

Ferroelectric perovskites: SSCHA systematically false-stabilises. On the same FE perovskites at
T ≤ 300 K where the screen reaches 0.77 recall, SSCHA correctly identifies the instability in only
7 of 30 units (recall 0.23), and the perovskite SSCHA runs include six numerical blow-ups (minimum
frequencies down to −2×10⁶ THz, concentrated in the float32 ORB-v2 runs). This is the cautionary
result: the expensive gold standard is less reliable than the cheap screen in the displacive regime
that dominates generative-CSP outputs (Fig. 5).

![**Fig. 3** Multi-mode SSCHA dynamic-stabilisation curves for bcc Ti/Zr/Hf, five models,
versus temperature (§3.3). All models stabilise the bcc phase by ≤50 K; the margin to the stability
boundary (MatterSim/ORB-v2 hugging ~0.4 THz, MACE-MP-0 firmly stable ~1.8 THz) discriminates the
models.](../results/figures/fig_sscha_bcc.png)

![**Fig. 4** Soft-mode screen vs gold-standard SSCHA minimum frequency on bcc
(§3.3): the cheap screen tracks SSCHA on the family where the gold standard is
trustworthy.](../results/figures/fig_method_agreement.png)

Root cause. A controlled diagnostic on cubic BaTiO₃ (MACE-MP-0, 100 K) isolates the mechanism. The
harmonic soft mode is −5.6 THz (correctly unstable), but `ForcePositiveDefinite` at initialisation
erases it (+2.88 THz); the SCHA auxiliary matrix then stays at +2.89 THz because at low temperature
its narrow Gaussian width never samples the anharmonic double well; and the free-energy Hessian
without the fourth-order term reproduces that positive curvature (+2.87 THz), giving a false-stable
call. Enabling the fourth-order term (`include_v4=True`) is the only route that could recover the
instability, but it ran for more than 18 min on a single unit without finishing (tens of hours per
unit at grid scale) and is numerically viable only in float64. The physically correct SSCHA
treatment of a deep displacive instability is to relax the structure into the distorted phase,
which changes the question from "is the cubic phase dynamically stable" to "what is the ground
state," and so cannot serve as the benchmark criterion. We therefore use SSCHA as the bcc gold
standard and the experimental transition temperature (the SrTiO₃ gate, §2.4) as the validation for
the perovskite screen.

Cubic fluorites confirm the failure is systematic, not numerical. Cubic ZrO₂ (the >2600 K phase,
harmonically unstable via the X-point oxygen mode at all temperatures studied) is a cleaner test
because its instability is shallower than the FE perovskites and does not trigger the float32
blow-ups. The soft-mode screen returns roughly −6.4 to −8 THz for all five models at every
temperature (100–900 K), correctly and consistently unstable. SSCHA instead returns ≈ +3 THz at
100 K for all five models (false-stable) and then destabilises with temperature (for example
MACE-MP-0 −4.6 THz and ORB-v2 −10.2 THz at 900 K), which is both the wrong sign at low T and the
wrong temperature trend. Because SSCHA fails the same way on a numerically well-behaved, shallower
instability, the false-stable behaviour is intrinsic to the fixed-reference SSCHA deployment (see
the root cause above) rather than an artifact of the deepest perovskite wells. HfO₂ shows the same
SSCHA false-stable pattern (all models +2.0 to +2.6 THz at 100 K, then destabilising to −4.3/−7.2
THz for CHGNet/ORB-v2 by 900 K). Across both fluorites SSCHA produced zero numerical blow-ups yet
the same systematic false-stable, which confirms that the failure is a methodological trap, not a
numerical one.

### 3.4 Ensemble disagreement as a guardrail (H3)

Because no single model is reliable across the set, we test whether cross-model disagreement flags
the units where the majority-vote consensus finite-temperature call is wrong. On the non-bcc,
non-borderline soft-mode units (n = 60; bcc excluded because its thermodynamic-T_c label is the
wrong reference for dynamic stability, §3.3), the majority-vote consensus is wrong on 16.7% of
units. The disagreement signal separates these: on the 18 units where the five models split on the
stable/unstable call, the consensus error rate is 0.39, against 0.07 on the 42 unanimous units, a
5.5× enrichment. As a ranked predictor of consensus error, the binary stable/unstable vote split
reaches AUC 0.75, whereas the continuous cross-model frequency standard deviation is uninformative
(AUC 0.52, no better than chance; Fig. 6).

![**Fig. 6** Ensemble-disagreement guardrail (§3.4): the discrete inter-model
vote split predicts consensus error (AUC 0.75) while the continuous cross-model frequency spread
does not (AUC 0.52).](../results/figures/fig_ensemble_guardrail.png)

This refines H3 into a rule with a caveat: the discrete inter-model vote split is a useful, cheap
guardrail (flag any candidate on which the foundation-MLIP ensemble disagrees), but the continuous
frequency spread that one might naively threshold is not. The physical reading is that disagreement
concentrates near the stability boundary, where split votes coincide with frequencies straddling
zero and the call is both most uncertain and most error-prone.

![**Fig. 5** Recall of the displacive (ferroelectric-perovskite) instability at
T ≤ 300 K: the cheap soft-mode screen (0.77) versus the expensive SSCHA (0.23) (§3.3). The
cautionary result is that the gold standard is less reliable than the screen in the regime that
matters most.](../results/figures/fig_displacive_recall.png)

### 3.5 Robustness: stochastic noise and finite size

Stochastic reproducibility. Repeating the bcc-Zr / MACE-MP-0 / 100 K SSCHA with four independent
random seeds gives a minimum free-energy-Hessian frequency of +1.798 ± 0.001 THz (range [+1.797,
+1.800]). The stochastic noise (≈0.001 THz) is two-to-three orders of magnitude smaller than the
cross-model bcc margins (~0.4 THz spread, §3.3), so those margins are real signal rather than
sampling noise.

Finite size. The stability call is insensitive to supercell size on the cases where the test is
well-posed (`results/convergence_study.parquet`). For bcc-Zr the dynamic-stabilisation verdict
holds from 2×2×2 to 3×3×3 (SSCHA +1.80 to +1.56 THz at 100 K, +1.80 to +1.55 at 300 K; soft-mode
screen +0.92 to +1.58 THz, stable throughout, the margin shifting by ≤0.25 THz). For the Γ-centred
ferroelectric mode of BaTiO₃ the soft-mode screen stays unstable across cell size (−8.5 to −7.3 THz
at 100 K). One caveat is intrinsic to the physics rather than the method: the SrTiO₃
antiferrodistortive instability lives at the zone-boundary R point = (½,½,½), which is commensurate
only with even supercells, so a 3×3×3 cell is blind to it by construction (the screen there reports
the Γ value, +0.0 THz, not the R instability at −2.8). A 2×2×2-versus-3×3×3 comparison is therefore
not a valid convergence test for R-point systems, and the definitive even-cell test (4×4×4, a
~320-atom SSCHA) is left to future work. The §3.3 SSCHA false-stable is not a missing-q artifact:
it occurs for the Γ ferroelectric mode of BaTiO₃, which is present in every cell including the
2×2×2 used, so the failure is the initialisation/Gaussian-width mechanism of the root-cause
diagnostic, not finite size.

## 4. Discussion

The harmonic layer (§3.1) reproduces the literature and confirms H1: PES softening produces a real
false-stable rate, localised to specific instabilities (bcc Zr/Hf) rather than uniform. The
finite-temperature results confirm H2 in a strong form. Harmonic accuracy is non-predictive of
finite-temperature accuracy, and the standard escalation ("if in doubt, run SSCHA") fails on the
displacive regime, because SSCHA driven by an MLIP false-stabilises deep double wells. The
practical recommendation inverts the usual cost/accuracy intuition: for screening
harmonically-unstable displacive candidates, the cheap single-mode soft-mode free energy is the
more reliable indicator, and SSCHA should be reserved for martensitic and entropy-stabilised cases
where its ansatz is sound.

Limitations. Ground-truth transition temperatures are approximate and scoring is qualitative. SSCHA
cells are 2×2×2 (finite size), so dynamic-stabilisation temperatures are approximate, though the
cross-model comparison at fixed cell is valid. ORB-v2's float32-only direct architecture is an
outlier in both layers and is flagged throughout rather than excluded.

## 5. Conclusions

Finite-temperature dynamic stability is a blind spot of foundation MLIPs and of the harmonic
benchmarks used to certify them. A cheap quantum soft-mode free-energy screen recovers the
finite-temperature stabilisation that harmonic accuracy does not predict, and, unlike MLIP-driven
SSCHA, it remains reliable on the deep displacive instabilities that dominate generative-CSP
screening. Harmonic accuracy does not certify a model for finite-temperature use, and neither does
an unexamined SSCHA cross-check.

## Conflicts of interest

There are no conflicts to declare.

## Data availability

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability and archived at Zenodo at
https://doi.org/10.5281/zenodo.20805824. The production results regenerate from
`results/ledger.parquet` (per-unit hashed, resumable) and the finite-size runs from
`results/convergence_study.parquet`; figures via `scripts/make_figures.py`, analysis in
`mlip_dynstab/analysis.py`, the SSCHA root-cause diagnostic in `scripts/sscha_v4_diag.py`, and the
stochastic-reproducibility study (§3.5) via `scripts/sscha_repro.py` (its per-seed frequencies
print to the run log rather than to the ledger).

## Acknowledgements

The author thanks the developers of the foundation interatomic-potential packages benchmarked
here, MACE-MP-0, CHGNet, ORB, SevenNet and MatterSim, and of phonopy, python-sscha /
cellconstructor, ASE and pymatgen, for making their codes and pre-trained models openly available;
this benchmark would not have been possible otherwise. Reference structures were obtained from the
Materials Project. Computational resources were provided by commercial cloud GPU providers. This
research received no specific grant from any funding agency in the public, commercial, or
not-for-profit sectors.

## References

1. A. Loew, D. Sun, H.-C. Wang, S. Botti and M. A. L. Marques, *npj Comput. Mater.*, 2025, **11**, 178.
2. X.-Q. Han, P.-J. Guo, Z.-F. Gao, W.-K. Li and Z.-Y. Lu, *arXiv*, 2025, arXiv:2512.21227.
3. G. Petretto, S. Dwaraknath, H. P. C. Miranda, D. Winston, M. Giantomassi, M. J. van Setten, X. Gonze, K. A. Persson, G. Hautier and G.-M. Rignanese, *Sci. Data*, 2018, **5**, 180065.
4. L. Monacelli, R. Bianco, M. Cherubini, M. Calandra, I. Errea and F. Mauri, *J. Phys.: Condens. Matter*, 2021, **33**, 363001.
5. O. Hellman, I. A. Abrikosov and S. I. Simak, *Phys. Rev. B*, 2011, **84**, 180301(R); O. Hellman, P. Steneteg, I. A. Abrikosov and S. I. Simak, *Phys. Rev. B*, 2013, **87**, 104111.
6. I. Batatia, P. Benner, Y. Chiang, A. M. Elena, D. P. Kovács, J. Riebesell *et al.*, *arXiv*, 2023, arXiv:2401.00096.
7. B. Deng, P. Zhong, K. Jun *et al.*, *Nat. Mach. Intell.*, 2023, **5**, 1031.
8. M. Neumann, J. Gin, B. Rhodes, S. Bennett, Z. Li, H. Choubisa, A. Hussey and J. Godwin, *arXiv*, 2024, arXiv:2410.22570.
9. Y. Park, J. Kim, S. Hwang and S. Han, *J. Chem. Theory Comput.*, 2024, **20**, 4857.
10. H. Yang, C. Hu, Y. Zhou *et al.*, *arXiv*, 2024, arXiv:2405.04967.
11. T. Tadano and S. Tsuneyuki, *Phys. Rev. B*, 2015, **92**, 054301.
12. W. Zhong, D. Vanderbilt and K. M. Rabe, *Phys. Rev. Lett.*, 1994, **73**, 1861.
13. A. Marronnier *et al.*, *ACS Nano*, 2018, **12**, 3477.
14. L. Monacelli and N. Marzari, *Chem. Mater.*, 2023, DOI: 10.1021/acs.chemmater.2c03475.
15. W. Petry *et al.*, *Phys. Rev. B*, 1991, **43**, 10933; A. Heiming *et al.*, *Phys. Rev. B*, 1991, **43**, 10948.
16. K. Parlinski, Z. Q. Li and Y. Kawazoe, *Phys. Rev. Lett.*, 1997, **78**, 4063.
17. D. A. Wood and N. Marzari, *Phys. Rev. B*, 2007, **76**, 134301.
18. A. Ranalli *et al.*, *Adv. Quantum Technol.*, 2023, **6**, 2200131.
