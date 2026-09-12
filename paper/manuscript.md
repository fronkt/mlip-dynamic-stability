# Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation machine-learning interatomic potential for finite-temperature dynamic stability

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
DFT literature, so that no new DFT ground truth is required. We introduce a cheap quantum
self-consistent-harmonic (SCHA) "soft-mode free energy" screen that evaluates every imaginary
commensurate mode and calls the high-symmetry phase unstable if any of them condenses, and we
cross-validate it against the gold-standard multi-mode stochastic SCHA (SSCHA). We report three
findings together with a practical ensemble-disagreement guardrail. (i) At the harmonic level the
models divide: MatterSim and SevenNet-0 reproduce every documented soft mode (accuracy 1.00),
whereas MACE-MP-0 and CHGNet soften the bcc Zr/Hf instabilities to zero (15% false-stable rate)
and ORB-v2 misses the SrTiO₃ soft mode (8%). (ii) Harmonic correctness does not transfer to the finite-temperature layer: on the matched set,
17 harmonically-correct units are mis-called at 300 K against only 4 the other way (McNemar exact
p = 0.007). CHGNet illustrates it, being the worst model harmonically yet second best at finite
temperature while MatterSim is harmonically perfect and only mid-table, though with fifteen
systems the per-model intervals overlap and we treat the finite-temperature ranking as suggestive
rather than established. We do not claim the stronger reading that harmonic accuracy is
uncorrelated with finite-temperature accuracy: once pairs are clustered by system no association
is significant at any temperature, and failing to reject that null is not evidence for it. Screening every
imaginary mode rather than the softest one is essential, because the deepest mode need not be the
one that condenses: in SrTiO₃ the Γ ferroelectric mode is deeper than the R-point tilt yet is
quantum-suppressed, and only the tilt drives the 105 K transition. (iii) Multi-mode SSCHA with an MLIP force
engine is a clean gold standard for the martensitic bcc metals, but in its default deployment, with
the free-energy Hessian truncated at the bubble, it systematically false-stabilises deep displacive
(ferroelectric) instabilities and can diverge numerically. Theory predicts this truncation to fail
for cubic metal halide perovskites; we measure the failure at benchmark scale and show it extends to
oxide perovskites and, free of stochastic blow-ups, to cubic fluorites. The cheap soft-mode screen
is therefore the more reliable finite-temperature indicator for the displacive instabilities that
dominate generative crystal-structure-prediction (CSP) screening.

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
the high-symmetry phase is the equilibrium phase above a transition temperature, and the
harmonic-only oracle that PhononBench and CSP screening rely on is what this regime breaks. Two 2025
studies probe finite-temperature reliability directly: four MACE foundation variants were
benchmarked for the dynamic stability of halide double perovskites,^19^ and the
ferroelectric-to-paraelectric transition of PbTiO₃ was used to expose a disconnect between static
accuracy and dynamic reliability.^20^ Both are confined to a single model family or a single
system. What remains untested is whether the effect is architecture-general and chemistry-general:
no study has classified finite-temperature dynamic stability across several independent MLIP
architectures and across the distinct anharmonic families, that is displacive and
antiferrodistortive oxide perovskites, halide perovskites, entropy-stabilised bcc refractory metals
and cubic fluorites, under a common free-energy criterion and a temperature ladder.

A separate line of work shows that machine-learned potentials can describe exactly the systems
we test, provided they are trained for them. On-the-fly active learning with Bayesian error
estimation has reproduced the entropy-driven phase transitions of hybrid perovskites,^27^ the
temperature-driven transitions and anharmonic thermal transport of zirconia,^28^ and the alpha-beta
transition of zirconium,^29^ which are respectively the perovskite, fluorite and bcc families of
our set. These are system-specific potentials, actively trained on configurations drawn from the
target's own dynamics, and their success sharpens rather than weakens the question here: the gap
we measure is a property of *foundation* models used as shipped, not of machine-learned potentials
in general. Active learning has also been applied directly to the anharmonic failure mode, with a
screen over more than a hundred materials identifying those where MLIP molecular dynamics gives
unphysical dynamic behaviour unless uncertainty is monitored.^30^ That literature descends from
Gaussian-process potentials^31^ and their sparse variants,^37^ and from on-the-fly Bayesian force
fields whose own predictive variance decides when new reference data are needed,^32^ and it supplies the natural alternative
reliability criterion to ours, namely committee or ensemble uncertainty propagated into the
simulation.^33^ We return to that comparison in §3.4 and §4.

The screening context is the other half of the motivation. Universal potentials now drive
high-throughput stability searches at the scale of millions of candidates,^34,35^ and the
community benchmark for machine-learned stability prediction evaluates precisely the model class
tested here.^36^ That benchmark, like the phonon benchmarks above, scores stability against 0 K
convex-hull and harmonic criteria. The distance between that notion of stability and the
finite-temperature dynamic stability a screened candidate actually needs is the gap this work
measures.

Research question: do foundation MLIPs reproduce finite-temperature dynamic stability,
specifically the harmonic-unstable to thermally-stabilised transition, or does PES softening make
their stability calls unreliable where the harmonic approximation fails?

We pre-registered three hypotheses.

- H1 (softening to false-stable). PES softening inflates harmonic stability, so MLIPs over-report
  stable with a measurable false-stable rate on harmonically-unstable references. This reproduces
  the literature picture on our model set and anchors the validity of the harness.
- H2 (the finite-temperature gap, the contribution). Harmonic accuracy does not certify a model
  for finite-temperature screening, and the harmonic ranking need not carry over. (The
  pre-registered form was "harmonic accuracy is uncorrelated with finite-temperature accuracy". The
  matched-set outcome is that harmonic correctness does not transfer — significantly so at 300 K, as
  a difference in layer difficulty — while the direction of any residual association is unresolved
  at this sample size, so the pre-registered form is neither cleanly confirmed nor cleanly rejected;
  see §3.2.)
- H3 (practical guardrail). Inter-model (ensemble) disagreement flags unreliable calls better than
  any single model's self-reported energetics.

Our contributions are a finite-temperature benchmark on the anharmonic regime spanning five
independent MLIP architectures and four anharmonic chemistry families; a cheap, validated
free-energy screen whose criterion is the phase rather than a single mode; a cautionary result that
MLIP-driven SSCHA at its default bubble truncation false-stabilises deep displacive instabilities,
which confirms and extends a published theoretical prediction and changes how such cross-checks
should be used; and a practical ensemble-disagreement guardrail (H3).

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

For each system we relax the structure, compute harmonic force constants, and enumerate **every
distinct imaginary mode** on the force-constant-commensurate **q** grid by exact `run_qpoints`
evaluation over the rational grid the supercell supports (bcc uses 6×6×6 force constants so the
ω-phase q = ⅔⟨111⟩ is commensurate). The three acoustic branches at Γ are removed as the branches
*nearest zero in magnitude*, not the three lowest: when the high-symmetry phase is unstable the soft
mode lies below the acoustic zeros, so masking the lowest three would delete the very instability
being sought. Modes are deduplicated over degenerate branches and symmetry-equivalent **q**, so a
triply degenerate T1u triplet and the three arms of a ⟨100⟩ star each cost one E(Q) map rather than
three. Each surviving mode is frozen into its minimal commensurate cell via phonopy modulation; we
map the static double well E(Q) with quadratic Q sampling, fit the well-plus-barrier window, and
minimise a single-mode quantum SCHA free energy over the order-parameter centroid (self-consistent
Gaussian width via bracketed root finding).

Variational free energy per mode. Each frozen mode defines a one-dimensional subsystem with
coordinate Q along the unit displacement pattern **u**, effective mass
M = Σᵢ mᵢ|**u**ᵢ|², and potential V(Q) = aQ² + bQ⁴ + cQ⁶ from the fit above; the Hamiltonian is
H = P²/2M + V(Q). We treat it with the self-consistent harmonic approximation in its original
single-mode form,^23,24^ built on the Peierls variational bound^25,26^

$$F \le \mathcal{F}(Q_0,\Omega;T) = F_0(\Omega,T) + \langle V \rangle_{Q_0,\sigma} - \tfrac{1}{2} M\Omega^2\sigma^2,$$

where the trial state is the thermal density of a harmonic oscillator of frequency Ω displaced to
centroid Q₀, F₀(Ω,T) = k_BT ln[2 sinh(ħΩ/2k_BT)] is its free energy, and the last term removes the
trial potential counted in F₀. The trial density is a Gaussian of mean Q₀ and quantum width

$$\sigma^2 = \frac{\hbar}{2M\Omega}\coth\!\left(\frac{\hbar\Omega}{2k_BT}\right),$$

which carries the nuclear quantum effects: at high T it recovers the classical k_BT/MΩ², and at
T → 0 it retains the zero-point width that suppresses condensation in shallow wells (quantum
paraelectricity). The Gaussian expectation of the even sextic follows from the moments
⟨Q²⟩ = Q₀² + σ², ⟨Q⁴⟩ = Q₀⁴ + 6Q₀²σ² + 3σ⁴, ⟨Q⁶⟩ = Q₀⁶ + 15Q₀⁴σ² + 45Q₀²σ⁴ + 15σ⁶. Stationarity
of 𝓕 with respect to Ω at fixed Q₀ gives the self-consistency condition

$$M\Omega^2 = \langle V''\rangle_{Q_0,\sigma} = 2a + 12b\,\langle Q^2\rangle + 30c\,\langle Q^4\rangle,$$

which we solve by bracketed root finding on σ² (avoiding the runaway large-σ fixed point a damped
iteration can reach), and 𝓕 is then minimised over the centroid on a Q₀ grid.

Criterion and observable. Two distinct quantities come out of 𝓕, and we keep them separate. The
**stability call** is the variational one: the mode has condensed at T if the global minimum of
𝓕(Q₀; T) sits at Q₀ > 0 — by the bound above, the displaced state then has the lower free energy.
The high-symmetry structure is dynamically unstable at T if **any** screened mode condenses, and
stable only when none does; dynamical stability is a property of the phase, not of one mode.
The **reported frequency** is the curvature of the free energy at the symmetric point,
ω_eff = sign(𝓕″(0)) · [|𝓕″(0)|/M]^½, the single-mode analogue of the SSCHA free-energy Hessian^21^
and a genuine signed observable (𝓕 is even in Q₀, so the curvature is evaluated by a symmetric
finite difference). The two can disagree, and the disagreement is physics rather than noise: for a
deep double well the variational transition is first-order-like, so 𝓕(0) remains a *local* minimum
(positive curvature) while a displaced minimum drops below it. A curvature criterion is blind to
that condensation by construction — the same blindness that afflicts any free-energy Hessian
evaluated at a fixed high-symmetry reference — and we record every such mode (`n_curv_blind` in
the ledger) rather than folding it into either number.

Screening only a single mode — even the globally softest — conflates physically distinct
instabilities. Cubic SrTiO₃ is the decisive case: its Γ ferroelectric mode is *deeper* than the
R-point antiferrodistortive tilt, yet the Γ mode is quantum-suppressed and never condenses while
the R tilt is what drives the 105 K transition, so a softest-mode screen inspects the wrong mode
and calls the cubic phase stable. The E(Q) maps are temperature-independent and cached, so each
temperature is a sub-second CPU solve over all modes. A cap of 24 modes per unit bounds the cost;
it binds on 20 of the 400 units, every one of which is already called unstable with at least three
condensing modes, so it cannot affect any call (a cap can only ever create a false-*stable*, which
would require all 24 screened modes to be non-condensing). Three earlier finite-temperature routes
(hand-rolled TDEP,^5^ one-shot hiPhive, and rattled-MD) were implemented and discarded after they
failed the SrTiO₃ gate; see the ESI. The variational criterion is not interchangeable with a
direct thermal-density criterion, and ESI §S1.4 makes the distinction concrete by solving the
same fitted potentials exactly: an isolated mode's thermal density stays bimodal at every
temperature, because it tends to exp(−V/k_BT), so the temperature dependence the screen needs
comes from the self-consistency rather than from the shape of the well. That comparison also
shows the screen never condenses a mode whose exact density is unimodal, in none of the 228
mode-temperature evaluations covering 57 (system, model) units.

Approximations, declared. Table 1 states what the screen neglects, the expected direction of the
bias, and where the consequence is visible in our own data.

**Table 1** The screen's approximation ledger.

| # | Approximation | What it excludes | Expected bias | Where it shows |
|---|---|---|---|---|
| A1 | One mode at a time | Mode–mode coupling and cooperative condensation | Either sign; absolute T* unreliable | PbTiO₃ T* ordering failure (§3.2) |
| A2 | Commensurate **q** only | Instabilities at incommensurate or finer-grid **q** | False-stable if the true soft mode is missed | R point needs even cells (§3.5); bcc ω needs 6×6×6 FCs |
| A3 | Sextic fit on a well+barrier window | Steep-wall anharmonicity beyond Q⁶ | Barrier-shape error at large Q | Fit window sensitivity (ESI) |
| A4 | Gaussian (SCHA) trial state | Tunnelling, non-Gaussian density; renders deep-well transitions first-order-like | Curvature misses condensation | `n_curv_blind` rows; §3.3 mechanism |
| A5 | Static E(Q) in a clamped cell | Thermal expansion and strain coupling | Under-stabilises martensitic systems | bcc labelled via SSCHA instead (§3.3) |
| A6 | Mode cap (24) | Modes beyond the 24 most imaginary | Could only create false-stables | Binds on 20/400 units, all already unstable (§2.4) |

Validation gate. Cubic SrTiO₃ resolves three distinct imaginary commensurate modes, and the screen
reports which of them condenses. At 100 K the Γ ferroelectric mode does **not** condense (Q₀ = 0),
which is the physically correct quantum-paraelectric result, while the R = (½,½,½) tilt does
(Q₀ = 0.14 Å); by 300 K neither condenses. The phase is therefore called unstable at 100 K and
stable at 300 K, bracketing the experimental transition at 105 K. The gate also exercises the
criterion/observable distinction above: at 100 K the condensing tilt still has a *positive*
symmetric-point curvature (+0.92 THz, hardening to +1.43 THz at 300 K), so this is a
first-order-like condensation that the free energy's argmin detects and its curvature does not —
the same mechanism, inside the cheap screen, that defeats the fixed-reference SSCHA Hessian in
§3.3. Three of the five models (MACE-MP-0, MatterSim, SevenNet-0) reproduce the R-tilt
condensation; CHGNet never condenses the tilt and ORB-v2 does not select the R point at all.
Because the gate identifies *which* instability each model captures rather than returning a single
number, it is a per-model test rather than a single-model demonstration.

### 2.5 Multi-mode SSCHA (gold-standard cross-check)

The full stochastic SCHA (python-sscha + cellconstructor)^4^ uses the MLIP as force engine: an ASE
finite-displacement harmonic dynamical matrix, then `ForcePositiveDefinite`, then stochastic SCHA
relaxation of the auxiliary dynamical matrix at T (root2 representation, with a per-population step
cap to force ensemble regeneration on noisy large cells), then a dedicated ensemble at the
converged matrix, and finally the free-energy (physical) Hessian. The minimum Hessian frequency
excluding the three acoustic modes is the anharmonic dynamic-stability indicator. Unlike the
free-energy screen, which treats each imaginary mode independently, this captures the coupled
multi-mode phonon entropy and is reliable for
entropy-stabilised martensitic transitions.

## 3. Results

### 3.1 Harmonic baseline: the models divide (H1)

The five models divide on the harmonic set (19 scored systems, KTaO₃ excluded):

| Model | Accuracy | False-stable rate | False-unstable rate |
|---|---|---|---|
| MatterSim | 19/19 = 1.000 [0.832, 1.000] | 0/13 = 0.000 [0.000, 0.228] | 0/6 = 0.000 [0.000, 0.390] |
| SevenNet-0 | 19/19 = 1.000 [0.832, 1.000] | 0/13 = 0.000 [0.000, 0.228] | 0/6 = 0.000 [0.000, 0.390] |
| MACE-MP-0 | 17/19 = 0.895 [0.686, 0.971] | 2/13 = 0.154 [0.043, 0.422] | 0/6 = 0.000 [0.000, 0.390] |
| ORB-v2 | 17/19 = 0.895 [0.686, 0.971] | 1/13 = 0.077 [0.014, 0.333] | 1/6 = 0.167 [0.030, 0.564] |
| CHGNet | 15/19 = 0.789 [0.567, 0.915] | 2/13 = 0.154 [0.043, 0.422] | 2/6 = 0.333 [0.097, 0.700] |

Rates are given as the count over its denominator with a Wilson score interval, here and
throughout. On nineteen systems those intervals are wide and they overlap heavily, so the
harmonic *ordering* below is the reportable finding and the individual accuracies are not
separated by this design. Per-model confusion matrices are in Table S4.

MatterSim and SevenNet-0 reproduce every documented soft mode with large imaginary frequencies.
MACE-MP-0 and CHGNet each carry two false-stable calls (rate 0.154) on the bcc Zr and Hf soft
modes, softened to −0.00 THz; this is the PES-softening bias of the literature, localised to
specific instabilities. ORB-v2 carries one false-stable (rate 0.077): it correctly flags bcc-Zr
(−0.43 THz) and bcc-Hf (−0.19 THz) unstable but reads SrTiO₃ at +0.00 THz. ORB-v2 (a direct-force
model, run in its default float32 precision) softens in both directions; besides that false-stable
it falsely calls MgO unstable (−1.07 THz), its one false-unstable on the oxide side. CHGNet carries the only non-ORB false-unstables (rate 0.333), CeO₂ (−0.27 THz) and
NaCl (−0.24 THz), both genuinely stable controls tripped just past the −0.1 THz tolerance; both
flip back to stable at a tolerance of ≈0.25 THz, so they are finite-displacement noise rather than
a real instability.

Because every binary call depends on the imaginary tolerance, we sweep it (Fig. 1). A strict
tolerance (0 THz) floods false-unstables (29 calls) as near-Γ finite-displacement noise dominates,
while a loose tolerance (0.3 THz) inflates false-stables (7 calls); the default −0.1 THz sits in
the stable basin between them (5 false-stable, 3 false-unstable), and the split holds across
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
phase is definitively dynamically unstable, and the multi-mode screen correctly calls it unstable in
16 of 30 model units (recall 0.533 [0.361, 0.698]). The screen recovers displacive instability
that harmonic accuracy does not anticipate and that, as §3.3 shows, the gold-standard SSCHA
misses more often still.
Across the other anharmonic families the screen is stronger: 23/25 = 0.920 [0.750, 0.978] on the
halide perovskites and 20/20 = 1.000 [0.839, 1.000] on the cubic fluorites, while on the
antiferrodistortive SrTiO₃ tilt it reaches 3/5 = 0.600 [0.231, 0.882] on a denominator too small
to support any comparison. On the six harmonically-stable controls it is correct on all 120 model
units (0/120, Wilson [0.000, 0.031]) and finds zero imaginary commensurate modes in every case.
One qualification belongs with that number: the harmonic classifier reads an interpolated
12×12×12 mesh while the screen reads only the **q** commensurate with the force-constant cell,
so the three marginal harmonic false-unstables (ORB-v2 on MgO at −1.07 THz, CHGNet on CeO₂ and
NaCl) have no commensurate counterpart for the screen to find. The control result is therefore
in part a consequence of approximation A2, and we report it as a consistency check rather than
as proof that the screen cannot manufacture an instability.

Multi-anchor comparison against experiment. Beyond the SrTiO₃ gate (§2.4) we compare the screen's
predicted stabilisation temperature T* (the lowest ladder T at which the cubic phase is called
stable) with the experimental transition temperatures. The comparison is only partly successful and
we report it as such. SrTiO₃ (T_c ≈ 105 K) stabilises earliest, at T* = 100–300 K, and BaTiO₃
(393 K) and KNbO₃ (708 K) follow at T* = 300 K for four of five models (ORB-v2 at 600 K), so the
SrTiO₃-below-ferroelectric ordering holds. PbTiO₃ (763 K), which has the highest transition
temperature of the four, does **not** follow: T* is 100 K for MACE-MP-0, MatterSim and ORB-v2,
600 K for SevenNet-0 and 900 K for CHGNet. A single-mode treatment is not expected to reproduce an
absolute T_c, but the PbTiO₃ failure is a genuine ordering failure rather than a scale error, and it
is the clearest limitation of the screen in this work. T* is therefore reported as a diagnostic
(Table S6 of the ESI), and the SrTiO₃ gate together with the control performance — not the T* ordering —
is what licenses the screen as the reference in §3.3.

Per-model false-stable rates on the displacive/anharmonic set (non-bcc, non-borderline, T ≤ 300 K;
bcc excluded because its thermodynamic-T_c label is the wrong reference for dynamic stability,
§3.3):

| Model | Finite-T false-stable rate | Finite-T accuracy | (Harmonic accuracy) |
|---|---|---|---|
| SevenNet-0 | 2/16 = 0.125 [0.035, 0.360] | 27/30 = 0.900 [0.744, 0.965] | 1.000 |
| CHGNet | 3/16 = 0.188 [0.066, 0.430] | 26/30 = 0.867 [0.703, 0.947] | 0.789 |
| MatterSim | 4/16 = 0.250 [0.102, 0.495] | 25/30 = 0.833 [0.664, 0.927] | 1.000 |
| MACE-MP-0 | 4/16 = 0.250 [0.102, 0.495] | 25/30 = 0.833 [0.664, 0.927] | 0.895 |
| ORB-v2 | 5/16 = 0.312 [0.142, 0.556] | 24/30 = 0.800 [0.627, 0.905] | 0.895 |

**These five accuracies are not separated by this design.** Every interval overlaps every
other, the whole spread is six units wide on a denominator of thirty, and we therefore treat
the finite-temperature ranking as suggestive rather than established. What the table is used
for below is the *joint* pattern with the harmonic column — a model can be strong in one layer
and unremarkable in the other — and not the position of any model within either column.

Read against the harmonic ranking (final column), the ordering is not preserved but neither is it
inverted, and we are explicit about this because a naive comparison invites a cleaner story than the
data support. The matched non-bcc, non-borderline comparison (n = 75 system×model pairs, clustered as five
models over fifteen systems) is temperature-dependent, and we separate two questions that a single
2×2 table invites conflating. The first is whether the two layers are equally hard, a question of
marginal homogeneity that McNemar's exact test addresses. At T = 300 K they are not: 17
harmonically-correct units are mis-called at finite temperature against only 4 the other way,
McNemar exact p = 0.007. Harmonic correctness therefore does not transfer, and this is the
significant result. The second question is whether harmonic and finite-temperature correctness are
*associated*, which McNemar does not test. There the effect is negative but not significant:
φ = −0.129 at 300 K, with a system-clustered permutation p = 0.31 over 10,000 permutations of whole
systems' finite-temperature rows, and a system-clustered bootstrap 95% interval of [−0.195, −0.057]
— an interval that reflects the precision of the point estimate rather than a test of the null, and
whose sign rides on a structurally empty both-incorrect cell. At T = 100 K the association is weakly
positive and likewise not significant (φ = +0.149, clustered permutation p = 0.12, McNemar exact
p = 0.73, matched per-model rank correlation ρ = +0.65). With five models and fifteen systems this
design cannot resolve the sign of the association, and we do not claim it. The transferable finding
is the failure of transfer itself. (The matched set excludes systems whose name carries the bcc
tag, which also removes the superionic AgI unit; the denominator is therefore fifteen systems, not
sixteen.)

Because the reordering rests on harmonic calls, we bound the harmonic estimator's own
reproducibility noise directly rather than assuming it is small. The v1 and v2 harmonic generations
(§2.3) are the same algorithm at the same settings — 0.01 Å displacements, 2×2×2 supercells,
12×12×12 meshes — re-measured in independently pinned environments, giving 100 paired (system,
model) re-measurements. The spread is strongly model-dependent and must not be pooled: CHGNet's
largest deviation is 0.0076 THz and MatterSim's is 0.00057 THz, with zero stability-call flips
across all forty of their re-measurements, whereas ORB-v2's reaches 2.01 THz and flips two calls,
one of them the KTaO₃ unit already excluded as borderline. The two models carrying the reordering
are therefore separated by two to four orders of magnitude more than the estimator's own noise, and
what noise exists is localised to the float32 direct-force model that §3.1 already identifies as the
weakest. This bounds environment and library nondeterminism at fixed displacement amplitude; it does
not probe sensitivity to the amplitude itself, which would require re-measurement on a displacement
grid (ESI §S1.2).

The defensible H2 statement is therefore about transfer, not about correlation: **harmonic
correctness does not carry over to the finite-temperature layer**, demonstrated by the 17-versus-4
asymmetry above. We deliberately do not upgrade this to "harmonic accuracy is non-predictive of
finite-temperature accuracy". That phrasing asserts a null, and the tests above fail to reject a
null rather than establishing one; with fifteen systems and five models the design cannot resolve
whether a weak association exists, still less its sign.
The clearest single illustration of the transfer failure is CHGNet: it is the **worst** model
harmonically (0.789) yet second **best** at finite temperature (0.867), while MatterSim is
harmonically perfect (1.000) and only mid-table at finite temperature (0.833). Because the
intervals in the table above overlap, this is an illustration and not a ranking claim. What it
does license, and what matters for practice, is negative and robust to the overlap: a top harmonic
score on the npj/PhononBench benchmarks does not identify a model as a good finite-temperature
screener, and a poor one does not disqualify it. We note explicitly that SevenNet-0 is top-equal in
*both* layers, tied with MatterSim harmonically, so the relationship is not a simple inversion and we do not claim one; an earlier version of this
analysis, computed before the mode-selection correction described in §2.4, reported that the
harmonic leaders sat behind MACE-MP-0 and CHGNet, and that specific ordering does not survive the
correction. ORB-v2 remains the weakest finite-temperature screener (0.800) and the only model that
fails to select the correct instability in SrTiO₃ (§2.4), consistent with its float32 direct
architecture over-softening the PES, though our design cannot separate the architecture from the
precision. The strongest evidence that finite temperature is a distinct, harder regime comes not
from this ranking comparison but from the per-family recall above and the SSCHA failure in §3.3.

CHGNet's harmonic accuracy is tolerance-dependent: 0.789 across the plateau tol ∈ [0.05, 0.20] used
throughout, rising to 0.895 at tol = 0.30, where two marginal false-unstables (CeO₂ at −0.267 THz
and NaCl at −0.238 THz) flip back and ORB-v2 becomes the uniquely worst model. We therefore state
the tolerance-robustness rather than the number: CHGNet sits below MatterSim harmonically at
every tolerance in [0.05, 0.50], and at tol = 0 the two tie rather than reversing. We do not
pair this with a finite-temperature ordering, because those intervals overlap. At tol = 0 all five models
collapse to 0.63–0.74 as the Γ acoustic numerical zeros flood the false-unstable count, and the
comparison is degenerate.

### 3.3 SSCHA: clean for bcc, false-stable for perovskites at the default truncation (the cautionary result)

We ran multi-mode SSCHA across both families to cross-validate the screen. The result is asymmetric
and is itself a central finding. We state at the outset that this concerns SSCHA as it is
realistically deployed with an MLIP force engine for a fixed high-symmetry reference, namely
`ForcePositiveDefinite` initialisation, a 2×2×2 cell, the v4=False free-energy Hessian, and
automatic stochastic relaxation, which is the recipe a practitioner escalating from a cheap screen
would actually run. It is not a claim that the SSCHA formalism is wrong; the formalism anticipates
this failure, and the expensive v4 Hessian is its own prescribed remedy, but that remedy is
impractical at screening scale (see the root cause below). The finding is that the default
escalation path, at the truncation its implementation ships with, is unsafe in this regime.

bcc metals: SSCHA is a clean gold standard, and the screen agrees with it where it counts. Across
Ti/Zr/Hf × 5 models × 5 temperatures (75 runs) SSCHA produced zero numerical failures, with minimum
free-energy-Hessian frequencies in a physical [0.06, 2.1] THz range. All five models anharmonically
stabilise the bcc phase by ≤50 K (the dynamic-stabilisation temperature, which is far below the
thermodynamic transition because bcc-to-hcp/ω is a martensitic, strain-coupled first-order
transition; using the thermodynamic T_c to label dynamic stability is the wrong comparison and is
what makes the models look false-stable on bcc). The margin to the stability boundary discriminates
the models: ORB-v2 hugs it (0.06–0.28 THz on Ti/Hf), SevenNet-0's Hf sits at ~0.3 THz, while
MatterSim and MACE-MP-0 are firmly stable (1.2–2.1 THz) (Fig. 3). Against the screen, the two
methods agree on the stability call in 35 of 45 paired bcc units (sign agreement 0.78; 30 of 36,
0.83, excluding ORB-v2), while the magnitudes are not rank-correlated (Spearman ρ = 0.11): a
single-mode symmetric-point curvature and a multi-mode free-energy Hessian are different
observables away from the sign change, so the cross-validation statistic we report is the call
agreement, not a magnitude correlation (Fig. 4).

Ferroelectric perovskites: SSCHA systematically false-stabilises, and its deepest wells do not run
at all. On the same FE perovskites at T ≤ 300 K where the screen reaches 0.53 recall, SSCHA
correctly identifies the instability in only 5 of the 27 units that return a physical number
(recall 0.19). Of the 30 attempted units, one blows up numerically and two fail outright; across
the full FE grid seven units die at cellconstructor symmetry or ensemble assertions, six of them
on PbTiO₃ — the deepest wells in the set (ORB-v2 at every temperature, MatterSim at 600 and
900 K). In the earlier, unpinned measurement these same units produced silent numerical blow-ups
to −2×10⁶ THz; in the pinned re-measurement they fail loudly instead, which is the better
behaviour but the same verdict: the method does not return usable answers where the instability is
deepest, and precision alone does not account for it (the failures span float32 and float64
models). This is the cautionary result: the expensive gold standard is less reliable than the
cheap screen in the displacive regime that dominates generative-CSP outputs — recall 0.19 against
0.53, with the deepest-well units unable to complete (Fig. 5).

Because both methods are evaluated on the same (system, model, temperature) units, the contrast
is a paired comparison and we test it as one rather than by inspecting whether the two marginal
intervals overlap. **We do not, however, attach a significance claim to it, and the reason is
worth setting out because an exact test on the paired units would supply a very small p.**

Over the combined displacive set the screen is right where SSCHA is wrong on 33 units against 3
the other way; excluding ORB-v2, 26 against 3. An exact McNemar over those discordant units
returns p = 2 × 10⁻⁷ and p = 1.5 × 10⁻⁵. Those p-values assume the units are independent, and
they are not: 47 units are five systems observed under five models at up to four temperatures.
That is the same objection that applies to the guardrail set in §3.4, and it applies here with
more force, because the number of systems is five rather than fifteen. Randomising the method
label over whole systems instead, which is exact at this size, gives **p = 0.125** for the
combined set, with four of the five systems favouring the screen (per-system net discordances
+6, +6, −1, +9, +10). With five systems the smallest attainable two-sided p is 2/2⁵ = 0.0625, so
no arrangement of these data could have reached conventional significance at the system level.
On the ferroelectric oxides alone (three systems) and the fluorites alone (two systems) the
clustered test is uninformative by construction.

What the data therefore support is an effect that is large, one-directional and consistent
across four of five systems, not a significant one. We report it that way. The argument that
the effect is real rests on mechanism rather than on sample size: the controlled diagnostic of
§S2.2, in which only the truncation order changes and the sign of the answer changes with it;
the fluorite within-cell control below; and the fact that the failure is predicted in the SCHA
literature for exactly this regime.^22^ Table S10 gives every system set with and without
ORB-v2, and both the unit-level and clustered tests. The ORB-v2 sensitivity is visible there
too: on the ferroelectric oxides alone the unit-level contrast weakens from 14-versus-3
(p = 0.013) to 10-versus-3 (p = 0.092) when ORB-v2 is removed.

![**Fig. 3** Multi-mode SSCHA dynamic-stabilisation curves for bcc Ti/Zr/Hf, five models,
versus temperature (§3.3). All models stabilise the bcc phase by ≤50 K; the margin to the stability
boundary discriminates them: ORB-v2 hugs it at 0.06 THz and SevenNet-0 at 0.30 THz, while
MACE-MP-0 and MatterSim sit firmly stable at 1.2 and 1.6 THz.](../results/figures/fig_sscha_bcc.png)

![**Fig. 4** Soft-mode screen vs gold-standard SSCHA minimum frequency on bcc
(§3.3): the cheap screen tracks SSCHA on the family where the gold standard is
trustworthy.](../results/figures/fig_method_agreement.png)

![**Fig. 5** Recall of the displacive (ferroelectric-perovskite) instability at
T ≤ 300 K: the cheap multi-mode soft-mode screen (0.53) versus the expensive SSCHA (0.19) (§3.3). The
cautionary result is that the gold standard is less reliable than the screen in the regime that
matters most; the paired test and its ORB-v2 sensitivity are in the text and Table S10.](../results/figures/fig_displacive_recall.png)

Root cause: the fourth-order truncation, not the reference structure. A controlled diagnostic on
cubic BaTiO₃ (MACE-MP-0, 100 K) isolates the mechanism, and the SCHA literature identifies it
precisely. The auxiliary SCHA matrix **Φ** is positive-definite by construction, since a
normalisable Gaussian trial state requires it, so it can never soften and searching its eigenvalues
for an instability is meaningless.^21,22^ The object that *can* detect a displacive transition from
a fixed high-symmetry reference is the free-energy Hessian,
∂²F/∂**R**∂**R** = **Φ** + **Φ**⁽³⁾**Λ**[**1** − **Φ**⁽⁴⁾**Λ**]⁻¹**Φ**⁽³⁾, which was derived for
exactly that purpose and demonstrated on the ferroelectric transitions of SnTe and GeTe.^21^ Our
diagnostic shows the failure enters at the truncation: the harmonic soft mode is −5.6 THz
(correctly unstable), yet with the fourth-order term dropped, which is the bubble approximation and
the `python-sscha` default, the Hessian returns +2.87 THz and the call is false-stable. That is the
regime in which the bubble is known to fail, since it breaks down for the cubic phase of metal
halide perovskites and produces qualitatively incorrect results there, and **Φ**⁽⁴⁾ plays a
significant role around the transition so that the complete RPA-like resummation is needed to
determine phase stability.^22^ Our contribution is to measure that breakdown at benchmark scale and
to show it is not confined to halides: it holds for oxide perovskites, and for cubic fluorites it
holds in a numerically clean form with zero stochastic blow-ups (above), which separates a
methodological limit from a numerical one. Enabling the fourth-order term (`include_v4=True`) is
therefore the physically correct escalation, consistent with that RPA requirement, but it ran for
more than 18 min on a single unit without finishing (tens of hours per unit at grid scale) and is
numerically viable only in float64, which is the practical barrier at screening scale. We therefore
use SSCHA as the bcc gold standard and the experimental transition temperature (the SrTiO₃ gate,
§2.4) as the validation for the perovskite screen. We add one structural observation from our own
data: even within the single-mode screen, deep wells condense first-order-like — the free energy
develops a lower displaced minimum while its curvature at the symmetric point stays positive
(§2.4, `n_curv_blind`) — and a fixed-reference Hessian criterion of any order is blind to a
condensation of that character. The screen survives it because it minimises over the centroid
rather than reading a curvature; a fixed-reference SSCHA deployment has no analogous escape short
of relaxing the centroids, which changes the question being asked.

Cubic fluorites confirm the failure is systematic, not numerical. Cubic ZrO₂ (the >2600 K phase,
harmonically unstable via the X-point oxygen mode at all temperatures studied) is a cleaner test
because its instability is shallower than the FE perovskites and does not trigger the float32
blow-ups. The soft-mode screen calls both fluorites unstable at every temperature and for every model
(recall 1.00), correctly and consistently. SSCHA instead returns +2.0 to +3.3 THz at 100 K for all
five models on ZrO₂ (false-stable) and then destabilises with temperature for three of them
(MACE-MP-0 −2.7 THz, SevenNet-0 −1.3 THz and ORB-v2 −24.8 THz at 900 K), which is both the wrong
sign at low T and the wrong temperature trend; CHGNet and MatterSim instead remain positive at
900 K (+1.2 and +2.3 THz), so they are false-stable across the whole ladder. Because SSCHA fails
the same way on a numerically well-behaved, shallower instability, the false-stable behaviour is
intrinsic to the truncated fixed-reference deployment (see the root cause above) rather than an
artifact of the deepest perovskite wells. HfO₂ shows the same pattern (all models +1.9 to +2.6 THz
at 100 K, then destabilising to −4.6/−10.0 THz for MACE-MP-0/ORB-v2 by 900 K). Across both
fluorites SSCHA produced zero numerical blow-ups and zero failed units, yet the same systematic
false-stable, which confirms that the failure is methodological, not numerical.

### 3.4 Ensemble disagreement as a guardrail (H3)

Because no single model is reliable across the set, we test whether cross-model disagreement flags
the units where the majority-vote consensus finite-temperature call is wrong. On the non-bcc,
non-borderline soft-mode units (n = 60; bcc excluded because its thermodynamic-T_c label is the
wrong reference for dynamic stability, §3.3), the majority-vote consensus is wrong on 12/60 =
0.200 [0.118, 0.318] of units. That set is fifteen systems × four temperatures, exactly the
fifteen systems of the matched set above (Table S7), and each unit already collapses the five
model votes, so the units are clustered by system and are not independent — every test below
resamples whole systems rather than units.

The disagreement signal separates the errors: on the 14 units where the five models split on the
stable/unstable call, the consensus error rate is 8/14 = 0.571 [0.326, 0.786], against 4/46 =
0.087 [0.034, 0.203] on the 46 unanimous units, a 6.6× enrichment. As a ranked predictor of
consensus error, the binary stable/unstable vote split reaches AUC 0.762, and it survives the
clustering: a permutation test over whole systems gives p = 0.004 and a cluster bootstrap gives a
95% interval of [0.590, 0.934], excluding 0.5. The naive unit-level permutation returns p < 0.001,
so treating the units as independent would indeed have overstated the significance, by about a
factor of four in p, without changing the conclusion. No such signal could be resolved for the continuous cross-model
frequency standard deviation (AUC 0.361, clustered p = 0.275, 95% interval [0.046, 0.625];
Fig. 6). We state that as a failure to resolve rather than as an absence. The interval spans
0.5, so the point estimate sitting below chance does not make it an inverted predictor; but the
interval also reaches 0.625, so these data do not exclude a moderately useful one either. The
practical reading is that the continuous spread cannot be relied on at this sample size,
whereas the discrete vote split can.

![**Fig. 6** Ensemble-disagreement guardrail (§3.4): the discrete inter-model
vote split predicts consensus error (AUC 0.762; system-clustered p = 0.004), while for the
continuous cross-model frequency spread no signal could be resolved (AUC 0.361, clustered 95%
interval [0.046, 0.625], which spans chance).](../results/figures/fig_ensemble_guardrail.png)

This refines H3 into a rule with a caveat: the discrete inter-model vote split is a useful, cheap
guardrail (flag any candidate on which the foundation-MLIP ensemble disagrees), while the
continuous frequency spread that one might naively threshold could not be shown to work here. The physical reading is that disagreement
concentrates near the stability boundary, where split votes coincide with frequencies straddling
zero and the call is both most uncertain and most error-prone.

### 3.5 Robustness: stochastic noise and finite size

Stochastic reproducibility. Repeating the bcc-Zr / MACE-MP-0 / 100 K SSCHA with four independent
random seeds gives a minimum free-energy-Hessian frequency of +1.798 ± 0.001 THz (range [+1.797,
+1.800]). The stochastic noise (≈0.001 THz) is two-to-three orders of magnitude smaller than the
cross-model bcc margins (§3.3), so those margins are real signal rather than sampling noise. The
same unit re-measured in the pinned environment, with the SSCHA initialiser rebuilt from phonopy
force constants, returns +1.80 THz at every temperature on the ladder, so the number also survives
an independent environment and initialiser.

Finite size. The SSCHA stability call is insensitive to supercell size on the cases where the test
is well-posed (`results/convergence_study.parquet`): for bcc-Zr the dynamic-stabilisation verdict
holds from 2×2×2 to 3×3×3 (+1.80 to +1.56 THz at 100 K, +1.80 to +1.55 at 300 K). We report the
soft-mode side of that study as superseded rather than as evidence: those runs were produced by the
earlier single-mode selection, before the correction described in §2.4, and have not been repeated
at 3×3×3.

We now state the limits of that test rather than letting the bcc result stand for the set. The
SrTiO₃ antiferrodistortive instability lives at the zone-boundary R point = (½,½,½), which is
commensurate only with even supercells, so a 3×3×3 cell is blind to it by construction and a
2×2×2-versus-3×3×3 comparison is not a valid convergence test for it. The same objection applies
to the fluorite X-point mode: X is commensurate with the 2×2×2 cell, so the mode is present, but
3×3×3 does not contain it either, so no cell-size comparison is available there. **The
zone-boundary systems therefore have no supercell-convergence test in this work, and we make no
convergence claim for them.** The definitive even-cell test (4×4×4, a ~320-atom SSCHA) remains
future work. The claim we do make is the narrower one the data support: the SSCHA verdict is
cell-robust where we were able to test it, which is bcc.

Separately, the §3.3 SSCHA false-stable is not a missing-**q** artifact, and this can be
established without a convergence test by comparing methods within a fixed cell. For BaTiO₃ the
instability is the Γ ferroelectric mode, present in every cell including the 2×2×2 used. That
argument covers one system, so we extend it to the fluorites by the stronger route: in the same
2×2×2 cell and with the same force engine, the harmonic calculation finds every one of the ten
fluorite units unstable, from −3.8 to −10.6 THz, while SSCHA at 100 K finds every one of them
stable, from +1.9 to +3.3 THz. The comparison is made at 100 K because that is where SSCHA is
uniformly false-stable; by 600 to 900 K three of the five models do destabilise, with the wrong
temperature trend, as §3.3 reports. The cell demonstrably resolves the instability, because the harmonic
calculation resolves it there; what loses it is the free-energy Hessian's truncation, exactly as
the root-cause diagnostic finds. A finite-size explanation would have to account for an
instability that is visible to one method and invisible to another in the same supercell.

## 4. Discussion

The harmonic layer (§3.1) reproduces the literature and confirms H1: PES softening produces a real
false-stable rate, localised to specific instabilities (bcc Zr/Hf) rather than uniform.

H2 is confirmed in the form §3.2 tests and no further. What the data establish is a **failure of
transfer**: harmonic correctness does not carry to the finite-temperature layer, with 17
harmonically-correct units mis-called at 300 K against 4 the other way (p = 0.007). That is a
statement about the relative difficulty of the two layers, and it is the one the title makes.
What the data do **not** establish is the pre-registered form, that harmonic and
finite-temperature accuracy are uncorrelated. Once pairs are clustered by system, no association
reaches significance at any temperature, and the sign of the point estimate itself changes across
the ladder; a failure to reject is not a demonstration, so we leave the association unresolved
rather than reporting it as absent. The per-model comparison should be read the same way: the five
finite-temperature accuracies span six units on a denominator of thirty and their intervals all
overlap, so the ranking is suggestive, not established. None of this weakens the practical
consequence, which is negative and needs no correlation to hold — a top harmonic benchmark score
does not certify a model for finite-temperature screening.

The standard escalation ("if in doubt, run SSCHA") fails on the displacive regime, because
MLIP-driven SSCHA at its default bubble truncation false-stabilises deep double wells. The practical recommendation inverts the usual cost/accuracy intuition: for screening
harmonically-unstable displacive candidates, the cheap all-imaginary-mode free-energy screen is the
more reliable indicator, and a default-truncation SSCHA escalation should be reserved for
martensitic and entropy-stabilised cases, or else run with the fourth-order resummation its own
theory prescribes.

Scope: foundation models as shipped, and why fine-tuning is outside it. Every result here is for
the released checkpoints used without adaptation. That is a deliberate scope rather than an
oversight, and it is the regime the benchmark is about. A practitioner filtering generative-CSP
output does not know in advance which candidates are strongly anharmonic, so there is no subset
to fine-tune on and no target-specific data to fine-tune with; the screening decision is taken
with the model as shipped, which is exactly the setting PhononBench and comparable filters
operate in. We would expect system-specific or family-specific fine-tuning to improve
finite-temperature performance substantially, and for a study of a *known* anharmonic material it
is clearly the right thing to do. What our results bound is the reliability of the untuned
foundation model in the screening role it is actually being given, and nothing here should be
read as a statement about what these architectures can achieve once adapted. Quantifying how much
of the finite-temperature gap fine-tuning closes, and how much training data per family that
takes, is the natural next study and is beyond what a benchmark of released checkpoints can say.

On model-reported uncertainty. A natural proposal is to use the models' own uncertainty on the
force predictions to decide where their stability calls should not be trusted. It is a
well-founded one: committee spreads propagated into molecular dynamics give more resilient
trajectories,^33^ a Bayesian force field's own predictive variance is a sound trigger for
acquiring new reference data,^32^ and monitoring uncertainty is what distinguishes physical from
unphysical MLIP dynamics on strongly anharmonic materials.^30^ We agree with the principle, and
§3.4 is that experiment: the five independent architectures form the ensemble, and
we test whether their disagreement flags the units where the consensus call is wrong. The outcome
is specific and, for anyone intending to threshold an uncertainty, worth knowing. The *discrete*
signal works, with split votes carrying a 6.6-fold enrichment in consensus error and surviving a
system-clustered permutation test. The *continuous* signal, the cross-model spread of the
predicted frequency, which is the propagated form of exactly the force-level disagreement the
proposal appeals to, yielded no signal we could resolve (AUC 0.361, 95% interval
[0.046, 0.625], spanning chance, so a moderately useful predictor is not excluded either). A
committee spread computed on a single architecture's own ensemble is a different quantity from
our cross-architecture spread and might behave better, but our measurement is the caution: the
continuous version of this idea failed on this set, and it failed while the discrete version
worked.

What an independent electronic-structure reference would add. Both layers compared here are
driven by the same MLIP potential-energy surface, so their agreement is a consistency check and
cannot by itself certify either. We are careful not to use it as one: the scoring labels and the
SrTiO₃ validation gate are experimental, and therefore independent of the surface, and the
SSCHA failure is attributed to its truncation by controls that hold the force engine fixed
(§3.3, §3.5, ESI §S2.2) rather than by comparison with the screen. What none of that supplies is
a check on the fitted double wells themselves. Computing first-principles energy profiles along
the soft-mode coordinates for a subset of systems, and benchmarking MLIP forces against them
along the SSCHA sampling paths, would separate a residual force-engine error from the
approximation error we have isolated here, and is the natural next study. It requires
electronic-structure resources beyond the scope of a benchmark built deliberately on published
ground truth.

Limitations. Ground-truth transition temperatures are approximate and scoring is qualitative. SSCHA
cells are 2×2×2 (finite size), so dynamic-stabilisation temperatures are approximate, though the
cross-model comparison at fixed cell is valid. ORB-v2's float32-only direct architecture is an
outlier in both layers and is flagged throughout rather than excluded. The screen is a
single-mode treatment applied mode-by-mode: it captures each instability independently but not
their coupling, which is the most likely reason the predicted stabilisation temperature fails to
order PbTiO₃ correctly (§3.2). Both the screen (0.53) and SSCHA (0.19) miss a substantial fraction
of the ferroelectric-oxide instabilities, so the comparison identifies the better of two imperfect
methods rather than a solved problem. Finally, α-AgI is modelled as an ordered CsCl-type
approximant of the bcc iodine sublattice; that is not the disordered superionic α phase, and its
screen call should be read as a placeholder.

## 5. Conclusions

Harmonic correctness does not carry over to the finite-temperature layer: on the matched set, 17
harmonically-correct units are mis-called at 300 K against 4 the other way. A harmonic benchmark
score therefore does not certify a foundation MLIP for the finite-temperature screening it is
routinely used for, and the obvious escalation does not rescue it, because MLIP-driven SSCHA at
its default bubble truncation systematically false-stabilises the deep displacive instabilities
that dominate generative-CSP output. A cheap quantum free-energy screen, applied to every
imaginary commensurate mode rather than to the softest one, is the more reliable indicator in
that regime. Over the combined displacive set that contrast survives removing the one
architecturally anomalous model, though on the ferroelectric oxides alone it does not (§3.3).
Which mode is examined matters as much as which method is used: the deepest instability need not
be the one that condenses, and in SrTiO₃ it is not. Where a single model cannot be trusted,
disagreement across an ensemble of independent architectures is a cheap and effective guardrail,
provided it is read as a discrete vote split; the continuous spread was not resolvable as a
predictor at this sample size.

We are deliberately not claiming a ranking of the five models at finite temperature. Their
accuracies span six units on a denominator of thirty and every interval overlaps, so which model
leads is not resolved by this design. The findings above are the ones that survive that
limitation, and all of them are negative or comparative rather than a leaderboard.

## Data availability

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability and archived at Zenodo at
https://doi.org/10.5281/zenodo.20805799 (concept DOI, resolving to the latest version). The production results regenerate from
`results/ledger.parquet` (per-unit hashed, resumable) and the finite-size runs from
`results/convergence_study.parquet`; figures via `scripts/make_figures.py`, analysis in
`mlip_dynstab/analysis.py`, the SSCHA root-cause diagnostic in `scripts/sscha_v4_diag.py`, and the
stochastic-reproducibility study (§3.5) via `scripts/sscha_repro.py` (its per-seed frequencies
print to the run log rather than to the ledger). The SSCHA results reported here are a full
re-measurement (`scripts/run_sscha_v2.py`, method version 3) of the original 208-unit grid in the
pinned environments, after two defects were found in the original analysis path: an acoustic-mode
identification inversion (the three translational modes were selected by lowest frequency rather
than smallest magnitude, which for an unstable phase discards the soft mode itself), and an
unrecorded software environment. The superseded generations are retained in the append-only ledger
— `mlip_dynstab.analysis.canonical` selects the current one — together with the in-place derived
correction (`scripts/fix_sscha_acoustic.py`, `*_v1` columns) and the uncorrected snapshot
(`results/ledger.parquet.pre-d1-fix`), so every stage of the correction is auditable.

The finite-temperature screen results reported here are the multi-mode grid described in §2.4
(5 models × 20 systems × 4 temperatures = 400 units). The ledger is append-only and also retains
the superseded single-mode rows; `mlip_dynstab.analysis.canonical` selects the current generation,
and every figure and statistic in this article is computed through it. The unit hash includes the
method version and the mode cap, so a re-run under a changed algorithm records new rows rather than
silently reusing old ones. Each model was run in its own pinned environment, with `pip freeze`
manifests deposited as `envs/lock-<model>-2026-08-16.txt`; the 220 cached E(Q) maps are included so
the temperature solves regenerate without a GPU.

## Author contributions

F.C. conceived the study, implemented the benchmark harness and analysis, performed all
computations, and wrote the manuscript.

## Conflicts of interest

There are no conflicts of interest to declare.

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
19. J. Yang, Z. Yin, L. Ao and S. Li, *Phys. Chem. Chem. Phys.*, 2026, **28**, 4459–4469.
20. D. Li, J. Yang, X. Chen, L. Yu and S. Liu, *J. Phys. Chem. C*, 2025, **129**, 21538-21544.
21. R. Bianco, I. Errea, L. Paulatto, M. Calandra and F. Mauri, *Phys. Rev. B*, 2017, **96**, 014111.
22. L. Monacelli, *Phys. Rev. B*, 2025, **112**, 014109.
23. D. J. Hooton, *Philos. Mag.*, 1955, **46**, 422.
24. N. R. Werthamer, *Phys. Rev. B*, 1970, **1**, 572.
25. R. Peierls, *Phys. Rev.*, 1938, **54**, 918.
26. R. P. Feynman, *Statistical Mechanics: A Set of Lectures*, W. A. Benjamin, Reading, MA, 1972.
27. R. Jinnouchi, J. Lahnsteiner, F. Karsai, G. Kresse and M. Bokdam, *Phys. Rev. Lett.*, 2019, **122**, 225701.
28. C. Verdi, F. Karsai, P. Liu, R. Jinnouchi and G. Kresse, *npj Comput. Mater.*, 2021, **7**, 156.
29. P. Liu, C. Verdi, F. Karsai and G. Kresse, *Phys. Rev. Mater.*, 2021, **5**, 053804.
30. K. Kang, T. A. R. Purcell, C. Carbogno and M. Scheffler, *Phys. Rev. Mater.*, 2025, **9**, 063801.
31. A. P. Bartók, M. C. Payne, R. Kondor and G. Csányi, *Phys. Rev. Lett.*, 2010, **104**, 136403.
32. J. Vandermause, S. B. Torrisi, S. Batzner, Y. Xie, L. Sun, A. M. Kolpak and B. Kozinsky, *npj Comput. Mater.*, 2020, **6**, 20.
33. G. Imbalzano, Y. Zhuang, V. Kapil, K. Rossi, E. A. Engel, F. Grasselli and M. Ceriotti, *J. Chem. Phys.*, 2021, **154**, 074102.
34. A. Merchant, S. Batzner, S. S. Schoenholz, M. Aykol, G. Cheon and E. D. Cubuk, *Nature*, 2023, **624**, 80-85.
35. C. Chen and S. P. Ong, *Nat. Comput. Sci.*, 2022, **2**, 718-728.
36. J. Riebesell, R. E. A. Goodall, P. Benner, Y. Chiang, B. Deng, G. Ceder, M. Asta, A. A. Lee, A. Jain and K. A. Persson, *Nat. Mach. Intell.*, 2025, **7**, 836-847.
37. A. Hajibabaei, C. W. Myung and K. S. Kim, *Phys. Rev. B*, 2021, **103**, 214102.
