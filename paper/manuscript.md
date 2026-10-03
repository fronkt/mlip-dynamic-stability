# Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation machine-learning interatomic potential for finite-temperature dynamic stability

Frank Cai^a^

^a^ Purdue University, West Lafayette, Indiana, USA. E-mail: cai485@purdue.edu.
ORCID: 0009-0003-0041-1459

## Abstract

Foundation machine-learning interatomic potentials (MLIPs) stand in for DFT in high-throughput
stability screening but are benchmarked mainly on harmonic (0 K) phonons. Cubic perovskites, bcc
metals and cubic fluorites are harmonically unstable yet stable above a transition
temperature. We test five foundation MLIPs, as shipped, on 20 systems. Harmonically the
models divide: MatterSim and SevenNet-0 score 19/19, while MACE-MP-0 and CHGNet flatten the bcc Zr
and Hf instabilities to zero. A single-mode quantum self-consistent harmonic approximation (SCHA)
screen shows that harmonic correctness does not certify finite-temperature behaviour: at 300 K,
17 harmonically correct units are mis-called against 4 the other way (7–24 against 4 under other
cell conventions). The asymmetry is not significant once units are clustered by system (p = 0.15).
On PBE energies along the same coordinates eight of the 17 (BaTiO₃, KNbO₃) are called correctly:
four models' wells there are about half the PBE depth. Multi-mode stochastic SCHA (SSCHA), with its
default free-energy Hessian at the symmetric structure, calls deep displacive wells stable against
unstable labels in 57 units; on the same MLIP energies the screen's free-energy comparison finds a
lower displaced minimum in 46 of them. A curvature read at the symmetric structure is local and
cannot certify stability against condensation into a deeper well. Under its
production convention the screen recovers 16/30 [0.36, 0.70] ferroelectric-perovskite
instabilities at T ≤ 300 K, against 5/27 [0.08, 0.37] for SSCHA.
<!-- PENDING-C1: the seed study shows converged = false and only the first population moving the auxiliary matrix on all 16 seeds of the four units (commit 2fdbb61; §S2.1 marker). If the converged-mode runs (C1c) do not keep the false-stables on BaTiO3/ZrO2, the SSCHA sentence above (and the title's second clause) must be restated as a property of the recipe as run, not of the default criterion; report whichever way it falls. -->
Ensemble vote splits flag consensus errors (AUC 0.76), but not robustly: 0.63 without ORB-v2.

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
harmonic-only oracle that PhononBench and CSP screening rely on is what this regime breaks. Two recent
studies probe finite-temperature reliability directly: four MACE foundation variants were
benchmarked for the dynamic stability of halide double perovskites,^19^ and the
ferroelectric-to-paraelectric transition of PbTiO₃ was used to expose a disconnect between static
accuracy and dynamic reliability.^20^ Both are confined to a single model family or a single
system. What remains untested is whether the effect is architecture-general and chemistry-general:
no study has classified finite-temperature dynamic stability across several independent MLIP
architectures and across the distinct anharmonic families, that is displacive and
antiferrodistortive oxide perovskites, halide perovskites, entropy-stabilised bcc refractory metals
and cubic fluorites, under a common free-energy criterion and a temperature ladder.

A separate line of work shows that machine-learned potentials can describe the families of
systems we test, provided they are trained for them. On-the-fly active learning with Bayesian error
estimation has reproduced the entropy-driven phase transitions of hybrid perovskites,^27^ the
temperature-driven transitions and anharmonic thermal transport of zirconia,^28^ and the alpha-beta
transition of zirconium,^29^ which are respectively the perovskite, fluorite and bcc families of
our set. These are system-specific potentials, actively trained on configurations drawn from the
target's own dynamics, and their success sharpens rather than weakens the question here, which is
about *foundation* models used as shipped and not about machine-learned potentials in general.
Active learning has also been aimed directly at the anharmonic failure mode: a screen of 112
materials found ten in which an insufficiently trained MLIP misdescribes the dynamics, and an
active-learning scheme driven by uncertainty estimates avoided those errors.^30^ That literature
descends from Gaussian approximation potentials, which were built on a sparse set of reference configurations
from the start,^31^ through a scalable sparse Gaussian-process regression with data-efficient
on-the-fly adaptive sampling, which reproduced the melting and glass-crystallisation temperatures
of the Li₇P₃S₁₁ solid electrolyte,^37^ and from on-the-fly Bayesian force fields whose own
predictive variance decides when new reference data are needed.^32^ The sparse Gaussian-process
line reaches practical accuracy from 100–1000 quantum calculations, requests a new one only when
the model's uncertainty exceeds a threshold, and combines expert models through a robust Bayesian
committee machine.^38^ A recent review of MLIPs for energy materials sets Gaussian-process
frameworks with uncertainty quantification beside the equivariant graph networks and foundation
models tested here, and lists foundation-model fine-tuning among the emerging directions.^39^ This
literature supplies the natural alternative reliability criterion to ours, namely committee or
ensemble uncertainty propagated into the simulation.^33^ §3.4 tests a cross-architecture version of
that criterion, and §4 discusses how it differs from a single model's committee and why the models
are tested here without fine-tuning.

The screening context is the other half of the motivation. High-throughput stability searches
now run over millions of candidates, using graph-network energy models trained at scale^34^ or a
universal interatomic potential,^35^ and sparse Gaussian-process potentials have been used to
accelerate high-throughput screening across materials spaces.^38^ The community benchmark for
machine-learned stability prediction evaluates precisely the model class tested here,^36^ and its
stability task scores thermodynamic stability against the 0 K convex hull; the phonon benchmarks
above score harmonic dynamical stability. Neither asks about the finite-temperature dynamic
stability a screened candidate actually needs, and that is the gap this work measures. The
soft-mode screen introduced here is meant as one more filter in such a pipeline, for candidates
that pass the hull check but show imaginary harmonic modes, to decide whether the high-symmetry
phase is stabilised at the temperature of interest.

Research question: do foundation MLIPs reproduce finite-temperature dynamic stability,
specifically the harmonic-unstable to thermally-stabilised transition, or does PES softening make
their stability calls unreliable where the harmonic approximation fails?

We pre-registered three hypotheses.

- H1 (softening to false-stable). PES softening inflates harmonic stability, so MLIPs over-report
  stable with a measurable false-stable rate on harmonically-unstable references. This reproduces
  the literature picture on our model set and anchors the validity of the harness.
- H2 (the finite-temperature gap). Harmonic accuracy does not certify a model for
  finite-temperature screening, and the harmonic ranking need not carry over. (The pre-registered
  form was "harmonic accuracy is uncorrelated with finite-temperature accuracy". The matched-set
  outcome in §3.2 is narrower. Harmonically correct units that the screen mis-calls at finite
  temperature outnumber the reverse in unit counts, 17 to 4 at 300 K under the screen's production
convention, but the asymmetry is not
  significant once units are clustered by system, most of it sits in three systems the screen
  mis-calls for at least four of the five models, and the direction of any association is not
  resolved at this sample size. The pre-registered form is therefore left open.)
- H3 (practical guardrail). Inter-model (ensemble) disagreement flags unreliable calls better than
  any single model's self-reported energetics. (§3.4 tests the ensemble signal only; the
  single-model comparator was not run. The vote split is suggestive, AUC 0.76, but falls to 0.63
  without ORB-v2.)

Our contributions are a finite-temperature benchmark on the anharmonic regime spanning five
independent MLIP architectures and four anharmonic chemistry families; a cheap free-energy screen
whose criterion is the phase rather than a single mode, reported with its sensitivity to its own
conventions; the observation that the default SSCHA cross-check, a free-energy Hessian at the
symmetric reference, answers a local question, and that as run it calls deep displacive wells
stable against unstable labels where, on the same MLIP energies, the screen's free-energy
comparison finds a lower displaced minimum, consistent with the published account of bubble
truncation;^22^ and an
ensemble-disagreement guardrail that is suggestive rather than robust (H3).

## 2. Methods

### 2.1 Systems and reference labels

The curated set (`configs/curated_systems.yaml`, 20 systems) comprises references whose
harmonic-imaginary and finite-temperature-stable behaviour is documented in the literature, so the
reference label needs no new DFT:

- Oxide displacive/antiferrodistortive perovskites (4): SrTiO₃,^11^ BaTiO₃, PbTiO₃, KNbO₃.^12^
- Halide perovskites (3): CsPbI₃,^13^ CsSnBr₃, CsSnI₃.^14^
- bcc refractory metals (3): Ti, Zr, Hf (phonon-entropy stabilisation).^15^
- Cubic fluorites (2): ZrO₂, HfO₂.^16^
- Superionic (1): α-AgI.^17^
- Quantum-paraelectric near-control (1): KTaO₃.^18^
- Harmonically-stable controls (6): Si, MgO, NaCl, Cu, diamond, CeO₂ (0 K harmonic stability
  documented in the DFPT phonon database^3^).

Transition temperatures are approximate, and scoring is qualitative (the correct side of the
transition, not a fitted T_c). The experimental labels mark thermodynamic, often first-order,
phase transitions, whereas the screen and SSCHA probe the dynamic stability of the high-symmetry
phase, so a label and a dynamic-stability call can differ near a transition without either being
wrong; the six control labels come from the DFPT phonon database^3^ rather than from experiment.
KTaO₃ is flagged borderline (incipient ferroelectric, DFT mode ≈0; three of five MLIPs call it
imaginary) and excluded from headline rates, which leaves 19 scored systems. CeO₂ and NaCl are
retained as scored stable controls, since they are genuinely stable and only CHGNet marginally
trips them (§3.1). The soft-mode screen and the harmonic baseline cover all 20 systems; the SSCHA
grid covers the four oxide perovskites, one halide perovskite (CsSnI₃), the two fluorites and the
three bcc metals. α-AgI is the one system where the soft-mode screen is on shaky physical ground,
because a superionic with a diffusive Ag sublattice has no single frozen order parameter; its
screen call is reported but flagged, and it is the natural target for a future
symmetry-breaking/MD probe.

### 2.2 Models

MACE-MP-0,^6^ CHGNet,^7^ ORB-v2,^8^ SevenNet-0,^9^ and MatterSim^10^ run through a
backend-agnostic ASE-calculator harness, with one virtual environment per model. All five are
the released checkpoints used as shipped, without fine-tuning (§4 discusses that scope). Testing
MatterSim directly probes PhononBench's own oracle. ORB-v2 is the only one of the five whose forces
are predicted directly rather than as gradients of the energy (non-conservative forces). Precision
does not single it out: as run, MACE-MP-0 evaluates in float64 and the other four models,
ORB-v2 included, return float32 forces. We carry ORB-v2 as an explicit caveat because it stands
apart in both the harmonic and the finite-temperature results, which this design cannot trace to
its direct forces or to anything else about the architecture, and we report the rates it could
drive with and without it (ESI Tables S9 and S15–S18).

### 2.3 Harmonic baseline (credibility anchor)

Finite-displacement harmonic phonons (phonopy, 2×2×2 force-constant supercells, 0.01 Å
displacements, 12×12×12 interpolation mesh) with each MLIP, classified by the minimum phonon
frequency against an imaginary tolerance (default −0.1 THz). The success criterion is to reproduce
the published harmonic stability split. This is a check on the harness and is not claimed as
novel.

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
three (a triplet whose components a model returns at slightly different frequencies, as CHGNet does
for BaTiO₃, is screened as separate modes; ESI §S1.3). Within a degenerate eigenspace the screened
direction is the one the eigensolver returns, which is arbitrary, and the well along another
direction of the same span can differ (ESI §S1.3). Each surviving mode is frozen into its minimal commensurate cell via phonopy modulation; we
map the static double well E(Q) with quadratic Q sampling, fit the well-plus-barrier window, and
minimise a single-mode quantum SCHA free energy over the order-parameter centroid (self-consistent
Gaussian width via bracketed root finding).

Variational free energy per mode. Each frozen mode defines a one-dimensional subsystem with
coordinate Q along the displacement pattern **u** (normalised as defined below), effective mass
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
iteration can reach), and 𝓕 is then minimised over the centroid on a Q₀ grid. ESI §S1.3 derives
each of these steps in full and states what the single-mode restriction neglects.

Definitions and numerical settings. The pattern **u** is phonopy's modulation rescaled so that its
largest Cartesian component is 1 (max over i and α of |u_iα| = 1), so Q, in ångström, is the
largest Cartesian displacement component of any atom in the distorted cell. M = Σᵢ mᵢ|**u**ᵢ|² and
V(Q) are both summed over the mode's minimal commensurate cell, the smallest diagonal supercell in
which the mode is a single commensurate distortion; in a cubic perovskite that is 1, 2, 4 or 8
formula units for a Γ, X, M or R mode. E(Q) is sampled at ten quadratically spaced points,
Q_k = 0.45 (k/9)² Å for k = 0, …, 9, which puts most points at small Q where narrow wells sit. The
fit keeps the sampled points within max(60 meV, 5|E_min|) of the sampled minimum E_min (the four
lowest-Q points if fewer qualify), fits the even sextic, and refits a quartic if the sextic comes
out unbounded below (c < 0). Sampling only Q ≥ 0 assumes V(−Q) = V(Q). Symmetry guarantees that for
most modes, but not for 24 bcc modes whose wavevector satisfies 3**q** ≡ **G**, where a cubic term
is allowed and parity depends on the modulation phase; we did not check those. 𝓕 is minimised over
Q₀ on 121 points spanning 0–0.6 Å (step 0.005 Å), and a mode is called condensed when the
minimising Q₀ exceeds 0.0075 Å, 1.5 grid steps. Where Brent's method fails on a bracket, the width
falls back to the grid value nearest the root; this happens on 4.6% of centroid evaluations and at
none of the minimising centroids that decide the 1512 mode-temperature calls. For 19 ORB-v2 modes
(18 on bcc Ti, one on PbTiO₃) every sampled energy is non-negative, yet the fitted sextic has a
minimum between sample points, 34.6–798.9 meV deep for the Ti modes and 0.2 meV for the PbTiO₃
one. These wells are fit artefacts; no unit call
depends on them, because each affected unit has another condensing mode with a sampled well.

Criterion and observable. Two distinct quantities come out of 𝓕, and we keep them separate. The
**stability call** is the variational one: the mode has condensed at T if the global minimum of
𝓕(Q₀; T) sits at a displaced centroid (Q₀ > 0.0075 Å, above); by the bound above, the displaced
state then has the lower free energy. The high-symmetry structure is dynamically unstable at T if
**any** screened mode condenses, and stable only when none does; dynamical stability is a property
of the phase, not of one mode. The **reported frequency** is the curvature of the free energy at
the symmetric point, ω_eff = sign(𝓕″(0)) · [|𝓕″(0)|/M]^½, the single-mode analogue of the SSCHA
free-energy Hessian^21^ (𝓕 is even in Q₀ under the parity assumption above, so the curvature is
evaluated by a symmetric finite difference). For a single mode that observable cannot signal an
instability: at the symmetric point the parity of V removes every term but ⟨V″⟩, so
𝓕″(0) = MΩ², the self-consistent trial stiffness, which is positive whenever a bound Gaussian
exists there (ESI §S1.3). The negative values it reports (95 of 1512 mode-temperature
evaluations) come from the width solver's grid fallback at a point of the finite-difference
stencil and from two fit-artefact modes, not from the physics. The call and the curvature
therefore disagree wherever a mode condenses: for a deep double well the variational transition
is first-order-like, so 𝓕(0) remains a *local* minimum (positive curvature) while a displaced
minimum drops below it. A curvature criterion is blind to that condensation by construction, and
so is any free-energy Hessian evaluated at a fixed high-symmetry reference when the reference is a
local minimum; we record every such mode (`n_curv_blind` in the ledger) rather than folding it into
either number.

Screening only a single mode — even the globally softest — conflates physically distinct
instabilities. Cubic SrTiO₃ is the decisive case: its Γ ferroelectric mode is *deeper* than the
R-point antiferrodistortive tilt, yet under the production normalisation (below) the Γ mode is
quantum-suppressed and does not condense, while the R tilt is what drives the 105 K transition, so
a softest-mode screen inspects the wrong mode and calls the cubic phase stable. The E(Q) maps are
temperature-independent and cached, so each temperature is a sub-second CPU solve over all modes. A cap of 24 modes per unit bounds the cost;
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
mode-temperature evaluations covering 57 (system, model) units (Table S12).

Approximations, declared. Table 1 states what the screen neglects, the expected direction of the
bias, and where the consequence is visible in our own data.

**Table 1** The screen's approximation ledger.

| # | Approximation | What it excludes | Expected bias | Where it shows |
|---|---|---|---|---|
| A1 | One mode at a time | Mode–mode coupling and cooperative condensation | Either sign; absolute T* unreliable | PbTiO₃ T* ordering failure; mis-calls shared by four or five models (§3.2) |
| A2 | Commensurate **q** only | Instabilities at incommensurate or finer-grid **q** | False-stable if the true soft mode is missed | R point needs even cells (§3.5); bcc ω needs 6×6×6 FCs |
| A3 | Sextic fit on a well+barrier window, sampled to Q = 0.45 Å | Steep-wall anharmonicity beyond Q⁶; wells beyond the sampled range | Barrier-shape error at large Q | No unit flip over the fit windows, ≤ 1 at T ≤ 300 K over sampling ranges (ESI Table S14) |
| A4 | Gaussian (SCHA) trial state | Tunnelling, non-Gaussian density; renders deep-well transitions first-order-like | Curvature misses condensation | `n_curv_blind` rows; §3.3 mechanism |
| A5 | Static E(Q) in a clamped cell | Thermal expansion and strain coupling | Under-stabilises martensitic systems | bcc excluded from scoring (§3.2) and compared with SSCHA only (§3.3) |
| A6 | Mode cap (24) | Modes beyond the 24 most imaginary | Could only create false-stables | Binds on 20/400 units, all already unstable (§2.4) |
| A7 | Energy and mass summed over the minimal commensurate cell | Any other frozen-cell normalisation | Larger cells condense more readily | Recall and T* depend on it (§2.4; ESI Table S14) |

SrTiO₃ gate. For MACE-MP-0, cubic SrTiO₃ resolves three distinct imaginary commensurate modes, and
the screen reports which of them condenses. At 100 K the Γ ferroelectric mode does **not** condense (Q₀ = 0),
which is the physically correct quantum-paraelectric result, while the R = (½,½,½) tilt does
(Q₀ = 0.14 Å); by 300 K neither condenses. The phase is therefore called unstable at 100 K and
stable at 300 K, bracketing the experimental transition at 105 K. The gate also exercises the
criterion/observable distinction above: at 100 K the condensing tilt still has a *positive*
symmetric-point curvature (+0.92 THz, hardening to +1.43 THz at 300 K), so this is a
first-order-like condensation that the free energy's argmin detects and its curvature does not,
the situation that any criterion read at the symmetric reference misses (§3.3). The gate is
coarse: on a 100/300 K ladder it passes for any transition between 100 and 300 K, so it brackets
the 105 K transition without locating it. Three of the five models (MACE-MP-0, MatterSim, SevenNet-0) reproduce the R-tilt
condensation; CHGNet never condenses the tilt and ORB-v2 does not select the R point at all.
Because the gate identifies *which* instability each model captures rather than returning a single
number, it is a per-model test rather than a single-model demonstration.

Frozen-cell normalisation. The cell over which M and V are summed is not a harmless convention.
Rescaling **u** changes no call, provided the settings given in ångström above (sampling range,
centroid box and threshold) are rescaled with it; summing over n minimal cells instead multiplies
a, b, c and M by n, which is the minimal-cell problem at mass n²M and temperature T/n, so larger cells
condense more readily. ESI Table S14 shows that the choice decides outcomes: the
ferroelectric-perovskite recall at T ≤ 300 K is 5/30 per formula unit, 16/30 in the minimal cell,
26/30 in the doubled cell and 28/30 in the common force-constant supercell, and the SrTiO₃ gate
passes for 0/5, 3/5, 2/5 and 3/5 of the models respectively. Accuracy cannot choose among them.
Scored at T ≤ 300 K the four give 105/150, 127/150, 136/150 and 139/150, but that set is nearly
blind to over-condensation, because only 9 of its 70 truly stable units carry an imaginary
commensurate mode at all. Over the full ladder they give 225/300, 243/300, 249/300 and 252/300,
with the larger cells paying in false-unstables, most of them at 600–900 K (36/190 and
41/190 stable-labelled units over the ladder, against 28/190),
and an eight-fold cell, which condenses most readily, matches the doubled cell at T ≤ 300 K but
falls below the minimal cell over the full ladder (234/300). What accuracy rewards therefore
depends on the balance of stable and unstable labels in the scored set, and those labels mark
thermodynamic transitions rather than the dynamic-stability boundary itself (§2.1). We use the
minimal cell because it is the smallest cell in which the mode is one commensurate distortion, so
M and V refer to a single repeat of the order parameter rather than to an arbitrary multiple of
it. That is an argument for the choice, not evidence that it is right, and every screen result in
this paper, the recall and T* especially, is reported under it and depends on it. That includes the
H2 count of §3.2: at 300 K it is 17 against 4 in the minimal cell and runs from 7 against 4 to 24
against 4 across the other conventions (ESI §S1.3).

The other axes of Table S14 matter far less. Fit windows of 3×, 5× and 8× the well depth with
floors of 30, 60 and 120 meV flip no unit call, although for 69% of modes the kept points are
identical across the multipliers at the 60 meV floor, so that sweep cannot move them. Sampling only
to Q ≤ 0.25–0.40 Å flips at most one unit call at T ≤ 300 K (eight over the full ladder at 0.25 Å,
seven of them halide-perovskite units at 600–900 K), and condensation thresholds of 0.5–3 grid
steps or a centroid box widened to 1.2 Å flip none. One axis was not varied: the force-constant
supercell, which fixes the **q**-points the screen searches, was not enlarged, so an instability
at a **q** outside that set would be missed (approximation A2).

### 2.5 Multi-mode SSCHA (default cross-check)

The full stochastic SCHA (python-sscha + cellconstructor)^4^ uses the MLIP as its force engine.
SSCHA here therefore inherits the MLIP potential-energy surface exactly as the screen does: it is
a second approximation to the same surface, not an independent reference. The starting dynamical
matrix is built from phonopy full force constants at a 0.03 Å displacement in the SSCHA supercell
(2×2×2, as for every unit reported here), made positive-definite (`ForcePositiveDefinite`) and
symmetrised. The auxiliary dynamical matrix is then relaxed stochastically at T in the root2
representation, with `min_step_dyn` = 0.5 and `meaningful_factor` = 10⁻⁴, the convergence
threshold. The minimiser's steps are capped at `max_ka` = 20, a cap that python-sscha 1.6.1
compares with the step count accumulated over all populations rather than resetting for each one
(ESI §S2.1), and the relaxation runs for up to 8 populations of 256 configurations (2,048 at
most). A dedicated 512-configuration ensemble at the final auxiliary matrix (2,560 configurations
per unit at most in all) then gives the free-energy
(physical) Hessian at the bubble level (`include_v4 = False`, the python-sscha default). Its minimum
frequency, excluding the three acoustic modes (taken as those nearest zero in magnitude), is called
unstable below the same −0.1 THz tolerance as the harmonic layer. The production grid did not
record how many populations each unit used, or whether each unit met the convergence threshold
before reaching the population cap, so neither is reported for it.
A re-run of four units with four seeds each that records them (BaTiO₃ and ZrO₂ with MACE-MP-0 at
100 K, bcc Zr with MatterSim at 50 K, SrTiO₃ with MACE-MP-0 at 600 K; ESI Table S21) shows that
no seed met the threshold. The cumulative cap ended the first population after 19 steps, each later
population took one step and discarded it, so only the first population moved the auxiliary
matrix, and the final gradient sat 7 × 10² to 10⁴ times above the threshold. The free-energy
Hessian was therefore evaluated close to the positive-definite start: within 0.24 THz of it for
BaTiO₃, ZrO₂ and bcc Zr, with a seed spread of 0.007–0.075 THz. SrTiO₃ at 600 K is the exception,
at −15.8 to −20.3 THz across seeds (standard deviation 2.0 THz) from a start of +0.90 THz. The
gradient error python-sscha records is a constant placeholder (ESI §S2.1), so the convergence test
it feeds is not a stochastic one.
<!-- PENDING-C1c/GRID: one or two sentences on the converged recipe (per-population cap, real gradient error) and what the converged grid shows, with its ESI table number. -->
Because this Hessian is evaluated at the fixed high-symmetry reference, its sign answers a local
question: whether the symmetric phase is a local minimum of the free energy. §3.3 shows why that
matters for deep double wells. Unlike the free-energy screen, which treats each imaginary mode
independently, SSCHA captures the coupled multi-mode phonon entropy, which matters most for the
entropy-stabilised martensitic transitions of the bcc metals.
### 2.6 First-principles reference (PBE)

To separate force-engine error from the approximations of the two finite-temperature methods we
computed PBE single points with pw.x (Quantum ESPRESSO 7.5)^44^ and the SSSP 1.3 PBE efficiency
pseudopotentials,^45^ for six systems (BaTiO₃, KNbO₃, SrTiO₃, CsSnBr₃, cubic ZrO₂, bcc Zr; 448
calculations, `scripts/dft_reference.py`). Cutoffs are the largest SSSP recommendation over the
elements (30–60 Ry for the wavefunctions, eight times that for the density), with
Marzari–Vanderbilt smearing^46^ of 0.005 Ry (0.02 Ry for Zr) and Γ-centred k-meshes at a spacing
of 0.25 Å⁻¹ (0.15 Å⁻¹ for Zr, 2π included). (i) Along the soft-mode coordinates (C3a): for each
model, the paths that decide its screen call were regenerated from its own force constants,
checked against the cached E(Q) map, and evaluated with PBE at ten amplitudes up to 0.45 Å, at that
model's relaxed lattice and along its eigenvector, not at the PBE lattice or along the PBE
eigenvector; the five MLIPs were re-evaluated on the identical geometries, and the screen of §2.4
was re-solved on the PBE-fitted E(Q) with every other setting unchanged. Where a model's screen
finds no instability in a system that another model does (bcc Zr for CHGNet, MACE-MP-0 and
SevenNet-0; SrTiO₃ for ORB-v2), the path is MatterSim's deciding coordinate. (ii) On
SSCHA-sampled configurations (C3b): twelve configurations from each of four seed-study ensembles
(BaTiO₃, ZrO₂ and SrTiO₃ with MACE-MP-0 at 100, 100 and 600 K; bcc Zr with MatterSim at 50 K),
scored for all five models against PBE forces and energies, with four rattled cells (0.03 Å
root-mean-square displacement) of the same supercell as the near-equilibrium baseline. These
ensembles come from the production recipe, which did not converge (§2.5), so at low temperature
they may sample only the neighbourhood of its positive-definite start rather than the double
well. Energies are compared only as differences within one code.

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
throughout. On nineteen systems those intervals are wide and they overlap heavily, so what we
report is which instabilities each model reproduces, not a ranking of the accuracies, which this
design does not separate. Per-model confusion matrices are in Table S4.

MatterSim and SevenNet-0 reproduce every documented soft mode with large imaginary frequencies.
MACE-MP-0 and CHGNet each carry two false-stable calls (2/13) on the bcc Zr and Hf soft modes,
which both models read at −0.000 THz, so neither sees a Zr or Hf instability at all; this is the
PES-softening bias of the literature, localised to specific instabilities. ORB-v2 carries one
false-stable (1/13): it correctly flags bcc-Zr (−0.47 THz) and bcc-Hf (−0.19 THz) unstable but
reads SrTiO₃ at +0.00 THz. Besides that false-stable it falsely calls MgO unstable (−1.07 THz), so
ORB-v2, the one model with directly predicted forces (§2.2), errs in both directions; ESI Tables S9
and S15–S18 report the rates it could plausibly drive with and without it. CHGNet carries the only
non-ORB false-unstables (2/6), CeO₂ (−0.27 THz) and NaCl (−0.24 THz), both genuinely stable
controls tripped just past the −0.1 THz tolerance. Both read stable at a tolerance of 0.3 THz and
at a 0.03 Å displacement (ESI Table S13), so we treat them as finite-displacement noise near the
tolerance rather than a real instability.

Because every binary call depends on the imaginary tolerance, we sweep it (Fig. 1). A strict
tolerance (0 THz) floods false-unstables (29 of 95 calls) as near-Γ finite-displacement noise
dominates, while a loose tolerance (0.3 THz) inflates false-stables (7 calls). The default −0.1 THz
sits in the stable basin between them with 5 false-stable and 3 false-unstable calls, the same at
0.05 THz and 6 and 3 at 0.2 THz; without ORB-v2 it gives 4 and 2 of 76 calls. The softening toward
zero, rather than any single rate, is the physics here. This reproduces the published picture and
is a check on the harness rather than a finding.

![**Fig. 1** Harmonic false-stable and false-unstable call counts versus the
imaginary-frequency tolerance (§3.1). The default −0.1 THz sits in the stable basin between the
false-unstable flood at strict tolerance and the false-stable inflation at loose tolerance.](../results/figures/fig_tolerance_sweep.png)

Fig. 2 shows the minimum harmonic frequency behind each call at the default tolerance, so the
division among the models can be read system by system. The five false-stable cells (bcc Zr and
Hf for MACE-MP-0 and CHGNet, SrTiO₃ for ORB-v2) read 0.00 THz, the same numerical zero of the
acoustic branch at Γ that the correctly called controls read: on those potentials no frequency on
the mesh lies below that zero, so these misses are not near-threshold calls. The three false-unstable
cells are CHGNet's CeO₂ (−0.27 THz) and NaCl (−0.24 THz) and ORB-v2's MgO (−1.07 THz).

![**Fig. 2** Harmonic layer (§2.3, §3.1): the minimum phonon frequency of each system on each
model's potential, over the Γ-centred 12×12×12 mesh interpolated from 2×2×2 finite-displacement
force constants, with negative values imaginary. Boxed cells are the units called unstable at the
production tolerance (minimum below −0.1 THz); FS and FU mark calls that disagree with the
reference label (false-stable and false-unstable). The minimum includes the acoustic branch at Γ,
so a system without an instability reads 0.00 THz and no cell lies above that numerical zero; the
five false-stable cells read the same zero. The colour scale is diverging and centred at zero (red
negative), linear within ±0.1 THz and logarithmic to −10 THz, and the colour bar shows only its
negative half, with the tolerance dashed; printed values are unclipped (HfO₂ on CHGNet, −10.60 THz, lies beyond the end of the scale). The
borderline KTaO₃ (*) is shown but not scored, so it carries no FS or FU
mark.](../results/figures/fig_harmonic_heat.png)

### 3.2 Finite-temperature soft-mode screen (H2)

On the ferroelectric perovskites at T ≤ 300 K, below every transition temperature, the label is
the distorted phase, and the multi-mode screen calls the cubic phase unstable in 16 of 30 model
units (recall 16/30 = 0.533 [0.361, 0.698]). That recall is quoted under the production
frozen-cell normalisation; across the conventions of §2.4 it runs from 5/30 to 28/30, so it is a
property of the screen as defined rather than a convention-free number. As §3.3 shows, the default
SSCHA cross-check recovers fewer of the same instabilities (5/27).
Across the other anharmonic families the screen is stronger: 23/25 = 0.920 [0.750, 0.978] on the
halide perovskites and 20/20 = 1.000 [0.839, 1.000] on the cubic fluorites, while on the
antiferrodistortive SrTiO₃ tilt it reaches 3/5 = 0.600 [0.231, 0.882] on a denominator too small
to support any comparison. Without ORB-v2 these recalls are 12/24, 20/20, 16/16 and 3/4. On the
six harmonically-stable controls the screen is correct on all 120 model units (0/120, Wilson
[0.000, 0.031]; 0/96 without ORB-v2) and finds zero imaginary commensurate modes in every case.
One qualification belongs with that number: the harmonic classifier reads an interpolated
12×12×12 mesh while the screen reads only the **q** commensurate with the force-constant cell,
so the three harmonic false-unstables (ORB-v2 on MgO at −1.07 THz, CHGNet on CeO₂ and NaCl at
−0.27 and −0.24 THz) have no commensurate counterpart for the screen to find. The control result is therefore
in part a consequence of approximation A2, and we report it as a consistency check rather than
as proof that the screen cannot manufacture an instability.

Only the screen's call is scored; its curvature observable cannot signal an instability (§2.4).
For a single mode the symmetric-point curvature equals the self-consistent trial stiffness and is
positive by construction, so it is positive on the units the screen calls unstable as well, where
a displaced minimum has dropped below a symmetric point that remains a local minimum. The
negative values that observable reports are numerical, not physical (ESI §S1.3).

Multi-anchor comparison against experiment. Beyond the SrTiO₃ gate (§2.4) we compare the screen's
predicted stabilisation temperature T* (the lowest ladder T at which the cubic phase is called
stable) with the experimental transition temperatures. The comparison is only partly successful and
we report it as such. SrTiO₃ (T_c ≈ 105 K) has T* = 300 K for the three models that pass the
gate and 100 K for CHGNet and ORB-v2, which never condense its tilt; BaTiO₃ (393 K) and KNbO₃
(708 K) both have T* = 300 K for four of five models (ORB-v2 at 600 K). SrTiO₃ therefore never
stabilises later than the two ferroelectrics, but on this coarse ladder it ties with them for the
three models that pass the gate, and the ladder does not separate BaTiO₃ from KNbO₃ at all.
PbTiO₃ (763 K), which has the highest transition temperature of the four, does **not** follow: T* is 100 K for MACE-MP-0, MatterSim and ORB-v2,
600 K for SevenNet-0 and 900 K for CHGNet. A single-mode treatment is not expected to reproduce an
absolute T_c, but the PbTiO₃ failure is a genuine ordering failure rather than a scale error, and it
is the clearest limitation of the screen in this work. T* also moves with the frozen-cell
normalisation (§2.4). It is therefore reported as a diagnostic (Table S6 of the ESI); the SrTiO₃
gate and the control performance, not the T* ordering, are what support using the screen's calls
in the comparison of §3.3.

Per-model false-stable rates on the displacive/anharmonic set (Table S5 gives the same rates over
the full temperature ladder and with the bcc metals included; non-bcc, non-borderline, T ≤ 300 K;
bcc excluded because its label marks a thermodynamic, strain-coupled transition rather than
dynamic stability, §2.1 and §3.3):

| Model | Finite-T false-stable rate | Finite-T accuracy | Harmonic accuracy, matched | (Harmonic, all 19) |
|---|---|---|---|---|
| SevenNet-0 | 2/16 = 0.125 [0.035, 0.360] | 27/30 = 0.900 [0.744, 0.965] | 15/15 = 1.000 [0.796, 1.000] | 1.000 |
| CHGNet | 3/16 = 0.188 [0.066, 0.430] | 26/30 = 0.867 [0.703, 0.947] | 13/15 = 0.867 [0.621, 0.963] | 0.789 |
| MatterSim | 4/16 = 0.250 [0.102, 0.495] | 25/30 = 0.833 [0.664, 0.927] | 15/15 = 1.000 [0.796, 1.000] | 1.000 |
| MACE-MP-0 | 4/16 = 0.250 [0.102, 0.495] | 25/30 = 0.833 [0.664, 0.927] | 15/15 = 1.000 [0.796, 1.000] | 0.895 |
| ORB-v2 | 5/16 = 0.312 [0.142, 0.556] | 24/30 = 0.800 [0.627, 0.905] | 13/15 = 0.867 [0.621, 0.963] | 0.895 |

The fourth column is the harmonic accuracy **on the same fifteen systems** the
finite-temperature column scores, and it is the only harmonic column that may legitimately be
compared with it. The fifth reproduces the §3.1 figure over all nineteen scored systems and is
shown only so the two are not confused: it includes the bcc metals, which the finite-temperature
layer excludes, and MACE-MP-0's and CHGNet's harmonic errors are concentrated exactly there.

On the matched set MACE-MP-0 is harmonically perfect (15/15), not 17/19, and CHGNet is 13/15 =
0.867 rather than 15/19 = 0.789, the same as its finite-temperature accuracy of 26/30. Setting
CHGNet's nineteen-system harmonic accuracy beside its fifteen-system finite-temperature accuracy
would suggest that the model lowest at the harmonic level is second best at finite temperature;
that reading comes from mismatched denominators, and we do not draw it. On matched systems
CHGNet's two harmonic errors are CeO₂ and NaCl, the marginal controls of §3.1.

The matched columns show the three harmonically perfect models at 25/30 to 27/30 at finite
temperature, and CHGNet and ORB-v2, both 13/15 harmonically, at 26/30 and 24/30. **These five
finite-temperature accuracies are not separated by this design**: every interval overlaps every
other, the whole spread is three units on a denominator of thirty, and 24/30 against 25/30 is one
unit. We make no ranking claim at finite temperature.

The matched non-bcc, non-borderline comparison (n = 75 system×model pairs, five models over
fifteen systems; ESI Table S7) is temperature-dependent, and we separate two questions that a
single 2×2 table invites conflating. The first is whether harmonic correctness carries over to the
finite-temperature layer, which the two discordant cells answer: b counts the units the harmonic
layer gets right and the screen gets wrong at T, and c the reverse. At 300 K, b = 17 and c = 4.
That asymmetry is large and one-directional in unit counts, but it is not significant once units
are clustered by system. The 21 discordant units come from nine systems, five with more b than c
and four with more c than b, and an exact test that randomises the sign of each system's net
discordance gives p = 0.152. McNemar's exact test on the units gives p = 0.007, but it treats the
75 pairs as independent when they are five models observed on each of fifteen systems, so we give
it only as a labelled companion. The whole ladder:

| T (K) | both correct | harm ✓, finite-T ✗ (b) | harm ✗, finite-T ✓ (c) | both wrong | b v c, system-clustered exact p | b v c, unit-level McNemar p (companion only) | φ | φ, system-clustered permutation p |
|---|---|---|---|---|---|---|---|---|
| 100 | 66 | 5 | 3 | 1 | 0.844 | 0.727 | +0.149 | 0.125 |
| 300 | 54 | 17 | 4 | 0 | 0.152 | 0.007 | −0.129 | 0.305 |
| 600 | 48 | 23 | 4 | 0 | 0.066 | <0.001 | −0.158 | 0.083 |
| 900 | 60 | 11 | 4 | 0 | 0.414 | 0.118 | −0.098 | 0.358 |

Under the production frozen-cell convention b exceeds c at every temperature, but the asymmetry is
not resolved at 100 K (5 v 3) and is not significant at the system level at any temperature; it is largest at 600 K (23 v 4, clustered
p = 0.066), and by 900 K most labels have flipped to stable and the comparison loses its force.
Without ORB-v2 (60 pairs) the counts are 3 v 2, 14 v 2, 20 v 2 and 10 v 2, with system-clustered
p = 1.0, 0.156, 0.055 and 0.25 (ESI Table S15), so removing the one model with directly predicted
forces changes neither the direction nor the conclusion. The frozen-cell convention changes the
size: at 300 K the count runs from 7 v 4 to 24 v 4 across the conventions of §2.4, and at 100 K
three of them give b ≤ c (ESI §S1.3).

Where the discordances sit matters as much as how many there are. We give the breakdown
descriptively, because it was selected after seeing the outcome and is not a test. At 300 K, 13 of
the 17 b-units fall in three systems that the screen mis-calls for at least four of the five
models: BaTiO₃ (4), KNbO₃ (4) and CsSnBr₃ (5). For BaTiO₃ and KNbO₃ the screen's T* falls below the
experimental transition, so the cubic phase is called stable at 300 K where it is labelled
unstable; CsSnBr₃ is labelled stable at 300 K, only 8 K above its 292 K transition, and the screen
still condenses it. Without those three systems the count is 4 v 4. Three of those four b units
are PbTiO₃, the screen's ordering failure (below), and all four c units are structural: three are
control false-unstables that the harmonic layer reads off its interpolated mesh and the
commensurate screen cannot reproduce (CeO₂ and NaCl with CHGNet, MgO with ORB-v2; approximation A2),
and the fourth is ORB-v2 on SrTiO₃, which sees no instability at any temperature. At 600 K five
systems hold 22 of the 23. An error that four or five architectures share on the same system can
come from the single-mode screen itself (approximation A1 of Table 1) or from a surface error common
to the models, for instance one inherited from similar training data. Running the screen on PBE
energies along the same coordinates separates the two (§2.6; ESI §S5, Table S19): a mis-call that
persists there belongs to the screen, the functional or the label, and one that disappears belongs
to the MLIPs. For BaTiO₃ and KNbO₃ the shared 300 K mis-calls disappear. All eight b units of
CHGNet, MACE-MP-0, MatterSim and SevenNet-0 are called unstable on PBE, as labelled, and along the
coordinates that decide them these four models' wells are 0.32–0.76 of the PBE depth (median 0.53).
That is the PES softening of H1, measured directly on the coordinate that matters. For CsSnBr₃ the
mis-calls persist: the MLIP wells match PBE to within 20 % for three of the five models, every PBE
curve has its minimum at the edge of the scan, and the label sits 8 K above the transition. KNbO₃ at
600 K, called stable by all five models and by PBE against an unstable label (T_c 708 K), is the
cleanest error of the screen or the functional. Over all 120 ladder units of the six systems,
swapping only the energy engine from each MLIP to PBE, on identical modes and amplitudes, raises
agreement with the labels from 81 to 101 (23 corrected, 3 newly wrong; five systems improve and
none worsens). Ten of the corrections are bcc Zr along MatterSim's deciding coordinate, which
CHGNet, MACE-MP-0 and SevenNet-0 flatten to zero. The counts are descriptive, over six systems, and
PBE is evaluated at each MLIP's relaxed lattice and along its coordinate, so they separate
force-engine error from the screen's approximation without measuring either in isolation.

The second question is whether harmonic and finite-temperature correctness are *associated*,
which McNemar does not test. There the answer is that this design cannot say. The
system-clustered permutation p for φ, over 10,000 permutations of whole systems'
finite-temperature rows, is above 0.08 at every temperature. φ is positive at 100 K and negative
at the three higher ones, but at each of those three the both-wrong cell is **structurally
empty**, so the sign is forced by a zero rather than measured. A system-clustered bootstrap
interval excludes zero at 300, 600 and 900 K; we give that no weight either, because it describes
the precision of a point estimate whose sign is already known to be an artifact. With five models
over fifteen systems we claim no association in either direction. (The matched set excludes
systems whose name carries the bcc tag, which also removes the superionic AgI unit; the
denominator is therefore fifteen systems, not sixteen.)

Because b and c rest on harmonic calls, we bound the harmonic estimator's own
reproducibility noise directly rather than assuming it is small. The v1 and v2 harmonic generations
(§2.3) are the same algorithm at the same settings — 0.01 Å displacements, 2×2×2 supercells,
12×12×12 meshes — re-measured in independently pinned environments, giving 100 paired (system,
model) re-measurements. The spread is strongly model-dependent and must not be pooled: CHGNet's
largest deviation is 0.0076 THz and MatterSim's is 0.00057 THz, with zero stability-call flips
across all forty of their re-measurements, whereas ORB-v2's reaches 2.01 THz and flips two calls
(Table S1). One is the KTaO₃ unit already excluded as borderline; the other is bcc Hf, which reads
−0.089 THz in one generation, stable at the 0.1 THz tolerance, and −0.194 THz in the other,
unstable. Neither CHGNet's nor MatterSim's replicate noise comes near its margins: CHGNet's two
matched-set harmonic errors sit 0.138 and 0.167 THz from the tolerance, about twenty times its
maximum replicate deviation of 0.0076 THz, and MatterSim's maximum deviation is 0.00057 THz. What
noise exists is concentrated in ORB-v2, the model with directly predicted forces, and both of its
flips fall outside the matched set.

That bounds environment and library nondeterminism at fixed displacement amplitude. The
amplitude itself is a separate axis, measured for all five models by re-running the scored systems
at 0.005, 0.02 and 0.03 Å (ESI Table S13). MACE-MP-0, MatterSim and SevenNet-0 change no call at
any amplitude. CHGNet changes calls only on controls (CeO₂ and NaCl, the same marginal units the
tolerance sweep moves, and Cu at 0.005 Å only), so its harmonic accuracy runs from 14/19 to 17/19 across amplitudes and
we treat it as a quantity sensitive to both knobs rather than as a number. ORB-v2 changes calls on
test systems as well (HfO₂, SrTiO₃, CsSnI₃, Ti and Hf, besides MgO), with accuracy from 16/19 to
17/19. It is the only model whose forces are predicted directly rather than as gradients of the
energy; CHGNet, SevenNet-0 and MatterSim also return float32 forces as run and do not show this, so
precision alone does not explain it.

The defensible H2 statement is therefore an existence statement about transfer, not a statement
about correlation: harmonically correct units that the screen mis-calls at finite temperature
exist and outnumber the reverse in unit counts, so a harmonic benchmark does not certify a model's
finite-temperature behaviour. The asymmetry is not significant at the system level, and most of it
is shared across models (above), so we do not read it as a measured difference between the
models' potential-energy surfaces. Nor do we upgrade it to the claim that harmonic accuracy carries
no information about finite-temperature accuracy. That claim asserts a null, the tests above fail
to reject a null rather than showing one, and with fifteen systems and five models the design
cannot resolve whether a weak association exists, still less its sign.
On matched data MatterSim, MACE-MP-0 and SevenNet-0 are all harmonically perfect (15/15) and reach
25/30, 25/30 and 27/30 at finite temperature, so a model that is perfect at the harmonic level on
these systems can still have one unit in six mis-called by the screen. Because the intervals
overlap, this is an illustration and not a ranking claim. What it does support, and what matters
for practice, is negative and robust to the overlap: a top harmonic score on the npj/PhononBench
benchmarks does not by itself identify a model as a good finite-temperature screener. SevenNet-0 is
top-equal in *both* layers, tied with MatterSim and MACE-MP-0 harmonically on the matched set, so
the relationship is not an inversion and we do not claim one. ORB-v2 is the only model that does
not select the R-point instability of SrTiO₃ (§2.4); its finite-temperature accuracy, 24/30, is one
unit below MACE-MP-0 and MatterSim and not separated from any of the others. What this layer shows
is the per-family recall above and, in §3.3, the comparison with SSCHA, not a ranking of models.

CHGNet's harmonic accuracy is tolerance-dependent: 15/19 across the plateau tol ∈ [0.05, 0.20]
used throughout, rising to 17/19 at tol = 0.30, where two marginal false-unstables (CeO₂ at
−0.267 THz and NaCl at −0.238 THz) flip back; CHGNet is the uniquely lowest model only for
tol ∈ [0.05, 0.20]. We therefore state the tolerance-robustness rather than the number: CHGNet sits
below MatterSim harmonically at every tolerance in [0.05, 0.50], and at tol = 0 the two tie rather
than reversing. We do not pair this with a finite-temperature ordering, because those intervals
overlap. At tol = 0 all five models collapse to 0.63–0.74 as the Γ acoustic numerical zeros flood
the false-unstable count, and the comparison is degenerate.

### 3.3 SSCHA: agreement on bcc, and a local stability criterion on deep displacive wells

We ran multi-mode SSCHA on the three bcc metals, the four oxide perovskites, CsSnI₃ and the two
fluorites as the default cross-check of §2.5: a `ForcePositiveDefinite` start, a 2×2×2 cell,
automatic stochastic relaxation, and the bubble-level free-energy Hessian (`include_v4 = False`)
evaluated at the fixed high-symmetry reference. That is the recipe a practitioner escalating from a
cheap screen would run. SSCHA here runs on the same MLIP energies as the screen, so agreement
between the two is a consistency check, and a disagreement shows that the two criteria differ, not
which of them matches first principles. Nothing below is a claim that the SSCHA formalism is
wrong; the results concern the question its default stability criterion asks.

bcc metals: clean runs, and agreement weaker than it first looks. Across Ti/Zr/Hf × 5 models × 5
temperatures (75 units) SSCHA returned no failed unit and no blow-up, and every lowest
free-energy-Hessian frequency is positive, between 0.06 and 2.10 THz (Fig. 3; ESI Table S3). All
five models are already positive at 50 K, the lowest temperature on the SSCHA ladder, so any
dynamic-stabilisation temperature is left-censored at 50 K. It lies far below the thermodynamic
transition because bcc to hcp/ω is a martensitic, strain-coupled, first-order transition, which is
why bcc is excluded from scoring (§3.2). Three things limit what these numbers show. First,
MACE-MP-0 and CHGNet have no harmonic instability in bcc Zr or Hf (§3.1), so a positive SSCHA
frequency on those four (metal, model) pairs tests no thermal stabilisation. Second, several curves
soften rather than harden with temperature, the opposite of entropy stabilisation: Ti with
MACE-MP-0 falls from 1.73 to 1.38 THz between 50 and 600 K, Ti with SevenNet-0 from 1.27 to
1.18 THz and Hf with SevenNet-0 from 0.33 to 0.30 THz. Third, the production grid recorded no
convergence diagnostics (§2.5), so it cannot show how far each value moved from the
positive-definite starting matrix. Within those limits, ORB-v2 lies closest to the stability boundary (0.06–0.28 THz
on Ti and Hf), SevenNet-0 near 0.3 THz on Hf, and MatterSim, whose harmonic bcc instabilities are
the deepest of the five models (−1.96 to −2.21 THz), at 1.6–2.1 THz. The seed-reproducibility and
cell-size tests of §3.5 were run on Zr with MACE-MP-0, one of the four pairs with nothing to
stabilise, so they bound SSCHA's noise on that surface and not on a unit near a sign change.
<!-- PENDING-C1: MatterSim bcc-Zr 50 K four-seed result with its recorded diagnostics (spread of the lowest Hessian frequency across seeds, lowest frequency after ForcePositiveDefinite against the final Hessian value, gradient history, populations used, convergence flag); one sentence here, whichever way it falls, pointing to the new ESI table. -->
<!-- PENDING-C5: bcc-Zr SSCHA 2x2x2 against 3x3x3 re-measured in the pinned environment for MatterSim and MACE-MP-0; one sentence here, whichever way it falls. -->

Over the 45 units where both methods ran (Ti, Zr and Hf at 100, 300 and 600 K), the two stability
calls agree in 31/45 = 0.69 [0.54, 0.80] (25/36 = 0.69 without ORB-v2). Twelve of the agreeing
units are MACE-MP-0 and CHGNet on Zr and Hf, where agreement is trivial; on the other 33 the calls
agree in 19. MatterSim agrees with SSCHA on the call on none of its 9 units: its screen calls all
three metals unstable at 100, 300 and 600 K. The screen's symmetric-point curvature is not compared
with SSCHA by sign, because for a single mode it is positive by construction (§2.4); the ten bcc
units on which it is negative are numerical (six width-solver fallbacks for MatterSim, one fit
artefact for ORB-v2 on Ti, and three Hf units in which ORB-v2 has no screened imaginary mode; ESI
Table S16). The frequency magnitudes show little rank correlation (Spearman ρ = 0.11, −0.003
without ORB-v2, given descriptively and without a test because the pairs cluster by system and
model), as expected of two different observables (Fig. 4; ESI Table S16).

![**Fig. 3** Multi-mode SSCHA on bcc Ti, Zr and Hf, one panel each: the lowest free-energy-Hessian
frequency, the default criterion evaluated at the bcc reference, against temperature for the five
models (§3.3). SSCHA runs on each model's own energies. Dashed lines with open markers are MACE-MP-0
and CHGNet on Zr and Hf, which have no harmonic instability on their own surfaces, so a positive
frequency there tests no thermal stabilisation. Every unit is positive from 50 K, the lowest
temperature computed, so any stabilisation temperature is left-censored there. In the Hf panel
ORB-v2 lies closest to the boundary (down to 0.06 THz), SevenNet-0 near 0.30 THz, MACE-MP-0
(dashed) near 1.2 THz and MatterSim near 1.6 THz. Some curves soften with temperature, for
example Ti with MACE-MP-0, from 1.73 to 1.38 THz between 50 and 600 K.](../results/figures/fig_sscha_bcc.png)

![**Fig. 4** The soft-mode screen's symmetric-point curvature frequency against the SSCHA lowest
free-energy-Hessian frequency on bcc Ti, Zr and Hf at 100, 300 and 600 K (45 paired units; §3.3).
Both are computed on the same MLIP energies, so agreement is a consistency check, not a
first-principles test. The stability calls agree in 31/45 = 0.69 [0.54, 0.80]; the screen's call
is its free-energy comparison, not its curvature (§2.4). The screen's curvature is positive by
construction wherever its width equation is solved, so its negative values here are numerical
(ESI Table S16) and quadrant agreement is not interpreted. Open markers are the 12 units with no
harmonic instability on that model's surface, where agreement is trivial. The
magnitudes show little rank correlation (Spearman ρ = 0.11, descriptive). A point beyond the
plotting window is drawn at its edge and labelled with its value.](../results/figures/fig_method_agreement.png)

Displacive systems: SSCHA's default criterion calls deep wells stable. On the ferroelectric oxide
perovskites at T ≤ 300 K, below every transition temperature, the screen recovers 16/30
instabilities (§3.2). SSCHA's default criterion recovers 5 of the 27 units that return a number
(5/27 = 0.185 [0.082, 0.367]), all five of them PbTiO₃ units; of the 30 attempted units one blew
up numerically and two failed. It calls cubic BaTiO₃ and KNbO₃ stable in every unit at
T ≤ 300 K (Fig. 5). On the fluorites the contrast is sharper: 20/20 for the screen against
1/20 = 0.05 [0.01, 0.24] for SSCHA (16/16 against 0/16 without ORB-v2). The false-stable is not
universal across displacive families. On CsSnI₃, the one halide perovskite in the SSCHA grid,
SSCHA calls the cubic phase unstable in 7 of 8 units at T ≤ 300 K.

Both methods see the same (system, model, temperature) units, so the contrast is paired. Over the
combined displacive set at T ≤ 300 K, the three ferroelectric oxides and the two fluorites (47
units with a number from both methods), the screen is right where SSCHA is wrong on 33 units and
SSCHA is right where the screen is wrong on 3 (26 against 3 without ORB-v2). Those 47 units are
five systems seen by five models at two temperatures, so we test at the level of systems:
randomising the method label over whole systems, which is exact at this size, gives p = 0.125
with or without ORB-v2. Four of the five systems favour the screen (net discordances +6 on
BaTiO₃, +6 on KNbO₃, +9 on ZrO₂ and +10 on HfO₂), and PbTiO₃ favours SSCHA by one. With five
systems the smallest attainable two-sided p is 2/2⁵ = 0.0625, so no arrangement of these data
could reach conventional significance at the system level; the ferroelectric oxides alone (three
systems, p = 0.5) and the fluorites alone (two systems) cannot resolve anything. The contrast is
large, one-directional and consistent across four of five systems, and it is not significant. It
also depends on the screen's frozen-cell convention, since the screen's ferroelectric recall runs
from 5/30 to 28/30 across the conventions of §2.4. ESI Table S10 gives each system set with and
without ORB-v2.

![**Fig. 5** Recall of the unstable cubic phase on the ferroelectric oxide perovskites (BaTiO₃,
KNbO₃, PbTiO₃) at T ≤ 300 K, below every transition temperature: the soft-mode screen under its
production frozen-cell convention (16/30) against SSCHA's default criterion (5/27), with Wilson 95%
intervals (§3.3). SSCHA's denominator excludes one numerical blow-up and two failed runs. Both
methods see the same units, so the comparison is paired; its system-clustered test (p = 0.125 on
the combined displacive set, 0.5 on these three systems alone) is in the text and ESI Table S10,
and two marginal intervals are not that test.](../results/figures/fig_displacive_recall.png)

Why the default criterion misses these wells. The labels record whether the high-symmetry phase is
the equilibrium phase at T, which is a global question about the free energy. The default SSCHA
criterion is the curvature of the free energy at the symmetric reference, which asks a local one:
is that phase a local minimum? For a shallow double well the two answers coincide. For a deep one
they need not, because within a Gaussian (SCHA) trial state the transition becomes
first-order-like (§2.4): the free energy keeps a local minimum at the symmetric point while a
displaced minimum lies lower, and a criterion read at the fixed reference then reports stable.
On the same MLIP energies the screen finds that situation on most of the units in question. Of the
57 non-bcc units, over the whole temperature ladder, where SSCHA calls the phase stable against an
unstable label, the screen's free-energy comparison finds a lower displaced minimum and calls the
phase unstable on 46 (46/57 = 0.81 [0.69, 0.89]; 39/50 without ORB-v2; ESI Table S16), while its
symmetric point remains a local minimum. That the symmetric point stays a local minimum is not
itself evidence: for a single mode the screen's symmetric-point curvature equals its
self-consistent trial stiffness and is positive by construction (§2.4). Cubic BaTiO₃ with
MACE-MP-0 at 100 K is a typical case: screen call unstable, SSCHA +2.87 THz. Where the screen gets
these units right, it does so because it minimises over the centroid instead of reading a
curvature; a fixed-reference Hessian has no such step, and relaxing the SSCHA centroid from a
displaced start would ask the global question directly, which we did not do. On these systems
the SSCHA false-stables are therefore consistent with what a local criterion returns for a deep
double well. They do not show that this is why the SSCHA Hessian is positive: a Hessian evaluated
after a relaxation that stopped near its positive-definite starting matrix is not evaluated at the
SCHA minimum and can read stable for that reason alone (§2.5; ESI §S2.1), and an MLIP error on
the sampled configurations is a third possibility (below).
<!-- PENDING-C1: the converged-mode runs (C1c) decide between the local-criterion account and early stopping for BaTiO3 and ZrO2 at 100 K; one or two sentences here, whichever way it falls, and restate the paragraph if the false-stables do not survive a converged relaxation. -->

This account is consistent with, and more specific than, the published analysis of the SSCHA
Hessian. The auxiliary SCHA matrix **Φ** is positive definite by construction, since a
normalisable Gaussian trial state requires it, so its eigenvalues cannot signal an
instability.^21,22^ The object that can is the free-energy Hessian,
∂²F/∂**R**∂**R** = **Φ** + **Φ**⁽³⁾**Λ**[**1** − **Φ**⁽⁴⁾**Λ**]⁻¹**Φ**⁽³⁾, derived to detect
displacive transitions from a fixed high-symmetry reference and demonstrated on SnTe and GeTe.^21^
The bubble approximation, the `python-sscha` default, drops **Φ**⁽⁴⁾; it breaks down for the cubic
phases of metal halide perovskites, where **Φ**⁽⁴⁾ matters around the transition and the full
resummation is needed to determine phase stability.^22^ The local-versus-global difference adds
that where a well is deep enough for the free energy to keep a local minimum at the symmetric
point, a curvature read there reports stable at any truncation order: the fourth-order term changes
the estimate of the curvature, not the question it answers. Whether including it would flip the
sign on these units is untested here.
<!-- PENDING-C1b: include_v4 = True free-energy Hessian on BaTiO3/MACE-MP-0/100 K (finished or not, value, wall time); one sentence here, whichever way it falls. -->
The account does not rest on the stage-by-stage diagnostic of ESI Table S2 (BaTiO₃, MACE-MP-0,
100 K), a single run that was not repeated in the pinned environments. Its rows are the harmonic
minimum (−5.64 THz), the positive-definite starting matrix (+2.88 THz), the auxiliary matrix at
the end of the relaxation (+2.89 THz), which is positive definite by construction and so says
nothing about stability, and the bubble-level Hessian (+2.87 THz). The run with the fourth-order term
(`include_v4 = True`) did not finish: it had run for more than 18 min on this one unit, against
about 200 s without the term. No two rows differ in truncation order alone, so the table does not
isolate the truncation as the cause. Whether an MLIP error on the thermally sampled configurations
also contributes to the false-stables is a separate question, which PBE forces on those
configurations address (ESI §S5, Table S20). They give no sign of it: on twelve configurations each
from the BaTiO₃ and ZrO₂ ensembles at 100 K (0.09 Å root-mean-square displacement), MACE-MP-0's
force error relative to the PBE forces is 0.10 and 0.18, against 0.10 and 0.23 on rattled
near-equilibrium cells.

The fluorites show the same pattern on a shallower, numerically well-behaved instability. Cubic
ZrO₂ and HfO₂ are the high-temperature phases (above about 2570 and 2800 K), unstable through the
X-point oxygen mode at every temperature studied. The screen calls all 20 fluorite units at
T ≤ 300 K unstable (38 of 40 over the ladder; MACE-MP-0 calls HfO₂ stable at 600 and 900 K).
SSCHA calls all ten units stable at 100 K, at +1.9 to +3.3 THz, although in the same 2×2×2 cell
the harmonic layer finds all ten unstable (−3.8 to −10.6 THz; §3.5). On the 31 fluorite units
SSCHA false-stabilises over the ladder, the screen's free-energy comparison calls all 31
unstable. Across both fluorites SSCHA produced no
blow-up and no failed unit, so the fluorite false-stable is not a numerical blow-up of the kind
seen on the deepest perovskite wells. With temperature SSCHA then destabilises ZrO₂ for three
models (MACE-MP-0 −2.7, SevenNet-0 −1.3 and ORB-v2 −24.8 THz at 900 K) and HfO₂ for two
(MACE-MP-0 −4.6 and ORB-v2 −10.0 THz), while CHGNet and MatterSim stay positive on both. The
fluorite labels are unstable at every temperature, so those negatives have the right sign, but the
trend with temperature runs the wrong way, as in the pattern below.

SSCHA false-unstables that grow with temperature. At high temperature SSCHA errs in the other
direction. Over the non-bcc units whose label is stable at that temperature, it calls the phase
unstable in 5/5 at 300 K, 10/13 at 600 K and 15/19 at 900 K (4/4, 8/11 and 12/16 without ORB-v2;
no non-bcc label is stable at 100 K), in SrTiO₃, BaTiO₃, KNbO₃, PbTiO₃ and CsSnI₃ (ESI
Table S17). SrTiO₃ is false-unstable in all 14 returned units above its 105 K transition, at −3.4
to −914 THz (11/11, down to −60.5 THz, without ORB-v2), against harmonic R-mode frequencies of −0.8
to −2.1 THz on the models that have one; SSCHA never calls SrTiO₃ stable at any temperature. Of the
23 non-bcc units SSCHA calls stable at 100 K, 14 turn negative by 600–900 K (10 of 19 without
ORB-v2); nine of those are false-unstables, and the other five are the fluorite units above, where
the sign is right and the trend is not. An instability that grows with thermal amplitude on a fixed
potential is the opposite of entropy stabilisation. It is what an MLIP extrapolating on
large-amplitude configurations, far from its training data, would produce; an instability of the
stochastic sampling itself, and a Hessian evaluated at an auxiliary matrix that has not reached
the SCHA minimum (§2.5), are the other candidates. PBE forces on twelve configurations from the
SrTiO₃ 600 K ensemble (ESI §S5, Table S20), at 0.34 Å root-mean-square displacement, show the
extrapolation error is real there: MACE-MP-0's relative force error rises from 0.11 near
equilibrium to 0.19, energy errors reach 44 meV per atom, and all five models fall in the same range
(0.12–0.20). Whether that error, rather than the sampling, drives the runaway is not settled by
twelve configurations from one unit.

Numerical failures. The SSCHA grid holds 208 of the 215 (system, model, temperature)
combinations its ten systems and two temperature ladders allow: ORB-v2 was not run on
CsSnI₃, and MACE-MP-0 on PbTiO₃ and SevenNet-0 on CsSnI₃ and SrTiO₃ were not run at 900 K (ESI
Table S17). Of the 208 attempted units, 201 returned a number. Seven stopped at
cellconstructor symmetry or ensemble assertions: ORB-v2 on PbTiO₃ at every temperature, MatterSim
on PbTiO₃ at 600 and 900 K, and SevenNet-0 on CsSnI₃ at 600 K. Eight returned blow-ups
(|f| > 50 THz): three for ORB-v2 (SrTiO₃ at 300, 600 and 900 K), three for MatterSim (PbTiO₃ at
300 K, CsSnI₃ at 600 and 900 K), one for CHGNet (SrTiO₃ at 900 K) and one for SevenNet-0 (PbTiO₃
at 900 K) (ESI Table S17). They fall on the deepest wells and on the high-temperature units above,
and together they involve every model except MACE-MP-0. In the superseded v1 grid retained in the
ledger, the seven failed units had returned values of −0.3 to −3653 THz instead of stopping; the
most extreme v1 value, about −2 × 10⁶ THz, belongs to SrTiO₃ with ORB-v2 at 600 K, which returns
−544.5 THz in the production grid (ESI §S2.3).

### 3.4 Ensemble disagreement as a guardrail (H3)

Because no single model is reliable across the set, we test whether cross-model disagreement flags
the units where the majority-vote consensus finite-temperature call is wrong. On the non-bcc,
non-borderline soft-mode units (n = 60; bcc excluded because its thermodynamic-T_c label is the
wrong reference for dynamic stability, §3.3), the majority-vote consensus is wrong on 12/60 =
0.200 [0.118, 0.318] of units. That set is fifteen systems × four temperatures, the fifteen
systems of the matched set above (ESI Table S7), and each unit already collapses the five model
votes, so the units are clustered by system and are not independent. Every test below resamples
whole systems rather than units.

ESI Table S8 lists every unit in that set. With all five models, the consensus is wrong on
8/14 = 0.571 [0.326, 0.786] of the units where the models split on the stable/unstable call,
against 4/46 = 0.087 [0.034, 0.203] of the unanimous units, a 6.6-fold enrichment. As a ranked
predictor of consensus error the vote split reaches AUC 0.762. A permutation test over whole
systems (10,000 permutations) gives p = 0.0035, and a cluster bootstrap gives a 95% interval of
[0.590, 0.934]. The naive unit-level permutation gives p = 0.0003, so treating the units as
independent would overstate the significance by a factor of about 12. Part of the enrichment is
built into the construction, since a unanimous unit can be wrong only when all five models are,
and the permutation null does not remove that part.

Without ORB-v2 the signal weakens. Only 8 units then split, against 14 with all five models; the
consensus is wrong on 4/8 = 0.500 [0.215, 0.785] of them and on 8/52 = 0.154 [0.080, 0.275] of the
unanimous units, and the vote-split AUC falls to 0.628 (system-clustered p = 0.047) with a
cluster-bootstrap interval, [0.438, 0.839], that spans 0.5. With four models three units tie 2–2;
the rule breaks a tie to "stable", and two of the three (PbTiO₃ at 100 and 300 K) are then wrong.
The vote split is therefore a suggestive guardrail, not one that is robust to removing ORB-v2.

For the continuous cross-model standard deviation of the screen's frequency no signal could be
resolved (AUC 0.361, clustered p = 0.275, 95% interval [0.046, 0.625]; without ORB-v2, 0.299
[0.076, 0.516]; Fig. 6). We state that as a failure to resolve rather than as an absence. The
interval with all five models spans 0.5, so the point estimate below chance does not make the
spread an inverted predictor, and it reaches 0.625, so these data do not exclude a moderately
useful one either. This spread is also not the force-level disagreement that an uncertainty
estimate usually means. It is a PES-level proxy, partly derived from forces: the E(Q) maps are
energies only, and forces enter only through each model's force constants, which fix the mode
patterns along which E(Q) is sampled. As pre-registered, H3 compares ensemble disagreement with a
single model's self-reported signal; that comparator was not run, so only the ensemble half of H3
is tested here. A force-level test was pre-registered separately (`scripts/force_spread.py`): the
spread of the five models' forces on thermally displaced configurations of the same 60 units, with
within-architecture committees (MACE-MP-0 small, medium and large; MatterSim 1M and 5M) as
secondary scores, the same system-clustered AUC and bootstrap, every score with and without
ORB-v2, and a check that no result comes from configurations in which atoms overlap. Each unit
gets 16 configurations drawn from the quantum harmonic distribution of the five models' mean
force constants, the same for every model, and the spread is the root-mean-square over atoms and
components of the across-model standard deviation of the forces, median over configurations.
The primary AUC is 0.681 [0.416, 0.908] (system-clustered p = 0.104), and 0.684 [0.413, 0.916]
without ORB-v2, so by the rule fixed in advance the force spread is not shown to flag
untrustworthy calls. Of the 19 secondary scores (ESI Table S18), five have a lower bound above
0.5: the MACE-MP-0 and MatterSim committees against the consensus (0.717 [0.535, 0.902] and
0.726 [0.516, 0.901]), the MACE-MP-0 committee against MACE-MP-0's own calls (0.722 [0.564,
0.873]), and the leave-one-out scores of MACE-MP-0 and SevenNet-0 without ORB-v2 (lower bounds
0.502). None is adjusted
for multiple comparisons, and all use the same configurations. The normalised spread, the check
fixed in advance against softness and amplitude, gives 0.578 [0.335, 0.795]. The overlap check
is the reason for caution: restricted to the 43 units in which no configuration brings two atoms
closer than 0.75 of the reference distance for that species pair, the primary AUC is 0.314
[0.146, 0.529], and those units hold only 4 of the 12 consensus errors. The soft halide and
ferroelectric spectra that carry most errors are the ones whose thermal draws push A-site cations
into the anions, where every model is extrapolating, so the positive estimates over the full set
rest largely on units in which the spread measures that extrapolation; the restriction was
pre-registered for the primary only and was not applied to the secondary scores. With fifteen
systems these data do not exclude a useful force-level signal, and they do not show one.

![**Fig. 6** Ensemble-disagreement guardrail (§3.4), with all five models and without ORB-v2: the
majority-vote consensus finite-temperature error rate on units where the models split on the
stable/unstable call and on units where they are unanimous (k/n with Wilson 95% intervals), and
the AUC of the vote split and of the cross-model frequency spread as predictors of consensus error,
with cluster-bootstrap 95% intervals over systems. With all five models the vote split gives AUC
0.762 [0.590, 0.934] (system-clustered p = 0.0035); without ORB-v2 it gives 0.628 [0.438, 0.839],
an interval that spans chance, with three tied 2–2 units called stable. For the frequency spread no
signal could be resolved in either set (0.361 [0.046, 0.625]; 0.299 [0.076, 0.516]).](../results/figures/fig_ensemble_guardrail.png)

In practice a split vote among foundation MLIPs is worth flagging, but the signal depends on which
models are in the ensemble, and a unanimous call is not clearance: 4 of the 46 unanimous units are
wrong, and 8 of 52 without ORB-v2. The physical reading is that disagreement concentrates near the
stability boundary, where the call is most uncertain.

### 3.5 Robustness: stochastic noise and finite size

Stochastic reproducibility. Repeating the bcc-Zr / MACE-MP-0 / 100 K SSCHA with four independent
random seeds gives a lowest free-energy-Hessian frequency of +1.798 ± 0.001 THz (range +1.797 to
+1.800 THz), and the same unit re-measured in the pinned environment, with the SSCHA initialiser
built from phonopy force constants, returns +1.79 to +1.80 THz at every temperature on the ladder. The test
is narrower than it looks. MACE-MP-0 finds no harmonic instability in bcc Zr (§3.1), so this unit
measures SSCHA's stochastic noise on a surface with nothing to stabilise. It says nothing about the
noise on a unit near a sign change, and it does not by itself show that the cross-model bcc margins
of §3.3 are signal rather than noise. A four-seed study that records the convergence diagnostics
(populations used, convergence flag, gradient history, and the spread of the lowest Hessian
frequency across seeds) extends the test to units that do carry an instability: bcc Zr with
MatterSim at 50 K, BaTiO₃ and ZrO₂ with MACE-MP-0 at 100 K, and SrTiO₃ with MACE-MP-0 at 600 K
(`scripts/sscha_seed_study.py`).
<!-- PENDING-C1: seed-study results for Zr/MatterSim 50 K, BaTiO3/MACE-MP-0 100 K, ZrO2/MACE-MP-0 100 K and SrTiO3/MACE-MP-0 600 K: per-unit seed spread of the lowest Hessian frequency, whether seed 0 reproduces the ledger value, populations used and convergence flag, and whether any call changes across seeds; two or three sentences here, whichever way it falls, pointing to the new ESI table. -->

Finite size. For bcc Zr with MACE-MP-0 the SSCHA call holds from 2×2×2 to 3×3×3 (+1.80 to
+1.56 THz at 100 K, +1.80 to +1.55 THz at 300 K; `results/convergence_study.parquet`). Three
caveats limit it. Those SSCHA rows come from the superseded June v1 generation, run in an
unpinned environment and before the phonopy-built initialiser, and they are again MACE-MP-0 on Zr,
which has no instability, so they show only that a surface with nothing to stabilise stays stable
in a larger cell. The two cells are also not nested: the bcc N point is commensurate only with
even cells and the ω point, q = ⅔⟨111⟩, only with multiples of three, so going from 2×2×2 to
3×3×3 changes the set of **q**-points the cell contains as well as its size. The soft-mode rows of
the same study are superseded as well: they were produced with a single softest-mode selection,
which is not the screen of §2.4, and are kept for audit only.
<!-- PENDING-C5: bcc-Zr SSCHA 2x2x2 against 3x3x3 re-measured in the pinned environment for MatterSim and MACE-MP-0; replace or confirm the numbers in this paragraph, whichever way it falls. Report it as a change of q-set, not a convergence test (N only in 2x2x2, omega only in 3x3x3), or add a nested 4x4x4 (and 6x6x6 for omega) run; a call that changes at 3x3x3 may be the omega mode entering the cell. -->

Zone-boundary systems. The SrTiO₃ antiferrodistortive instability lives at the zone-boundary
R point, (½,½,½), which is commensurate only with even supercells, so a 3×3×3 cell cannot contain
it and a 2×2×2-versus-3×3×3 comparison is not a convergence test for it. The fluorite X-point mode
has the same problem: the 2×2×2 cell contains it and the 3×3×3 cell does not. **The zone-boundary
systems therefore have no supercell-convergence test in this work, and we make no convergence
claim for them.** The even-cell test (4×4×4, a ~320-atom SSCHA) was not run.

Three narrower checks remain, and none of them is a convergence test. For the fluorites, a
missing **q**-point cannot be what produces the SSCHA false-stable: in the same 2×2×2 cell and
with the same force engine, the harmonic calculation finds all ten fluorite units unstable (−3.8 to
−10.6 THz), while SSCHA at 100 K finds all ten stable (+1.9 to +3.3 THz), so the cell contains the
instability that SSCHA calls stable. That is a finite-size argument and nothing more: it does not
show that the 2×2×2 result is converged, and it does not distinguish the local-criterion account
of §3.3 from an MLIP error on the sampled configurations. For BaTiO₃ the instability includes the
zone-centre (Γ) mode, which every supercell contains, so a missing **q**-point cannot produce its
false-stable either. For SrTiO₃ the objection does not arise: SSCHA never calls it stable at any
temperature, and a missing zone-boundary **q**-point could only hide an instability. What SrTiO₃
shows instead is the high-temperature divergence reported in §3.3.

## 4. Discussion

The harmonic layer (§3.1) reproduces the published picture and is consistent with H1: PES
softening appears as false-stable calls localised to specific instabilities rather than as a
uniform rate. MACE-MP-0 and CHGNet read bcc Zr and Hf at −0.000 THz (2/13 false-stable each),
while MatterSim and SevenNet-0 reproduce every documented soft mode.

For H2 the data support an existence statement about transfer and no more. Harmonically correct
units that the screen mis-calls at finite temperature exist and outnumber the reverse, 17 to 4 at
300 K on the matched set under the screen's production frozen-cell convention (7 to 24 against 4
under the others, §2.4). The asymmetry is large and one-directional in unit counts but not
significant once units are clustered by system (p = 0.152 at 300 K, 0.066 at 600 K), and 13 of the
17 sit in BaTiO₃, KNbO₃ and CsSnBr₃, three systems the screen mis-calls for at least four of the
five models. Sharing does not make them the screen's. On PBE energies along the same coordinates
the eight BaTiO₃ and KNbO₃ units are called correctly, with the MLIP wells at about half the PBE
depth, while the five CsSnBr₃ units stay mis-called (§3.2). So at least eight of the 17 are a
softening that four architectures share, and the five that persist are the screen's, the
functional's or the label's.
What survives is the point the title makes, and it does not depend on which layer is at fault in a
given unit: a top harmonic score does not certify the finite-temperature calls made with a model.
MatterSim, MACE-MP-0 and SevenNet-0 are harmonically perfect on the matched set and still have 5,
5 and 3 of their 30 finite-temperature units mis-called. The pre-registered form, that harmonic
and finite-temperature accuracy are uncorrelated, is left open: once pairs are clustered by system
no association is resolved at any temperature, and a failure to reject is not a demonstration. Nor
does the design separate the models at finite temperature. Their accuracies run from 24/30 to
27/30, a spread of three units on a denominator of thirty, and every interval overlaps every other.

The usual escalation, "if in doubt, run SSCHA", does not settle the displacive cases, and the
reason is the question its default criterion asks. The free-energy Hessian at the symmetric
reference tests whether the high-symmetry phase is a local minimum of the free energy; the labels
record whether it is the equilibrium phase. For deep double wells the two differ. On 57 non-bcc
units SSCHA's default criterion calls stable a phase labelled unstable, and on the same MLIP
energies the screen's free-energy comparison finds a lower displaced minimum on 46 of them.
Whatever the cause in a given unit, a positive SSCHA Hessian at the symmetric structure answers
the local question only, and the global question has to be asked separately, for instance by comparing the free energies of the symmetric and
displaced structures, which the screen does one mode at a time. On the displacive systems at
T ≤ 300 K the screen's free-energy comparison is right where SSCHA's default criterion is wrong on
33 units against 3. That contrast is consistent across four of five systems but not significant
(system-clustered p = 0.125), and on the ferroelectric oxides its size depends on the screen's
frozen-cell convention (§2.4). Whether the production SSCHA relaxations reached the SCHA minimum
was not recorded, and a Hessian read before they do can also report stable (§3.3).
<!-- PENDING-C1: one sentence on what the seed study and the converged-mode runs (C1c) show about the false-stables, whichever way it falls. -->
On the bcc metals, whose labels are not dynamic-stability labels, the two methods agree on the
call in 31/45 units, twelve of those agreements trivial.

Scope: foundation models as shipped. Every result here is for the released checkpoints used
without adaptation. That is the regime of generative-CSP screening: a practitioner filtering
candidates does not know in advance which are strongly anharmonic, so fine-tuning on the
candidates themselves is not available before the screen, and the screening decision is taken with
the model as shipped, which is the setting of PhononBench and comparable filters. Fine-tuning on
broad phonon data is available, and such a model has been deployed at screening scale;^41^ this
benchmark does not test one, and a phonon-fine-tuned checkpoint is the natural next model to put
through it. For a *known*
anharmonic material fine-tuning is the right step, and the literature shows what it does.
Universal MLIPs, CHGNet and MACE-MP-0 among them, systematically soften the potential-energy
surface, and fine-tuning on a small number of reference calculations removes much of that
softening.^40^ Fine-tuning a universal potential on harmonic phonon data cut its phonon-frequency
error about fourfold, and the tuned model was then used for SSCHA across 4669 inorganic
compounds.^41^ Partially frozen fine-tuning of MACE-MP reached the accuracy of a potential trained
from scratch with 10–20% of its data, including harmonic phonons and quasi-harmonic free energies,
and reproduced the phonon dispersions of α-, β- (bcc) and ω-Ti qualitatively.^42^ On PbTiO₃,
fine-tuning MACE on a small PBEsol dataset recovered the ferroelectric–paraelectric transition,
with the Curie temperature still about 160 K below experiment.^20^ Fine-tuning four of the five
checkpoints tested here (MACE-MP-0, SevenNet-0, a large MatterSim checkpoint and ORB-v2) on
system-specific first-principles data cut their force errors 5–15-fold and brought the
architectures to comparable accuracy, on diffusion, structure and energy pathways rather than
phonons.^43^ These results bear on our
findings unevenly. Fine-tuning could change the per-model inputs: the harmonic calls of §3.1 (the
flattened Zr and Hf instabilities are the kind of softening these studies address), the well
depths and shapes the screen reads, and so the per-model screen accuracies and the H2 counts built
from them. Whether it would remove the mis-calls shared across models, which carry most of the H2
count, is what a screen run on the PBE surface tests (§3.2), since that is the surface a PBE
fine-tune aims at, along the same coordinates (the mode pattern still comes from each MLIP's force
constants). On that surface the BaTiO₃ and KNbO₃ mis-calls disappear and the CsSnBr₃ ones do not,
so a fine-tune that deepened the softened ferroelectric wells toward PBE would plausibly remove
about half the shared count and leave the rest.
Fine-tuning could change which units fall in the deep-well regime of §3.3, since that depends on
the well depth, but not the locality of the criterion: a curvature read at the symmetric reference
reports stable on any surface whose double well is deep enough, fine-tuned or not. Nor could it
change the screen's
dependence on its frozen-cell convention (§2.4). And fine-tuning moves a model toward its reference
functional, not toward experiment, so whatever that functional gets wrong about a transition
remains. How much of the finite-temperature gap fine-tuning closes for each family, and with how
much data, is beyond what a benchmark of released checkpoints can say.

On model-reported uncertainty. A natural proposal is to use the models' own uncertainty on their
force predictions to decide where their stability calls should not be trusted. The principle is
well founded: committee spreads propagated into molecular dynamics give more resilient
trajectories,^33^ a potential's own predictive uncertainty is a sound trigger for acquiring new
reference data,^32,38^ and uncertainty-driven active learning avoids unphysical MLIP dynamics in
strongly anharmonic materials.^30^ The released checkpoints come as single models rather than
trained committees, so §3.4 tests the cross-architecture version, with the five models as the
ensemble. The discrete vote split flags consensus errors with all five models (AUC 0.762,
system-clustered p = 0.0035) but not robustly: without ORB-v2 it falls to 0.628, with an interval
that spans chance. For the one continuous metric tested, the cross-model spread of the screen's
frequency, no signal could be resolved (AUC 0.361, 95% interval [0.046, 0.625]), which does not
exclude a moderately useful predictor. That metric is a PES-level proxy rather than the force-level
disagreement the proposal names, and the pre-registered comparison with a single model's own signal
was not run. The force-level test of §3.4, with within-architecture committees for MACE-MP-0 and
MatterSim, is the direct measurement, and by its pre-registered rule it does not show that the
force spread flags untrustworthy calls (AUC 0.681 [0.416, 0.908]; 0.684 without ORB-v2). Some
secondary scores, among them the MACE-MP-0 committee against MACE-MP-0's own calls (0.722 [0.564,
0.873]), have intervals above 0.5, but they are unadjusted, share configurations, and were not
restricted to the configurations without overlapping atoms, on which the primary falls to 0.314.
Model-reported uncertainty is also a property of models as shipped: fine-tuning that brings the
architectures to comparable accuracy^43^ would remove much of the disagreement it relies on.

Limitations. Reference transition temperatures are approximate and scoring is qualitative. The
labels mark thermodynamic, often first-order, transitions, whereas the screen and SSCHA probe
dynamic stability; the two coincide for a continuous soft-mode transition and can differ for a deep
double well, which is where the local SSCHA criterion and the labels part company (§3.3). The
screen's T* and its ferroelectric recall depend on the frozen-cell convention (recall 5/30 to 28/30
across the conventions of §2.4), and accuracy on these labels cannot choose among the conventions.
The screen treats each mode on its own, and we have no quantitative bound on mode–mode coupling; it
is the most likely reason the predicted stabilisation temperature fails to order PbTiO₃ (§3.2),
and a likely source of the mis-calls shared across models. Both methods run on the same MLIP
potential-energy surface, so their agreement is a consistency check and SSCHA is not an independent
reference. What does not depend on the surface is the experimental labels, including the SrTiO₃
transition the gate is scored against (three of the five models pass it), and the argument of
§3.3, which concerns the criterion rather than the surface. None of these checks the fitted double
wells themselves, or the MLIP on the large-amplitude configurations SSCHA samples at high
temperature. First-principles energies along
the deciding soft-mode coordinates, and first-principles forces and energies on SSCHA-sampled
configurations, separate those errors from the approximations of the two methods. We computed
both for six systems (§2.6; ESI §S5). Along the deciding coordinates the MLIP wells are shallower
than PBE's for BaTiO₃, KNbO₃ and ZrO₂ (about half the depth for the four conservative models),
comparable for CsSnBr₃, and absent for bcc Zr in the three models that flatten it; with PBE energies
the screen agrees with the labels in 101 of 120 units against 81 with the MLIPs. On SSCHA-sampled
configurations at 50–100 K the MLIPs' relative force error is no larger than near equilibrium, so
those configurations show no extrapolation error, though they come from relaxations that did not
converge and may not reach the double well, where the error would matter. At 600 K in SrTiO₃, where sampling runs to
0.34 Å root-mean-square displacements, it rises 1.7-fold and energy errors reach 44 meV per atom.
The PBE checks rest on six systems, at each MLIP's relaxed lattice, and do not cover the remaining
fourteen.
SSCHA cells are 2×2×2, the only cell-size test is on a bcc unit with no instability and compares
cells that are not nested, and we make no convergence claim for the zone-boundary systems (§3.5);
the production SSCHA grid did not record per-unit convergence diagnostics, and its step cap counts
steps over the whole relaxation rather than per population (§2.5).
<!-- PENDING-C5 / PENDING-C1: once the pinned-environment bcc-Zr 3x3x3 re-measurement (MatterSim, MACE-MP-0) and the four-unit seed study land, rewrite "the only cell-size test is on a bcc unit with no instability" and add the seed-study coverage and diagnostics to this limitation, whichever way they fall. -->
ORB-v2 is the only model with directly predicted,
non-conservative forces. It shows in both layers, and the headline rates are given with and
without it (ESI Tables S9 and S15–S18). Both the screen (16/30) and SSCHA (5/27) miss a substantial
fraction of the ferroelectric-oxide instabilities, so neither is a solved method. Finally, α-AgI is
modelled as an ordered CsCl-type approximant of the bcc iodine sublattice; that is not the
disordered superionic α phase, and its screen call should be read as a placeholder.

## 5. Conclusions

We tested five foundation MLIPs, as released and without fine-tuning, on 20 systems whose
high-symmetry phase is harmonically unstable but thermally stabilised, together with stable
controls. Three findings come out of it, each stated at the strength that an analysis clustered
by system supports.

Harmonic correctness does not certify finite-temperature calls. On the matched set, 17
harmonically correct units are mis-called by the soft-mode screen at 300 K against 4 the other
way under its production frozen-cell convention (7 to 24 against 4 under the others). The
asymmetry is large and one-directional in unit counts but not significant once units are
clustered by system, and most of it sits in three systems the screen mis-calls for at least four of
the five models, so it establishes that such units exist, not a measured difference between the
models' surfaces. On PBE energies along the same coordinates the BaTiO₃ and KNbO₃ mis-calls
disappear, the four models' wells there being about half the PBE depth, and the CsSnBr₃ ones
persist.

The default SSCHA cross-check does not supply that certification for deep displacive wells. Its
free-energy Hessian at the symmetric reference asks whether that phase is a local minimum, and on
the same MLIP energies it calls stable phases in which the screen's free-energy comparison finds a
lower displaced minimum. On the displacive systems at T ≤ 300 K the screen is right where SSCHA's
default criterion is wrong on 33 units against 3, consistently across four of five systems
(system-clustered p = 0.125, where the smallest attainable value is 0.0625), under the screen's
production frozen-cell convention. Insofar as the limitation is the criterion's, a fine-tuned
surface with equally deep wells would give the same answer; whether the production relaxations
reached the SCHA minimum was not recorded (§3.3).
<!-- PENDING-C1: one clause on whether the false-stables survive a converged relaxation (C1c), whichever way it falls; restate this paragraph if they do not. -->
Screening every imaginary commensurate mode
rather than the softest one also matters: in SrTiO₃, under the production convention, the deepest
instability is not the one that condenses. PBE forces on the sampled configurations show no
extrapolation error behind the low-temperature false-stables, and a 1.7-fold rise in it on the
600 K SrTiO₃ ensemble, where the high-temperature false-unstables arise.

Disagreement across an ensemble of independent architectures is a suggestive guardrail, not a
robust one. The discrete vote split flags consensus errors (AUC 0.76) but falls to 0.63, with an
interval spanning chance, without ORB-v2, and the continuous frequency spread could not be resolved
as a predictor at this sample size. Nor did the pre-registered force-level spread show a signal:
its primary AUC interval spans chance, and its estimate falls to 0.31 once the units whose
configurations bring atoms into overlap are excluded.

We make no ranking of the five models at finite temperature. Their accuracies span three units on
a denominator of thirty and every interval overlaps, so which model leads is not resolved by this
design. The findings above are the ones that survive that limitation, and none of them is a
leaderboard.

## Data availability

The code supporting this article, together with the per-unit results ledger, is openly available
in the repository at https://github.com/fronkt/mlip-dynamic-stability. The version that produced
the numbers in this article will be pinned to a tagged release on the repository's default branch
and archived as a new version under the Zenodo concept DOI https://doi.org/10.5281/zenodo.20805799,
which resolves to the latest version.
<!-- PENDING-P5: release tag, commit hash and the version DOI of the new Zenodo version; name them here once minted (after the main merge and after the PENDING compute lands). -->

All production results regenerate from `results/ledger.parquet` (per-unit hashed, append-only,
resumable) through `mlip_dynstab.analysis.canonical`, which selects the current generation of each
layer; every figure and statistic in this article is computed through it. Figures are made by
`scripts/make_figures.py` and the analysis is in `mlip_dynstab/analysis.py`. The Wilson intervals
and the system-clustered tests are in `mlip_dynstab/stats.py`, run by `scripts/stats_hardening.py`
into `results/stats_hardening.json`. The ESI tables are generated from the ledger and that file by
`scripts/build_esi_tables.py`, whose `--check` option fails if the ESI on disk is out of date, and
`scripts/verify_claims.py` re-derives the headline numbers from the ledger. The sensitivity
analysis of the screen (ESI Table S14) re-solves the cached E(Q) maps with
`scripts/screen_sensitivity.py` into `results/screen_sensitivity.json`, and the
displacement-amplitude sweep (ESI Table S13) is run by `scripts/run_disp_sweep.py`, with its rows
in the ledger. The stage-by-stage SSCHA diagnostic of ESI Table S2 is `scripts/sscha_v4_diag.py`.
The bcc-Zr seed study of §3.5 is `scripts/sscha_repro.py`; its per-seed frequencies were printed to
the run log and are not deposited. The finite-size rows are in `results/convergence_study.parquet`,
whose SSCHA rows are the superseded v1 generation (§3.5). The SSCHA seed study with convergence
diagnostics and the 3×3×3 bcc-Zr re-measurement (`scripts/sscha_seed_study.py`), the force-level
ensemble test
(`scripts/force_spread.py`) and the first-principles reference calculations
(`scripts/dft_reference.py`) write their outputs under `results/revision/`. The identity between
the screen's symmetric-point curvature and its trial stiffness, and the origin of every negative
value of that observable, are checked by `scripts/curvature_identity_check.py` (into
`results/curvature_identity_check.json`), and the H2 count under each frozen-cell convention by
`scripts/h2_by_convention.py` (into `results/h2_by_convention.json`); both re-solve the cached
maps and need no GPU.

The ledger is append-only and keeps each superseded generation beside the current one, so every
correction is auditable. The screen results reported here are the multi-mode grid of §2.4
(5 models × 20 systems × 4 temperatures = 400 units, method version 4). The first screen
generation in the ledger, a single-mode one, had two defects. Its Γ-point acoustic mask removed
the three lowest branches rather than the three nearest zero, which for an unstable phase deletes
the ferroelectric soft mode, and it was computed with a **q**-point search that had been replaced
in the code, because the unit hash did not carry the algorithm version and the stale units were
skipped as already present. The unit hash now includes the method version and the mode cap, so a
re-run under a changed algorithm records new rows rather than reusing old ones. The SSCHA results
are a full re-measurement (`scripts/run_sscha_v2.py`, method version 3) of the 208-unit grid in the
pinned environments. The first SSCHA generation took the three acoustic modes to be the lowest
frequencies rather than those smallest in magnitude, which discards the soft mode of an unstable
phase, and was computed in an unrecorded software environment. It is retained together with the
in-place derived correction (`scripts/fix_sscha_acoustic.py`, `*_v1` columns) and the uncorrected
snapshot (`results/ledger.parquet.pre-d1-fix`). The harmonic layer was likewise re-measured in the
pinned environments, and its first generation is kept as the replicate of ESI §S1.2. Each model was
run in its own pinned environment, with `pip freeze` manifests deposited as
`envs/lock-<model>-2026-08-16.txt`. The cached E(Q) maps of the current screen generation
(`results/cache/softmode_v3m24_*.json`, one per system and model) are included, so the
temperature solves and the sensitivity analysis regenerate without a GPU.

## Author contributions

F.C.: conceptualization, methodology, software, formal analysis, investigation, data curation,
visualization, writing – original draft, and writing – review & editing.

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
2. X.-Q. Han, P.-J. Guo, Z.-F. Gao and Z.-Y. Lu, *arXiv*, 2025, arXiv:2512.21227v2.
3. G. Petretto, S. Dwaraknath, H. P. C. Miranda, D. Winston, M. Giantomassi, M. J. van Setten, X. Gonze, K. A. Persson, G. Hautier and G.-M. Rignanese, *Sci. Data*, 2018, **5**, 180065.
4. L. Monacelli, R. Bianco, M. Cherubini, M. Calandra, I. Errea and F. Mauri, *J. Phys.: Condens. Matter*, 2021, **33**, 363001.
5. (a) O. Hellman, I. A. Abrikosov and S. I. Simak, *Phys. Rev. B*, 2011, **84**, 180301(R); (b) O. Hellman, P. Steneteg, I. A. Abrikosov and S. I. Simak, *Phys. Rev. B*, 2013, **87**, 104111.
6. I. Batatia, P. Benner, Y. Chiang, A. M. Elena, D. P. Kovács, J. Riebesell *et al.*, *J. Chem. Phys.*, 2025, **163**, 184110.
7. B. Deng, P. Zhong, K. Jun *et al.*, *Nat. Mach. Intell.*, 2023, **5**, 1031.
8. M. Neumann, J. Gin, B. Rhodes, S. Bennett, Z. Li, H. Choubisa, A. Hussey and J. Godwin, *arXiv*, 2024, arXiv:2410.22570.
9. Y. Park, J. Kim, S. Hwang and S. Han, *J. Chem. Theory Comput.*, 2024, **20**, 4857.
10. H. Yang, C. Hu, Y. Zhou *et al.*, *arXiv*, 2024, arXiv:2405.04967.
11. T. Tadano and S. Tsuneyuki, *Phys. Rev. B*, 2015, **92**, 054301.
12. W. Zhong, D. Vanderbilt and K. M. Rabe, *Phys. Rev. Lett.*, 1994, **73**, 1861.
13. A. Marronnier *et al.*, *ACS Nano*, 2018, **12**, 3477.
14. L. Monacelli and N. Marzari, *Chem. Mater.*, 2023, **35**, 1702–1709.
15. (a) W. Petry *et al.*, *Phys. Rev. B*, 1991, **43**, 10933; (b) A. Heiming *et al.*, *Phys. Rev. B*, 1991, **43**, 10948.
16. K. Parlinski, Z. Q. Li and Y. Kawazoe, *Phys. Rev. Lett.*, 1997, **78**, 4063.
17. D. A. Wood and N. Marzari, *Phys. Rev. B*, 2007, **76**, 134301.
18. A. Ranalli *et al.*, *Adv. Quantum Technol.*, 2023, **6**, 2200131.
19. J. Yang, Z. Yin, L. Ao and S. Li, *Phys. Chem. Chem. Phys.*, 2026, **28**, 4459–4469.
20. D. Li, J. Yang, X. Chen, L. Yu and S. Liu, *J. Phys. Chem. C*, 2025, **129**, 21538–21544.
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
34. A. Merchant, S. Batzner, S. S. Schoenholz, M. Aykol, G. Cheon and E. D. Cubuk, *Nature*, 2023, **624**, 80–85.
35. C. Chen and S. P. Ong, *Nat. Comput. Sci.*, 2022, **2**, 718–728.
36. J. Riebesell, R. E. A. Goodall, P. Benner, Y. Chiang, B. Deng, G. Ceder, M. Asta, A. A. Lee, A. Jain and K. A. Persson, *Nat. Mach. Intell.*, 2025, **7**, 836–847.
37. A. Hajibabaei, C. W. Myung and K. S. Kim, *Phys. Rev. B*, 2021, **103**, 214102.
38. M. Ha, S. Pourasad, C. W. Myung and K. S. Kim, *Acc. Chem. Res.*, 2026, **59**, 103–113.
39. I. K. Park, S. Pourasad, J. Mun, A. C. Rajan, T. H. Park, A. Rohit, M. Zafari, T.-L. Pham, E. Jung, B. Koo, M. Ha, G. B. Sim, H. G. Park, W. W. Choi, K. Jo, S. Y. Willow, D. C. Yang, C. W. Myung, G. Lee and K. S. Kim, *Adv. Energy Mater.*, 2026, e71046, DOI: 10.1002/aenm.71046.
40. B. Deng, Y. Choi, P. Zhong, J. Riebesell, S. Anand, Z. Li, K. Jun, K. A. Persson and G. Ceder, *npj Comput. Mater.*, 2025, **11**, 9.
41. H. Lee, Z. Li, J. He and Y. Xia, *Small*, 2026, **22**, e00071.
42. M. Radova, W. G. Stark, C. S. Allen, R. J. Maurer and A. P. Bartók, *npj Comput. Mater.*, 2025, **11**, 237.
43. J. Hänseroth, A. Flötotto, M. N. Qaisrani and C. Dreßler, *J. Phys. Chem. Lett.*, 2026, **17**, 3152–3162.
44. (a) P. Giannozzi, S. Baroni, N. Bonini *et al.*, *J. Phys.: Condens. Matter*, 2009, **21**, 395502; (b) P. Giannozzi, O. Andreussi, T. Brumme *et al.*, *J. Phys.: Condens. Matter*, 2017, **29**, 465901.
45. G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet and N. Marzari, *npj Comput. Mater.*, 2018, **4**, 72.
46. N. Marzari, D. Vanderbilt, A. De Vita and M. C. Payne, *Phys. Rev. Lett.*, 1999, **82**, 3296.
