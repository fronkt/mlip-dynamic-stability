# Electronic Supplementary Information (ESI)

*Supplementary information for* "Neither harmonic benchmarks nor a default SSCHA cross-check
certifies a foundation machine-learning interatomic potential for finite-temperature dynamic
stability", F. Cai, *RSC Advances*. Section numbers without an S (§2.4, §3.3 and so on) and
numbered references refer to the main article. Tables S4-S11 and S13-S17 are generated from the
deposited ledger (`results/ledger.parquet`) and the deposited analysis outputs
(`results/stats_hardening.json`, `results/screen_sensitivity.json`) by
`scripts/build_esi_tables.py`; figures by `scripts/make_figures.py`.

*Citation convention: **sections** of this document are cited as §S1-§S4 and **tables** as
Table S1-Table S17. The two sequences are independent; a cross-reference to "Table S1" means the
table, not the section.*

## S1. Finite-T method development and discarded routes

The quantum-SCHA soft-mode screen (§2.4) was the fourth finite-T route we
implemented. The earlier three were built, tested against the SrTiO₃ gate (§2.4), and
discarded; we document them because the failures are instructive for anyone building MLIP-driven
finite-T stability screens.

- **Hand-rolled TDEP-lite** (NVT MD → least-squares effective harmonic force constants → minimum
  effective frequency). Conceptually the cheapest per-temperature route, but the effective
  force-constant fit is dominated by the anharmonic tails of the MD distribution and gave
  non-monotonic, supercell- and trajectory-length-sensitive frequencies that did not reproduce
  the SrTiO₃ hardening curve. The least-squares step also silently absorbs drift/relaxation off
  the high-symmetry geometry.

- **One-shot hiPhive.** Required a larger supercell so the pair cutoff stays below L/2, and the
  single-temperature effective-FC fit inherited the same tail-sensitivity as TDEP-lite without
  the robustness of a proper TDEP self-consistency loop.

- **Rattled-MD symmetry-breaking probe.** Intended for diffusive/superionic systems where
  effective-FC fitting is ill-posed; in practice the rattle amplitude was a free knob that set
  the answer, so it could not provide a calibrated stability call.

The surviving route replaces noisy MD-fitted effective force constants with a **static** double
well E(Q) along each imaginary commensurate mode plus an analytic single-mode quantum SCHA free
energy per mode, with the phase called unstable if any mode condenses (see §2.4; the original
implementation screened only the softest mode, which inspects the wrong instability in SrTiO₃). Moving the anharmonicity into a clean static map (cached, T-independent) and the thermal
physics into a 1-D self-consistent solve removed the trajectory/cutoff sensitivity and passed
the SrTiO₃ gate.

### S1.1 Soundness fixes during development

- **Γ-acoustic artifact.** The softest-mode search initially selected the acoustic sum-rule zero
  at Γ (well depth 0, convex E(Q)); the three acoustic modes at Γ are now masked.
- **Spurious flat-well modes from incommensurate q.** Searching the denominator-6 rational grid
  on 2×2×2 force constants interpolated force constants at q the cell does not actually resolve,
  inventing flat soft wells (notably for bcc with MACE/SevenNet). The search is now restricted to
  exactly force-constant-commensurate q; bcc uses 6×6×6 force constants so the ω-phase q = ⅔⟨111⟩
  is commensurate.
- **Unbounded fits.** Sextic E(Q) fits occasionally returned a negative leading coefficient
  (unbounded potential); the fit drops to a bounded quartic in that case, and only the
  well-plus-barrier window is fitted so a steep repulsive wall does not wash out a shallow well.
  **Window sensitivity (Table 1, A3 of the main text).** The production window keeps the
  sampled points within max(60 meV, 5|E_min|) of the minimum. Refitting every cached E(Q) with
  the multiplier at 3×, 5× and 8× and the floor at 30, 60 and 120 meV changes no unit call at
  any temperature, and at most 14 of 756 mode-level condensation calls at T ≤ 300 K (Table S14).
  The multiplier alone is a weak probe, because for 261 of the 378 screened modes (69%) the
  kept point set is the same at 3×, 5× and 8× with the 60 meV floor; that is why the floor is
  varied too.
- **SCHA self-consistency.** The width is solved with a bracketed root finder to avoid a runaway
  large-σ spurious root.

### S1.2 Harmonic estimator reproducibility: the v1/v2 paired replicate

The harmonic layer was measured twice. Generations v1 and v2 run the *same algorithm at the same
settings* (0.01 Å displacements, 2×2×2 supercells, 12×12×12 meshes) in independently pinned
software environments (`mlip_dynstab/__init__.py`), so the pair is a genuine replicate of the
estimator rather than a re-run of a cached result. That gives 100 paired (system, model)
measurements of the softest-mode frequency. Reproduced by `scripts/estimator_noise.py` from the
deposited ledger; no new computation is involved.

**Table S1** Harmonic estimator reproducibility: the v1/v2 paired replicate, per model.

| model | n | median \|Δ\| (THz) | p95 \|Δ\| | max \|Δ\| | stability-call flips |
|---|---|---|---|---|---|
| CHGNet | 20 | 5.7 × 10⁻⁵ | 3.6 × 10⁻³ | 0.0076 | 0 |
| MatterSim | 20 | 5.4 × 10⁻⁵ | 5.4 × 10⁻⁴ | 0.00057 | 0 |
| SevenNet-0 | 20 | 1.7 × 10⁻⁵ | 4.8 × 10⁻⁴ | 0.0010 | 0 |
| MACE-MP-0 | 20 | 1.5 × 10⁻¹² | 4.2 × 10⁻¹ | 0.536 | 0 |
| ORB-v2 | 20 | 4.3 × 10⁻² | 1.9 | 2.011 | 2 |

**The spread must not be pooled.** The pooled max of 2.01 THz comes entirely from ORB-v2 and
would overstate the replicate noise of CHGNet and MatterSim, the pair whose matched-set comparison
§3.2 discusses, by more than two orders of magnitude. ORB-v2's two flipped calls are `ktao3_cubic`,
which is already excluded from the matched set as borderline, and `hf_bcc`, which moves from
−0.0891 THz in v1 to −0.1937 THz in v2. At the production tolerance of 0.1 THz the v1 value is a
stable call and the v2 value an unstable one, so the replicate flips a scored ORB-v2 harmonic
call; at tol = 0 both values are unstable and nothing flips. ORB-v2 is the model with the largest
replicate noise. It is also the only one of the five whose forces are predicted directly rather
than as the gradient of an energy; precision does not distinguish it, because CHGNet, SevenNet-0
and MatterSim also return float32 forces as run and their replicate noise is below 0.01 THz.

**A note on the deposited flag.** `results/estimator_noise.json` records
`ordering_invariant: false` next to the statement that CHGNet sits strictly below MatterSim at
every tolerance in [0.05, 0.50]. Both are correct. The flag is evaluated over the full swept
range including tol = 0, where the two models *tie* at 0.684 rather than reversing, so the
strict inequality fails without the ordering ever inverting.

**This is a lower bound, not a full characterisation.** v1 and v2 share a code path, a displacement
amplitude and a seedless algorithm, so the pair probes environment and library nondeterminism only.
It does *not* probe sensitivity to the finite displacement amplitude itself, which is the axis a
reader familiar with κ_SRME's estimator noise will ask about. That axis is measured separately,
for all five models, by re-running the scored units at 0.005, 0.02 and 0.03 Å against the
production 0.01 Å (Table S13): MACE-MP-0, MatterSim and SevenNet-0 change no call, CHGNet changes
calls only on controls, and ORB-v2 changes calls on test systems as well.

The way the harness records that sweep is worth documenting, because the obvious implementation
is wrong twice over. `disp` was not part of the unit hash, so a re-run at
a new amplitude was skipped by `has_unit()` as already present; but simply adding it to the hash
would have changed the hash of all 100 deposited harmonic rows and detached the ledger from the
code that produced it. The production amplitude is therefore still absent from the hash, and only
a departure from it is recorded. Swept units are also written under the method name
`harmonic_dispsweep` rather than `harmonic`, because they carry the same `method_version` as
production rows and would otherwise pass through `analysis.canonical()` and inflate the
denominator of every harmonic rate in this paper.

### S1.3 Complete derivation of the single-mode quantum SCHA screen

Section 2.4 of the main text states the variational free energy and its self-consistency
condition. This section supplies what a reader reproducing the screen needs and the main text
compresses: the coordinate, the effective mass and the cell over which both are defined; the
implementation constants, which are fixed in the units of that coordinate; the stationarity step
that yields the self-consistency condition; the parity assumption; and what the single-mode
restriction neglects.

**Coordinate and effective mass.** A frozen mode is a rigid displacement pattern: atom *i* of
the mode's cell moves to **R**_i = **R**_i^(0) + Q **u**_i, where **u**_i is the (real) Cartesian
displacement of atom *i* in the pattern and Q is the scalar amplitude that parametrises the
distortion. The kinetic energy of that one-parameter motion is

$$T = \tfrac{1}{2}\sum_i m_i |\dot{\mathbf{R}}_i|^2
    = \tfrac{1}{2}\dot{Q}^2 \sum_i m_i |\mathbf{u}_i|^2
    \equiv \tfrac{1}{2} M \dot{Q}^2 ,
\qquad M = \sum_i m_i |\mathbf{u}_i|^2 ,$$

which is the definition used in the main text. Together with V(Q) from the fitted map this gives
the one-dimensional Hamiltonian H = P²/2M + V(Q).

M and Q are meaningful only once two things are fixed: the normalisation of **u**, and the cell
over which M and V are summed. The main text's phrase "unit displacement pattern" fixes neither.

*Normalisation of the pattern.* **u** is dimensionless and scaled so that its largest Cartesian
component is 1, max_i,α |u_i,α| = 1: phonopy's modulation output divided by its own largest
component (`finite_t._mode_pattern`). Q therefore carries the length unit and is the displacement,
in ångström, of the largest single Cartesian component in the pattern. It is not the largest
atomic displacement, which exceeds Q by up to a factor √3 when an atom moves off a Cartesian
axis. M is in atomic mass units and V in eV, so a, b and c are in eV Å⁻², eV Å⁻⁴ and eV Å⁻⁶.
The reported order parameter Q₀ (for example 0.14 Å for the SrTiO₃ R-point tilt under
MACE-MP-0) is read on the same scale. This convention differs from the unit-norm convention
Σ_i |**u**_i|² = 1 that "unit pattern" might suggest, and again from the mass-weighted eigenvector
**e**_i = √m_i **u**_i returned by diagonalising the dynamical matrix.

Rescaling the pattern, **u** → λ**u**, sends Q → Q/λ, M → λ²M, and the fitted coefficients
a → λ²a, b → λ⁴b, c → λ⁶c, so the one-dimensional problem is unchanged: ⟨V″⟩/M, every frequency
and every stability call are invariant, and only the numerical value of Q₀ moves. That
invariance belongs to the mathematics, not to the implementation, because four constants are
fixed in ångström: the sampling range Q ≤ 0.45 Å, the centroid scan 0 ≤ Q₀ ≤ 0.6 Å, the
condensation threshold Q₀ > 0.0075 Å and the finite-difference step of the curvature, 0.005 Å.
Under the max-component convention these are displacements of the most-displaced coordinate,
which is the scale on which they were chosen; a different normalisation of **u** would need them
rescaled with it.

*The cell over which M and V are summed.* Both M and V(Q) are sums over the mode's **minimal
commensurate cell**: the phonopy modulation cell whose dimensions are the denominators of
**q** (1×1×1 at Γ, 2×2×2 at R = (½,½,½), and the corresponding cells at X and M). In the
perovskites that cell holds 1, 2, 4 or 8 formula units depending on the mode. It is the smallest
cell in which the mode is a single commensurate distortion, and it is the production convention.
It is a choice rather than a consequence of the physics, and unlike the rescaling of **u** it
changes answers. Summing over n copies of the cell maps (a, b, c, M) → n(a, b, c, M), and since

$$H_n = \frac{P^2}{2nM} + nV(Q) = n\left[\frac{P^2}{2n^2M} + V(Q)\right],$$

the problem in the larger cell is the minimal-cell problem at mass n²M and temperature T/n. A
larger cell is a heavier, effectively colder and more classical mode, and it condenses more
readily. A single-mode treatment has no internal way to fix n: the collective coordinate of a
real crystal is correlated over a length that a single frozen mode does not contain. Table S14
shows the consequence. Taking the cell per formula unit, as the minimal cell, doubled, or as the
common force-constant supercell gives a ferroelectric-oxide recall of 5/30, 16/30, 26/30 and
28/30, and the SrTiO₃ gate passes for 0, 3, 2 and 3 of the five models. Accuracy on the scored
set cannot choose among these conventions (Table S14 notes). The screen's T* and its
ferroelectric recall are therefore conditional on the minimal-cell convention; we state that as
a limitation of the single-mode screen, not as a setting we could defend on the data.

**Sampling, fit and scan: the implementation constants.** Each map is sampled at ten points
Q_k = 0.45 (k/9)² Å, k = 0, …, 9, spaced quadratically so that most points fall at small Q, where
the narrow ferroelectric wells sit. V(Q) = aQ² + bQ⁴ + cQ⁶, with V(0) = 0 by construction, is
fitted by least squares to the sampled points lying within max(60 meV, 5|E_min|) of the sampled
minimum E_min, or to the four lowest-Q points if fewer than four lie in that window. If the
sextic returns c < 0, or b < 0 with c ≤ 0, the quartic is refitted with c = 0 so the potential
stays bounded below; on the production maps every refitted quartic has b > 0. The free
energy is minimised over the centroid on 121 points spanning 0 ≤ Q₀ ≤ 0.6 Å (step 0.005 Å), and
the mode is called condensed when the minimising Q₀ exceeds 1.5 steps, 0.0075 Å. The
symmetric-point curvature reported as ω_eff is a fourth-order finite difference of 𝓕(Q₀) with
step 0.005 Å; because 𝓕 is even in Q₀ the one-sided stencil is central. Table S14 varies the fit
window, the sampling range, the threshold and the scan; apart from the cell convention, none of
them changes more than one unit call at T ≤ 300 K.

**Stationarity, and why it gives the self-consistency condition.** With the trial state fixed to
a Gaussian of centroid Q₀ and width σ, the Gibbs–Bogoliubov (Peierls) bound is

$$\mathcal{F}(Q_0,\Omega;T) = F_0(\Omega,T) + \langle V\rangle_{Q_0,\sigma}
   - \tfrac{1}{2}M\Omega^2\sigma^2 ,\qquad
   F_0 = k_BT\ln\!\left[2\sinh\!\left(\tfrac{\hbar\Omega}{2k_BT}\right)\right],$$

with σ² = (ħ/2MΩ) coth(ħΩ/2k_BT). Minimising over the one free parameter Ω uses two identities.
The first is that the trial free energy's derivative is itself MΩσ²,

$$\frac{\partial F_0}{\partial\Omega}
  = k_BT\cdot\frac{\hbar}{2k_BT}\coth\!\left(\frac{\hbar\Omega}{2k_BT}\right)
  = \frac{\hbar}{2}\coth\!\left(\frac{\hbar\Omega}{2k_BT}\right) = M\Omega\sigma^2 ,$$

and the second is the Gaussian smoothing identity ∂⟨V⟩/∂σ² = ½⟨V″⟩, which holds for any V whose
Gaussian average exists. Differentiating 𝓕 at fixed Q₀ and collecting terms,

$$\frac{\partial\mathcal{F}}{\partial\Omega}
 = M\Omega\sigma^2
 + \tfrac{1}{2}\langle V''\rangle\frac{\partial\sigma^2}{\partial\Omega}
 - M\Omega\sigma^2
 - \tfrac{1}{2}M\Omega^2\frac{\partial\sigma^2}{\partial\Omega}
 = \tfrac{1}{2}\frac{\partial\sigma^2}{\partial\Omega}
   \left(\langle V''\rangle - M\Omega^2\right).$$

Since ∂σ²/∂Ω ≠ 0, the stationary point is exactly M Ω² = ⟨V″⟩, the condition quoted in the main
text. The two terms in MΩσ² cancelling is the reason the counter-term −½MΩ²σ² must be present:
without it the minimisation would not reduce to the self-consistency condition.

This establishes stationarity, not that the self-consistent Ω minimises 𝓕 at fixed Q₀; we have
not checked the second derivative in Ω. The Gibbs–Bogoliubov inequality holds for every Ω, so 𝓕
at the stationary point is an upper bound on the free energy whether or not it is the tightest
one available at that centroid, and the call compares these bounds across centroids.

For the sextic V(Q) = aQ² + bQ⁴ + cQ⁶ the Gaussian moments give
⟨V″⟩ = 2a + 12b⟨Q²⟩ + 30c⟨Q⁴⟩ with ⟨Q²⟩ = Q₀² + σ² and ⟨Q⁴⟩ = Q₀⁴ + 6Q₀²σ² + 3σ⁴.

**Solving it.** Substituting Ω(σ²) back gives a scalar fixed-point equation g(σ²) = σ²_sc(σ²) −
σ² = 0, solved in σ² rather than Ω because the physical branch is easier to bracket there. A
damped iteration on this equation can run away to a large-σ spurious root, so the implementation
brackets instead. The search grid is geometric, σ² ∈ [10⁻⁶, 2] Å² over 240 points, restricted to
the sub-interval on which ⟨V″⟩(σ²) > 10⁻⁸ eV Å⁻² (below that no normalisable Gaussian trial
state exists, since Ω would be imaginary); the first downward sign change of g on that grid is
refined by Brent's method to a tolerance of 10⁻¹⁰. If the grid has no sign change, or Brent's
method fails, the solver keeps the grid point at which |g| is smallest. That fallback fires on
8441 of 182,952 centroid evaluations (4.6%) and on none of the 1512 evaluations that decide a
call, the minimising centroid of each mode and temperature (Table S14). If the admissible
sub-interval is empty, no bound Gaussian exists at that centroid, and the centroid is rejected
with 𝓕 = +∞ so the outer minimisation moves to a displaced, bound centroid rather than
reporting a spurious value; on the production maps this never happens. The outer minimisation
over Q₀ is the 121-point scan described above.

**Parity.** The fitted polynomial is even and only Q ≥ 0 is sampled and scanned, so the screen
assumes V(−Q) = V(Q). For most modes an operation of the cubic phase, a lattice translation or a
point operation, maps the pattern onto its negative, and the assumption is exact. It is not
guaranteed for the 24 screened bcc modes with 3**q** ≡ **G** and 2**q** ≢ **G** (the ω-type
modes): no lattice translation reverses those patterns, and for this star the symmetry-allowed
cubic invariant vanishes only for particular modulation phases (multiples of π/3). Whether E(Q)
is even along the sampled pattern therefore depends on the phase phonopy returns, and we did not
check it. The bcc metals are outside the scored finite-temperature set, so no headline rate rests
on these modes; they enter only the bcc screen-versus-SSCHA comparison (Table S16) and T*
(Table S6).

**What the single-mode restriction neglects.** Writing the full Born–Oppenheimer surface in the
normal coordinates {Q_k} of the commensurate cell,

$$V(\{Q_k\}) = \sum_k V_k(Q_k) + \sum_{k<l}\lambda_{kl}Q_k^2Q_l^2
   + \sum_{k\neq l}\mu_{kl}\,Q_k^2Q_l + \sum_{k<l<m}\nu_{klm}\,Q_kQ_lQ_m + \ldots ,$$

where λ are biquadratic couplings and μ and ν cubic and trilinear ones, present only where
symmetry allows them. The screen evaluates each V_k(Q_k) with every other Q_l held at zero, so it
discards all of the inter-mode terms. Which of them are allowed depends on the irreducible
representations involved, and we did no symmetry analysis of the screened pairs. Nor do the
screened modes all belong to different representations: deduplication matches frequencies to
10⁻⁴ THz, so a triplet whose components a model returns at slightly different frequencies is
screened as separate modes of one representation (the T1u triplet of BaTiO₃ under CHGNet is an
example). The cubic and trilinear terms are therefore not excluded, and they, like a negative λ,
can drive a joint condensation that no single mode shows alone.

Keeping only the biquadratic term, a mean-field reading gives mode *k* an effective quadratic
coefficient

$$a_k^{\text{eff}} = a_k + \sum_{l\neq k}\lambda_{kl}\langle Q_l^2\rangle_T ,$$

so the sign of λ decides the direction of the error and the temperature dependence enters
through ⟨Q_l²⟩_T. Competing (λ > 0) coupling hardens mode *k* as other modes acquire amplitude,
so the isolated treatment over-predicts condensation; cooperative (λ < 0) coupling does the
opposite. This is why Table 1 lists A1's expected bias as "either sign".

We have no quantitative bound on the λ, μ or ν, and obtaining one would need the multi-mode map
the screen is designed to avoid. Nor are the reported calls far from where the coupling matters.
The deciding-mode wells at T ≤ 300 K are 28–43 meV deep per minimal cell in BaTiO₃, 25–59 meV
in KNbO₃, 0–110 meV in PbTiO₃ and 0–27 meV in SrTiO₃, against k_BT = 26 meV at 300 K, and for
16 of these 20 (system, model) pairs the screen's own T* is 300 K or lower (Table S6). The
T ≤ 300 K calls therefore sit close to the screen's stabilisation temperature, so a coupling
error moves them as directly as it moves T*; KNbO₃, 408 K below its transition, is called
stable at 300 K by four of the five models. The H2 decomposition points the same way: at 300 K,
13 of the 17 units in which the harmonic call is right and the screen call wrong fall in BaTiO₃,
KNbO₃ and CsSnBr₃, three systems the screen mis-calls for at least four of the five models
(Table S15). That pattern is what a shared error of the single-mode approximation would produce,
rather than an error specific to one model's PES; a feature of the surface that all five models
inherit from similar training data would look the same, and a screen run on a first-principles
surface separates the two. T* is reported as a diagnostic (Table S6) and not as a prediction for
the same reason.
<!-- PENDING-C3a: PBE E(Q) along the deciding soft-mode coordinates and PBE-backed screen calls
for BaTiO3, KNbO3, CsSnBr3 (and SrTiO3 R, bcc-Zr, ZrO2) decide whether these shared mis-calls
come from the single-mode approximation or from the MLIP surfaces; one sentence here -->

The multi-mode SSCHA retains the couplings, but here it cannot bound their effect on the call,
for three reasons. It runs on the same MLIP potential, so any PES error is common to both
methods. On the displacive systems its default criterion is local (§3.3, Table S16), so it
answers a different question from the screen's free-energy comparison. And on bcc, where both
apply, the agreement is weaker than a bound would need. The curvature signs agree in 35/45 paired
units (0.78 [0.64, 0.87]; 30/36 without ORB-v2), but the stability calls agree in 31/45
(0.69 [0.54, 0.80]; 25/36 without ORB-v2). Twelve of the agreeing units are CHGNet and
MACE-MP-0 on Zr and Hf, which have no harmonic instability to stabilise; without them the calls
agree in 19/33. MatterSim, whose bcc harmonic instabilities are the deepest of the five models,
agrees with SSCHA on the call in 0/9 units (Table S16). The size of the coupling correction is
unmeasured.

### S1.4 Why the criterion is variational: comparison with an exact 1D quantum solution

Each deciding mode is stored as an effective mass and three polynomial coefficients, so the
one-dimensional Schrödinger problem for the *same* fitted potential can be solved to machine
precision by diagonalising the tridiagonal Hamiltonian on a grid and Boltzmann-averaging the
resulting states (`finite_t._solve_1d`, run over the deposited ledger by
`scripts/scha_vs_exact.py`; 228 mode-temperature evaluations over 57 (system, model) units,
involving 75 distinct fitted potentials because the deciding mode can change along the ladder;
no solver failures, no new computation). The natural
stability criterion for that exact solution is the potential of mean force
W(Q) = −k_BT ln P(Q,T): the mode has condensed if the thermal density is bimodal.

Two results follow, and the first is a caution about the second.

**The exact isolated-mode criterion is temperature-independent, so it is not a finite-temperature
reference.** It calls 96.5% of modes condensed at 100, 300, 600 and 900 K alike, to four decimal
places. This is not a defect of the solver but the physics of an isolated coordinate: as k_BT
grows, P(Q,T) → exp(−V(Q)/k_BT), whose maxima are the minima of V, so a genuine double well
remains bimodal at every temperature. A single mode in isolation has no mechanism by which to
thermally stabilise. Any disagreement between this criterion and the screen must therefore
**not** be read as an error in the SCHA approximation; the two criteria answer different
questions.

**Table S12** The exact isolated-mode criterion against the variational one, over the same 228
mode-temperature evaluations: 57 (system, model) units, involving 75 distinct fitted potentials.

| T (K) | exact: fraction bimodal | SCHA screen: fraction condensed |
|---|---|---|
| 100 | 0.9649 | 0.8596 |
| 300 | 0.9649 | 0.6667 |
| 600 | 0.9649 | 0.5614 |
| 900 | 0.9649 | 0.4912 |

That contrast is itself the argument for the variational criterion. The temperature dependence
that lets the screen bracket the SrTiO₃ transition at 105 K is supplied by the SCHA
self-consistency, in which the Gaussian width broadens with T and re-enters the averaged
potential, and not by the shape of the well. It also explains why all three discarded routes of
§S1 failed the SrTiO₃ gate: each read a density or an effective force constant rather than
minimising a free energy.

**The screen never condenses a mode whose fitted potential lacks a displaced minimum.** Across
all 228 evaluations there is not a single case in which the screen condenses a mode whose exact
thermal density is unimodal; the disagreements are entirely in the other direction, and they
concentrate in shallow wells (median depth 27.9 meV, against 216 meV where the two agree). This
is the mode-level complement of the phase-level control result in §3.2, where the screen finds
zero imaginary commensurate modes on all 120 harmonically-stable control units.

The statement is about the **fitted** potential, not the sampled energies, and the two can
differ. In 19 of the 378 screened modes (18 on Ti under ORB-v2, one on PbTiO₃ under ORB-v2)
every sampled energy lies at or above E(0), yet the fitted polynomial has a minimum between
sample points, at Q ≈ 0.04–0.08 Å: a fit artefact, 34.6 to 798.9 meV deep per minimal cell for the Ti
modes and 0.2 meV for the PbTiO₃ one. These artefacts carry 48 mode-temperature condensation
calls. No unit call rests on them alone, because every affected unit has another condensing mode
whose well is present in the sampled energies, so no unit stability call changes if they are
removed (`scripts/screen_sensitivity.py`, diagnostics).

Neither result speaks to the accuracy of the underlying potential-energy surface. The
experimental anchors of §2.4 and §3.2 test the surface and the screen together; a first-principles
energy profile along the same coordinates is the direct test of the surface.
<!-- PENDING-C3a: PBE E(Q) along the deciding soft-mode coordinates (SrTiO3 R, BaTiO3, KNbO3,
CsSnBr3, bcc-Zr, ZrO2) against the MLIP maps, and the screen re-run on the PBE maps; new ESI
section, pointed to from here -->

## S2. SSCHA harness and the displacive-instability failure

### S2.1 Minimizer step cap

Multi-mode SSCHA on the 40-atom perovskite cells initially appeared to hang (CPU busy, GPU idle,
a per-population step counter climbing past 1000). The cause was an uncapped per-population
reweighting loop: with a noisy stochastic gradient on a large soft cell, one population chases
the gradient below its own stochastic-noise floor indefinitely instead of regenerating a fresh
ensemble. Capping the per-population steps (`minim.max_ka = 20`) forces ensemble regeneration;
with the cap SrTiO₃ converged in 231 s and 27 steps, against 1835 steps and a timeout without
it. Small cells (8-atom bcc) converge under the cap and are unchanged. The cap is not the
stopping criterion: the relaxation ends when python-sscha's convergence test, with threshold
`meaningful_factor = 1e-4`, is met, or when the population limit (8) is reached. Which of the two
ended each production unit was not recorded (Table S11).

### S2.2 Stage-by-stage diagnostic (cubic BaTiO₃, MACE-MP-0, 100 K)

`scripts/sscha_v4_diag.py`. The numbers in Table S2 come from a single run in June 2026, in the
unpinned environment of the v1 grid; they were not repeated in the pinned environments of the
production grid and are not asserted by `scripts/verify_claims.py`, so they are not directly
comparable with the canonical ledger values for this unit.

**Table S2** SSCHA stage-by-stage diagnostic, cubic BaTiO₃ / MACE-MP-0 / 100 K.

| Stage | min frequency (THz) |
|---|---|
| harmonic (MLIP, finite displacement) | −5.635 |
| after `ForcePositiveDefinite` (the SSCHA starting point) | +2.882 |
| converged SCHA auxiliary dynamical matrix | +2.892 |
| free-energy Hessian, bubble level, `include_v4=False` (production) | +2.872 |
| free-energy Hessian, `include_v4=True` | did not finish (> 18 min on this one unit, against ≈ 200 s with `include_v4=False`) |

What the table shows is limited, and it is worth saying exactly what. The auxiliary SCHA matrix is
positive definite by construction (a normalisable Gaussian trial state requires it), so the
+2.88 THz after `ForcePositiveDefinite` and the +2.89 THz at convergence say nothing about
stability: the auxiliary matrix cannot soften (refs 21, 22). The object that could detect the
instability is the free-energy Hessian, and at bubble level it lands within 0.02 THz of the
positive auxiliary curvature, so it reports the symmetric phase stable against a harmonic
instability of −5.6 THz on the same MLIP. No two rows differ in the truncation order alone: the
one row that would, `include_v4=True`, has no number. The table therefore does not isolate the
truncation as the cause.

The account the rest of the paper uses does not rest on this table. The free-energy Hessian is
the curvature of the SSCHA free energy at the high-symmetry reference, so it answers a local
question: is the symmetric phase a local minimum of the free energy? For a deep double well the
free energy can keep a local minimum at the symmetric point while a displaced minimum lies lower
(the first-order-like case of §2.4), and a curvature at the reference then reports stable at any
order of the expansion. The screen's own symmetric-point curvature, computed on the same MLIP
energies, behaves this way on 52 of the 57 non-bcc units SSCHA false-stabilises (Table S16),
including this one (screen curvature +2.03 THz, SSCHA +2.87 THz, screen call unstable). Adding
the fourth-order term changes the estimate of the curvature but not the question it answers. It
could flip the sign only where the bubble-level value overstates a curvature that is in fact
negative, and whether that happens here is untested.
<!-- PENDING-C1b: include_v4 = True free-energy Hessian on BaTiO3/MACE-MP-0/100 K; one sentence
here and in §3.3, whichever way it falls -->
Whether an MLIP error on the thermally sampled configurations contributes as well is a separate
question that these MLIP-only data cannot settle (§3.3).
<!-- PENDING-C3b: PBE forces and energies on the SSCHA-sampled configurations; one sentence here -->

### S2.3 Numerical outcome by family (complete grid)

**Table S3** SSCHA numerical outcome by chemistry family over the complete grid.

| Family | n measured | failed units | numerical blow-ups (\|f\|>50 THz) | min freq (THz) | max freq (THz) |
|---|---|---|---|---|---|
| bcc (Ti/Zr/Hf) | 75 | 0 | 0 | 0.06 | 2.10 |
| fluorite (ZrO₂/HfO₂) | 40 | 0 | 0 | −24.8 | 3.33 |
| perovskite (oxide + halide) | 86 | 7 | 8 | −914 | 3.72 |

`analysis.sscha_reliability(df)`. The bcc runs are numerically clean: all 75 return a positive
frequency between 0.06 and 2.10 THz, with no blow-up or failure. That says the runs behaved, not
that their answers are right; the bcc screen-versus-SSCHA agreement is examined in Table S16,
where 12 of the 45 paired units have no harmonic instability to stabilise. Several bcc
frequencies also fall with temperature, the opposite of entropy stabilisation: from 50 to 600 K,
Ti goes from 1.728 to 1.375 THz under MACE-MP-0 and from 1.268 to 1.175 THz under SevenNet-0, and
Hf from 0.331 to 0.300 THz under SevenNet-0. The production grid recorded no convergence
diagnostics (§S2.1, Table S11), so it cannot show how far each of these values moved from the
positive-definite starting matrix.

All failures and blow-ups sit in the perovskite family. Seven units stop at cellconstructor
symmetry or ensemble assertions before returning a number: ORB-v2 on PbTiO₃ at every temperature,
MatterSim
on PbTiO₃ at 600 and 900 K, and SevenNet-0 on CsSnI₃ at 600 K, a halide. The eight numerical
blow-ups are spread over four models: ORB-v2 three (SrTiO₃ at 300, 600 and 900 K), MatterSim
three (PbTiO₃ at 300 K, CsSnI₃ at 600 and 900 K), CHGNet one (SrTiO₃ at 900 K) and SevenNet-0 one
(PbTiO₃ at 900 K); Table S17 lists them. As run, CHGNet, SevenNet-0, MatterSim and ORB-v2 all
return float32 forces and only MACE-MP-0 returns float64. MACE-MP-0 has neither a blow-up nor a
failure, but it differs from the others in architecture too, and one model is no basis for
attributing the failures to precision; we do not. In the v1 grid (unpinned environment, rows
retained in the ledger as superseded) the same seven units returned a number instead of failing,
between −0.3 and −3653 THz, only two of them beyond the 50 THz blow-up threshold. The most
extreme v1 value, of order −2 × 10⁶ THz, belongs to SrTiO₃/ORB-v2/600 K, which is not a failed
unit and returns −544.5 THz in the production grid.

Cubic fluorites are numerically clean (no blow-ups, no failed units), and SSCHA nonetheless
calls every one of the ten fluorite units stable at 100 K, at +1.9 to +3.3 THz in the 2×2×2
cell used throughout, although cubic
ZrO₂ and HfO₂ are the high-temperature phases (above about 2570 K and 2800 K) and the label is
unstable. The screen's free-energy comparison calls all 20 fluorite units at T ≤ 300 K unstable.
Over the whole ladder SSCHA false-stabilises 31 fluorite units, and on 27 of them the screen's
own symmetric-point curvature is positive too (Table S16): the same local-versus-global
difference as on the perovskites, on a numerically well-behaved instability. Three of the five models
then *destabilise* with temperature, which is the wrong trend (the fluorite labels are unstable
at every temperature, so those high-temperature negatives are right in sign; Table S17). That a
numerically well-behaved instability is lost the same way shows the fluorite false-stable is not
a numerical blow-up of the deep perovskite wells. Whether it also involves an MLIP error on the
sampled configurations is not settled by these data (§3.3).

### S2.4 Stochastic reproducibility and finite-size convergence

`scripts/sscha_repro.py`, `scripts/run_revision_compute.sh`; finite-size runs in
`results/convergence_study.parquet` (the per-seed reproducibility frequencies of the bcc-Zr study
print to the run log rather than to a ledger).

- **Reproducibility.** bcc-Zr / MACE-MP-0 / 100 K SSCHA over four seeds gives
  +1.798 ± 0.001 THz, and the same unit re-measured in the pinned environment with the
  phonopy-built initialiser returns +1.80 THz at every ladder temperature. The test is narrower
  than it looks: MACE-MP-0 finds no harmonic instability in bcc Zr (−0.000 THz), so this unit
  probes the stochastic noise of SSCHA on a surface with nothing to stabilise, and says nothing
  about the noise on a unit near a sign change. It is the only seed study in the production data.
  <!-- PENDING-C1: four-seed results (spread of the lowest Hessian eigenvalue, gradient/error
  history, populations used, convergence flag) for BaTiO3/MACE-MP-0 100 K, ZrO2/MACE-MP-0 100 K,
  Zr/MatterSim 50 K and SrTiO3/MACE-MP-0 600 K; new ESI table referenced here -->
- **Finite size, bcc.** For bcc-Zr / MACE-MP-0 the SSCHA stability call holds from 2×2×2 to
  3×3×3 (+1.80 to +1.56 THz at 100 K, +1.80 to +1.55 THz at 300 K). Two caveats apply. These rows
  come from the June v1 generation (unpinned environment, before the phonopy initialiser), and
  they are again MACE-MP-0 on Zr, a unit with no harmonic instability. The soft-mode rows of the
  same study were produced by the earlier single-mode selection, before the correction described
  in §2.4, and are retained for audit only.
  <!-- PENDING-C5: 2x2x2 vs 3x3x3 SSCHA re-measured in the pinned environment on bcc-Zr for
  MatterSim and MACE-MP-0; replace or confirm the numbers in this bullet -->
- **Zone-boundary systems: no convergence claim.** The SrTiO₃ antiferrodistortive instability is
  at the R point (½,½,½), commensurate only with even supercells, so a 3×3×3 cell cannot contain
  it and a 2×2×2-versus-3×3×3 comparison is not a convergence test for it. The fluorite
  instability is at the X point: the 2×2×2 cell contains it, 3×3×3 does not, so no cell-size
  comparison is available there either. We therefore make no supercell-convergence claim for the
  R-point or X-point systems. The even-cell test (4×4×4, a ~320-atom SSCHA) was not run.
- **Within-cell control for the fluorites (missing q only).** In the same 2×2×2 cell and with the
  same force engine, the harmonic calculation finds every one of the ten fluorite units unstable
  (−3.8 to −10.6 THz), while SSCHA at 100 K finds every one of them stable (+1.9 to +3.3 THz).
  The cell resolves the instability, so the fluorite false-stable is not produced by a q-point the
  cell lacks. This is a finite-size argument and nothing more: it does not show that the 2×2×2
  result is converged, and it does not distinguish the local-criterion account of §3.3 from an
  MLIP error on the sampled configurations.
- **SrTiO₃.** SSCHA never calls SrTiO₃ stable at any temperature: it is correctly unstable at
  100 K (5/5) and false-unstable in all 14 returned units above the 105 K transition, at −3.4 to
  −914 THz (11/11 without ORB-v2, to −60.5 THz), against harmonic R-mode frequencies of −0.8 to
  −2.1 THz on the four models that have an R-point instability (ORB-v2 has no imaginary
  commensurate mode in SrTiO₃; Table S17). A missing zone-boundary q-point could only hide an
  instability, so it cannot explain this; the divergence at high temperature is the SrTiO₃
  result to explain, and §3.3 discusses it.
  <!-- PENDING-C3b: PBE check on SrTiO3 SSCHA-sampled configurations; one sentence here -->
- **BaTiO₃.** Its ferroelectric instability includes the zone-centre (Γ) mode, which every
  supercell contains, so a missing q-point cannot produce its false-stable either. Like the
  fluorite control, this rules out one explanation and is not a convergence test.

## S3. Supplementary tables

The tables below are generated by `scripts/build_esi_tables.py` from the deposited ledger and the
deposited analysis outputs (`results/stats_hardening.json`, `results/screen_sensitivity.json`);
the function or script behind each is named in its caption, so any number can be re-derived
independently, and `python scripts/build_esi_tables.py --check` fails if this document is out of
date with them. Rates are given as counts over their denominators, and the rates the text relies
on carry a Wilson score interval; counts reported only for completeness (the tolerance sweep, the
blow-up and failure tallies, the per-system breakdowns) do not. Where units cluster by system, the
test is a system-level permutation or exact enumeration, and
any unit-level p-value is labelled as a companion that should not be quoted.

**Reading order.** Table numbers are fixed identifiers, not an order of appearance. Four tables
sit in the section whose method they document: Table S1 (harmonic replicate, §S1.2), Table S12
(exact isolated-mode solution, §S1.4, so it is printed before Table S2), Table S2 (SSCHA
stage-by-stage diagnostic, §S2.2) and Table S3 (SSCHA numerical outcome by family, §S2.3). The
rest are collected here in numerical order:

- Table S4, harmonic confusion matrices per model; Table S5, finite-temperature screen rates per
  model; Table S6, predicted T* against experiment.
- Table S7, composition of the two analysis sets; Table S8, the guardrail set in full.
- Table S9, every pooled rate with and without ORB-v2; Table S10, the paired screen-versus-SSCHA
  tests.
- Table S11, SSCHA settings and numerical quality; Table S13, displacement-amplitude sensitivity
  of the harmonic layer.
- Table S14, sensitivity of the soft-mode screen (fit window, sampling range, frozen-cell
  normalisation, solver constants).
- Table S15, the H2 transfer asymmetry at every temperature, clustered by system.
- Table S16, screen-versus-SSCHA agreement on bcc scored as curvature sign and as the call, and
  the same-energy comparison behind the SSCHA false-stables.
- Table S17, SSCHA high-temperature false-unstables, the SrTiO₃ units above its transition, and
  SSCHA blow-ups and failures per model.
<!-- PENDING-C1 / PENDING-C2 / PENDING-C3a: the seed-study table (C1), the force-level ensemble
spread table (C2) and the PBE soft-mode comparison (C3a) are added to this list when they land -->

<!-- BEGIN GENERATED TABLES -->

**Table S4** Per-model harmonic confusion matrices over the 19 scored systems (KTaO₃ excluded as borderline). Positive = predicted dynamically stable, so a false-stable is the screening-dangerous error. Rates carry Wilson score intervals. `analysis.per_model_table(df, "harmonic")`.

| Model | TP | TN | False-stable | False-unstable | Accuracy [95% CI] | False-stable rate [95% CI] |
|---|---|---|---|---|---|---|
| CHGNet | 4 | 11 | 2 | 2 | 15/19 = 0.789 [0.567, 0.915] | 2/13 = 0.154 [0.043, 0.422] |
| MACE-MP-0 | 6 | 11 | 2 | 0 | 17/19 = 0.895 [0.686, 0.971] | 2/13 = 0.154 [0.043, 0.422] |
| MatterSim | 6 | 13 | 0 | 0 | 19/19 = 1.000 [0.832, 1.000] | 0/13 = 0.000 [0.000, 0.228] |
| ORB-v2 | 5 | 12 | 1 | 1 | 17/19 = 0.895 [0.686, 0.971] | 1/13 = 0.077 [0.014, 0.333] |
| SevenNet-0 | 6 | 13 | 0 | 0 | 19/19 = 1.000 [0.832, 1.000] | 0/13 = 0.000 [0.000, 0.228] |

**Table S5** Per-model finite-temperature (soft-mode screen) false-stable and accuracy rates, at the T ≤ 300 K restriction used for the headline table and over the full 100/300/600/900 K ladder, each with and without the bcc metals. bcc is excluded from the headline because its thermodynamic-T_c label is the wrong reference for dynamic stability (§3.3). `analysis.low_t_false_stable(df, t_max=..., exclude_bcc=...)`.

| Temperature set | bcc | Model | False-stable rate [95% CI] | Accuracy [95% CI] |
|---|---|---|---|---|
| T ≤ 300 K | excluded | CHGNet | 3/16 = 0.188 [0.066, 0.430] | 26/30 = 0.867 [0.703, 0.947] |
| T ≤ 300 K | excluded | MACE-MP-0 | 4/16 = 0.250 [0.102, 0.495] | 25/30 = 0.833 [0.664, 0.927] |
| T ≤ 300 K | excluded | MatterSim | 4/16 = 0.250 [0.102, 0.495] | 25/30 = 0.833 [0.664, 0.927] |
| T ≤ 300 K | excluded | ORB-v2 | 5/16 = 0.312 [0.142, 0.556] | 24/30 = 0.800 [0.627, 0.905] |
| T ≤ 300 K | excluded | SevenNet-0 | 2/16 = 0.125 [0.035, 0.360] | 27/30 = 0.900 [0.744, 0.965] |
| T ≤ 300 K | included | CHGNet | 7/24 = 0.292 [0.149, 0.492] | 30/38 = 0.789 [0.637, 0.889] |
| T ≤ 300 K | included | MACE-MP-0 | 10/24 = 0.417 [0.245, 0.612] | 27/38 = 0.711 [0.552, 0.830] |
| T ≤ 300 K | included | MatterSim | 4/24 = 0.167 [0.067, 0.359] | 33/38 = 0.868 [0.727, 0.942] |
| T ≤ 300 K | included | ORB-v2 | 9/24 = 0.375 [0.212, 0.573] | 28/38 = 0.737 [0.580, 0.850] |
| T ≤ 300 K | included | SevenNet-0 | 8/24 = 0.333 [0.180, 0.533] | 29/38 = 0.763 [0.608, 0.870] |
| full ladder | excluded | CHGNet | 4/22 = 0.182 [0.073, 0.385] | 49/60 = 0.817 [0.701, 0.894] |
| full ladder | excluded | MACE-MP-0 | 8/22 = 0.364 [0.197, 0.570] | 46/60 = 0.767 [0.646, 0.856] |
| full ladder | excluded | MatterSim | 6/22 = 0.273 [0.132, 0.482] | 49/60 = 0.817 [0.701, 0.894] |
| full ladder | excluded | ORB-v2 | 7/22 = 0.318 [0.164, 0.527] | 50/60 = 0.833 [0.720, 0.907] |
| full ladder | excluded | SevenNet-0 | 4/22 = 0.182 [0.073, 0.385] | 49/60 = 0.817 [0.701, 0.894] |
| full ladder | included | CHGNet | 14/36 = 0.389 [0.248, 0.551] | 53/76 = 0.697 [0.587, 0.789] |
| full ladder | included | MACE-MP-0 | 20/36 = 0.556 [0.396, 0.705] | 48/76 = 0.632 [0.519, 0.731] |
| full ladder | included | MatterSim | 6/36 = 0.167 [0.079, 0.319] | 63/76 = 0.829 [0.729, 0.897] |
| full ladder | included | ORB-v2 | 15/36 = 0.417 [0.271, 0.578] | 56/76 = 0.737 [0.628, 0.823] |
| full ladder | included | SevenNet-0 | 16/36 = 0.444 [0.295, 0.604] | 51/76 = 0.671 [0.559, 0.766] |

**Table S6** Predicted stabilisation temperature T* (the lowest ladder temperature at which the high-symmetry phase is called stable) against the experimental transition temperature, per system and model. T* is a **diagnostic, not a prediction**: a single-mode treatment is not expected to reproduce an absolute T_c, and the screen fails to order PbTiO₃ against the other perovskite anchors (§3.2). For the bcc metals, of the 15 (system, model) rows the screen places T* below the experimental transition in 11, and never calls the bcc phase stable on the ladder in the other 4 (MatterSim on Hf, Ti, Zr; ORB-v2 on Ti); the ladder ends at 900 K, below every bcc transition (1136 K and above), so those rows are censored and do not say on which side of T_c the screen's T* lies. Of the rows below the transition, the 4 for CHGNet and MACE-MP-0 on Zr and Hf have no harmonic instability at all, so their T* = 100 K is the bottom of the ladder rather than a measured stabilisation. A T* of 100 K means only that the phase is called stable at the lowest ladder temperature. `analysis.predicted_tstar(df)`.

| System | Model | Experimental T_c (K) | Predicted T* (K) |
|---|---|---|---|
| agi_bcc | CHGNet | 420 | never stable |
| agi_bcc | MACE-MP-0 | 420 | never stable |
| agi_bcc | MatterSim | 420 | never stable |
| agi_bcc | ORB-v2 | 420 | never stable |
| agi_bcc | SevenNet-0 | 420 | never stable |
| batio3_cubic | CHGNet | 393 | 300 |
| batio3_cubic | MACE-MP-0 | 393 | 300 |
| batio3_cubic | MatterSim | 393 | 300 |
| batio3_cubic | ORB-v2 | 393 | 600 |
| batio3_cubic | SevenNet-0 | 393 | 300 |
| c_diamond | CHGNet | -- | 100 |
| c_diamond | MACE-MP-0 | -- | 100 |
| c_diamond | MatterSim | -- | 100 |
| c_diamond | ORB-v2 | -- | 100 |
| c_diamond | SevenNet-0 | -- | 100 |
| ceo2_cubic | CHGNet | -- | 100 |
| ceo2_cubic | MACE-MP-0 | -- | 100 |
| ceo2_cubic | MatterSim | -- | 100 |
| ceo2_cubic | ORB-v2 | -- | 100 |
| ceo2_cubic | SevenNet-0 | -- | 100 |
| cspbi3_cubic | CHGNet | 600 | never stable |
| cspbi3_cubic | MACE-MP-0 | 600 | never stable |
| cspbi3_cubic | MatterSim | 600 | never stable |
| cspbi3_cubic | ORB-v2 | 600 | never stable |
| cspbi3_cubic | SevenNet-0 | 600 | never stable |
| cssnbr3_cubic | CHGNet | 292 | never stable |
| cssnbr3_cubic | MACE-MP-0 | 292 | 900 |
| cssnbr3_cubic | MatterSim | 292 | 900 |
| cssnbr3_cubic | ORB-v2 | 292 | 600 |
| cssnbr3_cubic | SevenNet-0 | 292 | never stable |
| cssni3_cubic | CHGNet | 426 | never stable |
| cssni3_cubic | MACE-MP-0 | 426 | never stable |
| cssni3_cubic | MatterSim | 426 | 900 |
| cssni3_cubic | ORB-v2 | 426 | 100 |
| cssni3_cubic | SevenNet-0 | 426 | never stable |
| cu_fcc | CHGNet | -- | 100 |
| cu_fcc | MACE-MP-0 | -- | 100 |
| cu_fcc | MatterSim | -- | 100 |
| cu_fcc | ORB-v2 | -- | 100 |
| cu_fcc | SevenNet-0 | -- | 100 |
| hf_bcc | CHGNet | 2013 | 100 |
| hf_bcc | MACE-MP-0 | 2013 | 100 |
| hf_bcc | MatterSim | 2013 | never stable |
| hf_bcc | ORB-v2 | 2013 | 100 |
| hf_bcc | SevenNet-0 | 2013 | 100 |
| hfo2_cubic | CHGNet | 2800 | never stable |
| hfo2_cubic | MACE-MP-0 | 2800 | 600 |
| hfo2_cubic | MatterSim | 2800 | never stable |
| hfo2_cubic | ORB-v2 | 2800 | never stable |
| hfo2_cubic | SevenNet-0 | 2800 | never stable |
| knbo3_cubic | CHGNet | 708 | 300 |
| knbo3_cubic | MACE-MP-0 | 708 | 300 |
| knbo3_cubic | MatterSim | 708 | 300 |
| knbo3_cubic | ORB-v2 | 708 | 600 |
| knbo3_cubic | SevenNet-0 | 708 | 300 |
| ktao3_cubic | CHGNet | -- | 100 |
| ktao3_cubic | MACE-MP-0 | -- | 100 |
| ktao3_cubic | MatterSim | -- | 100 |
| ktao3_cubic | ORB-v2 | -- | 100 |
| ktao3_cubic | SevenNet-0 | -- | 100 |
| mgo_rocksalt | CHGNet | -- | 100 |
| mgo_rocksalt | MACE-MP-0 | -- | 100 |
| mgo_rocksalt | MatterSim | -- | 100 |
| mgo_rocksalt | ORB-v2 | -- | 100 |
| mgo_rocksalt | SevenNet-0 | -- | 100 |
| nacl_rocksalt | CHGNet | -- | 100 |
| nacl_rocksalt | MACE-MP-0 | -- | 100 |
| nacl_rocksalt | MatterSim | -- | 100 |
| nacl_rocksalt | ORB-v2 | -- | 100 |
| nacl_rocksalt | SevenNet-0 | -- | 100 |
| pbtio3_cubic | CHGNet | 763 | 900 |
| pbtio3_cubic | MACE-MP-0 | 763 | 100 |
| pbtio3_cubic | MatterSim | 763 | 100 |
| pbtio3_cubic | ORB-v2 | 763 | 100 |
| pbtio3_cubic | SevenNet-0 | 763 | 600 |
| si_diamond | CHGNet | -- | 100 |
| si_diamond | MACE-MP-0 | -- | 100 |
| si_diamond | MatterSim | -- | 100 |
| si_diamond | ORB-v2 | -- | 100 |
| si_diamond | SevenNet-0 | -- | 100 |
| srtio3_cubic | CHGNet | 105 | 100 |
| srtio3_cubic | MACE-MP-0 | 105 | 300 |
| srtio3_cubic | MatterSim | 105 | 300 |
| srtio3_cubic | ORB-v2 | 105 | 100 |
| srtio3_cubic | SevenNet-0 | 105 | 300 |
| ti_bcc | CHGNet | 1155 | 600 |
| ti_bcc | MACE-MP-0 | 1155 | 100 |
| ti_bcc | MatterSim | 1155 | never stable |
| ti_bcc | ORB-v2 | 1155 | never stable |
| ti_bcc | SevenNet-0 | 1155 | 100 |
| zr_bcc | CHGNet | 1136 | 100 |
| zr_bcc | MACE-MP-0 | 1136 | 100 |
| zr_bcc | MatterSim | 1136 | never stable |
| zr_bcc | ORB-v2 | 1136 | 100 |
| zr_bcc | SevenNet-0 | 1136 | 100 |
| zro2_cubic | CHGNet | 2570 | never stable |
| zro2_cubic | MACE-MP-0 | 2570 | never stable |
| zro2_cubic | MatterSim | 2570 | never stable |
| zro2_cubic | ORB-v2 | 2570 | never stable |
| zro2_cubic | SevenNet-0 | 2570 | never stable |

**Table S7** Composition of the two analysis sets. Both rest on the **same 15 systems**, so they share their clustering structure: the 60 guardrail units are 15 systems × 4 temperatures, and the 75 matched pairs are 15 systems × 5 models. In the guardrail set the five model votes are already collapsed into each unit, so the models are not an independent axis there and the clustering unit is the system. Systems whose identifier contains `bcc` are dropped, which also removes the superionic `agi_bcc` (§3.2), so each set has 15 systems rather than 16.

| Analysis set | n units | n systems | Temperature axis | Model axis | Balanced |
|---|---|---|---|---|---|
| n = 60 (H3 guardrail, §3.4) | 60 | 15 | 4 (100/300/600/900 K) | collapsed into each unit | yes |
| n = 75 (H2 matched set, §3.2) | 75 | 15 | 1 per analysis (each T analysed separately) | 5 (an explicit axis) | yes |

The 15 systems are: `batio3_cubic`, `c_diamond`, `ceo2_cubic`, `cspbi3_cubic`, `cssnbr3_cubic`, `cssni3_cubic`, `cu_fcc`, `hfo2_cubic`, `knbo3_cubic`, `mgo_rocksalt`, `nacl_rocksalt`, `pbtio3_cubic`, `si_diamond`, `srtio3_cubic`, `zro2_cubic`.

**Table S8** The ensemble-disagreement guardrail set in full: every (system, temperature) unit behind the H3 analysis, with the cross-model frequency spread, the stable-vote fraction, and whether the majority consensus was correct. This is the n = 60 set. `analysis.h3_ensemble_guardrail(df, "softmode")`.

| System | T (K) | n models | Freq. std (THz) | Stable-vote frac. | Vote | Consensus | Label | Consensus correct |
|---|---|---|---|---|---|---|---|---|
| batio3_cubic | 100 | 5 | 0.064 | 0.00 | unanimous | unstable | unstable | yes |
| batio3_cubic | 300 | 5 | 0.078 | 0.80 | split | stable | unstable | **no** |
| batio3_cubic | 600 | 5 | 0.095 | 1.00 | unanimous | stable | stable | yes |
| batio3_cubic | 900 | 5 | 0.121 | 1.00 | unanimous | stable | stable | yes |
| c_diamond | 100 | 5 | 4.555 | 1.00 | unanimous | stable | stable | yes |
| c_diamond | 300 | 5 | 4.555 | 1.00 | unanimous | stable | stable | yes |
| c_diamond | 600 | 5 | 4.555 | 1.00 | unanimous | stable | stable | yes |
| c_diamond | 900 | 5 | 4.555 | 1.00 | unanimous | stable | stable | yes |
| ceo2_cubic | 100 | 5 | 0.405 | 1.00 | unanimous | stable | stable | yes |
| ceo2_cubic | 300 | 5 | 0.405 | 1.00 | unanimous | stable | stable | yes |
| ceo2_cubic | 600 | 5 | 0.405 | 1.00 | unanimous | stable | stable | yes |
| ceo2_cubic | 900 | 5 | 0.405 | 1.00 | unanimous | stable | stable | yes |
| cspbi3_cubic | 100 | 5 | 0.255 | 0.00 | unanimous | unstable | unstable | yes |
| cspbi3_cubic | 300 | 5 | 0.309 | 0.00 | unanimous | unstable | unstable | yes |
| cspbi3_cubic | 600 | 5 | 0.253 | 0.00 | unanimous | unstable | stable | **no** |
| cspbi3_cubic | 900 | 5 | 0.054 | 0.00 | unanimous | unstable | stable | **no** |
| cssnbr3_cubic | 100 | 5 | 0.323 | 0.00 | unanimous | unstable | unstable | yes |
| cssnbr3_cubic | 300 | 5 | 0.019 | 0.00 | unanimous | unstable | stable | **no** |
| cssnbr3_cubic | 600 | 5 | 0.022 | 0.20 | split | unstable | stable | **no** |
| cssnbr3_cubic | 900 | 5 | 0.024 | 0.60 | split | stable | stable | yes |
| cssni3_cubic | 100 | 5 | 0.236 | 0.20 | split | unstable | unstable | yes |
| cssni3_cubic | 300 | 5 | 0.108 | 0.20 | split | unstable | unstable | yes |
| cssni3_cubic | 600 | 5 | 0.136 | 0.20 | split | unstable | stable | **no** |
| cssni3_cubic | 900 | 5 | 0.155 | 0.40 | split | unstable | stable | **no** |
| cu_fcc | 100 | 5 | 0.612 | 1.00 | unanimous | stable | stable | yes |
| cu_fcc | 300 | 5 | 0.612 | 1.00 | unanimous | stable | stable | yes |
| cu_fcc | 600 | 5 | 0.612 | 1.00 | unanimous | stable | stable | yes |
| cu_fcc | 900 | 5 | 0.612 | 1.00 | unanimous | stable | stable | yes |
| hfo2_cubic | 100 | 5 | 12.225 | 0.00 | unanimous | unstable | unstable | yes |
| hfo2_cubic | 300 | 5 | 0.228 | 0.00 | unanimous | unstable | unstable | yes |
| hfo2_cubic | 600 | 5 | 0.283 | 0.20 | split | unstable | unstable | yes |
| hfo2_cubic | 900 | 5 | 0.314 | 0.20 | split | unstable | unstable | yes |
| knbo3_cubic | 100 | 5 | 0.407 | 0.00 | unanimous | unstable | unstable | yes |
| knbo3_cubic | 300 | 5 | 0.591 | 0.80 | split | stable | unstable | **no** |
| knbo3_cubic | 600 | 5 | 0.745 | 1.00 | unanimous | stable | unstable | **no** |
| knbo3_cubic | 900 | 5 | 0.849 | 1.00 | unanimous | stable | stable | yes |
| mgo_rocksalt | 100 | 5 | 0.932 | 1.00 | unanimous | stable | stable | yes |
| mgo_rocksalt | 300 | 5 | 0.932 | 1.00 | unanimous | stable | stable | yes |
| mgo_rocksalt | 600 | 5 | 0.932 | 1.00 | unanimous | stable | stable | yes |
| mgo_rocksalt | 900 | 5 | 0.932 | 1.00 | unanimous | stable | stable | yes |
| nacl_rocksalt | 100 | 5 | 0.423 | 1.00 | unanimous | stable | stable | yes |
| nacl_rocksalt | 300 | 5 | 0.423 | 1.00 | unanimous | stable | stable | yes |
| nacl_rocksalt | 600 | 5 | 0.423 | 1.00 | unanimous | stable | stable | yes |
| nacl_rocksalt | 900 | 5 | 0.423 | 1.00 | unanimous | stable | stable | yes |
| pbtio3_cubic | 100 | 5 | 0.830 | 0.60 | split | stable | unstable | **no** |
| pbtio3_cubic | 300 | 5 | 1.025 | 0.60 | split | stable | unstable | **no** |
| pbtio3_cubic | 600 | 5 | 1.184 | 0.80 | split | stable | unstable | **no** |
| pbtio3_cubic | 900 | 5 | 1.293 | 1.00 | unanimous | stable | stable | yes |
| si_diamond | 100 | 5 | 0.489 | 1.00 | unanimous | stable | stable | yes |
| si_diamond | 300 | 5 | 0.489 | 1.00 | unanimous | stable | stable | yes |
| si_diamond | 600 | 5 | 0.489 | 1.00 | unanimous | stable | stable | yes |
| si_diamond | 900 | 5 | 0.489 | 1.00 | unanimous | stable | stable | yes |
| srtio3_cubic | 100 | 5 | 0.157 | 0.40 | split | unstable | unstable | yes |
| srtio3_cubic | 300 | 5 | 0.087 | 1.00 | unanimous | stable | stable | yes |
| srtio3_cubic | 600 | 5 | 0.242 | 1.00 | unanimous | stable | stable | yes |
| srtio3_cubic | 900 | 5 | 0.356 | 1.00 | unanimous | stable | stable | yes |
| zro2_cubic | 100 | 5 | 2.156 | 0.00 | unanimous | unstable | unstable | yes |
| zro2_cubic | 300 | 5 | 1.066 | 0.00 | unanimous | unstable | unstable | yes |
| zro2_cubic | 600 | 5 | 0.220 | 0.00 | unanimous | unstable | unstable | yes |
| zro2_cubic | 900 | 5 | 0.253 | 0.00 | unanimous | unstable | unstable | yes |

**Table S9** Every pooled rate that ORB-v2 could plausibly drive, with and without it. ORB-v2 is the only one of the five models whose forces are predicted directly rather than as the gradient of an energy (non-conservative). As run, CHGNet, SevenNet-0, MatterSim and ORB-v2 all return float32 forces and only MACE-MP-0 returns float64, so precision does not single ORB-v2 out, and the design cannot separate its architecture from anything else about it. For reference: ORB-v2 is the only model that calls MgO harmonically unstable (−1.07 THz); its most negative bcc screen curvatures are −6.2 THz (Ti, 100 K) and −8.9 THz (Ti, 900 K), with Hf at −0.09 THz; and it accounts for 3 of the 8 SSCHA blow-ups and 4 of the 7 failed SSCHA units (all on pbtio3_cubic). Per-model rates (Tables S4 and S5) cannot change when another model is removed, so only pooled quantities are split here. The guardrail is where removing ORB-v2 changes the reading: without it the vote-split AUC falls from 0.762 to 0.628 and its cluster-bootstrap interval spans 0.5, so the guardrail is suggestive and not robust to removing ORB-v2. The frequency spread is the cross-model standard deviation of the screen's effective frequency, a potential-energy-surface proxy that is only partly force-derived (the mode patterns come from force constants; E(Q) is energy-only); it is not a force-level ensemble uncertainty. <!-- PENDING-C2: force-level ensemble spread (five-model spread plus MACE small/medium/large and MatterSim 1M/5M committees; clustered AUC; overlap-artifact check), with and without ORB-v2; new ESI table, one sentence here --> Four-model consensus uses the same rule as five-model consensus (stable when at least half the votes are stable), so a 2–2 split is called stable; this decides 3 units, 2 of them wrong. Removing ORB-v2 raises the bcc curvature-sign agreement from 0.78 to 0.83 but leaves the stability-call agreement at 0.69. Permutation p-values and bootstrap intervals resample whole systems (10,000 permutations and 10,000 bootstrap draws, seed 0; the 9,998 bootstrap draws that contain both outcomes are used), so each p carries Monte Carlo error. The H2 ladder without ORB-v2 is in Table S15, the paired screen-versus-SSCHA tests in Table S10, the bcc agreement by model in Table S16, and the SSCHA high-temperature false-unstables, blow-ups and failures by model in Table S17. `scripts/stats_hardening.py` (`orb_split_s3`, `bcc_agreement`); pooled finite-T accuracy from `scripts/screen_sensitivity.py`.

| Quantity | All five models | Excluding ORB-v2 |
|---|---|---|
| Harmonic, pooled false-stable / false-unstable at tolerance 0 THz | 1 / 29 of 95 | 0 / 23 of 76 |
| Harmonic, pooled false-stable / false-unstable at tolerance 0.05 THz | 5 / 3 of 95 | 4 / 2 of 76 |
| Harmonic, pooled false-stable / false-unstable at tolerance 0.1 THz | 5 / 3 of 95 | 4 / 2 of 76 |
| Harmonic, pooled false-stable / false-unstable at tolerance 0.2 THz | 6 / 3 of 95 | 4 / 2 of 76 |
| Harmonic, pooled false-stable / false-unstable at tolerance 0.3 THz | 7 / 1 of 95 | 5 / 0 of 76 |
| Harmonic, pooled false-stable / false-unstable at tolerance 0.5 THz | 14 / 1 of 95 | 6 / 0 of 76 |
| Screen, pooled scored accuracy, T ≤ 300 K | 127/150 = 0.847 [0.780, 0.896] | 103/120 = 0.858 [0.785, 0.910] |
| Screen, pooled false-stable rate, T ≤ 300 K | 18/80 = 0.225 [0.147, 0.328] | 13/64 = 0.203 [0.123, 0.317] |
| Screen recall of the unstable class, ferroelectric oxides, T ≤ 300 K | 16/30 = 0.533 [0.361, 0.698] | 12/24 = 0.500 [0.314, 0.686] |
| Screen recall of the unstable class, halide perovskites, T ≤ 300 K | 23/25 = 0.920 [0.750, 0.978] | 20/20 = 1.000 [0.839, 1.000] |
| Screen recall of the unstable class, fluorites, T ≤ 300 K | 20/20 = 1.000 [0.839, 1.000] | 16/16 = 1.000 [0.806, 1.000] |
| Screen recall of the unstable class, SrTiO₃, T ≤ 300 K | 3/5 = 0.600 [0.231, 0.882] | 3/4 = 0.750 [0.301, 0.954] |
| Screen false-unstable on the six controls, all four temperatures | 0/120 = 0.000 [0.000, 0.031] | 0/96 = 0.000 [0.000, 0.038] |
| SSCHA recall of the unstable class, ferroelectric oxides, T ≤ 300 K | 5/27 = 0.185 [0.082, 0.367] | 5/23 = 0.217 [0.097, 0.419] |
| SSCHA recall of the unstable class, halide perovskites (SSCHA ran on CsSnI₃ only), T ≤ 300 K | 7/8 = 0.875 [0.529, 0.978] | 7/8 = 0.875 [0.529, 0.978] |
| SSCHA recall of the unstable class, fluorites, T ≤ 300 K | 1/20 = 0.050 [0.009, 0.236] | 0/16 = 0.000 [0.000, 0.194] |
| SSCHA recall of the unstable class, SrTiO₃, T ≤ 300 K | 5/5 = 1.000 [0.566, 1.000] | 4/4 = 1.000 [0.510, 1.000] |
| SSCHA numerical blow-ups (\|f\| > 50 THz), of returned units | 8 of 201 | 5 of 166 |
| SSCHA failed units (no number returned), of the attempted grid | 7 of 208 | 3 of 169 |
| bcc screen-vs-SSCHA curvature-sign agreement | 35/45 = 0.778 [0.637, 0.875] | 30/36 = 0.833 [0.681, 0.921] |
| bcc screen-vs-SSCHA stability-call agreement | 31/45 = 0.689 [0.543, 0.805] | 25/36 = 0.694 [0.531, 0.820] |
| bcc frequency Spearman ρ (descriptive, no test) | +0.113 | −0.003 |
| Guardrail: consensus error on split-vote units | 8/14 = 0.571 [0.326, 0.786] | 4/8 = 0.500 [0.215, 0.785] |
| Guardrail: consensus error on unanimous units | 4/46 = 0.087 [0.034, 0.203] | 8/52 = 0.154 [0.080, 0.275] |
| Guardrail: vote-split AUC [cluster-bootstrap 95% CI], clustered permutation p | 0.762 [0.590, 0.934], clustered p = 0.0035 | 0.628 [0.438, 0.839], clustered p = 0.0469 |
| Guardrail: frequency-spread AUC [cluster-bootstrap 95% CI], clustered permutation p | 0.361 [0.046, 0.625], clustered p = 0.2748 | 0.299 [0.076, 0.516], clustered p = 0.1030 |
| Guardrail: 2–2 tied units (broken to 'stable'), of which wrong | 0, 0 wrong | 3, 2 wrong |

**Table S10** The central screen-versus-SSCHA contrast, tested as the paired comparison it is. Both methods see the same (system, model, temperature) units, so comparing their two marginal Wilson intervals would ignore the pairing. Two p-values are given and **the unit-level one should not be quoted**: an exact McNemar over discordant units also assumes those units are independent, and they cluster by system. The clustered column randomises the method label over whole systems and is exact at this size. Its resolution floor, 2/2^k, is given alongside, because with five displacive systems no arrangement of the data can reach p < 0.0625 and with two fluorite systems none can go below 0.5. The reportable content of this table is the size and the consistency of the effect (4 of 5 systems favour the screen, with per-system net discordances +6, +6, −1, +9, +10) rather than a significance claim. Why SSCHA loses these units is examined separately, on the same MLIP energies, in Table S16 (lower part) and §3.3: the screen's own symmetric-point curvature, the single-mode analogue of the SSCHA free-energy Hessian, is positive on most of them too. `scripts/stats_hardening.py`.

| System set | Model set | n paired | Discordant (screen/SSCHA) | Unit-level p (do not quote) | Systems favouring screen | Clustered p | Floor |
|---|---|---|---|---|---|---|---|
| fe oxide | all models | 27 | 14/3 | 0.013 | 2/3 | 0.5000 | 0.2500 |
| fe oxide | excl orb v2 | 23 | 10/3 | 0.092 | 2/3 | 0.5000 | 0.2500 |
| fluorite | all models | 20 | 19/0 | 3.8e-06 | 2/2 | 0.5000 | 0.5000 |
| fluorite | excl orb v2 | 16 | 16/0 | 3e-05 | 2/2 | 0.5000 | 0.5000 |
| displacive combined | all models | 47 | 33/3 | 2e-07 | 4/5 | 0.1250 | 0.0625 |
| displacive combined | excl orb v2 | 39 | 26/3 | 1.5e-05 | 4/5 | 0.1250 | 0.0625 |

**Table S11** SSCHA numerical quality, aggregated per family and model (not per unit). The SSCHA settings are identical for all 201 returned units and are therefore stated once here rather than tabulated. The starting dynamical matrix is built from phonopy full force constants at a 0.03 Å finite displacement, made positive definite (`ForcePositiveDefinite`) and symmetrised. The auxiliary dynamical matrix is relaxed in the root2 representation with `min_step_dyn = 0.5`; the convergence threshold is `meaningful_factor = 1e-4`. Each population of 256 configurations allows at most `max_ka = 20` reweighting steps, a cap that forces a fresh ensemble rather than a stopping criterion (§S2.1), and at most 8 populations are drawn (2048 configurations). The free-energy Hessian is then evaluated at bubble level (`include_v4 = False`) on a dedicated 512-configuration ensemble at the relaxed auxiliary matrix. The harness did not record how many populations each unit used or whether it met the convergence threshold before the population cap, so 2560 configurations is the configured maximum per unit, not a count. The SSCHA here inherits the MLIP potential-energy surface; nothing in this table tests that surface.

Two quantities are tabulated from the six lowest recorded Hessian frequencies. *Acoustic zeros resolved* counts units in which all three translational zeros (|ω| < 0.001 THz) appear within that window, and the residual column gives the largest of their magnitudes over those units. This is a check that the Hessian was symmetrised correctly; it is not a bound on the stochastic noise of the soft mode (§S2.4 gives the only seed spread measured). *Swamped* counts the opposite case: units with **no** recorded mode near zero, meaning at least six modes lie below the acoustic branches. Swamping is not itself an error: a deeply unstable phase genuinely has many imaginary modes, and the reported minimum frequency excludes the acoustic branches from the full spectrum rather than from this window. It does separate the families sharply. Read swamping as a fraction rather than a count, because the per-model denominators differ: on the perovskites it runs from 8/19 = 0.42 for MACE-MP-0 to 14/20 = 0.70 for CHGNet, so it is present for every architecture but is not uniform across them. On the fluorites it is **not** architecture-neutral, being 4/8 for ORB-v2 and 2/8 for MACE-MP-0 against 0/8 for the other three.

**What the harness did not retain**, and what would therefore need a re-run to supply: the per-iteration free-energy gradient history, and a per-unit uncertainty on the Hessian eigenvalues. The only uncertainty probe in the production data is the independent-seed study of §S2.4, which covers one unit, bcc-Zr/MACE-MP-0 at 100 K, on which the harmonic layer finds no bcc instability. <!-- PENDING-C1: four-seed SSCHA with recorded gradient/error history, population count and convergence flag for BaTiO3/MACE-MP-0 100 K, ZrO2/MACE-MP-0 100 K, Zr/MatterSim 50 K and SrTiO3/MACE-MP-0 600 K; the seed spread of the lowest Hessian eigenvalue is the Hessian uncertainty; results go in a new ESI table referenced here and in §S2.4 -->

| Family | Model | n units | Acoustic zeros resolved | Max zero residual (THz) | Swamped | Wall time (s) |
|---|---|---|---|---|---|---|
| bcc | CHGNet | 15 | 15/15 | 7.0e-22 | 0 | 166–203 |
| bcc | MACE-MP-0 | 15 | 15/15 | 2.3e-31 | 0 | 40–126 |
| bcc | MatterSim | 15 | 15/15 | 1.9e-22 | 0 | 127–148 |
| bcc | ORB-v2 | 15 | 15/15 | 0.0e+00 | 0 | 77–83 |
| bcc | SevenNet-0 | 15 | 15/15 | 2.5e-14 | 0 | 171–230 |
| fluorite | CHGNet | 8 | 8/8 | 4.9e-07 | 0 | 175–185 |
| fluorite | MACE-MP-0 | 8 | 6/8 | 4.3e-07 | 2 | 163–174 |
| fluorite | MatterSim | 8 | 8/8 | 4.6e-07 | 0 | 142–156 |
| fluorite | ORB-v2 | 8 | 3/8 | 6.0e-07 | 4 | 79–85 |
| fluorite | SevenNet-0 | 8 | 8/8 | 4.0e-07 | 0 | 179–184 |
| perovskite | CHGNet | 20 | 6/20 | 9.2e-07 | 14 | 199–248 |
| perovskite | MACE-MP-0 | 19 | 10/19 | 3.5e-07 | 8 | 184–240 |
| perovskite | MatterSim | 18 | 9/18 | 3.3e-07 | 9 | 157–180 |
| perovskite | ORB-v2 | 12 | 4/12 | 1.5e-06 | 8 | 96–104 |
| perovskite | SevenNet-0 | 17 | 9/17 | 3.5e-07 | 8 | 190–228 |

**Table S13** Sensitivity of the harmonic layer to the finite-displacement amplitude (0.005, 0.02 and 0.03 Å against the production 0.01 Å), all five models: an axis that ESI §S1.2's v1/v2 replicate cannot probe. Deviations are against the same model's production 0.01 Å row.

| Model | Amplitude (Å) | n | Median \|Δ\| (THz) | Max \|Δ\| (THz) | Call flips | Harmonic accuracy [95% CI] |
|---|---|---|---|---|---|---|
| CHGNet | 0.005 | 19 | 0.0162 | 2.0077 | 1 | 14/19 = 0.737 [0.512, 0.882] |
| CHGNet | 0.01 (production) | 20 | -- | -- | 0 | 15/19 = 0.789 [0.567, 0.915] |
| CHGNet | 0.02 | 19 | 0.0559 | 2.5497 | 1 | 16/19 = 0.842 [0.624, 0.945] |
| CHGNet | 0.03 | 19 | 0.1282 | 2.4710 | 2 | 17/19 = 0.895 [0.686, 0.971] |
| MACE-MP-0 | 0.005 | 19 | 0.0000 | 0.0251 | 0 | 17/19 = 0.895 [0.686, 0.971] |
| MACE-MP-0 | 0.01 (production) | 20 | -- | -- | 0 | 17/19 = 0.895 [0.686, 0.971] |
| MACE-MP-0 | 0.02 | 19 | 0.0002 | 0.1032 | 0 | 17/19 = 0.895 [0.686, 0.971] |
| MACE-MP-0 | 0.03 | 19 | 0.0004 | 0.2413 | 0 | 17/19 = 0.895 [0.686, 0.971] |
| MatterSim | 0.005 | 19 | 0.0002 | 0.0111 | 0 | 19/19 = 1.000 [0.832, 1.000] |
| MatterSim | 0.01 (production) | 20 | -- | -- | 0 | 19/19 = 1.000 [0.832, 1.000] |
| MatterSim | 0.02 | 19 | 0.0008 | 0.0367 | 0 | 19/19 = 1.000 [0.832, 1.000] |
| MatterSim | 0.03 | 19 | 0.0022 | 0.1000 | 0 | 19/19 = 1.000 [0.832, 1.000] |
| ORB-v2 | 0.005 | 19 | 0.1930 | 5.7574 | 5 | 16/19 = 0.842 [0.624, 0.945] |
| ORB-v2 | 0.01 (production) | 20 | -- | -- | 0 | 17/19 = 0.895 [0.686, 0.971] |
| ORB-v2 | 0.02 | 19 | 0.1095 | 2.0425 | 3 | 16/19 = 0.842 [0.624, 0.945] |
| ORB-v2 | 0.03 | 19 | 0.1510 | 2.9324 | 3 | 16/19 = 0.842 [0.624, 0.945] |
| SevenNet-0 | 0.005 | 19 | 0.0001 | 0.0098 | 0 | 19/19 = 1.000 [0.832, 1.000] |
| SevenNet-0 | 0.01 (production) | 20 | -- | -- | 0 | 19/19 = 1.000 [0.832, 1.000] |
| SevenNet-0 | 0.02 | 19 | 0.0006 | 0.0417 | 0 | 19/19 = 1.000 [0.832, 1.000] |
| SevenNet-0 | 0.03 | 19 | 0.0015 | 0.1050 | 0 | 19/19 = 1.000 [0.832, 1.000] |

**MACE-MP-0, MatterSim, SevenNet-0 change no stability call at any amplitude**; their harmonic accuracies are MACE-MP-0 17/19, MatterSim 19/19, SevenNet-0 19/19 at every amplitude. CHGNet changes calls only on harmonically stable controls (ceo2_cubic at 0.03 Å (-0.267 → -0.000 THz); cu_fcc at 0.005 Å (-0.000 → -0.476 THz); nacl_rocksalt at 0.02 Å (-0.238 → -0.000 THz); nacl_rocksalt at 0.03 Å (-0.238 → -0.000 THz)), so its harmonic accuracy runs from 14/19 to 17/19 across amplitudes; with its tolerance dependence (§3.1), it is not a robust number under either knob. ORB-v2 changes calls on test systems as well as controls (cssni3_cubic at 0.005 Å (-0.316 → -0.089 THz); hf_bcc at 0.005 Å (-0.194 → -0.084 THz); hf_bcc at 0.02 Å (-0.194 → +0.000 THz); hf_bcc at 0.03 Å (-0.194 → -0.000 THz); hfo2_cubic at 0.005 Å (-5.757 → -0.000 THz); mgo_rocksalt at 0.005 Å (-1.065 → -0.000 THz); srtio3_cubic at 0.005 Å (+0.000 → -0.162 THz); srtio3_cubic at 0.02 Å (+0.000 → -0.588 THz); srtio3_cubic at 0.03 Å (+0.000 → -1.050 THz); ti_bcc at 0.02 Å (-0.434 → -0.000 THz); ti_bcc at 0.03 Å (-0.434 → -0.000 THz)); its accuracy runs 16/19 to 17/19. Its harmonic calls are amplitude-dependent. Of the five models it is the only one whose forces are predicted directly rather than as gradients of an energy (non-conservative), a plausible cause that this design cannot isolate; CHGNet, SevenNet-0 and MatterSim also return float32 forces as run and do not show it, so precision alone does not explain it. Its harmonic calls should be read with that caveat. `scripts/run_disp_sweep.py`.

**Table S14** Sensitivity of the single-mode soft-mode screen to its fit window, sampling range, frozen-cell normalisation and solver constants. Every row re-solves the cached E(Q) maps with the production solver; no MLIP is re-evaluated. Each cell gives all five models, then ORB-v2 excluded (all · ex-ORB). Unit flips count (system, model, T) stability calls that differ from production. FE recall is the recall of the unstable class on BaTiO₃, KNbO₃ and PbTiO₃ at T ≤ 300 K. The SrTiO₃ gate counts models that condense at 100 K and are stable at 300, 600 and 900 K. Accuracy is on the scored finite-T set (T ≤ 300 K, bcc and KTaO₃ excluded); the all-T column scores the same systems over 100 to 900 K, with the false-unstable count in parentheses. Intervals are unit-level Wilson and do not account for clustering by system. The production baseline reproduces all 400 unit calls and all 1512 mode-temperature calls in the ledger. `scripts/screen_sensitivity.py` → `results/screen_sensitivity.json`.

| Axis | Setting | Unit flips, T ≤ 300 K | Unit flips, all T | FE recall [95% CI] | SrTiO₃ gate | Accuracy, T ≤ 300 K [95% CI] | Accuracy, all T [95% CI] (false-unstable) |
|---|---|---|---|---|---|---|---|
| Production | production: window 5× / floor 60 meV, Q ≤ 0.45 Å, minimal cell, threshold 1.5 steps, box 0.6 Å | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 3× well depth, floor 30 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 5× well depth, floor 30 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 8× well depth, floor 30 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 3× well depth, floor 60 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 8× well depth, floor 60 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 3× well depth, floor 120 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 5× well depth, floor 120 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (a) Fit window | window 8× well depth, floor 120 meV | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (b) Sampling range | sampled Q ≤ 0.25 Å | 1/200 · 1/160 | 8/400 · 7/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 250/300 [0.79, 0.87] (21/190) · 199/240 [0.78, 0.87] (19/152) |
| (b) Sampling range | sampled Q ≤ 0.30 Å | 0/200 · 0/160 | 2/400 · 2/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 245/300 [0.77, 0.86] (26/190) · 195/240 [0.76, 0.86] (23/152) |
| (b) Sampling range | sampled Q ≤ 0.35 Å | 0/200 · 0/160 | 2/400 · 2/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 245/300 [0.77, 0.86] (26/190) · 195/240 [0.76, 0.86] (23/152) |
| (b) Sampling range | sampled Q ≤ 0.40 Å | 0/200 · 0/160 | 1/400 · 1/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 244/300 [0.77, 0.85] (27/190) · 194/240 [0.75, 0.85] (24/152) |
| (c) Frozen-cell normalisation | common force-constant supercell (q-search cell) | 14/200 · 14/160 | 41/400 · 38/320 | 28/30 [0.79, 0.98] · 24/24 [0.86, 1.00] | 3/5 · 3/4 | 139/150 [0.87, 0.96] · 115/120 [0.91, 0.98] | 252/300 [0.79, 0.88] (41/190) · 203/240 [0.79, 0.89] (36/152) |
| (c) Frozen-cell normalisation | per formula unit | 39/200 · 31/160 | 89/400 · 73/320 | 5/30 [0.07, 0.34] · 5/24 [0.09, 0.40] | 0/5 · 0/4 | 105/150 [0.62, 0.77] · 85/120 [0.62, 0.78] | 225/300 [0.70, 0.80] (0/190) · 181/240 [0.70, 0.80] (0/152) |
| (c) Frozen-cell normalisation | doubled minimal cell | 11/200 · 11/160 | 23/400 · 19/320 | 26/30 [0.70, 0.95] · 22/24 [0.74, 0.98] | 2/5 · 2/4 | 136/150 [0.85, 0.94] · 112/120 [0.87, 0.97] | 249/300 [0.78, 0.87] (36/190) · 201/240 [0.79, 0.88] (30/152) |
| (c) Frozen-cell normalisation | 8× minimal cell | 18/200 · 18/160 | 58/400 · 52/320 | 28/30 [0.79, 0.98] · 24/24 [0.86, 1.00] | 0/5 · 0/4 | 136/150 [0.85, 0.94] · 112/120 [0.87, 0.97] | 234/300 [0.73, 0.82] (60/190) · 188/240 [0.73, 0.83] (52/152) |
| (d) Solver constants | condensation threshold 0.5 steps (0.0025 Å) | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (d) Solver constants | condensation threshold 3 steps (0.0150 Å) | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |
| (d) Solver constants | centroid box 1.2 Å, 241 points (same step) | 0/200 · 0/160 | 0/400 · 0/320 | 16/30 [0.36, 0.70] · 12/24 [0.31, 0.69] | 3/5 · 3/4 | 127/150 [0.78, 0.90] · 103/120 [0.78, 0.91] | 243/300 [0.76, 0.85] (28/190) · 193/240 [0.75, 0.85] (25/152) |

**Fit window.** None of the 8 non-production window settings (multiplier 3×, 5× or 8× the well depth, floor 30, 60 or 120 meV) changes a unit call at any temperature; mode-level condensation calls change on at most 14/756 mode-temperature evaluations at T ≤ 300 K. The multiplier on its own is a weak probe: at the production 60 meV floor the kept point set is identical at 3×, 5× and 8× for 261/378 modes (69%), including all 101 modes shallower than 7.5 meV, which is why the floor is varied as well.

**Sampling range.** Truncating each sampled E(Q) at 0.25 to 0.40 Å changes at most 1/200 unit calls at T ≤ 300 K. Over all four temperatures truncation at 0.25 Å changes 8/400 (cspbi3_cubic 2, cssnbr3_cubic 3, cssni3_cubic 2, ti_bcc 1) and at 0.40 Å 1/400 (cssnbr3_cubic 1). In the production maps 77/378 wells are still descending at the largest sampled Q (0.45 Å), 60/400 unit order parameters lie beyond it, and 26/400 sit at the 0.6 Å edge of the centroid scan; doubling the scan to 1.2 Å changes no call.

**Solver constants.** Every non-zero centroid is at least 0.035 Å, so condensation thresholds from 0.0025 to 0.015 Å (0.5 to 3 scan steps) cannot change a call. The bracketed width solve falls back to the grid-nearest σ² on 8441/182952 centroid evaluations (4.6%), and never at a deciding minimum (0/1512).

**Frozen-cell normalisation is outcome-determining.** Rescaling the cell over which M and V(Q) are summed by a factor n maps (a, b, c, M) to n(a, b, c, M), which is the minimal-cell problem at mass n²M and temperature T/n: a larger cell is a heavier, effectively colder and more classical mode, and condenses more readily. (The MLIP energies are extensive in the cell as assumed: for the SrTiO₃/MACE-MP-0 R mode the doubled-cell energy and mass are 2.000000 and 2.000000 times the minimal-cell values.) Taking the cell as per formula unit, the production minimal cell, a doubled minimal cell and the common force-constant supercell, FE recall runs 5/30, 16/30, 26/30 and 28/30 and the SrTiO₃ gate passes for 0/5, 3/5, 2/5 and 3/5 models. Accuracy cannot choose among the conventions. At T ≤ 300 K it is 105/150, 127/150, 136/150 and 139/150, but only 9/70 of the truly stable units in that set carry a screened imaginary mode, so the column is nearly blind to over-condensation. Over all four temperatures the extra condensation of larger cells shows up as false-unstables (0/190, 28/190, 36/190 and 41/190), and the 8× cell moves from above production at T ≤ 300 K (136/150 against 127/150) to below it over all T (234/300 against 243/300). Which temperatures are scored decides the ranking, so these data do not identify the physically right normalisation. The production convention is the minimal cell in which the mode is a single commensurate distortion (§S1.3), and the screen's T* and FE recall are conditional on it.

**Table S15** The H2 transfer asymmetry at every ladder temperature, with and without ORB-v2. b counts matched (system, model) pairs whose harmonic call is right and whose screen call at T is wrong; c counts the reverse. The matched set is 15 non-bcc, non-borderline systems × 5 models (75 pairs; 60 without ORB-v2). **The system-clustered p is the test**: exact enumeration of the 2^k per-system sign flips (`stats.cluster_exact_paired`), with the systems favouring b, favouring c and tied given alongside. The unit-level McNemar p treats the pairs as independent, which they are not (Table S7), and is given only as a labelled companion. The last three columns are a descriptive decomposition, not a test. A shared-screen-error system is one the screen mis-calls at that temperature for at least four of the five models; the set is defined on all five models and applied to both model sets. Because it is selected on the finite-temperature outcome, removing it removes b-type pairs by construction. The columns show where the asymmetry sits, in systems the screen mis-calls for nearly every model. They do not show that it is absent elsewhere, and they do not say whether an error shared by four or five models comes from the single-mode approximation or from a feature of the surface the models share (§S1.3). `scripts/stats_hardening.py` (`h2_clustered`).

| T (K) | Model set | b v c | Unit-level McNemar p (companion only) | System-clustered exact p | Systems b > c / c > b / tied | Shared-screen-error systems (models mis-called) | b v c inside them | b v c without them |
|---|---|---|---|---|---|---|---|---|
| 100 | all five | 5 v 3 | 0.727 | 0.844 | 3 / 3 / 9 | none | 0 v 0 | 5 v 3 (15 systems) |
| 100 | excluding ORB-v2 | 3 v 2 | 1.000 | 1.000 | 2 / 2 / 11 | none | 0 v 0 | 3 v 2 (15 systems) |
| 300 | all five | 17 v 4 | 0.007 | 0.152 | 5 / 4 / 6 | batio3_cubic (4/5), cssnbr3_cubic (5/5), knbo3_cubic (4/5) | 13 v 0 | 4 v 4 (12 systems) |
| 300 | excluding ORB-v2 | 14 v 2 | 0.004 | 0.156 | 4 / 2 / 9 | batio3_cubic (4/5), cssnbr3_cubic (5/5), knbo3_cubic (4/5) | 12 v 0 | 2 v 2 (12 systems) |
| 600 | all five | 23 v 4 | 0.0003 | 0.066 | 6 / 4 / 5 | cspbi3_cubic (5/5), cssnbr3_cubic (4/5), cssni3_cubic (4/5), knbo3_cubic (5/5), pbtio3_cubic (4/5) | 22 v 0 | 1 v 4 (10 systems) |
| 600 | excluding ORB-v2 | 20 v 2 | 0.0001 | 0.055 | 6 / 2 / 7 | cspbi3_cubic (5/5), cssnbr3_cubic (4/5), cssni3_cubic (4/5), knbo3_cubic (5/5), pbtio3_cubic (4/5) | 19 v 0 | 1 v 2 (10 systems) |
| 900 | all five | 11 v 4 | 0.118 | 0.414 | 4 / 4 / 7 | cspbi3_cubic (5/5) | 5 v 0 | 6 v 4 (14 systems) |
| 900 | excluding ORB-v2 | 10 v 2 | 0.039 | 0.250 | 4 / 2 / 9 | cspbi3_cubic (5/5) | 4 v 0 | 6 v 2 (14 systems) |

Leave-one-system-out: with all five models, dropping any single system never brings the clustered p below 0.05 at any temperature (lowest 0.059, at 600 K); excluding ORB-v2 at 600 K, dropping ceo2_cubic or nacl_rocksalt gives 0.047. Where one system's removal is enough to move p across 0.05, the value cannot carry a significance claim either way.

**Table S16** Screen-versus-SSCHA agreement on the bcc metals scored two ways, and the same-energy comparison behind the SSCHA false-stables on the displacive systems. *Curvature-sign agreement* compares the sign of the screen's symmetric-point free-energy curvature with the sign of the lowest SSCHA Hessian frequency. *Stability-call agreement* compares the two methods' calls. For SSCHA the call and the sign coincide on every row; for the screen the call is the variational argmin of §2.4, which can differ from the curvature sign. Both methods run on the same MLIP potential-energy surface, so agreement between them is a consistency check and says nothing about agreement with first principles. *Trivial* pairs are (system, model) combinations whose harmonic layer has no instability, so both methods agree without any thermal stabilisation having been tested. bcc pairs are Ti, Zr and Hf at 100, 300 and 600 K. `scripts/stats_hardening.py` (`bcc_agreement`, `criterion_blindness`).

| Subset | Curvature-sign agreement [95% CI] | Stability-call agreement [95% CI] |
|---|---|---|
| all pairs, all five models | 35/45 = 0.778 [0.637, 0.875] | 31/45 = 0.689 [0.543, 0.805] |
| all pairs, excluding ORB-v2 | 30/36 = 0.833 [0.681, 0.921] | 25/36 = 0.694 [0.531, 0.820] |
| trivial pairs (CHGNet and MACE-MP-0 on Zr and Hf) | 12/12 = 1.000 [0.758, 1.000] | 12/12 = 1.000 [0.758, 1.000] |
| non-trivial pairs, all five models | 23/33 = 0.697 [0.527, 0.826] | 19/33 = 0.576 [0.408, 0.728] |
| non-trivial pairs, excluding ORB-v2 | 18/24 = 0.750 [0.551, 0.880] | 13/24 = 0.542 [0.351, 0.721] |
| CHGNet | 9/9 = 1.000 [0.701, 1.000] | 7/9 = 0.778 [0.453, 0.937] |
| MACE-MP-0 | 9/9 = 1.000 [0.701, 1.000] | 9/9 = 1.000 [0.701, 1.000] |
| MatterSim | 3/9 = 0.333 [0.121, 0.646] | 0/9 = 0.000 [0.000, 0.299] |
| ORB-v2 | 5/9 = 0.556 [0.267, 0.811] | 6/9 = 0.667 [0.354, 0.879] |
| SevenNet-0 | 9/9 = 1.000 [0.701, 1.000] | 9/9 = 1.000 [0.701, 1.000] |

The 10 bcc units on which the two scores differ:

| System | Model | T (K) | Screen curvature (THz) | Screen call | SSCHA frequency (THz) | SSCHA call |
|---|---|---|---|---|---|---|
| hf_bcc | MatterSim | 600 | +0.276 | unstable | +1.683 | stable |
| hf_bcc | ORB-v2 | 100 | −0.092 | stable | +0.190 | stable |
| hf_bcc | ORB-v2 | 300 | −0.092 | stable | +0.093 | stable |
| hf_bcc | ORB-v2 | 600 | −0.092 | stable | +0.080 | stable |
| ti_bcc | CHGNet | 100 | +0.163 | unstable | +1.077 | stable |
| ti_bcc | CHGNet | 300 | +0.267 | unstable | +1.137 | stable |
| ti_bcc | MatterSim | 600 | +0.330 | unstable | +2.104 | stable |
| ti_bcc | ORB-v2 | 300 | +0.646 | unstable | +0.269 | stable |
| ti_bcc | ORB-v2 | 600 | +0.883 | unstable | +0.221 | stable |
| zr_bcc | MatterSim | 600 | +0.347 | unstable | +1.962 | stable |

The frequency magnitudes are given descriptively only, because the pairs cluster by system and model and no test is attached: Spearman ρ = +0.113 (n = 45), −0.003 without ORB-v2 (n = 36).

Lower part: on every non-bcc unit where SSCHA calls the phase stable against an unstable label, the screen's own symmetric-point curvature, the single-mode analogue of the SSCHA free-energy Hessian, is compared with the screen's call. Where the curvature is positive and the call is unstable, the screen's free energy has a local minimum at the symmetric point and a deeper one at a displaced centroid, so a criterion evaluated at the symmetric reference reports stable on the same energies.

| Quantity | All five models | Excluding ORB-v2 |
|---|---|---|
| non-bcc units SSCHA calls stable against an unstable label | 57 | 50 |
| screen's symmetric-point curvature positive | 52/57 = 0.912 [0.811, 0.962] | 47/50 = 0.940 [0.838, 0.979] |
| screen's free-energy comparison calls the phase unstable | 46/57 = 0.807 [0.687, 0.889] | 39/50 = 0.780 [0.648, 0.872] |
| both: curvature positive and call unstable | 41/57 = 0.719 [0.592, 0.819] | 36/50 = 0.720 [0.583, 0.825] |
|   of which batio3_cubic (n; curvature > 0; call unstable) | 10; 10; 6 | 8; 8; 4 |
|   of which cssni3_cubic (n; curvature > 0; call unstable) | 1; 0; 1 | 1; 0; 1 |
|   of which hfo2_cubic (n; curvature > 0; call unstable) | 16; 13; 16 | 14; 12; 14 |
|   of which knbo3_cubic (n; curvature > 0; call unstable) | 13; 13; 6 | 11; 11; 4 |
|   of which pbtio3_cubic (n; curvature > 0; call unstable) | 2; 2; 2 | 2; 2; 2 |
|   of which zro2_cubic (n; curvature > 0; call unstable) | 15; 14; 15 | 14; 14; 14 |
| all paired non-bcc units at T ≤ 300 K: SSCHA sign = screen curvature sign | 40/66 | 34/56 |
| all paired non-bcc units at T ≤ 300 K: SSCHA call = screen call | 21/66 | 20/56 |

**Table S17** Where SSCHA calls a phase unstable that its label calls stable, and where it fails numerically. Upper part: non-bcc SSCHA false-unstables by temperature, over the units whose label is stable at that temperature; numerical blow-ups (|f| > 50 THz) are included in the counts and also tallied separately. The count grows with temperature partly because more labels are stable at high temperature; the direct evidence is the per-unit trend below the table, in which units SSCHA calls stable at 100 K turn negative by 600 to 900 K. An instability that grows with thermal amplitude on a fixed potential-energy surface is the opposite of entropy stabilisation. It is what an MLIP extrapolating on large-amplitude thermal configurations would produce, and an instability of the stochastic sampling itself is the other candidate; these data do not separate the two. <!-- PENDING-C3b: PBE forces and energies on SSCHA-sampled configurations (SrTiO3 and the other high-T false-unstables) decide between MLIP extrapolation error and sampling instability; result goes in §3.3 and in a sentence here --> `scripts/stats_hardening.py` (`sscha_high_t`, `orb_split_s3`).

| T (K) | Model set | Non-bcc units returned | False-unstable / stable-labelled [95% CI] | Of which blow-ups | By system |
|---|---|---|---|---|---|
| 100 | all five | 33 | 0/0 (no stable label at this T) | 0 | -- |
| 100 | excluding ORB-v2 | 28 | 0/0 (no stable label at this T) | 0 | -- |
| 300 | all five | 33 | 5/5 = 1.000 [0.566, 1.000] | 1 | srtio3_cubic 5 |
| 300 | excluding ORB-v2 | 28 | 4/4 = 1.000 [0.510, 1.000] | 0 | srtio3_cubic 4 |
| 600 | all five | 31 | 10/13 = 0.769 [0.497, 0.918] | 2 | batio3_cubic 2, cssni3_cubic 3, srtio3_cubic 5 |
| 600 | excluding ORB-v2 | 26 | 8/11 = 0.727 [0.434, 0.903] | 1 | batio3_cubic 1, cssni3_cubic 3, srtio3_cubic 4 |
| 900 | all five | 29 | 15/19 = 0.789 [0.567, 0.915] | 4 | batio3_cubic 4, cssni3_cubic 3, knbo3_cubic 2, pbtio3_cubic 2, srtio3_cubic 4 |
| 900 | excluding ORB-v2 | 24 | 12/16 = 0.750 [0.505, 0.898] | 3 | batio3_cubic 3, cssni3_cubic 3, knbo3_cubic 1, pbtio3_cubic 2, srtio3_cubic 3 |

Of the 23 non-bcc units SSCHA calls stable at 100 K, 14/23 = 0.609 [0.408, 0.778] turn negative by 600 to 900 K (without ORB-v2 10/19 = 0.526 [0.317, 0.727]): batio3_cubic (CHGNet, MACE-MP-0, ORB-v2, SevenNet-0); cssni3_cubic (MACE-MP-0); hfo2_cubic (MACE-MP-0, ORB-v2); knbo3_cubic (CHGNet, ORB-v2); pbtio3_cubic (CHGNet, SevenNet-0); zro2_cubic (MACE-MP-0, ORB-v2, SevenNet-0). For 9 of the 14 the label is stable at that temperature, so the negative value is a false-unstable; the rest are hfo2_cubic and zro2_cubic units, labelled unstable at every temperature, where the sign is right but the trend with temperature runs the wrong way.

SrTiO₃ above its 105 K transition: SSCHA calls every returned unit unstable (14/14 = 1.000 [0.785, 1.000]; without ORB-v2 11/11 = 1.000 [0.741, 1.000]), at −3.4 to −914 THz (−3.4 to −60.5 THz without ORB-v2), each more negative than the same model's harmonic minimum. SSCHA never calls SrTiO₃ stable at any temperature, so a missing zone-boundary q-point cannot be what produces a false-stable here. ORB-v2 has no imaginary commensurate mode in SrTiO₃, so the screen condenses nothing and the mode listed for it is its softest commensurate mode, which is harmonically stable (+1.35 THz).

| Model | T (K) | SSCHA min frequency (THz) | Blow-up | Harmonic min frequency (THz) | Screen deciding mode q | Its harmonic frequency (THz) |
|---|---|---|---|---|---|---|
| CHGNet | 300 | −20.5 | no | −3.45 | (0.5, 0.5, 0.5) | −0.82 |
| MACE-MP-0 | 300 | −3.4 | no | −2.37 | (0.5, 0.5, 0.5) | −2.09 |
| MatterSim | 300 | −14.9 | no | −3.36 | (0.5, 0.5, 0.5) | −1.86 |
| ORB-v2 | 300 | −68.0 | yes | +0.00 | (0, 0.5, 0) | +1.35 |
| SevenNet-0 | 300 | −7.9 | no | −3.24 | (0.5, 0.5, 0.5) | −1.95 |
| CHGNet | 600 | −36.6 | no | −3.45 | (0.5, 0.5, 0.5) | −0.82 |
| MACE-MP-0 | 600 | −20.2 | no | −2.37 | (0.5, 0.5, 0.5) | −2.09 |
| MatterSim | 600 | −22.2 | no | −3.36 | (0.5, 0.5, 0.5) | −1.86 |
| ORB-v2 | 600 | −544.5 | yes | +0.00 | (0, 0.5, 0) | +1.35 |
| SevenNet-0 | 600 | −30.5 | no | −3.24 | (0.5, 0.5, 0.5) | −1.95 |
| CHGNet | 900 | −60.5 | yes | −3.45 | (0.5, 0.5, 0.5) | −0.82 |
| MACE-MP-0 | 900 | −27.1 | no | −2.37 | (0.5, 0.5, 0.5) | −2.09 |
| MatterSim | 900 | −35.5 | no | −3.36 | (0.5, 0.5, 0.5) | −1.86 |
| ORB-v2 | 900 | −913.7 | yes | +0.00 | (0, 0.5, 0) | +1.35 |

Blow-ups and failed units by model. A failed unit is one of the 208-unit attempted grid that returned no number: it stopped at a cellconstructor symmetry or ensemble assertion. The blow-ups spread over four models and the failures over three. As run, only MACE-MP-0 returns float64 forces, and it has neither; it is also a different architecture, and one model is no basis for attributing the failures to numerical precision.

| Model | Grid units | Returned | Failed | Failed units | Blow-ups | Blow-up units |
|---|---|---|---|---|---|---|
| CHGNet | 43 | 43 | 0 | -- | 1 | srtio3_cubic 900 K (−60.5 THz) |
| MACE-MP-0 | 42 | 42 | 0 | -- | 0 | -- |
| MatterSim | 43 | 41 | 2 | pbtio3_cubic 600 K, pbtio3_cubic 900 K | 3 | pbtio3_cubic 300 K (−801.3 THz), cssni3_cubic 600 K (−63.7 THz), cssni3_cubic 900 K (−108.6 THz) |
| ORB-v2 | 39 | 35 | 4 | pbtio3_cubic 100 K, pbtio3_cubic 300 K, pbtio3_cubic 600 K, pbtio3_cubic 900 K | 3 | srtio3_cubic 300 K (−68.0 THz), srtio3_cubic 600 K (−544.5 THz), srtio3_cubic 900 K (−913.7 THz) |
| SevenNet-0 | 41 | 40 | 1 | cssni3_cubic 600 K | 1 | pbtio3_cubic 900 K (−66.8 THz) |
| *total, all five* | 208 | 201 | 7 |  | 8 |  |
| *total, excluding ORB-v2* | 169 | 166 | 3 |  | 5 |  |

<!-- END GENERATED TABLES -->

## S4. Threats to validity (pre-registered, with outcomes)

- **Finite-displacement noise near Γ** → acoustic-sum-rule handling + swept imaginary tolerance
  (default −0.1 THz). *Outcome: the calls are stable over roughly 0.05 to 0.1 THz (pooled over
  models, 5 false-stable and 3 false-unstable of 95 calls at both 0.05 and 0.1 THz; 4 and 2 of 76
  without ORB-v2; Table S9). Beyond that CHGNet's CeO₂ and NaCl calls flip at about 0.25 THz,
  and at tol = 0 the calls are degenerate: every model collapses, with 29 of the 95 calls
  false-unstable. The displacement-amplitude axis is measured separately for all five models
  (Table S13): MACE-MP-0, MatterSim and SevenNet-0 change no call, CHGNet changes calls only on
  controls, and ORB-v2 changes calls on test systems.*
- **Single-mode vs multi-mode** → SSCHA cross-check. *Outcome: both methods run on the same MLIP
  surface, so their agreement is a consistency check. On bcc the curvature signs agree in 35/45
  paired units (0.78 [0.64, 0.87]; 30/36 without ORB-v2) and the stability calls in 31/45
  (0.69 [0.54, 0.80]; 25/36 without ORB-v2); 12 of the agreeing units have no harmonic
  instability, and MatterSim agrees on the call in 0/9 (Table S16). The frequency magnitudes
  give Spearman ρ = 0.11 (−0.003 without ORB-v2), reported descriptively with no test. On the
  displacive systems the two diverge because they answer different questions: the default SSCHA
  criterion is the curvature of the free energy at the symmetric reference, a local test, and
  it cannot see condensation into a deeper displaced minimum, which the screen's free-energy
  comparison does see. On the same MLIP energies the screen's own symmetric-point curvature is
  positive on 52/57 of the units SSCHA false-stabilises (Table S16; §3.3). Whether an MLIP error
  on the thermally sampled configurations also contributes is not settled by these data.*
  <!-- PENDING-C3b: PBE forces/energies on SSCHA-sampled configurations; outcome sentence here -->
- **Single-mode approximation itself** → derivation and sensitivity analysis (§S1.3, Table S14).
  *Outcome: varying the fit window, sampling range, condensation threshold and scan range changes
  at most one unit call at T ≤ 300 K, but the frozen-cell normalisation is outcome-determining
  (ferroelectric recall 5/30 to 28/30 across conventions), and mode–mode coupling is unbounded. The screen's T* and
  ferroelectric recall are conditional on the minimal-cell convention.*
  <!-- PENDING-C3a: PBE-backed screen calls; outcome sentence here -->
- **MLIP relaxation moving off the soft-mode geometry** → both at-reference and at-relaxed
  geometries recorded; relaxation hiding an instability is itself reported.
- **Supercell / cell-size convergence** → SSCHA at 2×2×2. *Outcome: the only cell-size test, bcc
  Zr 2×2×2 against 3×3×3, holds, but it was run on MACE-MP-0, which has no harmonic bcc-Zr
  instability, and in the v1 environment. No convergence claim is made for the zone-boundary
  (R-point and X-point) systems; the within-cell fluorite control rules out a missing q-point there
  and nothing more (§S2.4). For the screen, the q-points searched are those commensurate with
  the force-constant supercell (2×2×2, and 6×6×6 for the bcc metals), and the screen was not
  re-run with larger force-constant supercells, so an instability at a q outside that set would
  be missed (approximation A2 of Table 1); the common force-constant supercell row of Table S14
  changes the frozen-cell normalisation, not the set of q searched. Cross-model comparisons are
  made at a fixed cell; absolute stabilisation temperatures are approximate.*
  <!-- PENDING-C5: re-measured 2x2x2 vs 3x3x3 on bcc-Zr (MatterSim, MACE-MP-0); outcome here -->
- **Ground-truth uncertainty** → transition temperatures approximate; scoring qualitative
  (correct side of the transition). The labels of the test systems record thermodynamic, often
  first-order, transitions, and the control labels come from published harmonic phonon
  stability, whereas the screen and SSCHA probe dynamic stability. The two coincide for a
  continuous soft-mode transition and can differ for a deep double well, which is where the
  local SSCHA criterion and the labels part company (§3.3).
