# Electronic Supplementary Information (ESI)

*Supplementary information for* "Neither harmonic benchmarks nor a default SSCHA cross-check
certifies a foundation machine-learning interatomic potential for finite-temperature dynamic
stability", F. Cai, *RSC Advances*. Section numbers without an S (§2.4, §3.3 and so on) and
numbered references refer to the main article. Tables S4-S11 and S13-S23 are generated from the
deposited ledger (`results/ledger.parquet`) and the deposited analysis outputs
(`results/stats_hardening.json`, `results/screen_sensitivity.json`,
`results/curvature_identity_check.json`, `results/revision/force_spread/summary.json`,
`results/revision/dft/`, `results/revision/dft_checks/`, `results/revision/sscha_seeds/`,
`results/revision/sscha_converged_grid/`, `results/revision/grid_compare.json`) by
`scripts/build_esi_tables.py`; figures by `scripts/make_figures.py`.

*Citation convention: **sections** of this document are cited as §S1-§S5 and **tables** as
Table S1-Table S23. The two sequences are independent; a cross-reference to "Table S1" means the
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
set cannot choose among these conventions (Table S14 notes). The H2 transfer count moves with the
convention too. At 300 K b v c is 17 v 4 in the minimal cell, 7 v 4 in the common force-constant
supercell, 10 v 4 in the doubled cell, 11 v 4 in the eight-fold cell and 24 v 4 per formula unit,
with system-clustered p from 0.06 to 0.89; at 100 K b does not exceed c in the common supercell
(3 v 3), the doubled cell (3 v 3) or the eight-fold cell (2 v 3). Over the 20 convention and
temperature combinations the clustered p runs from 0.03 to 1.0, and two fall below 0.05 (per
formula unit at 100 K, 0.045; the eight-fold cell at 900 K, 0.031), which after twenty looks we do
not read as evidence (`scripts/h2_by_convention.py` → `results/h2_by_convention.json`). The
screen's T*, its ferroelectric recall and the size of the H2 count are therefore conditional on
the minimal-cell convention; we state that as a limitation of the single-mode screen, not as a
setting we could defend on the data.

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
step 0.005 Å; because 𝓕 is even in Q₀ the one-sided stencil is central. At the symmetric point
that curvature equals the self-consistent trial stiffness (shown after the stationarity step
below), which limits what it can report. Table S14 varies the fit
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

**What the symmetric-point curvature is.** Write F(Q₀) for 𝓕 at its stationary Ω. Because Ω is
stationary, dF/dQ₀ = ∂𝓕/∂Q₀ = ⟨V′⟩_{Q₀,σ}. Differentiating again, F″ = ⟨V″⟩ +
(∂⟨V′⟩/∂σ²)(dσ²/dQ₀). At Q₀ = 0 the second term vanishes for an even V, since ⟨V′⟩ is then zero
for every width and σ² is even in Q₀, so F″(0) = ⟨V″⟩_{0,σ} = MΩ²: the curvature of the screen's
free energy at the symmetric point is the self-consistent trial stiffness there. It is positive
whenever a bound Gaussian solves the width equation at Q₀ = 0, however deep the well, so it cannot
register condensation; only the comparison of 𝓕 across centroids, the screen's call, can. This is
the single-mode counterpart of the statement that the SSCHA auxiliary matrix is positive definite
by construction. In the multi-mode SSCHA Hessian the bubble term, built from third-order couplings
between the soft mode and other modes, need not vanish at the symmetric point, so the SSCHA
Hessian can be negative there where the single-mode curvature cannot. Numerically, over the 1512
mode-temperature evaluations of the production maps the reported ω_eff agrees with Ω/2π at
Q₀ = 0 to a median relative difference of 7 × 10⁻⁷, and every negative ω_eff (95 evaluations) is
numerical: 93 have the width solver's grid fallback (below) at a point of the finite-difference
stencil, and the other two are ORB-v2 modes of bcc Ti whose fitted polynomial has a positive
quadratic term and a well only between sample points, a fit artefact of the kind described in
§S1.4 (`scripts/curvature_identity_check.py` → `results/curvature_identity_check.json`). The
negative ω_eff values are therefore not physical curvatures, and the observable is not scored
against SSCHA anywhere in this work (Table S16).

**Solving it.** Substituting Ω(σ²) back gives a scalar fixed-point equation g(σ²) = σ²_sc(σ²) −
σ² = 0, solved in σ² rather than Ω because the physical branch is easier to bracket there. A
damped iteration on this equation can run away to a large-σ spurious root, so the implementation
brackets instead. The search grid is geometric, σ² ∈ [10⁻⁶, 2] Å² over 240 points, restricted to
the sub-interval on which ⟨V″⟩(σ²) > 10⁻⁸ eV Å⁻² (below that no normalisable Gaussian trial
state exists, since Ω would be imaginary); the first downward sign change of g on that grid is
refined by Brent's method to a tolerance of 10⁻¹⁰. If the grid has no sign change, or Brent's
method fails, the solver keeps the grid point at which |g| is smallest. That fallback fires on
8441 of 182,952 centroid evaluations (4.6%) and on none of the 1512 evaluations that decide a
call, the minimising centroid of each mode and temperature (Table S14). It does reach the
reported curvature, whose stencil uses Q₀ = 0, 0.005 and 0.010 Å: that is the source of 93 of the
95 negative ω_eff values above. If the admissible
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

Inside a degenerate eigenspace the screened direction is also not unique. The pattern is the
eigenvector the eigensolver returns, and any direction in the degenerate span is equally an
eigenvector, but the one-dimensional problem along it is not the same: regenerating the Γ
triplet of BaTiO₃ under MACE-MP-0 for the first-principles comparison gave a direction whose well
is 16.8 meV deep with M = 41 amu, against 20.9 meV and 53 amu along the production direction
(`scripts/dft_reference.py`, which therefore searches the degenerate span for the production
direction before comparing). The screen's call on a degenerate mode is thus a call along one
arbitrary direction of the span, and the anisotropy of the well within the span is not explored.

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
That test was run (§S5, Table S19): the shared BaTiO₃ and KNbO₃ mis-calls at 300 K disappear on
PBE energies along the same coordinates, where the MLIP wells are 0.32–0.76 of the PBE depth, so
they are surface errors that four models share; the CsSnBr₃ mis-calls and KNbO₃ at 600 K persist.
With PBEsol on the same structures most CsSnBr₃ calls turn correct and KNbO₃ at 600 K does not
(§S5.3), so the CsSnBr₃ errors follow the functional and KNbO₃ at 600 K is the one left for the
screen or the label.

The multi-mode SSCHA retains the couplings, but here it cannot bound their effect on the call,
for three reasons. It runs on the same MLIP potential, so any PES error is common to both
methods. On the displacive systems its default criterion is local (§3.3, Table S16), so it
answers a different question from the screen's free-energy comparison. And on bcc, where both
apply, the agreement is weaker than a bound would need: the stability calls agree in 31/45
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
That profile, for six systems, and the screen re-run on it are in §S5 (Table S19).

## S2. SSCHA harness and the displacive-instability failure

### S2.1 Minimizer step cap

Multi-mode SSCHA on the 40-atom perovskite cells initially appeared to hang (CPU busy, GPU idle,
a step counter climbing past 1000): with a noisy stochastic gradient on a large soft cell, the
minimiser chased the gradient below its own stochastic-noise floor without stopping. A step cap
(`minim.max_ka = 20`; the python-sscha default, −1, is no cap) ended that: with it the SrTiO₃ run
finished in 231 s after 27 steps, against 1835 steps and a timeout without it. The cap was
introduced on the understanding that it limits the steps within each population. It does not.
python-sscha 1.6.1 keeps the minimiser's step history across populations and compares `max_ka`
with the accumulated count (`SchaMinimizer.py`, line 1370 of the 1.6.1 source), so the cap bounds
the total number of steps over the whole relaxation. The relaxation otherwise ends when
python-sscha's convergence test, with threshold `meaningful_factor = 1e-4`, is met, or when the
population limit (8) is reached. Which of these ended each production unit was not recorded
(Table S11).
The seed study records it (Table S21): on all 16 seeds of the four units the convergence test was
not met, the cap ended population 1 after 19 kept steps, and populations 2 to 8 each took one
step and discarded it, changing the auxiliary matrix by less than 2 × 10⁻¹⁴ (relative); the final
dynamical-matrix gradient sat 7.3 × 10² to 1.0 × 10⁴ times above the threshold, having fallen by
a factor of only 1.1 to 2.8 from the first step. The threshold is itself built on a placeholder:
python-sscha 1.6.1 passes a constant in place of the stochastic error of the gradient
(`Ensemble.py`, line 2657), recorded as 0.433 per primitive-cell atom at every step of every run.
In the two production-recipe runs of the bcc-Zr cell-size check where MACE-MP-0's harmonic
matrix is already positive (3×3×3, 100 and 300 K) the start is the harmonic matrix and one
population of 17 and 20 steps meets the test.
The converged recipe (Table S22) replaces both: the minimiser is capped at 400 steps within each
population, the gradient error is the library's serial stochastic estimate (set to zero where
KL/N < 0.7, so convergence cannot be declared at a statistically invalid point) with
`meaningful_factor` = 0.2, populations hold 1,000 configurations up to 30, and the Hessian ensemble
has 2,000. With it, 159 of the 178 grid units meet the stopping test; 11 ORB-v2 units reach the
population cap, one CHGNet unit the wall cap, and 7 fail at the same assertions as in production.
An independent check on a fresh 2,000-configuration ensemble at the final matrix finds the
gradient consistent with a minimum on 152 of the 159 (the 7 that fail are listed in Table S22;
none carries a false-stable).

The stopping behaviour matters for how the SSCHA numbers are read. A relaxation that stops early
leaves the auxiliary matrix near its `ForcePositiveDefinite` start, in which imaginary harmonic
modes have been made real, and a free-energy Hessian evaluated there is not evaluated at the SCHA
minimum.

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
| SCHA auxiliary dynamical matrix at the end of the relaxation | +2.892 |
| free-energy Hessian, bubble level, `include_v4=False` (production) | +2.872 |
| free-energy Hessian, `include_v4=True` (re-run on the seed-0 production-recipe ensemble of Table S21) | +2.878 (bubble level on the same ensemble +2.878; 9,850 s against 11 s) |

What the table shows is limited, and it is worth saying exactly what. The auxiliary SCHA matrix is
positive definite by construction (a normalisable Gaussian trial state requires it), so the
+2.88 THz after `ForcePositiveDefinite` and the +2.89 THz at the end of the relaxation say
nothing about stability: the auxiliary matrix cannot soften (refs 21, 22). The two differ by
0.01 THz, so the relaxation left the matrix close to its positive-definite start (§S2.1). The object that could detect the
instability is the free-energy Hessian, and at bubble level it lands within 0.02 THz of the
positive auxiliary curvature, so it reports the symmetric phase stable against a harmonic
instability of −5.6 THz on the same MLIP. The `include_v4=True` row differs from the bubble-level
value on the same ensemble in truncation order alone and changes the lowest frequency by
1 × 10⁻⁴ THz, so on this unit the truncation is not the cause. (That row was evaluated on the
seed-0 ensemble of the production-recipe seed study, not on the June run of the other rows.)

The account the rest of the paper uses does not rest on this table. The free-energy Hessian is
the curvature of the SSCHA free energy at the high-symmetry reference, so it answers a local
question: is the symmetric phase a local minimum of the free energy? For a deep double well the
free energy can keep a local minimum at the symmetric point while a displaced minimum lies lower
(the first-order-like case of §2.4), and a curvature at the reference then reports stable at any
order of the expansion. On the same MLIP energies the screen finds that situation on half of the
34 non-bcc units converged SSCHA false-stabilises: its free-energy comparison finds a displaced
minimum below the symmetric point on 17 of them (Tables S16 and S22), including this one (57 and
46 with the production recipe). The screen's own
symmetric-point curvature cannot add to that, because for a single mode it is positive by
construction (§S1.3), and neither shows that SSCHA's positive Hessian has this origin rather than
a relaxation that stopped near its start (§S2.1); the converged recipe settles that for this
unit, which stays stable at +2.03 THz after a converged relaxation (Table S22). Adding
the fourth-order term changes the estimate of the curvature but not the question it answers. It
could flip the sign only where the bubble-level value overstates a curvature that is in fact
negative. Evaluated on the same seed-0 ensemble, the fourth-order Hessian gives +2.8784 THz
against +2.8783 THz at bubble level (9,850 s against 11 s, 24 threads): it does not flip the sign
on this unit. The ensemble is the production one, at an unconverged auxiliary matrix; the
converged value for this unit is +2.03 THz at bubble level (Table S22), and the fourth-order term
was not evaluated there.
Whether an MLIP error on the thermally sampled configurations contributes as well is a separate
question, which PBE forces on those configurations address (§S5, Table S20): for BaTiO₃ and
ZrO₂ at 100 K the MLIP's relative force error is 0.10 and 0.18 there against 0.10 and 0.23 near
equilibrium, so they give no sign of it.

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

Cubic fluorites are numerically clean (no blow-ups, no failed units). With the production recipe
SSCHA nonetheless called every one of the ten fluorite units stable at 100 K, at +1.9 to +3.3 THz
in the 2×2×2 cell, although cubic ZrO₂ and HfO₂ are the high-temperature phases (above about
2570 K and 2800 K) and the label is unstable, and it false-stabilised 31 fluorite units over the
ladder, all of which the screen's free-energy comparison calls unstable (Table S16). Three of the
five models then destabilised with temperature (Table S17). None of this survives convergence:
with the converged recipe SSCHA calls every fluorite unit unstable, as labelled (38 converged
units; the 2 ORB-v2 units that did not converge are unstable too; Table S22), so the production
fluorite false-stables were a relaxation that stopped near its positive-definite start (§S2.1),
not the local criterion.

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
    The revision adds four units with four seeds each, run with the production recipe and full
  diagnostics (Table S21). Every seed gives the same call. The seed spread of the lowest Hessian
  frequency is 0.007 THz for BaTiO₃/MACE-MP-0 at 100 K (+2.867 to +2.881), 0.009 THz for
  ZrO₂/MACE-MP-0 at 100 K (+3.069 to +3.087) and 0.075 THz for bcc-Zr/MatterSim at 50 K (+1.636 to
  +1.786), all within 0.24 THz of the positive-definite start, against 2.0 THz for
  SrTiO₃/MACE-MP-0 at 600 K (−15.8 to −20.3), where seed 0 gives −18.7 against the ledger's −20.2
  and the ledger value lies inside the seed range. The bootstrap over the Hessian ensemble gives a
  comparable uncertainty for BaTiO₃, ZrO₂ and SrTiO₃ (median 0.003, 0.009 and 3.0 THz) and none for
  bcc Zr, whose lowest Hessian frequency is the final auxiliary matrix's own; there the seed spread
  comes from the unconverged relaxation alone. None of the 16 seeds converged (§S2.1). Seed
  agreement is not convergence: the converged recipe changes two of these four calls (ZrO₂ at
  100 K, −22.4 THz; SrTiO₃ at 600 K, +2.44 THz) and moves the other two (BaTiO₃ +2.03 THz;
  bcc-Zr/MatterSim at 50 K +0.41 THz), Table S22. The converged recipe's own start dependence was
  measured on six units from two starts: all six keep their call, and SrTiO₃/MACE-MP-0 at 100 K
  (+1.13 against +1.02 THz) is flagged start-dependent because the difference exceeds its
  bootstrap error.
  <!-- PENDING-E3: pre-registered replicates of the converged grid (start B and a second seed on 65 units) started 2026-10-08; add their call agreement and spread here and in §3.5, or state that they were not run. -->
- **Finite size, bcc.** For bcc Zr the 2×2×2 and 3×3×3 cells are not nested: the N point is
  commensurate only with even cells and the ω point, q = ⅔⟨111⟩, only with multiples of three, so
  a comparison between them changes the set of q-points as well as the size, and it is reported
  as such rather than as a convergence test. With MACE-MP-0, which has no harmonic instability in
  bcc Zr, the call holds (+1.80 THz in 2×2×2, +1.56 and +1.55 THz in 3×3×3 at 100 and 300 K;
  re-measured in the pinned environment, Table S21 lower part, +1.562 and +1.547 THz, converged by
  the library's test). With MatterSim, which has one, it does not: converged from two starts, Zr at
  300 K is +0.92 THz in 2×2×2 and −0.92 THz in 3×3×3, consistent with the ω mode entering the
  larger cell; the production recipe gives the same sign change (+1.95 to −2.11 THz) without
  converging. The bcc SSCHA calls are therefore cell-dependent for MatterSim, and the converged
  grid's 3×3×3 bcc values (Table S22) are not a converged version of the 2×2×2 ones. A nested
  4×4×4 and 6×6×6 test was not run. The v1-generation rows of `results/convergence_study.parquet`
  and its soft-mode rows (produced by the earlier single-mode selection, before the correction
  described in §2.4) are retained for audit only.
- **Zone-boundary systems: no convergence claim.** The SrTiO₃ antiferrodistortive instability is
  at the R point (½,½,½) and the fluorite instability at the X point; both are commensurate with
  even supercells only, so neither has a cell-size comparison in this work, and the even-cell test
  (4×4×4, a ~320-atom SSCHA) was not run. We make no supercell-convergence claim for the R-point
  or X-point systems, and their SSCHA calls are stated for the 2×2×2 cell only.
- **SrTiO₃.** Converged SSCHA calls SrTiO₃ stable at 100 K, 5 K below its transition, in every
  model where it converged (+1.06 to +1.20 THz), and stable at every higher temperature (Table
  S22). The cell contains the R point, so this is not a missing q-point, but whether the
  finite-temperature renormalisation of the R mode is converged in 2×2×2 is untested, and that
  call may be a finite-size effect. The start-A/start-B pair at 100 K agrees in sign (+1.13 and
  +1.02 THz) and differs by 0.11 THz, more than its bootstrap standard deviation (0.03 THz), so it
  is flagged start-dependent. The production recipe never called SrTiO₃ stable: it was
  false-unstable in all 14 returned units above the transition, at −3.4 to −914 THz (Table S17),
  but those relaxations had not converged (Table S21), and PBE forces on twelve configurations of
  the production MACE-MP-0 600 K ensemble (0.34 Å root-mean-square displacement; §S5, Table S20),
  which put MACE-MP-0's relative force error at 0.19 against 0.11 near equilibrium, describe
  configurations the converged relaxation does not reach.
- **Fluorites.** Converged SSCHA calls every fluorite unit unstable in the 2×2×2 cell, as the
  labels and the screen do, so no result rests on a fluorite SSCHA false-stable; the production
  recipe's 100 K false-stables (+1.9 to +3.3 THz, against harmonic −3.8 to −10.6 THz in the same
  cell) were the unconverged relaxation. No claim is made that the 2×2×2 frequencies are converged.
- **BaTiO₃.** Its ferroelectric instability includes the zone-centre (Γ) mode, which every
  supercell contains, so a missing q-point cannot produce its false-stable. This rules out one
  explanation for one system and is not a convergence test.

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
- Table S16, screen-versus-SSCHA agreement on bcc scored on the call, and the same-energy
  comparison behind the SSCHA false-stables.
- Table S17, SSCHA high-temperature false-unstables, the SrTiO₃ units above its transition, and
  SSCHA blow-ups and failures per model.
- Table S18, the pre-registered force-level ensemble test, every score it names.
- Table S19, the screen's calls on PBE energies along the same coordinates, and the MLIP against
  PBE well depths path by path (§S5).
- Table S20, MLIP force and energy errors against PBE on SSCHA-sampled configurations (§S5).
- Table S21, the production-recipe SSCHA seed study and the bcc-Zr cell size at the production
  recipe (§S2.1, §S2.4).
- Table S22, the converged-recipe SSCHA grid beside the production numbers on the same units
  (§S2.1, §3.3).
- Table S23, checks on the errors that persist on PBE: k-point and cutoff convergence, PBEsol on
  the same structures, and the PBE lattice (§S5.3).

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

**Table S9** Every pooled rate that ORB-v2 could plausibly drive, with and without it. ORB-v2 is the only one of the five models whose forces are predicted directly rather than as the gradient of an energy (non-conservative). As run, CHGNet, SevenNet-0, MatterSim and ORB-v2 all return float32 forces and only MACE-MP-0 returns float64, so precision does not single ORB-v2 out, and the design cannot separate its architecture from anything else about it. For reference: ORB-v2 is the only model that calls MgO harmonically unstable (−1.07 THz); its most negative bcc screen curvatures are −6.2 THz (Ti, 100 K) and −8.9 THz (Ti, 900 K), both on fitted polynomials whose quadratic term is positive (fit artefacts of the kind described in §S1.4), with Hf at −0.09 THz, the softest commensurate harmonic value of a model with no screened imaginary mode there; and it accounts for 3 of the 8 SSCHA blow-ups and 4 of the 7 failed SSCHA units (all on pbtio3_cubic). Per-model rates (Tables S4 and S5) cannot change when another model is removed, so only pooled quantities are split here. The guardrail is where removing ORB-v2 changes the reading: without it the vote-split AUC falls from 0.762 to 0.628 and its cluster-bootstrap interval spans 0.5, so the guardrail is suggestive and not robust to removing ORB-v2. The frequency spread is the cross-model standard deviation of the screen's effective frequency, a potential-energy-surface proxy that is only partly force-derived (the mode patterns come from force constants; E(Q) is energy-only); it is not a force-level ensemble uncertainty; the force-level spread, pre-registered and given with and without ORB-v2, is in Table S18. Four-model consensus uses the same rule as five-model consensus (stable when at least half the votes are stable), so a 2–2 split is called stable; this decides 3 units, 2 of them wrong. Removing ORB-v2 leaves the bcc stability-call agreement at 0.69. Permutation p-values and bootstrap intervals resample whole systems (10,000 permutations and 10,000 bootstrap draws, seed 0; the 9,998 bootstrap draws that contain both outcomes are used), so each p carries Monte Carlo error. The H2 ladder without ORB-v2 is in Table S15, the paired screen-versus-SSCHA tests in Table S10, the bcc agreement by model in Table S16, and the SSCHA high-temperature false-unstables, blow-ups and failures by model in Table S17. The SSCHA rows here are the production recipe, which did not converge (§S2.1); the converged-recipe values, with and without ORB-v2, are in Table S22. `scripts/stats_hardening.py` (`orb_split_s3`, `bcc_agreement`); pooled finite-T accuracy from `scripts/screen_sensitivity.py`.

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
| bcc screen-vs-SSCHA stability-call agreement | 31/45 = 0.689 [0.543, 0.805] | 25/36 = 0.694 [0.531, 0.820] |
| bcc frequency Spearman ρ (descriptive, no test) | +0.113 | −0.003 |
| Guardrail: consensus error on split-vote units | 8/14 = 0.571 [0.326, 0.786] | 4/8 = 0.500 [0.215, 0.785] |
| Guardrail: consensus error on unanimous units | 4/46 = 0.087 [0.034, 0.203] | 8/52 = 0.154 [0.080, 0.275] |
| Guardrail: vote-split AUC [cluster-bootstrap 95% CI], clustered permutation p | 0.762 [0.590, 0.934], clustered p = 0.0035 | 0.628 [0.438, 0.839], clustered p = 0.0469 |
| Guardrail: frequency-spread AUC [cluster-bootstrap 95% CI], clustered permutation p | 0.361 [0.046, 0.625], clustered p = 0.2748 | 0.299 [0.076, 0.516], clustered p = 0.1030 |
| Guardrail: 2–2 tied units (broken to 'stable'), of which wrong | 0, 0 wrong | 3, 2 wrong |

**Table S10** The central screen-versus-SSCHA contrast, tested as the paired comparison it is. Both methods see the same (system, model, temperature) units, so comparing their two marginal Wilson intervals would ignore the pairing. Two p-values are given and **the unit-level one should not be quoted**: an exact McNemar over discordant units also assumes those units are independent, and they cluster by system. The clustered column randomises the method label over whole systems and is exact at this size. Its resolution floor, 2/2^k, is given alongside, because with five displacive systems no arrangement of the data can reach p < 0.0625 and with two fluorite systems none can go below 0.5. The reportable content of this table is the size and the consistency of the effect (4 of 5 systems favour the screen, with per-system net discordances +6, +6, −1, +9, +10, in the order BaTiO₃, KNbO₃, PbTiO₃, ZrO₂, HfO₂ where all five are present) rather than a significance claim. The *production* rows use the production-recipe SSCHA, which did not converge (§S2.1). The *converged* rows repeat the test with the converged-recipe SSCHA values of Table S22 on the units where that recipe converged (`results/revision/grid_compare.json`, block v): there 2 of 5 systems favour the screen (per-system net discordances +5, +6, 0, 0, 0), the fluorites show no discordance at all, and the clustered p is 0.5. Why SSCHA loses these units is examined separately, on the same MLIP energies, in Table S16 (lower part), Table S22 and §3.3. `scripts/stats_hardening.py`, `scripts/grid_compare.py`.

| SSCHA recipe | System set | Model set | n paired | Discordant (screen/SSCHA) | Unit-level p (do not quote) | Systems favouring screen | Clustered p | Floor |
|---|---|---|---|---|---|---|---|---|
| production | fe oxide | all models | 27 | 14/3 | 0.013 | 2/3 | 0.5000 | 0.2500 |
| production | fe oxide | excl orb v2 | 23 | 10/3 | 0.092 | 2/3 | 0.5000 | 0.2500 |
| production | fluorite | all models | 20 | 19/0 | 3.8e-06 | 2/2 | 0.5000 | 0.5000 |
| production | fluorite | excl orb v2 | 16 | 16/0 | 3e-05 | 2/2 | 0.5000 | 0.5000 |
| production | displacive combined | all models | 47 | 33/3 | 2e-07 | 4/5 | 0.1250 | 0.0625 |
| production | displacive combined | excl orb v2 | 39 | 26/3 | 1.5e-05 | 4/5 | 0.1250 | 0.0625 |
| converged | fe oxide | all models | 26 | 13/2 | 0.0074 | 2/3 | 0.5000 | 0.2500 |
| converged | fluorite | all models | 18 | 0/0 | 1 | 0/2 | 1.0000 | 0.5000 |
| converged | displacive combined | all models | 44 | 13/2 | 0.0074 | 2/5 | 0.5000 | 0.0625 |
| converged | fe oxide | excl orb v2 | 23 | 10/2 | 0.039 | 2/3 | 0.5000 | 0.2500 |
| converged | fluorite | excl orb v2 | 16 | 0/0 | 1 | 0/2 | 1.0000 | 0.5000 |
| converged | displacive combined | excl orb v2 | 39 | 10/2 | 0.039 | 2/5 | 0.5000 | 0.0625 |

**Table S11** SSCHA numerical quality, aggregated per family and model (not per unit). The SSCHA settings are identical for all 201 returned units and are therefore stated once here rather than tabulated. The starting dynamical matrix is built from phonopy full force constants at a 0.03 Å finite displacement, made positive definite (`ForcePositiveDefinite`) and symmetrised. The auxiliary dynamical matrix is relaxed in the root2 representation with `min_step_dyn = 0.5`; the convergence threshold is `meaningful_factor = 1e-4`. The minimiser's steps are capped at `max_ka = 20`, a cap that python-sscha 1.6.1 applies to the step count accumulated over all populations, not to each population (§S2.1); populations hold 256 configurations and at most 8 are drawn (2048 configurations). The free-energy Hessian is then evaluated at bubble level (`include_v4 = False`) on a dedicated 512-configuration ensemble at the final auxiliary matrix. The harness did not record how many populations each unit used or whether it met the convergence threshold before the population cap, so 2560 configurations is the configured maximum per unit, not a count. The SSCHA here inherits the MLIP potential-energy surface; nothing in this table tests that surface.

Two quantities are tabulated from the six lowest recorded Hessian frequencies. *Acoustic zeros resolved* counts units in which all three translational zeros (|ω| < 0.001 THz) appear within that window, and the residual column gives the largest of their magnitudes over those units. This is a check that the Hessian was symmetrised correctly; it is not a bound on the stochastic noise of the soft mode (§S2.4 gives the only seed spread measured). *Swamped* counts the opposite case: units with **no** recorded mode near zero, meaning at least six modes lie below the acoustic branches. Swamping is not itself an error: a deeply unstable phase genuinely has many imaginary modes, and the reported minimum frequency excludes the acoustic branches from the full spectrum rather than from this window. It does separate the families sharply. Read swamping as a fraction rather than a count, because the per-model denominators differ: on the perovskites it runs from 8/19 = 0.42 for MACE-MP-0 to 14/20 = 0.70 for CHGNet, so it is present for every architecture but is not uniform across them. On the fluorites it is **not** architecture-neutral, being 4/8 for ORB-v2 and 2/8 for MACE-MP-0 against 0/8 for the other three.

**What the production harness did not retain:** the per-iteration free-energy gradient history, and a per-unit uncertainty on the Hessian eigenvalues. The only uncertainty probe in the production data is the independent-seed study of §S2.4 on bcc-Zr/MACE-MP-0 at 100 K, where the harmonic layer finds no bcc instability. The revision re-ran four units that do carry an instability, four seeds each, with both recorded (Table S21): no seed converged, the seed spread of the lowest Hessian frequency is 0.007 to 0.075 THz on BaTiO₃, ZrO₂ and bcc Zr and 2.0 THz on SrTiO₃ at 600 K, and every seed gives the same call. The converged-recipe grid records all of these for every unit it ran (Table S22).

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

**Table S16** Screen-versus-SSCHA agreement on the bcc metals, scored on the stability call, and the same-energy comparison behind the SSCHA false-stables on the displacive systems, with the production-recipe SSCHA (2×2×2, not converged, §S2.1; the same comparisons with the converged recipe are in Table S22). For SSCHA the call is the sign of the lowest free-energy-Hessian frequency; for the screen it is the variational argmin of §2.4. Both methods run on the same MLIP potential-energy surface, so agreement between them is a consistency check and says nothing about agreement with first principles. *Trivial* pairs are (system, model) combinations whose harmonic layer has no instability, so both methods agree without any thermal stabilisation having been tested. bcc pairs are Ti, Zr and Hf at 100, 300 and 600 K. The screen's symmetric-point curvature is not scored against SSCHA. For a single mode with an even potential it equals the self-consistent trial stiffness at Q₀ = 0 (§S1.3), so it is positive wherever the width equation has a root and cannot register condensation; over all 1512 mode-temperature evaluations the positive values agree with the trial frequency to a median relative difference of 7 × 10⁻⁷ (95th percentile 1.0%). Of the 45 paired bcc units, 10 carry a negative screen value, and none of them is a physical curvature: 1 where the fitted polynomial has a positive quadratic term and a well only between sample points, a fit artefact of the kind described in §S1.4 (Ti/ORB-v2 100 K); 3 where the model has no screened imaginary mode and the value recorded is the softest commensurate harmonic frequency, not a curvature (Hf/ORB-v2 100 K, Hf/ORB-v2 300 K, Hf/ORB-v2 600 K); 6 where the width solver fell back to its nearest grid value at a point of the finite-difference stencil (Ti/MatterSim 100 K, Ti/MatterSim 300 K, Zr/MatterSim 100 K, Zr/MatterSim 300 K, Hf/MatterSim 100 K, Hf/MatterSim 300 K). `scripts/curvature_identity_check.py`. `scripts/stats_hardening.py` (`bcc_agreement`, `criterion_blindness`).

| Subset | Stability-call agreement [95% CI] |
|---|---|
| all pairs, all five models | 31/45 = 0.689 [0.543, 0.805] |
| all pairs, excluding ORB-v2 | 25/36 = 0.694 [0.531, 0.820] |
| trivial pairs (CHGNet and MACE-MP-0 on Zr and Hf) | 12/12 = 1.000 [0.758, 1.000] |
| non-trivial pairs, all five models | 19/33 = 0.576 [0.408, 0.728] |
| non-trivial pairs, excluding ORB-v2 | 13/24 = 0.542 [0.351, 0.721] |
| CHGNet | 7/9 = 0.778 [0.453, 0.937] |
| MACE-MP-0 | 9/9 = 1.000 [0.701, 1.000] |
| MatterSim | 0/9 = 0.000 [0.000, 0.299] |
| ORB-v2 | 6/9 = 0.667 [0.354, 0.879] |
| SevenNet-0 | 9/9 = 1.000 [0.701, 1.000] |

The frequency magnitudes are given descriptively only, because the pairs cluster by system and model and no test is attached: Spearman ρ = +0.113 (n = 45), −0.003 without ORB-v2 (n = 36).

Lower part: on every non-bcc unit where SSCHA calls the phase stable against an unstable label, the screen's call on the same MLIP energies. Where the screen calls the phase unstable, its free energy has a displaced minimum below the symmetric point, while the symmetric point of the single-mode problem keeps a positive curvature, so a criterion read at the symmetric reference would report stable. That shows the local and the global question have different answers on these energies. It does not show that SSCHA's positive Hessian has the same origin; the converged recipe of Table S22 keeps the false-stables on BaTiO₃ and KNbO₃ and removes those on the fluorites (§3.3).

| Quantity | All five models | Excluding ORB-v2 |
|---|---|---|
| non-bcc units SSCHA calls stable against an unstable label | 57 | 50 |
| screen's free-energy comparison calls the phase unstable | 46/57 = 0.807 [0.687, 0.889] | 39/50 = 0.780 [0.648, 0.872] |
| screen's symmetric-point curvature positive (by construction where the width equation is solved; §S1.3) | 52/57 = 0.912 [0.811, 0.962] | 47/50 = 0.940 [0.838, 0.979] |
|   of which batio3_cubic (n; call unstable) | 10; 6 | 8; 4 |
|   of which cssni3_cubic (n; call unstable) | 1; 1 | 1; 1 |
|   of which hfo2_cubic (n; call unstable) | 16; 16 | 14; 14 |
|   of which knbo3_cubic (n; call unstable) | 13; 6 | 11; 4 |
|   of which pbtio3_cubic (n; call unstable) | 2; 2 | 2; 2 |
|   of which zro2_cubic (n; call unstable) | 15; 15 | 14; 14 |
| all paired non-bcc units at T ≤ 300 K: SSCHA call = screen call | 21/66 | 20/56 |

**Table S17** Where SSCHA, with the production recipe (not converged, §S2.1), calls a phase unstable that its label calls stable, and where it fails numerically. Upper part: non-bcc SSCHA false-unstables by temperature, over the units whose label is stable at that temperature; numerical blow-ups (|f| > 50 THz) are included in the counts and also tallied separately. The count grows with temperature partly because more labels are stable at high temperature; the direct evidence is the per-unit trend below the table, in which units SSCHA calls stable at 100 K turn negative by 600 to 900 K. An instability that grows with thermal amplitude on a fixed potential-energy surface is the opposite of entropy stabilisation. It is what an MLIP extrapolating on large-amplitude thermal configurations would produce; an instability of the stochastic sampling itself, and a free-energy Hessian evaluated at an auxiliary matrix that has not reached the SCHA minimum, are the other candidates, and these data do not separate them. PBE forces on twelve configurations of the SrTiO₃ MACE-MP-0 600 K ensemble (Table S20) show the MLIPs extrapolating there (relative force error 0.19 against 0.11 near equilibrium, energy errors to 44 meV per atom), without showing that this rather than the sampling drives the runaway. The seed study shows the SrTiO₃ 600 K relaxation had not converged on any of four seeds (Table S21). With the converged recipe these counts are 0/4, 3/12 and 3/18 at 300, 600 and 900 K (all CsSnI₃), every SrTiO₃ unit above its transition is stable, and no unit turns negative with temperature (Table S22): the runaway is the production relaxation, not the MLIP. `scripts/stats_hardening.py` (`sscha_high_t`, `orb_split_s3`).

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

SrTiO₃ above its 105 K transition: SSCHA calls every returned unit unstable (14/14 = 1.000 [0.785, 1.000]; without ORB-v2 11/11 = 1.000 [0.741, 1.000]), at −3.4 to −914 THz (−3.4 to −60.5 THz without ORB-v2), each more negative than the same model's harmonic minimum. The production recipe never calls SrTiO₃ stable at any temperature; the converged recipe calls it stable in every converged unit, including 100 K, below the transition (Table S22). ORB-v2 has no imaginary commensurate mode in SrTiO₃, so the screen condenses nothing and the mode listed for it is its softest commensurate mode, which is harmonically stable (+1.35 THz).

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

**Table S18** Force-level ensemble spread as a predictor of unreliable finite-temperature calls, the test pre-registered at the head of `scripts/force_spread.py` (block hash 504d6031, recorded in the output). Units are the 60 consensus (system, temperature) units of Table S8, or the 300 per-model units behind them. For each unit, 16 thermally displaced configurations are drawn from the quantum harmonic distribution of the five models' mean force constants in the 2×2×2 supercell of the unrelaxed reference cell, the same configurations for every model. S is the median over configurations of the root-mean-square over atoms and components of the standard deviation of the forces across the models in the set. AUCs are for predicting the outcome in the Label column; intervals are cluster-bootstrap 95% intervals over systems and p is the system-clustered permutation p (10,000 draws each, seed 0); for (g) the blocks are unequal and only the interval is defined. The rule fixed in advance: the spread is reported as flagging untrustworthy calls only if the primary interval lies entirely above 0.5 and the four-model estimate (a) is also above 0.5. Verdict: not shown, because the primary interval spans 0.5. Of the 19 secondary scores, 5 have a lower bound above 0.5: (b) S, MACE-MP-0 small/medium/large; (c) S, MatterSim 1M/5M; (e) leave-one-out deviation of MACE-MP-0 from the mean of the other three, ORB-v2 excluded; (e) leave-one-out deviation of SevenNet-0 from the mean of the other three, ORB-v2 excluded; (f) S, MACE-MP-0 committee. None is adjusted for multiple comparisons, all use the same configurations, and the overlap restriction (g) was pre-registered for the primary only, so it was not applied to them. Restricted by (g), 43 units remain and hold 4 of the 12 consensus errors: the soft halide and ferroelectric spectra that carry most errors are the ones whose thermal draws bring A-site cations into the anions, where every model is extrapolating. The configurations sit near the unrelaxed reference cell rather than each model's relaxed cell, and the committees differ in size and training run, so they are family committees and not a deep ensemble of one model. Per-temperature primary AUCs (15 units each, descriptive only): 100 K 0.64 (1 wrong), 300 K 0.48 (4 wrong), 600 K 0.80 (5 wrong), 900 K 0.96 (2 wrong). `scripts/force_spread.py` → `results/revision/force_spread/summary.json`.

| Pre-registered score | Spread | Label | Units (wrong) | AUC [95% CI] | Clustered p |
|---|---|---|---|---|---|
| primary | force spread S, five models | consensus wrong | 60 (12) | 0.681 [0.416, 0.908] | 0.104 |
| (a) | S, four models (no ORB-v2) | four-model consensus wrong | 60 (12) | 0.684 [0.413, 0.916] | 0.103 |
| (b) | S, MACE-MP-0 small/medium/large | consensus wrong | 60 (12) | 0.717 [0.535, 0.902] | 0.040 |
| (c) | S, MatterSim 1M/5M | consensus wrong | 60 (12) | 0.726 [0.516, 0.901] | 0.054 |
| (d) | normalised S, five models | consensus wrong | 60 (12) | 0.578 [0.335, 0.795] | 0.541 |
| (d) | normalised S, four models | four-model consensus wrong | 60 (12) | 0.576 [0.323, 0.798] | 0.554 |
| (e) | leave-one-out deviation, pooled over five models | that model's call wrong | 300 (57) | 0.654 [0.425, 0.832] | 0.094 |
| (e) | leave-one-out deviation, pooled over four models | that model's call wrong | 240 (47) | 0.689 [0.438, 0.882] | 0.063 |
| (e) | leave-one-out deviation of MACE-MP-0 from the mean of the other four | MACE-MP-0's call wrong | 60 (14) | 0.699 [0.465, 0.907] | 0.082 |
| (e) | leave-one-out deviation of MACE-MP-0 from the mean of the other three, ORB-v2 excluded | MACE-MP-0's call wrong | 60 (14) | 0.731 [0.502, 0.927] | 0.041 |
| (e) | leave-one-out deviation of CHGNet from the mean of the other four | CHGNet's call wrong | 60 (11) | 0.681 [0.286, 0.931] | 0.152 |
| (e) | leave-one-out deviation of CHGNet from the mean of the other three, ORB-v2 excluded | CHGNet's call wrong | 60 (11) | 0.688 [0.307, 0.936] | 0.133 |
| (e) | leave-one-out deviation of ORB-v2 from the mean of the other four | ORB-v2's call wrong | 60 (10) | 0.542 [0.290, 0.768] | 0.736 |
| (e) | leave-one-out deviation of SevenNet-0 from the mean of the other four | SevenNet-0's call wrong | 60 (11) | 0.750 [0.481, 0.932] | 0.066 |
| (e) | leave-one-out deviation of SevenNet-0 from the mean of the other three, ORB-v2 excluded | SevenNet-0's call wrong | 60 (11) | 0.761 [0.502, 0.941] | 0.048 |
| (e) | leave-one-out deviation of MatterSim from the mean of the other four | MatterSim's call wrong | 60 (11) | 0.586 [0.331, 0.834] | 0.447 |
| (e) | leave-one-out deviation of MatterSim from the mean of the other three, ORB-v2 excluded | MatterSim's call wrong | 60 (11) | 0.584 [0.309, 0.844] | 0.453 |
| (f) | S, MACE-MP-0 committee | MACE-MP-0's call wrong | 60 (14) | 0.722 [0.564, 0.873] | 0.034 |
| (f) | S, MatterSim committee | MatterSim's call wrong | 60 (11) | 0.686 [0.473, 0.873] | 0.120 |
| (g) | primary, units with no atomic overlap in any configuration | consensus wrong | 43 (4) | 0.314 [0.146, 0.529] | -- |

**Table S19** PBE along the screen's own soft-mode coordinates. Each of the {tot[1]} ladder units (6 systems × 5 models × 100, 300, 600 and 900 K) is called twice by the screen's rule (unstable if any computed path condenses, §2.4), on the same structures: from the model's own energies (*MLIP, same paths*) and from Quantum ESPRESSO PBE single-point energies (*PBE-backed*). Both are scored against the finite-temperature label used throughout (stable iff T is at or above the experimental transition temperature). *Corrected by PBE* counts units whose MLIP call is wrong and whose PBE-backed call is right; *newly wrong* counts the reverse. **PBE is evaluated at each MLIP's own relaxed lattice, along that MLIP's coordinate**, so it is a different PBE potential for each model, not one reference curve per system. PBE covers 93 of the 139 screened paths: the deciding and reference paths and every other screened mode whose modulated cell has at most 12 atoms; of the 46 without PBE, 8 are symmetry copies of a computed path and 38 lie in cells of 18 to 216 atoms, so a PBE-backed *stable* means stable on the computed modes only. PBE covers 1 to 11 paths per unit, out of the 1 to 24 modes the screen maps; the mode that decided the ledger's call has a PBE path in 100 of 120 units. The MLIP column is the model's own call on those same paths and equals the ledger's call in 115 of 120 units. In 40 units (28 counting only deciding and reference paths) a path's regenerated E(Q) map did not reproduce the cached map the ledger was computed from (a substitute direction inside a degenerate eigenspace; `all_paths_match_cache` in the deposited table), so those calls rest partly on a substitute coordinate. Of the 23 corrections, 10 are bcc-Zr units on a reference coordinate and 8 are BaTiO₃ and KNbO₃ units, all at 300 K and all for CHGNet, MACE-MP-0, MatterSim, SevenNet-0. **The bcc-Zr paths of every model except MatterSim are MatterSim's deciding coordinate, q = (1/3, 2/3, 0)** (role `ref`: a coordinate borrowed from MatterSim, not one of the model's own), and ORB-v2's SrTiO₃ path is MatterSim's pattern at q = (1/2, 1/2, 1/2); 5 of the 93 paths below are of this kind, and on 4 of them the model's own curve has no well at all. The 3 newly wrong units are BaTiO₃/ORB-v2 at 300 K; bcc-Zr/MatterSim at 900 K; CsSnBr₃/ORB-v2 at 600 K; the 19 units that stay wrong under PBE are BaTiO₃ 1, KNbO₃ 5, CsSnBr₃ 9, bcc-Zr 4. Counts are descriptive: the units cluster by system (6 systems) and no test is attached. The label is a transition-temperature rule, so a PBE call that disagrees with it is not necessarily a PBE error. `scripts/dft_reference.py analyze` → `results/revision/dft/c3a_unit_calls.csv`.

| System | n units | MLIP, same paths: correct | PBE-backed: correct | Corrected by PBE | Newly wrong |
|---|---|---|---|---|---|
| batio3_cubic | 20 | 16 | 19 | 4 | 1 |
| knbo3_cubic | 20 | 11 | 15 | 4 | 0 |
| srtio3_cubic | 20 | 18 | 20 | 2 | 0 |
| cssnbr3_cubic | 20 | 9 | 11 | 3 | 1 |
| zr_bcc | 20 | 7 | 16 | 10 | 1 |
| zro2_cubic | 20 | 20 | 20 | 0 | 0 |
| *total, all five models* | 120 | 81 | 101 | 23 | 3 |
| *total, excluding ORB-v2* | 96 | 61 | 82 | 22 | 1 |

Lower part: the well depth of each of the 93 PBE paths (33 deciding, 5 reference, 55 other screened modes), the path model's own E(Q) against PBE on the same 10 structures. Depth is −min E(Q) over the sampled amplitudes, zero when none is below E(0), and Q_min is the sampled amplitude at that minimum, so both are limited to the 10-point scan to 0.45 Å. *Ratio* is own depth over PBE depth. The BaTiO₃ and KNbO₃ deciding paths of the four models other than ORB-v2 (12 paths: CHGNet, MACE-MP-0, MatterSim, SevenNet-0) have own-model depths 0.315 to 0.763 of PBE (median 0.525); with ORB-v2's 3 paths the range is 0.315 to 1.529. **The PBE minimum is at the scan edge (Q ≥ 0.449 Å) on 8 of 93 paths, all CsSnBr₃ deciding paths**; there the sampled depth is a lower bound on the PBE depth and the ratio is not a well-depth comparison. On 12 paths, all of role mode, PBE has no well at all (no sampled energy below E(0)) where the model has one; the ratio is not defined there.

| Path | Role | Own depth (meV) | PBE depth (meV) | Ratio own/PBE | Own Q_min (Å) | PBE Q_min (Å) | PBE minimum at scan edge |
|---|---|---|---|---|---|---|---|
| batio3_cubic_chgnet_q0-0-0-b0 | decide | 16.6 | 22.5 | 0.74 | 0.089 | 0.139 | no |
| batio3_cubic_chgnet_q0-0-0-b1 | mode | 30.3 | 39.3 | 0.77 | 0.089 | 0.089 | no |
| batio3_cubic_chgnet_q0-0-0-b2 | mode | 17.8 | 24.1 | 0.74 | 0.089 | 0.139 | no |
| batio3_cubic_chgnet_q0-1d2-0-b0 | mode | 24.9 | 49.6 | 0.50 | 0.089 | 0.089 | no |
| batio3_cubic_chgnet_q0-1d2-0-b1 | mode | 24.5 | 41.4 | 0.59 | 0.089 | 0.089 | no |
| batio3_cubic_chgnet_q1d2-0-0-b0 | mode | 29.4 | 61.7 | 0.48 | 0.089 | 0.089 | no |
| batio3_cubic_chgnet_q1d2-1d2-0-b0 | decide | 38.6 | 73.3 | 0.53 | 0.089 | 0.089 | no |
| batio3_cubic_mace_mp0_q0-0-0-b0 | mode | 20.8 | 34.3 | 0.61 | 0.089 | 0.089 | no |
| batio3_cubic_mace_mp0_q1d2-0-0-b0 | mode | 28.7 | 61.7 | 0.47 | 0.089 | 0.089 | no |
| batio3_cubic_mace_mp0_q1d2-0-1d2-b0 | decide | 42.6 | 73.5 | 0.58 | 0.089 | 0.089 | no |
| batio3_cubic_mattersim_q0-0-0-b0 | decide | 33.2 | 43.5 | 0.76 | 0.089 | 0.089 | no |
| batio3_cubic_mattersim_q0-0-1d2-b0 | mode | 37.7 | 59.0 | 0.64 | 0.089 | 0.089 | no |
| batio3_cubic_mattersim_q1d2-0-1d2-b0 | decide | 41.1 | 73.0 | 0.56 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q0-0-0-b0 | mode | 4.7 | 1.8 | 2.67 | 0.050 | 0.050 | no |
| batio3_cubic_orb_v2_q0-0-0-b1 | mode | 0.0 | 1.9 | 0.00 | no well | 0.050 | no |
| batio3_cubic_orb_v2_q0-0-1d2-b0 | mode | 14.8 | 11.1 | 1.34 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q0-0-1d2-b1 | mode | 9.3 | 10.9 | 0.86 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q0-1d2-0-b0 | mode | 14.7 | 10.6 | 1.39 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q0-1d2-0-b1 | mode | 8.3 | 11.0 | 0.76 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q1d2-0-0-b0 | mode | 20.2 | 13.7 | 1.47 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q1d2-0-0-b1 | mode | 19.0 | 13.9 | 1.37 | 0.089 | 0.089 | no |
| batio3_cubic_orb_v2_q1d2-1d2-0-b0 | decide | 32.8 | 21.4 | 1.53 | 0.089 | 0.089 | no |
| batio3_cubic_sevennet0_q0-0-0-b0 | mode | 21.2 | 25.5 | 0.83 | 0.089 | 0.089 | no |
| batio3_cubic_sevennet0_q1d2-0-0-b0 | decide | 28.2 | 42.4 | 0.67 | 0.089 | 0.089 | no |
| batio3_cubic_sevennet0_q1d2-0-1d2-b0 | decide | 33.3 | 69.0 | 0.48 | 0.050 | 0.089 | no |
| knbo3_cubic_chgnet_q0-0-0-b0 | decide | 30.3 | 57.8 | 0.52 | 0.089 | 0.089 | no |
| knbo3_cubic_mace_mp0_q0-0-0-b0 | mode | 16.3 | 32.2 | 0.51 | 0.089 | 0.089 | no |
| knbo3_cubic_mace_mp0_q0-0-1d2-b0 | mode | 15.2 | 45.5 | 0.33 | 0.050 | 0.089 | no |
| knbo3_cubic_mace_mp0_q0-1d2-1d2-b0 | decide | 27.9 | 87.8 | 0.32 | 0.050 | 0.089 | no |
| knbo3_cubic_mattersim_q0-0-0-b0 | decide | 25.4 | 48.8 | 0.52 | 0.089 | 0.139 | no |
| knbo3_cubic_mattersim_q0-0-1d2-b0 | mode | 20.6 | 63.8 | 0.32 | 0.050 | 0.089 | no |
| knbo3_cubic_mattersim_q0-1d2-1d2-b0 | decide | 28.9 | 91.8 | 0.31 | 0.050 | 0.089 | no |
| knbo3_cubic_orb_v2_q0-0-0-b0 | mode | 7.5 | 35.0 | 0.21 | 0.050 | 0.089 | no |
| knbo3_cubic_orb_v2_q0-0-0-b1 | mode | 7.2 | 26.5 | 0.27 | 0.050 | 0.089 | no |
| knbo3_cubic_orb_v2_q0-0-0-b2 | mode | 3.4 | 24.2 | 0.14 | 0.050 | 0.089 | no |
| knbo3_cubic_orb_v2_q0-0-1d2-b0 | mode | 28.4 | 64.1 | 0.44 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q0-0-1d2-b1 | mode | 24.3 | 63.4 | 0.38 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q0-1d2-0-b0 | mode | 33.8 | 72.2 | 0.47 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q0-1d2-0-b1 | mode | 34.5 | 71.6 | 0.48 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q0-1d2-1d2-b0 | decide | 58.8 | 97.6 | 0.60 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q1d2-0-0-b0 | mode | 29.1 | 67.4 | 0.43 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q1d2-0-0-b1 | mode | 25.0 | 66.6 | 0.38 | 0.089 | 0.139 | no |
| knbo3_cubic_orb_v2_q1d2-1d2-0-b0 | decide | 56.6 | 100.1 | 0.57 | 0.089 | 0.139 | no |
| knbo3_cubic_sevennet0_q0-0-0-b0 | mode | 16.1 | 30.7 | 0.53 | 0.089 | 0.089 | no |
| knbo3_cubic_sevennet0_q0-0-1d2-b0 | mode | 15.9 | 47.7 | 0.33 | 0.050 | 0.089 | no |
| knbo3_cubic_sevennet0_q0-1d2-1d2-b0 | decide | 31.6 | 94.6 | 0.33 | 0.089 | 0.089 | no |
| srtio3_cubic_chgnet_q0-0-0-b0 | mode | 0.1 | 6.1 | 0.02 | 0.006 | 0.050 | no |
| srtio3_cubic_chgnet_q0-0-1d2-b0 | mode | 0.2 | 0.9 | 0.18 | 0.006 | 0.022 | no |
| srtio3_cubic_chgnet_q1d2-1d2-1d2-b0 | decide | 5.8 | 23.4 | 0.25 | 0.139 | 0.139 | no |
| srtio3_cubic_mace_mp0_q0-0-0-b0 | mode | 0.5 | 7.3 | 0.07 | 0.022 | 0.050 | no |
| srtio3_cubic_mace_mp0_q1d2-1d2-1d2-b0 | decide | 27.4 | 27.7 | 0.99 | 0.139 | 0.139 | no |
| srtio3_cubic_mattersim_q0-0-0-b0 | mode | 3.6 | 8.9 | 0.40 | 0.050 | 0.050 | no |
| srtio3_cubic_mattersim_q1d2-1d2-1d2-b0 | decide | 17.2 | 26.5 | 0.65 | 0.139 | 0.139 | no |
| srtio3_cubic_orb_v2_q1d2-1d2-1d2-b0-refmattersim | ref | 57.4 | 26.7 | 2.15 | 0.139 | 0.139 | no |
| srtio3_cubic_sevennet0_q0-0-0-b0 | mode | 2.1 | 5.6 | 0.37 | 0.050 | 0.050 | no |
| srtio3_cubic_sevennet0_q1d2-1d2-1d2-b0 | decide | 22.9 | 26.1 | 0.88 | 0.139 | 0.139 | no |
| cssnbr3_cubic_chgnet_q0-0-0-b0 | mode | 8.5 | 0.0 | no PBE well | 0.200 | no well | no |
| cssnbr3_cubic_chgnet_q0-0-0-b1 | mode | 8.4 | 0.0 | no PBE well | 0.200 | no well | no |
| cssnbr3_cubic_chgnet_q0-0-0-b3 | mode | 2.6 | 0.0 | no PBE well | 0.272 | no well | no |
| cssnbr3_cubic_chgnet_q0-0-0-b4 | mode | 2.3 | 0.0 | no PBE well | 0.200 | no well | no |
| cssnbr3_cubic_chgnet_q0-0-1d2-b0 | mode | 13.8 | 0.0 | no PBE well | 0.356 | no well | no |
| cssnbr3_cubic_chgnet_q0-0-1d2-b2 | mode | 5.6 | 0.0 | no PBE well | 0.356 | no well | no |
| cssnbr3_cubic_chgnet_q0-1d2-0-b4 | mode | 3.9 | 0.0 | no PBE well | 0.272 | no well | no |
| cssnbr3_cubic_chgnet_q1d2-0-0-b0 | mode | 7.6 | 0.0 | no PBE well | 0.139 | no well | no |
| cssnbr3_cubic_chgnet_q1d2-0-0-b2 | mode | 3.6 | 0.0 | no PBE well | 0.272 | no well | no |
| cssnbr3_cubic_chgnet_q1d2-1d2-1d2-b0 | decide | 228.3 | 105.2 | 2.17 | 0.450 | 0.450 | yes |
| cssnbr3_cubic_mace_mp0_q1d2-1d2-0-b0 | decide | 55.7 | 60.6 | 0.92 | 0.450 | 0.450 | yes |
| cssnbr3_cubic_mace_mp0_q1d2-1d2-1d2-b0 | decide | 116.8 | 127.2 | 0.92 | 0.450 | 0.450 | yes |
| cssnbr3_cubic_mattersim_q0-0-0-b0 | mode | 1.4 | 0.0 | no PBE well | 0.139 | no well | no |
| cssnbr3_cubic_mattersim_q0-1d2-1d2-b0 | decide | 64.3 | 60.8 | 1.06 | 0.450 | 0.450 | yes |
| cssnbr3_cubic_mattersim_q1d2-0-0-b0 | mode | 2.5 | 0.0 | no PBE well | 0.200 | no well | no |
| cssnbr3_cubic_mattersim_q1d2-1d2-1d2-b0 | decide | 128.6 | 128.1 | 1.00 | 0.450 | 0.450 | yes |
| cssnbr3_cubic_orb_v2_q1d2-1d2-1d2-b1 | decide | 72.6 | 126.4 | 0.57 | 0.450 | 0.450 | yes |
| cssnbr3_cubic_orb_v2_q1d2-1d2-1d2-b2 | decide | 73.8 | 107.9 | 0.68 | 0.356 | 0.450 | yes |
| cssnbr3_cubic_sevennet0_q1d2-1d2-1d2-b0 | decide | 149.0 | 127.4 | 1.17 | 0.450 | 0.450 | yes |
| zr_bcc_chgnet_q1d3-2d3-0-b0-refmattersim | ref | 0.0 | 159.5 | 0.00 | no well | 0.200 | no |
| zr_bcc_mace_mp0_q1d3-2d3-0-b0-refmattersim | ref | 0.0 | 173.8 | 0.00 | no well | 0.200 | no |
| zr_bcc_mattersim_q1d2-0-0-b0 | mode | 50.1 | 67.7 | 0.74 | 0.200 | 0.200 | no |
| zr_bcc_mattersim_q1d2-1d2-0-b0 | mode | 100.1 | 136.3 | 0.73 | 0.200 | 0.200 | no |
| zr_bcc_mattersim_q1d3-0-0-b0 | mode | 63.6 | 53.4 | 1.19 | 0.272 | 0.200 | no |
| zr_bcc_mattersim_q1d3-1d2-0-b0 | mode | 64.7 | 31.3 | 2.07 | 0.200 | 0.139 | no |
| zr_bcc_mattersim_q1d3-1d3-0-b0 | mode | 206.3 | 0.0 | no PBE well | 0.200 | no well | no |
| zr_bcc_mattersim_q1d3-2d3-0-b0 | decide | 190.9 | 161.6 | 1.18 | 0.272 | 0.200 | no |
| zr_bcc_mattersim_q1d6-1d2-0-b0 | mode | 129.4 | 62.0 | 2.09 | 0.200 | 0.139 | no |
| zr_bcc_orb_v2_q1d3-2d3-0-b0-refmattersim | ref | 144.0 | 158.4 | 0.91 | 0.272 | 0.200 | no |
| zr_bcc_sevennet0_q1d3-2d3-0-b0-refmattersim | ref | 0.0 | 162.5 | 0.00 | no well | 0.200 | no |
| zro2_cubic_chgnet_q1d2-0-1d2-b0 | decide | 226.7 | 388.7 | 0.58 | 0.200 | 0.272 | no |
| zro2_cubic_mace_mp0_q0-1d2-1d2-b0 | decide | 173.4 | 385.4 | 0.45 | 0.200 | 0.272 | no |
| zro2_cubic_mattersim_q1d2-0-1d2-b0 | decide | 234.9 | 383.7 | 0.61 | 0.200 | 0.272 | no |
| zro2_cubic_orb_v2_q0-0-1d2-b0 | mode | 100.1 | 186.1 | 0.54 | 0.200 | 0.272 | no |
| zro2_cubic_orb_v2_q1d2-1d2-0-b0 | mode | 204.2 | 380.5 | 0.54 | 0.200 | 0.272 | no |
| zro2_cubic_orb_v2_q1d2-1d2-1d2-b0 | decide | 408.0 | 747.3 | 0.55 | 0.200 | 0.272 | no |
| zro2_cubic_sevennet0_q1d2-0-1d2-b0 | decide | 191.1 | 382.6 | 0.50 | 0.200 | 0.272 | no |

**Table S20** MLIP errors on SSCHA-sampled configurations, scored against PBE. The configurations are drawn from the production-recipe seed-study SSCHA ensembles of Table S21 (`scripts/sscha_seed_study.py`), one temperature per system; the baseline is near-equilibrium rattled configurations of the same supercell. Per set there are 4 baseline and 12 SSCHA configurations, so every figure rests on very few configurations and carries no uncertainty. The owner model is the one whose SSCHA ensemble produced the configurations; all five models are scored on the same configurations. Relative force RMSE is the RMSE over all Cartesian components divided by the RMS PBE force component; *Ratio* is the owner's relative force RMSE on the SSCHA configurations over its own baseline value. u_rms is the mean over configurations of the rms atomic displacement from the ideal supercell. Energy errors are per atom, taken relative to the undisplaced supercell in each code separately, and given as RMS / maximum absolute value over configurations; the baseline and SSCHA values are separated by a semicolon where two are given. Ranges run over the five evaluated models, with the range without ORB-v2 in brackets where it differs. **The owner's relative force error is 0.74 to 1.04 times its baseline for BaTiO₃, ZrO₂ and bcc-Zr, but 1.67 times for SrTiO₃ at 600 K** (0.113 to 0.190), where the configurations are displaced by 0.337 Å rms and the owner's energy error is 15.1 meV/atom rms and 44.3 meV/atom at its worst; its absolute force RMSE grows 15-fold while the PBE forces themselves grow 9-fold. **bcc-Zr's relative error has a PBE-force denominator of about 0.07 eV/Å on the baseline and is uninformative**: it goes from 0.596 to 0.439 on the SSCHA configurations although the owner's absolute force RMSE grows 2.3-fold, because the PBE force RMS (the denominator) grows 3.2-fold. `scripts/dft_reference.py analyze` → `results/revision/dft/c3b_units.csv`.

| System | Owner model | T (K) | Configs (baseline + SSCHA) | u_rms (Å), baseline / SSCHA | PBE force RMS (eV/Å), baseline / SSCHA | Owner force RMSE (eV/Å), baseline / SSCHA | Owner relative force RMSE, baseline | Owner relative force RMSE, SSCHA | Ratio | Relative force RMSE over five models, baseline; SSCHA | Owner energy error RMS / max (meV/atom), baseline; SSCHA | Energy error RMS over five models, SSCHA (meV/atom) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batio3_cubic | MACE-MP-0 | 100 | 4 + 12 | 0.033 / 0.091 | 0.271 / 0.496 | 0.027 / 0.052 | 0.100 | 0.104 | 1.04 | 0.041 to 0.135; 0.064 to 0.133 | 0.21 / 0.30; 0.54 / 1.46 | 0.4 to 2.0 |
| srtio3_cubic | MACE-MP-0 | 600 | 4 + 12 | 0.033 / 0.337 | 0.311 / 2.807 | 0.035 / 0.532 | 0.113 | 0.190 | 1.67 | 0.040 to 0.113; 0.124 to 0.203 | 0.37 / 0.55; 15.12 / 44.29 | 11.8 to 19.8 (11.8 to 17.7 without ORB-v2) |
| zr_bcc | MatterSim | 50 | 4 + 12 | 0.028 / 0.095 | 0.072 / 0.230 | 0.043 / 0.101 | 0.596 | 0.439 | 0.74 | 0.222 to 0.674 (0.577 to 0.674 without ORB-v2); 0.159 to 0.645 (0.439 to 0.645 without ORB-v2) | 0.73 / 0.87; 5.54 / 8.31 | 2.0 to 11.9 |
| zro2_cubic | MACE-MP-0 | 100 | 4 + 12 | 0.033 / 0.088 | 0.317 / 0.558 | 0.071 / 0.098 | 0.225 | 0.175 | 0.78 | 0.087 to 0.225; 0.124 to 0.175 | 0.94 / 1.10; 1.79 / 3.10 | 1.5 to 4.8 (1.5 to 3.2 without ORB-v2) |

**Table S21** SSCHA with the production recipe, re-run once per seed (`scripts/sscha_seed_study.py --preset revision`). **This is the production recipe, and no seed converged: 0 of 16 runs (4 seeds on each of 4 units) satisfy the minimiser's own stopping test.** Converged values are in the converged-recipe grid (Table S22), not here. Recipe: 256 configurations per population, at most 8 populations, `max_ka` = 20, `meaningful_factor` = 0.0001, then a separate Hessian ensemble of 512 configurations (256 antithetic pairs) at the final auxiliary matrix, without the fourth-order term. Seed 0 replays the production computation; the other seeds change only the random seed. *Populations × configs (moved)* is the populations drawn, the configurations in each and, in brackets, the populations whose kept steps changed the auxiliary matrix; *Steps taken (kept) / cap* counts the minimiser's steps over all populations, those not discarded when a population stops unconverged, and `max_ka`. **`max_ka` is a cumulative cap in python-sscha 1.6.1** (SchaMinimizer.py:1370 compares it with the step history accumulated over populations). In all 18 unconverged runs (these 16 and the 2 MatterSim runs of the lower part) the cap ended population 1 after 19 kept steps and each of populations 2 to 8 took one step and discarded it: the auxiliary matrix changes by less than 2 × 10⁻¹⁴ (relative) in populations 2 and later, so only the first population moved it. **The recorded gradient error is the library's placeholder.** python-sscha passes a constant in place of the stochastic error of the gradient (Ensemble.py:2657): the recorded value is the same at every step of every run of a unit and equals 0.433 per primitive-cell atom, so *Recorded gc error* is not a measurement. The convergence threshold is that value times `meaningful_factor`, and *Final gc ÷ threshold* is how far the last gradient sits above it: 7.3 × 10² to 1.0 × 10⁴ on the seed-study units, while the gradient falls by a factor of only 1.1 to 2.8 from the first step to the last. The structure gradient is at or below its threshold at the last step of every run, so the dynamical-matrix gradient alone is unconverged. **The Hessian lies close to the start on 3 of 4 units.** The start is the matrix after ForcePositiveDefinite (FPD), where the relaxation begins; the largest difference between it and the lowest Hessian frequency is 0.24 THz or less for BaTiO₃, ZrO₂ and bcc-Zr; for SrTiO₃ at 600 K it is 21 THz (start +0.90, Hessian −15.8 to −20.3), so there the finite-temperature correction, not the starting matrix, sets the lowest frequency. *Bootstrap SD* is the standard deviation of the lowest non-acoustic Hessian frequency over 30 resamples of the Hessian ensemble's antithetic pairs at the fixed final matrix. It measures the finite size of that ensemble only, whereas the seed spread (middle part) also contains the unconverged relaxation. For bcc-Zr/MatterSim at 50 K the lowest Hessian frequency is the final auxiliary matrix's own in every seed (identical in the recorded digits) and the bootstrap SD is 2.3 × 10⁻¹⁶ to 4.5 × 10⁻¹⁶, so the bootstrap says nothing there and the 0.075 THz seed spread comes entirely from the relaxation. *Ledger* is the deposited canonical SSCHA value for the unit, compared for seed 0 only (tolerance 0.02 THz): seed 0 reproduces it for BaTiO₃, ZrO₂ and bcc-Zr, but not for SrTiO₃ at 600 K, where seed 0 gives −18.71 THz against −20.23 and the ledger value lies inside the four-seed range (−20.26 to −15.77). `results/revision/sscha_seeds/*.json`.

| System | Model | T (K) | Cell | Seed | Populations × configs (moved) | Steps taken (kept) / cap | Converged | Stop reason | gc, first → final step | Recorded gc error (placeholder) | Final gc ÷ threshold | Start, after FPD (THz) | Hessian lowest (THz) | Bootstrap SD (THz) | Ledger (THz), seed 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BaTiO₃ | MACE-MP-0 | 100 | 2×2×2 | 0 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.52 → 0.36 | 2.165 | 1.7 × 10³ | +2.881 | +2.878 | 0.0027 | +2.868; Δ 0.011 (reproduced) |
| BaTiO₃ | MACE-MP-0 | 100 | 2×2×2 | 1 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.53 → 0.34 | 2.165 | 1.6 × 10³ | +2.881 | +2.881 | 0.0022 | -- |
| BaTiO₃ | MACE-MP-0 | 100 | 2×2×2 | 2 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.54 → 0.43 | 2.165 | 2.0 × 10³ | +2.881 | +2.867 | 0.0025 | -- |
| BaTiO₃ | MACE-MP-0 | 100 | 2×2×2 | 3 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.52 → 0.41 | 2.165 | 1.9 × 10³ | +2.881 | +2.868 | 0.0030 | -- |
| ZrO₂ | MACE-MP-0 | 100 | 2×2×2 | 0 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.25 → 0.094 | 1.299 | 7.3 × 10² | +2.962 | +3.082 | 0.011 | +3.090; Δ 0.008 (reproduced) |
| ZrO₂ | MACE-MP-0 | 100 | 2×2×2 | 1 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.27 → 0.097 | 1.299 | 7.5 × 10² | +2.962 | +3.087 | 0.0078 | -- |
| ZrO₂ | MACE-MP-0 | 100 | 2×2×2 | 2 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.27 → 0.11 | 1.299 | 8.7 × 10² | +2.962 | +3.069 | 0.0099 | -- |
| ZrO₂ | MACE-MP-0 | 100 | 2×2×2 | 3 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.26 → 0.10 | 1.299 | 7.8 × 10² | +2.962 | +3.069 | 0.0077 | -- |
| bcc-Zr | MatterSim | 50 | 2×2×2 | 0 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.19 → 0.16 | 0.433 | 3.8 × 10³ | +1.880 | +1.786 | 2.3 × 10⁻¹⁶ | +1.787; Δ 0.001 (reproduced) |
| bcc-Zr | MatterSim | 50 | 2×2×2 | 1 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.21 → 0.14 | 0.433 | 3.3 × 10³ | +1.880 | +1.636 | 2.3 × 10⁻¹⁶ | -- |
| bcc-Zr | MatterSim | 50 | 2×2×2 | 2 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.20 → 0.14 | 0.433 | 3.3 × 10³ | +1.880 | +1.638 | 2.3 × 10⁻¹⁶ | -- |
| bcc-Zr | MatterSim | 50 | 2×2×2 | 3 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.20 → 0.15 | 0.433 | 3.6 × 10³ | +1.880 | +1.742 | 4.5 × 10⁻¹⁶ | -- |
| SrTiO₃ | MACE-MP-0 | 600 | 2×2×2 | 0 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 2.2 → 2.0 | 2.165 | 9.4 × 10³ | +0.904 | −18.714 | 2.8 | −20.229; Δ 1.515 (not reproduced) |
| SrTiO₃ | MACE-MP-0 | 600 | 2×2×2 | 1 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 2.4 → 2.2 | 2.165 | 1.0 × 10⁴ | +0.904 | −17.046 | 2.2 | -- |
| SrTiO₃ | MACE-MP-0 | 600 | 2×2×2 | 2 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 2.2 → 1.9 | 2.165 | 8.9 × 10³ | +0.904 | −15.774 | 3.1 | -- |
| SrTiO₃ | MACE-MP-0 | 600 | 2×2×2 | 3 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 2.3 → 1.7 | 2.165 | 8.0 × 10³ | +0.904 | −20.264 | 3.7 | -- |

Middle part: the seed spread per unit. *SD* is the sample standard deviation (n − 1) of the lowest Hessian frequency over the seeds and *Seed SD ÷ bootstrap SD* its ratio to the median bootstrap SD (n/a where that is below 1 × 10⁻⁶ THz). *Call* is the sign rule used throughout: stable if the lowest non-acoustic frequency is at or above −0.1 THz. All seeds give the same call on every unit. None converged.

| System | Model | T (K) | Cell | Harmonic, before FPD (THz) | Start, after FPD (THz) | Final auxiliary matrix, min to max (THz) | Hessian, min to max (THz) | Hessian mean (THz) | SD over seeds (THz) | Median bootstrap SD (THz) | Seed SD ÷ bootstrap SD | Largest Hessian − start difference (THz) | Seeds converged | Call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BaTiO₃ | MACE-MP-0 | 100 | 2×2×2 | −6.299 | +2.881 | +2.891 to +2.902 | +2.867 to +2.881 | +2.874 | 0.0071 | 0.0026 | 2.8 | 0.014 | 0 of 4 | stable, 4 of 4 seeds |
| ZrO₂ | MACE-MP-0 | 100 | 2×2×2 | −5.521 | +2.962 | +3.159 to +3.179 | +3.069 to +3.087 | +3.077 | 0.0091 | 0.0088 | 1.0 | 0.12 | 0 of 4 | stable, 4 of 4 seeds |
| bcc-Zr | MatterSim | 50 | 2×2×2 | −2.030 | +1.880 | +1.636 to +1.786 | +1.636 to +1.786 | +1.700 | 0.075 | 2.3 × 10⁻¹⁶ | n/a | 0.24 | 0 of 4 | stable, 4 of 4 seeds |
| SrTiO₃ | MACE-MP-0 | 600 | 2×2×2 | −2.124 | +0.904 | +0.912 to +0.913 | −20.264 to −15.774 | −17.950 | 2.0 | 3.0 | 0.66 | 21 | 0 of 4 | unstable, 4 of 4 seeds |

Lower part: bcc-Zr cell size. The 3×3×3 cell was run once (seed 0) with the same production recipe in the current environments; the 2×2×2 value is the deposited canonical ledger row, not a re-run. The 3×3×3 minus 2×2×2 difference is −0.25 to −0.24 THz for MACE-MP-0 and −4.06 to −0.53 THz for MatterSim. The 2 runs that the library's test calls converged are MACE-MP-0 at 100 and 300 K: its harmonic matrix is already positive there, so the start is the harmonic matrix, and one population (17 and 20 steps) takes the gradient from 0.028 and 0.072 to 2.8 × 10⁻⁵ and 3.3 × 10⁻⁵, below the threshold of 4.3 × 10⁻⁵. The 2 MatterSim runs end at the cumulative cap as the seed-study units do (final gradient 7.1 × 10³ to 1.4 × 10⁴ times its threshold; the Hessian lies 0.26 to 3.7 THz from the final auxiliary matrix). The MatterSim value changes sign with cell size at 300 K (+1.95 THz in 2×2×2, −2.11 in 3×3×3). These are the production recipe's values and are not converged; Table S22 has the converged ones.

| Model | T (K) | 2×2×2, ledger (THz) | 3×3×3 Hessian (THz) | Bootstrap SD (THz) | 3×3×3 − 2×2×2 (THz) | Start, after FPD (THz) | Harmonic, before FPD (THz) | Populations × configs (moved) | Steps taken (kept) / cap | Converged | Stop reason | gc, first → final step | Final gc ÷ threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MACE-MP-0 | 100 | +1.800 | +1.562 | 5.2 × 10⁻⁴ | −0.238 | +1.567 | +1.567 | 1 × 256 (1) | 17 (17) / 20 | yes | converged ×1 | 0.028 → 2.8 × 10⁻⁵ | 0.64 |
| MACE-MP-0 | 300 | +1.797 | +1.547 | 0.0015 | −0.250 | +1.567 | +1.567 | 1 × 256 (1) | 20 (20) / 20 | yes | converged ×1 | 0.072 → 3.3 × 10⁻⁵ | 0.77 |
| MatterSim | 100 | +1.846 | +1.320 | 0.022 | −0.526 | +1.489 | −1.949 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.35 → 0.31 | 7.1 × 10³ |
| MatterSim | 300 | +1.953 | −2.106 | 0.15 | −4.059 | +1.488 | −1.949 | 8 × 256 (1) | 27 (19) / 20 | no | max_ka (cumulative) ×8 | 0.62 → 0.61 | 1.4 × 10⁴ |

**Table S22** SSCHA with the converged recipe on the units behind the §3.3 claims, beside the production numbers on the same units (`scripts/sscha_seed_study.py --preset grid`, compared with the production ledger by `scripts/grid_compare.py`). **Recipe:** 1000 configurations per population, at most 30 populations and at most 400 minimiser steps per population (a cap per population, where the production `max_ka` is cumulative over populations, Table S21), `meaningful_factor` = 0.2 applied to the real stochastic error of the gradient (the library's serial estimate, in place of the constant placeholder of Table S21; it is set to zero wherever KL/N < 0.7, so convergence cannot be declared at a statistically invalid point), Kong-Liu ratio 0.5, then a Hessian ensemble of 2000 configurations with 10 bootstrap resamples, and a 7200 s wall cap on each relaxation. Production: 256 configurations per population, at most 8 populations, `max_ka` = 20 cumulative, `meaningful_factor` = 0.0001 against the placeholder error, a Hessian ensemble of 512. **Start A only** (the production ForcePositiveDefinite start): start dependence is the six-unit start-A against start-B study (§3.5), not repeated across the grid. A call is the sign rule used throughout: stable iff the lowest free-energy-Hessian frequency is at or above −0.1 THz. **Run status:** 178 planned units, 171 finished, 159 of them (93% of those finished) met the library's stopping test (stop reasons: converged 159, max pop 11, wall cap 1); 1 ended at the wall cap, 7 failed, 0 are not finished, 4 returned a blow-up (|f| > 50 THz); production returned no value for 7 of the 178 units. Only runs that met the stopping test enter the claim table; the others are listed under *Planned, not evaluated* in the coverage table. **Coverage.** Of the 57 units where SSCHA calls a phase stable against an unstable label in Table S16, 57 are in the grid; of the 57 in the grid, 21 are still called stable by the converged recipe (the screen calls 11/21 of them unstable) and 32 are called unstable, and 4 have no converged value (finished without meeting the stopping test 4). The grid covers the non-bcc systems at 100, 300, 600 and 900 K in the 2×2×2 cell and the bcc metals at 100, 300 and 600 K in the 3×3×3 cell. **Fresh-ensemble check.** At the final matrix of each converged unit the gradient was recomputed on an independent ensemble of 2000 configurations; it is consistent with a minimum on 152 of the 159 converged units and not on 7 (bcc-Ti/MACE-MP-0 300 K, HfO₂/MACE-MP-0 100 K, HfO₂/CHGNet 300 K, bcc-Hf/MatterSim 300 K, bcc-Ti/MatterSim 600 K, bcc-Zr/MatterSim 600 K and ZrO₂/MatterSim 900 K), so *converged* means the library's stopping test, not an independent verification. None of the 34 converged false-stables is among them; among the false-stables, BaTiO₃/CHGNet at 100 K (+1.65 ± 1.50 THz) lies within two bootstrap standard deviations of zero. **The bcc cell is 3×3×3 in the grid and 2×2×2 in production**, so on bcc a change of call mixes the cell change with the convergence (Table S21, lower part, measures the cell change at the production recipe), and the bcc agreement below compares the screen with a different cell's SSCHA. **How the columns are made.** Each production claim is recomputed with the function that produced it (`stats_hardening.criterion_blindness` for claim (i), `analysis.displacive_recall` for (ii), `stats_hardening.bcc_agreement` for (iii), `stats_hardening.sscha_high_t` for (iv); Tables S16 and S17 and §3.3), so the unit sets, the ORB-v2 split and the blow-up rules are the production ones: SSCHA blow-ups leave the denominator in (ii), where they are counted beside, and stay in (i), (iii) and (iv). The first column is the production number on its full set, recomputed from the ledger and checked against `results/stats_hardening.json`; the second is the production number on exactly the units the grid evaluated; the third is the converged value on those same units; the fourth is the converged value on every unit evaluated, which includes units that production never returned (7 in the plan). A count of 0/0 means no unit of that kind was evaluated. `results/revision/grid_compare.json`.

| Claim | Quantity | Models | Production, full set | Production, same units as the grid | Converged, same units | Converged, all units evaluated |
|---|---|---|---|---|---|---|
| (i) | non-bcc units SSCHA calls stable against an unstable label (of those labelled unstable) | all five | 57/89 | 53/84 | 34/84 | 34/84 |
| (i) | non-bcc units SSCHA calls stable against an unstable label (of those labelled unstable) | without ORB-v2 | 50/75 | 49/74 | 30/74 | 30/74 |
| (i) | of those, the screen's free-energy comparison calls unstable | all five | 46/57 | 43/53 | 17/34 | 17/34 |
| (i) | of those, the screen's free-energy comparison calls unstable | without ORB-v2 | 39/50 | 39/49 | 14/30 | 14/30 |
| (i) | non-bcc, T ≤ 300 K: SSCHA call = screen call | all five | 21/66 | 20/60 | 42/60 | 42/60 |
| (i) | non-bcc, T ≤ 300 K: SSCHA call = screen call | without ORB-v2 | 20/56 | 19/55 | 40/55 | 40/55 |
| (ii) | FE perovskites, T ≤ 300 K: unstable recalled by the screen | all five | 16/30 | 15/26 | 15/26 | 15/26 |
| (ii) | FE perovskites, T ≤ 300 K: unstable recalled by the screen | without ORB-v2 | 12/24 | 12/23 | 12/23 | 12/23 |
| (ii) | FE perovskites, T ≤ 300 K: unstable recalled by SSCHA | all five | 5/27 (+1 blow-up) | 5/25 (+1 blow-up) | 4/26 | 4/26 |
| (ii) | FE perovskites, T ≤ 300 K: unstable recalled by SSCHA | without ORB-v2 | 5/23 (+1 blow-up) | 5/22 (+1 blow-up) | 4/23 | 4/23 |
| (iii) | bcc, 100-600 K: screen call = SSCHA call (production 2×2×2, grid 3×3×3) | all five | 31/45 | 28/41 | 33/41 | 33/41 |
| (iii) | bcc, 100-600 K: screen call = SSCHA call (production 2×2×2, grid 3×3×3) | without ORB-v2 | 25/36 | 25/36 | 32/36 | 32/36 |
| (iv) | non-bcc SSCHA false-unstable at 100 K (of stable-labelled units) | all five | 0/0 | 0/0 | 0/0 | 0/0 |
| (iv) | non-bcc SSCHA false-unstable at 100 K (of stable-labelled units) | without ORB-v2 | 0/0 | 0/0 | 0/0 | 0/0 |
| (iv) | non-bcc SSCHA false-unstable at 300 K (of stable-labelled units) | all five | 5/5 | 4/4 | 0/4 | 0/4 |
| (iv) | non-bcc SSCHA false-unstable at 300 K (of stable-labelled units) | without ORB-v2 | 4/4 | 4/4 | 0/4 | 0/4 |
| (iv) | non-bcc SSCHA false-unstable at 600 K (of stable-labelled units) | all five | 10/13 | 9/12 | 3/12 | 3/12 |
| (iv) | non-bcc SSCHA false-unstable at 600 K (of stable-labelled units) | without ORB-v2 | 8/11 | 8/11 | 3/11 | 3/11 |
| (iv) | non-bcc SSCHA false-unstable at 900 K (of stable-labelled units) | all five | 15/19 | 14/18 | 3/18 | 3/18 |
| (iv) | non-bcc SSCHA false-unstable at 900 K (of stable-labelled units) | without ORB-v2 | 12/16 | 12/16 | 3/16 | 3/16 |

Coverage of the production units behind each claim: how many are in the grid, how many have a converged value, which are not in the grid (by temperature) and which were planned but have no converged value.

| Claim | Production units | Count | In the grid plan | Evaluated | Not in the grid (by T) | Planned, not evaluated |
|---|---|---|---|---|---|---|
| (i) | the SSCHA false-stables of Table S16 (non-bcc, label unstable, SSCHA stable) | 57 | 57 | 53 | none | finished without meeting the stopping test 4 |
| (ii) | the ferroelectric-perovskite screen units at T ≤ 300 K (the SSCHA rows are a subset) | 30 | 30 | 26 | none | failed 2, finished without meeting the stopping test 2 |
| (iii) | the Ti/Zr/Hf units with both a screen and an SSCHA row | 45 | 45 | 41 | none | finished without meeting the stopping test 4 |
| (iv) | the non-bcc SSCHA units whose label is stable (the false-unstable denominators) | 37 | 37 | 34 | none | finished without meeting the stopping test 3 |

Run status by model.

| Model | Planned | ok | Converged | Wall cap | Failed | Not finished | Blow-ups (over 50 THz) |
|---|---|---|---|---|---|---|---|
| CHGNet | 37 | 37 | 36 | 1 | 0 | 0 | 0 |
| MACE-MP-0 | 36 | 36 | 36 | 0 | 0 | 0 | 0 |
| MatterSim | 37 | 35 | 35 | 0 | 2 | 0 | 0 |
| ORB-v2 | 33 | 29 | 18 | 0 | 4 | 0 | 4 |
| SevenNet-0 | 35 | 34 | 34 | 0 | 1 | 0 | 0 |
| total | 178 | 171 | 159 | 1 | 7 | 0 | 4 |

SrTiO₃ above its 105 K transition (the units of Table S17's second part that the grid or the converged mode evaluated; the label is stable at every one): the converged recipe calls 0/11 of them unstable.

| Model | T (K) | Production SSCHA minimum (THz) | Converged Hessian minimum (THz) | Production call | Converged call |
|---|---|---|---|---|---|
| CHGNet | 300 | −20.54 | +1.71 | unstable | stable |
| MACE-MP-0 | 300 | −3.39 | +1.83 | unstable | stable |
| MatterSim | 300 | −14.92 | +1.83 | unstable | stable |
| SevenNet-0 | 300 | −7.86 | +1.84 | unstable | stable |
| CHGNet | 600 | −36.64 | +2.36 | unstable | stable |
| MACE-MP-0 | 600 | −20.23 | +2.41 | unstable | stable |
| MatterSim | 600 | −22.18 | +2.42 | unstable | stable |
| SevenNet-0 | 600 | −30.52 | +2.42 | unstable | stable |
| CHGNet | 900 | −60.51 | +2.43 | unstable | stable |
| MACE-MP-0 | 900 | −27.14 | +2.76 | unstable | stable |
| MatterSim | 900 | −35.54 | +2.67 | unstable | stable |

Calls that differ from production, with the old and new Hessian minimum: 82 of the 171 finished units that have a production value change their call (stable → unstable 48, unstable → stable 34, whatever stopped the relaxation); of the 159 that met the stopping test, 72 change without a blow-up on either side (63 in the production cell and 9 bcc units where the cell changes as well), and on the non-bcc units 51 of those change from wrong to right against the label and 12 from right to wrong. *n/s*: the bcc label is the thermodynamic one and is not scored against a dynamical call (§3.3).

| System | Model | T (K) | Cell | Production Hessian (THz) | Converged Hessian (THz) | Call | Label | Screen | Converged | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| hf_bcc | MatterSim | 100 | 3×3×3 | +1.60 | −3.08 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| hf_bcc | MatterSim | 300 | 3×3×3 | +1.65 | −1.58 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| hf_bcc | ORB-v2 | 300 | 3×3×3 | +0.09 | −2.49 | stable → unstable | n/s | stable | no (max_pop) | cell and convergence change together |
| hf_bcc | MatterSim | 600 | 3×3×3 | +1.68 | −0.57 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| hf_bcc | ORB-v2 | 600 | 3×3×3 | +0.08 | −1.41 | stable → unstable | n/s | stable | no (max_pop) | cell and convergence change together |
| ti_bcc | MatterSim | 100 | 3×3×3 | +1.88 | −3.72 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| ti_bcc | MatterSim | 300 | 3×3×3 | +2.04 | −1.63 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| ti_bcc | ORB-v2 | 300 | 3×3×3 | +0.27 | −1.19 | stable → unstable | n/s | unstable | no (max_pop) | cell and convergence change together |
| zr_bcc | MatterSim | 100 | 3×3×3 | +1.85 | −2.14 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| zr_bcc | ORB-v2 | 100 | 3×3×3 | +1.16 | −3.10 | stable → unstable | n/s | stable | yes | cell and convergence change together |
| zr_bcc | MatterSim | 300 | 3×3×3 | +1.95 | −0.91 | stable → unstable | n/s | unstable | yes | cell and convergence change together |
| zr_bcc | ORB-v2 | 300 | 3×3×3 | +1.20 | −1.25 | stable → unstable | n/s | stable | yes | cell and convergence change together |
| batio3_cubic | CHGNet | 300 | 2×2×2 | +2.18 | −4.30 | stable → unstable | unstable | stable | no (wall_cap) |  |
| batio3_cubic | ORB-v2 | 300 | 2×2×2 | +2.62 | −1.77 | stable → unstable | unstable | unstable | no (max_pop) |  |
| batio3_cubic | CHGNet | 600 | 2×2×2 | −5.85 | +2.59 | unstable → stable | stable | stable | yes |  |
| batio3_cubic | ORB-v2 | 600 | 2×2×2 | −9.52 | +2.81 | unstable → stable | stable | stable | yes |  |
| batio3_cubic | CHGNet | 900 | 2×2×2 | −10.56 | +2.71 | unstable → stable | stable | stable | yes |  |
| batio3_cubic | MACE-MP-0 | 900 | 2×2×2 | −7.60 | +3.02 | unstable → stable | stable | stable | yes |  |
| batio3_cubic | ORB-v2 | 900 | 2×2×2 | −16.85 | +2.68 | unstable → stable | stable | stable | yes |  |
| batio3_cubic | SevenNet-0 | 900 | 2×2×2 | −4.24 | +2.99 | unstable → stable | stable | stable | yes |  |
| cssni3_cubic | MACE-MP-0 | 100 | 2×2×2 | +0.36 | −0.96 | stable → unstable | unstable | unstable | yes |  |
| knbo3_cubic | CHGNet | 600 | 2×2×2 | −4.27 | +2.52 | unstable → stable | unstable | stable | yes |  |
| knbo3_cubic | ORB-v2 | 600 | 2×2×2 | −13.75 | +2.40 | unstable → stable | unstable | stable | yes |  |
| knbo3_cubic | CHGNet | 900 | 2×2×2 | −16.89 | +2.74 | unstable → stable | stable | stable | yes |  |
| knbo3_cubic | ORB-v2 | 900 | 2×2×2 | −23.64 | +2.64 | unstable → stable | stable | stable | yes |  |
| pbtio3_cubic | CHGNet | 100 | 2×2×2 | +0.49 | −0.34 | stable → unstable | unstable | unstable | yes |  |
| pbtio3_cubic | SevenNet-0 | 100 | 2×2×2 | +0.09 | −0.19 | stable → unstable | unstable | unstable | yes |  |
| pbtio3_cubic | CHGNet | 300 | 2×2×2 | −1.26 | +0.57 | unstable → stable | unstable | unstable | yes |  |
| pbtio3_cubic | MACE-MP-0 | 300 | 2×2×2 | −8.11 | +0.79 | unstable → stable | unstable | stable | yes |  |
| pbtio3_cubic | MatterSim | 300 | 2×2×2 | −801.29 | +1.01 | unstable → stable | unstable | stable | yes | production blow-up |
| pbtio3_cubic | SevenNet-0 | 300 | 2×2×2 | −12.22 | +1.12 | unstable → stable | unstable | unstable | yes |  |
| pbtio3_cubic | CHGNet | 600 | 2×2×2 | −11.16 | +1.31 | unstable → stable | unstable | unstable | yes |  |
| pbtio3_cubic | MACE-MP-0 | 600 | 2×2×2 | −33.47 | +1.49 | unstable → stable | unstable | stable | yes |  |
| pbtio3_cubic | SevenNet-0 | 600 | 2×2×2 | −37.79 | +1.55 | unstable → stable | unstable | stable | yes |  |
| pbtio3_cubic | CHGNet | 900 | 2×2×2 | −30.62 | +1.36 | unstable → stable | stable | stable | yes |  |
| pbtio3_cubic | SevenNet-0 | 900 | 2×2×2 | −66.85 | +1.65 | unstable → stable | stable | stable | yes | production blow-up |
| srtio3_cubic | CHGNet | 100 | 2×2×2 | −1.36 | +1.06 | unstable → stable | unstable | stable | yes |  |
| srtio3_cubic | MACE-MP-0 | 100 | 2×2×2 | −0.49 | +1.12 | unstable → stable | unstable | unstable | yes |  |
| srtio3_cubic | MatterSim | 100 | 2×2×2 | −0.70 | +1.20 | unstable → stable | unstable | unstable | yes |  |
| srtio3_cubic | SevenNet-0 | 100 | 2×2×2 | −1.04 | +1.09 | unstable → stable | unstable | unstable | yes |  |
| srtio3_cubic | CHGNet | 300 | 2×2×2 | −20.54 | +1.71 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | MACE-MP-0 | 300 | 2×2×2 | −3.39 | +1.83 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | MatterSim | 300 | 2×2×2 | −14.92 | +1.83 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | SevenNet-0 | 300 | 2×2×2 | −7.86 | +1.84 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | CHGNet | 600 | 2×2×2 | −36.64 | +2.36 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | MACE-MP-0 | 600 | 2×2×2 | −20.23 | +2.41 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | MatterSim | 600 | 2×2×2 | −22.18 | +2.42 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | SevenNet-0 | 600 | 2×2×2 | −30.52 | +2.42 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | CHGNet | 900 | 2×2×2 | −60.51 | +2.43 | unstable → stable | stable | stable | yes | production blow-up |
| srtio3_cubic | MACE-MP-0 | 900 | 2×2×2 | −27.14 | +2.76 | unstable → stable | stable | stable | yes |  |
| srtio3_cubic | MatterSim | 900 | 2×2×2 | −35.54 | +2.67 | unstable → stable | stable | stable | yes |  |
| hfo2_cubic | CHGNet | 100 | 2×2×2 | +2.22 | −42.75 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | MACE-MP-0 | 100 | 2×2×2 | +2.39 | −10.12 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | MatterSim | 100 | 2×2×2 | +2.59 | −28.40 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | ORB-v2 | 100 | 2×2×2 | +1.90 | −49.39 | stable → unstable | unstable | unstable | no (max_pop) |  |
| hfo2_cubic | SevenNet-0 | 100 | 2×2×2 | +2.62 | −30.99 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | CHGNet | 300 | 2×2×2 | +2.13 | −8.67 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | MACE-MP-0 | 300 | 2×2×2 | +1.62 | −2.42 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | MatterSim | 300 | 2×2×2 | +2.53 | −13.91 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | ORB-v2 | 300 | 2×2×2 | +1.24 | −3.16 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | SevenNet-0 | 300 | 2×2×2 | +2.58 | −3.14 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | CHGNet | 600 | 2×2×2 | +1.71 | −3.66 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | MatterSim | 600 | 2×2×2 | +2.41 | −3.69 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | SevenNet-0 | 600 | 2×2×2 | +2.47 | −3.28 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | CHGNet | 900 | 2×2×2 | +1.56 | −3.89 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | MatterSim | 900 | 2×2×2 | +2.05 | −3.86 | stable → unstable | unstable | unstable | yes |  |
| hfo2_cubic | SevenNet-0 | 900 | 2×2×2 | +0.87 | −3.22 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | CHGNet | 100 | 2×2×2 | +3.07 | −28.01 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MACE-MP-0 | 100 | 2×2×2 | +3.09 | −23.26 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MatterSim | 100 | 2×2×2 | +3.30 | −26.46 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | ORB-v2 | 100 | 2×2×2 | +2.04 | −50.02 | stable → unstable | unstable | unstable | no (max_pop) | converged blow-up |
| zro2_cubic | SevenNet-0 | 100 | 2×2×2 | +3.33 | −22.85 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | CHGNet | 300 | 2×2×2 | +2.91 | −10.10 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MACE-MP-0 | 300 | 2×2×2 | +2.88 | −2.87 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MatterSim | 300 | 2×2×2 | +3.15 | −10.60 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | SevenNet-0 | 300 | 2×2×2 | +3.20 | −2.98 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | CHGNet | 600 | 2×2×2 | +2.35 | −3.29 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MACE-MP-0 | 600 | 2×2×2 | +1.93 | −2.86 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MatterSim | 600 | 2×2×2 | +3.02 | −3.42 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | SevenNet-0 | 600 | 2×2×2 | +2.71 | −3.27 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | CHGNet | 900 | 2×2×2 | +1.23 | −3.26 | stable → unstable | unstable | unstable | yes |  |
| zro2_cubic | MatterSim | 900 | 2×2×2 | +2.32 | −3.43 | stable → unstable | unstable | unstable | yes |  |

**Table S23** Checks on the errors that persist on PBE (`scripts/dft_checks.py analyze-checks` → `results/revision/dft_checks/`; 323 calculations). Upper part: k-point and cutoff convergence of the MACE-MP-0 deciding path of each system, as the relative change of the well depth (−min E(Q) over the ten sampled amplitudes, per modulated cell) from the production settings (k-spacing 0.25 Å⁻¹ with 2π included, SSSP cutoffs) to a k-spacing of 0.15 Å⁻¹ (*k*), cutoffs ×1.3 with the density cutoff kept at eight times the wavefunction cutoff (*ecut*), and both. The criterion, fixed before any variant ran, is depth within 5% and screen call identical at every T in [50.0, 100.0, 300.0, 600.0, 900.0], for each variant, on every path; the verdict is **NOT converged**. It fails on CsSnBr₃ only, by the depth, and no call changes anywhere; *Calls changed* is over 5 temperatures × 3 variants. Middle part: the screen re-solved with PBEsol on the same structures, over the ladder units of each system (5 models × 100, 300, 600, 900 K), along the 23 deciding paths; **PBEsol here is input_dft = 'pbesol' on the SSSP 1.3 PBE pseudopotentials at each MLIP's lattice, not the SSSP PBEsol set and not PBEsol's own lattice**. PBEsol corrects 9 of the 15 PBE errors and introduces 0; the 6 left wrong are BaTiO₃/ORB-v2 at 300 K, CsSnBr₃/ORB-v2 at 300 K, KNbO₃/CHGNet at 600 K, KNbO₃/MACE-MP-0 at 600 K, KNbO₃/MatterSim at 600 K and KNbO₃/ORB-v2 at 600 K. Lower part: the cubic lattice relaxed in PBE (vc-relax, production settings, final pressure within 0.02 kbar) against the MLIP lattices the curves of Table S19 use. The PBE force constants and E(Q) profiles at the PBE lattice (phases B and C of the check) were not computed, so whether the lattice or the eigenvector moves a call is not tested. Counts are descriptive (three systems).

| System | Path (MACE-MP-0, deciding) | Production depth (meV) | *k* | *ecut* | *k* + *ecut* | Calls changed (of 15) | PBE minimum at scan edge | Criterion |
|---|---|---|---|---|---|---|---|---|
| BaTiO₃ | q = (1/2, 0, 1/2), band 0 | 73.5 | −0.1 % | +1.7 % | +1.6 % | 0 | no | met |
| KNbO₃ | q = (0, 1/2, 1/2), band 0 | 87.8 | −0.2 % | −0.2 % | −0.4 % | 0 | no | met |
| CsSnBr₃ | q = (1/2, 1/2, 1/2), band 0 | 127.2 | −6.8 % | −0.0 % | −6.9 % | 0 | yes, all variants | missed (depth; k, ke) |

| System | Units | Correct, PBE | Correct, PBEsol | PBE errors corrected | New errors | Depth PBEsol/PBE | PBEsol minimum at scan edge |
|---|---|---|---|---|---|---|---|
| BaTiO₃ | 20 | 19 | 19 | 0 | 0 | 1.12–1.30 | no |
| KNbO₃ | 20 | 15 | 16 | 1 (SevenNet-0, 600 K) | 0 | 1.03–1.12 | no |
| CsSnBr₃ | 20 | 11 | 19 | 8 | 0 | 0.32–0.42 | no (0.36 Å; PBE at the 0.45 Å edge) |
| *total* | 60 | 45 | 54 | 9 | 0 |  |  |

| System | PBE a (Å) | MLIP a (Å), five models | MLIP − PBE (%) |
|---|---|---|---|
| BaTiO₃ | 4.0237 | 4.0340–4.0361 | +0.26 to +0.31 (largest CHGNet) |
| KNbO₃ | 4.0269 | 4.0567–4.0590 | +0.74 to +0.80 (largest MatterSim) |
| CsSnBr₃ | 5.8871 | 5.8941–5.9205 | +0.12 to +0.57 (largest CHGNet) |

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
  surface, so their agreement is a consistency check. On bcc the stability calls agree in 31/45
  with production SSCHA in 2×2×2 (0.69 [0.54, 0.80]; 25/36 without ORB-v2); 12 of the agreeing
  units have no harmonic instability, and MatterSim agrees on the call in 0/9 (Table S16). With
  converged SSCHA in 3×3×3 they agree in 33/41 (0.80 [0.66, 0.90]; 32/36 without ORB-v2), MatterSim
  in 7/9, a change that mixes cell and convergence (Table S22). The frequency magnitudes
  give Spearman ρ = 0.11 (−0.003 without ORB-v2), reported descriptively with no test. On the
  displacive systems the two diverge because they answer different questions: the default SSCHA
  criterion is the curvature of the free energy at the symmetric reference, a local test, and
  it cannot see condensation into a deeper displaced minimum, which the screen's free-energy
  comparison does see. On the same MLIP energies that comparison finds a displaced minimum below
  the symmetric point on 17/34 of the units converged SSCHA false-stabilises (Tables S16 and S22;
  §3.3; 46/57 with the production recipe). Outcome of the converged runs: the false-stables
  survive on BaTiO₃ and KNbO₃ (from both starts on BaTiO₃/MACE-MP-0), where the screen finds the
  displaced minimum on every 100 K unit, so early stopping is excluded there; they do not survive
  on the fluorites, where early stopping was the cause (Table S22).*
  *PBE forces on the sampled configurations (§S5, Table S20) show no extrapolation error behind
  the 100 K false-stables of BaTiO₃ (and ZrO₂, which is no longer a false-stable once
  converged).*
- **Single-mode approximation itself** → derivation and sensitivity analysis (§S1.3, Table S14).
  *Outcome: varying the fit window, sampling range, condensation threshold and scan range changes
  at most one unit call at T ≤ 300 K, but the frozen-cell normalisation is outcome-determining
  (ferroelectric recall 5/30 to 28/30 across conventions; the H2 count at 300 K, 17 v 4 in the
  production cell, runs from 7 v 4 to 24 v 4; §S1.3), and mode–mode coupling is unbounded. The
  screen's T*, its ferroelectric recall and the H2 count are conditional on the minimal-cell
  convention.*
  *On PBE energies along the same coordinates the screen agrees with the labels in 101/120 units
  of six systems against 81/120 on the MLIP energies (§S5, Table S19); the 16 errors that persist
  (KNbO₃ 600 K, CsSnBr₃ 300/600 K, bcc Zr 900 K) bound what the single-mode approximation, the
  functional and the labels contribute there; with PBEsol, 7 of the 8 CsSnBr₃ ones turn correct
  and 4 of the 5 KNbO₃ ones do not (§S5.3).*
- **MLIP relaxation moving off the soft-mode geometry** → both at-reference and at-relaxed
  geometries recorded; relaxation hiding an instability is itself reported.
- **Supercell / cell-size convergence** → SSCHA at 2×2×2. *Outcome: bcc Zr 2×2×2 against
  3×3×3, re-measured in the pinned environment, holds for MACE-MP-0 (no harmonic instability) and
  changes sign for MatterSim (converged at 300 K: +0.92 to −0.92 THz from both starts); the cells
  are not nested, so this is a change of q-set and not a convergence test (§S2.4). No convergence
  claim is made for the zone-boundary (R-point and X-point) systems, and their SSCHA calls are
  stated for the 2×2×2 cell only; the converged SrTiO₃ 100 K false-stable may be a finite-size
  effect (§S2.4). For the screen, the q-points searched are those commensurate with
  the force-constant supercell (2×2×2, and 6×6×6 for the bcc metals), and the screen was not
  re-run with larger force-constant supercells, so an instability at a q outside that set would
  be missed (approximation A2 of Table 1); the common force-constant supercell row of Table S14
  changes the frozen-cell normalisation, not the set of q searched. Cross-model comparisons are
  made at a fixed cell; absolute stabilisation temperatures are approximate.*
- **Ground-truth uncertainty** → transition temperatures approximate; scoring qualitative
  (correct side of the transition). The labels of the test systems record thermodynamic, often
  first-order, transitions, and the control labels come from published harmonic phonon
  stability, whereas the screen and SSCHA probe dynamic stability. The two coincide for a
  continuous soft-mode transition and can differ for a deep double well, which is where the
  local SSCHA criterion and the labels part company (§3.3).

## S5. First-principles reference along the soft-mode coordinates and on SSCHA samples

Settings are in §2.6; the analysis is `python scripts/dft_reference.py analyze` over the deposited
pw.x outputs (`results/revision/dft/`), and every number below is pinned in
`scripts/verify_claims.py`. All 981 calculations finished: 380 on the deciding and reference
paths, 533 on the other screened modes and 68 on SSCHA samples, 930 core-hours (three
BaTiO₃/CHGNet points were rerun with symmetry off after a symmetry error on the slightly
non-cubic CHGNet cell; `NOSYM_NOTE.txt`). The checks of §S5.3 add 323 (`scripts/dft_checks.py`,
`results/revision/dft_checks/`). The 246 fine-tuning single points in the same directory are not
counted here.

**S5.1 The screen on PBE energies (Referee 1.1).** For each of 120 ladder units of six systems
we compare the screen's call on the MLIP energies along the PBE-covered paths with the call on PBE
energies along the same paths (Table S19). The comparison changes the energy engine only: the mode
pattern, amplitudes, fit and solver are the same. PBE covers every mode the screen mapped
whose modulated cell has at most 12 atoms: 93 of 139 paths, 38 deciding or reference and 55
others. Of the 46 without PBE, 8 are symmetry copies of a computed path and 38 lie in cells of
18 to 216 atoms (19 CsSnBr₃, 13 MatterSim bcc Zr). The 55 other modes change none of the calls
below. On the covered paths the MLIP call reproduces
the full screen's call in 115 of 120 units (all 100 own-mode units; the five exceptions are ORB-v2
units scored on MatterSim's coordinate). Against the labels the MLIP calls are right in 81 units
and the PBE calls in 101: 23 corrected, 3 newly wrong, five systems improved and none worse
(without ORB-v2, 61 and 82 of 96). The corrections are of two kinds. Ten are bcc Zr, where
CHGNet, MACE-MP-0 and SevenNet-0 have no well along MatterSim's deciding coordinate and PBE has one
of 159–174 meV. Eight are the BaTiO₃ and KNbO₃ units at 300 K that four of the five models call
stable; along the coordinates that decide them, those models' wells are 0.32–0.76 of the PBE depth
(median 0.53), which is enough to move the screen's T* below 300 K. ZrO₂'s wells are similarly
shallow (0.45–0.61) without changing any call. The 16 errors that persist on PBE are KNbO₃ at
600 K (all five models; T_c 708 K, curves bracketed), CsSnBr₃ at 300 and 600 K (wells matching PBE
within 20 % for three models at the sampled amplitudes, the PBE minimum at the 0.45 Å edge of the
scan on every deciding path, and a 300 K label 8 K above the transition) and bcc Zr at 900 K.
They bound what the single-mode approximation, the PBE functional and the labels contribute;
§S5.3 splits the KNbO₃ and CsSnBr₃ ones further.

Along the other modes the picture is the same. The 26 BaTiO₃ and KNbO₃ paths of CHGNet,
MACE-MP-0, MatterSim and SevenNet-0 have wells 0.32–0.83 of PBE's (median 0.53). SrTiO₃'s Γ
modes have PBE wells of 5.6–8.9 meV at the MLIP lattices, which the models reproduce at 0.02–0.40
of the depth. On 11 CsSnBr₃ Γ and X modes CHGNet and MatterSim have wells of 1.4–13.8 meV where
PBE has none, and in bcc Zr MatterSim has a 206 meV well at q = (1/3, 1/3, 0) where PBE has none.
None of these changes a call.

Three limits apply. PBE is evaluated at each MLIP's relaxed lattice (within 0.06 % across models
for the oxide perovskites, 0.45 % for CsSnBr₃, where CHGNet's is the largest, and about 2 % in
volume for Zr; 0.1–0.8 % above the PBE lattice for BaTiO₃, KNbO₃ and CsSnBr₃, §S5.3) and along its
eigenvector, so each model is compared with a slightly different PBE surface. The PBE call covers
1 to 11 of the screen's modes per unit, so a PBE "stable" means stable on those modes only: 19
PBE-stable units have a mode in a larger cell without PBE, 16 of them correct calls such a mode
could overturn and 3 errors it could correct (BaTiO₃/ORB-v2 at 300 K, KNbO₃/ORB-v2 at 600 K,
bcc-Zr/MatterSim at 900 K). Twenty-eight units rest on a deciding or reference path whose
regenerated map differs from the cached one, and 40 on some covered path (Table S19). The counts are descriptive: six systems,
with the five models sharing each system's label, so no unit-level test is attached.

**S5.2 MLIP error on SSCHA-sampled configurations (Referee 1.2).** Table S20 scores all five
models against PBE on twelve configurations from each of four seed-study ensembles and on four
rattled near-equilibrium cells. At 50–100 K (0.09 Å root-mean-square displacement) the owner
model's force error relative to the PBE forces is 1.04, 0.78 and 0.74 times its near-equilibrium
value for BaTiO₃, ZrO₂ and bcc Zr; the absolute error grows 1.4–2.3-fold with the forces
themselves. For bcc Zr the ratio is uninformative, since the near-equilibrium PBE forces are only
0.07 eV Å⁻¹. For SrTiO₃ at 600 K (0.34 Å) the relative error rises 1.67-fold, to 0.19, energy
errors reach 15 meV per atom root-mean-square and 44 at most, and all five models fall at 0.12–0.20.
The low-temperature false-stables therefore show no sign of MLIP extrapolation; the
high-temperature SrTiO₃ ensemble does. These ensembles come from the production recipe, which did
not converge (§2.5), and twelve configurations per unit support a description, not a test.

**S5.3 Checks on the errors that persist on PBE (Referee 1.1).** BaTiO₃, KNbO₃ and CsSnBr₃ hold
13 of the 16 persistent errors and 15 of the 19 PBE errors. Three checks were run on them, with
the acceptance criterion fixed in `scripts/dft_checks.py` before any variant ran (Table S23; 323
calculations).

Convergence. On the MACE-MP-0 deciding path of each system the production settings were compared
with a k-spacing of 0.15 Å⁻¹, cutoffs 1.3 times higher, and both. The criterion, a depth within
5 % for every variant and the same call at every temperature, is met for BaTiO₃ and KNbO₃ (depth
changes of at most 1.7 %) and missed for CsSnBr₃, where the denser k-mesh lowers the depth by
6.8 %. By the criterion as fixed, the production settings are therefore not converged. The
failure is in the depth alone: every call on the three paths is unchanged at all five
temperatures under all three variants, and the CsSnBr₃ depth is the value at the 0.45 Å edge of
the scan, a lower bound on the well, in every variant. The other models' paths were not
recomputed.

Functional. The 23 deciding paths of the three systems were recomputed with PBEsol on the same
structures. PBEsol is applied through input_dft to the SSSP PBE pseudopotentials, not with the
SSSP PBEsol set, and at each MLIP's lattice, not at PBEsol's own, so it changes the functional at
fixed geometry and nothing else. It deepens the BaTiO₃ and KNbO₃ wells by 3–30 % and changes one
call there (KNbO₃/SevenNet-0 at 600 K, now unstable as labelled). It changes CsSnBr₃
qualitatively: the tilt wells fall to 0.32–0.42 of the PBE depth, the minimum moves inside the
scan (0.36 Å), and 8 of the 9 CsSnBr₃ PBE errors are called correctly, all four at 600 K among
them. Over the 60 units PBEsol corrects 9 of the 15 PBE errors and introduces none. Six remain:
KNbO₃ at 600 K for CHGNet, MACE-MP-0, MatterSim and ORB-v2, BaTiO₃/ORB-v2 at 300 K and
CsSnBr₃/ORB-v2 at 300 K.

Lattice. The cubic cells relaxed in PBE are 4.024 Å (BaTiO₃), 4.027 Å (KNbO₃) and 5.887 Å
(CsSnBr₃). The MLIP lattices the curves of §S5.1 use are larger by 0.26–0.31 %, 0.74–0.80 % and
0.12–0.57 %. The PBE force constants at that lattice and the E(Q) profile along the PBE
eigenvector were not computed, so whether the lattice or the eigenvector moves a call is not
tested.
<!-- PENDING-PL: pbe-lattice phases B and C (PBE force constants and E(Q) along the PBE eigenvector at the PBE lattice) not run; replace this paragraph and Table S23's lower-part note if they land. -->

Reading. KNbO₃ at 600 K is converged and persists under both functionals for four of the five
models, so among the errors tested it is the one left for the single-mode screen or the label.
The CsSnBr₃ calls are numerically stable but follow the functional: at 600 K, 308 K above the
transition, where the label is not in doubt, PBE condenses the tilt for four models' geometries
and PBEsol for none. The CsSnBr₃ errors therefore cannot be assigned to the screen. Which
functional describes CsSnBr₃ better is not settled here; three of the five models reproduce the
PBE wells within 20 % at the sampled amplitudes, which puts their wells at 2.4–3.1 times the
PBEsol depth.
