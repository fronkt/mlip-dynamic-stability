# Electronic Supplementary Information (ESI)

*Companion to `manuscript.md`. All tables regenerate from `results/ledger.parquet` via
`mlip_dynstab/analysis.py` and `scripts/build_esi_tables.py`; figures via
`scripts/make_figures.py`.*

*Citation convention: **sections** of this document are cited as §S1-§S4 and **tables** as
Table S1-Table S12. The two sequences are independent; a cross-reference to "Table S1" means the
table, not the section.*

## S1. Finite-T method development and discarded routes

The quantum-SCHA soft-mode screen (§2.4) was the fourth finite-T route we
implemented. The earlier three were built, tested against the SrTiO₃ validation gate, and
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
  **Window sensitivity (Table 1, A3 of the main text).** The production window is 5× the well
  depth (floor 60 meV). Refitting every cached E(Q) with the multiplier at 3× and 8× and
  re-solving the SCHA condensation call at 100 K and 300 K changes 4 of 756 mode-level calls
  (0.5%) at 3× and 10 of 756 (1.3%) at 8×, so no headline number depends on the window choice
  (e.g. BaTiO₃).
- **SCHA self-consistency.** The width is solved with a bracketed root finder to avoid a runaway
  large-σ spurious root.

### S1.2 Harmonic estimator reproducibility: the v1/v2 paired replicate

The harmonic layer was measured twice. Generations v1 and v2 run the *same algorithm at the same
settings* — 0.01 Å displacements, 2×2×2 supercells, 12×12×12 meshes — in independently pinned
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

**The spread must not be pooled.** The pooled max of 2.01 THz is entirely an ORB-v2 artifact and
overstates CHGNet and MatterSim — the two models that carry the §3.2 reordering — by roughly two
orders of magnitude. ORB-v2's two flipped calls are `ktao3_cubic`, which is already excluded from
the matched set as borderline, and `hf_bcc`, which moves from −0.0891 to −0.1937 THz (the same call
either way at the production tolerance, but a flip at tol = 0). This is consistent with ORB-v2's
float32 direct-force architecture, identified as the weakest model in §3.1 on independent grounds.

**This is a lower bound, not a full characterisation.** v1 and v2 share a code path, a displacement
amplitude and a seedless algorithm, so the pair probes environment and library nondeterminism only.
It does *not* probe sensitivity to the finite displacement amplitude itself, which is the axis a
reader familiar with κ_SRME's estimator noise will ask about. Closing that would require re-running
the 95 scored units on a displacement grid (0.005 / 0.01 / 0.02 / 0.03 Å). Note for anyone
attempting it: `disp` is hardcoded in `harmonic.py` and is *not* part of the unit hash, so it must
be added to the settings dict with a `METHOD_VERSION` bump or `has_unit()` will silently skip every
re-run unit.

### S1.3 Complete derivation of the single-mode quantum SCHA screen

Section 2.4 of the main text states the variational free energy and its self-consistency
condition. This section supplies the three elements a reader reproducing the screen needs and
the main text compresses: the origin and normalisation convention of the effective mass, the
stationarity step that yields the self-consistency condition, and an explicit account of what
the single-mode restriction neglects.

**Effective mass, and the normalisation convention.** A frozen mode is a rigid displacement
pattern: atom *i* in the commensurate supercell moves to **R**_i = **R**_i^(0) + Q **u**_i, where
**u**_i is the (real) Cartesian displacement of atom *i* in the pattern and Q is the scalar
amplitude that parametrises the distortion. The kinetic energy of that one-parameter motion is

$$T = \tfrac{1}{2}\sum_i m_i |\dot{\mathbf{R}}_i|^2
    = \tfrac{1}{2}\dot{Q}^2 \sum_i m_i |\mathbf{u}_i|^2
    \equiv \tfrac{1}{2} M \dot{Q}^2 ,
\qquad M = \sum_i m_i |\mathbf{u}_i|^2 ,$$

which is the definition used in the main text. Together with V(Q) from the fitted map this gives
the one-dimensional Hamiltonian H = P²/2M + V(Q).

M and Q are meaningful only once the normalisation of **u** is fixed, and the main text's phrase
"unit displacement pattern" is not specific enough. **The convention used throughout is that the
largest Cartesian component of any atom in the pattern equals 1 Å**, that is
max_i,α |u_i,α| = 1, which is phonopy's modulation output rescaled by its own maximum
(`finite_t._mode_pattern`). Q is therefore read directly as *the largest atomic displacement in
ångström*, which is why the reported order parameter Q₀ (for example 0.14 Å for the SrTiO₃
R-point tilt) is physically interpretable. Note that this differs from the unit-norm convention
Σ_i |**u**_i|² = 1 that "unit pattern" might suggest, and it differs again from the mass-weighted
eigenvector **e**_i = √m_i **u**_i returned by diagonalising the dynamical matrix.

Nothing observable depends on the choice. Rescaling **u** → λ**u** sends Q → Q/λ, M → λ²M, and
the fitted coefficients a → λ²a, b → λ⁴b, c → λ⁶c, so the curvature ratio ⟨V″⟩/M and hence every
frequency and every stability call is invariant; only the numerical value of Q₀ moves. We state
the convention because the reported Q₀ is convention-dependent even though the calls are not.

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

For the sextic V(Q) = aQ² + bQ⁴ + cQ⁶ the Gaussian moments give
⟨V″⟩ = 2a + 12b⟨Q²⟩ + 30c⟨Q⁴⟩ with ⟨Q²⟩ = Q₀² + σ² and ⟨Q⁴⟩ = Q₀⁴ + 6Q₀²σ² + 3σ⁴.

**Solving it.** Substituting Ω(σ²) back gives a scalar fixed-point equation g(σ²) = σ²_sc(σ²) −
σ² = 0, solved in σ² rather than Ω because the physical branch is easier to bracket there. A
damped iteration on this equation can run away to a large-σ spurious root, so the implementation
brackets instead. The search grid is geometric, σ² ∈ [10⁻⁶, 2] Å² over 240 points, restricted to
the sub-interval on which ⟨V″⟩(σ²) > 10⁻⁸ eV Å⁻² (below that no normalisable Gaussian trial
state exists, since Ω would be imaginary); the first sign change of g on that grid is refined by
Brent's method to a tolerance of 10⁻¹⁰. If the admissible sub-interval is empty, no bound
Gaussian exists at that centroid, and the centroid is rejected with 𝓕 = +∞ so the outer
minimisation moves to a displaced, bound centroid rather than reporting a spurious value. The
outer minimisation over Q₀ is a scan on a grid of 121 points spanning |Q₀| ≤ 0.6 Å.

**What the single-mode restriction neglects.** Writing the full Born–Oppenheimer surface in the
normal coordinates {Q_k} of the commensurate cell,

$$V(\{Q_k\}) = \sum_k V_k(Q_k) + \sum_{k<l}\lambda_{kl}Q_k^2Q_l^2 + \ldots ,$$

where the leading inter-mode term is biquadratic because the cubic cross terms Q_k²Q_l vanish by
symmetry whenever the two modes belong to different irreducible representations, which is the
case for every pair we screen. The screen evaluates each V_k(Q_k) with all other Q_l held at
zero, so it discards the λ_kl terms entirely. Within a mean-field reading of the neglected term,
mode *k* sees an effective quadratic coefficient

$$a_k^{\text{eff}} = a_k + \sum_{l\neq k}\lambda_{kl}\langle Q_l^2\rangle_T ,$$

so the sign of λ decides the direction of the error and the temperature dependence enters
through ⟨Q_l²⟩_T. Competing (λ > 0) coupling hardens mode *k* as other modes acquire amplitude,
so the isolated treatment over-predicts condensation; cooperative (λ < 0) coupling does the
opposite, and can produce a joint condensation that no single mode shows alone. This is why
Table 1 lists A1's expected bias as "either sign".

We do not have a quantitative bound on λ_kl, and obtaining one would require exactly the
multi-mode map the screen is designed to avoid; the honest statement is that the magnitude is
unmeasured. Two things can be said about its consequences. First, the effect is concentrated in
the *absolute* stabilisation temperature rather than in the low-temperature call: T* is where
a_k^eff crosses zero, so an error in a_k^eff translates directly into an error in T*, whereas at
T ≤ 300 K the ferroelectric wells are hundreds of meV deep and far from that crossing. This is
consistent with what we observe, namely that the screen's T ≤ 300 K calls are its reliable
output while its T* ordering fails for PbTiO₃ (§3.2), and it is why T* is reported as a
diagnostic (Table S6) and not as a prediction. Second, the multi-mode SSCHA cross-check does
retain these couplings, so on the family where it is trustworthy the agreement between the two
methods bounds the coupling's effect on the *call*: 0.78 sign agreement over 45 paired bcc units,
0.83 excluding ORB-v2 (§3.3).

### S1.4 Why the criterion is variational: comparison with an exact 1D quantum solution

Each deciding mode is stored as an effective mass and three polynomial coefficients, so the
one-dimensional Schrödinger problem for the *same* fitted potential can be solved to machine
precision by diagonalising the tridiagonal Hamiltonian on a grid and Boltzmann-averaging the
resulting states (`finite_t._solve_1d`, run over the deposited ledger by
`scripts/scha_vs_exact.py`; 228 modes, no solver failures, no new computation). The natural
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

| T (K) | exact: fraction bimodal | SCHA screen: fraction condensed |
|---|---|---|
| 100 | 0.9649 | 0.8596 |
| 300 | 0.9649 | 0.6667 |
| 600 | 0.9649 | 0.5614 |
| 900 | 0.9649 | 0.4912 |

**Table S12** The exact isolated-mode criterion against the variational one, over the same 228
fitted potentials.

That contrast is itself the argument for the variational criterion. The temperature dependence
that lets the screen bracket the SrTiO₃ transition at 105 K is supplied by the SCHA
self-consistency, in which the Gaussian width broadens with T and re-enters the averaged
potential, and not by the shape of the well. It also explains why all three discarded routes of
§S1 failed the SrTiO₃ gate: each read a density or an effective force constant rather than
minimising a free energy.

**The screen never manufactures a mode-level condensation.** Across all 228 modes there is not a
single case in which the screen condenses a mode whose exact thermal density is unimodal; the
disagreements are entirely in the other direction, and they concentrate in shallow wells
(median depth 27.9 meV, against 216 meV where the two agree). The screen's condensation calls
therefore always sit on a potential that genuinely possesses a displaced minimum. This is the
mode-level complement of the phase-level control result in §3.2, where the screen finds zero
imaginary commensurate modes on all 120 harmonically-stable control units.

Neither result speaks to the accuracy of the underlying potential-energy surface, which is a
separate question and is addressed by the experimental anchors of §2.4 and §3.2 rather than by
anything in this section.

## S2. SSCHA harness and the displacive-instability failure

### S2.1 Minimizer step cap

Multi-mode SSCHA on the 40-atom perovskite cells initially appeared to hang (CPU busy, GPU idle,
a per-population step counter climbing past 1000). The cause was an uncapped per-population
reweighting loop: with a noisy stochastic gradient on a large soft cell, one population chases
the gradient below its own stochastic-noise floor indefinitely instead of regenerating a fresh
ensemble. Capping the per-population steps (`minim.max_ka = 20`) forces ensemble regeneration;
SrTiO₃ then converged in 231 s / 27 steps (previously 1835 steps → timeout). Small cells (8-atom
bcc) converge under the cap and are unchanged.

### S2.2 Root-cause diagnostic (cubic BaTiO₃, MACE-MP-0, 100 K)

`scripts/sscha_v4_diag.py`:

**Table S2** SSCHA root-cause diagnostic, cubic BaTiO₃ / MACE-MP-0 / 100 K.

| Stage | min frequency (THz) |
|---|---|
| harmonic (ground truth) | −5.635 |
| after `ForcePositiveDefinite` | +2.882 |
| converged SCHA auxiliary dynamical matrix | +2.892 |
| free-energy Hessian, `include_v4=False` (production) | +2.872 |
| free-energy Hessian, `include_v4=True` | did not finish in >18 min (single unit) |

The auxiliary SCHA matrix is positive-definite by construction (a normalisable Gaussian trial
state requires it), so the +2.88 THz after `ForcePositiveDefinite` and the +2.89 THz at
convergence are not diagnostics of anything -- the auxiliary matrix cannot soften (main text,
refs 21, 22). The object that can detect the instability is the free-energy Hessian, and the
table shows the failure enters at its truncation: with the fourth-order term dropped (the bubble
approximation, the `python-sscha` default) the Hessian merely reproduces the positive auxiliary
curvature → false-stable. The fourth-order resummation is the theoretically prescribed remedy in
exactly this regime (ref 22) but is not viable at grid scale (≈tens of hours per unit) and is
numerically stable only in float64.

### S2.3 Reliability by family (complete grid)

**Table S3** SSCHA reliability by chemistry family over the complete grid.

| Family | n measured | failed units | numerical blow-ups (\|f\|>50 THz) | min freq (THz) | max freq (THz) |
|---|---|---|---|---|---|
| bcc (Ti/Zr/Hf) | 75 | 0 | 0 | 0.06 | 2.10 |
| fluorite (ZrO₂/HfO₂) | 40 | 0 | 0 | −24.8 | 3.33 |
| perovskite (oxide + halide) | 86 | 7 | 8 | −914 | 3.72 |

`analysis.sscha_reliability(df)`. bcc is the clean gold standard. All failures sit in the
perovskite family, on the deepest wells: seven units die at cellconstructor symmetry or ensemble
assertions before returning a number (six on PbTiO₃: ORB-v2 at every temperature, MatterSim at
600 and 900 K; one on CsSnI₃/SevenNet-0), and the eight numerical blow-ups span float32 (ORB-v2)
and float64 (SrTiO₃ high-T rows) alike, so precision alone does not account for them. In the
original unpinned measurement the same units blew up silently to −2×10⁶ THz instead of failing
loudly.
Cubic fluorites are numerically clean (no blow-ups, no failed units) but are nonetheless
**false-stabilised at low T** by the §S2.2 truncation: ZrO₂ and HfO₂ read ≈ +2 to +3.3 THz at
100 K in every model (cubic is the >2600 K phase, so the correct call is unstable — the screen's
condensation criterion calls all 20 fluorite units unstable, while its symmetric-point curvature
spans −7 to +27 THz, i.e. many of these condensations are of the first-order-like kind a
fixed-reference curvature cannot see, §2.4), and three of five models then *destabilise* with
temperature, the wrong trend. That a numerically well-behaved, shallower instability fails the
same way confirms the failure is methodological, not a numerical artifact of the deep perovskite
wells.

### S2.4 Stochastic reproducibility and finite-size convergence

`scripts/sscha_repro.py`, `scripts/run_revision_compute.sh`; finite-size runs in
`results/convergence_study.parquet` (the per-seed reproducibility frequencies print to the run
log rather than to a ledger).

- **Reproducibility.** bcc-Zr / MACE-MP-0 / 100 K SSCHA over 4 seeds: +1.798 ± 0.001 THz — the
  cross-model bcc margins exceed the stochastic noise by orders of magnitude. The same unit
  re-measured in the pinned environment with the phonopy-built initialiser returns +1.80 THz at
  every ladder temperature, so the number also survives an independent environment.
- **Finite size.** bcc-Zr SSCHA stability call holds 2×2×2 → 3×3×3 (+1.80 → +1.56 THz @100 K).
  The soft-mode rows of the convergence study are superseded (single-mode selection, pre-§2.4
  correction) and are retained for audit only. SrTiO₃'s R-point
  (½,½,½) instability is commensurate only with even cells, so the 2×2×2↔3×3×3 test is invalid for
  it (a 4×4×4 / ~320-atom SSCHA test is future work); the §3.3 false-stable is independent of this
  because it occurs for the always-present Γ mode of BaTiO₃.

## S3. Supplementary tables

The tables below are generated from the deposited ledger by `scripts/build_esi_tables.py`; the
function that produces each is named in its caption so any number can be re-derived
independently. Every rate is given as a count over its denominator with a Wilson score interval,
following the convention adopted throughout this revision.

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

**Table S6** Predicted stabilisation temperature T* (the lowest ladder temperature at which the high-symmetry phase is called stable) against the experimental transition temperature, per system and model. T* is a **diagnostic, not a prediction**: a single-mode treatment is not expected to reproduce an absolute T_c, the screen systematically under-estimates T* for the entropy-stabilised bcc metals, and it fails to order PbTiO₃ against the other perovskite anchors (§3.2). `analysis.predicted_tstar(df)`.

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

**Table S7** Composition of the two analysis sets, which the previous version of this ESI did not state. Both rest on the **same 15 systems**, so they share their clustering structure: the 60 guardrail units are 15 systems × 4 temperatures, and the 75 matched pairs are 15 systems × 5 models. In the guardrail set the five model votes are already collapsed into each unit, so the models are not an independent axis there and the clustering unit is the system. Systems whose identifier contains `bcc` are dropped, which also removes the superionic `agi_bcc` — retained deliberately for comparability with the published numbers and stated in §3.2.

| Analysis set | n units | n systems | Temperature axis | Model axis | Balanced |
|---|---|---|---|---|---|
| n = 60 (H3 guardrail, §3.4) | 60 | 15 | 4 (100/300/600/900 K) | collapsed into each unit | yes |
| n = 75 (H2 matched set, §3.2) | 75 | 15 | 1 per analysis (each T analysed separately) | 5 (an explicit axis) | yes |

The 15 systems are: `batio3_cubic`, `c_diamond`, `ceo2_cubic`, `cspbi3_cubic`, `cssnbr3_cubic`, `cssni3_cubic`, `cu_fcc`, `hfo2_cubic`, `knbo3_cubic`, `mgo_rocksalt`, `nacl_rocksalt`, `pbtio3_cubic`, `si_diamond`, `srtio3_cubic`, `zro2_cubic`.

**Table S8** The ensemble-disagreement guardrail set in full: every (system, temperature) unit behind the H3 analysis, with the cross-model frequency spread, the stable-vote fraction, and whether the majority consensus was correct. This is the n = 60 set. `analysis.h3_ensemble_guardrail(df, "softmode")`.

| System | T (K) | n models | Freq. std (THz) | Stable-vote frac. | Vote | Consensus | Ground truth | Consensus correct |
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

**Table S9** Every rate that ORB-v2 could plausibly drive, reported with and without it. ORB-v2 is float32 with non-conservative forces, which is an architecture confound for finite-difference force constants rather than a model property of interest, and it accounts for the MgO false-unstable, the ≈ −35 THz Ti/Hf outliers and most of the SSCHA blow-ups. Two entries deserve attention: the frequency rank correlation on bcc is ≈ 0.00 once ORB-v2 is removed, not merely smaller, which is why call agreement rather than magnitude correlation is the cross-validation statistic we report; and the FE-perovskite recall contrast narrows. See Table S10 for the paired tests.

| Quantity | Model set | Value |
|---|---|---|
| bcc screen-vs-SSCHA sign agreement | all five models | 0.778 over 45 pairs |
| bcc screen-vs-SSCHA Spearman ρ (magnitudes) | all five models | +0.113 |
| FE-perovskite recall, softmode | all five models | 16/30 = 0.533 [0.361, 0.698] |
| FE-perovskite recall, sscha | all five models | 5/27 = 0.185 [0.082, 0.367] |
| bcc screen-vs-SSCHA sign agreement | excluding ORB-v2 | 0.833 over 36 pairs |
| bcc screen-vs-SSCHA Spearman ρ (magnitudes) | excluding ORB-v2 | -0.003 |
| FE-perovskite recall, softmode | excluding ORB-v2 | 12/24 = 0.500 [0.314, 0.686] |
| FE-perovskite recall, sscha | excluding ORB-v2 | 5/23 = 0.217 [0.097, 0.419] |

**Table S10** The central screen-versus-SSCHA contrast tested **as a paired comparison**, which is what the design supports: both methods are evaluated on the same (system, model, temperature) units, so comparing their two marginal Wilson intervals would ignore the pairing. Counts are of discordant pairs; the test is an exact two-sided McNemar over them. The result to read honestly is the middle pair of rows: on the ferroelectric oxides **alone**, the contrast is significant with ORB-v2 included and not significant without it. The claim therefore rests on the combined displacive set, where the cubic fluorites — numerically clean, zero blow-ups, and essentially unaffected by ORB-v2 — carry it. `scripts/stats_hardening.py`.

| System set | Model set | n paired | Screen right, SSCHA wrong | SSCHA right, screen wrong | Exact p |
|---|---|---|---|---|---|
| fe oxide | all models | 27 | 14 | 3 | 0.013 |
| fe oxide | excl orb v2 | 23 | 10 | 3 | 0.092 |
| fluorite | all models | 20 | 19 | 0 | 0 |
| fluorite | excl orb v2 | 16 | 16 | 0 | 3e-05 |
| displacive combined | all models | 47 | 33 | 3 | 0 |
| displacive combined | excl orb v2 | 39 | 26 | 3 | 2e-05 |

**Table S11** SSCHA numerical quality per family and model. The sampling configuration is identical for all 201 units and is therefore stated once here rather than tabulated: 256 configurations per population, a cap of 8 populations, a dedicated 512-configuration ensemble for the free-energy Hessian, 2560 samples in total. The stopping criterion is `python-sscha`'s automatic stochastic relaxation under the per-population step cap `minim.max_ka = 20` (§S2.1), with the Hessian evaluated on a fresh ensemble at the converged auxiliary matrix.

Two quantities are tabulated from the six lowest recorded Hessian frequencies. *Acoustic zeros resolved* counts units in which all three translational zeros (|ω| < 0.001 THz) appear within that window, and the residual column gives the largest of their magnitudes over those units, which bounds the numerical noise on a quantity known analytically to be zero. *Swamped* counts the opposite case: units with **no** recorded mode near zero, meaning at least six modes lie below the acoustic branches. Swamping is not itself an error — a deeply unstable phase genuinely has many imaginary modes, and the reported minimum frequency excludes the acoustic branches from the full spectrum rather than from this window — but it separates the families sharply, and it marks the units on which the free-energy Hessian is furthest from the regime its bubble truncation is valid in (§3.3).

**What the harness did not retain**, and what would therefore need a re-run to supply: the per-iteration free-energy gradient history, and a per-unit uncertainty on the Hessian eigenvalues. The uncertainty probe that does exist is the independent-seed study of §S2.4, which this revision extends beyond bcc-Zr at Referee 1's request.

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

<!-- END GENERATED TABLES -->

## S4. Threats to validity (pre-registered, with outcomes)

- **Finite-displacement noise near Γ** → acoustic-sum-rule handling + swept imaginary tolerance
  (default −0.1 THz). *Outcome: stability calls are robust to the tolerance for all but the
  borderline KTaO₃.*
- **Single-mode vs multi-mode** → SSCHA cross-check. *Outcome: the two methods agree on the
  stability **call** for bcc (sign agreement 0.78 over 45 paired units; 0.83 excluding ORB-v2),
  which is the cross-validation statistic we report. Their **magnitudes** are not rank-correlated
  (Spearman ρ = 0.11, and ≈ 0.00 excluding ORB-v2), as expected for two different observables away
  from the sign change. Divergence on the perovskites is traced to an SSCHA failure mode, not a
  screen failure (§3.3).*
- **MLIP relaxation moving off the soft-mode geometry** → both at-reference and at-relaxed
  geometries recorded; relaxation hiding an instability is itself reported.
- **Supercell / cell-size convergence** → SSCHA at 2×2×2 (finite size); cross-model comparison at
  fixed cell is valid, absolute T_dyn approximate.
- **Ground-truth uncertainty** → transition temperatures approximate; scoring qualitative
  (correct side of the transition).
