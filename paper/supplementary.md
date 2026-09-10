# Electronic Supplementary Information (ESI)

*Companion to `manuscript.md`. All tables regenerate from `results/ledger.parquet` via
`mlip_dynstab/analysis.py`; figures via `scripts/make_figures.py`.*

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

- **Table S1** — full per-model harmonic confusion matrices (all 19 scored systems).
  `analysis.per_model_table(df, "harmonic")`.
- **Table S2** — per-model finite-T (softmode) false-stable rates over the full T-ladder, with
  and without bcc. `analysis.low_t_false_stable(df, exclude_bcc=...)`.
- **Table S3** — per-(system, model) predicted stabilisation temperature T* vs experimental
  transition temperature. `analysis.predicted_tstar(df)`. Note: the screen
  systematically *under*-estimates the absolute T* for entropy-stabilised bcc, and it fails to
  order PbTiO₃ against the other perovskite anchors (§3.2), so T* is reported as a diagnostic
  rather than a quantitative or ordinal T_c prediction.
- **Table S4** — per-(system, T) ensemble-disagreement guardrail table (H3).
  `analysis.h3_ensemble_guardrail(df, "softmode")`.

## S4. Threats to validity (pre-registered, with outcomes)

- **Finite-displacement noise near Γ** → acoustic-sum-rule handling + swept imaginary tolerance
  (default −0.1 THz). *Outcome: stability calls are robust to the tolerance for all but the
  borderline KTaO₃.*
- **Single-mode vs multi-mode** → SSCHA cross-check. *Outcome: agreement on bcc (ρ=0.78);
  divergence on perovskites traced to an SSCHA failure mode, not a screen failure (§3.3).*
- **MLIP relaxation moving off the soft-mode geometry** → both at-reference and at-relaxed
  geometries recorded; relaxation hiding an instability is itself reported.
- **Supercell / cell-size convergence** → SSCHA at 2×2×2 (finite size); cross-model comparison at
  fixed cell is valid, absolute T_dyn approximate.
- **Ground-truth uncertainty** → transition temperatures approximate; scoring qualitative
  (correct side of the transition).
