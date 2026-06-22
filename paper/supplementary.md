# Supplementary Information

*Companion to `manuscript.md`. All tables regenerate from `results/ledger.parquet` via
`mlip_dynstab/analysis.py`; figures via `scripts/make_figures.py`.*

## S1. Finite-T method development and discarded routes

The single-mode quantum-SCHA soft-mode screen (§2.4) was the fourth finite-T route we
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
well E(Q) along the softest commensurate mode plus an analytic single-mode quantum SCHA free
energy. Moving the anharmonicity into a clean static map (cached, T-independent) and the thermal
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
  well-plus-barrier window is fitted so a steep repulsive wall does not wash out a shallow well
  (e.g. BaTiO₃).
- **SCHA self-consistency.** The width is solved with a bracketed root finder to avoid a runaway
  large-σ spurious root.

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

`ForcePositiveDefinite` at initialisation removes the soft mode; at low temperature the
converged auxiliary matrix's narrow Gaussian width never samples the anharmonic double well, so
the `include_v4=False` free-energy Hessian merely reproduces the positive auxiliary curvature
→ false-stable. The fourth-order term is the only route that could recover the instability but is
not viable at grid scale (≈tens of hours per unit) and is numerically stable only in float64.

### S2.3 Reliability by family

| Family | n | numerical blow-ups (\|f\|>50 THz) | min freq (THz) | max freq (THz) |
|---|---|---|---|---|
| bcc | 75 | 0 | 0.02 | 2.12 |
| perovskite/fluorite | `[PENDING full grid]` | (perovskites: 6 blow-ups to −2×10⁶) | | |

Numerical blow-ups are concentrated in the float32 ORB-v2 runs on the deepest (FE perovskite)
wells. Cubic fluorites are numerically clean but still false-stabilised at low T by the §S2.2
mechanism `[PENDING confirmation on the full ZrO₂/HfO₂ grid]`.

## S3. Supplementary tables

- **Table S1** — full per-model harmonic confusion matrices (all 19 scored systems).
  `analysis.per_model_table(df, "harmonic")`.
- **Table S2** — per-model finite-T (softmode) false-stable rates over the full T-ladder, with
  and without bcc. `analysis.low_t_false_stable(df, exclude_bcc=...)`.
- **Table S3** — per-(system, model) predicted stabilisation temperature T* vs experimental
  transition temperature. `analysis.predicted_tstar(df)`. Note: the single-mode screen
  systematically *under*-estimates the absolute T* for entropy-stabilised bcc, so T* is reported
  as a qualitative ordering check, not a quantitative T_c prediction.
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
