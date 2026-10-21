# Pre-registration: multi-mode SSCHA free energy along the condensing soft mode (2026-10-21)

Committed before any unit below has been run (no box is rented at the time of writing). Deviations go
in the "Deviations" section at the end, dated, never by editing the sections above.

## Purpose

The second-round referee asks whether the converged SSCHA's stable call on the ferroelectric
perovskites is a limitation of where the Hessian is evaluated (a local test at the high-symmetry
centroid, blind to a lower displaced minimum) or of the theory itself, and whether the fourth-order
term would change the sign. The manuscript so far argues the first from the single-mode screen and
says (Section 3.3) that relaxing the centroid from a displaced start "we did not do". This run does it
in the full multi-mode SSCHA, for one unit, and measures the fourth-order term at a converged matrix.

What is already known for the unit (from the deposited ledger, screen v4): BaTiO₃ / MACE-MP-0 / 100 K
condenses at the M point q = (½, 0, ½) (softest band, harmonic −5.404 THz, M_eff 238.0 amu,
V(Q) = −14.258 Q² + 1080.6 Q⁴ + 5314 Q⁶ eV per 2×1×2 cell). The screen's single-mode SCHA free energy
F(Q₀) − F(0) at 100 K is +0.47 meV at Q₀ = 0.025 Å and −11.0 meV at 0.075 Å (global minimum) per
2×1×2 cell, i.e. about −22 meV per 40-atom 2×2×2 cell; the curvature at Q₀ = 0 is +2.03 THz. The
converged 2×2×2 SSCHA Hessian of the cubic centroid is +1.98 ± 0.055 THz (grid, start A), and its
free-energy error is 1.8 × 10⁻⁵ Ry (≈ 0.25 meV) per 40-atom cell.

## Units (fixed now)

All arms: BaTiO₃ cubic, MACE-MP-0 medium (pinned env), 100 K, 2×2×2 supercell (40 atoms) treated as
a Γ-only cell, the grid's converged recipe (CONV_DEFAULTS of scripts/sscha_seed_study.py: 1000
configurations per population, ≤ 30 populations, ≤ 400 steps per population, real serial gradient
error, Kong–Liu guard), start A built from the cubic harmonic matrix. The displacement pattern is the
screen's own: the softest band at q = (½, 0, ½), phonopy modulation, max atomic component 1, so Q₀ is
in the screen's units.

1. **M scan, fixed centroid** (`minim_struct = False`): Q₀ = 0, 0.04, 0.075, 0.10 Å.
2. **Free centroid** (`minim_struct = True`): start at the M pattern with Q₀ = 0.075 Å; record the final
   centroid's projection on the M pattern and its free energy.
3. **Γ polar, fixed centroid**: the Γ softest (T₁u) pattern along [111] (the experimental
   low-temperature direction), at the amplitude of the static E(Q) minimum along that pattern,
   computed on the box before the run and logged.
4. **Fourth-order term**: `include_v4` True vs False on the same 512-configuration subsample of arm 1's
   Q₀ = 0 Hessian ensemble (paired; the round-trip check of `v4_child` applies).

Gates (checked before arms 1-3 are read): the selected M band reproduces −5.404 ± 0.02 THz from the
pinned harmonic force constants; the Q₀ = 0 arm's Hessian is within 3 SD of +1.98 THz. A failed gate is
reported and the arm it concerns is not interpreted.

## Primary outcome and decision rule

ΔF_M = F(Q₀ = 0.075) − F(Q₀ = 0) per 40-atom cell, σ = (σ₁² + σ₂²)^½ from the two runs' final
free-energy errors. Only arms that reach the converged stop (gradient/real error < 0.2 at KL ≥ 0.7)
enter a comparison.

- ΔF_M < −2σ: the multi-mode SSCHA free energy has a displaced minimum below the cubic centroid
  while the Hessian at the cubic centroid is positive. The local-versus-global explanation of the
  false-stable call is SUPPORTED for this unit.
- ΔF_M > +2σ and no scan point below F(0) − 2σ: no displaced minimum along M at 100 K. The stable call
  is what the multi-mode Gaussian theory predicts for this potential, not a local trap; the geometric
  explanation is NOT supported for this unit, and the text says the failure is in the SCHA treatment
  (mode coupling within a Gaussian ansatz) or the potential, not in the evaluation point.
- Otherwise: inconclusive, reported as such.

The free-centroid arm (2) and the Γ arm (3) are secondary and reported as measured. Arm 4: the
fourth-order term "does not change the sign" if the v4 and v2 lowest non-acoustic frequencies have
the same sign; the shift is reported with the subsample's bootstrap SD beside it.

No directional prediction is made beyond the single-mode value above: the other soft modes'
fluctuations (Γ, X, the other M arms) renormalise the M stiffness in the multi-mode theory and may
remove the displaced minimum. Both outcomes are reported, in the manuscript and the response letter.

## Not done (budget)

One unit, one temperature, one model, one seed. No KNbO₃, no 300 K, no start B. Stated wherever the
result is used.

## Deviations

(none yet)
