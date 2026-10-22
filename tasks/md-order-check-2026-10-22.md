# Exploratory check, written before running: does MACE-MP-0's own surface order at 100 K? (2026-10-22)

NOT part of the pre-registration of 2026-10-21. Written after its result (primary NOT_SUPPORTED:
the multi-mode SSCHA free energy of BaTiO₃ / MACE-MP-0 / 100 K is lowest at the cubic centroid, by
+46.0 ± 2.2 meV per 40 atoms against the screen's M displacement and +55.3 ± 1.3 meV against the
static Γ [111] minimum) and before any MD below has run. Frank approved the run on 2026-10-22.
Reported as exploratory wherever it is used.

## Question

The SSCHA result leaves two explanations for the false-stable call on this unit: the potential (at
this fixed cubic volume MACE's surface does not order at 100 K) or the SSCHA itself (the quantum
Gaussian treatment disorders a surface that does order). Classical MD on the same surface, in the
same fixed cubic cell, samples the exact classical distribution with no Gaussian assumption. It has
no zero-point motion, which can only add disorder, so the check is decisive in one direction only.

## Runs (fixed now)

MACE-MP-0 medium (mace-torch 0.3.15, float32, CPU), cubic BaTiO₃, the SSCHA's own 40-atom 2×2×2
cell (results/revision/r2_centroid/gate/pattern.npz: cell, r0, MACE lattice 4.03405 Å), cell held
fixed. Langevin NVT, 2 fs step, friction 0.01 fs⁻¹, centre of mass fixed, frames every 10 fs.

- A: 100 K from the cubic structure; 5 ps equilibration + 25 ps production.
- B: 100 K from the static Γ [111] minimum (r0 + 0.0765 Å × u_g); same lengths.
- C (positive control): 20 K from the static Γ [111] minimum; 2 ps + 10 ps.

## Observable and reading rule

Per Ti, the off-centring u_i(t) = r_Ti − centroid of its six O neighbours (minimum image), and its
Fourier components A_q(t) = (1/8) Σ_i u_i cos(2π q·n_i) over the eight q of the 2×2×2 cell (Γ, 3 X,
3 M, R). Per run and q, the Binder cumulant of the 3-vector, U_q = 1 − (3/5)⟨|A_q|⁴⟩/⟨|A_q|²⟩²,
which is 0 for a Gaussian centred at zero (disordered) and 2/5 for a fixed magnitude (ordered,
whether or not the sample hops between equivalent variants). The q read is the one with the
largest ⟨|A_q|²⟩.

- Ordered: U ≥ 0.30. Disordered: U ≤ 0.10. Otherwise inconclusive. Each production half is also
  read; a run whose halves disagree is reported as not stationary.
- C must read ordered, or A and B are not interpreted.
- A and B both disordered: MACE's surface does not order at 100 K in this cell even classically, so
  the SSCHA's stable call is consistent with the surface, and the false-stable is the potential's
  at this volume (and in a cubic cell with no strain), not the Gaussian treatment's.
- A and B both ordered: the surface orders classically; whether zero-point motion or the Gaussian
  ansatz removes the order in the SSCHA is not separated here (it would need path-integral MD).
- A disordered and B ordered: hysteresis within the run length; inconclusive.

Descriptive, outside the rule: the pooled distribution of the single-Ti off-centring components
(unimodal, or bimodal as in an order–disorder picture, which a single Gaussian cannot represent),
its excess kurtosis, and the distributions of the projections on the SSCHA's M and Γ patterns
beside the SSCHA's own widths at the cubic centroid (0.0215 and 0.0184 Å).

## Limits

One model, one temperature, 40 atoms, fixed cubic cell (no strain coupling), classical, float32
where the SSCHA used float64, 25 ps.
