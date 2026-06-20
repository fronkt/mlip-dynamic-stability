# Lessons

(Updated after corrections / surprises. Each entry: pattern → rule.)

- 2026-06-20 — The naive "benchmark MLIPs on harmonic phonon dynamical stability" is already
  published (npj 2025; PhononBench Dec 2025). Rule: the contribution must be the
  **finite-temperature/anharmonic** wedge; the harmonic layer exists only to validate the
  harness against published numbers, never framed as novel.

- 2026-06-20 — The hand-rolled "TDEP-lite" finite-T estimator (unconstrained least-squares
  fit of a full 3N×3N effective force-constant matrix from MD snapshots) is **ill-conditioned
  and produced unphysical −32 to −40 THz frequencies, ~constant in T**. Root cause: ~14k
  parameters from a few hundred correlated snapshots with negligible ridge → spurious huge
  eigenvalues. Rule: finite-T effective force constants MUST be symmetry-constrained. Use
  **hiPhive** (TDEP-style, symmetry-reduced cluster expansion) or **SSCHA** (gold standard),
  not a raw matrix fit. The 5 bad SrTiO3 tdep rows were purged; do not report them.
  Verification gate before trusting any finite-T number: on SrTiO3, the method must show the
  soft mode HARDENING with T (negative→positive across the ~105 K transition).

- 2026-06-20 — hiPhive (symmetry-constrained TDEP) fits cleanly (rmse 0.05-0.30 eV/Å), but
  **one-shot TDEP-from-MD starting at the perfect cubic cell does NOT resolve SrTiO3's
  soft-mode instability**: the soft optical mode reads +0.81 THz at 50 K and stays ~flat to
  600 K (harmonic 0 K was -1.83). Trajectory check: thermal RMS 0.068 Å @50K (no persistent
  distortion) — the short MD never leaves the cubic basin, so the cubic-symmetric fit reports
  "stable" at all T. Rule: for soft-mode dynamic stability use **SSCHA** (self-consistent) OR
  symmetry-broken / rattled-start sampling; plain one-shot TDEP under-detects instabilities.
  IMPLICATION: the cross-model HARMONIC comparison is the solid near-term deliverable; the
  finite-T core needs the SSCHA upgrade before its numbers are trustworthy.
