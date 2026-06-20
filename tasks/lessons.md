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
