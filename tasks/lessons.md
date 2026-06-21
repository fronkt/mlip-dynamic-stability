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

- 2026-06-21 — orb-models 0.7.0 broke the calculator API the code was written against:
  `ORBCalculator` moved to `orb_models.forcefield.inference.calculator`, and the pretrained
  loaders (`orb_v2`, `orb_v3_conservative_inf_omat`) now return a `(model, atoms_adapter)`
  TUPLE that must both be passed: `ORBCalculator(model, atoms_adapter, device=...)`. Rule:
  smoke-test every new backend on si_diamond before launching the grid; pin/record the pkg
  version in the ledger (already done via model_version).

- 2026-06-21 — Per-model env recipe that WORKS on the Blackwell box: `uv venv
  --system-site-packages --python /venv/main/bin/python /root/env-<model>`, then
  `pip install -e . <model-pkg> -c /root/torch_constraint.txt` where torch_constraint.txt
  pins `torch==2.12.0+cu130`. The constraint + --system-site-packages stops the model pkg
  (sevenn/orb/mattersim) from pulling a non-Blackwell torch off PyPI. Verify
  `torch.__version__` and `torch.cuda.is_available()` after each install.

- 2026-06-21 — 5-model harmonic result (the real Layer-1 finding): models DISAGREE on which
  instabilities they capture, architecture-dependently. MatterSim & SevenNet reproduce every
  soft mode with large imaginary freqs (acc 1.00, 0 false-stable on 19 systems). MACE-MP-0 &
  CHGNet SOFTEN the bcc Zr/Hf soft modes to ~0 -> false-stable. ORB-v2 (direct, float32-only)
  softens broadly: SrTiO3 reads -0.0 (uniquely missed) and it falsely calls MgO unstable
  (-2.8). Rule: report per-system min_freq across models, not just binary rates — the
  "softening toward zero" is the physics, and float32-only direct models (ORB) need a caveat.
