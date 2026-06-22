# Lessons

(Updated after corrections / surprises. Each entry: pattern → rule.)

- 2026-06-22 — STAGE-2.5 INTEGRITY PASS catches (manuscript-vs-ledger read-through). Two recurring
  failure modes, both from prose drifting away from the data after revisions:
  (1) ATTRIBUTION DRIFT. §3.1 lumped "MACE, CHGNet AND ORB soften the bcc Zr/Hf modes" — true for
  MACE/CHGNet (FS = Zr+Hf) but ORB actually catches Zr (-0.43) and its 2 false-stables are Hf +
  SrTiO3. Rule: when a sentence attributes the SAME mechanism to a group of models, re-list each
  model's actual failing units from the ledger — don't generalize the majority's story to all.
  (2) STAT CONFLATION. "Spearman rho=0.78 rising to 0.78 once ORB excluded" silently swapped two
  different statistics: full-set Spearman=0.78 / sign-agree=0.64; excl-ORB the SIGN AGREEMENT rises
  to 0.78 while Spearman DROPS to 0.63. Rule: every "X rises to Y when Z removed" must name WHICH
  metric and report the ones that move the wrong way too. Also: prose Tc's (403/676) had drifted off
  the registry (393/708) — numbers in prose must match the config they're scored against. And a
  "all results regenerate from ledger.parquet" claim was false for a print-to-log-only repro study.
- 2026-06-22 — INTEGRITY CATCH (self-review overturned a headline). The "finite-T ranking inverts
  the harmonic ranking, Spearman rho=-0.26" claim was a DENOMINATOR ARTIFACT: it compared harmonic
  accuracy *including* bcc (where chgnet/mace/orb lose accuracy on bcc Zr/Hf softening) against
  finite-T accuracy *excluding* bcc. On the matched non-bcc set the correlation is weakly POSITIVE
  (phi=0.11 per-unit, n.s. McNemar p=0.34; model-level rho=+0.15 to +0.65). Rule: never compare two
  rates computed on different system subsets — match the denominator first. And never headline a
  rank correlation across n=5 models (p~0.7, statistically void); use the per-(system,model) paired
  test (phi/McNemar) which has dozens of points. The honest H2 is "harmonic accuracy is necessary
  not sufficient; harmonic leaders are not finite-T leaders." Always re-derive a striking number on
  a matched set before it becomes a headline.

- 2026-06-22 — SSCHA CANNOT adjudicate deep displacive (ferroelectric perovskite) instabilities
  as a black box, and this is THE central methods finding. Diagnostic (scripts/sscha_v4_diag.py,
  cubic BaTiO3/mace/100K): harmonic soft mode -5.6 THz, but ForcePositiveDefinite init erases it
  (+2.88), the SCHA auxiliary dyn stays stuck there (+2.89 — at low T the narrow Gaussian width
  never samples the double well), and the include_v4=False free-energy Hessian just echoes the
  positive curvature → +2.87 = FALSE-STABLE. include_v4=True is the only term that could recover
  the instability but ran >18 min on a SINGLE unit without finishing (≈tens of hours × 140 units)
  and is float64-only (orb_v2 float32 blows up to -2e6 THz). Quantified: FE-perovskite displacive
  recall (T<=300, cubic definitively unstable) softmode 0.77 vs SSCHA 0.23. Rule: SSCHA-with-MLIP
  is a clean gold-standard ONLY where the cubic reference is a genuine local min that becomes
  metastable (bcc martensitic: 0 blowups, tracks softmode rho=0.78). For deep displacive wells
  the cheap single-mode softmode (a static double-well + 1D quantum SCHA) is MORE reliable AND
  cheaper. Don't trust a converged SSCHA freq without checking the aux dyn actually moved off the
  ForcePositiveDefinite start.

- 2026-06-22 — SSCHA minimizer hang = uncapped max_ka, not a real convergence problem. A noisy
  stochastic gradient (large soft cells e.g. 40-atom perovskites) makes one population's
  reweighting chase the gradient below the noise floor for 1000+ steps (CPU thrash, GPU at 0%)
  WITHOUT ever regenerating a fresh ensemble → looks hung, hits timeout. Fix: set minim.max_ka
  (e.g. 20) so each population is bounded and the relaxer resamples forces. Small cells (8-atom
  bcc) converge under the cap so they're unchanged. Symptom to recognise: GPU idle while a SSCHA
  "minimization step" counter climbs into the hundreds.

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

- 2026-06-21 — Rattled-start MD finite-T FAILED the SrTiO3 gate, for TWO reasons found by
  diagnostics. (1) BUG: a hand-rolled brute finite-difference supercell Hessian to extract the
  soft eigenvector returned a spurious "imaginary" mode whose E(Q) was steeply CONVEX
  (+178 meV at 0.1 A) — i.e. not the physical mode. Rule: get soft-mode eigenvectors from
  PHONOPY (run_qpoints/run_modulations at the commensurate q), which already gives the correct
  R-point -2.09 THz; do NOT hand-roll the supercell Hessian. (2) PHYSICS: along the *correct*
  phonopy R-mode, MACE's SrTiO3 double well is only ~3.5 meV/f.u. deep (min at Q0~0.15 A).
  k_B*50K = 4.3 meV > barrier, so the cubic phase is thermally accessible at 50 K and MD
  legitimately melts any single-domain seed (order parameter psi ~0.02 A, flat in T). Rule:
  an order-parameter MD cannot resolve shallow soft-mode transitions (SrTiO3-like, Tc low,
  meV barrier). Use the 1D soft-mode FREE ENERGY instead: map E(Q) along the phonopy
  eigenvector (cheap, ~9 single-points), fit V=aQ^2+bQ^4(+cQ^6), then self-consistent
  (quantum) phonon renormalization Omega_eff^2 = a + 3 b <Q^2>(T,Omega) -> stability when
  Omega_eff^2 turns positive. GPU-light and transparent; deep-well systems (bcc Ti/Zr/Hf) are
  easier and the same machinery handles them.

- 2026-06-21 — 1D quantum SCHA finite-T method WORKS (passed the SrTiO3 gate). Three pitfalls
  fixed along the way: (1) SOFTEST-MODE SELECTION: searching argmin over all commensurate
  q-points grabs the spurious imaginary ACOUSTIC branch at Gamma (acoustic-sum-rule artifact,
  -2.365 THz, convex E(Q) garbage) instead of the real R-point soft mode (-2.089, true double
  well). Rule: mask the 3 acoustic branches at q=Gamma before argmin (a genuine zone-centre
  FE soft mode is optical, band>=3, still found). (2) POTENTIAL FIT must be BOUNDED: the sextic
  V=aQ^2+bQ^4+cQ^6 lstsq over Q up to 0.45 A returned c<0 (unbounded -> SCHA free energy
  diverges to -inf, fake "stable"). Rule: if c<0 drop it and refit the quartic (fits these
  shallow soft wells to <0.1 meV). (3) SCHA SELF-CONSISTENCY: the damped fixed-point on sigma^2
  falls into a spurious runaway large-sigma root (sigma~0.5A, F~-22 eV). The residual
  g(s2)=sigma2_sc(K(s2))-s2 is monotone on the K>0 interval -> bracket + brentq for the unique
  physical root. UNITS TRAP: <Q^2>=hbar/(2 m omega) coth needs hbar in J*s (m in kg -> m^2),
  but x=hbar*omega/2kT and the ZP energy need hbar in eV*s. Result: SrTiO3 Q0 condenses ~0.14A
  below Tc and melts at ~150K; eff_freq hardens -2.6 -> +1.8 THz through Tc~100-150K (expt 105).
  CAVEAT for the paper: the collective supercell mode's barrier (27.7 meV/2x2x2) scales with
  cell size, so the predicted Tc is supercell-dependent and only order-of-magnitude — report it
  as a qualitative stability call, not a quantitative Tc.

- 2026-06-21 — softmode method hardening pass (after the SrTiO3 gate), four more fixes from
  validating across well shapes (SrTiO3 shallow / BaTiO3 narrow-deep / MgO control / bcc Ti-Zr-Hf):
  (a) GET_CALCULATOR MUST CACHE: a grid calls run_unit per (system,T) and each reloaded MACE
  from disk (~40-90s under GPU contention) -> the grid looked "hung". Module-level handle cache
  keyed on (name,device,kw); inference models are stateless so reuse is correct.
  (b) NARROW WELL FIT: BaTiO3's soft well is narrow (min ~0.07A) and the static E(Q) climbs to
  +130 eV up the repulsive wall; a global poly fit over the full Q range washed a real -60 meV
  well down to -16 meV -> SCHA false-stable. Fix: quadratic Q spacing (denser at small Q) AND
  fit ONLY the well+barrier window (dE <= min + 5x|depth|, >=4 points). 
  (c) SOFTEST-q SEARCH: must scan the rational q-grid with denominator 6 (R=3/6, M, bcc omega
  2/3=4/6, 1/3, Gamma) via EXACT run_qpoints, NOT phonopy run_mesh -- run_mesh both injects the
  Gamma acoustic-sum-rule artifact (-2.4 THz) as a spurious global min and skips exact zone-
  boundary points (its irreducible set had R only as 5/12 at -0.63 vs exact -2.09). Freeze the
  winning q into its MINIMAL commensurate cell (2x2x2 perovskite R, 2x2x1 bcc N, etc.).
  (d) SCHA empty-grid guard: a purely-soft centroid (no K>0 bound Gaussian) returns F=+inf so
  the free-energy min falls to a displaced/stable centroid instead of crashing on argmin([]).
  RESULT: SrTiO3/BaTiO3/MgO all correct at the ladder temps; MatterSim captures bcc Ti/Zr/Hf
  instabilities (correctly U at low T) while MACE misses them (correctly false-stable).
  KNOWN LIMITATION (document in paper): single-mode SCHA UNDERESTIMATES absolute Tc for deep,
  phonon-entropy-stabilised transitions (bcc metals predicted T*~300-600 K vs expt 1100-2000 K;
  ordering preserved hf>ti~zr). So the HEADLINE metric is the LOW-T (100-300 K) false-stable
  rate, which cleanly separates instability-capturing models (MatterSim/SevenNet) from missers
  (MACE/CHGNet/ORB); the predicted T* vs Tc is a secondary, caveated comparison.

- 2026-06-22 — SOUNDNESS BUG in the softmode q-search: searching a denominator-6 rational
  q-grid on 2x2x2 force constants is UNSOUND. NxNxN finite-displacement FCs give EXACT
  frequencies ONLY at the N^3 commensurate q (Gamma/X/M/R for 2x2x2); at other q phonopy merely
  interpolates and can invent spurious soft modes with FLAT (non-physical) wells -> false-stable.
  Symptom: mace/sevennet bcc picked q=(5/6,5/6,1/6) dim[6,6,6] harm~-2 but well=-0.0. Fix:
  search ONLY FC-commensurate q. To capture the bcc omega mode (2/3<111>) AND N (1/2), the FC
  cell must be 6x6x6 (den-6 exact) -- cheap for bcc (1-atom primitive, cubic -> ~1 displacement).
  COROLLARY: the harmonic LAYER (2x2x2 FCs on a 12^3 mesh) is itself partly interpolated for
  bcc -- sevennet's "harmonic" ti/zr instabilities (-0.9/-1.1) largely VANISH with exact 6x6x6
  FCs (+0.5/+0.3). So bcc needs converged FCs at every layer.

- 2026-06-22 — single-mode SCHA is INSUFFICIENT for bcc Ti/Zr/Hf (user chose to invest in full
  SSCHA). bcc is stabilised by the FULL multi-mode phonon entropy at very high T (expt Tc
  1100-2000K); the single soft mode melts by ~300K. Decision: implement multi-mode stochastic
  SSCHA (python-sscha + cellconstructor) for quantitative finite-T dynamic stability. INSTALL
  notes: needs apt gfortran + libblas/liblapack-dev (box had none); pip install meson
  meson-python ninja cython, then cellconstructor then python-sscha, per model env. The "julia
  extension not available" warning is harmless. API gotcha: CC.Phonons.load_phonopy is DISABLED
  (raises NotImplementedError); bridge via ASE phonons instead:
  ASEPhonons(prim,calc,supercell,delta).run()/read(acoustic=True) ->
  CC.Phonons.get_dyn_from_ase_phonons(aph) -> ForcePositiveDefinite()+Symmetrize() ->
  sscha.Ensemble.Ensemble(dyn,T,sc) -> Relax.SSCHA(minim, ase_calculator=calc, N_configs,
  max_pop) -> relax() -> ensemble.get_free_energy_hessian() -> DiagonalizeSupercell(); min freq
  <0 => dynamically unstable. (RY_TO_THZ = CC.Units.RY_TO_CM * 0.0299792.)

- 2026-06-22 — KEY PHYSICS for the bcc benchmark: bcc->hcp/omega in Ti/Zr/Hf is a MARTENSITIC,
  strain-coupled FIRST-ORDER transition, NOT a soft-mode condensation. So the dynamic-
  stabilization temperature T_dyn (where the SSCHA free-energy Hessian turns positive = bcc is a
  metastable phonon minimum) is genuinely MUCH lower than the thermodynamic Tc (1100-2000 K).
  Therefore using transition_T_K (thermodynamic) as the ground-truth label for *dynamic*
  stability of bcc is the WRONG comparison -- it made every method look "false-stable" on bcc.
  The correct bcc deliverable is the SSCHA dynamic-stabilization CURVE, not a Tc classification.
  RESULT (2x2x2 SSCHA, 5 MLIPs, bcc-Zr): ALL five anharmonically stabilize bcc-Zr down to <=50 K
  (T_dyn < 50 K everywhere), but the margin to the boundary discriminates and tracks each model's
  harmonic-instability depth: mattersim(-2.03 harm)~orb closest (+0.4 THz @50K), then sevennet/
  chgnet (+1.0-1.1), mace furthest/flat (+1.8, it barely sees the instability). Caveat: 2x2x2 is
  a small SSCHA cell (q-grid Gamma/X/M/R, finite-size) -> T_dyn values approximate, but the
  cross-model comparison at fixed cell is valid. 4x4x4+ SSCHA is convergence-slow (>7 min,
  step-collapse) so 2x2x2 was the tractable choice.

- 2026-06-21 — 5-model harmonic result (the real Layer-1 finding): models DISAGREE on which
  instabilities they capture, architecture-dependently. MatterSim & SevenNet reproduce every
  soft mode with large imaginary freqs (acc 1.00, 0 false-stable on 19 systems). MACE-MP-0 &
  CHGNet SOFTEN the bcc Zr/Hf soft modes to ~0 -> false-stable. ORB-v2 (direct, float32-only)
  softens broadly: SrTiO3 reads -0.0 (uniquely missed) and it falsely calls MgO unstable
  (-2.8). Rule: report per-system min_freq across models, not just binary rates — the
  "softening toward zero" is the physics, and float32-only direct models (ORB) need a caveat.
