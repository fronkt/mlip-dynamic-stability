# Revised claims ledger (2026-09-27) — the single source for manuscript, ESI and letter

Every sentence the revision writes about a result must be one of these claims, with these numbers,
or a direct consequence. Sources: `results/stats_hardening.json` (keys named below),
`results/screen_sensitivity.json` / `.md`, `results/estimator_noise.json`, the ledger via
`analysis.load_canonical()`. `scripts/verify_claims.py` pins each number (56 PASS at da7cf29).
Compute results still running on the box are marked PENDING and get HTML-comment markers in the
text (`<!-- PENDING-C1 -->` etc.) so Phase 4 can find them.

## 0. The paper in one paragraph (what the revision now argues)

Five foundation MLIPs, 20 systems, three layers. (1) Harmonic layer: the models divide on
which instabilities they reproduce. (2) A cheap single-mode quantum SCHA screen: harmonically correct
units that the screen mis-calls at finite temperature exist and outnumber the reverse in unit counts
(17 vs 4 at 300 K), so a harmonic benchmark does not certify finite-temperature behaviour — but the
asymmetry is not significant at the system level (clustered p = 0.15), and most of it sits in three
systems where the screen mis-calls the phase for ≥4 of 5 models, i.e. it largely reflects the
screen's own approximation rather than model-specific PES error. (3) A default SSCHA cross-check
(free-energy Hessian at the symmetric reference, bubble level) calls deep displacive wells stable
against unstable labels; on the same MLIP energies the screen's own symmetric-point curvature does the
same on 52/57 of those units, while the screen's free-energy comparison finds the displaced minimum.
The default SSCHA criterion answers a local question (is the symmetric phase a local free-energy
minimum?) and cannot certify stability against condensation into a deeper displaced minimum, which is
what the thermodynamic labels record. Whether an MLIP error on thermally sampled configurations also
contributes is tested against PBE (PENDING-C3b). (4) Ensemble vote split is a suggestive guardrail,
not a robust one: AUC 0.762 with all five models, 0.628 without ORB-v2 (bootstrap CI spans 0.5).

## 1. Title (approved by Frank 2026-09-26)

"Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation machine-learning
interatomic potential for finite-temperature dynamic stability". Existence claim; survives clustering.

## 2. Harmonic layer (§3.1)

- Accuracy on 19 scored systems (`harmonic_per_model`): MatterSim 19/19 [0.832, 1.000], SevenNet-0
  19/19 [0.832, 1.000], MACE-MP-0 17/19 [0.686, 0.971], ORB-v2 17/19 [0.686, 0.971], CHGNet 15/19
  [0.567, 0.915]. Harmonic false-stable: MACE-MP-0 2/13, CHGNet 2/13, ORB-v2 1/13, others 0/13.
- MACE-MP-0 and CHGNet read −0.000 THz on bcc Zr and Hf (no instability). bcc-Zr ORB-v2 harmonic is
  **−0.47 THz** (canonical v2; −0.43 was the superseded v1 value). ORB-v2 MgO −1.07 THz (was −2.8).
- Tolerance: CHGNet uniquely lowest only for tol ∈ [0.05, 0.20] THz; the CHGNet < MatterSim ordering
  holds at every tol in [0.05, 0.50]; at tol = 0 all models collapse (degenerate).
- Displacement amplitude (Table S13, all five models, 0.005/0.02/0.03 vs 0.01 Å): MACE-MP-0,
  MatterSim, SevenNet-0 change **no** call; CHGNet changes calls only on controls (CeO₂, Cu, NaCl),
  accuracy 14/19–17/19; ORB-v2 changes calls on test systems too (HfO₂, SrTiO₃, CsSnI₃, Ti, Hf, MgO),
  accuracy 16/19–17/19. ORB-v2 is the only model with directly predicted (non-conservative) forces.
- **Precision fact:** as run, CHGNet, SevenNet-0, MatterSim AND ORB-v2 all return float32 forces;
  only MACE-MP-0 is float64 (calculators.py sets it). Never attribute a failure to "float32" alone.
- Harmonic replicate noise (ESI §S1.2): CHGNet max |Δ| 0.0076 THz, MatterSim 0.00057, zero flips;
  ORB-v2 2.01 THz and 2 flips. The v1→v2 hf_bcc ORB-v2 flip: v1 −0.089 THz was STABLE at the 0.1 THz
  tolerance, v2 −0.194 is unstable (supplementary.md currently inverts this).
- Fig. 2 plots the SCREEN's 100 K symmetric-point curvature, not harmonic frequencies; its colourbar
  label is inverted (blue = positive). Recaption; do not cite it as harmonic evidence.

## 3. Finite-T screen and H2 (§3.2)

- Scored finite-T accuracy (T ≤ 300 K, 15 systems, 30 units/model, `finite_t_per_model`): SevenNet-0
  27/30 [0.744, 0.965], CHGNet 26/30 [0.703, 0.947], MACE-MP-0 25/30, MatterSim 25/30
  [0.664, 0.927], ORB-v2 24/30 [0.627, 0.905]. **No ranking claim**: intervals overlap; 24 vs 25 is
  one unit. Delete "ORB-v2 remains the weakest finite-temperature screener" and "SevenNet-0 leads
  both layers"; say the design does not separate the models at finite T.
- Matched-set harmonic (15 systems): MACE-MP-0, MatterSim, SevenNet-0 15/15; CHGNet and ORB-v2 13/15.
  CHGNet is 13/15 = 0.867 harmonic and 26/30 = 0.867 finite-T on matched systems (the flagship
  "worst harmonic, second-best finite-T" illustration is withdrawn; say so once, no "the reordering").
- **H2 transfer asymmetry** (`h2_clustered`), harmonically-correct & finite-T-wrong (b) vs reverse (c),
  matched non-bcc non-borderline set, 15 systems, 75 pairs (60 ex-ORB):

  | T | b v c | unit McNemar p (companion) | system-clustered exact p | ex-ORB b v c, clustered p |
  |---|---|---|---|---|
  | 100 K | 5 v 3 | 0.727 | 0.844 | 3 v 2, 1.0 |
  | 300 K | 17 v 4 | 0.007 | **0.152** | 14 v 2, 0.156 |
  | 600 K | 23 v 4 | 0.0003 | **0.066** | 20 v 2, 0.055 |
  | 900 K | 11 v 4 | 0.118 | 0.414 | 10 v 2, 0.25 |

  Wording: "large and one-directional in unit counts, not significant once units are clustered by
  system". NEVER "significant", "confirmed", "established", "non-predictive". 100 K: "not resolved"
  (never "absent").
- **Shared screen error** (descriptive breakdown, selected on outcome — never presented as a test):
  at 300 K, 13 of the 17 b-units sit in BaTiO₃ (4), KNbO₃ (4) and CsSnBr₃ (5), systems the screen
  mis-calls for ≥4 of 5 models (BaTiO₃/KNbO₃: the screen's T* under-estimates; CsSnBr₃ is 8 K above
  its 292 K T_c). Without them 4 v 4. At 600 K five systems hold 22 of 23. So the counterexamples
  largely measure the single-mode screen's own approximation (A1), shared across models, not
  model-specific PES error. PENDING-C3a (PBE-backed screen on BaTiO₃ etc.) decides which.
- The reviewed φ = +0.11 and McNemar p = 0.34 were the **100 K** values (7 v 3; the reviewed text did
  not say so). Now at 100 K: φ +0.149, McNemar 0.727 (5 v 3). On the reviewed data φ was already
  negative at 300 K and above (−0.114, 14 v 4, McNemar 0.031 at 300 K). Never write that the
  re-measurement removed the positive correlation. Current 300 K: φ −0.129, McNemar 0.007, clustered
  φ permutation p 0.31 (estimator_noise.json). The association is not significant at any T.

## 4. Screen definition and sensitivity (§2.4, ESI §S1.3–S1.4, new Table S14) — Referee 1.5

- Q is the amplitude of the largest Cartesian component of the unit pattern u (max |u_iα| = 1),
  in Å; M = Σ m_i|u_i|² per the mode's MINIMAL commensurate cell (1, 2, 4 or 8 f.u. in perovskites);
  E(Q) is sampled on 10 quadratic points to Q = 0.45 Å; fit window max(60 meV, 5|E_min|); sextic,
  refit quartic if c < 0. Condensation threshold Q0 > 0.0075 Å (1.5 steps of the 0.6 Å, 121-point
  centroid scan); the brentq width root falls back to the grid-nearest value on 4.6% of evaluations,
  never at a deciding argmin. Even parity V(−Q) = V(Q) is assumed (Q ≥ 0 sampling); unchecked for 24
  bcc modes with 3q ≡ G.
- Table S14 (`results/screen_sensitivity.md`), unit-level flips at T ≤ 300 K:
  fit window (3×/5×/8× × 30/60/120 meV): 0 unit flips (69% of modes have an identical kept point set
  across multipliers at 60 meV — say so); sampling range Q ≤ 0.25–0.40 Å: ≤ 1 unit flip (8 over all T
  at 0.25 Å over CsPbI₃ 2, CsSnBr₃ 3, CsSnI₃ 2, Ti 1); threshold 0.5–3 steps and box 1.2 Å: 0 flips.
  **Frozen-cell normalisation is outcome-determining**: FE recall 16/30 (production minimal cell) →
  28/30 (common FC supercell), 26/30 (doubled), 5/30 (per formula unit); SrTiO₃ gate 3/5 → 3/5, 2/5,
  0/5. Accuracy cannot choose among them (T ≤ 300 K: 127/150, 139/150, 136/150, 105/150; all-T:
  243/300, 252/300, 249/300, 225/300). State this as a limitation of the single-mode screen: its T*
  and FE recall depend on the frozen-cell convention; the production choice is the minimal cell in
  which the mode is a single commensurate distortion.
- Mode–mode coupling: no quantitative bound is available; the earlier "hundreds of meV" premise is
  false (deciding-mode well depths at T ≤ 300 K: BaTiO₃ 28–43, KNbO₃ 25–59, PbTiO₃ 0–110, SrTiO₃
  0–27 meV per minimal cell). Split T1u triplets exist (BaTiO₃/CHGNet), so the "one irrep" claim goes.
  18 Ti/ORB-v2 modes and 1 PbTiO₃/ORB-v2 mode have fitted wells with no sampled well (fit artefacts,
  34.6–798.9 meV for Ti/ORB-v2 and 0.2 meV for PbTiO₃/ORB-v2); no unit call depends on them — disclose.

## 5. SSCHA (§2.5, §3.3, §3.5)

- §2.5 facts: initialiser = phonopy full force constants at **0.03 Å** displacement (not ASE),
  ForcePositiveDefinite + Symmetrize; root2 representation; min_step_dyn 0.5; meaningful_factor 1e-4
  (the convergence threshold); max_ka = 20 steps, **cumulative over all populations** in python-sscha
  1.6.1 (SchaMinimizer.py:1370; its default is −1, no cap) — NOT per population (integrator 09-27); up to max_pop = 8 populations of N = 256 configurations (2,048 max) then a dedicated
  512-configuration ensemble for the free-energy Hessian (include_v4 = False). The number of
  populations used and the per-unit convergence flag were not recorded in the production grid —
  say so; the new seed study records them (PENDING-C1). SSCHA here inherits the MLIP PES (say it).
- bcc agreement (`bcc_agreement`): screen-vs-SSCHA **curvature-sign** agreement 35/45 = 0.78
  [0.637, 0.875] (ex-ORB 30/36 = 0.83); **stability-call** agreement 31/45 = 0.69 [0.543, 0.805]
  (ex-ORB 25/36 = 0.69). **Integrator 09-27: curvature-sign agreement is withdrawn everywhere** — for
  a single even mode the screen's symmetric-point curvature = MΩ² > 0 by construction; all 10 bcc
  sign disagreements are numerical (6 width-solver fallback, 1 Ti/ORB fit artefact, 3 Hf/ORB with no
  screened mode; `scripts/curvature_identity_check.py`). Report call agreement only. 12 pairs agree trivially (MACE-MP-0/CHGNet on Zr/Hf, no harmonic
  instability); non-trivial: 23/33 sign, 19/33 call; MatterSim agrees in 3/9. Frequency Spearman
  0.113 (ex-ORB −0.003), descriptive only. Kill "clean gold standard", "tracks", "call agreement 0.78".
  The earlier bcc-Zr four-seed and 3×3×3 tests used MACE-MP-0 Zr, which has no harmonic instability
  (PENDING-C1: MatterSim Zr seeds; PENDING-C5: MatterSim/MACE 2×2×2 vs 3×3×3 re-measured).
  Several bcc SSCHA curves soften with T (ti/MACE 1.728 → 1.375 THz, 50 → 600 K) — note it.
- Displacive units: FE recall screen 16/30 vs SSCHA 5/27 (1 blow-up); fluorite 20/20 vs 1/20
  (ex-ORB 16/16 vs 0/16); CsSnI₃ SSCHA 7/8 correctly unstable at T ≤ 300 K. Paired screen-vs-SSCHA
  on the combined displacive set: 33 v 3 units, **system-clustered p = 0.125** (4 of 5 systems favour
  the screen; floor 0.0625); ex-ORB 26 v 3, clustered 0.125; FE oxides alone clustered p 0.5. Never
  quote the unit-level p (0.013, 0.092, 2e-5, 2e-7) as evidence.
- **Integrator 09-27: the 52/57 below is positive BY CONSTRUCTION (F''(0) = MΩ²), so it is not
  evidence.** The informative number is the screen's call: unstable on 46/57 [0.687, 0.889] (ex-ORB
  39/50). And the seed study shows the production SSCHA relaxations never converged (converged =
  false, one population moving the matrix, 19 steps; max_ka is cumulative in python-sscha 1.6.1), so
  the local-criterion account is 'consistent with', not 'therefore'; C1c decides.
- **Mechanism (`criterion_blindness`)**: of the 57 non-bcc units where SSCHA calls stable against an
  unstable label, the screen's own symmetric-point curvature is positive on 52/57 [0.811, 0.962]
  (ex-ORB 47/50), and on 41/57 it is positive while the screen's free-energy comparison says the phase
  condenses (BaTiO₃/MACE-MP-0 100 K: screen curvature +2.03 THz, SSCHA +2.87 THz, screen call
  unstable). Interpretation: the default SSCHA criterion is local (Hessian at the symmetric reference);
  for deep double wells the single-mode SCHA free energy keeps a local minimum at Q = 0 while a
  displaced minimum is lower (first-order-like), so any fixed-reference curvature criterion reports
  "stable". This is the global-vs-local distinction Referee 1 raised (R1.1b). It is consistent with,
  and more specific than, the bubble-truncation account of refs 21–22; it does NOT rest on Table S2.
- Table S2 (June, pre-audit, not re-measured): harmonic −5.635 → ForcePositiveDefinite start +2.88 →
  auxiliary +2.892 → bubble Hessian +2.872 THz; the include_v4 = True row never finished (> 18 min on
  one unit vs ~200 s for v4 = False). It does NOT isolate "only the truncation order changes". Replace
  every such sentence (manuscript 552-553 etc.) and "≈ tens of hours per unit" with the observed
  bound. Resolve the §3.3 internal contradiction in favour of: a fixed-reference Hessian (at any
  order) tests local stability; v4 is untested here (PENDING-C1b may add a number).
- **High-T SSCHA false-unstables (`sscha_high_t`)**, not previously discussed: non-bcc SSCHA
  false-unstable 0/0, 5/5, 10/13, 15/19 at 100/300/600/900 K (all models; ex-ORB 0, 4/4, 8/11,
  12/16); SrTiO₃ is false-unstable in all 14 above-T_c units, −3.4 to −914 THz (ex-ORB 11/11, to
  −60.5 THz), against harmonic R-mode values −0.8 to −2.1 THz; 14 of 23 units stable at 100 K turn
  negative by 600–900 K (ex-ORB 10/19). Instability growing with thermal amplitude on a fixed PES is
  the opposite of entropy stabilisation and is what an MLIP extrapolating on large-amplitude
  configurations (R1.2) would produce; sampling instability is the other candidate. PENDING-C3b.
- Blow-ups (|f| > 50 THz): 8 = ORB-v2 3 (SrTiO₃ 300/600/900 K), MatterSim 3 (PbTiO₃ 300 K, CsSnI₃
  600/900 K), CHGNet 1 (SrTiO₃ 900 K), SevenNet-0 1 (PbTiO₃ 900 K). Failed units (no row): 7 =
  ORB-v2 PbTiO₃ ×4, MatterSim PbTiO₃ 600/900 K, SevenNet-0 CsSnI₃ 600 K (a halide — "across the FE
  grid" is wrong). The "silent −2×10⁶ THz" value belongs to SrTiO₃/ORB-v2/600 K (v1), not to the
  failed units; the v1 values of the failed units were −0.3 to −3653 THz.
- Zone-boundary convergence (R3.2): no convergence claim for R-point (SrTiO₃) or X-point (fluorite)
  systems. Within-cell fluorite control: in the same 2×2×2 cell the harmonic layer sees the
  instability (−3.8 to −10.6 THz) that SSCHA at 100 K calls stable (+1.9 to +3.3 THz) → the fluorite
  false-stable is not a missing-q artefact (a finite-size argument, nothing more). SrTiO₃: SSCHA never
  false-stabilises it, so missing-q cannot apply; report its high-T divergence instead.

## 6. Guardrail (§3.4) — Referee 2.2 and 3.5

- Vote split (`orb_split_s3.guardrail`): all models split-vote error 8/14 vs unanimous 4/46,
  AUC 0.762, clustered permutation p = 0.0035 (10,000 permutations; exact enumeration 0.0030), cluster-bootstrap CI
  [0.590, 0.934]; **ex-ORB 4/8 vs 8/52, AUC 0.628, p = 0.047 (10,000 permutations; exact 0.044), CI [0.438, 0.839]
  spanning 0.5**; 3 tied 2–2 units broken to "stable" (2 wrong). Frequency-std AUC 0.361
  (ex-ORB 0.299). Wording: "suggestive, not robust to removing ORB-v2". Naive vs clustered p differ
  by a factor of ~12 (0.0003 vs 0.0035), not four.
- The frequency spread is a PES-level proxy partly derived from forces (mode patterns come from
  force constants; E(Q) is energy-only) — not "exactly the force-level disagreement".
- Force-level ensemble uncertainty (R2.2), DONE (`results/revision/force_spread/summary.json`, ESI
  Table S18): primary 0.681 [0.416, 0.908], clustered p 0.104; ex-ORB 0.684 [0.413, 0.916]; verdict
  "not shown". Report ALL secondaries (a)–(g): committees 0.717 [0.535, 0.902], 0.726 [0.516, 0.901];
  (d) normalised 0.578 [0.335, 0.795] / 0.576; (e) LOO pooled 0.654 [0.425, 0.832] / 0.689 ex-ORB,
  per model 0.542–0.761; (f) own-call MACE 0.722 [0.564, 0.873] p 0.034, MatterSim 0.686
  [0.473, 0.873]; (g) 43 units, 4 of 12 errors, 0.314 [0.146, 0.529]. 5 of 19 secondaries have lower
  bound > 0.5 (b, c, f-MACE, e-MACE ex-ORB, e-SevenNet ex-ORB at 0.502); unadjusted, same configs,
  (g) restriction not applied to them. Pre-registration: written before any config; amended once
  after a CPU smoke test, before any production config.
- H3's pre-registered comparator (a single model's self-reported signal) was not run — say so.
- Fine-tuning (R2.1): the paper tests models as shipped (the generative-CSP screening regime with no
  target data). Cite verified refs from `tasks/refs-2026-09-26.md` (F1 Deng npj 11, 9; F5 Radova
  npj 11, 237 — fine-tuned MACE-MP reproduces PBE phonons of bcc/α/ω-Ti; F2 Small 22, e00071; F3 =
  ref 20). State which findings fine-tuning could change (per-model screen accuracy, H1/H2 inputs)
  and which it could not (the local-criterion SSCHA result holds on any PES with a deep double well).

## 7. Words and claims that must not appear anywhere (manuscript, ESI, letter, captions, README)

"gold standard" / "clean gold standard"; "tracks" (for screen vs SSCHA); "validated"; "methodological
trap"; "systematically false-stabilises"; "call agreement 0.78"; "significant(ly)" for H2 or the
screen-vs-SSCHA contrast; "confirm(s) H2"; "non-predictive"; "absent at 100 K"; "the reordering";
"ORB-v2 (remains) the weakest"; "SevenNet-0 leads"; "only the truncation order changes"; "≈ tens of
hours per unit"; "float32 blow-ups" / float64 attribution; "hundreds of meV"; "factor of four in p";
"clean inversion"; "disagree sharply"; "harmonic leaders ≠ finite-T leaders"; "−35 THz outliers";
"this revision" / "previous version" inside the manuscript or ESI (the paper must stand alone; the
letter carries the history).

## 8. PENDING compute (box 52872517) and where it lands

- C1 seeds + diagnostics (BaTiO₃/MACE 100 K, ZrO₂/MACE 100 K, Zr/MatterSim 50 K, SrTiO₃/MACE 600 K)
  → §3.5, §S2.4, new ESI table; closes R1.4. C1b include_v4 → §3.3 one sentence either way.
- C2 force spread → §3.4, §4, ESI table; closes R2.2.
- C3a PBE along soft-mode coordinates (SrTiO₃ R, BaTiO₃, KNbO₃, CsSnBr₃, bcc-Zr, ZrO₂) + PBE-backed
  screen calls → new §3.x paragraph + ESI section; closes R1.1 and decides the H2 confound.
- C3b PBE forces/energies on SSCHA-sampled configurations → §3.3; closes R1.2.
- C5 bcc 2×2×2 vs 3×3×3 re-measured → §3.5.
Report each whichever way it falls.

## 9. Integrator corrections, 2026-09-27 (after the round-2 referee simulations)

- H2 count depends on the frozen-cell convention (`scripts/h2_by_convention.py` →
  `results/h2_by_convention.json`): 300 K b v c = 17 v 4 minimal (production), 7 v 4 common FC
  supercell (p 0.89), 10 v 4 doubled (0.50), 11 v 4 ×8 (0.42), 24 v 4 per f.u. (0.063). At 100 K
  b ≤ c in three conventions. Two of 20 convention×T combinations have clustered p < 0.05 (per f.u.
  100 K 0.045; ×8 900 K 0.031) — not evidence after twenty looks. Always qualify 17 v 4 with the
  production convention.
- At 300 K without BaTiO₃/KNbO₃/CsSnBr₃ (4 v 4): b = PbTiO₃ ×3 (MACE, MatterSim, ORB) + CsSnI₃/ORB;
  c = CeO₂/CHGNet, NaCl/CHGNet, MgO/ORB (interpolated-mesh false-unstables, A2) + SrTiO₃/ORB.
- Fine-tuning (R2.1): do not say fine-tuning "could not change the SSCHA result" — it can change
  which units are deep wells, not the locality of the criterion. Do not say there is "no target
  data to fine-tune with" — broad phonon fine-tunes exist (ref 41). New ref 43 = F4 (Hänseroth et
  al., JPCL 2026, 17, 3152; no phonons tested).
- Degenerate eigenspace: the screened direction is arbitrary (MACE BaTiO₃ Γ: production 20.9 meV,
  53 amu vs 16.8 meV, 41 amu along the LAPACK direction); disclosed in ESI §S1.3.
- bcc 2×2×2 vs 3×3×3 cells are not nested (N even cells only, ω multiples of 3 only).
- Add to §7 banned list: "curvature-sign agreement" as evidence; "max_ka per population";
  "the re-measurement removed the positive correlation"; "no target data to fine-tune with".


## 10. C3a / C3b results, analysed 2026-10-03 (`dft_reference.py analyze` at e7eac67; adversarially verified)

Outputs: `results/revision/dft/{summary.json,c3a_unit_calls.csv,c3a_paths.csv,c3b_units.csv,...}`.
448/448 pw.x outputs used, 0 flags; 38 PBE paths × 10 Q points (the 533 `ax_*` jobs were never run
and feed nothing). PBE curves sit at EACH MLIP's own relaxed lattice (0.1–0.3 % spread for the
perovskites; Zr ~2 % in volume) — lattice and functional error cannot be separated here.

**R1.1 (C3a) — like-for-like is PBE vs `mlip_same_paths_stable`, NOT vs the ledger call.**
- Baseline is **81/120, not 78**: the MLIP's call on the PBE-covered paths reproduces the ledger in
  115/120; all 100 own-mode rows reproduce, the 5 misses are ORB-v2 reference-coordinate rows.
- Swap the energy engine only (same modes, same Q, same `_fit_double_well`/`_solve_scha`, same T
  grid): **81/120 → 101/120** (23 corrected, 3 newly wrong: BaTiO₃/ORB 300 K, Zr/MatterSim 900 K,
  CsSnBr₃/ORB 600 K); ex-ORB-v2 61/96 → 82/96. By system: BaTiO₃ 16→19, CsSnBr₃ 9→11,
  KNbO₃ 11→15, SrTiO₃ 18→20, bcc-Zr 7→16, ZrO₂ 20→20 — 5 improve, 1 tie, 0 worsen.
- 10 of the 23 corrections are bcc-Zr reference-coordinate rows (MACE 4, CHGNet 3, SevenNet 3):
  those models' screens find no instability; PBE along **MatterSim's deciding coordinate
  q = (⅓,⅔,0)** does. Do NOT call it "the N point". Without Zr: 74 → 85 of 100. Own-mode rows
  75 → 84 of 100. Strict subset (cache-matching paths whose deciding mode has PBE, n = 72): 54 → 60.
- 16 errors persist on PBE: KNbO₃ 600 K ×5 (clean: T_c 708 K, curves bracketed — PBE + the
  single-mode screen put T_c below 600 K); CsSnBr₃ 300/600 K ×8 (**all 8 CsSnBr₃ PBE curves have
  their minimum at the scan edge Q = 0.45 Å**, wells unbracketed; 300 K is 8 K above the 292 K
  label); bcc-Zr 900 K ×3 (4 of 5 wrong on PBE counting the new MatterSim error). Attribute these to
  the screen, the PBE functional/lattice, or the label threshold — NOT to the screen alone.
- PBE-backed "stable" = stable on the 1–2 PBE-covered modes, not on the full screen; unstable
  calls are the safer direction. BaTiO₃/CHGNet 300 K's correction rests on the q(½½0) path whose
  i00/i01 points are nosym reruns — fragile, say so if quoted.
- Statistics: 6 systems, models share PES families → descriptive counts only. No McNemar on 23 v 3.

**R1.2 (C3b) — relative force RMSE (RMSE / RMS PBE force), baseline → SSCHA configs:**
BaTiO₃/MACE 100 K 0.100 → 0.104; ZrO₂/MACE 100 K 0.225 → 0.175; Zr/MatterSim 50 K 0.596 → 0.439
(u_rms 0.09–0.095 Å; absolute RMSE 1.4–2.3× larger because forces are larger);
**SrTiO₃/MACE 600 K 0.113 → 0.190 (1.7×), u_rms 0.34 Å, owner energy error RMS 15, max 44
meV/atom.** Zr's relative error is a ratio on a 0.07 eV/Å denominator — uninformative, say so.
4 sets × 12 SSCHA configs (+4 baseline). Not "roughly doubles".

Allowed manuscript wording (R1.1): "With the energy engine swapped from each MLIP to PBE on
identical modes and Q points, the screen's call matched experiment in 101/120 ladder units, against
81/120 for the same MLIPs on those paths (23 corrected, 3 newly wrong; 5 of 6 systems improved, none
worsened). Ten corrections are bcc-Zr units along MatterSim's deciding coordinate. The 16 errors
that persist with PBE (KNbO₃ 600 K; CsSnBr₃ 300/600 K, wells unbracketed at the scan edge; bcc-Zr
900 K) reflect the single-mode screen, the PBE functional and lattice, or the label threshold, not
force-engine error alone. Counts are descriptive, from six systems, with PBE evaluated at each
MLIP's relaxed lattice."


## 11. E1 / E2 / E5 results, analysed 2026-10-05 (committed d1e7c50; pinned in `scripts/verify_claims.py`, 95/95; adversarially re-checked against the raw outputs)

Supersedes the sentence in section 10 that the 533 `ax_*` jobs "were never run and feed nothing": they ran (box 2, 10-05), are analysed, and change no unit call. The ESI tables S19, S23 and S24 are generated (`scripts/build_esi_tables.py`; S23/S24 read the raw files through `scripts/ft_results.py`, and verify_claims recomputes the same quantities independently).

**E1 (R1.1) — `dft_reference.py analyze` over 1,227 pw.x (a_ 380, b_ 68, ax_ 533, ft_ 246), 0 flags.**
- Every job `ok`, SCF converged, `pw.in.ran == pw.in`, structure mismatch < 1e-4 Å.
- PBE now covers **93 of the 139** screened modes (38 before): 33 deciding, 5 reference, 55 extra-mode;
  20 of 30 (system, model) units fully covered (11 before); 40 of 120 ladder units rest on a
  cache-mismatched path (28 before; 22 of the 55 new paths mismatch).
- **0 of 150** (system, model, T) PBE-backed calls change when the 55 extra modes are added; every
  headline number of section 10 (81 → 101, 23/3, per-system, ex-ORB) is unchanged. Extra modes: own-model
  well median 0.516 of PBE (n = 42 with PBE well > 1 meV; deciding paths 0.602); none of the 55 PBE minima is at
  the scan edge (the 8 edge minima are all deciding CsSnBr₃ paths).
- Pins that had assumed 38 paths now filter to role decide/ref and a new block pins the 93.

**E2 (R1.1) — `analyze-checks` over 323 pw.x (cv_ 90, xs_ 230, pl_ vc-relax 3), 0 flags.**
- **Convergence: NOT converged by the pre-set criterion.** BaTiO₃ and KNbO₃ pass all three variants
  (depth ≤ 1.7 %, calls identical). **CsSnBr₃ fails the 5 % depth criterion on the two k-spacing variants
  (−6.82 %, −6.86 %; cutoff alone −0.04 %)**, no call changes anywhere (9/9 pairs), and its PBE minimum is at the
  scan edge. Do NOT write "the PBE wells are converged" for CsSnBr₃.
- **PBEsol (PBE pseudopotentials, MLIP lattices, unrelaxed): 60 ladder units; MLIP errors 24, PBE 15,
  PBEsol 6; 9 PBE errors fixed (8 of CsSnBr₃'s 9, KNbO₃/SevenNet-0 at 600 K), 0 new.** Remaining: KNbO₃ 600 K ×4
  (CHGNet, MACE, MatterSim, ORB), BaTiO₃/ORB 300 K, CsSnBr₃/ORB 300 K. CsSnBr₃ PBEsol/PBE depth 0.317–0.422
  (minimum 0.356 Å, off the edge, all 8); BaTiO₃/KNbO₃ 1.027–1.304. **So the CsSnBr₃ persistent errors are PBE's,
  not evidence about the screen; KNbO₃ 600 K is the unambiguous screen/label error.** The JSON key
  `persistent_errors` in dft_checks/summary.json lists ALL 15 PBE errors (including ones PBEsol fixes); use
  `pbesol_error`. PBEsol is not the SSSP PBEsol set and the lattice is not PBEsol's.
- **Lattice (phase A only):** PBE a = 4.0237 (BaTiO₃), 4.0269 (KNbO₃), 5.8871 Å (CsSnBr₃); MLIPs larger by
  0.26–0.31 %, 0.74–0.80 %, 0.12–0.57 %. **Phases B (PBE force constants) and C (profiles) were NOT run** (need a
  QE box); "lattice vs eigenvector" remains untested. Say so.

**E5 (R2.1) — fine-tuning trial, pre-registered (033b3d8; D1 contact redraw 0.6, D2 30 epochs, D3 details).**
Raw labels verified from the raw eval files (not summary.json); 32/32 runs, none mock; 120 labels per model
(108 train, 12 valid).
- **P1 well depth (band 0.8–1.2): MACE-MP-0 SUPPORTED** (median 0.448 → 1.005; replicate medians 1.071, 1.002,
  1.029; per-path 0.923–1.142). **CHGNet UNRESOLVED** (0.527 → 1.038 pooled; replicate-0 median 1.332; 3 paths,
  0.921–1.670). Negative control CsSnBr₃: MACE 1.475–1.728 (base 0.92), CHGNet 0.354–2.461 (base 2.17).
- **P2 screen calls, 10 registered cells: 'corrected' at 300 K REFUTED in all 4** (BaTiO₃, KNbO₃ × MACE, CHGNet; all 3
  replicates stable vs unstable label; MACE min-eff moves ≤ 0.107 THz). 'Persist': **5 supported** (CsSnBr₃ 300 K ×2,
  CsSnBr₃ 600 K MACE, KNbO₃ 600 K ×2), **1 unresolved** (CsSnBr₃ 600 K CHGNet: U S U). All 24 cells/model: MACE 11
  unchanged / 1 changed / 0 unresolved; CHGNet 9 / 0 / 3. **Unregistered: MACE CsSnBr₃ 900 K is a NEW error** (stable →
  unstable, 3/3); CHGNet KNbO₃ 100 K (U S U) and CsSnBr₃ 900 K (U S S) unresolved. **CHGNet replicate 1 has no
  imaginary phonon on KNbO₃ (+2.07 THz) and calls it stable at every T.**
- **P3 SUPPORTED** (one replicate, one system): converged SSCHA, start A, seed 0, replicate-0 fine-tuned MACE, BaTiO₃ 100 K:
  +1.735 THz (start +3.129), converged (6 populations, 343 steps), stable vs unstable label. Relaxed a = 4.016 Å (base 4.034,
  PBE 4.024). The base-model ledger value +2.868 is production-recipe and NOT a like-for-like comparison.
- **S2: MACE 6/6 controls unchanged; CHGNet 2 unchanged, 4 unresolved** (5 of 18 replicate calls differ: Si and Cu
  false-unstable in replicate 0; NaCl ×2 and CeO₂ ×1 corrected).
- **S1** (23 deciding paths, 230 pts; window force RMSE meV/Å): MACE BaTiO₃ 31.7 → 10.4/3.7/12.1; KNbO₃ 89.6 →
  13.9/5.9/13.9; CsSnBr₃ 1.3 → 3.9/3.9/2.7 (WORSE); CHGNet KNbO₃ 103.2 → 21.0/24.4/16.1; CsSnBr₃ 7.4 → 10.4/4.7/4.8.
- **THE P1/P2 GAP (key).** P1 is measured on the base model's held-out PBE geometries; the registered P2 pipeline relaxes the
  fine-tuned model and re-derives path and E(Q) map. The screen's rule on the fine-tuned models' held-out-path energies (stored
  in the c3a eval files; NOT registered, post hoc) gives the PBE-backed call in 12/12 replicate calls of the 4 corrected cells
  (48/48 over all BaTiO₃/KNbO₃ ladder cells). On the pipeline's own maps the MACE deciding well is BaTiO₃ 43.7/37.0/39.2 meV
  (base 42.6; PBE 73.5 on the base path), KNbO₃ 35.1/34.6/39.5 (base 27.9; PBE 87.8). PBE was not evaluated along the
  fine-tuned models' own paths → cause (lattice, eigenvector, force constants) NOT established.
- Departures to state: P1/S1 on the 23 deciding paths, not the registered full held-out set (extra-mode PBE finished after the
  GPU box was destroyed); configs generated under mace-torch 0.3.15, training/eval under 0.3.16 (base re-run reproduces the
  ledger in 24/24 cells). Weights gitignored (WEIGHTS.sha256).

Allowed wording: "Fine-tuning toward PBE brought MACE-MP-0's wells to 1.00 of the PBE depth on the held-out deciding paths
(CHGNet: unresolved), and the screen's rule applied there gives the PBE call, but re-running the screen on the fine-tuned
models corrected none of the four BaTiO₃ and KNbO₃ 300 K cases in any of three replicates (prediction refuted); why the two
differ is not established."

Banned: "fine-tuning fixes the softening and therefore the mis-calls"; "the P2 refutation shows the wells are not the cause"
(the held-out-path call says the opposite on those coordinates); "P3 shows the criterion, not the surface, decides" without the
one-replicate / unverified-own-path-wells caveat; "CsSnBr₃ persists on first principles" (it persists on PBE only); dropping the
CHGNet S2 / unresolved cells or MACE's CsSnBr₃ 900 K new error.
