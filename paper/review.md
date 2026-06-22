# Referee report — *Finite-temperature dynamic stability is a blind spot of foundation MLIPs*

*Stage 3 internal peer review, conducted against `manuscript.md` + `supplementary.md` at commit
5dc4315. Reviewer stance: a methods-minded referee for RSC Digital Discovery / npj Computational
Materials. Every quantitative claim was checked against `results/ledger.parquet`.*

## Recommendation

**Major revision.** The central thesis is timely, the framing is sharp, and the SSCHA cautionary
result is a genuine contribution. But three of the headline quantitative claims are currently
under-supported (one is statistically void as stated), the Methods do not match the data that was
actually run, and the primary screen is validated on a single system while being used as the
arbiter that declares the gold-standard method "wrong." None of these is fatal; all are fixable
with analysis already within reach of the existing ledger.

## Summary of submission

The paper argues that foundation MLIPs are benchmarked only for harmonic (0 K) dynamic stability
whereas dynamic stability is a finite-T property, and stress-tests five MLIPs on a curated
harmonically-unstable/thermally-stabilised set. It introduces a cheap single-mode quantum-SCHA
"soft-mode free energy" screen, cross-validates against multi-mode SSCHA, and reports: (H1) a
sharp harmonic model split; (H2) that finite-T accuracy is uncorrelated with — even inverts —
harmonic accuracy; (H3) that inter-model vote disagreement flags unreliable calls; and a
cautionary result that MLIP-driven SSCHA false-stabilises deep displacive instabilities.

## Significance

The gap is real and well-motivated against the two 2025 reference works. The most novel and
defensible result is the **SSCHA cautionary finding** (§3.3) — that the community's escalation
path "if unsure, run SSCHA" fails on exactly the displacive regime that dominates generative-CSP
output. That alone, properly scoped, is a publishable methods contribution. The cheap screen is
a useful secondary contribution if its validation is strengthened.

## Major concerns

**M1. The H2 "rank inversion" is not statistically supported as stated.** The headline number —
"Spearman ρ = −0.26 between harmonic and finite-T accuracy" — is computed across **n = 5
models**. With n = 5 this correlation is indistinguishable from zero (two-sided p ≈ 0.7); it
cannot carry the weight of "harmonic accuracy carries *no* predictive information." Worse, the
per-model finite-T rates differ by **one or two** mis-classified units (CHGNet and MACE both at
"0.062" = 1 false-stable of 16; the gap to SevenNet at 0.188 is two units), so the ranking is
fragile to a single system's label. *Required:* recast H2 on the **per-(system, model) paired**
data, where there are dozens of points — does harmonic correctness predict finite-T correctness
per unit (McNemar / φ coefficient, with CIs)? The `h2_harmonic_predictiveness` function already
computes φ; report it with a confidence interval and a bootstrap over systems. Demote the
5-model Spearman to an illustrative aside or drop it. The *qualitative* inversion (CHGNet
worst-harmonic→best-finite-T) can stay if accompanied by the per-unit statistic.

**M2. The primary screen is validated on a single system (SrTiO₃) yet is used to declare SSCHA
wrong.** The logic of §3.3 is: SSCHA disagrees with the screen on perovskites, the screen agrees
with "definitively unstable" physics, therefore SSCHA is at fault. This is persuasive for the
deep FE cases, but the screen itself rests on one validation gate (§2.4). The brief promised
"SSCHA cross-check on ≥3 textbook cases incl. SrTiO₃"; the manuscript should validate the screen
against experiment on **multiple** anchors before it is trusted as arbiter — e.g. the predicted
stabilisation ordering vs known T_c across SrTiO₃ (105 K), the FE perovskites (403/676/763 K),
and at least one case where the screen is *expected* to struggle (it under-estimates entropy-
stabilised T*, already noted for bcc). Table S3 (`predicted_tstar`) has the raw material; promote
a multi-anchor validation into the main text. Without it, M2 leaves the central comparison
partly circular.

**M3. Methods do not match the data actually run.** §2.1 lists the perovskite set as "SrTiO₃,
BaTiO₃, PbTiO₃, KNbO₃, CsSnI₃" and the controls, but the ledger contains **20 systems** including
**three** halide perovskites (CsPbI₃, CsSnBr₃, CsSnI₃) and the superionic **α-AgI** (`agi_bcc`),
which the Methods omit entirely. Either the extra systems were used (then describe them, and state
where they enter the rates) or they were excluded (then say so and why). As written, a reader
cannot reconstruct the "19 scored systems." This also affects the §3.2 table denominators.

**M4. The cautionary claim must be scoped to the *implementation*, not SSCHA in principle.** The
false-stable behaviour is induced by a specific (and standard) recipe: `ForcePositiveDefinite`
initialisation + 2×2×2 finite cell + `include_v4=False` + a narrow low-T auxiliary width. The
diagnostic (S2.2) is excellent and supports exactly that scoping. But the prose in places reads
as "SSCHA false-stabilises displacive instabilities" tout court. A SSCHA expert referee will
object that the *correct* SSCHA protocol for a structural instability is to let the centroids
relax (positions degree of freedom) and/or include v4, and that the cubic-constrained
free-energy Hessian is being asked a question it is not designed to answer at low T. Reframe
throughout as "SSCHA *as it is typically deployed with MLIP force engines for a fixed
high-symmetry reference*," and state plainly that the failure is a deployment trap, not a
theoretical refutation of the SSCHA. This strengthens, not weakens, the contribution.

**M5. Finite-size convergence is asserted, not shown.** Both methods use 2×2×2 cells (the screen
uses 6×6×6 force constants only for bcc). The brief pre-registered a SrTiO₃ supercell/MD
convergence sub-study; none is shown. At minimum, demonstrate that the *stability call* (sign),
for one perovskite and one bcc metal, is stable to 2×2×2 → 3×3×3 (screen) and 2×2×2 → 4×4×4
(SSCHA, at least for one tractable model). If a larger SSCHA cell changes the perovskite verdict,
M4 becomes essential.

## Minor concerns

- **m1.** Abstract says "Three findings" but the Introduction lists four contributions (H3 is the
  fourth). Reconcile.
- **m2.** §3.1: CHGNet's false-unstable rate (0.333) is the highest of any model on any layer and
  goes unexplained. Verified: it is CeO₂ (−0.26 THz) and NaCl (−0.24 THz) — marginal trips just
  past the −0.1 THz tolerance on genuinely stable controls. Say so, and show the call's
  sensitivity to the imaginary tolerance (the brief promised a tolerance sweep — include it, as
  these two calls likely flip at −0.3 THz).
- **m3.** §3.2 mixes two denominators without flagging it: "23 of 30" is the FE-perovskite
  displacive recall (3 systems × 5 models × 2 T), while the per-model table is the broader non-bcc
  set. State each population explicitly so the reader does not read 0.77 and 0.062 as the same
  metric.
- **m4.** ORB-v2 is a double confound (direct architecture *and* float32). Several claims lean on
  it ("broadly over-softens"). State that the design cannot separate architecture from precision,
  and avoid over-generalising from ORB-v2 to "direct models."
- **m5.** The −0.1 THz imaginary tolerance is central to every binary call but its sensitivity is
  never shown. A one-panel "rates vs tolerance" figure would pre-empt the obvious referee question
  and is already supported by the ledger (frequencies are stored).
- **m6.** SSCHA stochastic reproducibility: a fixed seed is used (good) but report the run-to-run
  spread of the min Hessian frequency on ≥1 unit, so the bcc margins (which discriminate models at
  the ~0.4 THz level) are shown to exceed the stochastic noise.
- **m7.** "n=45" bcc method-agreement with sign agreement 0.64 — for a *validation* figure this is
  only moderate; note that the disagreements are concentrated in ORB-v2's float32 outliers and
  report the agreement with ORB-v2 removed (0.78) alongside.
- **m8.** All DOIs/citations are placeholders; the deep-research pass must supply the
  anharmonic-stabilisation primary literature *per system* that underpins every ground-truth label
  (this is the evidentiary backbone of the whole benchmark and is currently un-cited).
- **m9.** Define "false-stable" and "recall" explicitly at first use; a screening audience will
  want the confusion-matrix convention stated once, early.

## Specific textual corrections

- Title is a strong claim ("is a blind spot"); ensure the conclusion's scope matches (it does).
- §2.5 "reliable for entropy-stabilised martensitic transitions" — supported for bcc here; soften
  to "reliable on the bcc martensitic cases studied here."
- Abstract: "can diverge numerically" — quantify (6 blow-ups; concentrated in float32 ORB-v2).
- §3.3 [PENDING] fluorite cell — when filled, explicitly state whether fluorite SSCHA is clean
  (it is numerically) *and* whether it still false-stabilises (preliminary ZrO₂ says yes); the
  distinction matters for M4.

## What would move this to "accept"

Address M1 (per-unit H2 statistic), M2 (multi-anchor screen validation), M3 (Methods/data
reconciliation) and M4 (scope the SSCHA claim to deployment). M5 and the tolerance sweep (m5)
would make the benchmark robust rather than illustrative. The science is there; the inferential
support needs to catch up to the framing.
