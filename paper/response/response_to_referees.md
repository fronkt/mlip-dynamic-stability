# Response to referees — RA-ART-07-2026-006452

**Manuscript** RA-ART-07-2026-006452
**Original title** *Finite-temperature dynamic stability is a blind spot of foundation
machine-learning interatomic potentials*
**Revised title** *Neither harmonic benchmarks nor a default SSCHA cross-check certifies a
foundation machine-learning interatomic potential for finite-temperature dynamic stability*
**Editor** Dr Lydia Rhyman, Associate Editor

> **DRAFT — not ready to send.** Items tagged **[PENDING]** are not yet complete. See the
> checklist at the end. Everything untagged is done and is in the revised manuscript.

---

Dear Dr Rhyman,

Thank you for the three reports, and please pass on my thanks to the referees. Referee 3 in
particular read the manuscript very closely, and several of the changes below are direct
consequences of that reading.

Before the point-by-point response I have to raise something that is not a referee item and
that the referees could not have known, because it changes numbers they quote.

## A. The referees reviewed a superseded version, and the numbers have moved against the paper

This manuscript reached RSC Advances by transfer from Digital Discovery (DD-ART-07-2026-000468),
submitted on 13 July 2026. In August, before the reports arrived, an internal audit of my own
deposited data found that **it was not reproducible by the deposited code**. Two defects were
responsible: a stale **q**-search grid that was skipped as "already present" because the unit
hash did not carry an algorithm version, and two inverted acoustic-mode masks which, for an
unstable phase, deleted the very instabilities under test. Every layer was re-measured from
scratch in independently pinned per-model environments. The audit trail is in
`tasks/audit-2026-08-16.md`; the re-measurement is described in the Data Availability section.

The re-measurement changed headline numbers. It is important to me that the editor sees the
direction of travel before the referees' comments are read against the revised text, because
most of the movement **cost the paper its cleanest claims**:

| Quantity | As reviewed | Revised | |
|---|---|---|---|
| Screen recall, FE perovskites | 23/30 = 0.77 | 16/30 = 0.53 | worse |
| SSCHA recall, FE perovskites | 7/30 = 0.23 | 5/27 = 0.19 | worse |
| bcc screen-vs-SSCHA rank correlation | ρ = 0.78 | ρ = 0.11 | **collapsed** |
| CHGNet finite-T accuracy | 0.933 | 0.867 | worse |
| MACE-MP-0 finite-T accuracy | 0.933 | 0.833 | worse |
| SevenNet-0 finite-T accuracy | 0.867 | 0.900 | better |
| ORB-v2 finite-T accuracy | 0.700 | 0.800 | better |
| Frequency-spread AUC | 0.52 | 0.36 | (uninformative either way) |

Two consequences bear directly on the reports. First, **the clean inversion is gone.** Referee 3's
summary describes "CHGNet and MACE-MP-0 reaching 0.933 while MatterSim falls to 0.833", which was
a genuine crossover between the harmonic and finite-temperature leaders. After re-measurement
SevenNet-0 leads both layers, and §3.2 says so explicitly. Second, **the cross-validation
statistic changed identity**: the reviewed version rested on a rank correlation of 0.78 between
the screen and SSCHA on bcc, which is now 0.11, so the reported statistic is the stability-call
agreement (0.78 over 45 paired units) and not a magnitude correlation.

This table is generated, not recalled: `scripts/reviewed_version_delta.py` extracts the numbers
from the reviewed commit and the revised text and diffs them, and fails loudly if a pattern stops
matching. The as-reviewed manuscript is deposited at
`paper/submissions/rsc-advances-2026-07/` so any reader can perform the same diff.

**Two further changes also predate the reports** and are noted here rather than presented as
responses to review. The title had already been changed because "blind spot" asserts that nobody
is looking, which two papers now in print falsify (refs 19 and 20); it has since been changed
again, for a reason given under Referee 3's item 1. And §2.4's variational derivation, the
clustered-statistics correction, and the measured harmonic estimator noise floor were added in
August and September for independent reasons, and happen to answer Referee 1's items 5 and 6 and
Referee 3's items 1 and 3. Where that is the case I say so and give the commit.

---

# Referee 1

> *the manuscript's primary claims currently exceed what the study design establishes*

This is the criticism I have taken most seriously, and the revision concedes more than the
referee asked for. The finite-temperature model ranking is now explicitly withdrawn as a claim
(see R1.6 and R3.3), the title no longer asserts a separation between models, and §3.2 no longer
asserts that harmonic accuracy is non-predictive.

## R1.1 — Independent first-principles validation along soft-mode coordinates

> *both the single-mode screen and SSCHA rely on the MLIP force engine, [so] disagreement between
> them cannot definitively establish which method is correct*

The premise is correct, and the manuscript should have been clearer about what does and does
not rest on that agreement. Being precise rather than sweeping: **no scoring, no validation gate
and no headline claim is licensed by screen-SSCHA agreement.** There is one place where the
agreement is used as evidence, and we now flag it as such rather than leaving it implicit: ESI
§S1.3 uses the bcc call agreement (0.78, or 0.83 excluding ORB-v2) to bound how much the screen's
neglect of mode-mode coupling can move a stability call, since the multi-mode SSCHA does retain
those couplings. That bound is legitimate only on bcc, where SSCHA is independently well behaved,
and it is a bound on an approximation rather than a validation of either method. The ground
truth is
experimental, not computational, and is therefore independent of the MLIP potential-energy
surface in a way that a DFT reference would only partly be:

- **Scoring labels** are documented experimental transition temperatures (§2.1). A unit is
  scored against which side of a measured transition it falls on.
- **The validation gate** is the SrTiO₃ transition at 105 K. The screen calls cubic SrTiO₃
  unstable at 100 K and stable at 300 K, bracketing the measured value, and it does so by
  condensing the R-point tilt while leaving the deeper Γ ferroelectric mode quantum-suppressed,
  which is the physically correct assignment (§2.4). No DFT enters this.
- **Controls**: 120/120 harmonically-stable control units correct, with zero imaginary
  commensurate modes found.

The referee also notes, correctly, that experimental transition temperatures reflect global
thermodynamic stability rather than local free-energy curvature. We agree, and this is why the
paper scores *which side of the transition* a unit falls on rather than a fitted T_c, why T* is
reported as a diagnostic and not a prediction (Table S6), and why the bcc metals are labelled
through their dynamic rather than their thermodynamic transition (§3.3).

On adding new DFT: I have not, and I want to be straightforward about why rather than imply it
was unnecessary. The study is designed around published ground truth so that no new
electronic-structure calculation is required, which is stated in the abstract; the comparison it
reports is between methods and between architectures on a shared surface, and a DFT reference for
a subset would not change any reported number. I do not have an allocation that covers this work.
I therefore regard independent first-principles profiles along the soft-mode coordinates as the
right next study rather than a revision item, and §4 now says so.

What I *can* separate at zero cost is the method error from the PES error on the screen side, and
the revision does this — see R1.2.

## R1.2 — Distinguishing MLIP force-engine error from anharmonic-method failure

> *the reported SSCHA false-stabilisation may stem from MLIP errors when sampling thermally
> accessible, highly distorted configurations out-of-distribution, rather than an intrinsic
> failure of the SSCHA formalism*

This is a sharp hypothesis and it is testable without DFT, because it makes a different
prediction from the truncation hypothesis. Three pieces of evidence separate them, two already
in the manuscript and one added here.

**1. A controlled within-unit diagnostic (ESI §S2.2, Table S2).** On cubic BaTiO₃ with
MACE-MP-0 at 100 K, the same force engine on the same structure gives a harmonic soft mode of
−5.635 THz, correctly unstable. The converged SCHA auxiliary matrix gives +2.892 THz, and the
free-energy Hessian with the fourth-order term dropped gives +2.872 THz, false-stable. The force
engine is identical across those rows. Only the truncation changes, and the sign changes with it.

**2. Architecture-uniformity.** An out-of-distribution force-engine failure should be
architecture-specific and precision-sensitive. It is not. The false-stabilisation occurs for all
five architectures, spans float32 and float64 models alike, and the new Table S11 shows the "swamped"
perovskite spectra present for every architecture: as fractions, since the denominators differ,
8/19 = 0.42 for MACE-MP-0 through 14/20 = 0.70 for CHGNet. It is not uniform across models, but
it is nowhere near confined to one.

**3. A within-cell control on the fluorites, new in this revision (§3.5).** In the same 2×2×2
cell and with the same force engine, the harmonic calculation finds all ten fluorite units
unstable, −3.8 to −10.6 THz, while SSCHA at 100 K finds all ten stable, +1.9 to +3.3 THz. The
comparison is stated at 100 K because that is where SSCHA is uniformly false-stable; by 600 to
900 K three of the five models destabilise, with the wrong temperature trend (§3.3). Both numbers now
have assertions in `scripts/verify_claims.py`. The force engine that is allegedly failing
out-of-distribution locates the instability correctly when asked harmonically, in the same cell,
at the same geometry.

The theoretical prediction is also not ours: Monacelli (*Phys. Rev. B* **112**, 014109, 2025) shows
the bubble approximation breaks down for exactly this class and that the fourth-order resummation
is required to determine phase stability. Our contribution is to measure that breakdown at
benchmark scale and show it extends beyond halides. Enabling `include_v4=True` did not finish in
18 minutes on a single unit, which is the practical barrier at grid scale.

**Separating method from PES on the screen side (ESI §S1.4, new).** Each deciding mode is stored
as a mass and three polynomial coefficients, so the same fitted potential can be solved exactly by
diagonalising the 1D Hamiltonian. Running that over the deposited ledger gives a result worth
reporting for its negative content: the exact isolated-mode criterion calls 96.5% of modes
condensed at 100, 300, 600 and 900 K alike, to four decimal places, because P(Q,T) → exp(−V/k_BT)
keeps a genuine double well bimodal at any temperature. An isolated one-dimensional mode cannot
thermally stabilise, so the exact solution is *not* a finite-temperature reference and the
divergence between it and the screen is not SCHA error. What it does establish is that the screen
never condenses a mode whose exact density is unimodal (0 of 228), and that the screen's
temperature dependence comes from the self-consistency rather than from the shape of the well.

## R1.3 — Contextualising MLIP architectures, training domain and screening methodologies

Done. The introduction has two new paragraphs and the bibliography eleven new entries (refs
27-37), covering both threads the referee names.

**Active learning, sparse Gaussian processes and on-the-fly potentials.** The added works are
more than context; three of them are on our own three families. On-the-fly Bayesian active
learning has reproduced the entropy-driven phase transitions of hybrid perovskites (ref 27), the
temperature-driven transitions and anharmonic thermal transport of zirconia (ref 28), and the
alpha-beta transition of zirconium (ref 29). We have used this to sharpen the paper's claim
rather than merely to acknowledge the field: these are *system-specific* potentials, actively
trained on configurations from the target's own dynamics, and their success establishes that the
gap we report belongs to **foundation models used as shipped**, not to machine-learned potentials
in general. That is a better statement of the contribution than the submitted version had, and it
came from this comment. Also added: the anharmonic-failure screen of ref 30, Gaussian
approximation potentials (ref 31), on-the-fly Bayesian force fields with variance-triggered data
acquisition (ref 32), committee uncertainty propagated into MD (ref 33), and the sparse-GP
potential of ref 37.

**ML-assisted high-throughput screening.** Added GNoME (ref 34), M3GNet (ref 35) and Matbench
Discovery (ref 36). The last is a useful foil rather than background: it benchmarks the same
model class on stability, but at the 0 K convex-hull level, so the distance between its criterion
and finite-temperature dynamic stability states our gap in one sentence.

**On identifying the requested references.** I could not match the request directly, because it
gives venues and years without titles or authors. Working from the venue-and-year fingerprints,
each of the six resolves to work from a single group. To be concrete: "PRB 2021" and "JPCL 2021"
correspond to Hajibabaei, Myung and Kim (*Phys. Rev. B* **103**, 214102) and Hajibabaei and Kim
(*J. Phys. Chem. Lett.* **12**, 8115); "Chem Phys Rev 2024" and "Chem Phys Rev 2025" to Willow
*et al.* (**5**, 041307 and **6**, 021401); "Acc Chem Res 2026" to Ha *et al.* (**59**, 103); and
"Adv Energy Mater 2026" to one of two review articles by the same group.

I have cited the sparse-Gaussian-process potential of *Phys. Rev. B* **103**, 214102 (ref 37),
which is a genuine primary source for the thread the referee raises. I have not cited the other
five. Four are review articles rather than primary sources, and none of the six treats
anharmonic crystals, phase transitions in the systems studied here, or the SCHA. I would also
gently note that the comment describes these works as showing how "Bayesian committee machine
potentials handle strongly anharmonic systems, phase transitions"; the *Chem. Phys. Rev.* **6**,
021401 paper is about oxygen-containing organic compounds, and I did not want to paraphrase a
characterisation into the manuscript that its abstract does not support.

I have instead cited the primary literature that bears most directly on this work, following the
editor's guidance in the decision letter. If the referee had specific papers in mind that I have
mis-identified, I would be glad to reconsider any of them on a title.

## R1.4 — Convergence diagnostics for SSCHA

The manuscript did not report these and should have.

**Now reported (Table S11, new).** The sampling configuration, stated once because it is
identical for all 201 units: 256 configurations per population, a cap of 8 populations, a
dedicated 512-configuration ensemble for the free-energy Hessian, 2560 samples in total. The
stopping criterion is `python-sscha`'s automatic stochastic relaxation under the per-population
step cap `minim.max_ka = 20` (ESI §S2.1), with the Hessian evaluated on a fresh ensemble at the
converged auxiliary matrix.

**Stated as not retained**, rather than glossed: the harness did not record per-iteration
free-energy gradient history or a per-unit uncertainty on the Hessian eigenvalues. Supplying
those requires re-running the grid, and Table S11 says so explicitly.

**A quality metric that *is* derivable from the deposited data**, added in its place: whether
the three translational acoustic modes, analytically zero, are resolvable within the recorded
spectrum, and how large their residual is. This separates the families sharply — bcc resolves
them in 75/75 units, with the largest residual over any model at 2.5 × 10⁻¹⁴ THz and most
orders of magnitude below that; the perovskites resolve them in 38 of 86.

**[PENDING] Four-seed extension beyond bcc-Zr**, to BaTiO₃ and one fluorite, as requested. Not
yet run. §S2.4 and Table S11 state that the seed study currently covers bcc-Zr only rather than
implying otherwise.

**A related axis is now measured, and it was not requested.** The paper's own robustness section
bounded the harmonic estimator using a paired replicate at fixed displacement amplitude, which
cannot speak to sensitivity to the amplitude itself. That sweep has now been run for CHGNet at
0.005, 0.02 and 0.03 Å against the deposited 0.01 Å (Table S13; the other four models need
compute I do not have). It reports in both directions. **No anharmonic test system changes its
call at any amplitude**, so the soft-mode detection the finite-temperature analysis rests on is
robust over a six-fold range. But CHGNet's harmonic accuracy runs from 0.737 to 0.895 across
that range, through the same three marginal control units the tolerance sweep moves, two of
which are its matched-set harmonic errors. Both are stated, and §3.2 now treats CHGNet's
harmonic standing as sensitive to both knobs rather than as a number.

## R1.5 — Formalisation and sensitivity of the soft-mode screen

Done, in new ESI §S1.3, which derives the screen from the Peierls bound. Note that §2.4's
derivation was added in August, before the reports (commit `e7375cc`); §S1.3 completes it against
the referee's three specific asks.

- **Mode mass-weighting**: derived from the kinetic energy of the frozen pattern. More usefully,
  the **normalisation convention is now pinned**, which the referee was right to flag as missing:
  it is *max atomic Cartesian component = 1 Å*, not unit norm and not the mass-weighted
  eigenvector. This is why the reported order parameter Q₀ reads directly as a displacement in
  ångström. §S1.3 also shows that every frequency and every stability call is invariant to the
  choice, and that only Q₀'s numerical value is not.
- **Gaussian-width self-consistency**: the stationarity step is now shown rather than asserted,
  including the identity ∂F₀/∂Ω = MΩσ² and the Gaussian smoothing identity ∂⟨V⟩/∂σ² = ½⟨V″⟩. The
  two MΩσ² terms cancel, which is precisely why the counter-term −½MΩ²σ² must be present.
- **Mode–mode coupling**: written out as the neglected biquadratic term with a mean-field
  reading, a_k^eff = a_k + Σ λ_kl⟨Q_l²⟩_T. This explains why Table 1 gives A1's bias as "either
  sign", and why the damage falls on the absolute T* rather than on the low-temperature call —
  which is what we observe, since T* mis-orders PbTiO₃ while the T ≤ 300 K calls are the screen's
  reliable output. I state plainly that λ_kl is unmeasured.
- **Solver specifics**: bracketing interval, the ⟨V″⟩ > 0 admissibility restriction, the
  rejection rule when no bound Gaussian exists, and the outer Q₀ grid.
- **Sensitivity**: the fit-window sweep is in ESI §S1.1 (3× and 8× against the production 5×
  changes 4 and 10 of 756 mode-level calls). Cell commensurability is now treated explicitly in
  §3.5 (see R3.2).

## R1.6 — Reconciling the statistical language

Agreed, and this is the change I regard as most important. The Abstract, §3.2, §4 and the title
all now carry the §3.2 formulation. The wording "harmonic accuracy is non-predictive of
finite-temperature accuracy" has been removed everywhere: it asserts a null, and the tests fail
to reject a null rather than establishing one. What the manuscript now claims is the failure of
*transfer* — 17 harmonically-correct units mis-called at 300 K against 4 the other way, McNemar
exact p = 0.007 — which is a statement about the relative difficulty of the two layers. See R3.1
and R3.3 for the details and for the title change.

---

# Referee 2

> *technically sound … [but] no mention of how fine-tuning would affect the results … no mention
> of what the ensemble uncertainty of the force predictions are*

Both gaps are real. "Fine-tuning" did not appear in the manuscript at all.

## R2.1 — Fine-tuning

§4 now carries an explicit scope subsection. The argument, in brief: every result is for released
checkpoints used without adaptation, and that is the regime the benchmark is about rather than an
omission. A practitioner filtering generative-CSP output does not know in advance which candidates
are strongly anharmonic, so there is no subset to fine-tune on and no target-specific data to
fine-tune with — a point the referee makes themselves. I agree that system- or family-specific
fine-tuning would improve finite-temperature performance substantially, and that for a study of a
*known* anharmonic material it is the right thing to do. The revision states that nothing here
bounds what these architectures achieve once adapted, and names quantifying that as the natural
next study.

## R2.2 — Ensemble uncertainty of the force predictions

I agree with the principle, and I would point out that §3.4 already runs this experiment; I think
its framing as a "guardrail" obscured that, and the revision makes the connection explicit.

The five independent architectures are the ensemble, and §3.4 tests whether their disagreement
flags the units where the consensus stability call is wrong. The result is specific and is exactly
the caution someone intending to threshold an uncertainty needs:

- The **discrete** signal works. Split votes carry a 6.6-fold enrichment in consensus error
  (8/14 = 0.571 against 4/46 = 0.087), AUC 0.762, and it survives a system-clustered permutation
  test at p = 0.004 with a cluster-bootstrap interval of [0.590, 0.934].
- The **continuous** signal fails. The cross-model spread of the predicted frequency — the
  propagated form of exactly the force-level disagreement the proposal appeals to — gives AUC
  0.361 with a clustered interval of [0.046, 0.625]. It carries no usable information.

§4 now states this, and distinguishes our cross-architecture spread from a within-architecture
committee spread, which is a different quantity and may behave better.

**[PENDING]** A force-level version, the cross-model spread of predicted forces on the displaced
configurations along E(Q), is in progress and will be added.

---

# Referee 3

I am grateful for this report. It found a self-contradiction I had missed, an ESI omission that
made the results uncheckable, and a statistical treatment that was genuinely wrong. All six items
are addressed; two of them are addressed further than requested, and one produced a result that
goes against the paper, reported below.

## R3.1 — §4 asserted the hypothesis that §3.2 rejects

Correct, and fixed. §4 said the results "confirm H2 in a strong form" and that harmonic accuracy
is "non-predictive"; §3.2 had already rejected the strong form. The first phrase was removed in
September (commit `dc11ed7`) for independent reasons; the second is removed now, and the §3.2
formulation is carried into the Discussion, the Abstract and the Conclusions.

**On the title, the referee's point had a consequence I did not anticipate.** Asked to bring the
title into line with §3.2, I checked it against the re-measured numbers and found it is not merely
overstated but **inverted**. The title claimed finite-temperature dynamic stability *separates*
models that harmonic benchmarks *rank equally*. In fact the finite-temperature accuracies span
0.100 (0.800 to 0.900) against the harmonic 0.211 (0.789 to 1.000), so that layer separates the
models *less*, and it still produces a tie (MatterSim = MACE-MP-0 = 0.833). The title has been
replaced with one that asserts only what is established:

> *Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation
> machine-learning interatomic potential for finite-temperature dynamic stability.*

## R3.2 — Zone-boundary convergence

I have taken the alternative the referee offers, and gone further on the part that matters.

**The claim is narrowed.** §3.5 now states plainly that the R-point (SrTiO₃) and X-point
(fluorite) systems have **no** supercell-convergence test in this work and that no convergence
claim is made for them. The referee is right that X is commensurate with 2×2×2 but that 3×3×3
does not contain it either, so no cell-size comparison exists there. The 4×4×4 test remains future
work. The surviving claim is the narrow one: the SSCHA verdict is cell-robust where it could be
tested, which is bcc.

**The missing-q rebuttal is now general.** The referee correctly notes that the Γ-mode argument
establishes this for BaTiO₃ only. The fluorites are now settled by a within-cell control that is
stronger than a convergence test and needs no new computation: in the same 2×2×2 cell with the
same force engine, harmonic calls all ten fluorite units unstable (−3.8 to −10.6 THz) and SSCHA
calls all ten stable (+1.9 to +3.3 THz). The cell resolves the instability, because the harmonic
calculation resolves it there. A finite-size account would have to explain an instability visible
to one method and invisible to another in the same supercell.

## R3.3 — Counts, intervals, and tests that respect the clustering

All correct, all adopted. A new module `mlip_dynstab/stats.py` and driver
`scripts/stats_hardening.py` implement these; the outputs are deposited in
`results/stats_hardening.json`.

**Counts and intervals.** Every rate in the results tables, every per-family recall and every
guardrail rate is now *k/n* with a Wilson score interval, and the referee's own arithmetic was
right: the harmonic accuracies are 19/19, 17/19 and 15/19. Two quantities are deliberately still
quoted without one, and I would rather name them than let the claim be read too broadly: the bcc
sign agreement (35/45 and 30/36), which is a concordance rather than an accuracy, and the 0/120
and 0/57 zero counts, which now carry their one-sided intervals in the text where they appear.
Wilson rather than Wald because several rates sit at the boundary, where Wald would assign
MatterSim's 19/19 the interval [1.000, 1.000].

**The consequence is a concession.** The intervals overlap heavily, so §3.1 and §3.2 now state
that the accuracies are not separated by this design and the finite-temperature ranking is
suggestive rather than established. That is the referee's own conclusion, and I have adopted it
in the text, in the Abstract, and in the title.

**And it exposed a worse problem in the same table, which I have fixed.** The §3.2 table put a
nineteen-system harmonic column beside a fifteen-system finite-temperature column. On the
fifteen systems the finite-temperature layer actually scores, CHGNet is 13/15 = 0.867
harmonically, which is *identical* to its finite-temperature accuracy; its apparent harmonic
deficit was bcc Zr/Hf, which that layer excludes, plus CeO₂ and NaCl, which §3.1 already calls
marginal finite-displacement noise. **The paper's flagship illustration, "CHGNet is worst
harmonically yet second best at finite temperature", was therefore a denominator artifact and is
withdrawn** from the Abstract and §3.2. The table now carries both columns with the matched one
named as the only comparable one, and the matched data support a better statement anyway: three
of five models are harmonically perfect on these systems and none exceeds 0.900 at finite
temperature.

**The full temperature ladder is now reported**, not the two points that made the case most
sharply. It helps rather than hurts: the transfer failure is significant at 300 K (p = 0.007)
**and at 600 K (p < 0.001)**, absent at 100 K, and not significant at 900 K where most labels
have flipped to stable. The discordance is one-directional at every temperature. On the
association, φ is positive at 100 K and negative above it, but at each of those three
temperatures the both-wrong cell is structurally empty, so the sign is forced by a zero; the
bootstrap interval excludes zero at three temperatures and the paper now explicitly gives that
no weight rather than showing the 300 K interval alone.

**Clustering.** The referee is right that the n = 60 units are clustered and that the model votes
are already collapsed into each one. The composition, which the ESI never gave, is now Table S7:
**both** analysis sets rest on the same 15 systems, with n = 60 being 15 systems × 4 temperatures
and n = 75 being 15 systems × 5 models. Every test now permutes or resamples whole systems:

| | AUC | clustered p | cluster-bootstrap 95% | naive unit-level p |
|---|---|---|---|---|
| Vote split | 0.762 | **0.004** | [0.590, 0.934] | <0.001 |
| Frequency spread | 0.361 | 0.275 | [0.046, 0.625] | 0.139 |

The referee was right that the naive test was anti-conservative, and wrong only in supposing the
result depended on it: the vote-split guardrail survives. The frequency-spread interval spans 0.5,
so I have **withdrawn "below chance"** — the honest statement is that it is uninformative.

**McNemar.** Adopted in full. The manuscript no longer reads any failure to reject as positive
evidence. The surviving p = 0.007 result is labelled as a difference in layer difficulty
(marginal homogeneity) wherever it appears, and never as an association.

**Spearman.** Adopted, with a sharper version of the referee's point. With five models there are
5! = 120 pairings, so the statistic has a resolution floor and cannot carry an inference at any
value. The floor is computed rather than asserted (the test is two-sided and the accuracy
vectors contain ties, so it is not 1/120): it is 0.107 at 300, 600 and 900 K and 0.603 at
100 K. Enumerated exactly: ρ = +0.645 (p = 0.60) at 100 K,
−0.667 (p = 0.41) at 300 K, −0.889 (p = 0.11) at 600 K, −0.167 (p = 1.00) at 900 K. It changes
sign across the ladder, which is itself the argument. It is now reported as descriptive only.

**A paired test the manuscript was missing, and a significance claim I have withdrawn as a
result of writing it.** Comparing the screen's 0.53 against SSCHA's 0.19 by inspecting two
marginal intervals ignores that both methods see the same units, so I added the paired test. An
exact McNemar over the discordant units gives p = 2 × 10⁻⁷ on the combined displacive set. I am
not quoting that number, because on inspection it commits the referee's own objection: those 47
units are five systems under five models at up to four temperatures, and treating them as
independent is exactly what §3.4 was corrected for. Randomising the method label over whole
systems instead, which is exact at this size, gives **p = 0.125**, with four of five systems
favouring the screen (net discordances +6, +6, −1, +9, +10). With five systems the floor is
2/2⁵ = 0.0625, so no arrangement of these data could have reached significance at the system
level.

§3.3 therefore now reports a large, one-directional, consistent effect and explicitly declines a
significance claim, and rests the case that the effect is real on mechanism instead: the
diagnostic in which only the truncation order changes and the sign of the answer changes with
it. Table S10 gives every system set both ways and both tests, with the unit-level column
labelled as one not to quote.

## R3.4 — Tables S1–S4

The referee is exactly right: §S3 was thirteen lines of bullets each ending in the name of a
function, with no table content anywhere, so no headline rate was checkable. The functions existed
and ran; the tables were never rendered.

`scripts/build_esi_tables.py` now renders them from the deposited ledger, and the ESI regenerates
rather than being hand-maintained (`--check` fails if it is stale). Two further defects found in
the same pass are fixed: the ESI numbered its *sections* S1–S4 **and** its *tables* S1–S4, so a
reader looking for Table S1 landed on a section, and the three tables that did have content had no
numbers or captions. Tables are now S1–S12 and every one has a caption, with a note at the head of the ESI
disambiguating them from the sections. One exception to strict order of appearance: Table S12,
the exact-1D comparison, sits in §S1.4 and so is rendered early; it is numbered last because it
was added last, and I have kept the number stable rather than renumbering the tables the referee
has already been asked to check.

Beyond the four requested, the ESI now also carries the analysis-set composition (S7), the ORB-v2
split (S9), the paired tests (S10), the SSCHA diagnostics (S11), and the exact-1D comparison (S12).

## R3.5 — Every rate with and without ORB-v2

Adopted, and it changed a conclusion.

Table S9 reports each affected rate both ways, and §3.3 and §3.5 carry the consequences. Two
findings the referee should see:

**1. A number the manuscript had reported only one way.** §3.3 quoted the bcc frequency rank
correlation as ρ = 0.11 without noting that this is the all-model value; excluding ORB-v2 it is
≈ −0.003. Both are now given. (This reinforces rather than undermines the existing choice to
report call agreement, 0.78 and 0.83, as the cross-validation statistic.)

**2. The referee's concern was partly justified, and I report it against interest.** The central
screen-versus-SSCHA contrast, tested paired on the ferroelectric oxides **alone**, is significant
with ORB-v2 (14 vs 3, p = 0.013) and **not** significant without it (10 vs 3, p = 0.092). That
subset is small once SSCHA's own failures are removed, and ORB-v2 contributes disproportionately
to it.

The claim therefore rests on the combined displacive set rather than on the ferroelectric oxides
alone, where the cubic fluorites carry it: **26 against 3, p = 2 × 10⁻⁵, with ORB-v2 excluded.**
The fluorites are the appropriate anchor for two reasons, and one thing I first claimed for them
is wrong and has been corrected. They are numerically clean, with zero blow-ups and no failed
units, and the paired discordance excluding ORB-v2 is 16 against 0. What I initially wrote, that
Table S11 shows their difficulty is not concentrated in any one architecture, does not survive
checking the denominators: fluorite spectra in which no acoustic zero is resolvable are 4/8 for
ORB-v2 and 2/8 for MACE-MP-0 against 0/8 for the other three, so that particular diagnostic *is*
ORB-weighted. The table and its caption now say so. The ORB-independence of the fluorite anchor
rests on the recall comparison, not on that diagnostic. §3.3 now states all of this explicitly, and Table S10 gives all three system
sets both ways so a reader can check the sensitivity themselves.

## R3.6 — Figure order

Fixed. Fig. 5 was placed after Fig. 6 despite being first cited in §3.3; it now sits immediately
after Fig. 4, and all six figures appear in order of first citation.

---

# Also changed

- **A stale number in the ESI.** The threats-to-validity table still carried ρ = 0.78 for the bcc
  cross-check, missed when the August re-measurement was propagated. Corrected, with the call
  agreement and the ORB-v2 split both given.
- **A contradiction in a deposited file.** `results/estimator_noise.json` carried
  `ordering_invariant: false` next to prose stating the ordering is never reversed. Both are
  true — at zero tolerance the two models tie, so a strict inequality fails — and the ESI now
  says so.
- **Reference 20** had a title and DOI but no author list. Completed: D. Li, J. Yang, X. Chen,
  L. Yu and S. Liu, *J. Phys. Chem. C*, 2025, **129**, 21538-21544.
- **`verify_claims.py`** now runs 29 assertions against the deposited ledger, including the new
  within-cell fluorite control. All pass.

---

# Outstanding before submission

| | Item |
|---|---|
| ☐ | **R1.4** four-seed stochastic test extended to BaTiO₃ and one fluorite |
| ☐ | **R2.2** force-level cross-model uncertainty on the E(Q) displaced configurations |
| ◐ | Displacement-amplitude sweep: CHGNet done (Table S13); four models remain |
| ☐ | Rebuild DOCX (clean + tracked changes), figures at 600 dpi, TOC entry |
| ☐ | CRediT author-contributions section; link ORCID at submission |
| ☐ | Decide on RSC's transparent peer review option |
| ☐ | New Zenodo version; confirm the concept DOI still resolves correctly |
| ☐ | Confirm APC coverage (Purdue read-and-publish) |

Yours sincerely,

Frank Cai
