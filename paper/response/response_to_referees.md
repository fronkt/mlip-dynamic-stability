# Response to the referees

**Manuscript** RA-ART-07-2026-006452, *RSC Advances*
**Revised title** Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation
machine-learning interatomic potential for finite-temperature dynamic stability
**Title as reviewed** Finite-temperature dynamic stability is a blind spot of foundation
machine-learning interatomic potentials
**Author** Frank Cai, Purdue University

Dr Lydia Rhyman
Associate Editor, *RSC Advances*

Dear Dr Rhyman,

Thank you for handling the manuscript. Please pass my thanks to the three referees, whose reports
changed what the paper claims. The revision:

- states H2 at the strength a system-clustered test supports: harmonically correct units that the
  screen mis-calls at finite temperature outnumber the reverse (17 to 4 at 300 K under the
  screen's production convention), not significantly (p = 0.152), and mostly in three systems the
  screen mis-calls for nearly every model; no model ranking is claimed (R1.6, R3.1, R3.3);
- relates the SSCHA false-stables to the local question its default criterion asks and reports
  SSCHA's high-temperature false-unstables (R1.1, R1.2);
- adds PBE single points along the screen's soft-mode coordinates and on SSCHA-sampled
  configurations (R1.1, R1.2), and a four-seed SSCHA study with recorded convergence diagnostics
  (R1.4);
    on PBE energies the shared BaTiO₃ and KNbO₃ mis-calls disappear (MLIP wells about half the PBE
  depth) and the CsSnBr₃ ones persist;
    MLIP force error does not grow on the 100 K SSCHA samples and rises 1.7-fold at 600 K in SrTiO₃;
  <!-- PENDING-C1: one short clause with the outcome of the seed study and of the converged-mode runs (C1c). The seed study is complete (commit 2fdbb61): no seed of any unit converged, and the auxiliary matrix moved only in the first population. If C1c does not keep the false-stables, rewrite the SSCHA bullet above as well. -->
- adds the literature on active-learning potentials and ML-driven screening (R1.3);
- completes the screen's derivation and reports its sensitivity, including the frozen-cell
  convention that decides its ferroelectric recall (R1.5);
- discusses fine-tuning and reports a pre-registered force-level ensemble test, which did not show
  a signal (R2.1, R2.2);
- drops the zone-boundary convergence claims, gives rates as counts with intervals, clusters by
  system every test used as evidence, splits every pooled rate by ORB-v2, renders the ESI tables
  and orders the figures by citation (R3.2–R3.6).

## Before the point-by-point answers

Three things bear on the reports that the referees could not have known.

### (a) Several numbers the referees quote have moved, most of them against the paper

The manuscript was submitted to *Digital Discovery* on 13 July 2026 and transferred to *RSC
Advances*. In mid-August, before the reports arrived, an audit of my own deposit found that the
deposited results could not be reproduced by the deposited code. The cause was the hash that
decides whether a unit has already been computed: it did not include the algorithm version, so
units computed with a superseded **q**-point search were skipped as present instead of being
recomputed. The audit also found a defect that the code and the deposit shared: the acoustic-mode
masks of the screen and of the SSCHA post-processing removed the three lowest-frequency branches
instead of the three nearest zero, and for an unstable phase the lowest branch is the soft mode,
so the mask removed the very instability under test. The first SSCHA generation had also been
run in a software environment that was not recorded. I
re-measured every layer from scratch in pinned per-model environments. The superseded generations
remain in the deposited ledger beside the current ones, and the Data availability section
describes both defects.

The numbers below are the ones the referees quote or rely on, as they appear in the version they
read and as they are now.

| Quantity (where it is quoted) | As reviewed | Now |
|---|---|---|
| Harmonic accuracy, 19 systems (R3.3) | 1.000, 0.895, 0.789 | unchanged: 19/19, 17/19, 15/19; ORB-v2 0.842 → 17/19 |
| Finite-T accuracy, T ≤ 300 K: CHGNet, MACE-MP-0, MatterSim (R3 summary) | 0.933, 0.933, 0.833 | 26/30, 25/30, 25/30 |
| Screen recall, ferroelectric perovskites, T ≤ 300 K (R3 summary) | 23/30 | 16/30 [0.36, 0.70] |
| SSCHA recall, same units (R3 summary) | 7/30 | 5/27 [0.08, 0.37] (one blow-up and two failed runs excluded) |
| H2 association φ (R3.1) | +0.11, the 100 K value (the text did not give the temperature) | +0.149 at 100 K; −0.129 at 300 K (−0.114 on the reviewed data); not resolved at any temperature |
| H2 McNemar exact p (R3.3) | 0.34, the 100 K value (7 v 3) | 0.727 at 100 K (5 v 3); at 300 K 0.007 unit level (17 v 4; 0.031 and 14 v 4 on the reviewed data), 0.152 clustered by system |
| Per-model Spearman ρ, harmonic vs finite-T (R3.1, R3.3) | +0.15 to +0.65 | removed (R3.3) |
| Vote-split AUC (R3 summary, R3.3) | 0.745 | 0.762 [0.590, 0.934]; 0.628 [0.438, 0.839] without ORB-v2 |
| Frequency-spread AUC (R3 summary) | 0.52 | 0.361 [0.046, 0.625] |
| bcc screen–SSCHA rank correlation ρ (reviewed §3.3) | 0.78 | 0.11, descriptive only |
| bcc screen–SSCHA sign agreement (R3.5) | 0.64; 0.78 without ORB-v2 | not used: the screen's curvature is positive by construction and its negative bcc values are numerical (R1.1); the stability calls agree in 31/45 = 0.69 (25/36 without ORB-v2) |
| ORB-v2 finite-T false-stable rate (R3.5) | 0.562 | 5/16 = 0.312 |
| ORB-v2 MgO harmonic frequency (R3.5) | −2.8 THz | −1.07 THz |
| ORB-v2 bcc screen outliers (R3.5) | near −35 THz on Ti and Hf | none; most negative −8.9 THz (Ti, 900 K); Hf −0.09 THz |
| SSCHA blow-ups (R3.5) | six, concentrated in ORB-v2, down to −2×10⁶ THz | eight with \|f\| > 50 THz over four models, three of them ORB-v2; most negative −914 THz |

Intervals are Wilson 95% intervals for rates and cluster-bootstrap 95% intervals over systems for
AUCs. The screen's recall fell from 23/30 to 16/30, and the finite-temperature layer no longer
resolves the crossover in Referee 3's summary, in which CHGNet and MACE-MP-0 lead MatterSim at
finite temperature while trailing it harmonically: MACE-MP-0 now ties MatterSim at 25/30, CHGNet's
26/30 is one unit ahead, and the five finite-temperature accuracies (24/30 to 27/30) are not
separated by this design. On the fifteen systems the finite-temperature layer scores, CHGNet is
13/15 harmonically, the same fraction as its 26/30 at finite temperature. The bcc
agreement that the reviewed text used as a cross-validation is also weaker than it read: the
stability calls agree in 31/45 units, and 12 of the agreements are trivial (R1.1).

The version the referees read (commit 76d3a84) is kept in the repository beside the revision, so
each of these numbers can be checked against both.
<!-- PENDING-P5: name the release tag, commit and Zenodo version DOI that contain paper/submissions/rsc-advances-2026-07/, once the branch is merged and the version is minted. -->

### (b) The title

The title the referees read asserted that finite-temperature dynamic stability had not been
examined. Two studies in print had examined it, each for a single model family or a single system
(J. Yang, Z. Yin, L. Ao and S. Li, *Phys. Chem. Chem. Phys.*, 2026, **28**, 4459, ref 19; D. Li,
J. Yang, X. Chen, L. Yu and S. Liu, *J. Phys. Chem. C*, 2025, **129**, 21538, ref 20), so in
August, before the reports, I changed it to one saying that finite-temperature dynamic stability
separates models that harmonic benchmarks rank equally (commit 0fbea7d, 16 August 2026). The
re-measured data do not support that title either, since the finite-temperature accuracies are not
separated, and Referee 3 asked that the title follow the formulation the data support (R3.1). The
revised title makes two existence claims, neither of which rests on a significance test, so
clustering by system cannot overturn them: a top harmonic score does not certify a model's
finite-temperature calls (MatterSim, MACE-MP-0 and SevenNet-0 are harmonically perfect on the
matched set, and the screen still mis-calls 5, 5 and 3 of their 30 units), and the default SSCHA
cross-check, as run, does not supply that certification for deep displacive wells (§3.3).
The PBE check of R1.1 bears on the first clause in the opposite direction from the one this
paragraph anticipated: most of the shared mis-calls behind it are not the screen's. Eight of the
17 units (BaTiO₃ and KNbO₃ at 300 K) are called correctly on PBE energies along the same
coordinates, so they are softened MLIP wells that four architectures share, and the clause stands
as a statement about the models.
<!-- PENDING-C1: if the converged-mode runs (C1c) do not keep the SSCHA false-stables, the second clause must be revisited with the author before submission. -->

### (c) What was changed before the reports arrived

Some changes were made in August and early September for reasons of my own and bear on items the
referees raise. I list them here so that none of them is presented as a response:

- 16 August 2026, commits 4c8e478 and 0fbea7d: the phrase "methodological trap" removed from the
  Abstract and §3.3 (R1.4), and the SSCHA result related to the published analyses of the SSCHA
  free-energy Hessian (R. Bianco *et al.*, ref 21; L. Monacelli, ref 22). In 0fbea7d, the sentence
  "The finite-temperature results confirm H2 in a strong form" removed from §4 (R3.1), the phrase
  "necessary but not sufficient" removed from the Abstract, §1 and §3.2 (R3.1), and the title
  changed as in (b).
- 16 August 2026, commit a7b375f: an Author contributions section added and the back matter put
  in the journal's order (Editor's comments).
- 16 August 2026, commit e7375cc: the variational derivation of the screen in §2.4 and the
  approximation ledger of Table 1 (R1.5, in part; ESI §S1.3 now completes it).
- 10 September 2026, commit dc11ed7: the harmonic replicate-noise measurement (ESI §S1.2, Table
  S1) and a system-clustered permutation test for the association φ (R3.3, in part).

The phrase "non-predictive" that R1.6 and R3.1 quote was still in §4 after these changes, and it is
removed now. Everything else below was done after the reports arrived.

---

# Referee 1

> However, the manuscript's primary claims currently exceed what the study design establishes.
> Crucially, both the single-mode screen and SSCHA are driven by the same underlying MLIP
> potential energy surface (PES), making their agreement a consistency test rather than an
> absolute measure of physical accuracy.

I agree, and the revision is organised around this point. It withdraws three claims the design did
not support: that harmonic accuracy is non-predictive of finite-temperature accuracy, that the
harmonic leaders are not the finite-temperature leaders, and that SSCHA driven by an MLIP is a
methodological trap. It describes the agreement between the screen and SSCHA as a consistency check
on one surface, and it adds first-principles calculations that test the surface itself.

## R1.1 Independent first-principles validation

> Establish Independent First-Principles Validation: Because both the single-mode screen and SSCHA
> rely on the MLIP force engine, disagreement between them cannot definitively establish which
> method is correct. Experimental phase transition temperatures are governed by global
> thermodynamic stability, not necessarily local free-energy curvature. The authors must
> incorporate independent first-principles (e.g., DFT or high-level reference) evaluations along
> soft-mode coordinates or anharmonic PES profiles for a key subset of systems (such as SrTiO3,
> BaTiO3, or bcc-Zr).

**Response.** I agree with all three sentences, and the second one changed the paper more than any
other comment.

*First-principles evaluations.* I have added PBE single-point calculations along the screen's own
soft-mode coordinates for the three systems the referee names (the R-point tilt of SrTiO₃, BaTiO₃
and bcc Zr) and for three more on which the paper's argument depends: KNbO₃ and CsSnBr₃, which
with BaTiO₃ hold most of the H2 counterexamples (R1.6), and cubic ZrO₂, a fluorite on which
SSCHA's false-stable involves no numerical failure. Each model's deciding coordinate is regenerated
with the production code and checked against the cached E(Q) map the paper was computed from, all
five MLIPs are evaluated on the identical geometries, and the screen is re-solved on the
PBE-fitted potential. Where a model has no instability of its own on a system (MACE-MP-0 and CHGNet
on bcc Zr, for example), its path follows MatterSim's deciding pattern at that model's relaxed
lattice. The comparison answers two questions: whether PBE, evaluated along each model's deciding
coordinate at that model's relaxed lattice, gives the same well, and whether a mis-call shared
across models persists on the PBE surface, in which case it belongs to the single-mode
approximation (or to the functional), or disappears, in which case it belongs to the MLIP surfaces.
It does not re-derive the coordinate or the lattice in PBE.
All 448 calculations finished, and the outcome splits by system (new §2.6 and §3.2; ESI §S5,
Table S19). (i) Along the coordinates that decide BaTiO₃, KNbO₃ and ZrO₂, the wells of CHGNet,
MACE-MP-0, MatterSim and SevenNet-0 are shallower than PBE's (0.32–0.76 of the depth for the first
two systems, 0.45–0.61 for ZrO₂) at the same minimum position; for SrTiO₃'s R tilt they are
0.25–0.99 of it; for CsSnBr₃ three of the five match PBE within 20 %; and along MatterSim's
deciding coordinate in bcc Zr, PBE has a 159–174 meV well where CHGNet, MACE-MP-0 and SevenNet-0
have none. (ii) The shared 300 K mis-calls on BaTiO₃ and KNbO₃ disappear on PBE: all eight units
are called unstable, as labelled, so they belong to the MLIP surfaces. The CsSnBr₃ mis-calls
persist, with every PBE minimum at the edge of the scan and a 300 K label 8 K above the
transition, and so does KNbO₃ at 600 K (T_c 708 K), called stable by all five models and by PBE.
(iii) Over the 120 ladder units of the six systems, swapping only the energy engine raises
agreement with the labels from 81 to 101 (23 corrected, 3 newly wrong; no system worse); ten of
the corrections are bcc Zr. The counts are descriptive (six systems, PBE at each model's lattice);
I did not repeat the profiles at the PBE-relaxed lattice along the PBE eigenvector.

*Global stability against local curvature.* This distinction is now the centre of §3.3. For the
anharmonic systems the labels record which phase is the equilibrium phase at T, a global question
about the free energy. SSCHA's default criterion, the free-energy Hessian at the symmetric
reference, asks a local one: is the symmetric phase a local minimum? For a deep double well
treated with a Gaussian trial state the free energy can keep a local minimum at the symmetric
point while a displaced minimum lies lower, and a curvature read at the reference then reports
stable. On the same MLIP energies the screen finds that situation on most of the units in
question: of the 57 non-bcc units where SSCHA calls the phase stable against an unstable label,
the screen's free-energy comparison, the minimum over the centroid (§2.4), finds a lower displaced
minimum and calls the phase unstable on 46 (46/57 = 0.81 [0.69, 0.89]; 39/50 without ORB-v2). The
screen's own curvature at the symmetric point adds nothing here, and §2.4 and ESI §S1.3 now say
why: for a single mode with an even potential it equals the self-consistent trial stiffness MΩ²,
so it is positive whenever a bound Gaussian exists and cannot register condensation. That the
screen finds a lower displaced minimum on these units shows that the local and the global question
have different answers on these energies. It does not show that SSCHA's positive Hessian has the
same origin; a relaxation that stopped short of the SCHA minimum is the alternative, and R1.4
describes how it is being tested. Because the comparison is made on one surface, it also says
nothing about which criterion matches first principles. §2.1 and the Limitations now also say
that the labels mark thermodynamic, often first-order, transitions, whereas both methods probe
dynamic stability, so a label and a dynamic-stability call can differ near a transition without
either being wrong.

*Agreement on one surface.* §2.5 now states that SSCHA inherits the MLIP potential-energy surface
exactly as the screen does, and §3.3 and the caption of Fig. 4 describe screen–SSCHA agreement as a
consistency check. The descriptions of SSCHA as a gold standard and of the screen as tracking it
are gone. The bcc agreement is also reported more carefully than before: the stability calls agree
in 31/45 = 0.69 [0.54, 0.80]; 12 of the agreeing units are MACE-MP-0 and CHGNet on Zr and Hf, where
neither model has a harmonic instability to stabilise (Fig. 3 now marks them); and MatterSim, whose
bcc instabilities are the deepest, agrees with SSCHA on the call in none of its 9 units (Table
S16). The reviewed text compared the two methods' frequencies directly. For the reason just given,
the screen's curvature is not scored against SSCHA by sign: the ten bcc units on which it is
negative are all numerical (six fall-backs of the width solver, one fit artefact, and three units
in which the model has no screened instability; Table S16). The screen's calls are scored against
the reference labels (experimental for the anharmonic systems, the DFPT phonon database for the six
stable controls), not against SSCHA. Its SrTiO₃ gate, which on the 100/300 K ladder brackets the
105 K transition without locating it, is passed by three of the five models (§2.4).

**Changes.** §2.1; §2.4 ("Criterion and observable", SrTiO₃ gate); §2.5; §3.3 (the bcc paragraphs,
and "Why the default criterion misses these wells"); §4 (Limitations); captions of Figs. 2, 3 and
4; ESI §S1.3 ("What the symmetric-point curvature is", last paragraph), §S2.2, §S4, Table S16;
`scripts/curvature_identity_check.py`.
New §2.6 (first-principles reference) and the C3a paragraph of §3.2; ESI §S5.1 and Table S19.

## R1.2 MLIP force-engine errors against method failures

> Differentiate MLIP Force Engine Errors from Anharmonic Method Failures: The reported SSCHA
> false-stabilization may stem from MLIP errors when sampling thermally accessible, highly distorted
> configurations out-of-distribution, rather than an intrinsic failure of the SSCHA formalism
> itself. The authors should benchmark MLIP forces/energies against first-principles reference
> calculations along the sampling paths.

**Response.** I have run the benchmark the referee describes: PBE energies and forces on
configurations drawn from the SSCHA ensembles of four units, compared with the same MLIP-vs-PBE
errors on near-equilibrium rattled configurations of the same cells. The configurations are
subsamples of the final Hessian ensembles of the production recipe, re-run with four seeds in the
study described under R1.4. The units are BaTiO₃ and ZrO₂ with MACE-MP-0 at 100 K, where SSCHA
false-stabilises; SrTiO₃ with MACE-MP-0 at 600 K, where it returns −20.2 THz against a harmonic
−2.37 THz; and bcc Zr with MatterSim at 50 K, a bcc unit that carries a harmonic instability.
All 68 calculations finished (ESI §S5.2, Table S20). At 100 K (0.09 Å root-mean-square
displacement, against 0.03 Å for the rattled baseline) MACE-MP-0's force error relative to the PBE
forces is 0.10 for BaTiO₃ and 0.18 for ZrO₂, against 0.10 and 0.23 near equilibrium, so the error
does not grow on these samples; bcc Zr behaves the same way, though its relative error is a ratio
on small forces. These ensembles come from relaxations that did not converge (R1.4), so at 100 K
they may not reach the double well, and a small error on them does not clear the MLIP inside it.
At 600 K in SrTiO₃ (0.34 Å) the relative error rises from 0.11 to 0.19, energy errors reach
44 meV per atom, and all five models fall at 0.12–0.20: the MLIPs are extrapolating where the
runaway happens, although twelve configurations cannot show that this, rather than the sampling,
causes it.

Two things in the MLIP-only data bear on the hypothesis, one for each direction of SSCHA error.

*The false-stables.* The reviewed text attributed them to the formalism. That attribution is
withdrawn, and §3.3 now states that none of its results is a claim that the SSCHA formalism is
wrong; they concern the question its default criterion asks. The revised account is narrower: the
default criterion is local, and on the same energies the screen's free-energy comparison finds a
lower displaced minimum on 46 of the 57 units (R1.1). That account, if it holds, needs no MLIP
error on the sampled configurations, but it does not exclude one, and the manuscript says so. Nor
does it exclude a relaxation that stopped short of the SCHA minimum, which R1.4 addresses. The
reviewed text also presented the stage-by-stage diagnostic of ESI Table S2 (numbered as in the
revised ESI) as isolating the mechanism, with the fourth-order term as the remedy. It does not
isolate it: the one run that would change only the truncation, with the fourth-order term
included, did not finish (more than 18 min on this one unit, against about 200 s without the
term), so no two rows of the table differ in truncation order alone, and the remedy is untested.
§3.3 and ESI §S2.2 now describe the table as what it is, and the reviewed per-unit estimate of
that run time is replaced by the observed bound.

*The high-temperature false-unstables.* The reviewed version did not discuss them, and it should
have, because they are the pattern the referee's hypothesis predicts. Over the non-bcc units whose
label is stable, SSCHA calls the phase unstable in 5/5 at 300 K, 10/13 at 600 K and 15/19 at 900 K
(4/4, 8/11 and 12/16 without ORB-v2). SrTiO₃ is false-unstable in all 14 returned units above its
105 K transition, at −3.4 to −914 THz, against harmonic R-mode frequencies of −0.8 to −2.1 THz on
the four models that have an R-point instability (ORB-v2 has none), and 14 of the 23 non-bcc units
that SSCHA calls stable at 100 K turn negative by 600–900 K (nine of them against a stable label;
the other five are fluorite units, where the sign is right and the trend is not). An instability
that grows with thermal amplitude on a fixed potential is the opposite of entropy stabilisation.
It is what an MLIP extrapolating on large-amplitude configurations would produce; an instability of
the stochastic sampling itself, and a free-energy Hessian evaluated at an auxiliary matrix that has
not reached the SCHA minimum, are the other candidates, and MLIP-only data cannot separate them.
The SrTiO₃ unit at 600 K in the PBE benchmark above was chosen to test the first.
In the seed study this unit is the one where the finite-temperature correction, not the start,
sets the answer: the start is +0.90 THz and the four seeds give −15.8 to −20.3 THz, none of them
converged, so the third candidate is not excluded by the production recipe.
<!-- PENDING-C1c/GRID: whether the converged recipe keeps SrTiO3 600/900 K unstable (the grid includes SrTiO3 units). -->

**Changes.** §3.3 ("Why the default criterion misses these wells", "SSCHA false-unstables that grow
with temperature", "Numerical failures"); §4 (Limitations); ESI §S2.1, §S2.2 (Table S2), §S2.3,
§S2.4 (SrTiO₃), Tables S16 and S17.
New §2.6; §3.3 ("Why the default criterion misses these wells" and "SSCHA false-unstables that
grow with temperature", last sentences); §4; ESI §S5.2 and Table S20.

## R1.3 Context: MLIP architectures, training domain and screening

> Contextualize MLIP Architectures, Training Domain, and Screening Methodologies: To provide readers
> with a comprehensive perspective on how MLIPs perform in energy materials and high-throughput
> screening, the introduction and discussion should be enriched by discussing established paradigms
> in machine-learning interatomic potentials and data-driven screening strategies (Adv Energy Mater
> 2026). Specifically, the authors are encouraged to implicitly address and cite key developments
> in: Sparse Gaussian Process and On-the-Fly MLIPs (PRB 2021, JPCL 2021): Discuss how
> active-learning, sparse Gaussian process, or Bayesian committee machine potentials handle strongly
> anharmonic systems, phase transitions, and complex potential energy surfaces with high sample
> efficiency (e.g., in solid electrolytes, energy materials, and molecular dynamics contexts) (Chem
> Phys Rev 2024, 2025; Adv Energy Mater 2026); ML-Assisted High-Throughput Screening: Situate the
> proposed single-mode soft-mode screen within the broader context of ML-driven screening
> frameworks used for structural stability, functional energy materials, and catalyst discovery
> (Adv Energy Mater/Acc Chem Res 2026).

**Response.** The Introduction has two new paragraphs, one for each thread, and the Discussion
returns to this literature where it discusses model-reported uncertainty.

*Active learning, sparse Gaussian processes and on-the-fly potentials.* This thread turned out to
bear directly on the paper. On-the-fly active learning with Bayesian error estimation has reproduced
the entropy-driven phase transitions of hybrid perovskites (R. Jinnouchi *et al.*, *Phys. Rev.
Lett.*, 2019, **122**, 225701, ref 27), the temperature-driven transitions and anharmonic thermal
transport of zirconia (C. Verdi *et al.*, *npj Comput. Mater.*, 2021, **7**, 156, ref 28) and the
α–β transition of zirconium (P. Liu *et al.*, *Phys. Rev. Mater.*, 2021, **5**, 053804, ref 29),
which are the perovskite, fluorite and bcc families of this paper. Those are system-specific
potentials trained on configurations from the target's own dynamics, and their success places the
gap measured here with foundation models used as shipped, not with machine-learned potentials in
general. The Introduction now says so. Also added: an active-learning screen aimed at MLIP failures
in strongly anharmonic materials (K. Kang *et al.*, ref 30); Gaussian approximation potentials (A.
P. Bartók *et al.*, ref 31); on-the-fly Bayesian force fields whose predictive variance triggers
new reference data (J. Vandermause *et al.*, ref 32); committee uncertainty propagated into
molecular dynamics (G. Imbalzano *et al.*, ref 33); scalable sparse Gaussian-process regression
with data-efficient on-the-fly sampling, which reproduced the melting and glass-crystallisation
temperatures of the Li₇P₃S₁₁ solid electrolyte (A. Hajibabaei, C. W. Myung and K. S. Kim, *Phys.
Rev. B*, 2021, **103**, 214102, ref 37); the Account by M. Ha, S. Pourasad, C. W. Myung and K. S. Kim
(*Acc. Chem. Res.*, 2026, **59**, 103, ref 38), which covers sample efficiency (practical accuracy
from 100–1000 quantum calculations), uncertainty-triggered sampling and the robust Bayesian
committee machine; and the review of MLIPs for energy materials by I. K. Park *et al.* (*Adv. Energy
Mater.*, 2026, e71046, ref 39), which sets Gaussian-process frameworks with uncertainty
quantification beside the equivariant graph networks and foundation models tested here and lists
foundation-model fine-tuning among the emerging directions.

*ML-assisted high-throughput screening.* Added GNoME (A. Merchant *et al.*, ref 34), M3GNet (C.
Chen and S. P. Ong, ref 35) and Matbench Discovery (J. Riebesell *et al.*, ref 36), and the use of
sparse Gaussian-process potentials to accelerate high-throughput screening (ref 38). The
Introduction now places the soft-mode screen as one filter in such a pipeline, for candidates that
pass the convex-hull check but show imaginary harmonic modes. Matbench Discovery scores stability
against the 0 K convex hull and the phonon benchmarks score harmonic stability; neither asks about
the finite-temperature dynamic stability a screened candidate needs, which is the gap this paper
measures. I have kept the discussion to the screening of crystal stability; catalyst discovery is
outside what this benchmark can speak to.

I also considered the other works the comment points to, and the paper cites those that bear
directly on its subject, the finite-temperature dynamic stability of the crystal families studied
here, or on the uncertainty and screening questions it raises. Against the version the referees
read, the reference list has grown from 18 entries to 46. The additions made before the reports
are the two finite-temperature studies behind the title change, the two analyses of the SSCHA
Hessian, and the four sources of the §2.4 derivation; three cite the codes of the new
first-principles reference (R1.1); the rest were added for this comment and for Referee 2's first
point.

**Changes.** §1 (third and fourth paragraphs); §4 ("On model-reported uncertainty"); refs 27–39.

## R1.4 Convergence diagnostics for SSCHA

> Rigorous Convergence Diagnostics for SSCHA: The claim that SSCHA is a "methodological trap"
> requires stronger justification. The authors must report standard convergence metrics: sample
> sizes, gradient history, stopping criteria, and Hessian uncertainties. The four-seed stochastic
> test should be expanded beyond bcc-Zr to include representative displacive systems (BaTiO3 or
> fluorites).

**Response.** *"Methodological trap".* I have withdrawn the claim rather than tried to justify it.
The phrase was removed in August (see (c) above), and what replaces it in §3.3 is narrower: the
default criterion is local, and on the same energies the screen's free-energy comparison finds a
lower displaced minimum on 46 of the 57 units SSCHA false-stabilises (R1.1). §3.3 presents that as
consistent with the SSCHA false-stables, not as their demonstrated cause, because a relaxation
that stops short of the SCHA minimum can also report stable, and the convergence metrics the
referee asks for are what decide between the two.
<!-- PENDING-C1b: one sentence, whichever way it falls: the include_v4 = True (fourth-order) Hessian on BaTiO3/MACE-MP-0/100 K, the unit of ESI Table S2: whether it finished within the cap, its lowest frequency and wall time, and whether the call changes; match §3.3 and ESI §S2.2. -->

*Sample sizes and stopping criteria.* These are now stated in §2.5, ESI §S2.1 and Table S11. The
starting dynamical matrix is built from phonopy full force constants at a 0.03 Å displacement in
the 2×2×2 SSCHA cell, made positive definite (`ForcePositiveDefinite`) and symmetrised. The
auxiliary matrix is relaxed in the root2 representation with `min_step_dyn` = 0.5, and the
convergence threshold is `meaningful_factor` = 10⁻⁴. Up to 8 populations of 256 configurations are
drawn (2,048 at most), and the bubble-level free-energy Hessian is evaluated on a dedicated ensemble
of 512 configurations at the final auxiliary matrix, 2,560 configurations per unit at most. The
minimiser's steps are capped at `max_ka` = 20. I introduced that cap to stop a single population
from iterating indefinitely, on the understanding that it counts the steps within each
population, and the reviewed text described it that way. It does not: python-sscha 1.6.1 keeps
the minimiser's step history across populations and compares `max_ka` with the accumulated count,
so the cap bounds the steps over the whole relaxation. §2.5, ESI §S2.1 and Table S11 now say so.
The re-run with recorded histories (R1.4 below) shows what the cap did: on all 16 seeds of the
four units the first population was ended after 19 steps, each later population took one step and
discarded it, and the convergence test was never met, so the free-energy Hessian was evaluated
close to the positive-definite starting matrix (within 0.24 THz of it on BaTiO₃, ZrO₂ and bcc Zr).
<!-- PENDING-C1c/GRID: what the converged recipe (per-population cap, real gradient error) shows for the same units and across the grid; carry the consequence into R1.1, R1.2, the summary and §3.3. -->

*What the production grid did not record.* It did not record how many populations each unit used,
whether each unit met the convergence threshold before the population cap, the gradient history,
or an uncertainty on the Hessian. 2,560 is therefore the configured maximum, not a count, and §2.5
and Table S11 say so rather than reconstruct the missing values.

*The seed study.* The reviewed four-seed test (+1.798 ± 0.001 THz) was weaker than the text
implied: it ran on bcc Zr with MACE-MP-0, which has no harmonic instability in Zr, so it measured
SSCHA's stochastic noise on a surface with nothing to stabilise, and §3.5 now says so. To supply the
missing metrics where they matter, I re-ran the production recipe with four seeds on four units:
BaTiO₃ and ZrO₂ with MACE-MP-0 at 100 K, the displacive and fluorite false-stables the referee
asks about; bcc Zr with MatterSim at 50 K, a bcc unit that does carry a harmonic instability; and
SrTiO₃ with MACE-MP-0 at 600 K, a high-temperature false-unstable. Every run records the populations
used, the per-population free-energy and gradient histories, the stopping test that fired, the
lowest frequency of the positive-definite starting matrix against the final Hessian value, and
the lowest Hessian eigenvalue, whose spread over the seeds is the Hessian uncertainty. Seed 0 is the
production computation, so the study also checks that the deposited value is reproduced.
The outcome (ESI Table S21) is unfavourable to the production recipe and I report it as such. No
seed of any unit met the convergence test: the cumulative step cap stopped the first population,
and later populations did not move the auxiliary matrix. The seed spread of the lowest Hessian
frequency is small where the start dominates (BaTiO₃ +2.867 to +2.881 THz, ZrO₂ +3.069 to +3.087,
bcc Zr +1.636 to +1.786, against starts of +2.881, +2.962 and +1.880) and large for SrTiO₃ at
600 K (−15.8 to −20.3 THz, standard deviation 2.0 THz, with a bootstrap over the Hessian ensemble
of 2.2–3.7 THz per seed). Seed 0 reproduces the deposited value for the first three units and not
for SrTiO₃ (−18.7 against −20.2 THz), whose deposited value lies inside the seed range. No call
changes across seeds. The recorded gradient error turned out to be a constant placeholder passed
by python-sscha 1.6.1, so the library's own convergence test is not a stochastic one (ESI §S2.1).
<!-- PENDING-C1c/GRID: then the converged recipe's result for these units and the grid. -->

**Changes.** §2.5; §3.3; §3.5 ("Stochastic reproducibility"); §4 (Limitations); ESI §S2.1,
§S2.2 (Table S2), §S2.4, Table S11.
ESI Table S21 (new: the seed study and the bcc cell-size re-measurement).

## R1.5 Formalisation and sensitivity of the soft-mode screen

> Formalization and Sensitivity of the Soft-Mode Screen: The mathematical formulation of the
> single-mode quantum SCHA screen needs complete derivation in the ESI (mode mass-weighting,
> Gaussian width self-consistency equation, and mode-mode coupling limitations). Sensitivity to
> polynomial fit windows, sampling ranges, and cell commensurability must be systematically
> reported.

**Response.** The derivation is now complete in ESI §S1.3, which extends the §2.4 derivation added
in August.

- *Mode mass-weighting.* The effective mass M = Σᵢ mᵢ|**u**ᵢ|² follows from the kinetic energy of
  the frozen pattern. §S1.3 also fixes what the reviewed text left open: the pattern is normalised
  so that its largest Cartesian component is 1, so Q in ångström is the largest Cartesian
  displacement component in the cell (not the largest atomic displacement); M and V(Q) are summed
  over the mode's minimal commensurate cell; and four implementation constants are fixed in
  ångström (sampling range, centroid scan, condensation threshold and finite-difference step), so a
  different normalisation of **u** would need them rescaled with it.
- *Gaussian-width self-consistency.* The stationarity step is shown in full, including the two
  identities that make it work (∂F₀/∂Ω = MΩσ² and ∂⟨V⟩/∂σ² = ½⟨V″⟩), and the solver is described:
  a bracketed root in σ² refined by Brent's method, a fallback to the nearest grid value on 4.6% of
  centroid evaluations and at none of the 1512 that decide a call, and the parity assumption
  V(−Q) = V(Q), which holds for most modes but was not checked for 24 bcc modes with 3**q** ≡ **G**.
  §S1.3 states that stationarity is shown, not that the stationary point is a minimum in Ω. ESI §S1.4
  (Table S12) solves the same fitted potentials exactly: an isolated mode's thermal density stays
  bimodal at every temperature, so the screen's temperature dependence comes from the
  self-consistency, and the screen never condenses a mode whose exact density is unimodal. §S1.3
  now also shows what the screen's reported curvature is: at the symmetric point it equals the
  self-consistent trial stiffness MΩ², so it is positive by construction and only the call can
  register condensation (R1.1).
- *Mode–mode coupling.* The neglected terms are written out (biquadratic, cubic and trilinear
  couplings, where symmetry allows them), with a mean-field reading of the biquadratic term that
  shows why the bias can have either sign (Table 1, A1). I have no quantitative bound on the
  couplings and say so. Nor are the reported calls far from where coupling matters: the deciding
  wells at T ≤ 300 K are 28–43 meV deep per minimal cell in BaTiO₃, 25–59 meV in KNbO₃, 0–110 meV
  in PbTiO₃ and 0–27 meV in SrTiO₃, against k_BT = 26 meV at 300 K, so those calls sit close to the
  screen's own stabilisation temperature and a coupling error can move them directly.
  No symmetry analysis of the screened pairs was done, and components of a degenerate triplet that
  a model returns at slightly different frequencies are screened as separate modes (BaTiO₃ under
  CHGNet). Within a degenerate eigenspace the screened direction is the one the eigensolver
  returns, and the well is not the same along every direction of the span: for MACE-MP-0 on
  BaTiO₃ at Γ the production direction gives a 20.9 meV well with M = 53 amu, and another direction
  of the same triplet 16.8 meV with 41 amu. §2.4 and §S1.3 now say so.

*Sensitivity (ESI Table S14, with ORB-v2 excluded as well).* Every setting re-solves the cached
E(Q) maps with the production solver.

- *Fit window.* Multipliers of 3×, 5× and 8× the well depth with floors of 30, 60 and 120 meV
  change no unit call at any temperature. For 69% of modes the kept points are identical across the
  multipliers at the 60 meV floor, which is why the floor is varied as well.
- *Sampling range.* Truncating each map at 0.25–0.40 Å changes at most one unit call at T ≤ 300 K
  (eight of 400 over the full ladder at 0.25 Å, seven of them halide-perovskite units at
  600–900 K).
  Condensation thresholds of 0.5–3 grid steps and a centroid scan widened to 1.2 Å change none.
- *Cell commensurability.* This axis has two parts, and one of them decides outcomes. The cell over
  which M and V are summed is not a harmless convention: summing over n minimal cells is the
  minimal-cell problem at mass n²M and temperature T/n, so larger cells condense more readily. The
  ferroelectric-perovskite recall at T ≤ 300 K is 5/30 per formula unit, 16/30 in the production
  minimal cell, 26/30 in the doubled cell and 28/30 in the common force-constant supercell, and the
  SrTiO₃ gate passes for 0, 3, 2 and 3 of the five models. Accuracy on the labels cannot choose among
  them (105, 127, 136 and 139 of 150 at T ≤ 300 K). The paper therefore reports the screen's recall
  and T* as properties of a stated convention, the minimal cell in which the mode is a single
  commensurate distortion, and not as convention-free numbers. The H2 count depends on it as well:
  at 300 K it is 17 against 4 in the production cell and runs from 7 against 4 to 24 against 4
  across the other conventions, and at 100 K three conventions give no excess at all (ESI §S1.3,
  `scripts/h2_by_convention.py`). This weakens the screen, and the Abstract, §3.2, §3.3, §5 and
  the Limitations say so. The other part, the force-constant supercell
  that fixes which **q**-points are searched, was not enlarged; §2.4 and ESI §S4 say so, and Table 1
  lists it as approximation A2. Cell size for SSCHA is under R3.2.

**Changes.** §2.4 ("Definitions and numerical settings", "Criterion and observable", "Frozen-cell
normalisation" and the paragraph after it); Table 1 (A3, A7); §3.2; §4 (Limitations); §5; ESI
§S1.1, §S1.3, §S1.4 (Table S12), Table S14, §S4.

## R1.6 Statistical language and claims

> Reconcile Statistical Language and Claims: The Abstract and Discussion currently frame harmonic
> accuracy as "non-predictive" of finite-temperature performance, whereas the Results section notes
> a weak positive correlation. Align the headline claims with the statistical analysis to avoid
> overstatement.

**Response.** Agreed. "Non-predictive" asserts a null that the tests could not establish, and it is
gone from every section. The weak positive correlation the reviewed Results reported, φ = 0.11,
was the value at 100 K, and the reviewed text did not give the temperature. It is now +0.149 at
100 K, and φ is negative at 300, 600 and 900 K (−0.129 at 300 K), as it already was on the
reviewed data (−0.114 at 300 K). At each of those three temperatures the cell of units both layers
get wrong is empty, so the sign is forced by a zero rather than measured. Permuting whole systems
gives p above 0.08 at every temperature. Neither framing survives, and the paper now claims no
association in either direction.

What it claims instead concerns transfer. On the matched set, 17 harmonically correct units are
mis-called by the screen at 300 K against 4 the other way (23 against 4 at 600 K), under the
screen's production frozen-cell convention; across the other conventions the 300 K count runs from
7 against 4 to 24 against 4 (R1.5). That asymmetry is large and one-directional in unit counts, but it is not significant once units are clustered by
system (exact system-level p = 0.152 at 300 K and 0.066 at 600 K). The unit-level McNemar p (0.007
at 300 K) treats five models on each of fifteen systems as independent, and appears only as a
labelled companion. The paper also says where the discordances sit, as a description selected on
the outcome and not as a test: 13 of the 17 at 300 K fall in BaTiO₃, KNbO₃ and CsSnBr₃, three
systems the screen mis-calls for at least four of the five models, and without them the count is 4
against 4. Three of those four are PbTiO₃, the screen's clearest ordering failure, and all four
reverse units are structural: three are control units that the harmonic layer reads off its
interpolated mesh as unstable and the commensurate screen cannot reproduce, and the fourth is
ORB-v2 on SrTiO₃, which finds no instability at any temperature (§3.2). An error four or five
architectures share could be the single-mode screen's own approximation or a surface error the
models have in common, and the PBE-backed screen of R1.1 tests which. For most of them it is the
models: on PBE energies along the same coordinates the eight BaTiO₃ and KNbO₃ units are called
correctly, so they are a softening four models share, while the five CsSnBr₃ units stay
mis-called; the revised §3.2 and §4 say so.

The Abstract, the H2 statement in §1, §3.2, §4 and the Conclusions carry this wording, and none of
them reads a failure to reject as a finding.

**Changes.** Abstract; §1 (H2); §2.4 ("Frozen-cell normalisation"); §3.2 (the temperature-ladder
table and the paragraphs after it); §4; §5; ESI §S1.3, Table S15.

---

# Referee 2

> The manuscript provides a test of universal MLIPs for different dynamically stabilized systems.
> The work focuses exclusively on a stability argument and does not mention how fine-tuning would
> affect the results, which most experts would assume is needed for these systems. While for crystal
> structure prediction you would not necessarily have the prior knowledge to make that assumption,
> there is no mention of what the ensemble uncertainty of the force predictions are. This metric can
> be used as a reliable measure of where the MLIP predictions should not be trusted. Because of this
> I don't think the manuscript is suitable for publication, despite it being technically sound.

Thank you for this report. Both omissions were real: the reviewed manuscript did not mention
fine-tuning at all, and it did not test uncertainty at the level of forces. The revision adds a
discussion of the first with references, and a pre-registered measurement of the second. On the
observation that the work "focuses exclusively on a stability argument": the revision also
compares each model's energies and forces with first-principles values, along the soft-mode paths
the screen integrates and on configurations the SSCHA samples (R1.1, R1.2).
Along the soft-mode paths the four conservative models' wells are about half as deep as PBE's
in BaTiO₃, KNbO₃ and ZrO₂, and three models have no well at all in bcc Zr;
on the SSCHA samples the force error does not grow at 100 K and rises 1.7-fold, with energy errors
to 44 meV per atom, at 600 K in SrTiO₃.

## R2.1 Fine-tuning

> The work focuses exclusively on a stability argument and does not mention how fine-tuning would
> affect the results, which most experts would assume is needed for these systems.

**Response.** The Abstract now says that the models are tested as shipped, §2.2, §4 and the
Conclusions say that none was fine-tuned, and §4 says why. That is the regime of generative-CSP
screening: a practitioner filtering candidates does not know in advance which of them are strongly
anharmonic, so fine-tuning on the candidates themselves is not available before the screen, as
the referee's report notes for crystal structure prediction. Fine-tuning on broad phonon data is
available, and ref 41 below deploys such a model at screening scale; this benchmark does not test
one, and §4 now names a phonon-fine-tuned checkpoint as the natural next model to put through it.
For a known anharmonic material I agree that fine-tuning is the right step, and §4 now cites what
it has been shown to do: universal MLIPs, CHGNet and MACE-MP-0 among them, systematically soften the potential-energy surface, and
fine-tuning on a small number of reference calculations removes much of that softening (B. Deng
*et al.*, *npj Comput. Mater.*, 2025, **11**, 9, ref 40); fine-tuning a universal potential on
harmonic phonon data cut its phonon-frequency error about fourfold, and the tuned model was then
used for SSCHA across 4669 inorganic compounds (H. Lee *et al.*, *Small*, 2026, **22**, e00071, ref
41); partially frozen fine-tuning of MACE-MP matched a potential trained from scratch with 10–20% of
its data and reproduced the phonon dispersions of α-, β- (bcc) and ω-Ti qualitatively (M. Radova
*et al.*, *npj Comput. Mater.*, 2025, **11**, 237, ref 42); and on PbTiO₃, fine-tuning MACE on a
small PBEsol dataset recovered the ferroelectric–paraelectric transition, with the Curie temperature
still about 160 K below experiment (ref 20). One further study bears on both of the referee's
points: fine-tuning four of the five checkpoints tested here (MACE-MP-0, SevenNet-0, a large
MatterSim checkpoint and ORB-v2) on system-specific first-principles data cut their force errors
5–15-fold and brought the architectures to comparable accuracy (J. Hänseroth, A. Flötotto, M. N.
Qaisrani and C. Dreßler, *J. Phys. Chem. Lett.*, 2026, **17**, 3152, ref 43). It tested diffusion,
structure and energy pathways rather than phonons, and §4 cites it with that restriction; it also
implies that cross-architecture disagreement is a signal of the models as shipped (R2.2).

§4 then sorts the paper's findings by whether fine-tuning could change them. It could change the
per-model inputs: the harmonic calls (the flattened Zr and Hf instabilities of MACE-MP-0 and CHGNet
are the kind of softening these studies address), the well depths the screen reads, and so the
per-model screen accuracies and the H2 counts built from them. Whether it would remove the
mis-calls shared across models, which carry most of the H2 count, is what the PBE-backed screen of
R1.1 tests, since it re-solves the screen on the surface a PBE fine-tune aims at, along the same
coordinates (the mode pattern still comes from each MLIP's force constants).
It does so in part: the BaTiO₃ and KNbO₃ mis-calls disappear on that surface and the CsSnBr₃ ones
do not, so a fine-tune that deepened the softened ferroelectric wells toward PBE would plausibly
remove about half the shared count.
Fine-tuning could change which units fall in the deep-well regime of §3.3, since that depends on
the well depth, but not the locality of the criterion: a curvature read at the symmetric reference
reports stable on any surface whose double well is deep enough, fine-tuned or not. Nor could it
change the screen's dependence on its frozen-cell convention. And fine-tuning moves a model toward
its reference functional, not toward experiment. I have not fine-tuned any model here, and how much
of the finite-temperature gap fine-tuning closes for each family, and with how much data, is beyond
what a benchmark of released checkpoints can say; §4 says that too.

**Changes.** Abstract; §2.2; §4 ("Scope: foundation models as shipped", "On model-reported
uncertainty"); §5; refs 20 and 40–43.

## R2.2 Ensemble uncertainty of the force predictions

> While for crystal structure prediction you would not necessarily have the prior knowledge to make
> that assumption, there is no mention of what the ensemble uncertainty of the force predictions
> are. This metric can be used as a reliable measure of where the MLIP predictions should not be
> trusted.

**Response.** I agree that a model's own uncertainty is the natural trust metric, and §4 now sets
out the literature behind it (committee spreads propagated into molecular dynamics, ref 33;
predictive variance as a trigger for new reference data, refs 32 and 38; uncertainty-driven active
learning that avoids unphysical dynamics in strongly anharmonic materials, ref 30). The released
checkpoints come as single models rather than trained committees, so I tested the idea in three
forms on the same 60 consensus units (15 systems at four temperatures; ESI Tables S7 and S8),
asking in each case whether the signal flags the units where the five models' majority call is
wrong.

*The vote split across architectures (§3.4).* With all five models, the consensus is wrong on 8/14
units where the models split and on 4/46 where they agree, and the split predicts consensus error
with AUC 0.762 [0.590, 0.934] (system-clustered p = 0.0035). Without ORB-v2 it falls to 0.628
[0.438, 0.839], an interval that spans 0.5, with three 2–2 ties broken to "stable" (two of them
wrong). It is a suggestive guardrail, not a robust one, and the paper now says so; the reviewed
version reported it with all five models only.

*The spread of the screen's frequency across models (§3.4).* No signal could be resolved (AUC 0.361
[0.046, 0.625]). This spread is a proxy at the level of the potential-energy surface, not the
force-level uncertainty the referee means: the E(Q) maps are energies, and forces enter only
through the force constants that fix each mode pattern.

*The force-level test.* This is new. Its protocol and decision rule were written before any
configuration was generated or any force evaluated, and amended once after a CPU smoke test but
before any production configuration existed (the amendment made the overlap check resolve species
pairs and stated the missing-data rule and the softness confound; the primary was unchanged). They
are the dated block at the head of `scripts/force_spread.py`, and the block's hash is recorded in
the output. For each consensus unit,
16 thermally displaced configurations are drawn from the quantum harmonic distribution of the five
models' mean force constants at that temperature, the same configurations for every model. The
score is the across-model spread of the forces (the root-mean-square over atoms and components of
the standard deviation across models, median over configurations), and the primary statistic is its
system-clustered AUC for predicting consensus error. The rule fixed in advance was to report the
spread as flagging untrustworthy calls only if the cluster-bootstrap interval lay entirely above 0.5
and the four-model estimate (a) was also above 0.5. The pre-registration names seven secondary
scores, each to be reported whatever the primary shows: (a) the same spread without ORB-v2; (b, c)
within-family committees, MACE-MP-0 small, medium and large and MatterSim 1M and 5M, against the
consensus; (d) the spread normalised by the size of the mean force, the check fixed in advance
against softness and amplitude; (e) each model's leave-one-out deviation from the others against
that model's own call; (f) each committee's spread against its own model's call, which is the
metric closest to the one the referee describes; and (g) the primary restricted to units in which
no configuration brings two atoms closer than 0.75 of the shortest distance between the same two
species in the reference cell. ESI Table S18 lists all of them.

The primary AUC is 0.681 (system-clustered p = 0.104) with a 95% interval of [0.416, 0.908], which
spans 0.5; without ORB-v2, (a), it is 0.684 [0.413, 0.916]. By the pre-registered rule the force
spread is not shown to flag untrustworthy calls. The secondary scores are: (b) 0.717 [0.535, 0.902]
and (c) 0.726 [0.516, 0.901]; (d) 0.578 [0.335, 0.795], 0.576 without ORB-v2; (e) pooled over
models 0.654 [0.425, 0.832], 0.689 [0.438, 0.882] without ORB-v2, and 0.542 to 0.761 per model;
(f) 0.722 [0.564, 0.873] for MACE-MP-0 (system-clustered p = 0.034) and 0.686 [0.473, 0.873] for
MatterSim. Five of the nineteen secondary scores have a lower bound above 0.5: (b), (c), (f) for
MACE-MP-0, and (e) for MACE-MP-0 and for SevenNet-0 without ORB-v2, each of those two at 0.502.
None is adjusted for multiple comparisons, all use the same configurations, and I do not promote
any of them after the fact. The overlap check (g) is the reason for caution. Restricted to the
43 units in which no configuration brings atoms unphysically close, the primary AUC is 0.314
[0.146, 0.529], and those 43 units hold only 4 of the 12 consensus errors. The soft
halide-perovskite and ferroelectric spectra that carry most of the errors are the ones whose
thermal draws push A-site cations into the anions, where every model is extrapolating, so the
positive estimate over the full set rests on units in which the spread measures that extrapolation
rather than disagreement about thermal forces. The restriction was pre-registered for the primary
only and was not applied to the secondary scores, which use the same configurations, so the same
caution applies to them. The configurations also sit near the unrelaxed reference cell rather than
each model's relaxed cell, and the committees differ in size and training run, so they are family
committees and not a deep ensemble of one model.

With fifteen systems these data do not exclude a useful force-level signal; they do not show one.
The comparison H3 pre-registered, ensemble disagreement against a single model's own signal, was
not run, and the paper says so.

**Changes.** §3.4 (the ORB-v2 split, the tie rule, the part of the enrichment built into the
construction, the frequency spread as a PES-level proxy, the force-level test and its result); §4
("On model-reported uncertainty"); §5; Fig. 6 (both model sets); ESI Tables S8, S9 and S18.

---

# Referee 3

Thank you for a report that read the manuscript and its supplement closely enough to find a
contradiction between two sections, an ESI that named tables it did not contain, and statistics that
treated clustered units as independent. All three were right. Several numbers in the report's
summary have since moved (table in (a) above).

## R3.1 §4 against §3.2

> Reconcile §4 with §3.2; the Discussion asserts the hypothesis that the Results reject. Section
> 3.2 tests the strong form of H2, states explicitly that it is rejected, and reports a weak
> positive association (phi = 0.11, per-model rho = +0.15 at T <= 300 K rising to +0.65 at T = 100
> K), settling on "necessary but not sufficient"; Section 4 then states that the results "confirm
> H2 in a strong form" and that "harmonic accuracy is non-predictive of finite-temperature
> accuracy," which is the opposite of a positive correlation. The §3.2 formulation is the one the
> data support and should be carried consistently into the Discussion, the abstract, and the
> framing of the title.

**Response.** Correct: §4 contradicted §3.2. "Confirm H2 in a strong form" was removed on 16 August
(commit 0fbea7d), before the report; "non-predictive" remained in §4 and is removed now.

I have to add that the §3.2 formulation the referee endorses does not survive a reading of the
whole temperature ladder. The weak positive association (φ = 0.11) was the 100 K value, which the
reviewed text did not label; it is now +0.149 at 100 K (system-clustered p = 0.125), and at 300 K
and above φ is negative, as it already was on the reviewed data, with a sign forced by an empty
cell. The per-model rank correlations are not interpretable (R3.3). "Necessary but not
sufficient" would need an association the data do not show, so it is not used. What the
Discussion, the Abstract, the Conclusions and the title now carry is narrower: harmonically correct
units that the screen mis-calls at finite temperature exist and outnumber the reverse, 17 to 4 at
300 K under the screen's production convention (7 to 24 against 4 under the others; R1.5); the
asymmetry is not significant at the system level (p = 0.152) and most of it sits in three systems
with an error shared across models (R1.6); no association between the layers is resolved; and the
pre-registered form of H2 is left open rather than rejected or confirmed. The title is discussed in
(b) above.
The PBE check (R1.1) now locates most of that shared error in the models rather than the screen:
the BaTiO₃ and KNbO₃ mis-calls disappear on PBE energies along the same coordinates, the CsSnBr₃
ones persist, and the title's first clause stands as a statement about the models.

**Changes.** Title; Abstract; §1 (H2); §2.4; §3.2; §4; §5; ESI §S1.3.

## R3.2 Zone-boundary convergence

> Complete the deferred 4x4x4 SSCHA for SrTiO3 and one fluorite, or remove all convergence claims
> for zone-boundary systems. The manuscript concedes that the SrTiO3 antiferrodistortive instability
> at R = (1/2,1/2,1/2) is commensurate only with even supercells and that the 3x3x3 comparison is
> therefore invalid for it, which leaves the R-point and X-point systems with no convergence test at
> all, while the 2x2x2 cell used throughout is marginal for the ZrO2/HfO2 X-point mode; the argument
> that the §3.3 false-stable is not a missing-q artifact because it also occurs for the Gamma mode
> of BaTiO3 establishes this for one system only and does not extend to the fluorites or to SrTiO3.

**Response.** I have taken the second option; the 4×4×4 SSCHA (about 320 atoms) was not run. §3.5
and ESI §S2.4 now state that the R-point and X-point systems have no supercell-convergence test in
this work and that no convergence claim is made for them. I agree that the 2×2×2 cell is marginal
for the fluorite X-point mode: it is the smallest cell that contains X, 3×3×3 does not, and so no
cell-size comparison exists there. The SSCHA results of §3.3 are stated for the 2×2×2 cell used
throughout, and the Limitations repeat this.

The referee is also right that the BaTiO₃ argument covers one system. What remain are three
narrower checks, each stated as ruling out a missing **q**-point and nothing more:

- *Fluorites.* In the same 2×2×2 cell and with the same force engine, the harmonic calculation finds
  all ten fluorite units unstable (−3.8 to −10.6 THz) while SSCHA at 100 K finds all ten stable
  (+1.9 to +3.3 THz), so the cell contains the instability SSCHA calls stable. This does not show
  that the 2×2×2 result is converged, and it does not distinguish the local-criterion account of
  §3.3 from an MLIP error on the sampled configurations.
- *BaTiO₃.* Its instability includes the zone-centre mode, which every supercell contains; §3.5
  restricts this argument to BaTiO₃.
- *SrTiO₃.* SSCHA never calls it stable at any temperature, and a missing zone-boundary **q**-point
  could only hide an instability, so the objection does not arise. What SrTiO₃ shows instead is the
  high-temperature divergence discussed under R1.2.

The one cell-size comparison in the paper, bcc Zr at 2×2×2 against 3×3×3, was run with MACE-MP-0,
which has no harmonic instability in bcc Zr, and in the superseded software environment; §3.5 now
says both. The same commensurability objection applies to it: the bcc N point is commensurate
only with even cells and the ω point only with multiples of three, so the two cells are not nested
and the comparison changes the set of **q**-points as well as the cell size. §3.5 and ESI §S2.4
now say that too.
<!-- PENDING-C5: rewrite the paragraph above once the pinned-environment re-measurement lands (it will no longer be "the one cell-size comparison ... run with MACE-MP-0"): bcc-Zr SSCHA at 2x2x2 against 3x3x3 for MatterSim and MACE-MP-0 at 100 and 300 K, whichever way it falls, described as a change of q-set (N only in 2x2x2, omega only in 3x3x3) unless a nested 4x4x4 (and 6x6x6) run is added; point to §3.5. -->

**Changes.** §3.5 ("Finite size", "Zone-boundary systems" and the paragraph after it); §4
(Limitations); ESI §S2.4, §S4, Table S17.

## R3.3 Counts, intervals and tests that respect the clustering

> Report every rate with counts and an interval, and replace the AUC and rank correlations with
> tests that respect the clustering. Accuracies quoted as 1.000, 0.895 and 0.789 are 19/19, 17/19
> and 15/19 on nineteen systems and should be given as such with bootstrap or Wilson intervals; the
> AUC of 0.745 is computed on sixty consensus units that are repeated measurements on a much smaller
> number of systems — the composition of that set is never stated, and the five model votes are
> already collapsed into each unit — so the units are clustered by system and cannot be treated as
> independent, and the system list and temperature ladder behind n = 60 should be given alongside
> those behind the n = 75 of §3.2; the Spearman coefficients relating harmonic to
> finite-temperature accuracy are computed across five models, where +0.15 to +0.65 is
> indistinguishable from noise; and the McNemar exact p = 0.34 is a failure to reject a null of no
> difference, which is not positive evidence for H2 and is currently read as though it were. A
> permutation test at the system level would be the appropriate substitute, and the honest summary
> of the finite-temperature ranking at this sample size is that it is suggestive rather than
> established.

**Response.** All adopted. The Wilson intervals and the system-level tests are in
`mlip_dynstab/stats.py`, run by `scripts/stats_hardening.py`, with the outputs deposited in
`results/stats_hardening.json`.

*Counts and intervals.* Rates are now given as k/n, with a Wilson 95% interval in the tables, in
Figs. 5 and 6 and at the main-text rates the argument turns on: the harmonic accuracies are
19/19 [0.832, 1.000], 17/19 [0.686, 0.971] and 15/19 [0.567, 0.915]. Some rates the text quotes
in passing carry their interval only in the ESI (the SSCHA high-temperature false-unstables, in
Table S17). I used Wilson rather than Wald intervals because several rates sit at 0 or 1, where a
Wald interval collapses to a point. Counts given only for completeness (the tolerance sweep, the
blow-up and failure tallies) carry none, and the head of ESI §S3 says so.

*The AUC and the composition of the sets.* Table S7 gives the composition: both sets rest on the
same 15 systems, n = 60 being 15 systems at four temperatures and n = 75 being 15 systems under five
models. (The filter that removes the bcc metals also removes the superionic AgI, which is why there
are 15 systems and not 16; §3.2 says so.) Table S8 lists every one of the 60 units. Every test
used as evidence on these sets now permutes or resamples whole systems, and the unit-level values
appear only as labelled companions. The vote-split AUC of 0.762 has a system-clustered permutation
p of 0.0035 and a cluster-bootstrap interval of [0.590, 0.934]; the naive unit-level p (0.0003)
overstated the significance by a factor of about 12. Without ORB-v2 the AUC falls to 0.628
(system-clustered p = 0.047) with an interval, [0.438, 0.839], that spans 0.5 (R2.2), and §3.4
also notes that part of the enrichment is built into the construction, since a unanimous unit can
be wrong only when all five models are.

*Spearman coefficients.* Removed from the manuscript. With five models there are 120 possible
rankings, and the coefficients change sign across the ladder: +0.645 at 100 K, −0.667 at 300 K,
−0.889 at 600 K and −0.167 at 900 K, with exact permutation p of 0.60, 0.41, 0.11 and 1.00, and no
value at 300 K or above could go below p = 0.107. They are deposited in
`results/stats_hardening.json` and are not used as evidence.

*McNemar.* The reviewed p = 0.34 was the 100 K value (7 against 3), and the reviewed text did not
say so; at 100 K it is now 0.727 (5 against 3). The revision reports every temperature. At 300 K
the counts are 17 against 4 (14 against 4 on the reviewed data), with a unit-level McNemar p of
0.007. The referee's objection applies with more force to that number, because the 75 pairs are
five models observed on each of fifteen systems. The
test is therefore now an exact system-level test that randomises the sign of each system's net
discordance: p = 0.844, 0.152, 0.066 and 0.414 at 100, 300, 600 and 900 K (1.0, 0.156, 0.055 and
0.25 without ORB-v2), not significant at any temperature, with a leave-one-system-out check in Table
S15. The unit-level McNemar p appears only as a labelled companion. The association φ is tested by
permuting whole systems (p above 0.08 at every temperature). No failure to reject is read as a
finding: the pre-registered form of H2 is left open, and the 100 K result is described as not
resolved.

*The ranking.* The manuscript goes one step beyond "suggestive": it makes no ranking claim at
finite temperature. The five accuracies run from 24/30 to 27/30 and every interval overlaps every
other (§3.2, §5).

The same correction applies to a comparison the referee did not raise, between the screen and SSCHA
on the displacive systems. The two methods see the same units, so the test is paired and clustered
by system: the screen is right where SSCHA is wrong on 33 units against 3, four of five systems
favour the screen, and the system-level p is 0.125, where the smallest value five systems can give
is 0.0625. §3.3 reports this as a large and consistent difference that is not significant, and ESI
Table S10 marks its unit-level p-values as not to be quoted.

**Changes.** §3.1 (table); §3.2 (both tables and the text after them); §3.3; §3.4; §5; Figs. 5 and
6; ESI §S3 (head), Tables S4–S10 and S15.

## R3.4 Tables S1–S4

> Supply Tables S1–S4. The ESI refers to per-model harmonic confusion matrices, per-model finite-T
> rates with and without bcc, per-system predicted T* against experiment, and the ensemble-guardrail
> table, but only the names of the functions that generate them appear; these carry the per-system
> numbers underlying every headline rate in the paper, and none of the results are checkable without
> them.

**Response.** The referee is right: the reviewed ESI named the functions, and the tables were never
rendered. They are now generated from the deposited ledger and analysis outputs by
`scripts/build_esi_tables.py`, whose `--check` option fails if the ESI is out of date with them. The
four requested tables have new numbers:

| Requested (reviewed ESI) | Now |
|---|---|
| Table S1, per-model harmonic confusion matrices | Table S4 |
| Table S2, per-model finite-T rates with and without bcc | Table S5 |
| Table S3, per-system predicted T* against experiment | Table S6 |
| Table S4, the ensemble-guardrail table | Table S8 |

Tables S1–S3 are now tables that sit in the section whose method they document (the harmonic
replicate in §S1.2, the SSCHA stage-by-stage diagnostic in §S2.2, and the SSCHA numerical outcome by
family in §S2.3), and a note at the head of the ESI separates section numbers (§S1–§S4) from table
numbers. The ESI also adds Table S7 (composition of the analysis sets), S9 (every pooled rate with
and without ORB-v2), S10 (the paired screen-versus-SSCHA tests), S11 (SSCHA settings and numerical
quality), S12 (the exact one-dimensional comparison, printed in §S1.4 before Table S2), S13
(displacement-amplitude sensitivity of the harmonic layer), S14 (sensitivity of the screen), S15 (the
H2 ladder clustered by system), S16 (screen–SSCHA agreement on bcc and the same-energy comparison
behind the SSCHA false-stables), S17 (SSCHA high-temperature false-unstables, blow-ups and
failures by model) and S18 (the pre-registered force-level ensemble test, R2.2).
It adds §S5 with Table S19 (the screen on PBE energies and the MLIP-versus-PBE well depths,
R1.1) and Table S20 (MLIP errors on SSCHA-sampled configurations, R1.2).
<!-- PENDING-C1: add the new seed-study and converged-grid tables to this list and to the Changes line below. -->
Per-unit values not tabulated are in the deposited ledger, from which every table regenerates. In
rendering Table S6 I also corrected its note: the screen does not systematically under-estimate T*
for the bcc metals, since in 4 of the 15 bcc rows it never calls the phase stable on the ladder.

**Changes.** ESI head note; §S3 (reading-order note, Tables S4–S11 and S13–S18); Tables S1–S3 in
§S1.2, §S2.2 and §S2.3; Table S12 in §S1.4.

## R3.5 Every rate with and without ORB-v2

> Report all rates with and without ORB-v2. The float32, non-conservative-force architecture is a
> confound for finite-difference force constants and for SSCHA stability rather than a model
> property of interest, and ORB-v2 accounts for its sole false-unstable at MgO (-2.8 THz), the
> softmode outliers near -35 THz on Ti and Hf, the six SSCHA blow-ups concentrated in its runs (down
> to -2x10^6 THz), and the outlying finite-T false-stable rate of 0.562; the with-and-without split
> is already performed for one bcc sign-agreement number and should be applied throughout, since the
> design cannot separate precision from architecture and the manuscript says so.

**Response.** Adopted, and it changed one conclusion. I agree that ORB-v2 is a confound this design
cannot resolve. Two of the report's premises have moved, and I should say so before describing the
split.

- *Precision.* As run, CHGNet, SevenNet-0 and MatterSim return float32 forces as ORB-v2 does; only
  MACE-MP-0 runs in float64. Precision therefore does not single ORB-v2 out, and the manuscript no
  longer attributes any failure to it. What distinguishes ORB-v2 is that its forces are predicted
  directly rather than as gradients of the energy (§2.2).
- *The ORB-v2 figures.* After the re-measurement its MgO false-unstable is −1.07 THz (it is still
  its only harmonic false-unstable); there are no screen values near −35 THz (its most negative bcc
  screen curvatures are −6.2 THz on Ti at 100 K and −8.9 THz on Ti at 900 K, both from fitted
  polynomials whose well lies only between sample points, and Hf is at −0.09 THz, the softest
  commensurate harmonic value of a model with no screened instability there); the SSCHA blow-ups
  (|f| > 50 THz) number eight and are spread over four models, three
  for ORB-v2 (SrTiO₃ at 300, 600 and 900 K), three for MatterSim, one for CHGNet and one for
  SevenNet-0; the −2×10⁶ THz value came from the superseded grid (SrTiO₃ with ORB-v2 at 600 K,
  which now returns −544.5 THz); ORB-v2 accounts for 4 of the 7 SSCHA runs that fail outright, all
  on PbTiO₃; and its finite-temperature false-stable rate is 5/16 = 0.312.

With the split applied throughout, one conclusion changes: the ensemble guardrail, whose vote-split
AUC falls from 0.762 to 0.628 with an interval spanning 0.5, so §3.4 now calls it suggestive and not
robust. The others hold. The bcc stability-call agreement stays at 0.69 (31/45 and 25/36). The
reviewed sign-agreement score is no longer used, because the screen's curvature is positive by
construction (R1.1). The H2 asymmetry keeps its direction and its conclusion (14 against 2 at
300 K, clustered p = 0.156). The screen-versus-SSCHA contrast has the same clustered p with and
without it (0.125; 26 against 3 without ORB-v2). In the same-energy comparison behind the SSCHA
false-stables the screen's free-energy comparison finds a lower displaced minimum on 39 of 50 units
without ORB-v2, against 46 of 57 with it, and the force-level test is split the same way (Table
S18). Two results single ORB-v2 out on the harmonic side: its replicate noise reaches 2.01 THz and
flips two calls, while no other model's call flips and their largest deviations are 0.536 THz for
MACE-MP-0, 0.0076 THz for CHGNet, 0.0010 THz for SevenNet-0 and 0.00057 THz for MatterSim (ESI
§S1.2); and in the displacement-amplitude sweep it is the only model whose calls on test systems
change (Table S13).

Per-model rates (Tables S4 and S5) cannot change when another model is removed, so the split is
applied to every pooled quantity: Tables S9 and S15–S18 give each one both ways, and the main text
gives the values without ORB-v2 beside the pooled results its argument uses (§3.1–§3.4).

**Changes.** §2.2; §3.1–§3.4; §4 (Limitations); ESI Table S9 (every pooled rate both ways, including
the guardrail), and the split in Tables S10, S13, S14, S15, S16, S17 and S18.

## R3.6 Figure order

> Confirm the figure order. Fig. 5 is discussed in §3.3 but printed after Fig. 6, which may be an
> artifact of the proof rather than of the submitted manuscript; figures should appear in order of
> first citation, and the production editor may wish to check this.

**Response.** It was in the submitted manuscript, not the proof: Fig. 6 was placed before Fig. 5 in
the source. All six figures now appear in order of first citation, with Fig. 5 placed after Fig. 4
in §3.3, and they are also supplied as separate numbered TIFF files at 600 dpi in that order.
Every caption except that of Fig. 1 changed in substance. Fig. 2 is a new figure: the reviewed
Fig. 2 plotted the screen's 100 K curvature observable, which the reviewed text cited in §3.1 as
though it showed harmonic frequencies; that observable is positive by construction (ESI §S1.3), so
its negative values were numerical. The new Fig. 2 shows what §3.1 argues from, the harmonic
minimum frequency of every system under every model, with the unstable calls boxed and the
disagreements with the reference label marked; it also puts the per-system harmonic numbers behind
Table S4 on the record (item 4). Fig. 3 no longer says that the SSCHA margins discriminate the models, and marks
the units with nothing to stabilise. Fig. 4 no longer describes SSCHA as a "gold standard" or says
the screen tracks it, and gives the call agreement. Fig. 5 no longer says SSCHA is less reliable
than the screen, and gives both recalls as counts with intervals. Fig. 6 adds the panel without
ORB-v2 and says that no signal could be resolved for the frequency spread.

**Changes.** Placement of Figs. 5 and 6; captions of Figs. 2–6.
<!-- PENDING-FIG: Figs. 2 and 4 still plot the screen's curvature including its numerical negatives (Fig. 4's quadrants imply a sign comparison the text no longer makes); the captions now say so. Decide whether to regenerate them (e.g. mark the fallback and fit-artefact cells, or drop the quadrant shading) with scripts/make_figures.py before the TIFFs are uploaded. -->

---

# Editor's comments

> If any of the reviewer reports below contain recommended references to be included in your
> manuscript, please check them carefully and only include those you think are relevant and help
> improve your manuscript.

I checked each of the suggested works against its published record and abstract, and included
those I judged relevant to the paper's argument (R1.3). The references added for fine-tuning
(R2.1) were checked in the same way.

The decision letter also asked for an Author contributions section in CRediT form, a linked ORCID
for the submitting author, and a choice on transparent peer review. The manuscript now has an
Author contributions section in CRediT roles, immediately above Conflicts of interest and
Acknowledgements. My ORCID (0009-0003-0041-1459) is on the title page and linked to the submitting
account. I have opted in to transparent peer review. The author name, affiliation and funding
statement have been checked.
<!-- PENDING-P4: before upload, link the ORCID to the submitting account in the portal and finish the affiliation, funding and AI-use checks (tasks/todo.md P4); if either is not done, rewrite the two sentences above to say what was done. -->

# Other changes

- The harmonic layer's sensitivity to the finite-displacement amplitude is measured for all five
  models (0.005, 0.02 and 0.03 Å against the production 0.01 Å; §3.2, ESI Table S13). MACE-MP-0,
  MatterSim and SevenNet-0 change no call; CHGNet changes calls only on controls, so its harmonic
  accuracy runs from 14/19 to 17/19; ORB-v2 changes calls on test systems.
- The MACE-MP-0 reference (ref 6) now cites its published version (*J. Chem. Phys.*, 2025, **163**,
  184110).

# Files uploaded

1. This point-by-point response.
2. The revised manuscript with the changes marked (Word tracked changes against the version the
   referees read).
3. The revised manuscript as a clean .docx with the figures embedded.
4. The revised ESI, clean, and with the changes marked against the version the referees read.
5. Figs. 1–6 as separate numbered TIFF files at 600 dpi.
6. The table of contents entry: a graphic within 8 cm × 4 cm and a text of at most 250 characters.

Yours sincerely,

Frank Cai
Purdue University, West Lafayette, Indiana, USA
