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
- relates the SSCHA false-stables found after a converged and replicated relaxation (27 of 77
  label-unstable non-bcc units with a resolved call, all oxide perovskites; 23 of 73 without the
  four SrTiO₃ units at 100 K, whose cell has no size test; seven more unresolved between
  replicates) to the local question its default criterion asks, and reports that the fluorite
  false-stables and the high-temperature false-unstables of the first recipe were unconverged
  relaxations (R1.1, R1.2, R1.4);
- adds PBE single points along the screen's soft-mode coordinates and on SSCHA-sampled
  configurations (R1.1, R1.2), and a four-seed SSCHA study with recorded convergence diagnostics
  (R1.4);
  on PBE energies at the MLIPs' lattices the shared BaTiO₃ and KNbO₃ mis-calls disappear (MLIP
  wells there about half the PBE depth) and the CsSnBr₃ ones persist, though mostly not with PBEsol, so they follow the
  functional; MLIP force error does not grow on the 100 K SSCHA samples and rises 1.7-fold at
  600 K in SrTiO₃; the production SSCHA recipe did not converge on any of the 16 seeds on which
  convergence was recorded (cumulative step cap, placeholder gradient error), and re-run to convergence on 178 units (159 converged) the BaTiO₃ and KNbO₃
  false-stables remain while the fluorite false-stables and the high-temperature runaway do not;
  a pre-registered replicate of 65 units (second start, second seed) gives the same call on 43 of
  the 51 with three values and leaves 8 unresolved, which the counts now report as such (R1.4);
  at its own lattice PBE makes the BaTiO₃ and KNbO₃ 300 K errors itself on the two modes
  profiled (its X-point mode was not; R1.1);
- adds the literature on active-learning potentials and ML-driven screening (R1.3);
- completes the screen's derivation and reports its sensitivity, including the frozen-cell
  convention that decides its ferroelectric recall (R1.5);
- reports a pre-registered fine-tuning trial (R2.1), whose central prediction was refuted, and a
  pre-registered force-level ensemble test (R2.2), which did not show a signal;
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
| SSCHA recall, same units (R3 summary) | 7/30 | production recipe 5/27 [0.08, 0.37] (one blow-up and two failed runs excluded); converged recipe 4/22 [0.07, 0.39], with the screen at 14/22 [0.43, 0.80] on those units (two failed, two unconverged and four replicate-unresolved runs excluded; 4/26 against 15/26 from one start; paired, system-clustered p = 0.5) |
| H2 association φ (R3.1) | +0.11, the 100 K value (the text did not give the temperature) | +0.149 at 100 K; −0.129 at 300 K (−0.114 on the reviewed data); not resolved at any temperature |
| H2 McNemar exact p (R3.3) | 0.34, the 100 K value (7 v 3) | 0.727 at 100 K (5 v 3); at 300 K 0.007 unit level (17 v 4; 0.031 and 14 v 4 on the reviewed data), 0.152 clustered by system |
| Per-model Spearman ρ, harmonic vs finite-T (R3.1, R3.3) | +0.15 to +0.65 | removed (R3.3) |
| Vote-split AUC (R3 summary, R3.3) | 0.745 | 0.762 [0.590, 0.934]; 0.628 [0.438, 0.839] without ORB-v2 |
| Frequency-spread AUC (R3 summary) | 0.52 | 0.361 [0.046, 0.625] |
| bcc screen–SSCHA rank correlation ρ (reviewed Section 3.3) | 0.78 | 0.11, descriptive only |
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
(J. Yang, Z. Yin, L. Ao and S. Li, *Phys. Chem. Chem. Phys.*, 2026, **28**, 4459, ref 4; D. Li,
J. Yang, X. Chen, L. Yu and S. Liu, *J. Phys. Chem. C*, 2025, **129**, 21538, ref 5), so in
August, before the reports, I changed it to one saying that finite-temperature dynamic stability
separates models that harmonic benchmarks rank equally (commit 0fbea7d, 16 August 2026). The
re-measured data do not support that title either, since the finite-temperature accuracies are not
separated, and Referee 3 asked that the title follow the formulation the data support (R3.1). The
revised title makes two existence claims, neither of which rests on a significance test, so
clustering by system cannot overturn them: a top harmonic score does not certify a model's
finite-temperature calls (MatterSim, MACE-MP-0 and SevenNet-0 are harmonically perfect on the
matched set, and the screen still mis-calls 5, 5 and 3 of their 30 units), and the default SSCHA
cross-check, even converged, does not supply that certification for deep displacive wells (Section 3.3).
The PBE checks of R1.1 change what the first clause is about, and I have kept the title but
made the text say so. Eight of the 17 units behind it (BaTiO₃ and KNbO₃ at 300 K) are called
correctly on PBE energies along the same coordinates at the MLIPs' lattices, but at its own,
slightly smaller lattice PBE calls the same two systems stable at 300 K on the two modes profiled
(the X-point mode was not), so the MLIP wells are shallow relative to PBE at the MLIP geometry,
not relative to PBE at its own equilibrium, and the pre-registered fine-tuning trial of R2.1 did
not correct these calls either. Four of the five CsSnBr₃ units among the 17 are called correctly
with PBEsol on the same structures, so they follow the functional, and three more are PbTiO₃, the
screen's ordering failure. The first clause is therefore a statement about the finite-temperature
calls a screen makes with a model, the model and the screen together, and for eight of the 17
units the reference functional at its own equilibrium gives the same call; Sections 3.2, 4 and 5 now
say this, and do not read the count as a difference between the MLIPs and their reference. The converged
SSCHA re-run (R1.4) keeps the second clause for the ferroelectric perovskites and removes the
fluorites from it: converged, SSCHA still calls BaTiO₃ and KNbO₃ stable against their labels at
100 K in every model, and calls every fluorite unit unstable.

### (c) What was changed before the reports arrived

Some changes were made in August and early September for reasons of my own and bear on items the
referees raise. I list them here so that none of them is presented as a response:

- 16 August 2026, commits 4c8e478 and 0fbea7d: the phrase "methodological trap" removed from the
  Abstract and Section 3.3 (R1.4), and the SSCHA result related to the published analyses of the SSCHA
  free-energy Hessian (R. Bianco *et al.*, ref 38; L. Monacelli, ref 44). In 0fbea7d, the sentence
  "The finite-temperature results confirm H2 in a strong form" removed from Section 4 (R3.1), the phrase
  "necessary but not sufficient" removed from the Abstract, Sections 1 and 3.2 (R3.1), and the title
  changed as in (b).
- 16 August 2026, commit a7b375f: an Author contributions section added and the back matter put
  in the journal's order (Editor's comments).
- 16 August 2026, commit e7375cc: the variational derivation of the screen in Section 2.4 and the
  approximation ledger of Table 1 (R1.5, in part; ESI Section S1.3 now completes it).
- 10 September 2026, commit dc11ed7: the harmonic replicate-noise measurement (ESI Section S1.2, Table
  S1) and a system-clustered permutation test for the association φ (R3.3, in part).

The phrase "non-predictive" that R1.6 and R3.1 quote was still in Section 4 after these changes, and it is
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
All 981 calculations finished, covering the deciding paths and every other screened mode in
cells of up to 12 atoms (93 of 139 paths), and the outcome splits by system (new Sections 2.6 and 3.2;
ESI Section S5, Table S19). (i) Along the coordinates that decide BaTiO₃, KNbO₃ and ZrO₂, the wells of CHGNet,
MACE-MP-0, MatterSim and SevenNet-0 are shallower than PBE's (0.31–0.76 of the depth for the first
two systems, 0.45–0.61 for ZrO₂) at the same minimum position; for SrTiO₃'s R tilt they are
0.25–0.99 of it; for CsSnBr₃ three of the five match PBE within 20 %; and along MatterSim's
deciding coordinate in bcc Zr, PBE has a 159–174 meV well where CHGNet, MACE-MP-0 and SevenNet-0
have none. (ii) The shared 300 K mis-calls on BaTiO₃ and KNbO₃ disappear on PBE: all eight units
are called unstable, as labelled, so at the MLIPs' geometries they belong to the MLIP surfaces
(but see the PBE-lattice check at the end of this response). The CsSnBr₃ mis-calls
persist, with the PBE minimum at the edge of the scan on every deciding path and a 300 K label 8 K
above the transition, and so does KNbO₃ at 600 K (T_c 708 K), called stable by all five models and
by PBE.
(iii) Over the 120 ladder units of the six systems, swapping only the energy engine raises
agreement with the labels from 81 to 101 (23 corrected, 3 newly wrong; no system worse); ten of
the corrections are bcc Zr, and the 55 modes beyond the deciding ones change no call. (iv) On the
errors that persist I ran three further checks for BaTiO₃, KNbO₃ and CsSnBr₃ (382 calculations
in all, 323 for the convergence and functional checks and 59 for the PBE-lattice check below;
ESI Section S5.3, Table S23). Denser k-points and higher cutoffs change no call at any temperature; by
the criterion I fixed in advance the production settings still fail, because the denser k-mesh
lowers the CsSnBr₃ well depth, a value at the edge of the scan, by 6.8 %. PBEsol on the same
structures corrects 8 of the 9 CsSnBr₃ PBE errors (7 of the 8 where the MLIP call is wrong too)
and introduces none, so those errors follow the
functional and are not shown to be the screen's; this shows that the call is sensitive to the
functional, not that PBEsol is right. KNbO₃ at 600 K persists under both functionals for four of
the five models. The counts are descriptive (six systems, PBE at each model's lattice, which is
0.1–0.8 % above the PBE lattice for the three perovskites checked).

*At the PBE lattice.* The calculations above put PBE at each MLIP's lattice and along its
eigenvector, which is what separates the force engine from the screen but not what PBE predicts
at its own equilibrium, so I also computed PBE force constants at the PBE lattice (finite
differences, 2×2×2) and re-solved the screen on PBE E(Q) along PBE's own softest band at the
q-point that decides MACE-MP-0's call, and along its softest mode (59 further calculations; ESI
Section S5.3, Table S23). The result goes against the paper's first reading of these units. At its own
lattice PBE's deciding wells are 48.7 meV on BaTiO₃ and 30.0 meV on KNbO₃, against 73.5 and
87.8 meV at MACE-MP-0's lattice along MACE-MP-0's eigenvector, and PBE then calls both systems
stable at 300 K, the MLIPs' own error; MACE-MP-0's wells at its own lattice (42.6 and 27.9 meV)
are close to PBE's at PBE's. The MLIP lattices are only 0.26–0.31 % (BaTiO₃) and 0.74–0.80 %
(KNbO₃) larger than PBE's, so these 300 K calls sit on a knife edge, and the eight corrections
hold at the MLIPs' lattices only. The check changes lattice and eigenvector together and does not
say which of the two moves the call, and only the two profiled modes were tested: PBE at its own
lattice has a third imaginary mode, at X (−6.63 THz in BaTiO₃, −5.47 THz in KNbO₃, between the two
profiled ones in frequency), which was not profiled, so PBE's own 300 K call is stable on the
modes tested and not established. KNbO₃ at 600 K
and CsSnBr₃ at 300 and 600 K are mis-called by PBE at both lattices. Sections 3.2, 4, 5 and the Abstract
now state the eight corrections, and PBE's own-lattice call, with these qualifiers.

*Global stability against local curvature.* This distinction is now the centre of Section 3.3. For the
anharmonic systems the labels record which phase is the equilibrium phase at T, a global question
about the free energy. SSCHA's default criterion, the free-energy Hessian at the symmetric
reference, asks a local one: is the symmetric phase a local minimum? For a deep double well
treated with a Gaussian trial state the free energy can keep a local minimum at the symmetric
point while a displaced minimum lies lower, and a curvature read at the reference then reports
stable. On the same MLIP energies the screen finds that situation on half of the units in
question: of the 27 non-bcc units where converged SSCHA calls the phase stable against an unstable
label in every finished replicate, the screen's free-energy comparison, the minimum over the
centroid (Section 2.4), finds a lower displaced minimum and calls the phase unstable on 15 (15/27 = 0.56
[0.37, 0.72]; 12/23 without ORB-v2; 17/34 from one start, before seven units became unresolved;
with the production recipe 46/57). The
screen's own curvature at the symmetric point adds nothing here, and Section 2.4 and ESI Section S1.3 now say
why: for a single mode with an even potential it equals the self-consistent trial stiffness MΩ²,
so it is positive whenever a bound Gaussian exists and cannot register condensation. That the
screen finds a lower displaced minimum on these units shows that the local and the global question
have different answers on these energies. It does not by itself show that SSCHA's positive
Hessian has the same origin; a relaxation that stopped short of the SCHA minimum was the
alternative, and R1.4 shows it was the cause on the fluorites and not on BaTiO₃ and KNbO₃. Because the comparison is made on one surface, it also says
nothing about which criterion matches first principles. Section 2.1 and the Limitations now also say
that the labels mark thermodynamic, often first-order, transitions, whereas both methods probe
dynamic stability, so a label and a dynamic-stability call can differ near a transition without
either being wrong.

*Agreement on one surface.* Section 2.5 now states that SSCHA inherits the MLIP potential-energy surface
exactly as the screen does, and Section 3.3 and the caption of Fig. 5 describe screen–SSCHA agreement as a
consistency check. The descriptions of SSCHA as a gold standard and of the screen as tracking it
are gone. The bcc agreement is also reported more carefully than before: the stability calls agree
in 31/45 = 0.69 [0.54, 0.80]; 12 of the agreeing units are MACE-MP-0 and CHGNet on Zr and Hf, where
neither model has a harmonic instability to stabilise (Fig. 4 now marks them); and MatterSim, whose
bcc instabilities are the deepest, agrees with production SSCHA on the call in none of its 9
units (Table S16); with converged SSCHA in the 3×3×3 cell the calls agree in 33/40 units with a
resolved call (33/41 from one start) and MatterSim in 7/9 (Tables S22 and S24), although for MatterSim the cell itself changes the call (Section 3.5). The reviewed text compared the two methods' frequencies directly. For the reason just given,
the screen's curvature is not scored against SSCHA by sign: the ten bcc units on which it is
negative are all numerical (six fall-backs of the width solver, one fit artefact, and three units
in which the model has no screened instability; Table S16). The screen's calls are scored against
the reference labels (experimental for the anharmonic systems, the DFPT phonon database for the six
stable controls), not against SSCHA. Its SrTiO₃ gate, which on the 100/300 K ladder brackets the
105 K transition without locating it, is passed by three of the five models (Section 2.4).

**Changes.** Section 2.1; Section 2.4 ("Criterion and observable", SrTiO₃ gate); Section 2.5; Section 3.3 (the bcc paragraphs,
and "Why the default criterion misses these wells"); Section 4 (Limitations); captions of Figs. 2, 4 and
5; ESI Section S1.3 ("What the symmetric-point curvature is", last paragraph), Sections S2.2, S4, Table S16;
`scripts/curvature_identity_check.py`.
New Section 2.6 (first-principles reference) and the PBE paragraph of Section 3.2; new Fig. 3 (the PBE
call at its own lattice and at the MLIP's, with the well depths behind it); ESI Sections S5.1, S5.3,
Table S19 and Table S23.

## R1.2 MLIP force-engine errors against method failures

> Differentiate MLIP Force Engine Errors from Anharmonic Method Failures: The reported SSCHA
> false-stabilization may stem from MLIP errors when sampling thermally accessible, highly distorted
> configurations out-of-distribution, rather than an intrinsic failure of the SSCHA formalism
> itself. The authors should benchmark MLIP forces/energies against first-principles reference
> calculations along the sampling paths.

**Response.** I have run the benchmark the referee describes, with one limitation I state first:
the configurations come from the SSCHA ensembles of the production recipe, which did not converge
(R1.4), and I did not repeat the check on the converged ensembles, so the false-stables that
survive convergence (BaTiO₃ and KNbO₃ at 100 K) have no first-principles check on their own
sampled configurations. The benchmark is PBE energies and forces on
configurations drawn from the SSCHA ensembles of four units, compared with the same MLIP-vs-PBE
errors on near-equilibrium rattled configurations of the same cells. The configurations are
subsamples of the final Hessian ensembles of the production recipe, re-run with four seeds in the
study described under R1.4. The units are BaTiO₃ and ZrO₂ with MACE-MP-0 at 100 K, where SSCHA
false-stabilises; SrTiO₃ with MACE-MP-0 at 600 K, where the production recipe returns −20.2 THz
against a harmonic −2.37 THz (+2.44 THz converged); and bcc Zr with MatterSim at 50 K, a bcc unit that carries a harmonic instability.
All 68 calculations finished (ESI Section S5.2, Table S20). At 100 K (0.09 Å root-mean-square
displacement, against 0.03 Å for the rattled baseline) MACE-MP-0's force error relative to the PBE
forces is 0.10 for BaTiO₃ and 0.18 for ZrO₂, against 0.10 and 0.23 near equilibrium, so the error
does not grow on these samples; bcc Zr behaves the same way, though its relative error is a ratio
on small forces. These ensembles come from relaxations that did not converge (R1.4), so at 100 K
they may not reach the double well, and a small error on them does not clear the MLIP inside it.
At 600 K in SrTiO₃ (0.34 Å) the relative error rises from 0.11 to 0.19, energy errors reach
44 meV per atom, and all five models fall at 0.12–0.20: the MLIPs are extrapolating where the
runaway happens; the converged relaxation of the same unit is stable (+2.44 THz) and never reaches
those configurations (below).

Two things in the MLIP-only data bear on the hypothesis, one for each direction of SSCHA error.

*The false-stables.* The reviewed text attributed them to the formalism. That attribution is
withdrawn, and Section 3.3 now states that none of its results is a claim that the SSCHA formalism is
wrong; they concern the question its default criterion asks. The revised account is narrower: the
default criterion is local, and on the same energies the screen's free-energy comparison finds a
lower displaced minimum on 15 of the 27 converged false-stables with a resolved call (R1.1). That account, if it
holds, needs no MLIP error on the sampled configurations, but it does not exclude one, and the
manuscript says so. Nor did it exclude a relaxation that stopped short of the SCHA minimum, which
R1.4 shows was the cause on the fluorites. The
reviewed text also presented the stage-by-stage diagnostic of ESI Table S2 (numbered as in the
revised ESI) as isolating the mechanism, with the fourth-order term as the remedy. The one run
that changes only the truncation, with the fourth-order term, has now finished on the seed-0
ensemble of the production recipe: +2.878 THz against +2.878 THz at bubble level on the same
ensemble (2.7 h against 11 s), so on this unit the fourth-order term is not the remedy. Section 3.3 and
ESI Section S2.2 now say so.

*The high-temperature false-unstables.* The reviewed version did not discuss them, and it should
have, because they are the pattern the referee's hypothesis predicts. Over the non-bcc units whose
label is stable, production SSCHA calls the phase unstable in 5/5 at 300 K, 10/13 at 600 K and 15/19 at 900 K
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
The converged recipe settles it for this unit and for the grid: SrTiO₃ with MACE-MP-0 converges
to +2.44 THz at 600 K and +2.71 THz at 900 K in the converged-mode runs (+2.41 and +2.76 THz in the
grid; stable, as labelled), and over the converged grid the
false-unstables fall to 0/4, 3/12 and 3/18 at 300, 600 and 900 K, all CsSnI₃; no unit stable at
100 K turns negative with temperature, and every converged SrTiO₃ unit above 105 K is stable
(0/11). The runaway was the unconverged relaxation, which the MLIP's extrapolation error on the
configurations it reached may have amplified; the MLIP-error hypothesis is not needed for it, and
Section 3.3 no longer presents it as the signature the referee describes.

**Changes.** Section 3.3 ("Why the default criterion misses these wells", "High-temperature
false-unstables were the unconverged recipe", "Numerical failures", now summarised there with the
full list in ESI Section S2.3); Fig. 7; Section 4 (Limitations); ESI Section S2.1,
Section S2.2 (Table S2), Sections S2.3, S2.4 (SrTiO₃), Tables S16, S17 and S22.
New Section 2.6; Section 3.3 ("Why the default criterion misses these wells" and "High-temperature
false-unstables were the unconverged recipe", last sentences); Section 4; ESI Section S5.2 and Table S20.

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
Lett.*, 2019, **122**, 225701, ref 6), the temperature-driven transitions and anharmonic thermal
transport of zirconia (C. Verdi *et al.*, *npj Comput. Mater.*, 2021, **7**, 156, ref 7) and the
α–β transition of zirconium (P. Liu *et al.*, *Phys. Rev. Mater.*, 2021, **5**, 053804, ref 8),
which are the perovskite, fluorite and bcc families of this paper. Those are system-specific
potentials trained on configurations from the target's own dynamics, and their success places the
gap measured here with foundation models used as shipped, not with machine-learned potentials in
general. The Introduction now says so. Also added: an active-learning screen aimed at MLIP failures
in strongly anharmonic materials (K. Kang *et al.*, ref 9); Gaussian approximation potentials (A.
P. Bartók *et al.*, ref 10); on-the-fly Bayesian force fields whose predictive variance triggers
new reference data (J. Vandermause *et al.*, ref 13); committee uncertainty propagated into
molecular dynamics (G. Imbalzano *et al.*, ref 17); scalable sparse Gaussian-process regression
with data-efficient on-the-fly sampling, which reproduced the melting and glass-crystallisation
temperatures of the Li₇P₃S₁₁ solid electrolyte (A. Hajibabaei, C. W. Myung and K. S. Kim, *Phys.
Rev. B*, 2021, **103**, 214102, ref 11), and its use to screen lithium diffusion across hundreds
of candidate solid electrolytes and to merge the per-crystal models into transferable Li–P–S and
Li–Sb–S potentials (A. Hajibabaei and K. S. Kim, *J. Phys. Chem. Lett.*, 2021, **12**, 8115, ref
12); the review of sparse Gaussian-process potentials and their library for batteries, solar
cells, catalysts and macromolecular systems (S. Y. Willow *et al.*, *Chem. Phys. Rev.*, 2024,
**5**, 041307, ref 15); the Account by M. Ha, S. Pourasad, C. W. Myung and K. S. Kim
(*Acc. Chem. Res.*, 2026, **59**, 103, ref 14), which covers sample efficiency (practical accuracy
from 100–1000 quantum calculations), uncertainty-triggered sampling and the robust Bayesian
committee machine; and the review of MLIPs for energy materials by I. K. Park *et al.* (*Adv. Energy
Mater.*, 2026, e71046, ref 16), which sets Gaussian-process frameworks with uncertainty
quantification beside the equivariant graph networks and foundation models tested here and lists
foundation-model fine-tuning among the emerging directions.

*ML-assisted high-throughput screening.* Added GNoME (A. Merchant *et al.*, ref 18), M3GNet (C.
Chen and S. P. Ong, ref 19) and Matbench Discovery (J. Riebesell *et al.*, ref 20), and the use of
sparse Gaussian-process potentials to accelerate high-throughput screening (ref 14). The
Introduction now places the soft-mode screen as one filter in such a pipeline, for candidates that
pass the convex-hull check but show imaginary harmonic modes. Matbench Discovery scores stability
against the 0 K convex hull and the phonon benchmarks score harmonic stability; neither asks about
the finite-temperature dynamic stability a screened candidate needs, which is the gap this paper
measures. I have kept the discussion to the screening of crystal stability; catalyst discovery is
outside what this benchmark can speak to.

I also considered the other works the comment points to, and the paper cites those that bear
directly on its subject, the finite-temperature dynamic stability of the crystal families studied
here, or on the uncertainty and screening questions it raises. Of the works the comment names, one
is not cited: the 2025 *Chem. Phys. Rev.* paper on a sparse Bayesian committee machine potential
for oxygen-containing organic compounds (S. Y. Willow *et al.*, *Chem. Phys. Rev.*, 2025, **6**,
021401). Its systems are molecular, gas-phase to solid organic compounds, and do not include the
inorganic crystal families or the phase transitions studied here; the committee-machine method it
applies is cited through ref 14. Against the version the referees read, the reference list has
grown from 18 entries to 48. The additions made before the reports
are the two finite-temperature studies behind the title change, the two analyses of the SSCHA
Hessian, and the four sources of the Section 2.4 derivation; three cite the codes of the new
first-principles reference (R1.1); the rest were added for this comment and for Referee 2's first
point.

**Changes.** Section 1 (third and fourth paragraphs); Section 4 ("On model-reported uncertainty"); refs 6–20.

## R1.4 Convergence diagnostics for SSCHA

> Rigorous Convergence Diagnostics for SSCHA: The claim that SSCHA is a "methodological trap"
> requires stronger justification. The authors must report standard convergence metrics: sample
> sizes, gradient history, stopping criteria, and Hessian uncertainties. The four-seed stochastic
> test should be expanded beyond bcc-Zr to include representative displacive systems (BaTiO3 or
> fluorites).

**Response.** *"Methodological trap".* I have withdrawn the claim rather than tried to justify it.
The phrase was removed in August (see (c) above), and what replaces it in Section 3.3 is narrower: the
default criterion is local, and on the same energies the screen's free-energy comparison finds a
lower displaced minimum on 15 of the 27 units converged SSCHA false-stabilises in every finished
replicate (46 of 57 with the production recipe; R1.1). A relaxation that stops short of the SCHA minimum can also report stable,
and the convergence metrics the referee asks for are what decide between the two. They decide it
both ways: early stopping on the fluorites, the criterion on BaTiO₃ and KNbO₃ (below).
The run with the fourth-order term (`include_v4 = True`) on the unit of ESI Table S2,
BaTiO₃/MACE-MP-0/100 K, finished within its 4 h cap (9,850 s, against 11 s at bubble level) on the
seed-0 production ensemble, and gives +2.878 THz against +2.878 THz at bubble level, so the call
does not change; Section 3.3 and ESI Section S2.2 now say so.

*Sample sizes and stopping criteria.* These are now stated in Section 2.5, ESI Section S2.1 and Table S11. The
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
so the cap bounds the steps over the whole relaxation. Section 2.5, ESI Section S2.1 and Table S11 now say so.
The re-run with recorded histories (R1.4 below) shows what the cap did: on all 16 seeds of the
four units the first population was ended after 19 steps, each later population took one step and
discarded it, and the convergence test was never met, so the free-energy Hessian was evaluated
close to the positive-definite starting matrix (within 0.24 THz of it on BaTiO₃, ZrO₂ and bcc Zr).
I therefore re-ran SSCHA with a recipe that converges: at most 400 steps within each population,
the library's stochastic gradient error in place of the placeholder, populations of 1,000
configurations up to 30, and a 2,000-configuration Hessian ensemble with bootstrap resampling, on
every unit a claim of Section 3.3 rests on (178 units). 171 finished and 159 met the stopping test (11
ORB-v2 units reached the population cap, one CHGNet unit the wall cap, and the seven units that
failed in production failed again). The converged values change 72 of the 154 calls that have a
converged value and a production value that is not a blow-up (75 of the 159 if the production
blow-ups count as unstable); on the non-bcc units 51 change from wrong to right against the label
and 12 from right to wrong. The consequences, carried into R1.1, R1.2, Section 3.3 and the abstract, with the replicates below
applied: converged SSCHA calls 27 of the 77 label-unstable non-bcc units with a resolved call
stable, every one an oxide perovskite (18 surviving from production, all on BaTiO₃ and KNbO₃, and
9 new under convergence: PbTiO₃ at 300–600 K, SrTiO₃ at 100 K and one KNbO₃ unit); the fluorite
false-stables and the high-temperature false-unstables do not survive; and the ferroelectric
recall is 4/22 for converged SSCHA against 14/22 for the screen on the same units (ESI Tables S22
and S24). (From the first start alone, before the replicates, these were 34 of 84, 21 surviving
and 13 new, and 4/26 against 15/26.)

*What the production grid did not record.* It did not record how many populations each unit used,
whether each unit met the convergence threshold before the population cap, the gradient history,
or an uncertainty on the Hessian. 2,560 is therefore the configured maximum, not a count, and Section 2.5
and Table S11 say so rather than reconstruct the missing values.

*The seed study.* The reviewed four-seed test (+1.798 ± 0.001 THz) was weaker than the text
implied: it ran on bcc Zr with MACE-MP-0, which has no harmonic instability in Zr, so it measured
SSCHA's stochastic noise on a surface with nothing to stabilise, and Section 3.5 now says so. To supply the
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
by python-sscha 1.6.1, so the library's own convergence test is not a stochastic one (ESI Section S2.1).
With the converged recipe, from two starting matrices each: BaTiO₃ +2.03 and +1.85 THz (still
stable against an unstable label), ZrO₂ −22.4 and −26.0 THz (now unstable, as labelled), bcc
Zr/MatterSim at 50 K +0.41 THz from both, and SrTiO₃/MACE-MP-0 at 600 K +2.44 THz from the first
start (stable, as labelled). The seed agreement of the production recipe was therefore agreement on
where it stopped, not convergence. Across the converged grid each unit reports its stopping
reason, the number of populations, the gradient against its stochastic error, a bootstrap SD of
the Hessian minimum, and a check of the final gradient on a fresh ensemble, which 152 of the 159
converged units pass (ESI Table S22).

*Replicates of the converged grid.* A bootstrap over one ensemble is not a Hessian uncertainty
across runs, so before running any replicate I registered a protocol for one
(`tasks/preregistration-repeats-2026-10-03.md`; its one deviation, dated before any replicate
existed, fixes the random draw as without replacement): every grid unit whose converged call
disagrees with its comparison (53) and 12 of the rest drawn at random, each re-run from a second
starting matrix (imaginary modes at +0.3 THz instead of |ω|, re-run once from 1.0 THz after the
cellconstructor complex-dynamical-matrix assertion) and with a second seed; a call that differs
between replicates is counted as unresolved, not by majority, and no replicate is dropped. The
outcome is mixed, and Section 3.5, ESI Section S2.4 and Table S24 report it in full. The second seed finished
on all 65 units. The second start did not: 19 units hit the assertion at 0.3 THz and 7 of them
failed again at 1.0 THz, 6 stopped at other cellconstructor symmetry errors (not retried, as
registered) and one wrote no result, so 51 units have three values. On those the call is the same
in 43 (43/51 = 0.84 [0.72, 0.92]; 32/39 without ORB-v2), with a median range of 0.22 THz in the
lowest Hessian frequency. Eight are unresolved. Seven are PbTiO₃ and KNbO₃ with CHGNet and
MatterSim at 300–600 K (PbTiO₃ with MatterSim at 600 K failed under both recipes), where the second start ended at its wall or population cap at −209 to
−6230 THz, unconverged blow-ups, while the first start and the second seed agree on a converged
stable call; the eighth is Ti with ORB-v2 at 600 K (+2.53 THz from the first start, −26.4 and
−3.73 THz, both unconverged, from the other two). As registered, Section 3.3 now counts these as
unresolved: the converged false-stables become 27 of the 77 label-unstable non-bcc units with a
resolved call (34/84 from one start; 23/67 without ORB-v2), the ferroelectric recall 4/22 for SSCHA
against 14/22 for the screen, the paired contrast 12 against 2 (p = 0.5, unchanged) and the bcc
agreement 33/40. Two observations were not registered and are labelled as such in Section 3.5: the
second seed alone reproduces the first start's call on 64 of the 65 units, and on the 31 units
that converged in all three replicates every call agrees. On the ten BaTiO₃ and KNbO₃ units at
100 K that carry the local-criterion account, the second seed gives a converged stable call on all
ten and the second start a stable call on the seven where it finished. One of the ten is
marginal: BaTiO₃ with CHGNet reads +1.65 THz on the full Hessian ensemble, but its bootstrap
(B = 10 resamples) has a mean of −0.33 THz and 6 of the 10 resamples are unstable; Section 3.3 now says
so. No resample is unstable on the other nine.

**Changes.** Section 2.5; Section 3.3; new Fig. 7 (the converged call on every unit of the grid, the
units the replicates leave unresolved hatched); Section 3.5 ("Stochastic reproducibility"); Section 4
(Limitations); ESI Section S2.1,
Section S2.2 (Table S2), Section S2.4, Table S11.
ESI Table S21 (new: the seed study and the bcc cell-size re-measurement) and ESI Table S22 (new:
the converged-recipe grid beside the production values).

## R1.5 Formalisation and sensitivity of the soft-mode screen

> Formalization and Sensitivity of the Soft-Mode Screen: The mathematical formulation of the
> single-mode quantum SCHA screen needs complete derivation in the ESI (mode mass-weighting,
> Gaussian width self-consistency equation, and mode-mode coupling limitations). Sensitivity to
> polynomial fit windows, sampling ranges, and cell commensurability must be systematically
> reported.

**Response.** The derivation is now complete in ESI Section S1.3, which holds the derivation added to
Section 2.4 in August, extended; Section 2.4 keeps the free energy, the width and the
self-consistency condition, and points to it.

- *Mode mass-weighting.* The effective mass M = Σᵢ mᵢ|**u**ᵢ|² follows from the kinetic energy of
  the frozen pattern. Section S1.3 also fixes what the reviewed text left open: the pattern is normalised
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
  Section S1.3 states that stationarity is shown, not that the stationary point is a minimum in Ω. ESI Section S1.4
  (Table S12) solves the same fitted potentials exactly: an isolated mode's thermal density stays
  bimodal at every temperature, so the screen's temperature dependence comes from the
  self-consistency, and the screen never condenses a mode whose exact density is unimodal. Section S1.3
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
  of the same triplet 16.8 meV with 41 amu. Sections 2.4 and S1.3 now say so.

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
  across the other conventions, and at 100 K three conventions give no excess at all (ESI Section S1.3,
  `scripts/h2_by_convention.py`). This weakens the screen, and the Abstract, Sections 3.2, 3.3, 5 and
  the Limitations say so. The other part, the force-constant supercell
  that fixes which **q**-points are searched, was not enlarged; Section 2.4 and ESI Section S4 say so, and Table 1
  lists it as approximation A2. Cell size for SSCHA is under R3.2.

**Changes.** Section 2.4 ("Definitions and numerical settings", with its constants now in ESI
Section S1.3, "Criterion and observable", "Frozen-cell normalisation" and the paragraph after it); Table 1 (A3, A7); Section 3.2; Section 4 (Limitations); Section 5; ESI
Sections S1.1, S1.3, S1.4 (Table S12), Table S14, Section S4.

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
ORB-v2 on SrTiO₃, which finds no instability at any temperature (Section 3.2). An error four or five
architectures share could be the single-mode screen's own approximation or a surface error the
models have in common, and the PBE-backed screen of R1.1 tests which. On PBE energies along the
same coordinates the eight BaTiO₃ and KNbO₃ units are called correctly, so at the MLIPs' lattices
they are a softening four models share, but at its own, slightly smaller lattice PBE makes the
same 300 K error on the two modes profiled (R1.1); the five CsSnBr₃ units stay mis-called on PBE,
and four of those five are called correctly with PBEsol on the same structures, so they follow the
functional rather than the screen. Most of the count is therefore not a difference between the
MLIPs and their reference functional, and the H2 statement concerns the calls a screen makes with
a model. The revised Sections 3.2, 4 and 5 say so.

The Abstract, the H2 statement in Sections 1, 3.2, 4 and the Conclusions carry this wording, and none of
them reads a failure to reject as a finding.

**Changes.** Abstract; Section 1 (H2); Section 2.4 ("Frozen-cell normalisation"); Section 3.2 (Table 4, the temperature-ladder
table, and the paragraphs after it); Section 4; Section 5; ESI Section S1.3, Table S15.

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
Along the soft-mode paths, at each model's lattice, the four conservative models' wells are about
half as deep as PBE's in BaTiO₃, KNbO₃ and ZrO₂, and three models have no well at all in bcc Zr;
on the SSCHA samples the force error does not grow at 100 K and rises 1.7-fold, with energy errors
to 44 meV per atom, at 600 K in SrTiO₃.

## R2.1 Fine-tuning

> The work focuses exclusively on a stability argument and does not mention how fine-tuning would
> affect the results, which most experts would assume is needed for these systems.

**Response.** The Abstract now says that the models are tested as shipped, Sections 2.2, 4 and the
Conclusions say that none was fine-tuned outside the trial described below, and Section 4 says why. That is the regime of generative-CSP
screening: a practitioner filtering candidates does not know in advance which of them are strongly
anharmonic, so fine-tuning on the candidates themselves is not available before the screen, as
the referee's report notes for crystal structure prediction. Fine-tuning on broad phonon data is
available, and ref 45 below deploys such a model at screening scale; this benchmark does not test
one, and Section 4 now names a phonon-fine-tuned checkpoint as the natural next model to put through it.
For a known anharmonic material I agree that fine-tuning is the right step, and Section 4 now cites what
it has been shown to do: universal MLIPs, CHGNet and MACE-MP-0 among them, systematically soften the potential-energy surface, and
fine-tuning on a small number of reference calculations removes much of that softening (B. Deng
*et al.*, *npj Comput. Mater.*, 2025, **11**, 9, ref 46); fine-tuning a universal potential on
harmonic phonon data cut its phonon-frequency error about fourfold, and the tuned model was then
used for SSCHA across 4669 inorganic compounds (H. Lee *et al.*, *Small*, 2026, **22**, e00071, ref
45); partially frozen fine-tuning of MACE-MP matched a potential trained from scratch with 10–20% of
its data and reproduced the phonon dispersions of α-, β- (bcc) and ω-Ti qualitatively (M. Radova
*et al.*, *npj Comput. Mater.*, 2025, **11**, 237, ref 47); and on PbTiO₃, fine-tuning MACE on a
small PBEsol dataset recovered the ferroelectric–paraelectric transition, with the Curie temperature
still about 160 K below experiment (ref 5). One further study bears on both of the referee's
points: fine-tuning four of the five checkpoints tested here (MACE-MP-0, SevenNet-0, a large
MatterSim checkpoint and ORB-v2) on system-specific first-principles data cut their force errors
5–15-fold and brought the architectures to comparable accuracy (J. Hänseroth, A. Flötotto, M. N.
Qaisrani and C. Dreßler, *J. Phys. Chem. Lett.*, 2026, **17**, 3152, ref 48). It tested diffusion,
structure and energy pathways rather than phonons, and Section 4 cites it with that restriction; it also
implies that cross-architecture disagreement is a signal of the models as shipped (R2.2).

Section 4 then sorts the paper's findings by whether fine-tuning could change them. It could change the
per-model inputs: the harmonic calls (the flattened Zr and Hf instabilities of MACE-MP-0 and CHGNet
are the kind of softening these studies address), the well depths the screen reads, and so the
per-model screen accuracies and the H2 counts built from them. Whether it would remove the
mis-calls shared across models, which carry most of the H2 count, is what the PBE-backed screen of
R1.1 tests, since it re-solves the screen on the surface a PBE fine-tune aims at, along the same
coordinates (the mode pattern still comes from each MLIP's force constants).
The PBE-backed screen only says what the PBE surface gives along each MLIP's coordinates, so I
also tested fine-tuning itself, in a trial registered before any data were computed
(`tasks/preregistration-finetune-2026-10-03.md`; Section 4 and ESI Section S6, Tables S25 and S26). MACE-MP-0
and CHGNet were each fine-tuned three times toward PBE on 120 PBE-labelled thermal configurations
of BaTiO₃, KNbO₃ and CsSnBr₃ and tested on held-out PBE data along the screen's coordinates; a call
counts as changed only if all three replicates agree. The predictions and their outcomes:

- The ferroelectric well depths would come within 20 % of PBE. Supported for MACE-MP-0 (median
  fine-tuned/PBE ratio 0.99, base 0.45); unresolved for CHGNet (pooled 1.04, one replicate 1.33).
- The BaTiO₃ and KNbO₃ 300 K mis-calls would be corrected. **Refuted** in all four (system, model)
  cells: every replicate of both models still calls both stable at 300 K.
- The KNbO₃ 600 K and CsSnBr₃ 300 K mis-calls would persist. Supported; CsSnBr₃ at 600 K persists
  for MACE-MP-0 and is unresolved for CHGNet.
- Converged SSCHA on a fine-tuned MACE-MP-0 would still call BaTiO₃ stable at 100 K. Supported
  (+1.73 THz).
- The six stable controls would keep their harmonic calls. Supported for MACE-MP-0; four of six
  unresolved for CHGNet.

Fine-tuning also cost MACE-MP-0 a correct call, CsSnBr₃ at 900 K, in all three replicates, which
no prediction covered. One step was not executed as registered: the first MACE-MP-0 fine-tunes
trained 6 epochs, not 30, because their run script predated the epoch budget. I recorded that as
a deviation before re-running MACE-MP-0 at 30 epochs on the same data and seeds; the 30-epoch run
is the registered result, the 6-epoch run gave the same verdicts, and both are in Table S25.

A check made after the results, and labelled as such, locates the refuted prediction. Trained
without stresses, the 30-epoch MACE-MP-0 replicates relax BaTiO₃ and KNbO₃ about 0.5 % smaller than
the base model, to within 0.3 % of the PBE lattice, and there all six replicate and system pairs
call 300 K stable; at the base model's lattice, where the held-out PBE data sit, all six call it
unstable. PBE does the same at its own lattice (R1.1). A fine-tune faithful to PBE therefore
inherits PBE's own 300 K errors at the PBE lattice, and the estimate the previous draft of Section 4
made, that a PBE fine-tune would remove about half of the shared count, is withdrawn. The trial
does not show that fine-tuning cannot fix these calls: it covers two models, three systems, data
at one lattice without stresses, and for CsSnBr₃ a reference functional that is itself wrong.
Fine-tuning still cannot change the locality of the SSCHA criterion, nor the screen's dependence
on its frozen-cell convention, and it moves a model toward its reference functional, not toward
experiment. How much of the finite-temperature gap fine-tuning closes in general, for each family
and with how much data, is beyond what this benchmark can say; Section 4 says that too.

**Changes.** Abstract; Section 2.2; Section 4 ("Scope: foundation models as shipped": the trial, replacing the
estimate of how much a fine-tune would remove, "On model-reported uncertainty"); Fig. 3 (the fine-tunes at
their own and at the base model's lattice); Section 5; refs 5 and
45–48; ESI Section S6 and Tables S25 and S26; the registration, its deviations and all trial data are in
the deposited repository (`results/revision/finetune/`, `results/revision/finetune_mace30/`,
`scripts/finetune_lattice_diag.py`).

## R2.2 Ensemble uncertainty of the force predictions

> While for crystal structure prediction you would not necessarily have the prior knowledge to make
> that assumption, there is no mention of what the ensemble uncertainty of the force predictions
> are. This metric can be used as a reliable measure of where the MLIP predictions should not be
> trusted.

**Response.** I agree that a model's own uncertainty is the natural trust metric, and Section 4 now sets
out the literature behind it (committee spreads propagated into molecular dynamics, ref 17;
predictive variance as a trigger for new reference data, refs 13 and 14; uncertainty-driven active
learning that avoids unphysical dynamics in strongly anharmonic materials, ref 9). The released
checkpoints come as single models rather than trained committees, so I tested the idea in three
forms on the same 60 consensus units (15 systems at four temperatures; ESI Tables S7 and S8),
asking in each case whether the signal flags the units where the five models' majority call is
wrong.

*The vote split across architectures (Section 3.4).* With all five models, the consensus is wrong on 8/14
units where the models split and on 4/46 where they agree, and the split predicts consensus error
with AUC 0.762 [0.590, 0.934] (system-clustered p = 0.0035). Without ORB-v2 it falls to 0.628
[0.438, 0.839], an interval that spans 0.5, with three 2–2 ties broken to "stable" (two of them
wrong). It is a suggestive guardrail, not a robust one, and the paper now says so; the reviewed
version reported it with all five models only.

*The spread of the screen's frequency across models (Section 3.4).* No signal could be resolved (AUC 0.361
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

**Changes.** Section 3.4 (the ORB-v2 split, the tie rule, the part of the enrichment built into the
construction, the frequency spread as a PES-level proxy, the force-level test and its result); Section 4
("On model-reported uncertainty"); Section 5; Fig. 8 (both model sets); ESI Tables S8, S9 and S18.

---

# Referee 3

Thank you for a report that read the manuscript and its supplement closely enough to find a
contradiction between two sections, an ESI that named tables it did not contain, and statistics that
treated clustered units as independent. All three were right. Several numbers in the report's
summary have since moved (table in (a) above).

## R3.1 Section 4 against Section 3.2

> Reconcile §4 with §3.2; the Discussion asserts the hypothesis that the Results reject. Section
> 3.2 tests the strong form of H2, states explicitly that it is rejected, and reports a weak
> positive association (φ = 0.11, per-model ρ = +0.15 at T ≤ 300 K rising to +0.65 at T = 100 K),
> settling on "necessary but not sufficient"; Section 4 then states that the results "confirm H2 in
> a strong form" and that "harmonic accuracy is non-predictive of finite-temperature accuracy,"
> which is the opposite of a positive correlation. The §3.2 formulation is the one the data support
> and should be carried consistently into the Discussion, the abstract, and the framing of the
> title.

**Response.** Correct: Section 4 contradicted Section 3.2. "Confirm H2 in a strong form" was removed on 16 August
(commit 0fbea7d), before the report; "non-predictive" remained in Section 4 and is removed now.

I have to add that the Section 3.2 formulation the referee endorses does not survive a reading of the
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
The PBE checks (R1.1) locate that shared error: the BaTiO₃ and KNbO₃ mis-calls disappear on PBE
energies along the same coordinates at the MLIPs' lattices but return with PBE at its own lattice
on the modes profiled, and the CsSnBr₃ ones persist on PBE but four of the five turn correct with
PBEsol. The title is kept, and the text reads its first clause as a statement about the
finite-temperature calls a screen makes with a model, not about the model's surface alone.

**Changes.** Title; Abstract; Section 1 (H2); Section 2.4; Section 3.2; Section 4; Section 5; ESI Section S1.3.

## R3.2 Zone-boundary convergence

> Complete the deferred 4×4×4 SSCHA for SrTiO₃ and one fluorite, or remove all convergence claims
> for zone-boundary systems. The manuscript concedes that the SrTiO₃ antiferrodistortive
> instability at R = (½,½,½) is commensurate only with even supercells and that the 3×3×3
> comparison is therefore invalid for it, which leaves the R-point and X-point systems with no
> convergence test at all, while the 2×2×2 cell used throughout is marginal for the ZrO₂/HfO₂
> X-point mode; the argument that the §3.3 false-stable is not a missing-q artifact because it also
> occurs for the Γ mode of BaTiO₃ establishes this for one system only and does not extend to the
> fluorites or to SrTiO₃.

**Response.** I have now attempted the first option and, where it did not finish, kept the
second. The 4×4×4 run was pre-registered before any unit ran
(`tasks/preregistration-4x4x4-2026-10-09.md`, commit f47cc2a): the four SrTiO₃ units that
converged SSCHA calls stable at 100 K, and the fluorite unit closest to a sign change, HfO₂ with
MACE-MP-0 at 600 K, each with the converged recipe of R1.4 unchanged except for the supercell
(320 atoms for SrTiO₃) and a 9 h cap per unit. For the fluorite the relaxation converged and the
lowest free-energy-Hessian frequency is −3.22 THz in 4×4×4 against −2.20 THz in 2×2×2, so its
call is unchanged. That is one start, and the unit reached the cap before its bootstrap, so it has
no uncertainty in 4×4×4. For SrTiO₃ no unit reached a 4×4×4 Hessian within the cap: CHGNet's
relaxation converged and the run stopped while computing the Hessian, and the other three had not
converged after 10 to 14 populations. SrTiO₃ therefore still has no cell-size test. Section 3.5
and ESI Section S2.4 give the five results, state that no supercell-convergence claim is made for
the R-point and X-point systems, and state their SSCHA calls for the 2×2×2 cell only. I agree that
the 2×2×2 cell is marginal for the fluorite X-point mode: it is the smallest cell that contains X,
and 3×3×3 does not; the one 4×4×4 unit checks one call and is not a convergence test of the
fluorite frequencies. The Limitations repeat this.

Since the first version of this reply the SSCHA has been re-run with a recipe that converges
(R1.4), and that changes two of the points I would otherwise make here. First, the fluorite
false-stable the referee was concerned about does not survive convergence: converged SSCHA calls
every fluorite unit unstable in the 2×2×2 cell, as labelled, so the paper no longer rests anything
on a fluorite SSCHA false-stable. Second, I had argued that for SrTiO₃ the objection does not arise
because SSCHA never called it stable; that is no longer true. Converged SSCHA calls SrTiO₃ stable at
100 K, 5 K below its transition, for all four models where it converged (+1.06 to +1.20 THz). The
R point is in the 2×2×2 cell, so this is not a missing **q**-point, but whether the R mode's
finite-temperature renormalisation is converged in that cell is exactly the question the referee
raises, and Section 3.5 now says that this call may be a finite-size effect; the 4×4×4 attempt above did not
reach a call for any of these units, so it remains untested. Those four units are also in the
headline count of converged SSCHA false-stables, so Section 3.3 and the Conclusions give that count
without them as well: 23 of 73, all on the three ferroelectrics. For BaTiO₃ the zone-centre
argument stands, restricted to BaTiO₃, as the referee says.

On bcc Zr, the cell-size comparison has been re-measured in the pinned environment with MatterSim,
which does carry a harmonic instability, as well as with MACE-MP-0, which does not. With MACE-MP-0
the call holds; with MatterSim, converged from two starts, it changes sign between 2×2×2
(+0.92 THz) and 3×3×3 (−0.92 THz) at 300 K. The two cells are not nested (N is commensurate only
with even cells, ω only with multiples of three), so Section 3.5 reports this as a change of **q**-set,
not a convergence test, and as a reason not to read the 2×2×2 bcc margins as converged.

**Changes.** Section 3.5 ("Finite size", "Zone-boundary systems", now with the 4×4×4 attempt);
Section 4 (Limitations); ESI Sections S2.4 (new item "The 4×4×4 attempt", with its five results), S4,
Tables S17, S21 and S22.

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
Figs. 6 and 8 and at the main-text rates the argument turns on: the harmonic accuracies are
19/19 [0.832, 1.000], 17/19 [0.686, 0.971] and 15/19 [0.567, 0.915]. Some rates the text quotes
in passing carry their interval only in the ESI (the SSCHA high-temperature false-unstables, in
Table S17). I used Wilson rather than Wald intervals because several rates sit at 0 or 1, where a
Wald interval collapses to a point. Counts given only for completeness (the tolerance sweep, the
blow-up and failure tallies) carry none, and the head of ESI Section S3 says so.

*The AUC and the composition of the sets.* Table S7 gives the composition: both sets rest on the
same 15 systems, n = 60 being 15 systems at four temperatures and n = 75 being 15 systems under five
models. (The filter that removes the bcc metals also removes the superionic AgI, which is why there
are 15 systems and not 16; Section 3.2 says so.) Table S8 lists every one of the 60 units. Every test
used as evidence on these sets now permutes or resamples whole systems, and the unit-level values
appear only as labelled companions. The vote-split AUC of 0.762 has a system-clustered permutation
p of 0.0035 and a cluster-bootstrap interval of [0.590, 0.934]; the naive unit-level p (0.0003)
overstated the significance by a factor of about 12. Without ORB-v2 the AUC falls to 0.628
(system-clustered p = 0.047) with an interval, [0.438, 0.839], that spans 0.5 (R2.2), and Section 3.4
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
other (Sections 3.2, 5).

The same correction applies to a comparison the referee did not raise, between the screen and SSCHA
on the displacive systems. The two methods see the same units, so the test is paired and clustered
by system: the screen is right where converged SSCHA is wrong on 12 units against 2 (13 against 2
from one start, before the replicates of R1.4 left four units unresolved), with the net
difference on two of the five systems (BaTiO₃ and KNbO₃), and the system-level p is 0.5, where the smallest value five
systems can give is 0.0625; with the production recipe it was 33 against 3 and p = 0.125. Section 3.3
reports this as a difference that is not significant, and ESI Table S10 gives both recipes and
marks its unit-level p-values as not to be quoted.

**Changes.** Section 3.1 (table); Section 3.2 (both tables and the text after them); Section 3.3; Section 3.4; Section 5; Figs. 6, 7 and
8; ESI Section S3 (head), Tables S4–S10 and S15.

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
replicate in Section S1.2, the SSCHA stage-by-stage diagnostic in Section S2.2, and the SSCHA numerical outcome by
family in Section S2.3), and a note at the head of the ESI separates section numbers (Sections S1–S4) from table
numbers. The ESI also adds Table S7 (composition of the analysis sets), S9 (every pooled rate with
and without ORB-v2), S10 (the paired screen-versus-SSCHA tests), S11 (SSCHA settings and numerical
quality), S12 (the exact one-dimensional comparison of Section S1.4), S13
(displacement-amplitude sensitivity of the harmonic layer), S14 (sensitivity of the screen), S15 (the
H2 ladder clustered by system), S16 (screen–SSCHA agreement on bcc and the same-energy comparison
behind the SSCHA false-stables), S17 (SSCHA high-temperature false-unstables, blow-ups and
failures by model) and S18 (the pre-registered force-level ensemble test, R2.2).
It adds Section S5 with Table S19 (the screen on PBE energies and the MLIP-versus-PBE well depths,
R1.1), Table S20 (MLIP errors on SSCHA-sampled configurations, R1.2) and Table S23 (convergence,
functional and lattice checks on the errors that persist on PBE, R1.1). It adds Table S21 (the
production-recipe SSCHA seed study and bcc-Zr cell size, R1.4) and Table S22 (the converged-recipe
SSCHA grid beside the production values, R1.4), Table S24 (the pre-registered replicates of the
converged grid, R1.4) and Tables S25 and S26 (the pre-registered fine-tuning trial, R2.1). Every
table is printed in numerical order.
Per-unit values not tabulated are in the deposited ledger, from which every table regenerates. In
rendering Table S6 I also corrected its note: the screen does not systematically under-estimate T*
for the bcc metals, since in 4 of the 15 bcc rows it never calls the phase stable on the ladder.

**Changes.** ESI head note; Section S3 (reading-order note, Tables S4–S26); Tables S1–S3 in
Sections S1.2, S2.2 and S2.3; Section S5; Section S6.

## R3.5 Every rate with and without ORB-v2

> Report all rates with and without ORB-v2. The float32, non-conservative-force architecture is a
> confound for finite-difference force constants and for SSCHA stability rather than a model
> property of interest, and ORB-v2 accounts for its sole false-unstable at MgO (−2.8 THz), the
> softmode outliers near −35 THz on Ti and Hf, the six SSCHA blow-ups concentrated in its runs
> (down to −2×10⁶ THz), and the outlying finite-T false-stable rate of 0.562; the with-and-without
> split is already performed for one bcc sign-agreement number and should be applied throughout,
> since the design cannot separate precision from architecture and the manuscript says so.

**Response.** Adopted, and it changed one conclusion. I agree that ORB-v2 is a confound this design
cannot resolve. Two of the report's premises have moved, and I should say so before describing the
split.

- *Precision.* As run, CHGNet, SevenNet-0 and MatterSim return float32 forces as ORB-v2 does; only
  MACE-MP-0 runs in float64. Precision therefore does not single ORB-v2 out, and the manuscript no
  longer attributes any failure to it. What distinguishes ORB-v2 is that its forces are predicted
  directly rather than as gradients of the energy (Section 2.2).
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
AUC falls from 0.762 to 0.628 with an interval spanning 0.5, so Section 3.4 now calls it suggestive and not
robust. The others hold. The bcc stability-call agreement stays at 0.69 (31/45 and 25/36). The
reviewed sign-agreement score is no longer used, because the screen's curvature is positive by
construction (R1.1). The H2 asymmetry keeps its direction and its conclusion (14 against 2 at
300 K, clustered p = 0.156). The screen-versus-SSCHA contrast has the same clustered p with and
without it (converged, p = 0.5; 12 against 2 with it, 9 against 2 without). In the same-energy
comparison behind the converged SSCHA false-stables the screen's free-energy comparison finds a
lower displaced minimum on 12 of 23 units without ORB-v2, against 15 of 27 with it, and the
ferroelectric recall on the units where converged SSCHA returned a resolved value is 4/19 for
SSCHA against 11/19 for the screen without ORB-v2 (4/22 against 14/22 with it). The replicates of
R1.4 are split the same way: the call is the same in 32 of 39 complete units without ORB-v2 and 43
of 51 with it (ESI Table S24). The force-level test is split
the same way (Table S18). The converged SSCHA grid is the ORB-v2 caveat in practice: every one of
its 11 runs that reached the population cap and all four of its blow-ups (down to −33,506 THz) are
ORB-v2 (Table S22). Two results single ORB-v2 out on the harmonic side: its replicate noise reaches 2.01 THz and
flips two calls, while no other model's call flips and their largest deviations are 0.536 THz for
MACE-MP-0, 0.0076 THz for CHGNet, 0.0010 THz for SevenNet-0 and 0.00057 THz for MatterSim (ESI
Section S1.2); and in the displacement-amplitude sweep it is the only model whose calls on test systems
change (Table S13).

Per-model rates (Tables S4 and S5) cannot change when another model is removed, so the split is
applied to every pooled quantity: Tables S9 and S15–S18 give each one both ways, and the main text
gives the values without ORB-v2 beside the pooled results its argument uses (Sections 3.1–3.4).

**Changes.** Section 2.2; Sections 3.1–3.4; Section 4 (Limitations); ESI Table S9 (every pooled rate both ways, including
the guardrail), and the split in Tables S10, S13, S14, S15, S16, S17 and S18.

## R3.6 Figure order

> Confirm the figure order. Fig. 5 is discussed in §3.3 but printed after Fig. 6, which may be an
> artifact of the proof rather than of the submitted manuscript; figures should appear in order of
> first citation, and the production editor may wish to check this.

**Response.** It was in the submitted manuscript, not the proof: Fig. 6 was placed before Fig. 5 in
the source. The revision has eight figures, and all of them now appear in order of first citation,
with the recall figure placed after the method-agreement figure in Section 3.3; they are also
supplied as separate numbered TIFF files at 600 dpi in that order. Two figures are new, so the
numbers below are the revised ones: reviewed Figs. 1 and 2 keep their numbers, reviewed Figs. 3,
4, 5 and 6 are now Figs. 4, 5, 6 and 8, and the new Figs. 3 and 7 show the PBE lattice check and
the fine-tuned models at two lattices (R1.1, R2.1) and the converged SSCHA call on every unit of
the grid (R1.4, R3.3). The point the referee raised, the recall figure printed before the
method-agreement figure, concerned reviewed Figs. 5 and 6.
Every caption except that of Fig. 1 changed in substance. Fig. 2 is a new figure: the reviewed
Fig. 2 plotted the screen's 100 K curvature observable, which the reviewed text cited in Section 3.1 as
though it showed harmonic frequencies; that observable is positive by construction (ESI Section S1.3), so
its negative values were numerical. The new Fig. 2 shows what Section 3.1 argues from, the harmonic
minimum frequency of every system under every model, with the unstable calls boxed and the
disagreements with the reference label marked; it also puts the per-system harmonic numbers behind
Table S4 on the record (item 4). Fig. 4 no longer says that the SSCHA margins discriminate the
models, marks the units with nothing to stabilise, and now adds the converged 3×3×3 values beside
the production curves, with the one converged unit that the replicates leave unresolved (Ti with
ORB-v2 at 600 K) marked separately. Fig. 5 no longer describes SSCHA as a "gold standard" or says
the screen tracks it, and gives the call agreement; it now plots the SSCHA frequency against the
screen's stability call instead of against the screen's curvature, since that curvature is
positive by construction and its negative values are numerical. Fig. 6 no longer says SSCHA is less reliable than
the screen, gives both recalls as counts with intervals, and now plots converged SSCHA (4/22)
against the screen on the same 22 units (14/22), the units where the replicates of R1.4 leave the
call resolved, with the production value (5/27) beside them. Fig. 8 adds the panel without
ORB-v2 and says that no signal could be resolved for the frequency spread.

**Changes.** Placement of Figs. 6 and 8 (reviewed Figs. 5 and 6); captions of Figs. 2 and 4–8; Figs. 2,
3 and 7 new; Figs. 4, 5 and 6 regenerated. The three main-text tables that had no number now have numbers and captions
(Tables 2–4, in order of first citation).

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
Acknowledgements. My ORCID (0009-0003-0041-1459) is on the title page and is to be linked to the submitting
account in the portal at upload. My choice on transparent peer review is the one made in the
portal at upload. The author name, affiliation and funding statement in the manuscript were
checked against the title page and back matter.
<!-- PENDING-P4: before upload, link the ORCID to the submitting account in the portal and finish the affiliation, funding and AI-use checks (tasks/todo.md P4); if either is not done, rewrite the two sentences above to say what was done. -->

# Other changes

- The harmonic layer's sensitivity to the finite-displacement amplitude is measured for all five
  models (0.005, 0.02 and 0.03 Å against the production 0.01 Å; Section 3.2, ESI Section S1.2 and Table S13). MACE-MP-0,
  MatterSim and SevenNet-0 change no call; CHGNet changes calls only on controls, so its harmonic
  accuracy runs from 14/19 to 17/19; ORB-v2 changes calls on test systems.
- The MACE-MP-0 reference (ref 29) now cites its published version (*J. Chem. Phys.*, 2025, **163**,
  184110).
- The references are now numbered in order of first citation, as the journal's style requires;
  the numbers in this letter follow the new list. Ref 27 (α-AgI) cited the wrong work by the same
  authors and now cites B. C. Wood and N. Marzari, *Phys. Rev. Lett.*, 2006, **97**, 166401.
- The pre-registration of H1–H3 is now dated in Section 1: the research brief committed before any
  calculation (commit 9c2599f, 20 June 2026).
- A statement on the use of AI tools is added at the end of the Acknowledgements, in line with the
  journal's policy.
<!-- DRAFT-FOR-FRANK: the AI-use statement in the manuscript is Option A of paper/response/ai_use_statement_DRAFT.md; confirm or replace its wording, and keep or drop this bullet accordingly. -->
- Two figures are new and carry results the revision added: Fig. 3 shows the lattice knife edge
  behind the PBE check and the fine-tuning trial (R1.1, R2.1), and Fig. 7 maps converged SSCHA's
  call on every unit of the grid, with the units its replicates leave unresolved hatched (R1.4,
  R3.3). To keep the paper readable the main text is shorter by about 2,900 words: methods detail and
  robustness material that predate the revision (the screen's full derivation and numerical
  constants, the harmonic replicate study, the T* comparison, the production-recipe bcc caveats,
  the full list of SSCHA failures) now stand in the ESI as they were written, each with a pointer
  in the main text, and results stated several times are stated once. Nothing added in answer to
  the referees left the main text.
- The PBE first-principles reference now carries its own convergence, functional and lattice checks
  (Section 2.6 (iii), ESI Section S5.3, Table S23), and the SSCHA results of Section 3.3 rest on a converged re-run of
  the SSCHA grid (Section 2.5, ESI Table S22).

# Files submitted with this response

<!-- PENDING-B1: build the clean .docx files and the marked-changes manuscript and ESI (scripts/build_docx.py, scripts/build_marked_changes.py against 76d3a84) after the last text edit; drop any item below that is not uploaded. -->
1. This point-by-point response.
2. The revised manuscript with the changes marked (Word tracked changes against the version the
   referees read).
3. The revised manuscript as a clean .docx with the figures embedded.
4. The revised ESI, clean, and with the changes marked against the version the referees read.
5. Figs. 1–8 as separate numbered TIFF files at 600 dpi.
6. The table of contents entry: a graphic within 8 cm × 4 cm and a text of at most 250 characters.

Yours sincerely,

Frank Cai
Purdue University, West Lafayette, Indiana, USA
