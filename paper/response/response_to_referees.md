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
changed what the paper claims. In brief, the revision:

- states H2 only as strongly as a system-clustered test allows, and claims no model ranking (R1.6, R3.1, R3.3);
- adds PBE calculations along the screen's soft-mode coordinates and on SSCHA-sampled
  configurations, including a check at PBE's own lattice that goes against the paper's first reading (R1.1, R1.2);
- re-runs the SSCHA grid (178 units) with a recipe that converges, with pre-registered replicates;
  the fluorite false-stables and the high-temperature false-unstables do not survive, the
  ferroelectric false-stables do (R1.4);
- adds the requested literature, the screen's full derivation and its sensitivity (R1.3, R1.5);
- reports a pre-registered fine-tuning trial, whose central prediction was refuted, and a
  pre-registered force-level ensemble test, which showed no signal (R2.1, R2.2);
- attempts the 4×4×4 SSCHA, gives rates as counts with intervals, clusters every test by system,
  splits every pooled rate by ORB-v2, renders the ESI tables and orders the figures by citation (R3.2–R3.6).

## Before the point-by-point answers

### (a) Several numbers the referees quote have moved, most of them against the paper

In mid-August, before the reports arrived, an audit of my own deposit found that the deposited
results could not be reproduced by the deposited code: the hash that decides whether a unit is
already computed omitted the algorithm version, so units from a superseded **q**-point search were
kept, and the acoustic-mode masks removed the three lowest branches instead of the three nearest
zero, which for an unstable phase removes the soft mode under test. I re-measured every layer in
pinned per-model environments; the Data availability section describes both defects, and the
version the referees read (commit 76d3a84) is kept in the repository beside the revision.
<!-- PENDING-P5: name the release tag, commit and Zenodo version DOI that contain paper/submissions/rsc-advances-2026-07/, once the branch is merged and the version is minted. -->

| Quantity (where quoted) | As reviewed | Now |
|---|---|---|
| Harmonic accuracy, 19 systems (R3.3) | 1.000, 0.895, 0.789 | 19/19, 17/19, 15/19 (unchanged) |
| Finite-T accuracy, T ≤ 300 K: CHGNet, MACE-MP-0, MatterSim | 0.933, 0.933, 0.833 | 26/30, 25/30, 25/30 |
| Screen recall, ferroelectrics, T ≤ 300 K | 23/30 | 16/30 [0.36, 0.70] |
| SSCHA recall, same units | 7/30 | converged recipe 4/22 [0.07, 0.39] against the screen's 14/22 on the same units |
| H2 φ (R3.1) | +0.11 (100 K, unlabelled) | +0.149 at 100 K; −0.129 at 300 K; not resolved at any temperature |
| H2 McNemar p (R3.3) | 0.34 (100 K) | 0.727 at 100 K; at 300 K 17 v 4, system-clustered p = 0.152 |
| Vote-split AUC | 0.745 | 0.762 [0.590, 0.934]; 0.628 [0.438, 0.839] without ORB-v2 |
| Frequency-spread AUC | 0.52 | 0.361 [0.046, 0.625] |
| ORB-v2 figures (R3.5) | MgO −2.8 THz; −35 THz outliers; six blow-ups to −2×10⁶ THz; FS rate 0.562 | −1.07 THz; none; eight blow-ups over four models (three ORB-v2); 5/16 |

Intervals are Wilson 95% for rates and cluster-bootstrap 95% over systems for AUCs. The
finite-temperature accuracies (24/30 to 27/30) no longer separate the models, so the crossover in
Referee 3's summary is not resolved.

### (b) The title

In August, before the reports, I changed the title because two studies in print had examined
finite-temperature stability for single model families (refs 4 and 5). The re-measured data do not
support that second title either, and Referee 3 asked that the title follow what the data support
(R3.1). The revised title makes two existence claims that need no significance test: a top harmonic
score does not certify a model's finite-temperature calls (the screen mis-calls 5, 5 and 3 of 30
units for the three harmonically perfect models), and the default SSCHA cross-check, even
converged, does not supply that certification for deep displacive wells. After the PBE checks of
R1.1 the text reads the first clause as a statement about the calls a screen makes with a model,
not about the MLIP surface alone.

### (c) What was changed before the reports arrived

On 16 August (commits 4c8e478, 0fbea7d, a7b375f, e7375cc) I removed "methodological trap",
"confirm H2 in a strong form" and "necessary but not sufficient", changed the title, added the
Author contributions section and the screen's variational derivation; on 10 September (dc11ed7) I
added the harmonic replicate-noise study and a system-clustered test of φ. "Non-predictive" was
still in Section 4 and is removed now. Everything else below was done after the reports arrived.

---

# Referee 1

> However, the manuscript's primary claims currently exceed what the study design establishes.
> Crucially, both the single-mode screen and SSCHA are driven by the same underlying MLIP
> potential energy surface (PES), making their agreement a consistency test rather than an
> absolute measure of physical accuracy.

I agree. The revision withdraws three claims the design did not support (harmonic accuracy is
non-predictive; the harmonic leaders are not the finite-temperature leaders; MLIP-driven SSCHA is a
methodological trap), describes screen–SSCHA agreement as a consistency check on one surface, and
adds first-principles calculations that test the surface itself.

## R1.1 Independent first-principles validation

> Establish Independent First-Principles Validation: Because both the single-mode screen and SSCHA
> rely on the MLIP force engine, disagreement between them cannot definitively establish which
> method is correct. Experimental phase transition temperatures are governed by global
> thermodynamic stability, not necessarily local free-energy curvature. The authors must
> incorporate independent first-principles (e.g., DFT or high-level reference) evaluations along
> soft-mode coordinates or anharmonic PES profiles for a key subset of systems (such as SrTiO3,
> BaTiO3, or bcc-Zr).

**Response.** *First-principles evaluations.* I added PBE single points along the screen's own
soft-mode coordinates for the three systems named and for KNbO₃, CsSnBr₃ and ZrO₂ (981
calculations; all five MLIPs on identical geometries; the screen re-solved on the PBE-fitted
potential). (i) Along the deciding coordinates the four conservative models' wells are shallower
than PBE's (0.31–0.76 of the depth in BaTiO₃ and KNbO₃, 0.45–0.61 in ZrO₂), and in bcc Zr PBE
has a 159–174 meV well where three models have none. (ii) Swapping only the energy engine raises
agreement with the labels from 81 to 101 of 120 units (23 corrected, 3 newly wrong): the shared
300 K mis-calls on BaTiO₃ and KNbO₃ disappear; those on CsSnBr₃ and KNbO₃ at 600 K persist.
(iii) On the persisting errors denser k-points and higher cutoffs change no call, and PBEsol
corrects 8 of the 9 CsSnBr₃ PBE errors, so those follow the functional.

*At the PBE lattice.* The calculations above place PBE at each MLIP's lattice. At PBE's own lattice
(59 further calculations) its deciding wells are 48.7 meV (BaTiO₃) and 30.0 meV (KNbO₃) instead of
73.5 and 87.8 meV, and PBE then calls both stable at 300 K, the MLIPs' own error. The MLIP lattices
are only 0.26–0.80 % larger, so these calls sit on a knife edge and the eight corrections hold at
the MLIP lattices only (new Fig. 3). PBE's X-point mode at its own lattice was not profiled, so its
own 300 K call is stable on the modes tested, not established.

*Global stability against local curvature.* This is now the centre of Section 3.3. The labels record
the equilibrium phase, a global question; SSCHA's default criterion, the Hessian at the symmetric
reference, asks a local one. On the same MLIP energies the screen's free-energy comparison finds a
lower displaced minimum on 15 of the 27 converged SSCHA false-stables (15/27 = 0.56 [0.37, 0.72];
12/23 without ORB-v2), so the local and global questions have different answers there; R1.4 shows
that early stopping, the other candidate, explains the fluorites but not BaTiO₃ and KNbO₃. The
screen's own curvature at the symmetric point equals MΩ² and is positive by construction (ESI
Section S1.3), so only its call is scored. Sections 2.1 and 4 note that the labels mark
thermodynamic transitions, which a dynamic-stability call can differ from near T_c.

*Agreement on one surface.* Sections 2.5 and 3.3 now state that SSCHA inherits the MLIP surface; the
"gold standard" wording is gone. The bcc call agreement is 31/45 [0.54, 0.80], 12 of them trivial
(no instability to stabilise; marked in Fig. 4); converged SSCHA in 3×3×3 agrees in 33/40.

**Changes.** Sections 2.1, 2.4, 2.5, new 2.6; Section 3.2 (PBE paragraph); Section 3.3 ("Why the
default criterion misses these wells"); Section 4 (Limitations); new Fig. 3; captions of Figs. 2, 4
and 5; ESI Sections S1.3, S2.2, S4, S5.1, S5.3, Tables S16, S19, S23.

## R1.2 MLIP force-engine errors against method failures

> Differentiate MLIP Force Engine Errors from Anharmonic Method Failures: The reported SSCHA
> false-stabilization may stem from MLIP errors when sampling thermally accessible, highly distorted
> configurations out-of-distribution, rather than an intrinsic failure of the SSCHA formalism
> itself. The authors should benchmark MLIP forces/energies against first-principles reference
> calculations along the sampling paths.

**Response.** I ran this benchmark: PBE energies and forces on SSCHA-sampled configurations of four
units, against rattled near-equilibrium configurations of the same cells (68 calculations; ESI
Section S5.2, Table S20). At 100 K MACE-MP-0's relative force error is 0.10 (BaTiO₃) and 0.18
(ZrO₂) against 0.10 and 0.23 near equilibrium, so it does not grow on these samples. At 600 K in
SrTiO₃ it rises from 0.11 to 0.19 and energy errors reach 44 meV per atom for all five models. One
limitation: the samples come from the production recipe, which did not converge (R1.4), and the
check was not repeated on the converged ensembles.

The reviewed attribution of the false-stables to the SSCHA formalism is withdrawn; Section 3.3 now
says its results concern the question the default criterion asks (R1.1), which needs no MLIP error
but does not exclude one. The fourth-order term is not the remedy either: on BaTiO₃/MACE-MP-0/100 K
it gives +2.878 THz, identical to bubble level. The high-temperature false-unstables, which the
reviewed version did not discuss and which match the referee's hypothesis, turned out to be
unconverged relaxations: converged, SrTiO₃/MACE-MP-0 is stable at 600 K (+2.44 THz) and no unit
stable at 100 K turns unstable with temperature (remaining false-unstables: CsSnI₃ only, 6 units).

**Changes.** Section 2.6; Section 3.3 ("Why the default criterion misses these wells",
"High-temperature false-unstables were the unconverged recipe", "Numerical failures", full list in
ESI Section S2.3); Fig. 7; Section 4; ESI Sections S2.1–S2.4, S5.2, Tables S2, S16, S17, S20, S22.

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

**Response.** The Introduction has two new paragraphs and the Discussion returns to this literature.
On-the-fly active learning has reproduced the transitions of hybrid perovskites, zirconia and
zirconium (refs 6–8), the families studied here; those are system-specific potentials, which places
the gap measured here with foundation models used as shipped, not with MLIPs in general. Also cited:
refs 9–17, including the sparse Gaussian-process work the referee points to (PRB 2021, ref 11; JPCL
2021, ref 12; Chem. Phys. Rev. 2024, ref 15; Acc. Chem. Res. 2026, ref 14; Adv. Energy Mater. 2026,
ref 16). For screening, GNoME, M3GNet and Matbench Discovery (refs 18–20) are added, and the screen
is placed as one filter for candidates that pass the convex hull but show imaginary modes; catalyst
discovery is outside this benchmark. One suggested work is not cited: the 2025 *Chem. Phys. Rev.*
paper on oxygen-containing organic compounds, whose molecular systems do not overlap with the
crystals studied; its method is cited through ref 14.

**Changes.** Section 1 (third and fourth paragraphs); Section 4 ("On model-reported uncertainty"); refs 6–20.

## R1.4 Convergence diagnostics for SSCHA

> Rigorous Convergence Diagnostics for SSCHA: The claim that SSCHA is a "methodological trap"
> requires stronger justification. The authors must report standard convergence metrics: sample
> sizes, gradient history, stopping criteria, and Hessian uncertainties. The four-seed stochastic
> test should be expanded beyond bcc-Zr to include representative displacive systems (BaTiO3 or
> fluorites).

**Response.** I withdrew "methodological trap" rather than justify it (see (c)). The settings are now
stated in Section 2.5 and Table S11. The convergence metrics showed a problem: python-sscha 1.6.1
applies the step cap `max_ka` cumulatively across populations, not per population, and passes a
constant placeholder as the gradient error. A four-seed study on BaTiO₃ and ZrO₂ (MACE-MP-0,
100 K), bcc Zr (MatterSim, 50 K) and SrTiO₃ (MACE-MP-0, 600 K), with full histories, found that no
seed of any unit converged; on three of the four units the Hessian sat within 0.24 THz of the starting matrix, so the earlier
seed agreement was agreement on where the recipe stopped (ESI Table S21).

I therefore re-ran every unit Section 3.3 rests on (178) with a recipe that converges: ≤400 steps
per population, the library's stochastic gradient error, populations of 1,000 up to 30, a
2,000-configuration Hessian ensemble with bootstrap. 159 converged, each reporting its stopping
reason, gradient against error, bootstrap SD and a fresh-ensemble check (152 pass; Table S22). Then,
under a protocol registered before any replicate ran, 65 units (53 disagreeing + 12 random) were
repeated from a second start and with a second seed; a call that differs is counted unresolved.
The call is the same in 43 of the 51 units with three values (0.84 [0.72, 0.92]); 8 are unresolved
(Table S24).

Result: converged SSCHA calls 27 of 77 label-unstable non-bcc units stable, all oxide perovskites
(23/67 without ORB-v2); BaTiO₃ stays stable (+2.03, +1.85 THz from two starts); ZrO₂ becomes
unstable (−22.4, −26.0 THz), as labelled; and the ferroelectric recall is 4/22 for SSCHA against
14/22 for the screen on the same units. One of the ten BaTiO₃/KNbO₃ 100 K units is marginal
(CHGNet: +1.65 THz, but 6 of 10 bootstrap resamples unstable), and Section 3.3 says so.

**Changes.** Section 2.5; Section 3.3; new Fig. 7; Section 3.5 ("Stochastic reproducibility");
Section 4 (Limitations); ESI Sections S2.1, S2.2, S2.4, Tables S2, S11, S21, S22, S24.

## R1.5 Formalisation and sensitivity of the soft-mode screen

> Formalization and Sensitivity of the Soft-Mode Screen: The mathematical formulation of the
> single-mode quantum SCHA screen needs complete derivation in the ESI (mode mass-weighting,
> Gaussian width self-consistency equation, and mode-mode coupling limitations). Sensitivity to
> polynomial fit windows, sampling ranges, and cell commensurability must be systematically
> reported.

**Response.** The derivation is complete in ESI Section S1.3: the effective mass M = Σᵢ mᵢ|**u**ᵢ|²
and the normalisation of Q; the Gaussian-width stationarity condition in full, with the solver and
its fallback (4.6 % of centroid evaluations, none of the 1512 that decide a call); and the neglected
mode–mode couplings written out (Table 1), with no quantitative bound, which I state. An exact
one-dimensional solution of the same potentials is compared in ESI Section S1.4 (Table S12).

Sensitivity (ESI Table S14): fit windows of 3–8× the well depth with 30–120 meV floors change no
call; truncating the sampling range at 0.25–0.40 Å changes at most one call at T ≤ 300 K. Cell
commensurability does matter: summing M and V over n minimal cells is the minimal-cell problem at
n²M and T/n, so the ferroelectric recall at T ≤ 300 K runs from 5/30 (formula unit) through 16/30
(production minimal cell) to 28/30 (force-constant supercell), and accuracy on the labels cannot
choose among them. The paper now reports the screen's recall and T* as properties of a stated
convention, and the H2 count with its range across conventions (7 to 24 against 4 at 300 K).

**Changes.** Section 2.4 (constants now in ESI Section S1.3); Table 1 (A3, A7); Section 3.2;
Section 4 (Limitations); Section 5; ESI Sections S1.1, S1.3, S1.4, S4, Tables S12, S14.

## R1.6 Statistical language and claims

> Reconcile Statistical Language and Claims: The Abstract and Discussion currently frame harmonic
> accuracy as "non-predictive" of finite-temperature performance, whereas the Results section notes
> a weak positive correlation. Align the headline claims with the statistical analysis to avoid
> overstatement.

**Response.** Agreed. "Non-predictive" is gone. The positive φ = 0.11 was the unlabelled 100 K value;
it is now +0.149 at 100 K and negative at 300–900 K, with system-level p above 0.08 at every
temperature, so the paper claims no association in either direction. It claims instead that
harmonically correct units the screen mis-calls at finite temperature outnumber the reverse, 17 to 4
at 300 K, not significantly once clustered by system (p = 0.152), and that 13 of the 17 sit in
BaTiO₃, KNbO₃ and CsSnBr₃, where the error is shared across models and, by R1.1, is shared by PBE
at its own lattice or follows the functional. The Abstract, Sections 1, 3.2, 4 and 5 carry this
wording and none reads a failure to reject as a finding.

**Changes.** Abstract; Section 1 (H2); Section 2.4; Section 3.2 (Table 4 and the text after it);
Sections 4 and 5; ESI Section S1.3, Table S15.

---

# Referee 2

> The manuscript provides a test of universal MLIPs for different dynamically stabilized systems.
> The work focuses exclusively on a stability argument and does not mention how fine-tuning would
> affect the results, which most experts would assume is needed for these systems. While for crystal
> structure prediction you would not necessarily have the prior knowledge to make that assumption,
> there is no mention of what the ensemble uncertainty of the force predictions are. This metric can
> be used as a reliable measure of where the MLIP predictions should not be trusted. Because of this
> I don't think the manuscript is suitable for publication, despite it being technically sound.

Thank you for this report. Both omissions were real. The revision adds a pre-registered test of
each, and it now also compares each model's energies and forces with PBE along the screen's paths
and on SSCHA samples (R1.1, R1.2).

## R2.1 Fine-tuning

> The work focuses exclusively on a stability argument and does not mention how fine-tuning would
> affect the results, which most experts would assume is needed for these systems.

**Response.** The Abstract and Sections 2.2, 4 and 5 now state that the models are tested as shipped,
the regime of generative-CSP screening, where the anharmonic candidates are not known in advance.
For a known anharmonic material I agree fine-tuning is the right step, and Section 4 cites what it
has been shown to do (refs 5 and 45–48): it removes much of the systematic softening of universal
MLIPs and has been used for SSCHA at screening scale.

I also tested it, in a trial registered before any data were computed
(`tasks/preregistration-finetune-2026-10-03.md`; ESI Section S6, Tables S25 and S26). MACE-MP-0 and
CHGNet were each fine-tuned three times toward PBE on 120 thermal configurations of BaTiO₃, KNbO₃
and CsSnBr₃; a call changes only if all three replicates agree.

- Ferroelectric well depths within 20 % of PBE: supported for MACE-MP-0 (median ratio 0.99, base
  0.45); unresolved for CHGNet.
- BaTiO₃ and KNbO₃ 300 K mis-calls corrected: **refuted** for both models.
- KNbO₃ 600 K and CsSnBr₃ 300 K mis-calls persist: supported.
- Converged SSCHA on fine-tuned MACE-MP-0 still calls BaTiO₃ stable at 100 K: supported (+1.73 THz).
- Stable controls keep their calls: supported for MACE-MP-0; four of six unresolved for CHGNet.

Fine-tuning also cost MACE-MP-0 one correct call (CsSnBr₃, 900 K). One deviation was recorded
before re-running: the first MACE-MP-0 fine-tunes trained 6 epochs instead of 30; both runs give
the same verdicts. A labelled post-hoc check locates the refutation: the fine-tunes relax to within
0.3 % of the PBE lattice, where they call 300 K stable, as PBE itself does (R1.1); at the base
model's lattice they call it unstable (Fig. 3). A fine-tune faithful to PBE inherits PBE's own
300 K errors, so the earlier estimate that it would remove about half of the shared count is
withdrawn. The trial covers two models and three systems and does not show that fine-tuning cannot
fix these calls.

**Changes.** Abstract; Section 2.2; Section 4 ("Scope: foundation models as shipped", "On
model-reported uncertainty"); Fig. 3; Section 5; refs 5, 45–48; ESI Section S6, Tables S25 and
S26; trial data in `results/revision/finetune/` and `results/revision/finetune_mace30/`.

## R2.2 Ensemble uncertainty of the force predictions

> While for crystal structure prediction you would not necessarily have the prior knowledge to make
> that assumption, there is no mention of what the ensemble uncertainty of the force predictions
> are. This metric can be used as a reliable measure of where the MLIP predictions should not be
> trusted.

**Response.** I agree it is the natural trust metric, and Section 4 now sets out the literature
(refs 9, 13, 14, 17). The released checkpoints are single models, so I tested the idea in three
forms on the same 60 consensus units, asking whether the signal flags units where the majority call
is wrong.

- *Vote split across architectures.* AUC 0.762 [0.590, 0.934] with all five models, 0.628
  [0.438, 0.839] without ORB-v2: suggestive, not robust, and the paper now says so.
- *Spread of the screen's frequency.* No signal (0.361 [0.046, 0.625]); this is a PES-level proxy,
  not force-level uncertainty.
- *Force-level spread (new, pre-registered).* Protocol and decision rule were fixed, and hashed, in
  `scripts/force_spread.py` before any configuration existed. Sixteen thermal configurations per
  unit, identical for every model; the score is the across-model force spread. The primary AUC is
  0.681 [0.416, 0.908] (0.684 without ORB-v2), which spans 0.5, so by the registered rule the
  spread is not shown to flag untrustworthy calls. Five of nineteen secondary scores, including
  within-family committees, have lower bounds above 0.5; none is adjusted for multiplicity, and
  restricted to units without unphysically close atoms the primary falls to 0.314, so I do not
  promote any of them (ESI Table S18).

With fifteen systems these data do not exclude a useful force-level signal; they do not show one.

**Changes.** Section 3.4; Section 4 ("On model-reported uncertainty"); Section 5; Fig. 8 (both
model sets); ESI Tables S8, S9 and S18.

---

# Referee 3

Thank you for a close reading that found a contradiction between two sections, an ESI that named
tables it did not contain, and statistics that treated clustered units as independent. All three
were right. Several numbers in the report's summary have since moved (table in (a)).

## R3.1 Section 4 against Section 3.2

> Reconcile §4 with §3.2; the Discussion asserts the hypothesis that the Results reject. Section
> 3.2 tests the strong form of H2, states explicitly that it is rejected, and reports a weak
> positive association (φ = 0.11, per-model ρ = +0.15 at T ≤ 300 K rising to +0.65 at T = 100 K),
> settling on "necessary but not sufficient"; Section 4 then states that the results "confirm H2 in
> a strong form" and that "harmonic accuracy is non-predictive of finite-temperature accuracy,"
> which is the opposite of a positive correlation. The §3.2 formulation is the one the data support
> and should be carried consistently into the Discussion, the abstract, and the framing of the
> title.

**Response.** Correct. "Confirm H2 in a strong form" was removed in August; "non-predictive" is
removed now. The Section 3.2 formulation does not survive the full temperature ladder either: φ is
positive only at 100 K and negative from 300 K, and the rank correlations are not interpretable
(R3.3), so "necessary but not sufficient" is not used. The Discussion, Abstract, Conclusions and
title now carry the narrower statement of R1.6, and the pre-registered form of H2 is left open.

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

**Response.** I attempted the first option and, where it did not finish, kept the second. The
4×4×4 run was pre-registered before any unit ran (`tasks/preregistration-4x4x4-2026-10-09.md`,
commit f47cc2a): the four SrTiO₃ units that converged SSCHA calls stable at 100 K, and the fluorite
unit closest to a sign change (HfO₂/MACE-MP-0, 600 K), with the converged recipe of R1.4 and a 9 h
cap per unit. For HfO₂ the relaxation converged and the lowest Hessian frequency is −3.22 THz in
4×4×4 against −2.20 THz in 2×2×2, so its call is unchanged (one start; the cap came before the
bootstrap). No SrTiO₃ unit reached a 4×4×4 Hessian within the cap, so SrTiO₃ still has no
cell-size test. Section 3.5 and ESI Section S2.4 therefore make no supercell-convergence claim for
the R-point and X-point systems and state their calls for 2×2×2 only.

Two points have changed since the first version of this reply. The fluorite false-stable does not
survive convergence (R1.4). Converged SSCHA now calls SrTiO₃ stable at 100 K for all four models
(+1.06 to +1.20 THz); the R point is in the 2×2×2 cell, but Section 3.5 says this may be a
finite-size effect, and the headline count is also given without those units: 23 of 73. The
zone-centre argument stands for BaTiO₃ only, as the referee says. For bcc Zr with MatterSim the
call changes sign between 2×2×2 (+0.92 THz) and 3×3×3 (−0.92 THz) at 300 K; the cells are not
nested, so Section 3.5 reports this as a change of **q**-set, not a convergence test.

**Changes.** Section 3.5 ("Finite size", "Zone-boundary systems", with the 4×4×4 attempt);
Section 4 (Limitations); ESI Sections S2.4 ("The 4×4×4 attempt"), S4, Tables S17, S21 and S22.

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

**Response.** All adopted (`mlip_dynstab/stats.py`, `scripts/stats_hardening.py`).

- *Counts and intervals.* Rates are given as k/n with Wilson 95% intervals: 19/19 [0.832, 1.000],
  17/19 [0.686, 0.971], 15/19 [0.567, 0.915].
- *Composition and clustering.* Both sets rest on the same 15 systems (n = 60: four temperatures;
  n = 75: five models; Tables S7 and S8). Every test used as evidence now resamples whole systems:
  the vote-split AUC 0.762 has a system-clustered p of 0.0035 (unit-level 0.0003 overstated it).
- *Spearman.* Removed; with five models the coefficients change sign across the ladder and none is
  significant.
- *McNemar.* The reviewed 0.34 was the 100 K value. Every temperature is now reported, with an
  exact system-level sign test: p = 0.844, 0.152, 0.066 and 0.414 at 100–900 K; the unit-level p
  appears only as a labelled companion, and no failure to reject is read as a finding.
- *Ranking.* The paper makes no finite-temperature ranking claim; the accuracies (24/30 to 27/30)
  have overlapping intervals.

The same correction applies to the screen-versus-SSCHA comparison: paired and clustered, 12 against
2 in the screen's favour, system-level p = 0.5, reported as not significant (Table S10).

**Changes.** Sections 3.1–3.4; Section 5; Figs. 6, 7 and 8; ESI Section S3 (head), Tables S4–S10 and S15.

## R3.4 Tables S1–S4

> Supply Tables S1–S4. The ESI refers to per-model harmonic confusion matrices, per-model finite-T
> rates with and without bcc, per-system predicted T* against experiment, and the ensemble-guardrail
> table, but only the names of the functions that generate them appear; these carry the per-system
> numbers underlying every headline rate in the paper, and none of the results are checkable without
> them.

**Response.** Correct: the tables were never rendered. They are now generated from the deposited
data by `scripts/build_esi_tables.py`, whose `--check` fails if the ESI is out of date. The
requested tables are now Table S4 (harmonic confusion matrices), S5 (finite-T rates with and
without bcc), S6 (T* against experiment) and S8 (ensemble guardrail); Tables S1–S3 now sit with the
methods they document, and a head note separates section from table numbers. Tables S7 and S9–S26
are new and carry the revision's results.

**Changes.** ESI head note; Section S3 (Tables S4–S26); Tables S1–S3 in Sections S1.2, S2.2 and
S2.3; Sections S5 and S6.

## R3.5 Every rate with and without ORB-v2

> Report all rates with and without ORB-v2. The float32, non-conservative-force architecture is a
> confound for finite-difference force constants and for SSCHA stability rather than a model
> property of interest, and ORB-v2 accounts for its sole false-unstable at MgO (−2.8 THz), the
> softmode outliers near −35 THz on Ti and Hf, the six SSCHA blow-ups concentrated in its runs
> (down to −2×10⁶ THz), and the outlying finite-T false-stable rate of 0.562; the with-and-without
> split is already performed for one bcc sign-agreement number and should be applied throughout,
> since the design cannot separate precision from architecture and the manuscript says so.

**Response.** Adopted, and it changed one conclusion. Two premises have moved: CHGNet, SevenNet-0
and MatterSim also run in float32, so precision does not single ORB-v2 out (its direct, non-gradient
forces do; Section 2.2); and its figures changed after re-measurement (table in (a)).

With the split applied to every pooled quantity (ESI Table S9; main text beside each pooled
result), the ensemble guardrail falls to 0.628 with an interval spanning 0.5 and is now called
suggestive. The others hold: bcc call agreement 31/45 and 25/36; H2 at 300 K 14 against 2
(clustered p = 0.156); converged SSCHA false-stables 23/67; ferroelectric recall 4/19 against
11/19; replicate agreement 32/39. ORB-v2 does stand out in convergence: all 11 converged-grid runs
that hit the population cap and all four of its blow-ups are ORB-v2, and it is the only model whose
harmonic calls flip under replicate noise, and the only one whose calls on test systems change
with the displacement amplitude.

**Changes.** Section 2.2; Sections 3.1–3.4; Section 4 (Limitations); ESI Table S9, and the split
in Tables S10 and S13–S18.

## R3.6 Figure order

> Confirm the figure order. Fig. 5 is discussed in §3.3 but printed after Fig. 6, which may be an
> artifact of the proof rather than of the submitted manuscript; figures should appear in order of
> first citation, and the production editor may wish to check this.

**Response.** It was in the submitted source, not the proof. All eight figures now appear in order of
first citation and are supplied as numbered 600 dpi TIFF files. Two figures are new, so reviewed
Figs. 3, 4, 5 and 6 are now Figs. 4, 5, 6 and 8; new Fig. 3 shows the PBE lattice check and the
fine-tunes (R1.1, R2.1), and new Fig. 7 the converged SSCHA call on every grid unit (R1.4). Fig. 2
is also new: the reviewed Fig. 2 plotted the screen's curvature, which is positive by construction,
as though it were a harmonic frequency; it now shows the harmonic minimum frequency of every system
under every model. Every caption except Fig. 1's changed in substance (no "gold standard", counts
with intervals, converged SSCHA values).

**Changes.** Placement of Figs. 6 and 8; captions of Figs. 2 and 4–8; Figs. 2, 3 and 7 new; Figs. 4,
5 and 6 regenerated; main-text Tables 2–4 numbered and captioned.

---

# Editor's comments

> If any of the reviewer reports below contain recommended references to be included in your
> manuscript, please check them carefully and only include those you think are relevant and help
> improve your manuscript.

I checked each suggested work against its published record and included those relevant to the
argument (R1.3); the fine-tuning references (R2.1) were checked the same way. The manuscript now has
an Author contributions section in CRediT roles above Conflicts of interest and Acknowledgements,
and my ORCID (0009-0003-0041-1459) on the title page, to be linked in the portal at upload, where I
also make my choice on transparent peer review.
<!-- PENDING-P4: before upload, link the ORCID to the submitting account in the portal and finish the affiliation, funding and AI-use checks (tasks/todo.md P4); if either is not done, rewrite the two sentences above to say what was done. -->

# Other changes

- References are numbered in order of first citation; ref 27 (α-AgI) now cites the correct work,
  and the MACE-MP-0 reference cites its published version.
- The pre-registration of H1–H3 is dated in Section 1 (commit 9c2599f, 20 June 2026).
- A statement on the use of AI tools is added at the end of the Acknowledgements.
<!-- DRAFT-FOR-FRANK: the AI-use statement in the manuscript is Option A of paper/response/ai_use_statement_DRAFT.md; confirm or replace its wording, and keep or drop this bullet accordingly. -->
- To keep the paper readable the main text is about 2,900 words shorter: methods detail that
  predates the revision moved verbatim to the ESI with pointers, and repeated results are stated
  once. Nothing added in answer to the referees left the main text.

# Files submitted with this response

<!-- PENDING-B1: build the clean .docx files and the marked-changes manuscript and ESI (scripts/build_docx.py, scripts/build_marked_changes.py against 76d3a84) after the last text edit; drop any item below that is not uploaded. -->
1. This point-by-point response.
2. The revised manuscript with the changes marked.
3. The revised manuscript as a clean .docx with the figures embedded.
4. The revised ESI, clean and with the changes marked.
5. Figs. 1–8 as separate numbered TIFF files at 600 dpi.
6. The table of contents entry.

Yours sincerely,

Frank Cai
Purdue University, West Lafayette, Indiana, USA
