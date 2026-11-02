# Response to the referee (second revision)

**Manuscript** RA-ART-07-2026-006452.R1, *RSC Advances*
**Title** Neither harmonic benchmarks nor a default SSCHA cross-check certifies a foundation
machine-learning interatomic potential for finite-temperature dynamic stability
**Author** Frank Cai, Purdue University

Dr Lydia Rhyman
Associate Editor, *RSC Advances*

Dear Dr Rhyman,

Thank you for sending the revision back to the referee, and please pass my thanks to them for a
second close reading. Each answer below gives the physical argument the comment asks for and the number in the deposited
data that tests it. In brief, the revision:

- **C1** tests the first revision's explanation directly: a pre-registered multi-mode SSCHA test of
  BaTiO₃ with MACE-MP-0 at 100 K puts the free energy at the screen's zone-boundary minimum 46 meV per
  40 atoms above the cubic point, and the fourth-order term at the converged matrix does not change
  the sign, so on that unit the disagreement with the screen comes from mode coupling, not from where
  the curvature is read; Section 3.3 now sets out why a symmetric-point Hessian cannot by itself
  exclude a lower minimum, and what the test implies for the screen's own recall;
- **C2** explains the lattice sensitivity quantitatively (the soft-mode κ₂ is a near-cancellation,
  the well depth goes as κ₂²/4κ₄, and the measured d ln(depth)/d ln a is 130–161 for BaTiO₃), shows
  that the MLIP lattices are the Materials Project PBE lattice shared by all five models, and states
  the constant-volume protocol and what it leaves out;
- **C3** turns the step-cap finding into a lesson in validating a stochastic minimiser: Section 3.5
  now opens with the minimiser's mechanics, the role of the population size, a
  population-by-population table (new Table 5) and the checks that catch a non-converged
  relaxation, with an ESI figure for BaTiO₃ and bcc Zr;
- **C4** removes the "guardrail" framing, shows why the force-level spread fails (it tracks
  extrapolation into short contacts, while about half or more of the error on thermally sampled
  configurations is common to all five models), and adds a recommendation for committee design in
  anharmonic regions.

Every new number is checked against the deposited data by `scripts/verify_claims.py`, and the code
and data are deposited as release v2.2.0 (commit 1344faf), archived at Zenodo as
https://doi.org/10.5281/zenodo.23855585.

# Referee 1

> The manuscript addresses a timely and significant topic in computational materials science:
> evaluating whether foundation MLIPs reliably capture finite-temperature dynamic stability,
> particularly for harmonically unstable yet thermally stabilized systems (such as cubic
> perovskites, bcc refractory metals, and fluorites). [...] Given that the first revision is
> already made substantially, the manuscript is publishable after second revision considering the
> following comments.

**Response.** I thank the referee for this assessment.

## C1 Why converged SSCHA and the screen disagree on deep double wells

> 1. The manuscript points out that converged SSCHA (using the free-energy Hessian at the symmetric
> reference) fails to flag many ferroelectric perovskites as unstable at T≤300 K, calling them
> "stable" because the symmetric reference remains a local minimum of the free energy in a deep
> double-well potential. However, the explanation is glossed over with generic terminology rather
> than a rigorous articulation of free-energy landscapes, order-parameter paths, and the breakdown
> of the adiabatic/harmonic approximation under high-order anharmonic tensor contributions
> \Phi^(4). The author needs to expand Section 3.3 to explicitly explain why the bubble-level SSCHA
> free-energy Hessian misses displacive phase transitions in deep wells. They should incorporate a
> dedicated paragraph discussing how the auxiliary dynamical matrix enforces positive-definiteness
> for the trial Gaussian density, and how the absence of a global centroid displacement search
> (unlike the single-mode screen) traps the Hessian evaluation at the high-symmetry point. To prove
> deep understanding, the author should explicitly discuss whether higher-order terms would alter
> the sign or if the limitation is fundamentally geometric (local vs. global minimum search).

**Response.** The referee is right that Section 3.3 named the limitation without deriving it.
Deriving it, and then testing the explanation the first revision gave, changed the answer: on the
one unit I tested, the stable call does not come from reading the curvature at the wrong point.
Section 3.3 now gives the argument, the test, and the mechanism that replaces the old explanation.

*Positive definiteness, and what a symmetric-point Hessian can see.* A Gaussian trial density is
normalisable only if the auxiliary matrix Φ is positive definite. The workflow enforces this twice:
the starting matrix is made positive definite (`ForcePositiveDefinite`, imaginary frequencies
replaced by their moduli), and the minimiser steps in the square root of Φ (`python-sscha`'s root
representation), so Φ stays positive definite at every step; in Table 5 its lowest frequency stays
between +2.4 and +2.9 THz throughout. Its eigenvalues therefore cannot signal an instability. The
test is the free-energy Hessian, Φ + Φ⁽³⁾Λ[1 − Φ⁽⁴⁾Λ]⁻¹Φ⁽³⁾ (ref 38), which is also the static
(ω → 0) limit of the SSCHA phonon self-energy: the test is static, and the frequency dependence of
the self-energy shapes the phonon spectra, not the curvature of the free energy. Every
correction to Φ carries Φ⁽³⁾ on both sides, and at bubble level, where Λ is negative definite, it
can only lower the curvature. At the cubic reference the soft mode's coupling to itself, Φ⁽³⁾_sss,
is forbidden by symmetry: by inversion for the polar Γ mode, and by translation for the X, M and R
modes of the cubic perovskite, for which 3q is not a reciprocal-lattice vector. Along that mode
alone, then, F″(0) = MΩ² > 0 whether Φ⁽⁴⁾ is kept or dropped. The Hessian can turn negative only
through the soft mode's couplings to pairs of other modes (among them the other two arms of the M
star, which sum with it to a reciprocal-lattice vector), and its sign marks where the symmetric
phase stops being a local minimum of the free energy. For a second-order transition that is the
transition; for a first-order one, a lower minimum at a finite displacement can coexist with a
positive Hessian. I have now evaluated Φ⁽⁴⁾ at the converged matrix (the first revision's test
used an unconverged one): the lowest frequency moves from +2.137 to +2.152 THz, against a bootstrap
standard deviation of 0.14 THz, and the sign does not change.

*The missing centroid search.* The referee is also right that the default relaxation cannot leave
the high-symmetry point: the centroid gradient vanishes there by symmetry (`python-sscha`
symmetrises it, and it is recorded as exactly zero on all 54 perovskite units of the converged grid
with MACE-MP-0, MatterSim and SevenNet-0; CHGNet on BaTiO₃ and ORB-v2 record small non-zero
values). The first revision said a displaced start had not been tried. I have now compared free
energies at displaced centroids for one unit, BaTiO₃ with MACE-MP-0 at 100 K in the 40-atom cell,
under a protocol, gates and decision rule committed to the public repository before any run
(`tasks/preregistration-centroid-2026-10-21.md`). On this unit the screen condenses an M-point and
an X-point mode, not the polar Γ mode; the scan follows the M mode, and the X mode was not tested.

With the centroid held at the screen's M-point minimum (0.075 Å), the multi-mode SSCHA free energy
lies 46.0 ± 2.2 meV per 40 atoms above the cubic point, although the static energy there is 90 meV
lower. At 0.10 Å it lies 149 meV above. Under the registered rule there is no displaced minimum
along M. Held at the static minimum along the polar Γ mode in [111], the free energy lies
55.3 ± 2.2 meV above the cubic point, against a static energy 226 meV lower. A free centroid
started at 0.075 Å along M did not converge in 30 populations; it left the M pattern for a small
polar displacement along [010] (on average Ti +0.031 Å, O −0.033 Å) at a free energy
3.6 ± 2.4 meV above the cubic point, so it found no lower minimum but does not exclude a shallow
polar one. A held-centroid free energy is an upper bound on the lowest free energy with the other
centroid coordinates free. Under the registered rule, the first revision's explanation is not
supported for this unit.

*What replaces it.* The screen moves one mode and freezes all the others. In the multi-mode theory
the displacement stiffens the other modes: at 0.075 Å the auxiliary frequencies rise by 3.6 % on
geometric average, and in the SCHA's own split (which depends on Φ, so it describes the cost within
the theory rather than decomposing it uniquely) their harmonic free energy rises by 62 meV per 40
atoms, 52 meV of it zero-point energy (35 meV would remain in the classical limit), while the
anharmonic remainder falls by 16 meV; statically the displacement gains 90 meV. The two methods
disagree through mode coupling, approximation A1 of Table 1, and the screen's minimal-cell
normalisation (A7) favours the zone-boundary modes. This has a consequence I state in Section 3.3
and the Discussion: the screen's unstable calls on the three ferroelectrics rest mostly on
zone-boundary modes (the polar Γ mode condenses in 5 of its 16 unstable calls at T ≤ 300 K), so its
higher recall is not evidence that it better approximates the MLIP's own free energy. On the unit
tested, the cruder treatment is the one that matches the label.

*The closing question.* On this unit, within the SCHA and along the directions tested, the answer
is neither: the higher-order term does not change the sign, and the free energy at the displaced
centroids tested lies above the cubic one. What stays open is whether the multi-mode Gaussian
answer is right for the MLIP's surface. An exploratory classical molecular-dynamics check on the
same potential and cell, whose reading rule was written after the SSCHA result and before the runs
(`tasks/md-order-check-2026-10-22.md`; not pre-registered), is inconclusive under that rule at
100 K. Pooled over the run, each Cartesian component of the Ti off-centring is bimodal near
±0.10 Å, which a Gaussian centred on the cubic site cannot reproduce, though the statistic does not
separate local disorder from a coherent polarisation that reorients. Classical dynamics omits
zero-point motion, so separating the potential from the Gaussian, zero-point treatment would need
path-integral simulations. Section 3.3 says so, and says that only one unit, at one temperature
and in one cell size, was tested; the registration made no directional prediction and committed
to reporting either outcome.

**Changes.** Abstract; Section 1 (contributions); Table 1 (rows A1 and A4); Section 2.4 (two
pointer sentences); Section 2.5 (one sentence); Section 3.3 (retitled; the local-versus-global
explanation of the false-stables replaced by the tested mode-coupling result, the general caveat
kept; new paragraphs on what a symmetric-point Hessian can exclude, the direct test, the mechanism
and what remains open; the screen's deciding modes); Section 4 (Discussion paragraph on SSCHA; one
sentence each in the fine-tuning paragraph and Limitations); Section 5; ESI Sections S1.3, S2.2 and
S2.3, the Table S16 caption, two outcomes in Section S4, new Section S7 and Tables S28–S29;
`scripts/sscha_centroid_scan.py`, `scripts/centroid_decomposition.py`, `scripts/md_order_check.py`,
`results/revision/r2_centroid/`, `results/revision/r2_md/`;
`tasks/preregistration-centroid-2026-10-21.md`, `tasks/md-order-check-2026-10-22.md`.

## C2 Lattice, volume and stress

> 2. Section 3.2 highlights that PBE at its own lattice (0.3–0.8% smaller than the MLIP relaxed
> lattices) gives different stability outcomes for BaTiO3 and KNbO3, and that fine-tuning toward
> PBE without stress tensor inclusion led to lattice contraction and altered stability calls (Fig.
> 3). The author treats this somewhat as a statistical artifact rather than exploring the
> fundamental coupling between volume/stress and soft-mode behavior. Based on Quasi-harmonic
> approximation (QHA) principles and volume-dependent phonon softening, the author should add a
> discussion explaining why a fraction of a percent change in lattice parameter drastically alters
> well depths (due to the exponential sensitivity of overlap integrals and electrostatic forces in
> ionic/perovskite oxides). The author needs to explicitly analyze whether the MLIPs' volume biases
> are systematic (e.g., overestimation of lattice constants due to training set distributions) and
> how constant-volume vs. constant-pressure constraints affect finite-temperature screening.

**Response.** The referee is right that the revision gave the lattice dependence as an empirical
"knife edge" without its mechanism. Section 3.2 now gives it, in three parts, and Section 2.5
states the constant-volume protocol explicitly.

*Why a fraction of a percent matters.* The polar instability of these perovskites is a near
cancellation between the long-range dipole–dipole interaction, made large by the anomalous Born
charges of B-site d–O 2p hybridisation, and the short-range repulsion between ion cores (refs 44–47).
The soft mode's harmonic coefficient κ₂ (V = κ₂Q² + κ₄Q⁴ + …) is the small difference of the two.
The repulsion is the steep, overlap-controlled term the referee mentions, and it is the
stabilising one: a small expansion weakens it far more than the slowly varying dipolar term, so
|κ₂| grows. On PBE, d ln|κ₂|/d ln a = 46 (BaTiO₃) and 59 (KNbO₃). The well depth is κ₂²/4κ₄, so the
square doubles that, and in BaTiO₃ κ₄ also falls on expansion (d ln κ₄/d ln a = −37): together
d ln(depth)/d ln a = 130 from the fits and 161 from the sampled depths (115 and 146 for KNbO₃).
Because the PBE pair changes the eigenvector along with the lattice, I checked it with a model
evaluated at two lattices along its own eigenvectors of the same M-point mode (the fine-tuned
MACE-MP-0 replicates): 135–148. A 0.1 % change in lattice parameter therefore changes the BaTiO₃
well depth by about 15 %, and the 0.26–0.80 % differences in Fig. 3 by a factor of 1.5 to 3. I also
swept each of the five models' own lattices from −1.5 % to +1 % (new ESI Table S27): d ln(depth)/d ln a
= 109–147 for BaTiO₃ in four models (79 for ORB-v2, whose deciding mode lies at another q), and the
correct 100 K BaTiO₃ call turns stable 0.33–0.59 % below each model's lattice (1.40 % for ORB-v2). The
section cites the classic evidence for this volume sensitivity: KNbO₃ stable at the LDA lattice and
unstable at the experimental one (ref 45), the −4.8 GPa negative pressure that effective-Hamiltonian
studies of BaTiO₃ applied to undo the LDA lattice (ref 22), and BaTiO₃'s Curie temperature falling
by 46–59 K GPa⁻¹ under pressure (ref 48).

*Is the MLIP volume bias systematic?* Yes, and it is inherited from the training data, not from the
architectures. All five models relax BaTiO₃, KNbO₃, SrTiO₃ and cubic ZrO₂ to within 0.03–0.06 % of
one another, 5 to 14 times closer than they sit to our PBE lattice, and all fifteen model–system
pairs with a PBE lattice lie above it. For the three perovskites the models reproduce, to within
0.06 %, the Materials Project's own PBE lattices (ref 49), the reference all five training sets
share (Section 2.2 now names them); those lie 0.6 % (BaTiO₃) and 1.0 % (SrTiO₃) above experiment,
the usual PBE overestimate (refs 50–52). Architecture-specific departures appear only on the
halides and bcc metals. Our own PBE lattice (Quantum ESPRESSO, SSSP) lies 0.29 % (BaTiO₃) and
0.75 % (KNbO₃) below the Materials Project's. The difference is not numerical: raising both
plane-wave cutoffs by factors of 1.5 and 2 moves our lattice by at most 0.06 %, and an 11×11×11
k-mesh in place of 7×7×7 leaves it unchanged, so it comes from the pseudopotentials and codes (SSSP
in Quantum ESPRESSO against the projector-augmented-wave datasets of VASP behind the Materials
Project). At a sensitivity near 150 that difference between two calculations of one functional is
itself a factor of 1.5 to 3 in well depth, which is why the 300 K calls of Fig. 3 change with it,
and on KNbO₃ at 100 K three of the five models call the cubic phase unstable at their own lattice
and stable at ours.

*Constant volume against constant pressure, and the fine-tuning.* Every screen and SSCHA calculation
is made at constant volume, in each model's static 0 K cell at zero pressure. The quasi-harmonic
approximation cannot provide the thermal expansion for these phases, whose harmonic spectra are
unstable; the self-consistent route is SSCHA at constant pressure through its stress tensor
(ref 53), which I did not run, and the text says so. With SrTiO₃'s thermal expansion as a guide (at
least 0.1 % in lattice parameter per 100 K, ref 54), the omission is the same order as the
differences that flip the calls: the stable, and wrong, 300 K BaTiO₃ call of four models turns
unstable when their lattice is expanded by 0.15–0.34 %. For the polar modes of BaTiO₃ and KNbO₃,
whose wells deepen on expansion, a constant-volume screen therefore leans towards stability relative
to each model's own finite-temperature cell, more so as T rises, while PBE's larger lattice pulls the
other way relative to experiment. SrTiO₃'s deepest well behaves the other way, deepening on
compression (d ln(depth)/d ln a = −76 to −89 in four models, ESI Table S27), so for it constant
volume leans towards instability. The fine-tuning result follows from the same sensitivity:
trained on energies and forces in one cell, without stresses, the fine-tuned models relax
0.47–0.53 % below the base model on BaTiO₃, and the measured sensitivity predicts wells 1.8–2.0 times
shallower, against 1.9–2.1 measured. Fine-tuning aimed at soft-mode physics needs stresses, or
several volumes, in its data.

**Changes.** Section 2.2 (training sets named); Section 2.5 (constant-volume protocol); Section 3.2
(three new paragraphs); refs 44–54; `scripts/lattice_sensitivity.py`,
`results/revision/r2/external_lattices.json`; new ESI Table S27 (the lattice sweep),
`scripts/lattice_sweep.py`, `results/revision/r2_lattice/`; `scripts/qe_lattice_check.py`,
`results/revision/r2_qe_lattice/`.

## C3 The step cap as a lesson in validating a stochastic minimiser

> 3. A major technical finding of the work is that the "production" SSCHA recipe failed to converge
> because of a cumulative step-cap flaw in python-sscha, rendering previous literature practices
> questionable. However, the transition from unconverged production results to fully converged
> 3×3×3 results is presented as a passive observation rather than an analytical lesson in numerical
> workflow validation. The author needs to rewrite the SSCHA methodology and results discussion to
> emphasize why monitoring step caps, gradient thresholds, and stochastic population sizes is
> critical in machine-learning-driven phonon pipelines. The author should include a clear
> comparative table or flowchart showing how the gradient norm evolved across populations for at
> least one representative system (BaTiO3 or bcc-Zr), in view of the mechanics of stochastic
> minimization rather than just rerunning scripts generated by AI.

**Response.** Agreed. Section 3.5 now opens with the mechanics and a population-by-population table
(new Table 5, BaTiO₃ with MACE-MP-0 at 100 K, both recipes); ESI Fig. S1 adds bcc Zr with MatterSim
at 300 K (3×3×3). Every number in them is read from the per-step histories the original runs
recorded.

The mechanics: the SSCHA gradient is a Monte Carlo average over the current population; as the
auxiliary matrix moves, the same configurations are reused with importance weights until the
Kong–Liu ratio KL/N falls to a floor and a new population is drawn. A relaxation can therefore stop
for three reasons that are not convergence (a step cap, the Kong–Liu floor, a threshold set against
the wrong error). The population size N sets the gradient's statistical error (∝ 1/√N), and so the
smallest gradient a stopping test can resolve (256 configurations per population in production,
1,000 in the converged recipe); because the Kong–Liu floor is a fraction of N, a larger population
does not let the matrix move further before a new one is drawn, but each estimate along the way is
less noisy. Section 3.5 now says this.

What the table shows. In production the cumulative cap was spent by the first population (20 steps);
each later population took one step, discarded, and moved the matrix by less than 10⁻¹⁴, so the
gradient stayed at 0.34–0.36. The threshold it was tested against (10⁻⁴ × a constant placeholder
error, 2.2 × 10⁻⁴) lies 70–110 times below the real stochastic error of the gradient even at 1,000
configurations, so on this unit the test could only fail; the Hessian echoed the start (+2.878
against +2.881 THz). In the converged recipe populations 1–3 end at the Kong–Liu floor while the
gradient falls from 15 to 2.4 times its real error; population 4 reports a gradient at 0.02 of its
error, but at KL/N = 0.58, and the guard refuses it. The fresh population at the same matrix
measures 0.047, 2.7 times the error and a hundred times what the degenerate reweighting reported;
the run then converges at 0.19 of the error with KL/N = 0.97, and the Hessian (+1.98 ± 0.06 THz) lies
0.9 THz below the start. bcc Zr repeats the pattern (population 10 refused at 0.18 of the error and
KL/N = 0.50; population 11 starts at 2.9 times it).

Section 3.5 then lists the checks, each with what it caught in this study: a per-population step
cap; a threshold against the real gradient error; convergence refused below KL/N = 0.7 (the two false
stops above); a fresh-ensemble gradient check (7 of 159 converged units fail it); the distance of
the final matrix and Hessian from the start; a bootstrap on the Hessian; and a second start and
seed. Three of these would each have exposed the production failure alone. All are numbers a run
already computes; an MLIP-driven pipeline that runs hundreds of units unattended has to test them
per unit, because an unconverged relaxation returns a Hessian in the same form as a converged one.

**Changes.** Section 2.5 (pointer); Section 3.5 (new opening paragraph with the population size, Table 5 and the list of
checks); ESI Section S2.1 and new Fig. S1; `scripts/sscha_convergence_history.py`,
`results/revision/r2/sscha_history.json`.

## C4 Ensemble uncertainty

> 4. Section 3.4 shows that the ensemble vote-split AUC drops significantly and loses robustness
> when ORB-v2 (the only non-conservative, direct-force model) is removed, and that the force-level
> spread fails to predict errors once atomic overlap configurations are filtered out. The author
> should tone down any claims regarding ensemble uncertainty as a plug-and-play guardrail, and
> analyze why force-level spread fails in regions where atoms approach overlap (i.e.,
> high-temperature or strongly anharmonic rattling where MLIPs extrapolate wildly outside their
> training convex hull). The author should add a concluding recommendation on how future
> active-learning or committee strategies must be designed specifically for anharmonic regions
> rather than relying on naive inter-architecture variance.

**Response.** Agreed on the tone; and the analysis the referee asked for sharpened the conclusion.

*Toned down.* Section 3.4 is retitled "Ensemble disagreement: an extrapolation signal, not an error
estimate"; "guardrail" is gone from the abstract, the hypotheses, the contributions, Section 3.4,
the Fig. 8 caption, the Conclusions and the ESI text (the word survives only in a function name
and a figure file name in the deposited code, kept so that the code still runs); the vote split is described as suggestive, not
robust to removing ORB-v2, and as not certifying a call.

*Why the spread fails.* The data bear the comment out at short contacts and show a second failure
nearer equilibrium, where the wrong calls lie. Where atoms approach overlap the spread rises
steeply (median 0.059 eV Å⁻¹ at contact ratios of 0.85 and above, 2.1 eV Å⁻¹ below 0.50; it falls
with the contact ratio in all fifteen systems), because each model extrapolates on its own terms
and the five diverge; but on SrTiO₃ at 600 K, the one PBE-checked set that reaches such contacts,
the error of the mean force is still four times the spread, because 0.94 of the squared error is
shared by all five models. Nearer equilibrium the spread is small and misses the errors that decide
the calls: the four consensus errors left after the overlap filter are oxide false-stables at
below-median spread, called stable by at least three of the five models (KNbO₃ at 600 K by all
five). PBE forces on 64 configurations put a number on why: the part of each model's squared force
error shared by all five, which no spread can see, is 0.32–0.46 on rattled near-equilibrium cells,
0.45–0.59 on the SSCHA-sampled configurations of BaTiO₃, ZrO₂ and bcc Zr, and 0.94 on SrTiO₃ at
600 K. Within the thermal ensembles the spread ranks the force error poorly on BaTiO₃ and bcc Zr
(Spearman 0.23 and 0.19 over twelve configurations each) and better on ZrO₂ and SrTiO₃ (0.62 and
0.87). The models share a reference functional and most of their training configurations
(Section 2.2 now names the training sets), so they share their error.

*Recommendation* (Section 4, summarised in Section 5). A committee meant to guard finite-temperature
stability calls needs members that differ in their training data, not only in architecture;
training and active-learning configurations drawn from the target finite-temperature ensemble
(SSCHA or MD samples at T, and frozen-mode paths along the soft coordinates at more than one
lattice parameter, given C2); uncertainty carried through to the well depth or free-energy Hessian
rather than left on per-atom forces; and a standing check of the shared error against
first-principles single points, since no variance among members can expose a bias they share.
The section cites the committee and active-learning literature this builds on (refs 13, 17, 60–63).

**Changes.** Abstract; Section 1 (H3 and contributions); Section 2.2; Section 3.4 (retitled, new
paragraph); Fig. 8 caption; Section 4 (new recommendation); Section 5; ESI Tables S7–S9 wording;
refs 60–63; `scripts/spread_mechanism.py`, `results/revision/r2/spread_mechanism.json`.

# Editor's comments

No references were requested in this round. The 15 references added (44–54 and 60–63) were each checked
against Crossref for authors, volume and pages and against the source for the statement they
support.

# Other changes

- Section 3.3 and ESI Section S2.2 cited the six-unit converged value for BaTiO₃/MACE-MP-0 at 100 K
  (+2.03 THz) to Table S22, which holds the grid value (+1.98 ± 0.06 THz); the text now uses the
  grid value where Table S22 is cited and points the six-unit study to ESI Section S2.4.
- The table of contents graphic's second panel drew the first revision's local-versus-global account;
  it now plots the comment 1 result (free energy along the screen's M mode of BaTiO₃ at 100 K: static,
  single-mode screen and multi-mode SSCHA), and the entry's text says the same.
- References renumbered by first citation (`scripts/renumber_refs.py`): the eleven added for C2 are now
  refs 44–54 and the four added for C4 refs 60–63; the first revision's refs 44–48 are now 55–59.

# Files submitted with this response

1. This point-by-point response.
2. The revised manuscript as a clean .docx with the figures embedded.
3. The revised manuscript with the changes highlighted (insertions in yellow, no tracked changes),
   against the first revision as submitted.
4. The revised ESI, clean and with the changes highlighted in the same way.
5. Figs. 1–8 as separate numbered TIFF files at 600 dpi (unchanged from the first revision).
6. The table of contents entry (graphic and text revised; see Other changes).
7. The data availability statement.

Yours sincerely,
Frank Cai
