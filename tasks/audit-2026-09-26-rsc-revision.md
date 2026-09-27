# RSC Advances revision: completeness audit, 2026-09-26

Provenance: 18-agent read-only audit (9 checker groups, each with an independent adversarial
verifier) of branch `rsc-figure-fixes` at a7af4c7 against the verbatim referee reports
(decision email 2026-09-11). FINAL = the verifier's verdict. The key H2 clustering result was
re-run by hand in the main session:
  T=300 K: 17 vs 4, unit-level McNemar p = 0.007, system-clustered exact p = 0.152 (ex-ORB 0.156)
  T=600 K: 23 vs 4, unit-level p < 0.001,          system-clustered exact p = 0.066 (ex-ORB 0.055)
Context: RSC reminder escalated 2026-09-26 ("due shortly", possible withdrawal). Frank emailed
advances@rsc.org 2026-09-26 07:04 UTC asking for time to **9 October 2026**.
Not committed. The tasks/todo.md [x] marks predate this audit and several are wrong (E4, G1, I2, F1 partly).


##### GROUP R1.1-R1.2
- R1.1: checker=answered_by_argument FINAL=answered_by_argument agree=True
    * No first-principles E(Q) profile or single point exists for any of SrTiO3, BaTiO3 or bcc-Zr. The mandatory ask is declined, at response_to_referees.md:114-120 and manuscript.md:767-772.
    * The letter's resource defence ('no allocation', :118) sits beside tasks/todo.md:225-226, which budgets a rented box of about $20-60 for MLIP work. tasks/compute-runbook-rsc-revision.md has no R1.1/R1.2 or DFT item; grep returns zero hits.
    * Letter mismatch (a): 'one place' versus at least eight manuscript and ESI uses of screen-SSCHA agreement as cross-validation or reference.
    * Letter mismatch (b): the SrTiO3 gate is stated without the 3/5-model qualifier given at manuscript.md:266-267.
    * Letter mismatch (c): 120/120 controls are quoted without the A2 caveat at manuscript.md:342-347.
    * Letter mismatch (d): the claim that §S1.4 'separates method error from PES error' is contradicted by the letter's own :168-170 and by ESI :268-270.
    * The minimal fix still stands: PBE single points on the displaced structures behind the MLIP E(Q) maps. The cache stores q, band, dim, m_eff, a, b, c, Qs, dE and well_depth_meV but no eigenvector (results/cache/softmode_v3m24_*.json keys), so the patterns must be regenerated with phonopy modulation. SrTiO3's R mode is 2x2x2 (40 atoms) and MatterSim's deciding Zr mode is 3x3x1. For BaTiO3, MACE-MP-0's deciding mode is M=(1/2,0,1/2) in a 2x1x2 cell, not the 1x1x1 Γ cell.
- R1.1b (experimental Tc = global thermodynamics vs local curvature): checker=answered_by_argument FINAL=answered_by_argument agree=True
    * Add one Limitations sentence stating that the experimental labels mark thermodynamic, often first-order, transitions, while the screen and SSCHA probe dynamic stability.
    * Reword the letter at :111-112: bcc is excluded from scoring and is not labelled by SSCHA. Where included (Tables S5 and S6) it keeps the thermodynamic Tc.
    * Optional: in the letter, cite the screen's argmin-over-centroid criterion (manuscript.md:210-214). It is a free-energy comparison, not a local curvature.
- RESIDUE-R1.1 (remaining absolute-accuracy claims): checker=partial FINAL=partial agree=True
    * Replace 'gold standard', 'clean gold standard', 'tracks', 'validated' and 'harmonic (ground truth)' at the locations listed. Add one sentence in §2.5 that SSCHA here inherits the MLIP PES.
    * The 'clean gold standard for bcc' claim has to confront the paper's own §3.1 result. MACE-MP-0 and CHGNet have no Zr/Hf harmonic instability, so their SSCHA stability at 50 K is not evidence of thermal stabilisation. Report the bcc agreement with those 12 trivially agreeing units separated out (23/33 = 0.70), and note that MatterSim agrees in only 3/9.
    * The bcc-Zr repro and finite-size tests were run on zr_bcc/MACE-MP-0, whose harmonic Zr is -0.000 THz. Either say so, or rerun on MatterSim or SevenNet-0 Zr.
    * Fig. 4 caption: change 'tracks' to agreement in the stability call. Magnitudes are uncorrelated.
- R1.1-published-DFT (the plan's promised zero-compute route): checker=not_done FINAL=not_done agree=True
    * Add a table comparing ledger deciding-mode harmonic frequencies and well depths (ft_decide_harm_thz, ft_well_depth_meV) with published first-principles values for BaTiO3, SrTiO3 and ZrO2/HfO2, with the normalisation matched and each value verified at the source.
    * Set the bcc SSCHA stability-at-50 K claim (manuscript.md:505-509) against published first-principles finite-T bcc phonons (ref 5; possibly SCAILD work, to be verified), especially given that MACE-MP-0 and CHGNet have no Zr/Hf harmonic instability.
    * The letter never mentions this route. That contradicts the author's recorded plan (todo.md:225).
- R1.2: checker=answered_by_argument FINAL=answered_by_argument agree=True
    * No first-principles force or energy reference exists along any sampling path, and the ensembles must be regenerated before one can be made.
    * The OOD hypothesis is absent from the manuscript. Add it to §3.3 or §4, and say what does and does not exclude it.
    * The high-T anti-stabilisation (30 SSCHA false-unstable units, SrTiO3 down to -914 THz, 14/23 sign reversals with T) goes undiscussed. It is the pattern most consistent with R1.2's hypothesis, and it must be addressed in the manuscript and the letter.
    * The letter's §S1.4 'separating method from PES' heading (:163) is contradicted by its own :168-170 and by ESI :268-270.
- R1.2-args (validity of the three no-DFT arguments): checker=partial FINAL=partial agree=True
    * Withdraw or reword 'only the truncation changes' (letter :139; manuscript.md:552-553). No Table S2 row pair isolates truncation.
    * Drop the Table S11 swamping citation as evidence of uniform false-stabilisation (letter :143-146).
    * Present the within-cell fluorite control as a finite-size argument (its original purpose), not as a discriminator against R1.2.
    * Unused zero-compute evidence could be added: E(Q) maps out to Q=0.45 on the same MLIPs, and MatterSim's finite-T training data (verify against ref 10). Neither replaces a DFT benchmark.
- R1.2-v4 (one decisive MLIP-only test not run): checker=not_done FINAL=not_done agree=True
    * No completed include_v4=True Hessian exists for any unit.
    * ESI :305 ('≈tens of hours per unit') contradicts manuscript.md:595-597, which declines to extrapolate.
    * Table S2 comes from a pre-audit run (June 2026) that was never re-run in the pinned environments and is not asserted in verify_claims.py.
    * The claim that v4 is 'the physically correct escalation' / 'its own prescribed remedy' (manuscript.md:495-496, 592-593, 725-726) is in tension with manuscript.md:600-605 and with the small bubble correction in Table S2. It needs verification against Bianco 2017 and Monacelli 2025 before it is restated.
- RESIDUE-R1.2 (causal attribution to truncation still asserted): checker=partial FINAL=partial agree=True
    * Narrow the wording in the abstract, §3.3 (:575, :583-584, :616-621), §4 (:765-766, :770), the Conclusions, ESI :302 and ESI S4 :649-650 to 'consistent with the bubble-truncation failure predicted in ref 22; an MLIP error on thermally sampled configurations is not excluded without a first-principles benchmark'.
    * Resolve the internal contradiction between manuscript.md:583-593 and 600-605 about whether a v4 (any-order) fixed-reference Hessian could detect these condensations.
    * 'Methodological, not numerical' (:621; ESI :332) should read 'not a numerical blow-up'. It does not separate method from PES.
    * The letter repeats the attribution (:139, :155, :157-161) and needs the same narrowing.
  NEW ISSUES:
    + The bcc 'clean gold standard' is undercut by the paper's own harmonic data. Canonical ledger harmonic rows give -0.000 THz for MACE-MP-0 and CHGNet on both zr_bcc and hf_bcc (confirmed in manuscript.md:303-304). Of the 35 agreeing bcc screen/SSCHA pairs, 12 are these units, where the MLIP PES has no instability at all, so both methods agree trivially. Without them agreement is 23/33 = 0.70. MatterSim, with the deepest bcc harmonic instabilities (-1.96/-2.03/-2.21 THz), agrees with SSCHA in only 3/9 bcc units, all at 600 K. This bears directly on R1.1's named bcc-Zr and on ESI §S1.3's 'bound' (supplementary.md:215-218), which the letter cites as the one legitimate use of agreement.
    + The SSCHA robustness evidence for bcc-Zr was run on a unit with no instability. The 4-seed reproducibility test (+1.798 THz; manuscript.md:663-665; scripts/sscha_repro.py:11-12 defaults zr_bcc/mace_mp0) and the 3x3x3 finite-size test (manuscript.md:672-673; results/convergence_study.parquet contains only zr_bcc/mace_mp0 SSCHA rows) both use MACE-MP-0 on Zr, whose harmonic PES has no Zr instability (-0.000 THz). Phrases such as 'firmly stable (1.2-2.1 THz)' for MACE-MP-0 (manuscript.md:511; Fig. 3 caption :563-564) reflect the PES softening the paper reports in §3.1.
    + The paper's SSCHA data show systematic high-T false-unstables that neither the manuscript nor the letter discusses. Canonical non-bcc SSCHA gives 0, 5, 10 and 15 false-unstable units at 100, 300, 600 and 900 K. SrTiO3 is false-unstable in all 14 above-Tc units, at -3.4 to -914 THz against harmonic R-mode values of about -0.8 to -2.1 THz. 14 of the 23 units that are stable at 100 K go negative by 600-900 K (BaTiO3 CHGNet/MACE/ORB/SevenNet; KNbO3 CHGNet/ORB; ZrO2 and HfO2 for three models). Growth of instability with thermal amplitude is the signature R1.2's OOD hypothesis predicts. The manuscript mentions the wrong trend only for the fluorites (612-614, 696-697), and the letter confines its control to 100 K (:150-152).
    + The manuscript contradicts itself on the root cause. It says the failure enters at the truncation and that v4 is the remedy (manuscript.md:495-496, 583-593, 725-726). It also says 'a fixed-reference Hessian criterion of any order is blind to a condensation of that character' (manuscript.md:600-605). The author's own audit note (tasks/audit-2026-08-16.md:146-152) says that for even V the Hessian at Q0=0 reduces to the auxiliary curvature. Table S2 shows the whole bubble correction is only -0.02 THz (+2.892 to +2.872), so a resummation that rescales that term is unlikely to flip the sign. This weakens the letter's central R1.2 defence ('controls that hold the force engine fixed', manuscript.md:765-766). The direction of the v4 correction must be checked in refs 21 and 22; I have not verified it against the papers.
    + ESI §S1.3 (supplementary.md:211-213) says 'at T <= 300 K the ferroelectric wells are hundreds of meV deep and far from that crossing'. The ledger contradicts this. Deciding-mode well depths at T<=300 K are BaTiO3 28-43, KNbO3 25-59, PbTiO3 0-110 and SrTiO3 0-27 meV. KNbO3 is called stable at 300 K by 4 of 5 models, so those calls sit at the crossing. This is the same §S1.3 paragraph the letter's R1.1 answer relies on.
    + The letter's 'Also changed' (response_to_referees.md:532) says verify_claims.py 'now runs 29 assertions'. Today the script prints 32 PASS lines and exits 0, and tasks/todo.md L1 says 32/32. The count in the letter is stale.
    + Letter R1.1 (:99-100) says 'Scoring labels are documented experimental transition temperatures'. The six control labels come from the DFPT phonon database (manuscript.md:135-136), and the harmonic layer is scored against 'the published harmonic stability split' (:159). Minor overstatement.
    + The DOCX title is still the Digital Discovery-era title, 'Finite-temperature dynamic stability separates foundation machine-learning interatomic potentials that harmonic benchmarks rank equally' (paper/manuscript.docx, built 2026-09-10). It matches neither the revised title nor the letter's stated 'Original title'.
  REFUTED:
    - Not refuted but corrected: the 'DOCX contains none of the R1.1/R1.2 revisions' issue is accurate (0 hits for 'independent electronic-structure', 'within-cell', 'S1.4', 'Table S12' and 'Neither harmonic'; 'tens of hours' and 'clean gold standard' present). However, the letter already lists 'Rebuild DOCX (clean + tracked changes)' as outstanding (response_to_referees.md:544), so the author has acknowledged it and it is not a hidden gap.
    - Partly overstated: letter mismatch (e) in R1.1. ESI §S1.3 (supplementary.md:215-218) does frame the bcc agreement as a bound ('bounds the coupling's effect on the call'), consistent with the letter's 'bound on an approximation'. What is missing is any same-PES caveat, and the ESI asserts SSCHA is 'trustworthy'. The bound framing itself is present.
    - Partly wrong: R1.2-args (2)(a), 'the argument appears only in the letter, not the manuscript'. The manuscript does report all-five-model false-stabilisation on the fluorites (manuscript.md:611-612, 618) and SSCHA failures spanning float32 and float64 (527-528). What is absent is framing it as evidence against an MLIP-specific OOD artefact.
    - Factual slips in the checker's R1.1 and R1.2-args gaps. The cited cache, results/cache/softmode_v3_batio3_cubic_mace_mp0_sc222.json, is a superseded generation; a v3m24 file also exists. Both hold three modes (Γ 1x1x1, X 2x1x1, and M 2x1x2 or 1x2x2), not a single 1x1x1 map. MACE-MP-0's deciding BaTiO3 mode in the ledger is M=(1/2,0,1/2) in a 2x1x2 (20-atom) cell. The E(Q) maps reach Qmax = 0.45, not 'at least 0.14'. Neither slip changes a verdict.
    - Cover letter (paper/cover_letter.md:7, 32, 37-38) is confirmed stale (Digital Discovery, 'validated', 'gold-standard'), but RSC's required-files list for the revision does not include a cover letter. This is low severity and matters only if the file is reused.

##### GROUP R1.3
- R1.3-overall: checker=partial FINAL=partial agree=True
    * Rewrite the declined-references paragraph in the response letter (response_to_referees.md:198-217)
    * DOCX/PDF contain no R1.3 changes; no changes-marked version exists anywhere in the repo
    * References not in order of first citation (worse than the as-reviewed version)
    * The letter's R1.3 answer (:176-177) mentions only the Introduction and omits the Discussion clause at manuscript.md:742-747
- R1.3a-active-learning-sGP-thread: checker=complete FINAL=complete agree=True
    * Optional wording fix at manuscript.md:82: GAP (ref 31) was itself sparse; describe ref 37 as a 'scalable sparse-GP formulation with on-the-fly adaptive sampling'
    * Optional: 'sample efficiency' and 'solid electrolytes' (the referee's words) are not engaged explicitly; one clause on ref 37 (Li7P3S11) would cover both
- R1.3b-HT-screening-thread: checker=partial FINAL=partial agree=True
    * No Discussion sentence linking the soft-mode screen to ML-driven screening frameworks (refs 34-36)
    * Energy-materials and catalyst screening are not mentioned, and the letter does not state that this narrowing is deliberate
    * Wording at manuscript.md:88: GNoME is a GNN energy-model search, not a search driven by a 'universal potential'
    * Wording at :90-91: Matbench Discovery scores convex-hull stability; 'harmonic' belongs to the phonon benchmarks
- R1.3-refs-27-37-verification: checker=complete FINAL=partial agree=False
    * manuscript.md:82 GAP/sparse-GP attribution
    * manuscript.md:88 GNoME cited for 'universal potentials' driving searches
    * manuscript.md:746-747 overstates ref 30
    * No deposited record of the Crossref verification that commit 7ea88e6 claims (todo I2 is marked [x], but the repo has no artefact)
- R1.3-declined-suggestions-letter-tone-and-accuracy: checker=partial FINAL=partial agree=True
    * 'Four are review articles' is wrong: three of the five are reviews (JPCL 12, 8115 and CPR 6, 021401 are primary research)
    * The exclusion criteria are applied inconsistently: refs 31-36 fail the 'treats anharmonic crystals / phase transitions / SCHA' test, and ref 20 is a perspective, yet both are cited
    * The universal negative 'none of the six treats anharmonic crystals' is contestable (superionic Li7P3S11 melting and crystallization in PRB 103)
    * The quote at :210-211 is a selective truncation that the referee can rebut from their own sentence
    * Tone: the single-group identification plus named authors plus 'on a title' reads as an accusation; move any such concern to confidential comments to the editor
    * Consider citing ACR 59, 103 or AEM 71046 on their merits: they speak directly to sample efficiency, uncertainty quantification, foundation-model fine-tuning and HT screening, which also bears on R2.1 and R2.2
- R1.3-transparent-peer-review: checker=blocked_on_frank FINAL=blocked_on_frank agree=True
    * Transparent peer review decision not made (response_to_referees.md:546, todo.md:208)
    * Rewrite :198-217 before deciding
- R1.3-ref20-author-list: checker=complete FINAL=complete agree=True
    * The fix is not in manuscript.docx or manuscript.pdf (tracked under the DOCX item)
    * Hyphen and page range here versus the en dash in ref 19 (style item)
- R1.3-rsc-reference-style: checker=partial FINAL=partial agree=True
    * Renumber in order of first citation, then update the ESI (supplementary.md:301,305) and letter ref numbers
    * Make pages consistent (first page only, or ranges throughout)
    * Make et al. usage consistent
    * Ref 14: give vol/page (35, 1702)
    * Ref 2: fix author order (Han, Gao, Li, Guo, Lu) and pin the arXiv version
    * Split or label the compound refs 5 and 15
- R1.3-docx-propagation: checker=not_done FINAL=not_done agree=True
    * Rebuild the clean manuscript.docx and supplementary.docx from the current .md after renumbering the references
    * Produce the changes-marked version against manuscript-as-reviewed.md
    * manuscript.pdf is also stale
  NEW ISSUES:
    + response_to_referees.md:207-209 applies two exclusion rules ('review articles rather than primary sources'; 'none ... treats anharmonic crystals, phase transitions in the systems studied here, or the SCHA') that the revision's own citations fail. Refs 31 (GAP: C/Si/Ge), 32 (rare events), 33 (water/liquid Ga), 34, 35 and 36 do not meet the second rule. Ref 20 is self-described as 'This perspective' (OpenAlex abstract), so it fails the first. A referee will see a double standard.
    + manuscript.md:81-82 'Gaussian-process potentials^31^ and their sparse variants,^37^' is historically inaccurate. The GAP PRL (ref 31) already used 'a sparsification procedure ... 300 such sparse configurations' (arXiv:0910.1019 full text). Ref 37's contribution is a scalable SGPR with on-the-fly adaptive sampling.
    + manuscript.md:88 cites GNoME (ref 34) for 'Universal potentials now drive high-throughput stability searches'. The GNoME abstract describes 'graph networks trained at scale' finding 2.2 M structures below the hull, which is an energy-model search, not one driven by an interatomic potential.
    + The ACR 59, 103 conspectus (a declined item) matches the referee's request point for point: sample efficiency ('100–1000 quantum calculations', '10–100× reduction in data requirements'), 'robust Bayesian committee machine', Li7P3S11 solid electrolytes and 'accelerate high-throughput screening'. Declining it as not bearing on the work is the weakest part of the letter.
    + PhononBench count: manuscript.md:52 '~1.1×10⁵ generated structures' matches arXiv:2512.21227 v1/v2 (108,843). The current v3 (updated 2026-06-11) says 133,838, and ref 2 (:858) gives no version. Pin 'arXiv:2512.21227v2' or update the number. Ref 2's author order is also wrong (arXiv: Han, Gao, Li, Guo, Lu).
    + The ESI cites main-text references by number (supplementary.md:301 'refs 21, 22', :305 'ref 22'), so any renumbering must also be propagated there, not only to the letter.
    + paper/toc_entry/toc_blurb.txt (2026-07-08, pre-revision) still says the five MLIPs 'disagree sharply at finite temperature'. That contradicts the revised position that the finite-T ranking is suggestive and that this layer separates models less (response_to_referees.md:353-355; manuscript.md:803-806). This is outside R1.3 but part of the upload package.
    + paper/manuscript.pdf (2026-08-30) is also stale and contains the old title-only ref 20. If a PDF is used as the changes-marked or clean upload, it has the same problem as the DOCX.
    + The letter's R1.3 answer (:176-177) counts 'eleven new entries (refs 27-37)'. Against the version the referees read, the list grew from 18 to 37, because refs 19-26 were added in August. §A of the letter explains 19-20 and the §2.4 derivation, but the changes-marked file will show 19 new references. Add one clause saying so.
  REFUTED:
    - Checker R1.3b gap 'This thread appears in the Introduction only' is overstated. The Discussion does place the screen within a CSP screening pipeline at manuscript.md:721-726 ('for screening harmonically-unstable displacive candidates, the cheap all-imaginary-mode free-energy screen is the more reliable indicator') and :730-734 ('A practitioner filtering generative-CSP output ... PhononBench and comparable filters'). What is missing is a link to the ML-driven screening frameworks of refs 34-36 and to energy/catalyst screening.
    - Checker calls the letter's :210-211 quote a misquote. It is a verbatim substring of the referee's sentence (reports line 26). The fair criticism is selective truncation that changes the scope, not misquotation. The recommendation to delete it stands.
    - Checker other_issue #4 and R1.3a optional gap 1 suggest citing refs 27-29 to support 'fine-tuning would improve finite-temperature performance substantially' (manuscript.md:734-736). Refs 27-29 are system-specific potentials trained from scratch on the fly, not fine-tuned foundation models, so they do not support a fine-tuning claim. Ref 20's abstract ('can be addressed with targeted fine-tuning') does, and so does AEM 10.1002/aenm.71046 ('foundation-model fine-tuning').
    - Checker other_issue #1 asserts the editor's sentence (reports line 6) is RSC boilerplate 'not a targeted signal'. That is plausible but was not verified today (rsc.org returned 403). The conclusion stands either way: keep any citation-stacking framing out of documents sent to RSC.

##### GROUP R1.4-R3.2
- R1.4a-methodological-trap: checker=partial FINAL=partial agree=True
    * Letter silent on the 'methodological trap' sentence. Add a paragraph: the phrase has been removed, it is replaced by manuscript.md:488/:495-498, and the justification now sits at :575-605 and :689-700.
    * manuscript.md:552-553, response_to_referees.md:138-139 and :460-461, and the Table S10 caption (supplementary.md:588) all describe Table S2 as a controlled change of truncation order with a sign change. No such row exists, because v4=True never finished (supplementary.md:296). Rewrite as 'consistent with the truncation predicted by ref 22' and state that the v4 remedy is untested.
    * The residual unhedged synonyms ('systematically', 'confirms ... methodological') at manuscript.md:37, :518, :591-592, :621, :792 and supplementary.md:332 conflict with the declined significance at :550-551.
    * Halide overreach: the SSCHA grid has only CsSnI3 (manuscript.md:143 says it 'covers the perovskites'), and there SSCHA is not false-stable at T<=300 K (7/8 correctly unstable) but false-unstable at 600/900 K (6/6).
    * The mechanism change against the as-reviewed text is not disclosed in the letter.
- R1.4b-sample-sizes-and-stopping-criteria: checker=partial FINAL=partial agree=True
    * State that 2560 is the configured maximum and that the number of populations used and the convergence status of each unit were not recorded (finite_t.py:1058, :1083).
    * Report the actual convergence thresholds (meaningful_factor=1e-4, min_step_dyn=0.5) and stop calling max_ka=20 the stopping criterion (supplementary.md:599; response :226-227).
    * Fix §2.5 (manuscript.md:273-274): the initialiser is built from phonopy full force constants, not ASE, at 0.03 Å displacement. Report the 0.03 Å.
    * Change 'per-unit' at manuscript.md:501-502 to 'per family and model', or supply a per-unit table.
- R1.4c-gradient-history-and-Hessian-uncertainty: checker=answered_by_argument FINAL=not_done agree=False
    * Neither the gradient history nor a Hessian uncertainty is reported for any SSCHA unit. R1 said 'must report'.
    * The acoustic-zero residual is a symmetrisation check (residuals at 1e-31 to 1e-14 THz on bcc), not a noise bound on the soft mode. Relabel it at supplementary.md:601 and response :234-238.
    * The letter item at :230-232 is untagged despite being undone. Tag it [PENDING] or supply the data.
    * Cheapest close: in the R1.4d re-run, save the minimiser's gradient and error history per population, and use the seed spread of the lowest Hessian eigenvalue as the Hessian uncertainty.
- R1.4d-four-seed-extension-to-displacive-systems: checker=not_done FINAL=not_done agree=True
    * Run sscha_repro.py for BaTiO3 and ZrO2 (MACE-MP-0, 100 K, seeds 0-3) on a rented GPU box. The code already uses the phonopy initialiser via compute_finite_t_sscha (finite_t.py:1038).
    * Make the per-seed values persist (a file or ledger rows). Today the existing bcc-Zr per-seed values are not deposited anywhere, despite the Data availability wording (manuscript.md:816-817).
    * If it cannot be run before submission, replace 'work in progress' (supplementary.md:603) with a declared limitation in §3.5, §S2.4 and the letter, and correct the letter's claim that §S2.4 states it.
    * Report the result whichever way it falls (runbook :53-58).
- R1.4e-bcc-SSCHA-convergence-evidence: checker=not_done FINAL=not_done agree=True
    * In the same re-run as R1.4d, record the lowest frequency after ForcePositiveDefinite alongside the converged Hessian value for at least one bcc unit at a low and a high temperature (e.g. zr_bcc/MatterSim at 50 and 600 K), plus the gradient history. Alternatively, state the limitation explicitly.
    * Qualify 'clean gold standard' (manuscript.md:35-36, :500) and 'real signal rather than sampling noise' (:665-666) until that check exists, and address the non-monotone and softening temperature trends visible in Fig. 3 data.
- R3.2a-main-text-narrowing: checker=complete FINAL=partial agree=False
    * Change manuscript.md:699-700 so it rules out a missing q-point, not finite size generally (e.g. 'A missing-q explanation would have to account for ...').
    * Soften manuscript.md:616-618 to say the fluorite false-stable is consistent with the truncation at the 2x2x2 cell used.
    * Fix the misquote at response_to_referees.md:367-368 and engage R3's word 'marginal'. Drop 'stronger than a convergence test' (:374).
- R3.2b-missing-q-rebuttal-extension: checker=partial FINAL=partial agree=True
    * Add one sentence (manuscript §3.5 and letter) on SrTiO3: SSCHA never false-stabilises it, so the missing-q objection cannot apply; report its high-T false-unstable divergence.
    * Replace 'finite-size explanation' at manuscript.md:699-700 with 'missing-q explanation', and 'exactly as ... finds' at :698-699 with 'consistent with'.
    * Soften the letter's 'now general' (:372) and 'stronger than a convergence test' (:374).
- R3.2c-leftover-zone-boundary-claims-in-ESI-and-elsewhere: checker=partial FINAL=partial agree=True
    * Rewrite ESI §S2.4 bullet (supplementary.md:345-350) to match manuscript.md:678-700: name the R and X points, state 'no convergence claim for zone-boundary systems', and replace the BaTiO3-only sentence with the within-cell fluorite control (framed as missing-q only).
    * Add the zone-boundary caveat to ESI S4 (supplementary.md:653-654).
    * Qualify supplementary.md:325-333 ('consistent with', 2x2x2) and optionally the Abstract at manuscript.md:38-40.
    * Optionally fix the verify_claims.py:131-133 comment so the deposited code does not repeat the overreach.
- PKG-docx-stale-for-R1.4-and-R3.2: checker=not_done FINAL=not_done agree=True
    * Rebuild the clean manuscript and ESI DOCX from the current .md sources after the text fixes, and produce the marked-changes version against paper/submissions/rsc-advances-2026-07/*-as-reviewed.md.
    * Write an RSC Advances revision cover note. The existing cover_letter.md is the superseded DD submission letter.
  NEW ISSUES:
    + manuscript.md:552-553 ('the controlled diagnostic of §S2.2, in which only the truncation order changes and the sign of the answer changes with it'), response_to_referees.md:460-461 (same wording) and the Table S10 caption at supplementary.md:588 ('the case that the effect is real rests on the mechanism isolated in Table S2') all misdescribe Table S2. The only row that changes truncation order (include_v4=True, supplementary.md:296) has no number, and the sign change is between the harmonic and ForcePositiveDefinite rows. Since §3.3 explicitly declines significance and rests the effect on this mechanism, this is the load-bearing justification R1.4 asked to be strengthened.
    + The response letter contradicts itself on the unit-level p-values. R3.3 (:450-452) says 'I am not quoting that number', and Table S10 (supplementary.md:588) says the unit-level column 'should not be quoted'. Yet R3.5 (:497-503) calls the FE-oxide contrast 'significant with ORB-v2 (14 vs 3, p = 0.013) and not significant without it (10 vs 3, p = 0.092)' and says 'The claim therefore rests on the combined displacive set ... 26 against 3, p = 2 × 10⁻⁵'. manuscript.md:557-558 and the Conclusions at :795-796 lean on the same unit-level p-values.
    + The only cell-convergence evidence the paper retains, bcc-Zr SSCHA 2x2x2 -> 3x3x3 (+1.80 -> +1.56/+1.55 THz; manuscript.md:671-673, :686-687; supplementary.md:345), comes from the superseded June v1 generation. results/convergence_study.parquet was last committed in a24cbc6 on 2026-06-22, has no method_version column, and has model_version mace-torch=0.3.16. It predates the pinned re-measurement and the phonopy initialiser. The manuscript flags the soft-mode rows of that file as superseded (:674-676) but not the SSCHA rows. The Data availability statement (:817-822) says the reported SSCHA results are a full v3 re-measurement.
    + manuscript §2.5 (:273-274) describes the SSCHA start as 'an ASE finite-displacement harmonic dynamical matrix', but the production code builds it from phonopy full force constants (finite_t.py:1036-1038), as §3.5 itself says (:667-668). It also uses disp=0.03 Å (finite_t.py:1002; cli.py:150 passes only supercell), not the 0.01 Å harmonic amplitude, and that value is never reported.
    + Several bcc SSCHA curves soften with temperature (ledger: ti/MACE 1.728 -> 1.375 THz, ti/SevenNet 1.268 -> 1.175, hf/SevenNet 0.331 -> 0.300, zr/MACE 1.796 -> 1.789 from 50 to 600 K). That is the opposite of the entropy stabilisation the paper uses to call bcc SSCHA a 'clean gold standard' (manuscript.md:35-36, :500-511), and no convergence diagnostic rules out that these values echo the ForcePositiveDefinite start (tasks/lessons.md:46-47).
    + response_to_referees.md:10-11 says 'Everything untagged is done and is in the revised manuscript', but the R1.4 gradient-history and Hessian-uncertainty paragraph (:230-232) is untagged and the data are not supplied.
    + manuscript.md:501-502 says Table S11 gives 'the per-unit numerical diagnostics', but Table S11 (supplementary.md:605-621) is aggregated per family and model.
    + SrTiO3 SSCHA is false-unstable at 300-900 K in all 15 units with runaway magnitudes (e.g. MACE -3.4/-20.2/-27.1 THz; ORB-v2 to -913.7 THz; ledger). The main text never reports this; only supplementary.md:322 mentions 'SrTiO₃ high-T rows' among the blow-ups. It bears on R3.2's SrTiO3 clause and on the 'systematically false-stabilises' framing.
    + paper/cover_letter.md is still the July Digital Discovery letter (old title at :11-12, 'AUC 0.75' at :44, addressed to DD editors at :7) and must not be reused for the RSC revision.
  REFUTED:
    - Checker other_issue on halides: 'On CsSnI3, SSCHA calls the phase unstable at T<=300 K in 6 of 7 units'. The count is wrong. The ledger has 8 CsSnI3 SSCHA units at T<=300 K (CHGNet, MACE-MP-0, MatterSim and SevenNet-0 at 100 and 300 K; ORB-v2 was never run on CsSnI3), and 7 of 8 are unstable. Only MACE-MP-0 at 100 K (+0.355 THz) is stable. The substance stands and is stronger: all 6 units at 600/900 K are false-unstable (-5.7 to -108.6 THz).
    - Checker R1.4a detail: the list of 'trap' hits in manuscript.md omits :596 ('extrapolating'). This is trivial and does not change the verdict; no genuine 'trap' wording remains in the .md sources.
    - Checker R1.4c label 'answered_by_argument': not refuted on substance, but mislabelled. The letter offers no argument that gradient history or Hessian uncertainty are unnecessary, only a concession that they were not retained. Relabelled not_done.

##### GROUP R1.5
- R1.5a-mass-weighting: checker=complete FINAL=partial agree=False
    * supplementary.md:117-143 never says that M and V(Q) are summed over the mode's minimal commensurate cell, whose size varies by mode (1 to 8 f.u. in perovskites). 'Nothing observable depends on the choice' (140) is true only for λ-rescaling. Changing the cell normalisation changes calls, and the SrTiO3 gate fails for MACE-MP-0 under a doubled cell (see R1.5f).
    * The unit wording is loose (131-133). u is dimensionless with max|u_i,α|=1, and Q is in Å.
    * Q is the largest Cartesian component, not the largest atomic displacement (134-135). It can be off by up to √3 for off-axis polarisations.
    * The invariance claim (140-142) ignores Q-unit implementation constants: q_max 0.45 Å, the Q0 box 0.6 Å and the 0.0075 Å threshold (finite_t.py:797, 740, 757).
    * The letter at response_to_referees.md:262-266 repeats the invariance claim. It also says 'the referee was right to flag [normalisation] as missing', but the referee asked only for a mass-weighting derivation.
- R1.5b-gaussian-width-self-consistency: checker=complete FINAL=complete agree=True
    * The root-finder fallback is undisclosed (finite_t.py:717). It fires on 4.6% of Q0 evaluations and never at the deciding argmin.
    * The condensation threshold (Q0 > 0.0075 Å) is undisclosed. The text says 'Q₀ > 0' (manuscript.md:212). The comment at finite_t.py:755-756 describes a ~kT check that is not implemented.
    * The even-parity assumption V(−Q)=V(Q) and Q≥0-only sampling are never stated or justified. For 24 bcc modes with 3q≡G, parity depends on the phonopy modulation phase and is unchecked.
    * Only stationarity is shown, not that it is a minimum. This is low priority.
- R1.5c-mode-mode-coupling: checker=partial FINAL=partial agree=True
    * No estimate or bound on λ_kl is given (supplementary.md:207-209), although todo E3 promised one.
    * The premise 'hundreds of meV' at supplementary.md:212 is false (max FE wells 28-110 meV per cell), so the T* versus call argument lacks support.
    * The 'bound' at supplementary.md:215-218 uses curvature-sign agreement (35/45), not call agreement (31/45). On units where a mode was actually screened, call agreement is 4/18.
    * The irrep claim at supplementary.md:193-195 is false for the screened set (split T1u triplet in BaTiO3/CHGNet). It omits trilinear couplings and has no symmetry analysis behind it.
    * The letter at response_to_referees.md:90-94 and 270-274 inherits these errors.
- R1.5d-sensitivity-fit-window: checker=complete FINAL=partial agree=False
    * There is no committed script or results JSON for the 3×/8× numbers, and verify_claims.py has no assertion for them.
    * The 60 meV floor is never varied. The kept point set is identical at 3×, 5× and 8× for 261/378 modes, so the sweep cannot probe them. Floor at 30 and 120 meV gives 2/756 and 8/756 flips; report this.
    * A dangling leftover '(e.g. BaTiO₃)' sits at supplementary.md:56.
    * The result is a prose bullet under 'Soundness fixes', not a systematic table, and the mode-level versus unit-level distinction is not stated.
    * Fitted wells with no sampled well exist (Ti/ORB-v2, 18 modes, up to 376 meV). This is undisclosed; it does not change any unit call.
- R1.5e-sensitivity-sampling-range: checker=not_done FINAL=not_done agree=True
    * The production sampling range (0.45 Å, 10 quadratic points) is not stated anywhere.
    * Unbracketed wells (77/378 modes) and scan-edge Q0 values (26/400 units) are undisclosed.
    * No sweep is reported. The truncation test (3/756, 9/756, 0 unit flips) can be added without compute; extension needs compute.
    * The letter omits this axis while claiming 'Done'.
- R1.5f-sensitivity-cell-commensurability: checker=not_done FINAL=not_done agree=True
    * Screen sensitivity to cell commensurability is not reported anywhere. §3.5 and §S2.4 are SSCHA-only and call the screen's convergence rows superseded.
    * The frozen-cell normalisation convention is undisclosed and outcome-determining. FE recall ranges over 5/30, 16/30, 26/30 and 28/30 depending on convention, and the SrTiO3 gate breaks under doubling or per-f.u. normalisation.
    * The FC-supercell axis (which q are searched) was not re-run at 3×3×3 or 4×4×4 after the §2.4 correction.
    * The letter at response_to_referees.md:278-279 points to the wrong section. todo E4 [x] is false.
- R1.5g-systematic-sensitivity-table: checker=not_done FINAL=not_done agree=True
    * There is no ESI table covering fit window (multiplier and floor), sampling range, and cell/frozen-cell convention, with columns for mode flips, unit flips, FE recall and SrTiO3 gate.
    * No committed script or results JSON produces the numbers, and there are no verify_claims assertions.
    * The todo E4 [x] mark is false.
- R1.5h-response-letter-accuracy: checker=partial FINAL=partial agree=True
    * 'Done' at line 257 overstates: the sampling range and screen commensurability are missing.
    * The pointer at 278-279 goes to the SSCHA-only §3.5.
    * Lines 88-94 and 270-274 inherit the coupling errors, and lines 55-56 mislabel curvature agreement as call agreement.
    * The 'added in August, before the reports' aside (257-259) should be recast as new in this revision.
    * The outstanding checklist (540-549) omits all R1.5 work.
- PKG-esi-docx-lacks-S1.3: checker=not_done FINAL=not_done agree=True
    * Rebuild both .docx files, clean and marked-changes, from the current .md after the R1.5 fixes. The existing .docx files ship none of §S1.3/§S1.4 and no Tables S12-S13.
  NEW ISSUES:
    + supplementary.md:52-56: the fit-window sweep cannot change the fit for 261 of 378 screened modes (69%). With the 60 meV floor and 10 samples, the kept point set is identical at 3×, 5× and 8×; this includes all 101 modes shallower than 7.5 meV. The floor itself is never varied. My compute-free test: floor 30 meV gives 2/756 flips, 120 meV gives 8/756, so the conclusion likely holds but is not reported.
    + supplementary.md:56: '(e.g. BaTiO₃)' is a leftover. Before commit f24e913 it closed the sentence '…does not wash out a shallow well (e.g. BaTiO₃)'; now it dangles after 'no headline number depends on the window choice'.
    + Fit artefact, undisclosed: 18 Ti/ORB-v2 modes (softmode_v3m24_ti_bcc_orb_v2_sc666.json) and 1 PbTiO3/ORB-v2 mode have every sampled dE ≥ 0 (well_depth_meV = 0), yet the fitted sextic has a 100-376 meV minimum at Q≈0.042 Å between sample points. 48 mode-temperature condensation calls rest on these invented wells. No unit call depends on them, since each affected unit has another condensing mode with a real sampled well. ESI §S1.4 (supplementary.md:263-264), 'condensation calls therefore always sit on a potential that genuinely possesses a displaced minimum', is true only of the fitted potential, not of the sampled data.
    + supplementary.md:645-646 (§S4 Threats to validity) also calls 0.78 agreement 'on the stability **call**'. The checker flagged manuscript.md:512 and the letter, but not this third instance. The call agreement is 31/45.
    + response_to_referees.md:475 says 'Tables are now S1–S12', and lines 481-482 list additions only through S12. The ESI now has Table S13 (supplementary.md:623), so the letter is internally stale.
    + response_to_referees.md:263 says the referee 'was right to flag [the normalisation convention] as missing'. R1.5 asks only for a 'mode mass-weighting' derivation; the normalisation point is the author's own addition.
    + SrTiO3 gate detail under the common 2×2×2 convention: MatterSim's Γ mode condenses at 100 K. The unit call is unchanged, but the manuscript's narrative that 'the Γ mode is quantum-suppressed and never condenses' (manuscript.md:226-228) is also convention-dependent.
  REFUTED:
    - Checker other_issue on ESI table order (Table S12 printed before Table S2): this is not unaddressed. response_to_referees.md:476-479 explicitly discloses the out-of-order placement and justifies keeping the number stable. Only the caption-after-table layout (supplementary.md:243-251, unlike every other table) remains a genuine but cosmetic inconsistency.
    - Checker R1.5b gap 'even-parity … fails for 24 of 378 screened modes': overstated. I confirm 24 modes have 3q≡G and 2q≢G (all bcc), so no lattice translation reverses the pattern. For monatomic bcc, however, inversion through an atom site maps a cos-phase modulation Q→−Q, and the symmetry-allowed cubic invariant for this star is Im(η³), which vanishes when the modulation phase is a multiple of π/3. E(Q)=E(−Q) is therefore unverified (it depends on phonopy's eigenvector phase), not shown to fail. The correct gap is 'unstated and unchecked'.
    - Checker's frozen-cell mode-level counts '91/756 (N=2)' and '178/756 (N=8)' reproduce as 89/756 and 176/756 with my independent solver, which matches the ledger on all 1512/1512 calls (the checker's matched 1511/1512). Its unit-level numbers reproduce exactly: common-FC cell 14 flips and recall 28/30, per-f.u. 39 flips and recall 5/30, doubled-cell recall 26/30. Its fallback count 8,439 reproduces as 8,442, and 'never at argmin' is confirmed (1512/1512 roots). These are minor numeric corrections; the substance stands.
    - Checker R1.5e count '76 of 378 modes still decreasing at Q=0.45 Å' is 77 by a strict E[-1]<E[-2] test (CsPbI3 24, not 23). This is immaterial to the verdict.
    - Checker's verdicts of 'complete' for R1.5a and R1.5d are downgraded to partial (see items): the frozen-cell normalisation is omitted from the mass-weighting convention, and the window sweep is inert for 69% of modes and has no committed script.

##### GROUP R1.6-R3.1-R3.3
- R1.6: checker=partial FINAL=partial agree=True
    * The replacement headline, McNemar p = 0.007 (Abstract :26, intro :107, §3.2 :402-405/:415, §4 :709), is unclustered. System-clustered p is 0.152 at 300 K and 0.066 at 600 K (reproduced with two methods).
    * 13 of the 17 discordances at 300 K sit in three systems where every model fails identically (CsSnBr3 is 8 K above T_c; BaTiO3/KNbO3 are the screen's T* underestimates). Without CsSnBr3 the count is 12 vs 4, unit-level p = 0.077.
    * The Abstract does not label p = 0.007 as marginal homogeneity (layer difficulty), although the letter :436-437 says it is labelled that way 'wherever it appears'.
    * manuscript.docx still says 'non-predictive' and carries the intermediate title.
    * The letter never tells R1 or R3 that φ moved from 0.11 to −0.129 and McNemar from 0.34 to 0.007. The §A table (:39-48) omits these rows, although number_changes.md:37-39 has them.
- R3.1: checker=partial FINAL=partial agree=True
    * The aligned H2 claim is worded as confirmed/significant/established (:107, :404-405, :415, :707), but the test behind it is unclustered (clustered p = 0.152 at 300 K).
    * The letter does not declare that the §3.2 formulation R3 endorsed ('weak positive', 'necessary but not sufficient', 'we test, and reject') was replaced by 'neither confirmed nor rejected' after re-measurement. It says only that the §3.2 formulation 'is carried'.
    * The letter's :346-347 cites the wrong commit and month for removing 'confirm H2 in a strong form' (actually 0fbea7d, 2026-08-16, not dc11ed7 in September).
    * The letter's title-span argument (:353-355) uses the 19-system harmonic column, the denominator artifact the letter withdraws at :400-407.
    * Old framing survives in the TOC graphic ('harmonic leaders ≠ finite-T leaders'), toc_blurb.txt ('disagree sharply'), cover letter, DA statement and manuscript.docx.
- R3.3-counts-intervals: checker=partial FINAL=partial agree=True
    * The Abstract quotes bare rates: accuracy 1.00, 15%, 8%, 0.900 (:22-28).
    * The SSCHA recall 0.19 and the fluorite recall 1.00 in §3.3, the Fig. 5 caption and §4 limitations have no interval in the main text (:519-530, :571, :611, :780).
    * The letter's claim that 0/57 and 0/120 carry one-sided intervals is false (:390-391).
    * The figures show counts or rates without intervals (fig_displacive_recall, fig_ensemble_guardrail).
- R3.3-AUC-clustered: checker=complete FINAL=complete agree=True
    * The 'factor of four in p' at manuscript.md:641 is wrong against the deposited JSON (0.0035/0.0003 ≈ 12). This is a one-phrase fix.
    * The fig_ensemble_guardrail title shows AUCs without the clustered p/CI (cosmetic).
- R3.3-Spearman: checker=complete FINAL=partial agree=False
    * The letter (:445) says Spearman is 'now reported as descriptive only', but it appears in neither the manuscript nor the ESI. Either add one ESI sentence or table, or say it was removed.
    * manuscript.md:513 and supplementary.md:647-648 assert 'not rank-correlated' from an untested ρ = 0.11 over clustered pairs.
    * The §3.2 table header puts 'clustered p' (the φ test) right after 'McNemar exact p' (:408), which invites the reading that McNemar was clustered.
- R3.3-composition-n60-n75: checker=complete FINAL=complete agree=True
- R3.3-McNemar-failure-to-reject: checker=complete FINAL=partial agree=False
    * manuscript.md:415 and letter :414 say 'absent at 100 K' for p = 0.727. It should read 'not significant' / 'not resolved'.
    * manuscript.md:513 and supplementary.md:647-648 read an untested ρ = 0.11 as 'not rank-correlated'.
    * The letter's R2.2 (:326) says 'It carries no usable information', contradicting manuscript :644-648 ('failure to resolve rather than ... absence').
- R3.3-H2-McNemar-clustering: checker=not_done FINAL=not_done agree=True
    * Add a system-clustered paired test for the H2 transfer asymmetry at every ladder T, next to the unit-level McNemar in the §3.2 table (:408-413).
    * Downgrade 'significant' / 'confirmed' / 'establish' at the Abstract :25-26, intro :107, §3.2 :404-405 and :415, and §4 :707-709 to the §3.3-style wording: large, one-directional in unit counts, not significant at system level (clustered p 0.15 / 0.07). Alternatively, argue explicitly why clustering does not apply here.
    * Correct the letter at :424 ('Every test ...'), :413-414 and R1.6 :287-288.
    * Add a verify_claims assertion for the clustered H2 p (currently :82-85 pins only unit-level p = 0.007).
- R3.3-ranking-suggestive: checker=partial FINAL=partial agree=True
    * manuscript.md:473 'ORB-v2 remains the weakest finite-temperature screener' is a ranking claim, and it is contradicted by the Table S5 full-ladder row.
    * manuscript.md:442 and supplementary.md:84 wrongly attribute 'weakest' to §3.1.
    * The letter §A :52-53 says 'SevenNet-0 leads both layers ... §3.2 says so explicitly', contradicting the withdrawn-ranking stance and §3.2's own wording.
    * verify_claims.py:80-81 pins a ranking.
    * The TOC graphic asserts a ranking that is false on the revised numbers.
- RESIDUE-screen-vs-SSCHA-claims: checker=partial FINAL=partial agree=True
    * The §5 :795-796 survives/does-not-survive distinction exists only in the unquotable unit-level p-values.
    * The Abstract :40-42 and §4 :722-724 state the screen is 'more reliable' without the §3.3 non-significance caveat.
    * The letter's R3.5 :497-503 uses unit-level significance language that its R3.3 :450-453 disowns.
- RESIDUE-title: checker=blocked_on_frank FINAL=blocked_on_frank agree=True
    * Frank's sign-off on title (1) is pending.
    * Once decided, propagate it to the DA statement, the ESI header, the rebuilt DOCX, the cover letter (if uploaded) and the TOC blurb.
- PKG-docx-claims: checker=not_done FINAL=not_done agree=True
    * Rebuild the clean manuscript DOCX, the ESI DOCX, and a tracked-changes DOCX against manuscript-as-reviewed.md. Figures at 600 dpi.
- PKG-TOC-claims: checker=not_done FINAL=not_done agree=True
    * Regenerate the TOC graphic from the current numbers with no ranking claim, and rewrite the blurb to match the revised title and §3.2.
- RESIDUE-letter-stats-consistency: checker=partial FINAL=partial agree=True
    * §A table omits the H2 φ and McNemar changes the referees quoted.
    * ':424 Every test now permutes or resamples whole systems' is false for the H2 McNemar.
    * The one-sided-interval claim (:390-391) is false.
    * 'Reported as descriptive only' (:445) is not true of the manuscript.
    * The title-span numbers (:353-355) are on the 19-system basis.
    * R3.5 quotes unit-level p-values that R3.3 disowns.
    * R2.2 :326 'no usable information'.
    * §A :52-53 'SevenNet-0 leads both layers'.
    * 'Tables S1–S12' (:475) but the ESI now has S13.
    * Wrong commit and month for the 'strong form' removal (:346-347).
    * 'absent at 100 K' (:414).
    * '29 assertions' (:532) against 32 executed.
    * DRAFT banner and two [PENDING] tags still present.
  NEW ISSUES:
    + response_to_referees.md:346-347 says 'confirm H2 in a strong form' 'was removed in September (commit dc11ed7)'. git log -S"confirm H2 in a strong form" -- paper/manuscript.md shows it was removed in 0fbea7d on 2026-08-16. dc11ed7 (2026-09-10) does not touch that phrase; it left 'non-predictive' in place (git show dc11ed7:paper/manuscript.md line 535). The letter cites a commit the referee could check, and that commit does not show the change.
    + manuscript.md:415 ('absent at 100 K') and letter :414 read McNemar p = 0.727 (5 vs 3) as evidence of absence, the exact R3.3 failure-to-reject error. The same sentence correctly says 'not significant' for 900 K.
    + manuscript.md:640-641: 'overstated the significance, by about a factor of four in p'. The deposited values (stats_hardening.json h3_guardrail.auc_vote_disagreement) are clustered 0.0035 and naive 0.0003, a factor of about 12.
    + ESI Table S7 caption (supplementary.md:501) says the n = 60 and n = 75 sets 'share their clustering structure'. The paper therefore declares the n = 75 set clustered in its own words, yet quotes an unclustered McNemar on it as 'the significant result' (manuscript.md:404-405).
    + Of the 17 H2 discordances at 300 K, 13 fall in three systems where all or four of the models fail identically: BaTiO3 4, KNbO3 4, CsSnBr3 5. KNbO3 is 408 K below its T_c and is called stable at 300 K by the screen for four models (manuscript.md:354-355). CsSnBr3 is only 8 K above its 292 K T_c (configs/curated_systems.yaml:133). The 'transfer failure' therefore largely measures the single-mode screen's own T* error (approximation A1), shared across models, rather than MLIP-specific PES error. That is a design confound in the H2 headline which the manuscript does not state.
    + The public repo README.md:21 (linked from the Data Availability section) still reads 'H2 — Harmonic error is *not predictive* of finite-T error (the core blind spot)', and its Layer-2 description names TDEP, which was discarded. It is labelled as a hypothesis, so this is low risk, but it should be updated before the new Zenodo version (todo K6).
    + The mlip_dynstab/analysis.py:303-305 docstring for h2_paired_summary still says 'harmonic accuracy is a WEAK (not negative) predictor ... the harmonic leaders are not the finite-T leaders'. This deposited code contradicts the revised claims (low severity).
    + The letter :532 says verify_claims runs 29 assertions. Running scripts/verify_claims.py (read-only, exit 0) prints 32 PASS lines; tasks/todo.md L1 says 32/32.
  REFUTED:
    - The checker's R3.1 letter-status note says the dc11ed7 date was 'verified to predate the reports', treating the letter's attribution as sound. The date is correct, but dc11ed7 did not remove 'confirm H2 in a strong form'; 0fbea7d (2026-08-16) did. The letter's claim is wrong on the commit, not merely unverified.
    - The checker's R3.1 gap calls manuscript.md:431/:438 and supplementary.md:80 ('the reordering', 'the two models carrying the reordering') residue of the withdrawn CHGNet crossover. That is only partly right. On the matched set, a CHGNet-vs-MatterSim point-estimate reordering still exists: harmonic 13/15 vs 15/15, finite-T 26/30 vs 25/30. What :383-385 withdrew was 'CHGNet is worst harmonically'. The problem is a framing inconsistency (leaning on a reordering the paper says is unresolved), not a withdrawn number resurfacing. The severity is lower than implied.
    - The checker's RESIDUE-title and other_issues cite tasks/todo.md:28-29 as naming the intermediate title as current. The actual location is tasks/todo.md:41-43 (item D2). Lines 28-29 are the number-change table.
    - The checker rated R3.3-Spearman 'complete'. Under the task's standard, which requires the letter to be accurate, it is partial: the letter says Spearman 'is now reported as descriptive only' (:445), but it appears nowhere in the manuscript or ESI.
    - The checker rated R3.3-McNemar-failure-to-reject 'complete'. It is partial because manuscript.md:415 and letter :414 call the non-significant 100 K result 'absent', the same error class the referee flagged.

##### GROUP R2
- R2.1: checker=partial FINAL=partial agree=True
    * No citation in the fine-tuning paragraph (manuscript.md:728-740), and no fine-tuning reference in the list (857-893), even though todo.md:156 G1 [x] says 'Cite the relevant work'. Any candidate such as Deng et al., 'Systematic softening in universal MLIPs', npj Comput. Mater. 2025, must be verified before it is cited.
    * The paragraph does not sort which findings fine-tuning could change (the per-model screen accuracy, findings i/ii) from those it could not (the SSCHA truncation false-stabilisation, controlled with a fixed engine in ESI S2.2 Table S2: harmonic -5.635 against Hessian +2.872 THz; the SrTiO3 'screen every mode' result). One or two sentences would turn the scope disclaimer into a substantive answer.
    * The as-shipped scope is not stated in the abstract (manuscript.md:8-42) or the Conclusions (786-806), and there is no ESI mention.
    * Letter :302 says 'subsection' but the manuscript has an unnumbered run-in paragraph (manuscript.md:728).
    * Letter :295-296 presents a paraphrase as an italic verbatim quote ('no mention of how fine-tuning' against R2's 'does not mention how fine-tuning') and moves 'technically sound' to the front with an inserted '[but]'.
    * paper/manuscript.docx (2026-09-10) does not contain the paragraph and still has the old title, so the DOCX must be rebuilt.
- R2.2a-reframing-of-§3.4: checker=answered_by_argument FINAL=answered_by_argument agree=True
    * R2 read the as-reviewed guardrail (manuscript-as-reviewed.md:20-21, 321-339) and still raised the point. 'Section 3.4 already runs this experiment' (letter :314) re-offers material R2 judged insufficient.
    * 'propagated form of exactly the force-level disagreement' (manuscript.md:752-754; letter :324-325) is inaccurate: it is the std of an energy-derived free-energy curvature (analysis.py:352; finite_t.py:604-617, 813-817). Describe it as a PES-level proxy.
    * The concession at manuscript.md:756-757 means R2's committee metric was not tested.
    * NEW: the guardrail is not reported without ORB-v2 (Table S9, supplementary.md:575-588, omits it despite R3.5). Recomputed without ORB-v2: AUC 0.628, cluster-bootstrap CI [0.439, 0.835] spans 0.5, split 4/8 against unanimous 8/52, with 3 tied units. The conclusion at manuscript.md:800-803 ('cheap and effective guardrail') and the letter's 'discrete signal works' (:321) overstate its robustness.
    * NEW: the letter (:326 'It carries no usable information'; :433 'uninformative') contradicts the manuscript's 'failure to resolve ... moderately useful not excluded' (manuscript.md:644-648, 754-755). §4 at 757-758 ('failed') is also stronger than §3.4.
    * NEW: the raw-std metric is dominated by far-from-boundary spreads on unanimous-correct units (c_diamond 4.55 THz x4 temperatures; hfo2 100 K 12.2 THz), so 'the continuous version of this idea failed' (757-758) generalises from one naive metric. An exploratory boundary-proximity score gives AUC about 0.76 (post hoc; do not report as a finding).
    * NEW: H3's pre-registered comparator, a single model's self-reported signal (manuscript.md:111-112), is never tested, and it is the comparison closest to R2's request.
    * The enrichment is partly built in (unanimous units can be wrong only if all five models are), and the clustered permutation null of no association does not account for this. It is not disclosed.
    * NEW (minor): manuscript.md:640-642 'overstated the significance, by about a factor of four in p'. stats_hardening.json gives naive 0.0003 against clustered 0.0035, about 12 times.
    * The todo note 'R2 appears to have missed this' (tasks/todo.md:159-160) is contradicted by the as-reviewed abstract, and the same note keeps the withdrawn 'below chance' wording.
- R2.2b-force-level-ensemble-uncertainty: checker=not_done FINAL=not_done agree=True
    * No force-spread computation or output exists anywhere in scripts/, mlip_dynstab/, results/ or the ledger. It needs a five-model GPU run (runbook:6-9).
    * The cache stores no displacement vector (only q/band/dim/Qs/dE), so each model's force constants must be regenerated to rebuild the displaced structures. Runbook:28-30's 'regenerable from the stored mode pattern' understates this.
    * Design: E(Q) configurations are T-independent (cli.py:125-126), so the score must be made T-dependent (e.g. sample at sigma(T)) or declared per-system. runbook:32's 'per-(system, model, T)' cross-model spread is incoherent as written.
    * The per-model E(Q) paths differ, so a shared configuration set must be defined and stated.
    * Even when done, it is cross-architecture, not the within-architecture committee R2 likely means (manuscript.md:756-757). Add a committee-like variant or argue explicitly that released checkpoints ship no committee.
    * ORB-v2 (float32, non-conservative) must be reported split (runbook:36; R3.5). Given that the existing guardrail already weakens to AUC 0.628 without ORB-v2, the split is essential.
    * Letter :331-332 says 'in progress' although nothing has run. It must be replaced by a result or by an explicit limitation before sending.
- R2-overall: checker=partial FINAL=partial agree=True
    * R2.2 has no new measurement, and the force-level item is [PENDING] with inaccurate 'in progress' wording (letter :331-332, :542).
    * The R2.2 argument overstates the force link (manuscript.md:752-754) and the robustness of the guardrail (no ORB-v2 split; recomputed AUC 0.628 with CI spanning 0.5).
    * The letter contradicts the manuscript on the frequency spread (letter :326/:433 against manuscript.md:644-648).
    * R2.1 is uncited and its scope is missing from the abstract and Conclusions.
    * The DOCX (paper/manuscript.docx, 2026-09-10) contains none of the R2 text and must be rebuilt, along with the supplementary DOCX, before anything can be submitted.
    * Given the escalated RSC reminder, Frank has to decide between running the force-level compute (runbook item 1, ~$20-60 budget already agreed) and replacing the [PENDING] line with an explicit limitation. The second is a weak answer for this referee.
  NEW ISSUES:
    + The guardrail is not robust to excluding ORB-v2 and is not reported that way. Recomputed read-only with analysis.h3_ensemble_guardrail plus stats.cluster_permutation_auc/cluster_bootstrap_auc on df[model!='orb_v2']: vote-split AUC 0.628, clustered p about 0.046, cluster-bootstrap 95% CI [0.439, 0.835] (spans 0.5); split 4/8 against unanimous 8/52; 3 tied 2-2 units broken to 'stable' by the >=0.5 rule (analysis.py:348). Table S9 (supplementary.md:575-588) omits the guardrail, although R3.5 asks for every rate with and without ORB-v2. This undercuts the 'discrete signal works' answer to R2 (manuscript.md:750-752; letter :321-323) and the Conclusions' 'cheap and effective guardrail' (manuscript.md:800-803).
    + response_to_referees.md:326 ('It carries no usable information') and :433 ('the honest statement is that it is uninformative') contradict manuscript.md:644-648 ('We state that as a failure to resolve rather than as an absence ... these data do not exclude a moderately useful one either') and manuscript.md:754-755. Inside the manuscript, §4 (757-758: 'the continuous version of this idea failed on this set') is also stronger than §3.4.
    + The continuous metric tested is the raw np.nanstd of a signed frequency (analysis.py:352), which is dominated by far-from-boundary magnitudes on unanimous-correct units: c_diamond 4.55 THz at all four temperatures, hfo2_cubic 100 K 12.2 THz, zro2 100 K 2.16 THz. It therefore does not test 'the continuous version of this idea' (manuscript.md:757-758) in general. In an exploratory post-hoc check, -|median frequency| gives AUC 0.759 (clustered p about 0.036) and std/|mean| gives 0.601. Flag as exploratory only.
    + H3 as pre-registered (manuscript.md:111-112) compares ensemble disagreement with 'any single model's self-reported energetics', and that comparison is never run in §3.4 (623-659) or the ESI. It is the part of the paper closest to R2's request, so a reject-leaning referee focused on uncertainty is the one most likely to notice.
    + manuscript.md:640-642 says the naive test overstated significance 'by about a factor of four in p'. stats_hardening.json has p_perm_naive_unit_level 0.0003 against p_perm_clustered 0.0035, about 12 times.
    + The intro (manuscript.md:84-85) says 'We return to that comparison in §3.4 and §4', but §3.4 (623-659) never mentions forces, committees or model-reported uncertainty. The comparison appears only in §4, which contradicts the letter's 'the revision makes the connection explicit' (:315) if that is read as referring to §3.4.
    + paper/manuscript.docx (2026-09-10 08:05) has the OLD title ('Finite-temperature dynamic stability separates...') and the withdrawn wording 'carries no usable signal (AUC 0.36, below chance on this set)'. The same stale state applies to paper/supplementary.docx (2026-09-10 08:05).
  REFUTED:
    - Partly refuted: the checker says manuscript.md:833's '220 cached E(Q) maps' wrongly counts the 20 superseded v2 maps. There are 220 softmode map files in results/cache (200 v3 + 20 v2), so the sentence is literally true and only imprecise, since the current generation uses 200. The statement that is actually wrong is compute-runbook-rsc-revision.md:27-28, which says softmode_v3* 'holds 222 cached E(Q) maps'. There are 200 v3 files, and 222 counts the 20 v2 files plus 2 MD extxyz trajectories.
    - Softened: the checker says the fine-tuning sentence (manuscript.md:734-735) 'contradicts the paper's own evidence' for the SSCHA layer. The sentence is about the models' finite-temperature performance, not about the SSCHA method, so it is an over-broad, uncited concession rather than a contradiction. The suggested fix (state which findings fine-tuning cannot change) is still worth making.
    - Qualified: the checker says the frequency spread 'is not from forces'. The E(Q) values are energy-only (finite_t.py:604-617), but the mode patterns along which E(Q) is sampled come from each model's harmonic force constants, which are computed from forces. 'A PES-level proxy partly derived from forces' is more accurate than 'not from forces'. The manuscript's 'exactly the force-level disagreement' is still an overstatement.

##### GROUP R3.4-R3.5-R3.6
- R3.4-content: checker=complete FINAL=partial agree=False
    * Letter R3.4 (response_to_referees.md:475-482): change 'S1–S12' to 'S1–S13'; remove 'numbered last because it was added last' and 'kept the number stable rather than renumbering the tables the referee has already been asked to check'; add an explicit mapping (your S1 -> Table S4, S2 -> S5, S3 -> S6, S4 -> S8).
    * Table S12's caption sits after its table (supplementary.md:243-251); every other table has its caption first. The out-of-order placement itself is already acknowledged in the letter.
    * Table S6 caption (supplementary.md:396): 'systematically under-estimates T* for the entropy-stabilised bcc metals' is wrong for 5 of the 15 bcc-metal rows, which are 'never stable', i.e. over-estimates (MatterSim Ti/Zr/Hf at 442/487/492; ORB-v2 Ti at 488). Soften it.
    * Per-system harmonic minimum frequencies appear nowhere in the paper or ESI (Fig. 2 is the screen's 100 K curvature, not harmonic; see figure-files). Optional: a system x model harmonic min-frequency table, so the referee's 'per-system numbers underlying every headline rate' covers the harmonic layer too.
- R3.4-esi-docx: checker=not_done FINAL=not_done agree=True
    * Rebuild supplementary.docx from the current supplementary.md. Then confirm Tables S1-S13 render as real Word tables with captions, and check the file for any leftover function-name-only lines.
- R3.5-coverage: checker=partial FINAL=partial agree=True
    * Add ex-ORB rows for the H2 ladder at all four temperatures, and rewrite manuscript.md:415-416 (ex-ORB at 900 K the unit-level result is 10v2, p=0.039).
    * Add the four-model guardrail (split/unanimous 4/8 vs 8/52, vote AUC 0.628, clustered p≈0.047, CI [0.438, 0.839] spanning 0.5; freq-std AUC 0.299), state the 2-2 tie rule, and disclose the weakening in §3.4.
    * Add ex-ORB family recalls (halide 20/20, fluorite 16/16, SrTiO3 3/4, controls 0/96) and ex-ORB tolerance-sweep counts to Table S9 or the text.
    * Tabulate SSCHA blow-ups and failed units per model (ORB-v2 3/8, MatterSim 3, CHGNet 1, SevenNet-0 1).
    * Fix the Table S9 caption (supplementary.md:575). '≈ −35 THz Ti/Hf outliers' is stale: ORB-v2's bcc screen minimum is −6.20 THz (Ti 100 K) or −8.89 THz (Ti 900 K), and Hf is −0.09. 'Most of the SSCHA blow-ups' is also wrong: ORB-v2 has 3 of 8.
    * Fix manuscript.md:307-308 ('Table S9 reports every rate it could plausibly drive'), or make it true.
    * Put the ex-ORB ρ (−0.003) into §3.3 at manuscript.md:513, as the letter claims was done.
    * Letter R3.5: remove the §3.5 claim (488), stop quoting the unit-level 'p = 2 × 10⁻⁵' (503), and say that the referee's premises have moved (MgO −2.8 → −1.07 THz; FS rate 0.562 → 0.312; 'six blow-ups concentrated in its runs' → 3 of 8, spread over four models). No [PENDING] tag is present although the item is incomplete.
- R3.6-manuscript-order: checker=complete FINAL=complete agree=True
- R3.6-docx: checker=not_done FINAL=not_done agree=True
    * Rebuild the clean manuscript.docx and a tracked/highlighted version from the current manuscript.md; verify caption order 1-6 and the revised title in the built files.
- R3.6-figure-files: checker=partial FINAL=partial agree=True
    * Fix the stale Fig. 4 annotation (make_figures.py:102-110): compute the value (−6.2 THz, Ti 100 K) or drop it, then regenerate.
    * Fig. 2: either plot harmonic per-system minimum frequencies, or re-caption it as the screen's 100 K curvature and move its first citation out of the harmonic §3.1 argument. Fix the inverted colorbar label ('blue<0 unstable').
    * Flatten the TIFFs to RGB, and export them as separately numbered files in placement order: Fig1=tolerance_sweep, Fig2=softmode_heat, Fig3=sscha_bcc, Fig4=method_agreement, Fig5=displacive_recall, Fig6=ensemble_guardrail.
    * The docx-embedded images are the 300-dpi PNGs. Embed the 600-dpi versions, or upload the numbered TIFFs as well.
    * Cosmetic: replace the code-name legends in Figs. 2-4 with the paper's model names. The Fig. 3 caption values (0.06/0.30/1.2/1.6 THz, manuscript.md:563-564) are Hf-panel values and should say so.
  NEW ISSUES:
    + The bcc cross-validation statistic is mislabelled as call agreement. manuscript.md:511-513 says the methods 'agree on the stability call in 35 of 45 paired bcc units (sign agreement 0.78; 30 of 36, 0.83, excluding ORB-v2)'. analysis.method_agreement (analysis.py:165-181) actually scores the sign of min_eff_freq (>= 0), i.e. the symmetric-point curvature, which §2.4 (manuscript.md:210-223) says differs from the variational call. Recomputed on the variational calls (pred_stable), agreement is 31/45 = 0.689 with all models and 25/36 = 0.694 without ORB-v2. The one ORB split the referee called 'already performed' then shows no improvement from excluding ORB-v2. Ten units differ: CHGNet Ti 100/300 K and MatterSim Ti/Zr/Hf 600 K have positive curvature but are called unstable; ORB-v2 Ti 300/600 K likewise; ORB-v2 Hf has −0.09 THz curvature but is called stable. The same mislabel appears in supplementary.md:645-647 (§S4, 'agree on the stability **call**'), supplementary.md:216-218 (§S1.3, 'bounds the coupling's effect on the *call*') and the letter §A ('the reported statistic is the stability-call agreement').
    + Fig. 2 (results/figures/fig_softmode_heat.png; make_figures.py fig_softmode_heat) plots the screen's 100 K effective-frequency curvature, not harmonic minimum frequencies. It is cited in the harmonic §3.1 (manuscript.md:318-319, caption 326-328) as showing harmonic softening. Its colorbar label 'THz (blue<0 unstable)' is inverted: blue marks positive values.
    + manuscript.md:307-308 says 'Table S9 reports every rate it could plausibly drive both with and without it'. This is false in the manuscript itself, not just in the letter: the H2 ladder, H3 guardrail, family recalls, controls, tolerance sweep and SSCHA blow-up attribution are absent from S9.
    + The Conclusions (manuscript.md:795-796) say the screen-vs-SSCHA contrast 'survives removing' ORB-v2 on the combined set, 'though on the ferroelectric oxides alone it does not'. That verdict rests on the unit-level McNemar (0.092 vs 0.013), which Table S10 (supplementary.md:588) says must not be quoted. The clustered p is identical with and without ORB-v2 (0.125 combined, 0.5 FE oxides; supplementary.md:592-597).
    + manuscript.md:521-523 says 'across the full FE grid seven units die at cellconstructor symmetry or ensemble assertions'. One of the seven is CsSnI3/SevenNet-0, a halide (supplementary.md:319-321), so it is not part of the FE grid.
    + Letter R3.4 (response_to_referees.md:478-479) says the table numbers were kept stable for 'the tables the referee has already been asked to check'. In fact exactly those four were renumbered (S1->S4, S2->S5, S3->S6, S4->S8), and no mapping is given.
    + Ex-ORB, the H2 system-clustered paired test (stats.cluster_exact_paired over 15 systems) gives p = 0.156 at 300 K, 0.055 at 600 K and 0.25 at 900 K. With all models it gives 0.152, 0.066 and 0.414. This confirms the checker's HIGH other_issue that the title-level 17v4 p = 0.007 does not survive clustering, with or without ORB-v2.
  REFUTED:
    - Checker R3.4 gap 'Section/table collision is disambiguated, not removed ... say so rather than fixed': the letter already says this. response_to_referees.md:475-476 describes the fix as 'a note at the head of the ESI disambiguating them from the sections', which matches supplementary.md:7-9. No letter change is needed on that point.
    - Checker R3.4 gap 'Table S12 is out of sequence': the letter already acknowledges the out-of-order placement (response_to_referees.md:476-479). What remains wrong is its rationale ('numbered last because it was added last'; S13 was added later) and the caption-after-table placement.
    - Checker R3.4 gap 'Per-system harmonic minimum frequencies behind Table S4 appear only graphically (Fig. 2)': this is wrong. Fig. 2 plots the soft-mode screen's 100 K curvature (make_figures.py fig_softmode_heat, method=='softmode' at min T), so the harmonic per-system minima appear nowhere, not even graphically. The gap is larger than stated.
    - Checker other_issue on Fig. 4 ('the MatterSim points sit at -1.4 to -0.6 THz on the screen axis'): this holds only for 6 of the 9 MatterSim points (100 and 300 K). At 600 K the MatterSim screen values are +0.28 to +0.35 THz. The conclusion that the 'tracks SSCHA' caption (manuscript.md:566-568) overstates the relationship still stands (ρ = 0.113; the MatterSim cluster sits far off y=x).
    - Checker's ex-ORB guardrail interval [0.439, 0.828] came from 2,000 resamples. With the repo's default of 10,000 (seed 0) I get a clustered p of 0.047 and a CI of [0.438, 0.839]. The conclusion (CI spans 0.5) is unchanged; only the numbers to report differ slightly.

##### GROUP PACKAGE
- PKG-a: checker=partial FINAL=partial agree=True
    * DRAFT banner at response_to_referees.md:10-11 must be removed
    * [PENDING] R1.4 four-seed extension (:240-242): run it, or rewrite as an explicit decline consistent with supplementary.md:603
    * [PENDING] R2.2 force-level ensemble spread (:331-332): run it or delete the promise; R2 is the reject-leaning referee
    * Internal checklist :537-548 (3 open science rows plus 5 package rows) must be stripped from the letter
    * Stale ':532 29 assertions' should be 32 (verified by running verify_claims.py today)
    * Stale ':475 Tables are now S1–S12' and the list at :481-482 omit S13, which the letter cites at :247
    * Letter claims at :12 and :516-517 are true only of manuscript.md, not of the .docx that would be uploaded
    * No .docx/.pdf export of the letter exists for the portal
- PKG-b: checker=not_done FINAL=not_done agree=True
    * No marked-changes manuscript (.docx or .pdf) exists anywhere in paper/; RSC lists it as required
    * No marked-changes ESI, recommended given R3.4 and R1.5 changes live in the ESI
- PKG-c: checker=not_done FINAL=not_done agree=True
    * Rebuild paper/manuscript.docx from manuscript.md; the current file lacks the new title, the fine-tuning scope, Wilson intervals, refs 27-37 and the Fig. 5/6 order fix
    * Rebuild paper/supplementary.docx; the current file lacks §S1.3/§S1.4 and rendered Tables S4-S13 (the R3.4 and R1.5 deliverables)
    * Quarantine the stale paper/manuscript.pdf (old title) so it cannot be uploaded
- PKG-d: checker=partial FINAL=partial agree=True
    * Five of the six embedded figures print at about 300 dpi; embed the 600-dpi versions or upload the TIFFs separately
    * Separate TIFFs are not numbered by figure (Fig1=tolerance_sweep, Fig2=softmode_heat, Fig3=sscha_bcc, Fig4=method_agreement, Fig5=displacive_recall, Fig6=ensemble_guardrail)
    * TIFFs are RGBA; flatten to RGB for production safety
    * The Fig. 4 caption (manuscript.md:566-568) says 'tracks SSCHA' while the figure's own title shows ρ=0.113
- PKG-e: checker=partial FINAL=partial agree=True
    * Regenerate the TOC graphic with the revised recall (0.53 vs 0.19) and remove the falsified 'harmonic leaders ≠ finite-T leaders' / 'SevenNet-0 finite-T not #1' panel
    * Rewrite the blurb (≤250 characters) to the revised claims; 'disagree sharply at finite temperature' is contradicted by the revision
    * Flatten the TIFF alpha channel
- PKG-f: checker=partial FINAL=partial agree=True
    * Optional: restate the section in CRediT role terms (conceptualization, methodology, software, formal analysis, investigation, data curation, visualization, writing – original draft, writing – review & editing)
    * Remove the stale 'CRediT' row from the letter checklist (:545)
- PKG-g: checker=blocked_on_frank FINAL=blocked_on_frank agree=True
    * Frank's decision: keep Gmail or switch the corresponding email to cai485@purdue.edu in case the Purdue read-and-publish agreement requires it (terms unverified in the repo; confirm with Purdue Libraries or RSC)
    * Link ORCID in the RSC portal (portal action)
    * Optional: add the department and postcode to the affiliation
- PKG-h: checker=not_done FINAL=not_done agree=True
    * Default branch main (793f4dc) is the pre-audit July code; merge rsc-figure-fixes into main and push, or pin the DAS to a tag or commit
    * Letter-cited artefacts scripts/reviewed_version_delta.py and paper/submissions/rsc-advances-2026-07/ are on neither main nor any Zenodo version
    * DAS (manuscript.md:808-835) does not name the revision's new code and outputs (mlip_dynstab/stats.py, scripts/stats_hardening.py, results/stats_hardening.json, scripts/build_esi_tables.py, scripts/run_disp_sweep.py)
- PKG-i: checker=not_done FINAL=not_done agree=True
    * Mint a new Zenodo version from the final commit; v2.0.0 lacks the 2026-09-11 ledger rows (Table S13), stats_hardening outputs and build_esi_tables.py, so the DAS claim that the numbers regenerate from the archived deposit is false for the revision
    * Update the .zenodo.json:4 title/description before minting
    * Sequence the mint after the main merge (PKG-h) and after any PENDING runs
- PKG-j: checker=not_done FINAL=not_done agree=True
    * No revision cover note to Dr Rhyman / RSC Advances / RA-ART-07-2026-006452 exists; the current cover_letter.md/.docx is the Digital Discovery original with a stale title and stale AUC 0.75 and must not be uploaded
- PKG-k: checker=blocked_on_frank FINAL=blocked_on_frank agree=True
    * Frank must choose opt-in or opt-out in the portal; if opting in, re-read the letter's tone on R1.3 (:198-217) as a public document
- PKG-l: checker=partial FINAL=partial agree=True
    * No AI-use disclosure anywhere; Frank decides the wording and placement (Acknowledgements or Methods) per current RSC policy, which I could not fetch directly (403)
  NEW ISSUES:
    + ESI §S2.4 at supplementary.md:345-350 is unchanged from the as-reviewed ESI (supplementary-as-reviewed.md:107-109). It still says 'the §3.3 false-stable is independent of this because it occurs for the always-present Γ mode of BaTiO₃', which is the BaTiO₃-only argument R3.2 said 'establishes this for one system only'. It does not mention the fluorite within-cell control or the narrowing ('no convergence claim' for R/X-point systems) that manuscript §3.5 (manuscript.md:682-694) and the letter (:365-378) now make. R3 will see the old argument in the revised ESI.
    + ESI §S4 threats-to-validity at supplementary.md:642-644 still says 'Outcome: stability calls are robust to the tolerance for all but the borderline KTaO₃', copied verbatim from the as-reviewed ESI (:127). It contradicts the revised manuscript: :479-481 says 'CHGNet's harmonic accuracy is tolerance-dependent' (CeO₂/NaCl flip at tol 0.30), and :314-316 says a strict tolerance 'floods false-unstables (29 calls)'. It also contradicts Table S13's text (supplementary.md:634), which says the amplitude flips are 'the same marginal units §3.1 identifies from the tolerance sweep'.
    + Abstract length: manuscript.md abstract (:10-42) is 461 words, up from 277 in the as-reviewed version. An rsc.org-restricted search summary gives RSC Advances' abstract guidance as 'around 50 to 250 words' (not verbatim-verified; the RSC page returns 403 to fetch). A third-party site's 200-word figure is unverified. Either way the revision moved further over the guidance, not closer.
    + Letter :240-242 says '§S2.4 and Table S11 state that the seed study currently covers bcc-Zr only'. §S2.4 (supplementary.md:341-344) only describes the bcc-Zr study and never states the limitation. The explicit statement is at :603, in 'work in progress and is not reported here' wording that should not ship in a final ESI.
    + The letter checklist has a third open science row the checker omitted: :543 'Displacement-amplitude sweep: CHGNet done (Table S13); four models remain'. The text already declares it partial (manuscript.md:445-447; supplementary.md:623 'Partial: CHGNet only'), so it needs only removal from the letter, not new compute.
    + tasks/todo.md:216 'L2 Independent read-through: every number in the manuscript and ESI traced to the ledger' is unchecked. The letter at :12 nonetheless asserts 'Everything untagged is done'.
    + tasks/todo.md:41-43 (D2, marked [x]) still records the 'Current title' as the intermediate '...separates foundation MLIPs that harmonic benchmarks rank equally', not the 'Neither ... certifies' title now at manuscript.md:1. The repo's own plan does not record adoption of the new title, which is consistent with it still awaiting Frank's sign-off.
  REFUTED:
    - PKG-h response_letter_status says the files the letter cites 'None ... is in any Zenodo version'. That is partly wrong. The Zenodo v2.0.0 zip (mlip-dynamic-stability-v2.0.0-da9c077.zip, downloaded read-only) contains tasks/audit-2026-08-16.md (cited at letter :32-33), scripts/verify_claims.py and envs/lock-*-2026-08-16.txt. Only scripts/reviewed_version_delta.py (:58) and paper/submissions/rsc-advances-2026-07/ (:60-61) are absent from every archive. The PKG-h verdict stands.
    - PKG-d says the embedded PNGs 'are 300 dpi'. At their displayed size in the docx, five of six print at 299-334 dpi, but fig_sscha_bcc (rId25, 3600 px across 14.82 cm) prints at 617 dpi. The embedded route fails for 5 of 6 figures, not all. The verdict stands.
    - PKG-j says the revision 'now qualifies' the cover letter's 'cheap screen is the more reliable indicator' framing. The revised abstract keeps 'The cheap soft-mode screen is therefore the more reliable finite-temperature indicator' (manuscript.md:40-42). Only the cover letter's 'inverting the usual cost/accuracy intuition' goes beyond the manuscript. The verdict stands.
    - Minor count errors in PKG-c that do not change the verdict: 'fine-tun' occurs 5 times in manuscript.md, not 4, and 'Table S' occurs 19 times in supplementary.md, not more than 20.
    - All eight of the checker's other_issues checked out: Fig. 4 caption at :566-568 (verified, and the figure title itself shows ρ=0.113); ESI :96 vs Table S13; ESI :3 'Companion to manuscript.md' and :603 'work in progress'; stale standalone DAS title at data_availability_statement.md:3; new title not signed off (consistent with todo.md:41-43); 461-word abstract; 32 vs 29 assertions (32 PASS observed by running the script); docx Fig. 6 before Fig. 5 (paras 159/162). None refuted.

##### GROUP RESIDUE
- RESIDUE-1: checker=complete FINAL=partial agree=False
    * manuscript.md:473-475: remove the 'weakest finite-temperature screener' ranking claim, or restate it as a one-unit difference that is not separated.
    * manuscript.md:442 and supplementary.md:84: §3.1 does not identify ORB-v2 as the weakest (it is tied at 0.895 and above CHGNet at 0.789). Reword to 'the model with the largest replicate noise'.
- RESIDUE-2: checker=not_done FINAL=not_done agree=True
    * Rewrite the letter for RSC Advances revision RA-ART-07-2026-006452 with the Sep-11 title and current numbers, and rebuild the .docx, or do not upload it.
- RESIDUE-3: checker=not_done FINAL=not_done agree=True
    * Update the title in data_availability_statement.md:3 and its .docx to the Sep-11 title, or drop the file.
- RESIDUE-4: checker=not_done FINAL=not_done agree=True
    * Rewrite toc_blurb.txt (at most 250 characters) around the surviving claims.
    * Regenerate the TOC graphic: drop the 'harmonic leaders ≠ finite-T leaders' panel and read the recalls (0.53 and 0.19) from the ledger instead of hardcoding them.
- RESIDUE-5: checker=not_done FINAL=not_done agree=True
    * Rebuild clean manuscript.docx and supplementary.docx from the current .md files.
    * Build a tracked-changes or highlighted version against manuscript-as-reviewed.md, as RSC requires (rsc_referee_reports.md:12).
    * Re-scan the rebuilt docx for the stale phrases, including 'ρ=0.78'.
    * Delete or rebuild the stale manuscript.pdf.
- RESIDUE-6: checker=not_done FINAL=not_done agree=True
    * Change the Fig. 4 caption (manuscript.md:566-568) to call-agreement framing: 35/45 = 0.78, magnitudes not rank-correlated (ρ = 0.11).
    * Replace the hardcoded '~ -35 THz' annotation (make_figures.py:110) with the data value (−6.2 THz, Ti only) and regenerate the .png and .tiff.
- RESIDUE-7: checker=not_done FINAL=not_done agree=True
    * Correct the Table S9 caption in build_esi_tables.py:199 (MgO false-unstable, bcc softmode to −8.9 THz on Ti, 3 of 8 blow-ups, 4 of 7 failed units), rerun the builder and keep --check green.
    * In the R3.5 answer, tell Referee 3 that the 0.562 rate, MgO −2.8 THz, the −35 THz outliers and the ORB-concentrated blow-ups they cited no longer hold after re-measurement.
- RESIDUE-8: checker=not_done FINAL=not_done agree=True
    * Replace '(≈tens of hours per unit)' at supplementary.md:305 with the observed bound: did not finish in 18 min, against ~200 s for v4=False.
- RESIDUE-9: checker=not_done FINAL=not_done agree=True
    * Rewrite the §S4 tolerance outcome (supplementary.md:642-644): calls are stable over roughly 0.05–0.1 THz; CHGNet's CeO₂ and NaCl flip at ≈0.25 THz; tol = 0 is degenerate; point to Table S13 for the amplitude axis.
- RESIDUE-10: checker=not_done FINAL=not_done agree=True
    * Change supplementary.md:95-97 to: measured for CHGNet (Table S13); the other four models are outstanding.
- RESIDUE-11: checker=not_done FINAL=not_done agree=True
    * Reword manuscript.md:431 and 438-442 and supplementary.md:80 so they no longer refer to 'the reordering'.
- RESIDUE-12: checker=not_done FINAL=not_done agree=True
    * Update §S2.4 (supplementary.md:345-350) to mirror §3.5: no convergence test and no claim for the R-point and X-point systems, plus the fluorite within-cell control (−3.8 to −10.6 THz harmonic vs +1.9 to +3.3 THz SSCHA).
- RESIDUE-13: checker=not_done FINAL=not_done agree=True
    * Rewrite Conclusions manuscript.md:795-796 in effect-size terms: the clustered p is 0.125 with or without ORB-v2, 26 vs 3 excluding it, and 10 vs 3 on the FE oxides alone.
    * Rewrite letter R3.5 (response_to_referees.md:496-512) to drop the significant/not-significant language and 'p = 2 × 10⁻⁵', and cite the clustered p.
    * Fix the letter claim at line 511 that '§3.3 now states all of this explicitly', or add the content to §3.3.
- RESIDUE-14: checker=not_done FINAL=not_done agree=True
    * Change response_to_referees.md:354 to the matched harmonic spread: 0.133 (0.867 to 1.000).
- RESIDUE-15: checker=not_done FINAL=not_done agree=True
    * Either add the four across-model ρ values and their floors to the ESI, marked descriptive, or change letter line 445 to say they were removed from the manuscript and are given only here and in stats_hardening.json.
- RESIDUE-16: checker=not_done FINAL=not_done agree=True
    * Change response_to_referees.md:52-53 to 'top-equal in both layers; the per-model accuracies are not separated by this design'.
- RESIDUE-17: checker=not_done FINAL=not_done agree=True
    * Line 475: S1–S13. Line 477: drop 'numbered last'.
    * Line 532: 32 assertions.
    * Line 391: say two-sided Wilson, and give the 0/228 (57 units) interval or drop it.
    * Lines 346-347: commit 0fbea7d, August 2026.
    * Line 545: reword to 'convert the Author contributions section to CRediT roles'.
    * Lines 481-482: add S13 (displacement sweep).
- RESIDUE-18: checker=not_done FINAL=not_done agree=True
    * Rephrase manuscript.md:381-386 and 470-473 so they stand alone.
    * Remove the 'previous version' and 'this revision' wording at supplementary.md:357 and 501.
    * Fix or remove the 'earlier, unpinned measurement' sentences at manuscript.md:523-525 and supplementary.md:322-324.
- RESIDUE-19: checker=partial FINAL=partial agree=True
    * Resolve the letter mismatches in RESIDUE-13 to 17, plus the line 511 claim about §3.3 and the S13 omission at lines 481-482.
- RESIDUE-20: checker=complete FINAL=partial agree=False
    * manuscript.md:306: change bcc-Zr −0.43 THz to −0.47 THz (canonical v2), and consider adding an assertion to verify_claims.py.
  NEW ISSUES:
    + manuscript.md:306: 'bcc-Zr (−0.43 THz)' is the superseded v1/July value (manuscript-as-reviewed.md:169; ledger v1 row −0.4334). The canonical v2 value is −0.4706 THz, and the adjacent Hf value (−0.19) is already v2.
    + manuscript.md:473-475: 'ORB-v2 remains the weakest finite-temperature screener (0.800)' is residue of the withdrawn July ranking claim (manuscript-as-reviewed.md:241-243). It contradicts manuscript.md:392-395 and 803-806 (not separated, no ranking claim); 24/30 vs 25/30 is one unit with overlapping intervals.
    + manuscript.md:442 and supplementary.md:84 say §3.1 identifies ORB-v2 as 'the weakest' model. The §3.1 table (manuscript.md:291-295) has ORB-v2 at 17/19, tied with MACE-MP-0 and above CHGNet at 15/19. It is 'uniquely worst' only at tol = 0.30 (manuscript.md:481).
    + supplementary.md:82-83 inverts the hf_bcc flip. It says '(the same call either way at the production tolerance, but a flip at tol = 0)'. In fact the v1 value −0.0891 THz is STABLE at the 0.1 THz production tolerance (ledger dynamically_stable = True) and the v2 value −0.1937 is unstable. estimator_noise.py:90 computes flips from those stored production-tolerance calls, and at tol = 0 both values are unstable (no flip). So the replicate flips a scored ORB-v2 harmonic call at production tolerance.
    + manuscript.md:523-525 and supplementary.md:322-324 say that in the earlier unpinned measurement 'these same units produced silent numerical blow-ups to −2×10⁶ THz'. The ledger v1 rows for the 7 failed units read −3.6, −363, −3653, −15.9, −4.8, −0.30 and −8.6 THz: only 2 exceed 50 THz and none approaches −2×10⁶. The −2.03×10⁶ value (ft_sscha_min_freq_thz_v1) belongs to SrTiO3/ORB-v2/600 K, which is not a failed unit. This is July residue from 'six blow-ups ... down to −2×10⁶'.
    + supplementary.md:321-322 ('the eight numerical blow-ups span float32 (ORB-v2) and float64 (SrTiO₃ high-T rows)') and manuscript.md:527-528 ('the failures span float32 and float64 models'). Only 1 of the 5 non-ORB blow-ups is a SrTiO3 row (CHGNet 900 K); the others are MatterSim PbTiO3 300 K, MatterSim CsSnI3 600/900 K and SevenNet-0 PbTiO3 900 K. The harness sets float64 explicitly only for MACE-MP-0 (mlip_dynstab/calculators.py:76), which has zero blow-ups and zero failures; CHGNet, SevenNet-0 and MatterSim use package defaults (calculators.py:80-105). The float64 attribution is therefore unsupported by the repo. PLAUSIBLE: I did not check the package default dtypes.
    + manuscript.md:608-610: 'does not trigger the float32 blow-ups' is July wording (manuscript-as-reviewed.md:308). It attributes the blow-ups to float32, contradicting manuscript.md:527-528 and the ledger, where 5 of 8 blow-ups are non-ORB.
    + manuscript.md:640-642 says treating units as independent overstates significance 'by about a factor of four in p'. stats_hardening.json gives clustered p 0.0035 against naive p 0.0003, a factor of about 12. The 'four' only follows from the rounded values '<0.001' and 0.004.
    + supplementary.md:250: the Table S12 caption says 'over the same 57 fitted potentials', but supplementary.md:226-227 says the 228 evaluations cover '57 (system, model) units, involving 75 distinct fitted potentials'. The caption also sits after its table, unlike every other ESI caption.
    + response_to_referees.md:481-482 (R3.4) lists the added tables as S7 and S9–S12 and omits S13, the displacement sweep.
    + response_to_referees.md:484-512 (R3.5) does not tell Referee 3 that the ORB-v2 figures they cited have moved: 0.562 finite-T false-stable is now 5/16 = 0.312 (manuscript.md:373), MgO −2.8 THz is now −1.07 THz (manuscript.md:309), the −35 THz Ti/Hf outliers no longer exist, and the ORB-concentrated blow-ups are now 3 of 8.
    + supplementary.docx still contains 'Outcome: agreement on bcc (ρ=0.78)', the exact stale number the letter's 'Also changed' list (response_to_referees.md:523-525) says was corrected. The .md is correct, but the file RSC would receive is not.
  REFUTED:
    - Other issue 6 (number_changes.md stale) is only partly right. number_changes.md is generated by scripts/reviewed_version_delta.py from manuscript.md, and its line saying the ex-ORB ρ is 'not reported in the current version' is accurate for the main text: −0.003 appears only in ESI Table S9 (supplementary.md:584) and as '≈ 0.00' in ESI §S4 (supplementary.md:648). The 'SevenNet-0 now leads both layers' line and the tasks/todo.md D2 'Current title' are genuinely stale, but both files are internal and not part of the submission.
    - RESIDUE-17 sub-item 6 (the CRediT checklist line should be marked done) is partly refuted. manuscript.md:836-839 has an Author contributions section, but in free prose rather than CRediT roles, so the checklist item 'CRediT author-contributions section' (response_to_referees.md:545) is legitimately still open. It needs rewording, not ticking.
    - RESIDUE-13 calls letter p = 2 × 10⁻⁵ vs manuscript 1.5 × 10⁻⁵ a numeric mismatch. The JSON value 1.52e-05 rounds to 2e-05 at one significant figure, so this is an inconsistency of precision, not a wrong number. The real defect, quoting a unit-level p the letter itself declines and using significant/not-significant language, stands.
    - RESIDUE-2's risk rating of 'medium' is arguably high. RSC's revision file list (rsc_referee_reports.md:10-17) does not ask for a cover letter, so the stale letter only matters if it is uploaded. The gap itself is real.
    - Other issues 1, 2, 3, 4, 5, 7 and 8 are all confirmed: the consensus_error_rate field holds 48/60 = 0.800 correct; manuscript.pdf is dated Aug 30; Author contributions is free prose; the defect descriptions differ between response_to_referees.md:28-31 and manuscript.md:818-822; §3.3 lacks the fluorite ORB weighting claimed at response_to_referees.md:511; the Table S12 caption sits after its table; and make_figures.py:102-103 carries the stale comment.
