# Audit fixes, 2026-10-09 (final adversarial audit of RA-ART-07-2026-006452)

The audit report itself lives outside the repo, in the session scratchpad (`final_audit_2026-10-09.md`).
Every finding was checked before acting on it. Line references are to the files after the last
commit of this pass: **M** = paper/manuscript.md, **E** = paper/supplementary.md, **L** =
paper/response/response_to_referees.md, **C** = paper/response/cover_note_revision.md.
Checks at the end: `verify_claims.py` 215/215 pass (was 206, plus 9 new assertions),
`build_esi_tables.py --check` is clean, `renumber_refs.py --check` is ok,
`build_docx.py --check --allow-pending` passes (4/4 main-text tables captioned, 0 uncaptioned),
and the abstract is 247 words.

Commits: 7580317 (manuscript, ESI, scripts, figures), e678a94 (letter, cover note,
number_changes), 1bb0262 (minor manuscript), 60f0dc3 (§ → Section, kept separate), plus this file.

## Blockers

| # | Outcome | Where |
|---|---|---|
| B1 docx / marked changes not built | **Left for Frank, as briefed** (separate step). The letter's file list is now "Files submitted with this response" and carries a `PENDING-B1` marker: build with `build_docx.py` and `build_marked_changes.py` (reviewed = 76d3a84), then drop any item that is not uploaded. `build_docx.py --check --allow-pending` parses both markdown files cleanly. | L 1205–1206 |
| B2 PENDING/DRAFT markers; false ORCID/TPR claims | Fixed where the text was false. The letter and the cover note no longer say the ORCID is linked or that transparent review was opted into: both are now described as portal actions taken at upload. PENDING-P5 is kept (M Data availability, L 101) because the tag and DOI are genuinely unknown, and the "will be pinned" future tense is true as written. PENDING-P4 is kept in L and added in C. The DRAFT-FOR-FRANK AI-use markers are untouched, as briefed. | L 1179–1181; C 36–39 |
| B3 stale cover note | Fixed. The PENDING-C1c/GRID line is gone. New bullets cover the converged grid (178 units, Table S22), the pre-registered replicates (Table S24), the fluorite and high-T results not surviving convergence, PBE at its own lattice, and the fine-tuning trial with its refuted central prediction. | C 20–34 |
| B4 unnumbered main-text tables | Fixed. They are now Tables 2, 3 and 4, each with a caption and each cited before it appears. Table 1 was already first. The letter's R3.3 and R3.6 "Changes" lines mention them. | M 450–453, 559–564, 606–608; L 1160–1162 |

## Major

| # | Verified? | Outcome | Where |
|---|---|---|---|
| M1 X mode not profiled | Yes. summary.json profiles M (deciding) and Γ (softest); X is −6.63 / −5.47 THz and lies between them. | Qualified with no new DFT, in the abstract ("on both modes profiled"), Section 3.2 (X frequencies, "stable on the modes tested, not established"), Discussion, Limitations, Conclusions and the letter (summary, (b), R1.1, R1.6, R3.1). verify_claims asserts that X was not profiled and lies between the two profiled modes, and pins the printed frequencies. | M 19, 652–664, 1206–1208, 1366, 1401–1404; L 38–39, 120–131, 234–238 |
| M2 title / H2 about model+screen | Yes. 8 + 5 + 3 + 1 = 17 is confirmed against the text. | Title unchanged (Frank's decision). The abstract now says "does not certify the finite-temperature calls made with a model". Section 3.2 replaces "identify a model as a good finite-temperature screener" and adds the model+screen sentence (8 of 17 called the same way by PBE at its own lattice). The Discussion gives the 8/5/3 breakdown. The Conclusions add the model+screen sentence and say that for 8 of 17 the reference at its own equilibrium gives the same call. Letter (b), R1.6 and R3.1 no longer say "the clause stands as a statement about the models". | M 14–16, 708–709, 741–745, 1209–1217, 1386–1388; L 120–131, 669–678, 905–909 |
| M3 "PES softening of H1" overstated | Yes. Own-lattice ratios are 0.87 and 0.93. | Changed to "a softening relative to PBE at fixed geometry", with the own-equilibrium wells added (42.6 against 48.7 and 27.9 against 30.0 meV, within 15 %). Limitations now says "at each MLIP's lattice". L 687 already said "at each model's lattice" (no change). verify_claims checks the ratio and the printed numbers. | M 648–657, 1350; L 30–31 |
| M4 ref 27 wrong | Yes, on Crossref. 10.1103/PhysRevB.76.134301 is "Proton dynamics in superprotonic CsHSO4". 10.1103/PhysRevLett.97.166401 is Wood & Marzari, "Dynamical Structure, Bonding, and Thermodynamics of the Superionic Sublattice in α-AgI", PRL 97, 166401 (2006); the first author is **B. C.** Wood, not D. A. | Replaced. The citing sentence ("Superionic (1): α-AgI") still matches what the paper is about. renumber_refs check ok. The letter's "Other changes" says so. | M 1557; L 1194–1196 |
| M5 SrTiO3 100 K in headline 27 | Yes. All 4 units are SrTiO3 at 100 K. | 23/73 = 0.32 [0.22, 0.43] (all ferroelectric) is given beside 27/77 in Section 3.3, and appears in Section 1, the abstract (4 SrTiO₃ units untested for cell size), the Conclusions and the letter (summary, R3.2). It is computed from data in verify_claims `text_vs_data`. | M 22–24, 128–129, 922–924, 1415–1416; L 22–24, R3.2 |
| M6 PBE force check on unconverged ensembles | Yes | Disclosed in text only, with no new DFT. Section 3.3 now states it first, and Limitations names the surviving false-stables as unchecked. The R1.2 response opens with the limitation, and the old last sentence is removed. | M 976–985, 1354–1356; L 301–305 |
| M7 Fig. 4 x-axis is an artefact | Yes | Least invasive fix that keeps the numbering: Fig. 4 is regenerated by `make_figures.py` (600 dpi, PNG/TIFF/upload Fig4.tif) to plot SSCHA frequency against the screen's **call**. The caption is rewritten, the text cites Fig. 4 at the call agreement, and Spearman ρ now cites Table S16 only. Letter R3.6 updated. | scripts/make_figures.py fig_method_agreement; M 815, 828–829, 846–857; L R3.6 |
| M8 BaTiO3/CHGNet bootstrap 6/10 unstable | Yes. B = 10, mean −0.33, frac 0.6; the other nine are 0/10. | Stated in Section 3.3. The "every model" claim is qualified as holding on the full-ensemble value. Letter R1.4 updated. verify_claims asserts it. | M 934–936, 944–949; L 558–561 |
| M9 negative existence claim | Yes | Changed to "to our knowledge". | M 53 |

## Minor

| # | Outcome | Where |
|---|---|---|
| m1 H1–H3 provenance | Evidence exists: `docs/research_brief.md` in the first commit, 9c2599f (2026-06-20), states H1–H3 before any result. Cited in Section 1 and in the letter's Other changes. | M 104–105 |
| m2 causal over-attribution | Changed to "its settings, chiefly the cumulative step cap, produced…". | M 1240–1241 |
| m3 untested reading | Changed to "One reading, not tested here, is…". | M 1111 |
| m4 PbTiO3 A1 only | Now names A1 and A5 (clamped cell, strain coupling). | M 1336–1340 |
| m5 abstract at 250 | Now 247 words (whitespace count; verify_claims ≤ 250). The convention range (7–24) is kept: dropping it would hide the convention dependence. | M 10–30 |
| m6 "converged and replicated" | **Skipped.** The audit calls it acceptable with "in every finished replicate". The abstract keeps "converged and replicated" and the 77 denominator is "with a resolved call". | — |
| m7 Fig. 3 unresolved unit | Ti/ORB-v2/600 K is drawn as a cross ("converged, unresolved by replicates") via `make_figures.py` (reads e3 summary). Caption updated. | M 833–836 |
| m8 30-epoch summary "pending" | Summarizer gap, not a data error: the 30-epoch re-run did not re-evaluate the base model. `scripts/finetune/summary.py` gets a `--base-eval-from` fallback (CLI in finetune_trial.py) that records `base_loaded_from` and marks CHGNet "not part of this run". The summary was regenerated with 0 "pending", P2 statuses unchanged and base reruns agreeing with the ledger. ESI Section S6 names both summaries. | results/revision/finetune_mace30/summary.json; E 1839–1845 |
| m9 plan codes / referee labels in ESI | C3a/C3b removed from Section S6 and from the Table S26 caption. "(Referee x.y)" removed from the S5.1–S5.3 run-in heads, the S6 heading and the S24/S25 captions (generator). | E S5, S6; build_esi_tables.py |
| m10 ESI table order | Table S12 is now generated (from results/scha_vs_exact.json) and printed between S11 and S13. Section S1.4 points to it, the head note and reading-order note are updated, and letter R3.4 too. Every ESI table now prints in numerical order. | E 391, 664–672, 985; L R3.4 |
| m11 number_changes rows 35–36 | Marked superseded, with the call agreement. | number_changes.md 35–36 |
| m12 L210 "323"; L476–482 order | Now "382 calculations in all, 323 … and 59 …". R1.4 leads with 27/77 and 14/22 v 4/22, with the start-A values in parentheses. | L 215–217, 489–496 |
| m13 R3.6 Changes omit Fig. 2 | "Fig. 2 new" added. | L 1160 |
| m14 seven unresolved read as eight | "(PbTiO₃ with MatterSim at 600 K failed under both recipes)" added in M, E and L. | M 1155; E 603; L 547 |
| m15 back-matter order / e-mail | **Left for Frank**: check the RSC template order at upload. The Purdue e-mail was a deliberate earlier choice. | — |
| m16 DOIs in references | **Skipped** (optional; RSC does not require them). | — |
| m17 ex-ORB p v bootstrap; TOC bars | Explanation clause added. **TOC graphic not changed** (optional design change, left for Frank). | M 1057–1060 |
| Section 3 table: truncation order | "…at any truncation order of the SCHA free-energy Hessian". | M 967 |
| Section 3: "places the gap in the lattice" | Now "is consistent with the gap lying in the lattice, though it does not separate the lattice from the eigenvector". | M 1288–1290 |
| Section 5: S11, S25, S26 not cited in main text | Now cited (Section 2.5, Section 4 E5 paragraph). | M ~375, 1274 |
| Section 4: PBE-lattice B/C labelled pre-registered? | **Skipped, no change needed.** The text never calls them pre-registered: "criteria fixed before the runs" covers the convergence acceptance rule only, and the pl_ stages were coded in 30d5674 (10-03) before they ran. | — |

## § → Section (last, separate commit 60f0dc3)

Conversion order: ranges, then lists, then singles. It ran over manuscript.md (122 → 0), supplementary.md (101 → 0), response_to_referees.md (211 → 6: the six left are all inside referee blockquotes, kept verbatim), number_changes.md (4 → 0), build_esi_tables.py (40 → 0) and verify_claims.py (32 → 0, so the text assertions still match). Inline code spans and HTML comments were skipped. After conversion: verify 215/215, ESI --check clean, docx dry-run passes.

## Left for Frank

- B1: build the docx and marked-changes files, then grep them for "non-predictive", "34/84" and "Table S26". Delete the PENDING-B1 marker once the uploaded set matches the list.
- PENDING-P4 (L and C): link the ORCID and make the transparent-review choice in the portal, then let the sentences say so directly if wanted.
- PENDING-P5: the release tag, commit and Zenodo version DOI.
- DRAFT-FOR-FRANK: the AI-use wording.
- Optional: the TOC bars (m17), the back-matter order against the RSC template (m15), DOIs (m16). The M1 X-mode profile and the M6 PBE forces on the converged ensembles are cheap compute if a stronger answer is wanted.
