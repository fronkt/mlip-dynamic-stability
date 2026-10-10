# Reviewer-coverage audit, 2026-10-08 (RA-ART-07-2026-006452, branch rsc-figure-fixes)

Adversarial check of every referee and editor ask against the current revision. Asks are
paraphrased (no report text is reproduced here). Line numbers: **M** = paper/manuscript.md,
**E** = paper/supplementary.md, **L** = paper/response/response_to_referees.md, all as of HEAD 5ac1a79
plus the working tree on 10-08. The 09-26 audit was re-verified rather than trusted; items it raised
that are now fixed are not repeated.

Status key: DONE / PARTIAL / ARGUED-ONLY / NOT DONE / **awaiting fold: X** (X = GRID converged-SSCHA
grid incl. C1c, C5 bcc cell size, E1 full-PBE coverage, E2 DFT checks, E5 fine-tuning trial).

## Ranked gap list (most likely to sink the paper first)

1. **awaiting fold: GRID. This fold rewrites results; it does not just fill in a sentence.**
   `results/revision/grid_compare.json` (committed f45e77e) shows the converged recipe changes 82
   of 171 SSCHA calls. Of the 57 production SSCHA false-stables that §3.3, the abstract and the
   title's second clause rest on, 32 become unstable, 21 stay false-stable (11 of those 21 with a
   screen-unstable call) and 4 do not converge. Every fluorite false-stable disappears: ZrO₂/MACE-MP-0
   at 100 K goes from +3.08 to about −23 THz. The SrTiO₃/MACE-MP-0 600 and 900 K "false-unstables"
   become stable (+2.4 and +2.8 THz). BaTiO₃/MACE-MP-0 100 K stays false-stable (+1.98 THz). Text
   that has to be rewritten: the title's second clause (M 1); the abstract (M 20-26); the
   contributions (M 120-125); all of §3.3 (M 743-877, including 46/57, fluorites 20 v 1, the 33 v 3
   paired test, "false-unstables that grow with temperature"); §4 (M 1031-1047); §5 (M 1167-1181);
   the TOC panel and blurb; the cover note; letter R1.1, R1.2 and R1.4 and the opening summary
   (L 18-30); Fig. 5; Tables S10, S16 and S17. Table S21 already points to a "Table S22" that does
   not exist (E 1181). Referee 1's central worry was that the SSCHA failure is an artefact. The
   converged grid partly agrees with them, so the letter has to say that plainly.
2. **Editor package NOT DONE.** No marked-changes manuscript exists. `scripts/build_marked_changes.py`
   exists, but nothing it would produce is in paper/. `manuscript.docx` (09-10) is stale: it has no
   new title, no PBE, no Table S21, and it **still contains "non-predictive"**, so uploading it
   would re-trigger R1.6 and R3.1. `supplementary.docx` (09-10, 12 kB) has no rendered tables, so
   uploading it would re-trigger R3.4. `manuscript.pdf` (08-30) is stale too. The letter's "Files
   uploaded" list (L 1006-1014) describes files that do not exist yet.
3. **awaiting fold: C5, but the data are already in the ESI and contradict the text.** Table S21,
   lower part (E 1211-1218), shows bcc Zr with MatterSim changing sign with cell size at 300 K
   (+1.95 THz at 2×2×2, −2.11 at 3×3×3). The manuscript still says the call "holds" and that the
   only cell-size test is on a unit with no instability: M 973-982, M 1138-1139, E 571-582 and §S4
   E 1259-1262. The letter does the same (L 777-782). Referee 3 can see the contradiction inside
   the ESI. The bcc N point is also a zone-boundary q-point, so the bcc SSCHA conclusions of §3.3
   need the same "no convergence claim" caveat that §3.5 gives the R- and X-point systems.
4. **awaiting fold: E2 and E1.** E2 shows the CsSnBr₃ PBE k-mesh is not converged (well depth off
   by 6.8%), and PBEsol fixes 9 of the 15 PBE errors, 8 of them CsSnBr₃. The repeated claim that the
   CsSnBr₃ mis-calls "persist on PBE" and belong to "the screen, the functional or the label"
   (M 598-602, 1019-1021, 1163-1165; E 349-351, 1300-1306; L 27-28, 189, 541-543) must change,
   probably toward "functional". E1 now covers every mode (93 of 139 paths), so "the PBE call
   covers one or two of the screen's modes per unit" (E 1309) and the 81→101 count may move. The
   PBE-lattice and PBE-eigenvector checks (pbe-lattice phases B and C) are still unrun. The letter
   says so (L 193), but the referee asked for an independent PBE reference.
5. **R2.1 is ARGUED-ONLY; awaiting fold: E5.** The fine-tuning discussion (M 1049-1087) is
   literature-based and says "I have not fine-tuned any model here" (L 619). The pre-registered E5
   trial ran on 10-04 and has not been written up. Referee 2 recommended rejection over exactly
   this point. Folding E5, whatever it shows, is the single best lever with Referee 2.
6. **R1.3 is PARTIAL (neutral, but vague).** Referee 1 named six works, all from one group. The
   paper cites PRB 2021 (ref 37), Acc. Chem. Res. 2026 (ref 38) and Adv. Energy Mater. 2026
   (ref 39). It does not cite JPCL 2021 (Hajibabaei and Kim, solid-electrolyte survey) or Chem.
   Phys. Rev. 2024 or 2025. The letter says only that it "considered the other works" (L 358-360)
   and does not name the omitted three or give a reason. The tone is neutral, which matches the
   editor's instruction; one sentence naming what was left out and why would close it. Separately,
   **the references are not numbered in order of first citation**, which RSC style requires.
   §1 cites 1, 3, 2, …, 37 before 32, and refs 27-39 appear before refs 6-18. All the references
   the letter adds were verified (tasks/refs-2026-09-26.md has DOIs). Plain web search did not
   index refs 38, 41 or 43; that is a coverage gap, not evidence that they don't exist.
7. **R1.4 and R1.2 coverage is PARTIAL.** Convergence diagnostics (gradient history, stop reason,
   bootstrap Hessian SD) exist only for the 4 seed-study units, and 0 of 16 seeds converged (Table
   S21). The production grid recorded none (M 370-372). The PBE-on-samples benchmark uses 12
   configurations from each of 4 *unconverged* ensembles (M 409-411). Once GRID is folded, the
   diagnostics should be reported grid-wide, and the letter should say whether the PBE sampling
   check was repeated on converged ensembles or why it was not.
8. **PENDING markers remain:** 12 in M (27, 383, 707, 708, 800, 815, 971, 983, 1045, 1142, 1176,
   1202), 7 in E (11, 450, 499, 586, 654, 1245, 1272), 9 in L (30, 93, 115, 297, 384, 402, 430, 783,
   894, plus PENDING-FIG at 975 and PENDING-P4 at 995) and 1 in response/cover_note_revision.md.
   C1b (the include_v4 run on BaTiO₃) is still open.
9. **No AI-use statement (NOT DONE).** response/ai_use_statement_DRAFT.md is not in the
   Acknowledgements (M 1259-1267). RSC policy expects a disclosure. Frank has to choose the wording.
10. **Internal plan codes are in text that will be published.** "(C3a)" and "(C3b)" appear at
    M 397 and 405. "plan item C3a/C3b/C1/C5" and "Referee 1.x" appear in ESI captions and headings
    (E 1116, 1172, 1181, 1211, 1287, 1312). Remove them.
11. **Portal items need Frank to confirm.** The ORCID link (PENDING-P4, L 995) and transparent
    peer review: both the letter (L 993) and the cover note assert opt-in as already done.
12. **Minor items:**
    - R3.5: the main text gives the SSCHA ferroelectric recall 5/27 without its ex-ORB value
      (5/23 is in Table S9) at M 26, 746 and 771, and gives no ORB split for T*.
    - R3.3: bare rates or AUCs without counts or intervals at M 28, 115, 675 ("0.63–0.74") and
      1184-1187 ("AUC 0.76", "0.63", "0.31").
    - Figs. 2 and 4 still plot the screen's curvature, including its numerical negatives
      (PENDING-FIG, L 975).
    - Three letter quotes are ASCII-transliterated rather than verbatim: R3.1 (L 715: phi, rho,
      <=), R3.2 (L 748: 4x4x4, (1/2,1/2,1/2), Gamma) and R3.5 (L 904: -2.8, -2x10^6). Every other
      quoted passage is verbatim.
    - `paper/cover_letter.md`/.docx is the stale Digital Discovery letter ("gold-standard",
      "systematically false-stabilises", AUC 0.75). Do not upload it; response/cover_note_revision.md
      is the right file.
    - `response/number_changes.md` (09-11, public repo) still says "SevenNet-0 now leads both
      layers… the paper says so", which contradicts the no-ranking claim.
    - The ESI header gives table ranges as S1-S20 (E 6, 14), but Table S21 exists.
    - The abstract is 246 words (277 as reviewed). The RSC Advances limit was not re-fetched (web
      use was restricted to checking references); the 09-26 audit recorded "about 50-250, unverified".
    - The corresponding email is cai485@purdue.edu (M 5), against Frank's canonical
      gmail address. This is a deliberate choice recorded in the 09-26 audit (PKG-g); keep it
      consistent across the portal.

## Coverage table

| # | Ask (paraphrase, ≤15 words) | Manuscript / ESI | Letter | Status | Gap |
|---|---|---|---|---|---|
| R1.s1 | Shared MLIP surface makes screen–SSCHA agreement a consistency test, not accuracy | M 356-358, 684-687, 734-735, 1119-1121 | L 142-151 | DONE | none |
| R1.s2 | Primary claims exceed the design; scale them back | M 545-549, 648-667, 1190-1193 | L 147-151 | DONE | the SSCHA claims now exceed the converged data (gap 1) |
| R1.s3 | Reach Digital Discovery standard: FP validation, convergence, context | see R1.1-R1.4 | L 46-47 (transfer noted) | ARGUED-ONLY | journal is now RSC Adv.; no explicit answer needed beyond the items |
| R1.1a | First-principles evaluation along soft-mode coordinates, key subset (SrTiO₃, BaTiO₃, bcc Zr) | M 389-411, 592-608; E 1280-1310, Table S19 | L 166-193 | DONE; **awaiting fold: E1, E2** | CsSnBr₃ claim rests on an unconverged k-mesh; PBE only at the MLIP lattice |
| R1.1b | Disagreement cannot establish which method is right | M 684-687, 1119-1123 | L 217-230 | DONE | — |
| R1.1c | Transitions are global thermodynamics, not local curvature | M 146-149, 777-799, 1112-1114 | L 195-215 | DONE; **awaiting fold: GRID** | the local-criterion account loses most of its units once converged |
| R1.2a | Separate MLIP force-engine errors from SSCHA method failure | M 822-828, 845-863; E 1312-1322, Table S20 | L 246-297 | PARTIAL; **awaiting fold: GRID** | 4 units × 12 configs from unconverged ensembles; the GRID shows early stopping was a major cause |
| R1.2b | Benchmark MLIP forces/energies vs FP along sampling paths | Table S20 | L 246-262 | PARTIAL | as R1.2a; consider re-sampling the converged ensembles or saying why not |
| R1.3a | Contextualise MLIP architectures and training domain | M 57-81 | L 321-346 | DONE | — |
| R1.3b | Discuss sparse-GP / on-the-fly MLIPs; cite PRB 2021, JPCL 2021 | M 68-75 (ref 37 = PRB 2021) | L 338-340 | PARTIAL | JPCL 2021 not cited, and the decline is not stated |
| R1.3c | Bayesian committee/active learning on anharmonic transitions; cite CPR 2024, 2025 | M 57-67, 72-75 (refs 27-30, 38) | L 324-346, 358-360 | ARGUED-ONLY (implicit decline) | CPR 2024/2025 not cited or named; add one neutral sentence |
| R1.3d | Cite Adv Energy Mater 2026 | M 75-77 (ref 39) | L 343-346 | DONE | — |
| R1.3e | Situate screen in ML-driven high-throughput screening; cite Acc Chem Res 2026 | M 83-93 (refs 34-36, 38) | L 348-356 | DONE | catalyst discovery declined explicitly (fine) |
| R1.3f | Editor: include only relevant references | refs 27-46 | L 985-987 | DONE | references not in order of first citation (RSC style) |
| R1.4a | Justify or drop "methodological trap" | phrase absent (grep) | L 377-383 | DONE | — |
| R1.4b | Report SSCHA sample sizes | M 362-372; E 427-445; Table S11 | L 386-397 | DONE | production counts not recorded (stated) |
| R1.4c | Report gradient history | Table S21 (4 units) | L 415-429 | PARTIAL; **awaiting fold: GRID** | production grid has none; give it grid-wide |
| R1.4d | Report stopping criteria | M 363-366, 373-382; E 427-445 | L 393-401 | DONE | the cumulative max_ka cap is disclosed |
| R1.4e | Report Hessian uncertainties | Table S21 (bootstrap SD) | L 418-427 | PARTIAL; **awaiting fold: GRID** | 4 units only |
| R1.4f | Extend four-seed test beyond bcc Zr to BaTiO₃/fluorites | M 373-382, 966-971; Table S21 | L 409-429 | DONE (run); **awaiting fold: GRID** | 0/16 seeds converged, so the test shows the recipe failed; §3.5 PENDING at M 971 |
| R1.5a | ESI derivation: mode mass-weighting | E 131-166 | L 447-453 | DONE | — |
| R1.5b | ESI derivation: Gaussian-width self-consistency | E 213-285 | L 454-465 | DONE | the minimum in Ω is not shown (stated) |
| R1.5c | ESI: mode–mode coupling limitations | E 298-362; Table 1 A1 | L 466-478 | DONE | no quantitative bound (stated) |
| R1.5d | Sensitivity to polynomial fit window | M 345-347; E 58-64; Table S14 | L 483-485 | DONE | — |
| R1.5e | Sensitivity to sampling range | M 347-350; Table S14 | L 486-489 | DONE | — |
| R1.5f | Sensitivity to cell commensurability | M 321-343, 350-352; E 168-195, 1263-1267 | L 490-504 | PARTIAL | frozen-cell convention reported (it decides outcomes); FC supercell / q-set not enlarged |
| R1.6a | Remove "non-predictive" framing from Abstract and Discussion | grep clean in all .md | L 517-524 | DONE in .md; **NOT DONE in manuscript.docx** | stale docx still says "non-predictive" |
| R1.6b | Align headline claims with weak positive correlation | M 16-18, 610-620, 1011-1029 | L 526-546 | DONE | — |
| R2.s1 | Not only a stability argument; compare with reference energies/forces | M 389-411; E §S5 | L 563-572 | DONE | — |
| R2.1 | Discuss how fine-tuning would affect results | M 164, 1049-1087, 1152; refs 40-43 | L 579-623 | ARGUED-ONLY; **awaiting fold: E5** | pre-registered trial run 10-04, not yet written up |
| R2.2a | Report ensemble uncertainty of force predictions | M 917-941; Table S18 | L 653-698 | DONE | the main text gives AUCs only, not spread magnitudes (minor) |
| R2.2b | Test whether that uncertainty flags untrustworthy predictions | M 925-941, 1089-1109 | L 675-698 | DONE (negative result, pre-registered) | — |
| R3.s1 | Harmonic leaders ≠ finite-T leaders (summary premise) | M 545-549 (no ranking) | L 63-89 | DONE | premise moved; disclosed |
| R3.s2 | Screen 23/30 vs SSCHA 7/30 (summary premise) | M 25-26, 743-751 | L 67-68 | DONE; **awaiting fold: GRID** | the SSCHA recall will change |
| R3.s3 | Vote AUC 0.745 vs spread 0.52 (summary premise) | M 893-911 | L 72-73 | DONE | — |
| R3.1a | §4 must not assert H2 strong form | M 1011-1029 | L 724-725 | DONE | — |
| R3.1b | Remove "non-predictive" | grep clean (.md) | L 725 | DONE (.md) | docx stale (gap 2) |
| R3.1c | Carry §3.2 formulation into abstract | M 15-18 | L 733-739 | DONE | — |
| R3.1d | Carry it into the title framing | M 1 | L 95-115 | DONE for clause 1; **awaiting fold: GRID** for clause 2 | — |
| R3.2a | Run 4×4×4 SSCHA (SrTiO₃ + fluorite) or drop zone-boundary convergence claims | M 985-990; E 583-588 | L 756-761 | DONE (claims dropped) | — |
| R3.2b | 2×2×2 marginal for the fluorite X mode | M 987-1002; E 589-595 | L 758-770 | DONE | — |
| R3.2c | BaTiO₃ Γ argument covers one system only | M 998-1000 | L 763-775 | DONE | — |
| R3.2d | (implied) no convergence claim the cells can't support | M 973-982, 1138-1139; E 571-582, 1259-1262 | L 777-783 | **PARTIAL; awaiting fold: C5** | Table S21 shows a MatterSim bcc-Zr sign flip at 3×3×3 that the text contradicts; bcc N is zone-boundary too |
| R3.3a | Rates as k/n with Wilson/bootstrap intervals | M 421-430, 478-489, 524-530, Fig. 5/6 | L 809-815 | DONE | residual bare values at M 28, 115, 675, 1184-1187 |
| R3.3b | Treat the AUC as clustered by system | M 883-898 | L 817-827 | DONE | — |
| R3.3c | State the composition of the n=60 and n=75 sets | M 551-552, 619-620, 885-886; Tables S7, S8 | L 817-820 | DONE | — |
| R3.3d | Drop five-model Spearman as evidence | removed from H2; bcc ρ=0.11 descriptive (M 718, 740) | L 829-833 | DONE | — |
| R3.3e | McNemar p is non-rejection, not support | M 559-561, 654-656, 1026-1027 | L 835-846 | DONE | — |
| R3.3f | Use a system-level permutation test | M 557-568, 612-613, 757-763 | L 840-846 | DONE | — |
| R3.3g | Call the finite-T ranking suggestive, not established | M 545-549, 1190-1193 | L 848-850 | DONE (no ranking claimed) | — |
| R3.4 | Supply per-system Tables S1-S4 | Tables S4, S5, S6, S8 (renumbered); E 618-656 | L 870-900 | DONE | ESI ranges say S1-S20; S22 dangling; supplementary.docx stale |
| R3.5a | Report all rates with and without ORB-v2 | Table S9 (E 872-902); S15-S18; M passim | L 912-951 | PARTIAL | main text lacks 5/23 beside 5/27 and a T* split; GRID adds new ORB numbers |
| R3.5b | ORB precision/architecture confound acknowledged | M 165-171 | L 916-919 | DONE | — |
| R3.5c | ORB extremes (MgO, −35 THz, blow-ups, 0.562) | M 437-438, 865-877, 530 | L 920-929 | DONE | premises corrected; the GRID adds −33,506 THz ORB blow-ups |
| R3.6 | Figures in order of first citation | cited at M 445, 457, 691, 720, 748, 909; printed at 453, 464, 722, 732, 769, 943 | L 959-975 | DONE | Fig. 2/4 regeneration open (PENDING-FIG); Figs. 3-5 change with GRID |
| Ed1 | Check suggested references; include only relevant | refs 37-39 | L 985-987 | DONE | numbering order (gap 6) |
| Ed2 | Point-by-point response | L | — | PARTIAL | 11 PENDING markers; no upload-format file |
| Ed3 | Marked-changes manuscript | none | L 1009-1010 | **NOT DONE** | build_marked_changes.py exists, no output |
| Ed4 | Clean .docx with figures | manuscript.docx 09-10 stale | L 1011 | **NOT DONE** | rebuild after the folds |
| Ed5 | Figures ≥600 dpi | results/figures/upload/Fig1-6.tif, 600 dpi RGB, 11.7-17.0 cm | L 1013 | DONE; **awaiting fold: GRID** | Figs. 3-5 and 2/4 to regenerate |
| Ed6 | TOC graphic 8×4 cm, text ≤250 chars | toc_graphic 1889×944 px at 600 dpi = 8.0×4.0 cm; blurb 235 chars | L 1014 | DONE (format); **awaiting fold: GRID** (content) | the SSCHA clause and right panel assume production false-stables |
| Ed7 | Revised ESI | supplementary.md | L 1012 | PARTIAL | docx stale; no marked ESI |
| Ed8 | ORCID linked | M 6 | L 992-995 | PARTIAL (Frank) | portal link PENDING-P4 |
| Ed9 | CRediT author contributions above COI + Acknowledgements | M 1250-1257 | L 989-992 | DONE | — |
| Ed10 | Transparent peer review option | — | L 993; cover note | PARTIAL (Frank) | asserted as done; confirm in the portal |
| Pol | AI-use disclosure (RSC policy, not a referee ask) | absent | — | NOT DONE (Frank) | draft not inserted |

## Verification notes

- Letter quotes: 15 of 18 blockquotes are verbatim substrings of the reports. R3.1, R3.2 and R3.5
  are ASCII-transliterated (checked by script).
- Forbidden language: "non-predictive", "strong form", "blind spot", "methodological trap",
  "gold standard" and "necessary but not sufficient" do not appear in manuscript.md,
  supplementary.md, toc_blurb.txt or the AI draft. They do appear in the stale cover_letter.md
  ("gold-standard") and in manuscript.docx ("non-predictive").
- The letter's "changed X" claims were checked against M and E for R1.1-R3.6. All were found in
  the text, except the C5, GRID and E5 items still marked PENDING, and the "Files uploaded" list,
  whose files do not exist yet.
- Figure order was checked against first-citation lines (table above).
- Converged-grid numbers are from results/revision/grid_compare.json (variants.converged_only,
  fs57_outcomes, changed_calls, c1c_overlap). They are reported here only to rank risk; the fold
  belongs to the agent doing it.
