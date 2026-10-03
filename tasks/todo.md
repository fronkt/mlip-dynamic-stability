# RSC Advances major revision: phased plan (rewritten 2026-09-26)

**Manuscript** RA-ART-07-2026-006452. **Decision** major revision, 2026-09-11 (Assoc. Ed. Dr Lydia
Rhyman). Referees: R1 major, R2 reject-leaning, R3 minor.
**Committed date: 9 October 2026.** Frank emailed advances@rsc.org on 2026-09-26 07:04 UTC asking
for time to that date. If it slips, send a follow-up before 9 Oct, not after.
**Source of truth for what is open:** `tasks/audit-2026-09-26-rsc-revision.md` (18-agent audit,
each finding adversarially verified). Item IDs below (R1.1, PKG-c, RESIDUE-12 ...) refer to it.
**Predecessor plan:** `tasks/todo-archive-2026-09-26.md`. Its [x] marks for E4, G1, I2 and F1 were
false.

## ▶ RESUME HERE — paused 2026-09-27 at Frank's request ("pause after the runs are done, save everything")

**Done and pushed (branch `rsc-figure-fixes`):** Phase 2 stats; Phase 3 text (manuscript, ESI S1–S18, figures,
TOC, README, DAS); response letter rewritten + 3 referee-simulation critiques + fact-check; packaging tools
(`build_docx.py`, `build_marked_changes.py`, `renumber_refs.py`); C2, C4, C1, C1c, C5 results; Fig. 2 replaced.
Everything that feeds the paper is in `tasks/revision-claims-2026-09-27.md` (§9 = later corrections).

**Compute results (all committed under `results/revision/`):**
- C2 force spread (R2.2): pre-registered primary AUC 0.681, CI [0.416, 0.908] → NOT shown; committees 0.72–0.73;
  no-overlap check 0.314 → the signal is extrapolation on overlapping configs. Folded into §3.4/§4/§5 + Table S18.
- C4 displacement sweep, all five models: MACE/MatterSim/SevenNet no flips; CHGNet controls only; ORB-v2 test systems.
- C1 (production SSCHA recipe, 4 seeds): NONE converged; `max_ka=20` is a CUMULATIVE cap in python-sscha 1.6.1,
  and the library passes a placeholder gradient error (Ensemble.py:2657). BaTiO3 Hessian +2.87 ≈ its
  ForcePositiveDefinite start +2.88. SrTiO3 600 K seeds −15.8…−20.3. MACE Zr 2×2×2 +1.80 → 3×3×3 +1.55.
- **C1c converged SSCHA (real error, per-population cap, two starts):**
  BaTiO3/MACE 100 K +2.03 (A) / +1.85 (B at 1.0 THz) → STILL stable vs unstable label;
  SrTiO3/MACE 100 K +1.13 / +1.02 → stable vs label (T_c 105 K; production had −0.49);
  ZrO2/MACE 100 K −22.4 / −26.0 → UNSTABLE, correct (production +3.09 was the non-convergence);
  MatterSim Zr 2×2×2: 50 K +0.41/+0.41, 300 K +0.92/+0.93; **3×3×3 300 K −0.92/−0.95** (bcc 2×2×2 stability is a
  finite-size effect). BaTiO3 start B at 0.3 THz fails (complex-dyn assertion) — the 1.0 THz retry is in `_b1/`.
- **C3a PBE along the screen paths (6 systems incl. KNbO3, CsSnBr3): 380/380 pw.x jobs JOB DONE. C3b PBE on
  SSCHA configs + rattled baseline: 68/68 JOB DONE.** Queue logs: 377 + 68 ok, 0 SCF-not-converged; the 3 C3a
  failures (QE d_matrix error 14 on the CHGNet-relaxed, slightly off-cubic BaTiO3 cell:
  `a_batio3_cubic_chgnet_q0-0-0-b0_i00`, `..._q1d2-1d2-0-b0_i00`, `..._i01`) were rerun with `nosym=.true.`
  (pw.in edited by hand; original in `pw.in.orig_sym`, reason in `NOSYM_NOTE.txt`, failed output in
  `pw.out.failed`) → ok. job.json does not record nosym; **do not run `qe-inputs --force`** on these three or
  analyze will mark their outputs `stale_output`. NOT YET ANALYZED. Everything committed (5549086, 222465e):
  `results/revision/dft/{geom,geom_c3b,mlip,qe,logs}`; `.gitattributes` keeps that tree byte-exact because
  analyze hashes the geometry files and byte-compares pw.in with pw.in.ran (verified: committed blob sha =
  working copy = the `geom_sha256` MLIP-eval recorded on the box). Not in git: the 533 unselected `ax_*` C3a
  inputs (never run) and pw.err (gfortran IEEE notes only) — both in the raw archive.
- **Box 52872517 DESTROYED 2026-09-27 ~22:32 UTC** after all runs finished (vast credit left $17.80). Nothing
  runs anywhere. Raw archives in `C:\Users\frank\mlip-rsc-revision-raw\` (local only, not yet on Zenodo):
  `box_final_2026-09-27.tar.gz` (64 MB, 6,093 files, verified count + gzip: full `results/revision/dft`,
  C1c `sscha_converged{,_b1}` incl. `work/` ensembles, box `logs/`, `aside/`, c1c/c3b driver scripts);
  `sscha_work_2026-09-27.tar.gz` (C1 seeds); `dft_c3a_2026-09-27.tar.gz` (superseded by box_final). The C1c
  `work/` ensembles are also extracted into the repo tree (gitignored). A new box needs `setup_rsc_box.sh` +
  the as-run notes in `scripts/box/as_run/README.md` (SSSP API URL).
- **Found by the letter integrator:** the screen's symmetric-point curvature is positive BY CONSTRUCTION (= trial
  stiffness), so "52/57" was circular; the informative count is the screen CALL (46/57). Fixed everywhere; lesson
  in `tasks/lessons.md`. H2's 17 v 4 depends on the frozen-cell convention (7 v 4 to 24 v 4).

**Next session, in order:**
1. `python scripts/dft_reference.py analyze` on the committed C3a/C3b outputs (runs locally, no box) → R1.1 (does the PBE-backed screen reproduce
   the MLIP calls on BaTiO3/KNbO3/CsSnBr3? decides whether the H2 counterexamples are screen error or model error) and
   R1.2 (MLIP force error on SSCHA-sampled vs near-equilibrium configs). Adversarially verify before writing.
2. Decide the converged-SSCHA GRID (`--preset grid`, ~105 units, start A, ~5–7 GPU-h, ~$4): needed if §3.3 keeps a
   screen-vs-SSCHA comparison; the paper's SSCHA numbers are otherwise unconverged. **Code ready, NOT run**
   (`--preset grid`, reviewed; 113 units = 58 displacive T<=300 K + 10 SrTiO3 + 45 bcc 3x3x3 at 100/300/600 K;
   start A; `scripts/box/grid_run.sh` runs a COPY `scripts/sscha_seed_study_grid.py`; est. ~40 serial GPU-h =
   ~4-5 h wall with 14 workers (3 per env, 2 for ORB; check GPU memory on a 24 GB card), worst case much longer at
   the 7200 s unit cap; all five envs on the old box had SSCHA, a rebuilt box must too). NOT covered by the grid:
   non-bcc 600/900 K and bcc 50/200 K, which §3.3's false-unstable counts, the 57-unit ladder and Fig. 3 use —
   decide whether to extend or to scope the claims to what is re-measured. Copy the 4 overlapping C1c startA
   JSONs only if their recipe matches (the summary flags `other_recipe`).
3. **Title (Frank):** the second clause "nor a default SSCHA cross-check" now rests on: converged SSCHA still calls
   BaTiO3/SrTiO3 stable at 100 K but gets ZrO2 right; the production "default" failure was partly our step cap.
   Bring Frank a concrete proposal with the grid numbers.
4. Phase 4 fold-in: all PENDING markers (manuscript 23, ESI 16, letter 23); rebuild Figs. 3, 4, 5 from converged SSCHA
   (Fig. 4's x-axis is the positive-by-construction curvature — redesign or drop).
5. Phase 6 package: `renumber_refs.py --letter`, `build_docx.py --pdf`, `build_marked_changes.py --pdf`, CRediT
   (done), AI-use (Frank's choice: `paper/response/ai_use_statement_DRAFT.md`), DAS, merge to main + tag, Zenodo
   (include `C:\Users\frank\mlip-rsc-revision-raw\` archives).
6. Phase 7 fresh adversarial re-audit; Phase 8 Frank uploads by 9 Oct.

**Frank's open decisions:** title (after the grid), AI-use wording, gmail vs cai485@purdue.edu.

## Decisions (Frank, 2026-09-26)

- [x] **Title approved:** "Neither harmonic benchmarks nor a default SSCHA cross-check certifies a
  foundation machine-learning interatomic potential for finite-temperature dynamic stability".
  The title is an existence claim (harmonically correct models that are wrong at finite T exist),
  so it survives the clustering result below. The *significance* wording under it does not.
- [x] **Rent a machine** for the compute arm (Phase 1). vast.ai credit was $11.63 on 09-26; Frank
  to top up if the run needs more.
- [x] **Transparent peer review: opt in** ("sure, or your call"). Precondition: the R1.3 decline
  paragraph is rewritten neutrally before upload (Phase 5).
- [x] Corresponding e-mail: **cai485@purdue.edu** (Frank, 2026-10-03) — Purdue RSC read-and-publish covers the APC; Purdue affiliation + portal record must match.
- [x] **Converged-SSCHA grid: RUN IT** (Frank, 2026-10-03), `--preset grid`, start A, on vast.ai credit.
- [ ] AI-use statement wording. I draft it, Frank confirms. Open.

---

## Phase 0 — Setup (09-26)

- [x] 0.1 Archive the old plan; write this one.
- [x] 0.2 Commit the audit file and this plan on `rsc-figure-fixes` (6717c4b).

## Phase 1 — Compute on the rented box (critical path, 09-26 → ~09-29)

One box: GPU for the MLIPs and SSCHA, ≥30 real cores (read `cpu.max`, not `nproc`) for DFT.
Nothing writes to `results/ledger.parquet` except C4 (which uses method `harmonic_dispsweep`).
Every study writes its own JSON/parquet under `results/revision/` and records the env lock.

- [ ] **C1 SSCHA seeds + diagnostics (R1.4).** `scripts/sscha_seed_study.py`. Four seeds each on
  batio3/MACE/100 K (the Table S2 false-stable), zro2/MACE/100 K (fluorite false-stable),
  zr_bcc/MatterSim/50 K (a bcc unit that HAS a harmonic instability; the old study used MACE Zr,
  which has none), srtio3/MACE/600 K (high-T false-unstable runaway). Per run, persist: number of
  populations, per-population gradient and free-energy history, converged flag, lowest frequency
  after ForcePositiveDefinite (the start), final Hessian lowest-6, and a bootstrap-over-configs
  spread of the lowest Hessian eigenvalue (the Hessian uncertainty R1.4 asks for). Save a subsample
  of the final ensemble (positions + MLIP forces/energies) for C3b.
- [ ] **C1b include_v4=True** on batio3/MACE/100 K with a hard time cap (try the julia backend). If it
  finishes, it is the only direct test of "the truncation is what goes wrong"; report either way.
- [x] **C2 Force-level ensemble uncertainty (R2.2).** DONE 09-27 (5ad84cd): pre-registered primary AUC 0.681, CI [0.416, 0.908] → NOT SHOWN; committees 0.72-0.73 (CI just above 0.5) but the no-overlap check (g) gives 0.314 on 43 units → the signal is extrapolation onto overlapping configurations, not wrong calls. `scripts/force_spread.py`. Quantum
  phonon-rattled configurations at each ladder T from ensemble-mean harmonic FCs (|ω| for imaginary
  modes); evaluate all five models plus two within-architecture committees (MACE-MP-0
  small/medium/large, MatterSim 1M/5M). Scores: cross-model force RMS deviation per (system, T) for
  the consensus test, and per-model deviation from the ensemble mean for the per-unit test. Clustered
  AUC, with and without ORB-v2.
- [x] **C3a PBE along the soft-mode coordinates (R1.1).** RUN 09-27: 380/380 JOB DONE (3 via nosym rerun); analysis pending. `scripts/dft_reference.py` + QE (conda-forge),
  SSSP pseudopotentials. SrTiO3 R tilt, BaTiO3 Γ/deciding mode (each model's pattern), bcc-Zr N
  point, ZrO2 X point. All five MLIPs re-evaluated on the identical geometries. Outputs: E(Q) curves,
  well depths, and the single-mode screen's call re-solved on the PBE-fitted potential.
- [x] **C3b PBE forces on SSCHA-sampled configurations (R1.2).** RUN 09-27: 68/68 JOB DONE; analysis pending. 12–16 configurations per C1 unit.
  MLIP-vs-PBE force and energy error on thermally sampled configurations, against a near-equilibrium
  baseline. This is the out-of-distribution test R1.2 asks for.
- [x] **C4 Displacement sweep, remaining four models.** DONE 09-27 (a27e1e3): MACE/MatterSim/SevenNet no flips; CHGNet controls only; ORB-v2 flips test systems. `scripts/run_disp_sweep.py --device cuda`.
- [ ] **C5 bcc finite-size, re-measured.** zr_bcc SSCHA 2×2×2 vs 3×3×3 at 100 and 300 K for
  MatterSim and MACE-MP-0 in the current envs (the deposited 3×3×3 rows are June v1).
- [x] C6 Pull everything back; destroy the box the moment it is idle. DONE 09-27 ~22:32 UTC (see RESUME block).

## Phase 2 — Zero-compute statistics and sensitivity (parallel with Phase 1)

- [x] **S1 H2 clustered.** (e81c904) System-clustered exact paired test at every ladder T, with and without
  ORB-v2, per-system discordance counts, and a leave-one-system-out check. Into
  `stats_hardening.py` / `.json` and `verify_claims.py`. (R1.6, R3.1, R3.3-H2)
- [x] **S2 bcc call agreement** on `pred_stable` (31/45, ex-ORB 25/36), with the 12 trivially
  agreeing MACE/CHGNet pairs split out (23/33) and MatterSim's 3/9. Curvature-sign agreement kept,
  labelled as such. (R1.1 residue, R3.5)
- [x] **S3 ORB-v2 split everywhere (R3.5):** four-model guardrail (AUC 0.628, CI spanning 0.5, tie
  rule stated), H2 ladder, family recalls, controls, tolerance sweep, SSCHA blow-ups and failures per
  model.
- [x] **S4 SSCHA high-T false-unstables** table (0/5/10/15 by T; SrTiO3 all above-Tc units; the 14/23
  sign reversals). (R1.2)
- [x] **S5 Screen sensitivity (R1.5).** `scripts/screen_sensitivity.py` → `results/screen_sensitivity.json`
  and one ESI table: fit-window multiplier and floor, sampling-range truncation, frozen-cell
  normalisation convention (minimal / common FC cell / per f.u. / doubled), with mode flips, unit
  flips, FE recall and the SrTiO3 gate as columns. Plus unbracketed wells and scan-edge Q0 counts.
- [x] **S6 Literature (verified before use):** `tasks/refs-2026-09-26.md` fine-tuning refs for R2.1; ACR 59, 103 and AEM
  10.1002/aenm.71046 on their merits for R1.3; published PBE values to cross-check C3a.

## Phase 3 — Manuscript and ESI text (after Phase 2, ~09-28 → 10-01)

DONE 09-27 (4decd1b) from `tasks/revision-claims-2026-09-27.md`; T1–T11 all applied and adversarially
reviewed (manuscript, ESI Tables S14–S17, figures 600 dpi RGB + upload/Fig1–6.tif, TOC 8×4 cm + 235-char blurb,
README, DAS). Added: S7 local-vs-global criterion (`criterion_blindness`, da7cf29): SSCHA false-stables
reproduced by the screen's own curvature on 52/57. PENDING markers remain for C1/C1b/C2/C3a/C3b/C5/P5.

- [ ] T1 H2 reworded everywhere: large and one-directional in unit counts, not significant at
  system level (clustered p 0.15 at 300 K, 0.07 at 600 K); the screen-T* confound stated.
- [ ] T2 bcc: call vs curvature agreement relabelled in §3.3, Fig. 4, §S1.3, §S4; "clean gold standard"
  and "tracks" removed; the MACE-Zr robustness tests described as what they are until C5 lands.
- [ ] T3 §3.3 mechanism: "consistent with the truncation predicted in ref 22"; Table S2 described
  correctly; the v4 contradiction (583-593 vs 600-605) resolved; OOD hypothesis stated.
- [ ] T4 High-T SSCHA false-unstables reported and discussed (§3.3/§4), including SrTiO3.
- [ ] T5 R1.5: ESI derivation fixes (cell normalisation, Q definition, threshold 0.0075 Å, root
  fallback, parity assumption), coupling premise ("hundreds of meV" is false), sensitivity table.
- [ ] T6 R1.4 text: §2.5 initialiser (phonopy FCs, 0.03 Å), real stopping criteria, sample sizes.
- [ ] T7 ESI §S2.4 rewritten to match §3.5 (R3.2); §S4 tolerance outcome; stale numbers (Zr −0.47,
  hf_bcc flip direction, the −2×10⁶ attribution, float64 attribution, factor ≈12, "absent at 100 K",
  "ORB weakest", "the reordering").
- [ ] T8 R2.1 fine-tuning paragraph cited, and it says which findings fine-tuning could and could not
  change; scope in abstract and Conclusions.
- [ ] T9 R1.3 wording fixes (GAP, GNoME, Matbench), refs renumbered in citation order, style.
- [ ] T10 Figures: Fig. 2 recaptioned (screen curvature, not harmonic) + colorbar fixed; Fig. 4
  caption and −35 THz annotation; model names in legends; numbered 600-dpi RGB TIFFs.
- [ ] T11 Abstract ≤ 250 words; every rate with k/n and an interval.

## Phase 4 — Fold compute results (~09-30 → 10-02)

- [ ] F1 C1/C1b → §3.3, §3.5, §S2.4, Table S11 (+ new seed table). Report whichever way it falls.
- [ ] F2 C2 → §3.4, §4, Table S9. C3a/C3b → new §3.x paragraph + ESI section + table.
- [ ] F3 C4 → Table S13 complete; C5 → §3.5 finite-size sentence.
- [ ] F4 `verify_claims.py` assertions for every new number; `build_esi_tables.py --check` green.

## Phase 5 — Response letter (~10-02 → 10-03)

- [ ] L1 Rewrite on the final numbers: every audit letter-slip fixed; the moved numbers disclosed
  (H2 clustering, bcc label, ex-ORB guardrail, R3's ORB premises); no [PENDING], no internal
  checklist, no DRAFT banner.
- [ ] L2 R1.3 paragraph rewritten neutrally (transparent review), ACR/AEM considered on merits.

## Phase 6 — Package (~10-03 → 10-05)

- [ ] P1 Clean manuscript.docx + supplementary.docx from the .md (figures embedded at 600 dpi).
- [ ] P2 Marked-changes manuscript against `manuscript-as-reviewed.md` (and ESI).
- [ ] P3 TOC graphic regenerated from the ledger (no ranking claim) + blurb ≤ 250 characters.
- [ ] P4 CRediT roles; AI-use statement; affiliation; corresponding e-mail per Frank.
- [ ] P5 DAS names the new scripts/outputs; merge `rsc-figure-fixes` → `main`; tag; new Zenodo version.
- [ ] P6 Short cover note to Dr Rhyman; response letter exported to .docx/.pdf; stale PDF/cover
  letter quarantined.

## Phase 7 — Verification before done (~10-05 → 10-06)

- [ ] V1 Fresh adversarial re-audit of the final package against the verbatim reports.
- [ ] V2 Every number in manuscript + ESI + letter traced to the ledger or a deposited JSON.
- [ ] V3 Built .docx files re-scanned for stale phrases and the old title.

## Phase 8 — Frank (~10-06 → 10-09)

- [ ] Read-through; confirm AI-use wording and e-mail; upload; opt in to transparent review; link
  ORCID. If anything slips past 9 Oct, e-mail the editor before 9 Oct.

---

## Standing traps for this repo

- `disp` is not in the unit hash; sweep rows must use method `harmonic_dispsweep`.
- The matched set's `str.contains("bcc")` also drops `agi_bcc` (n = 15). Documented; do not "fix".
- `h3_ensemble_guardrail` defaults to `method="tdep"`, for which no rows exist.
- Regenerate through `analysis.canonical()`, never raw `load_ledger()`.
- Unit-level p-values on the clustered sets are not quotable as evidence (R3.3); every new test
  clusters by system.
- ACCESS CHE260157 is STS-only: never for this paper.
