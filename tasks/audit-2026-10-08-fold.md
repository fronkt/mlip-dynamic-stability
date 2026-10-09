# Audit of the 2026-10-08 fold (commits 67d9bdd..a9fa30a)

Adversarial check of the converged-grid, E1/E2 and R1.3 fold into `paper/manuscript.md`,
`paper/supplementary.md`, `paper/response/response_to_referees.md`, `scripts/verify_claims.py`,
`scripts/grid_compare.py` and `scripts/build_esi_tables.py`. Read-only on paper/, scripts/ and
results/. The PENDING-E3/E5/PL/P4/P5 markers were not scored.

**Method.** Independent scripts (scratchpad `audit/grid_audit.py`, `prod_audit.py`, `fs57.py`,
`changed.py`, `quotes.py`) read the raw per-unit JSONs in `results/revision/sscha_converged_grid/`
(not `summary.csv` or `grid_compare.json`), labels from `configs/curated_systems.yaml` (stable iff
T ≥ T_c), screen calls from `ledger.parquet` (softmode v4) and production SSCHA from the ledger
(sscha v3, `ft_sscha_min_freq_thz`, unstable below −0.1 THz, blow-up |f| > 50 THz). DFT numbers
were recomputed from `dft/c3a_*.csv`, `dft_checks/*.csv` and `summary.json`. Wilson intervals were
recomputed (z = 1.95996). References were checked on Crossref and OpenAlex (DOI lookups only).

**Bottom line.** All the headline numbers are reproduced exactly by code that shares nothing with
the pipeline. The problems are in the wording and in one rendering bug: 3 high, 5 medium, 11 low.

## High

**H1. ESI Table S19 caption prints a raw template field.** `paper/supplementary.md:1146` reads
"Each of the {tot[1]} ladder units". `scripts/build_esi_tables.py:1300`: the string lost its `f`
prefix when "C3a)." was cut in eeb10aa. The 67d9bdd version (line 1252) was an f-string.
`build_esi_tables.py --check` still passes ("up to date") because it compares the generated text
with itself, so the check cannot catch this. *Fix:* restore the `f` prefix and rebuild. A full scan
of the ESI for unformatted `{name}` fields finds no others.

**H2. The abstract's recall sentence lost its convention qualifier and does not give the paired
test.** `manuscript.md:25-26`: "At T ≤ 300 K the screen recovers 15/26 [0.39, 0.74] ... against
4/26 [0.06, 0.34] for converged SSCHA." The 67d9bdd abstract opened this sentence with "Under its
production convention". §3.2 says the screen's recall runs from 5/30 to 28/30 across conventions.
The two non-overlapping Wilson intervals read as a significant difference, but the paired
system-clustered test gives p = 0.5, and also p = 0.5 on the three ferroelectric systems alone
(recomputed). The Fig. 5 caption itself warns that "two marginal intervals are not that test".
Letter R1.5 (`response_to_referees.md:562-563`) says "the Abstract ... say so" about the
convention. For the recall, the abstract no longer does. *Fix:* restore "Under its production
frozen-cell convention" and add "(paired, system-clustered p = 0.5)". Words can be found by
trimming the 7–24 parenthesis, since the abstract is at 248/250.

**H3. The claim that the production recipe "never converged" is broader than the evidence, and the
ESI contradicts it.** The claim appears at `manuscript.md:128` (Intro), `:724` (§3.3) and `:1277`
(§5: "The default recipe as first run never converged"), and in the letter at
`response_to_referees.md:32`. What was measured: 0 of 16 seeds on 4 units, and the production grid
did not record convergence (§2.5). ESI §S2.1 (`supplementary.md:451-453`) reports two
production-recipe runs (bcc Zr/MACE-MP-0, 3×3×3, 100 and 300 K) that do meet the test. *Fix:* "did
not converge on any of the 16 recorded seeds; the cumulative cap ends the relaxation after about
19 steps wherever the start is far from the minimum". The fluorite and high-T statements can keep
their basis, which is the unit-by-unit comparison with the converged grid.

## Medium

**M1. "34 of 84" is attributed to the ferroelectric perovskites, and "still" is inaccurate.** The
sentences are at `manuscript.md:20-22` (abstract: "still calls the ferroelectric perovskites
stable ... in 34 of 84 units"), `:125-127` (Intro) and `:1272-1273` (§5: "remain on the
ferroelectric perovskites (34 units)"). Recomputed: the 34 are BaTiO₃ 8, KNbO₃ 15, PbTiO₃ 7
(300/600 K) and SrTiO₃ 4 (100 K, antiferrodistortive). The 84 is every label-unstable converged
non-bcc unit, which includes 38 fluorite and 8 CsSnI₃ units where SSCHA has 0 false-stables.
Recomputed by system: oxide perovskites 34/38, ferroelectric oxides 30/34. Only 21 of the 34 were
production false-stables. The other 13 are new under convergence (PbTiO₃ 7, SrTiO₃ 4, KNbO₃ 2), so
"still" holds for 21. §3.3:863 and §4:1125-1126 say it correctly ("ferroelectric or
antiferrodistortive"). *Fix:* use "the oxide perovskites, 34 of their 38 label-unstable units
(30/34 ferroelectric)" or "34 of 84 label-unstable non-bcc units, all oxide perovskites", and drop
"still" or make it "21 of them already in the first recipe".

**M2. The letter says 34 false-stables survive.** `response_to_referees.md:451-452`: "the
false-stables survive on BaTiO₃ and KNbO₃ (34/84 label-unstable units in all, with PbTiO₃ at
300–600 K and SrTiO₃ at 100 K)". 21 of the 57 survive, and the PbTiO₃ and SrTiO₃ false-stables are
new. ESI S22 (`supplementary.md:1305`) has it right (21/32/4). *Fix:* "21 of the 57 survive, all
BaTiO₃/KNbO₃; convergence adds 13 (PbTiO₃ 300–600 K, SrTiO₃ 100 K, two KNbO₃), 34/84 in all".

**M3. A conclusion rests on the unconverged ensembles without saying so.** `manuscript.md:1281-1282`
(§5): "PBE forces on the sampled configurations show no extrapolation error behind the
low-temperature false-stables". Those configurations come from the production BaTiO₃ and ZrO₂
100 K ensembles. ZrO₂ is no longer a false-stable once converged, and the converged BaTiO₃ matrix
moved from +2.87 to +2.03 THz. §3.3:914-915 and §4:1229 hedge this, but §5 does not. *Fix:* add
"on the ensembles of the unconverged recipe (BaTiO₃ and ZrO₂, 100 K)", or drop the clause.

**M4. A production-only bcc inference is stated as fact, and the converged grid falsifies it for
two models.** `manuscript.md:738-741`: "All five models are already positive at 50 K ... any
dynamic-stabilisation temperature is left-censored at 50 K. It lies far below the thermodynamic
transition because bcc to hcp/ω is a martensitic ... transition". Converged 3×3×3: MatterSim is
unstable at 100 and 300 K on Ti, Zr and Hf (Hf still −0.57 THz at 600 K), and ORB-v2 on Zr at 100
and 300 K. In those units the stabilisation temperature is 300–600 K or higher, not ≤ 50 K. The
next paragraph reports the change, but this sentence still reads as a finding. *Fix:* prefix "With
the production recipe in the 2×2×2 cell", and drop or qualify the causal "because" clause.

**M5. The new verify_claims checks mostly pin pipeline outputs.** They compare literals with
`grid_compare.json["...short"]` strings and with `summary.csv` columns (`stable_call`,
`gt_stable`, `screen_stable_call`, `converged`). Both files are pipeline products, so these are
regression pins, not re-derivations. The "both self-checks pass" check is also internal to
grid_compare. No check ties the manuscript text to the numbers: §3.2's "median 0.53" (L1) passes
under a ±1e-3 tolerance around 0.525. The §3.2 "seven of the eight" PBEsol claim is not tested
(only 8/9 is). The "ferroelectric" label on the 34 (M1) is not tested. Run result: 119 PASS, 0
FAIL. No number is wrong, because the independent recomputation above matches every one. *Fix
(optional):* derive the grid checks from the per-unit JSONs plus the yaml labels, as the
scratchpad script does, and add text-match checks for the abstract numbers.

## Low

- **L1** `manuscript.md:646`: "(median 0.53)" for the 26 BaTiO₃/KNbO₃ paths. Actual median is
  0.5247, which rounds to 0.52. The 12 deciding paths give 0.5255, so 0.53 at `:624` is right.
- **L2** `manuscript.md:421-422`: the list of units scored on MatterSim's deciding coordinate
  omits ORB-v2 on bcc Zr. In `c3a_unit_calls.csv` there are 5 reference-coordinate units: Zr for
  CHGNet, MACE-MP-0, SevenNet-0 and ORB-v2, plus SrTiO₃ for ORB-v2. ESI S19 says "every model
  except MatterSim".
- **L3** `manuscript.md:923-924` says "the 2 ORB-v2 units that did not converge are unstable too",
  but ZrO₂/ORB-v2/100 K (−50.02 THz) is listed as a blow-up at `:955`. HfO₂/ORB-v2/100 K is
  −49.4 THz.
- **L4** `manuscript.md:763-764` and `:1072`: the 300 K Zr/MatterSim pair "isolates the cell ...
  converged from two starts". The 2×2×2 start A fails the fresh-ensemble check (AB verdict: "NOT
  shown consistent with a minimum for A").
- **L5** `manuscript.md:875-879`: "survives a converged relaxation in every model ... early
  stopping is excluded". Each unit has one start and one seed (only BaTiO₃/MACE-MP-0 has two
  starts), and BaTiO₃/CHGNet/100 K is +1.65 ± 1.50 THz (bootstrap SD), within 2 SD of the
  boundary, as ESI S22 notes.
- **L6** `response_to_referees.md:449`: "change 72 of the 159 calls". The 72 are counted over the
  154 units with a non-blow-up production value. Counting production blow-ups as unstable gives
  75 of 159. The non-bcc 51/12 split is consistent with 72.
- **L7** `manuscript.md:629-631` gives "seven of the eight persistent CsSnBr₃ errors" (MLIP and PBE
  both wrong). The letter at `:206` gives "8 of the 9 CsSnBr₃ PBE errors". Both are correct;
  harmonise or name the denominator.
- **L8** `manuscript.md:1225`: CsSnBr₃ wells are called "comparable to PBE's". That holds for 3 of
  5 models (0.92–1.17). CHGNet's well is 2.17× PBE's and ORB-v2's is 0.57–0.68.
- **L9** Ref 15, `manuscript.md:1399`: Crossref has given name "Soohaeng Yoo", family "Willow", so
  in RSC style the author is "S. Y. Willow". Otherwise the metadata is correct (10.1063/5.0231265).
- **L10** `manuscript.md:1226`: "101 of 120 units against 81 with the MLIPs". The 81 is the MLIP
  call on the same paths; the ledger calls give 78. Add "along the same coordinates".
- **L11** Fig. 5 caption, `manuscript.md:843`: "the 26 units where converged SSCHA returned a
  value". The two unconverged BaTiO₃ 300 K units also returned values, so say "converged".

## Verified correct (independent recomputation)

**Converged grid**
- 159/178 = 0.893 [0.839, 0.931]; 171 ok, 7 failed (the same 7 as production), 11 ORB-v2 at the
  population cap, 1 CHGNet at the wall cap.
- Fresh-ensemble check: 152/159 consistent, and the 7 that fail match ESI S22.
- 4 blow-ups, all unconverged ORB-v2; 0 among converged units.

**False-stables and recall**
- 34/84 [0.306, 0.512] (30/74); 17/34 [0.341, 0.659] (14/30).
- Breakdown: BaTiO₃ 8, KNbO₃ 15, PbTiO₃ 7, SrTiO₃ 4. All ten BaTiO₃/KNbO₃ units at 100 K are
  screen-unstable.
- Fate of the 57 production false-stables: 21 still stable, 32 now unstable, 4 unconverged;
  11/21 screen-unstable; 18/50 without ORB-v2. 46/57 reproduced.
- Ferroelectric recall: 4/26 [0.062, 0.335], 15/26 [0.390, 0.745]; 4/23 and 12/23; the four
  SSCHA hits are PbTiO₃ at 100 K (−0.12 to −0.37 THz). Screen 16/30. Production 5/27.

**Paired tests**
- 13 v 2 on 44 units (10 v 2 on 39); system nets +5, +6, 0, 0, 0; exact sign-flip p = 0.5
  (ferroelectric systems only: 0.5).
- Production: 33 v 3 on 47, p = 0.125.

**High temperature, SrTiO₃, fluorites, CsSnI₃**
- Converged false-unstables 0/4, 3/12, 3/18 (0/4, 3/11, 3/16), all CsSnI₃ (−0.32 to −1.09 THz).
- Production 5/5, 10/13, 15/19 (4/4, 8/11, 12/16); 14 of 23 turn negative; 14 SrTiO₃ units at
  −3.4 to −914 THz.
- SrTiO₃ 100 K: +1.06 to +1.20 THz (4 models). Above 105 K: +1.71 to +2.76 THz (0/11).
- 14 non-bcc units stable at 100 K, none turning negative.
- Fluorites: 38/38 converged unstable, 18/18 at T ≤ 300 K. Production 19/20, 10/10 at 100 K
  (+1.90 to +3.33 THz), 31/40.
- CsSnI₃: 8/8 (production 7/8).

**bcc**
- Agreement 33/41 [0.660, 0.898] (32/36); MatterSim 7/9.
- Production 31/45 [0.543, 0.805] (25/36).
- Ti/MACE-MP-0: +1.57, +1.66, +1.71 THz.

**PBE (E1)**
- 101 v 81 of 120 on the same paths: 23 corrected (10 bcc Zr), 3 newly wrong, 5 systems improve.
- 93/139 paths (8 symmetry copies, 38 larger cells; extra modes ≤ 12 atoms); the 55 extra modes
  change no call on PBE or on the MLIPs.
- 12 deciding paths 0.315–0.763 (median 0.526); 26 paths 0.315–0.828.
- ZrO₂ 0.45–0.61; SrTiO₃ 0.25–0.99; bcc Zr PBE wells 159–174 meV.
- 981 = 380 + 533 + 68 (ft_ 246 is E5).

**DFT checks (E2)**
- PBEsol: 9 of 15 errors fixed, 0 new; CsSnBr₃ 8/9 (7/8 persistent), 4 of the 5 at 300 K, every
  one at 600 K.
- CsSnBr₃ PBEsol/PBE depth 0.32–0.42.
- k-mesh −6.8 %; 323 jobs; MLIP lattices 0.1–0.8 % above PBE.

**References**
- 11, 12, 14 and 15 exist, and the claims attributed to them match the abstracts. 12:
  10.1021/acs.jpclett.1c01605, JPCL 12, 8115–8120. 14: Acc. Chem. Res. 59, 103–113.
- Citation order is monotone (1–48, none missing). The letter's ref numbers match the new list.
  18 → 48 entries.

**Letter quotes**
- All 17 block quotes are verbatim against the reports (whitespace-normalised).

**Style**
- No em dashes in added prose (two pre-existing in §2.4/§3.2 at `:277` and `:664`).
- No plan codes in prose.
- Abstract 248 words; TOC 247 characters.
- "§" is used throughout. This predates the fold (a pre-existing convention, not a regression).
