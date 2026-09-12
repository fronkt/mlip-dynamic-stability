# RSC Advances major revision — plan

**Manuscript** RA-ART-07-2026-006452, *Finite-temperature dynamic stability is a blind spot of
foundation machine-learning interatomic potentials* (submitted title; the working title has since
changed — see D1 below).
**Decision** Major revision, 2026-09-11, Assoc. Ed. Dr Lydia Rhyman. Three referees: R1 major,
R2 reject-leaning, R3 minor.
**Deadline** None stated ("as soon as possible", automatic reminders).
**Predecessor** `tasks/todo-archive-2026-09-11.md` (superseded Aug-17 snapshot).

---

## D. The three disclosures that must lead the response

These are not referee items. They are things the referees do not know, and burying any of them
would be worse than any criticism in the reports.

- [x] **D1 — The referees reviewed a superseded version.** They read commit `76d3a84` (submitted
  2026-07-13). In August an internal audit found the deposited data was not reproducible by the
  deposited code (a stale q-search grid skipped because the unit hash carried no algorithm version;
  two inverted acoustic masks that deleted the very instabilities under test). Everything was
  re-measured in pinned environments. **Headline numbers moved, and they moved against the paper:**

  | Quantity | As reviewed (Jul) | Now | Direction |
  |---|---|---|---|
  | FE-perovskite screen recall | 23/30 = 0.77 | 16/30 = 0.53 | **worse** |
  | SSCHA FE recall | 7/30 = 0.23 | 5/27 = 0.19 | worse |
  | CHGNet finite-T accuracy | 0.933 | 0.867 | worse |
  | MACE-MP-0 finite-T accuracy | 0.933 | 0.833 | worse |
  | SevenNet-0 finite-T accuracy | 0.867 | 0.900 | better |
  | ORB-v2 finite-T accuracy | 0.700 | 0.800 | better |
  | ORB-v2 harmonic accuracy | 0.842 | 0.895 | better |
  | Freq-spread AUC | 0.52 | 0.36 | (below chance either way) |
  | Vote-split AUC | 0.75 | 0.76 | ~same |

  The consequence that matters: **in the reviewed version the harmonic leaders were not the
  finite-T leaders — a clean inversion. That inversion is gone.** SevenNet-0 now leads both
  layers. R3's entire reading of H2 ("CHGNet and MACE-MP-0 reaching 0.933 while MatterSim falls to
  0.833") describes a result that no longer exists. Lead with this; the numbers moving against the
  author's own story is the evidence that the correction was honest.
- [x] **D2 — The title has changed.** "Blind spot" asserts nobody is looking, which two 2025/2026
  papers (refs 19, 20) falsify in print. Current title: *"Finite-temperature dynamic stability
  separates foundation MLIPs that harmonic benchmarks rank equally."* R3 independently asks for the
  title framing to be brought into line with §3.2, so this is convergent, but the change predates
  the reports and must be declared as such.
- [x] **D3 — Several referee items were already addressed before the reports arrived**, for
  independent reasons (the August audit and a 2026-09-10 estimator-noise revision): the §2.4
  variational derivation (R1-5), the clustered-statistics correction (R1-6, R3-1, R3-3), and the
  measured harmonic noise floor. Say so plainly with commit references rather than presenting them
  as new responses to review.

---

## Phase A — Foundations (compute-free)

- [x] A1 Archive the superseded `tasks/todo.md` with a provenance header.
- [x] A2 Baseline `scripts/verify_claims.py` — 27/27 pass before any change.
- [x] A3 Snapshot the as-reviewed manuscript to `paper/submissions/rsc-advances-2026-07/` from
  `76d3a84`, so every later diff is against what the referees actually read.
- [x] A4 `scripts/reviewed_version_delta.py` — mechanically extract and compare every headline
  number between the as-reviewed and current versions; emit the D1 table rather than hand-copying
  it. Output `paper/response/number_changes.md`.

## Phase B — Statistics hardening (R3-3, R1-6) — the largest technical phase

New module `mlip_dynstab/stats.py`. No scipy (deliberately absent from the pinned envs); hand-roll.

- [x] B1 Wilson score intervals `wilson(k, n, alpha)`. Every rate in the paper becomes *k/n with an
  interval*, not a bare decimal: harmonic accuracies (currently 1.000 / 0.895 / 0.789 = 19/19,
  17/19, 15/19), finite-T accuracies, per-family recalls, false-stable rates, consensus error rates.
- [x] B2 **System-level permutation test for both AUCs.** R3's sharpest statistical point: the
  n = 60 consensus units are repeated measurements over a much smaller number of systems, the five
  model votes are already collapsed into each unit, so they are not independent. Permute whole
  systems; report a clustered p and a cluster-bootstrap CI for AUC 0.76 and AUC 0.36.
- [x] B3 Demote the five-model Spearman. ρ is computed on **n = 5** (+0.645 at 100 K, −0.667 at
  300 K — it flips sign with temperature, which is itself the argument). Report as descriptive with
  the n stated inline, add a system-level permutation companion, and never let it carry an inference.
- [x] B4 **State the composition of n = 60 and n = 75 explicitly** — the system list and the
  temperature ladder behind each. R3 notes the n = 60 composition "is never stated". Emit as ESI
  tables, not prose.
- [x] B5 Keep McNemar, but ensure no sentence anywhere reads a failure to reject as positive
  evidence. The surviving marginal claim (17 vs 4, p = 0.007) is a difference in layer difficulty
  and must be labelled as such every time it appears.
- [x] B6 ~~Factor the clustered machinery out of `scripts/estimator_noise.py:184-256` and have it
  import from `stats.py`.~~ **Deliberately NOT done, after checking.** The deposited
  `results/estimator_noise.json` is byte-reproducible from its own seed (verified at all four
  temperatures), and its numbers are quoted in ESI §S1.2. That script shares one RNG across the
  temperature ladder while `stats.py` seeds per call, so importing would silently move published
  p-values and interval endpoints for no gain. Both docstrings now record the duplication and the
  reason. If ever unified, re-deposit the JSON and re-propagate the ESI in the same commit.
- [x] B7 Emit `results/stats_hardening.json`; extend `verify_claims.py` to pin every new number.

## Phase C — ORB-v2 with/without split throughout (R3-5)

- [x] C1 `analysis.py` contains no ORB handling at all — every split in the repo is ad hoc at the
  call site. Add an `exclude_models` parameter to `per_model_table`, `low_t_false_stable`,
  `displacive_recall`, `h3_guardrail_summary`, `sscha_reliability`.
- [x] C2 Recompute every headline rate with and without ORB-v2; paired columns in the ESI, compact
  in the main text.
- [x] C3 **Disclose the ρ discrepancy found while planning:** §3.3 quotes the bcc frequency
  Spearman ρ = 0.11, which is the all-model value; excluding ORB-v2 it is ≈ −0.003. The manuscript
  reports the sign-agreement split (0.78 / 0.83) but not the ρ split. Report both.

## Phase D — ESI rebuild (R3-4)

- [x] D-1 **Generate the missing tables.** Confirmed: §S3 of the ESI is thirteen lines of bullets,
  each ending in a Python function call, and contains no table content at all. All four generator
  functions exist and run (`per_model_table` :96, `low_t_false_stable` :115, `predicted_tstar` :148,
  `h3_ensemble_guardrail` :337). This is a rendering omission, not a data gap. New script
  `scripts/build_esi_tables.py`.
- [x] D-2 **Fix the S-numbering collision.** The ESI numbers its *sections* S1–S4 and its *tables*
  S1–S4. A referee scanning for "Table S1" lands on "§S1 Finite-T method development". Give every
  table a number and caption and disambiguate section references in text.
- [x] D-3 Number and caption the three tables that *do* have content but are currently bare
  (§S1.2 noise floor, §S2.2 v4 diagnostic, §S2.3 reliability by family).
- [x] D-4 Correct the Table S2 caption: it promises "over the full T-ladder" but the call given
  (`low_t_false_stable(df, exclude_bcc=...)`) takes the `t_max = 300.0` default. Report both the
  T ≤ 300 K restriction and the full ladder.
- [x] D-5 Fold in the Phase B4 composition tables and the Phase C ORB tables.
- [x] D-6 Fix the `ordering_invariant: false` flag in `results/estimator_noise.json`, which sits
  next to prose claiming the ordering is never reversed. Both are true (at tol = 0 the models tie,
  so a strict `<` fails) but a referee reading the deposited JSON sees a contradiction.

## Phase E — Complete the SCHA derivation in the ESI (R1-5)

The derivation exists but is in the main text (§2.4, ~24 lines) and covers only one of R1's three
asks properly.

- [x] E1 Mass-weighting: `M = Σᵢ mᵢ|uᵢ|²` is currently a bare formula, and the normalisation
  convention for **u** (unit pattern vs mass-weighted eigenvector) is never pinned. Derive it and
  pin the convention.
- [x] E2 Gaussian-width self-consistency: substantially complete, but the stationarity step is
  asserted rather than shown and the bracketing interval is unspecified. Show both.
- [x] E3 Mode–mode coupling: currently one table cell (Table 1 row A1). Give an explicit statement
  of the neglected term and an estimate or bound of its size — this is the approximation R1 is
  actually worried about, and PbTiO₃'s T* failure is the measured symptom.
- [x] E4 Systematic sensitivity table: fit window (already measured, 3×/8× vs production 5×),
  sampling range, cell commensurability — collected in one place as R1-5 asks.

## Phase F — SSCHA convergence diagnostics (R1-4, R3-2)

- [x] F1 Report the diagnostics **already in the ledger** — `ft_n_configs`, `ft_max_pop`,
  `ft_n_hessian` per unit — as an ESI table, plus the stopping criteria and the per-population step
  cap (§S2.1). Much of R1-4 needs no new compute.
- [ ] F2 Extend the four-seed stochastic test beyond bcc-Zr to BaTiO₃ and one fluorite, as R1-4
  explicitly asks. Cheap MLIP compute.
- [x] F3 Zone-boundary convergence: **take R3's offered alternative and narrow the claim**, which
  it explicitly accepts, as the primary route. The 4×4×4 SrTiO₃ SSCHA (~320 atoms) is a bonus if
  the measured cost turns out tolerable — decide from a costed pilot, not from optimism.

## Phase G — Referee 2's two asks (the reject-leaning report)

- [x] G1 **Fine-tuning.** The word appears nowhere in the manuscript — R2 is simply right. New
  scoped subsection: the paper tests foundation models *as shipped*, which is precisely the
  generative-CSP screening regime where no target-specific training data exists; fine-tuning is the
  obvious remedy and its absence is a scope boundary, not an oversight. Cite the relevant work.
- [~] G2 **Ensemble force uncertainty.** Two-part answer. (i) §3.4 *already* tests ensemble
  disagreement as a trust metric and finds the continuous version carries no usable signal
  (AUC 0.36, below chance) while the discrete vote split works (AUC 0.76) — R2 appears to have
  missed this, and it is a direct measured answer to the exact proposal. (ii) The on-target
  addition is the *force-level* version: cross-model force spread on the displaced configurations
  along E(Q). Cheap MLIP compute.

## Phase H — Displacement-amplitude sweep (self-declared open gap)

Not asked for by any referee, but it is the one axis the paper's own revision note names as
uncovered, and it hardens B1–B3 against "your estimator is just noisy".

- [x] H1 `disp` is hardcoded at `harmonic.py:40` and is **not in the unit hash** (`cli.py:54`).
  Add it to the settings dict and bump `METHOD_VERSION`, or `has_unit()` silently skips every
  re-run unit. This trap is recorded; do not walk into it.
- [~] H2 Run the grid 0.005 / 0.01 / 0.02 / 0.03 Å over the 95 scored units.
- [~] H3 Report as an ESI table plus one sentence in §3.2.

## Phase I — Citations (R1-3)

- [x] I1 R1's citation request is **citation stacking**: it names "Adv Energy Mater 2026",
  "Chem Phys Rev 2024, 2025", "PRB 2021", "JPCL 2021", "Acc Chem Res 2026" with no titles and no
  authors. The editor pre-empted it in her own comments ("only include those you think are
  relevant"). Identify what genuinely exists in sparse-GP / on-the-fly / active-learning MLIPs and
  in ML-driven high-throughput screening, and cite only what actually bears on this work.
- [x] I2 **Adversarial verification pass on every candidate** before it enters the bibliography —
  DOI, authors, title, and that the claim attributed is actually the paper's. Prior measured rate
  of unusable LLM-sourced findings on this account: 17%.
- [x] I3 Record the declined suggestions and the reason, for the response letter.
- [x] I4 Fix ref 20, which currently has a title and DOI but no authors.

## Phase J — Text alignment (R3-1, R1-6) — cheapest, highest value

Do this **after** B and C, because the numbers move.

- [x] J1 §4 Discussion still says the results "support H2" and that harmonic accuracy is
  "non-predictive". §3.2 says the association is not significant at any temperature and the design
  cannot resolve its sign. "Non-predictive" asserts a null from a failure to reject. Carry §3.2's
  formulation into §4 verbatim in substance.
- [x] J2 Abstract and §5 Conclusions: same alignment.
- [x] J3 Intro H2 bullet: already hedged; confirm consistency with the final J1 wording.
- [x] J4 **Figure order** (R3-6): Fig. 5 is first cited at line 422 but placed at line 501, after
  Fig. 6 at 491. Move it to just after the Fig. 4 block so placement order matches citation order.

## Phase K — Package and response

- [x] K1 Point-by-point response letter, leading with D1–D3, then R1/R2/R3 in order, every item
  numbered to match the report.
- [ ] K2 Tracked-changes manuscript + clean manuscript, both .docx (RSC requires both).
- [ ] K3 Rebuild figures (600 dpi) and all DOCX; TOC entry (8 × 4 cm, ≤250 characters).
- [ ] K4 CRediT author contributions section; link ORCID at submission.
- [ ] K5 Decide on RSC's transparent-peer-review option — Frank's call, flag it.
- [ ] K6 Mint a new Zenodo version and update the concept-DOI reference.
- [ ] K7 **APC**: confirm whether Purdue's RSC read-and-publish agreement covers Frank as
  submitting author before final acceptance. RSC Advances is fully open access; list APC ~£1,600.

## Phase L — Verification before done

- [x] L1 `verify_claims.py` green (32/32) plus `build_esi_tables.py --check`, including every new assertion from B, C, F, H.
- [ ] L2 Independent read-through: every number in the manuscript and ESI traced to the ledger.
- [x] L3 Adversarial self-review: walk all three referee reports item by item and confirm each
  numbered point has a located, specific response. No item silently dropped.
- [x] L4 Confirm no claim anywhere reads a non-significant result as positive evidence.

---

## Compute budget (decided 2026-09-11)

No new DFT. R1-1 and R1-2 are answered from published first-principles results plus the
experimental gate the design already uses. Cheap MLIP compute (~$20–60, rented box; no local GPU,
8 CPUs) covers F2, G2(ii) and H2. ACCESS CHE260157 is **STS-scope only and must not be used for
this paper.**

## Standing traps for this repo

- `disp` is not in the unit hash — bump `METHOD_VERSION` or re-runs are silently skipped (H1).
- The matched set's `str.contains("bcc")` also drops the superionic `agi_bcc`, so n = 15 systems
  not 16. Documented in §3.2, deliberately not fixed; do not "fix" it without re-running everything.
- `h3_ensemble_guardrail` defaults to `method="tdep"`, for which no rows exist.
- Figures and claims must regenerate through `analysis.canonical()`, never `load_ledger()` raw.
