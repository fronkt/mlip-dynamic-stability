# Pre-registration: does fine-tuning toward PBE remove the finite-temperature mis-calls? (2026-10-03)

Committed before any training data is computed, any model is fine-tuned or any result is seen. The
commit hash of this file is the registration timestamp. Deviations are allowed only for crashes or
tool failures, and every deviation is recorded in a "Deviations" section appended below, with the
reason and the date, never by editing the sections above.

## Motivation

Referee 2 (RSC Advances RA-ART-07-2026-006452) objected that the paper does not address
fine-tuning. The revision's own results make a falsifiable prediction about it. The PBE check
(§3.2, ESI §S5, Table S19) found that the shared 300 K screen mis-calls on BaTiO₃ and KNbO₃ are
softened MLIP wells (0.32–0.76 of the PBE depth), while those on CsSnBr₃ persist on PBE energies.
The paper also argues (§3.3) that SSCHA's low-temperature false-stables come from its local
criterion, not from the surface. If both readings are right, fine-tuning toward PBE should fix the
first, leave the second and the third alone, and not break stable controls.

## Design

- **Base models (2):** MACE-MP-0 (medium, the checkpoint of the paper) and CHGNet 0.4.2. Both are
  among the four models that mis-call BaTiO₃ and KNbO₃ at 300 K, and both have documented
  fine-tuning paths in the pinned environments.
- **Systems (3 test + controls):** BaTiO₃ and KNbO₃ (predicted fixed), CsSnBr₃ (negative control:
  predicted not fixed). Forgetting control: the harmonic calls of the six stable control systems.
- **Replicates:** 3 independent fine-tunes per base model (seeds 0, 1, 2), identical data and
  settings. A call counts as changed only if all three replicates agree; otherwise it is reported
  as unresolved.
- **Training data (PBE, same settings as §2.6):** per test system, 40 phonon-rattled 2×2×2
  supercells (40 atoms), drawn at 100, 300 and 600 K (13, 13 and 14) from the base model's
  harmonic force constants with |ω| for imaginary modes, the generator of the force-spread study
  (C2). One training set per system and base model, pooled over the three systems for a single
  fine-tune per replicate. Random 90/10 train/validation split by configuration, fixed seed.
- **Leakage guard:** no training configuration may lie within 0.05 Å RMS displacement of any C3a
  path geometry or C3b configuration of the same system and model; violators are redrawn.
- **Held-out test set (never used for training, validation or early stopping):** every PBE C3a
  path point (the 448 analysed on 2026-10-03 plus the 533 extra-mode points run on 2026-10-03/04)
  and every C3b configuration.
- **Training settings:** each tool's documented fine-tuning defaults for a small dataset, fixed
  before training and recorded in the run config; a fixed epoch budget; no hyperparameter search;
  model selection only on the validation split.

## Outcomes and predictions

- **P1 (surface, primary).** Along the held-out deciding paths of BaTiO₃ and KNbO₃, the median
  ratio of fine-tuned to PBE well depth over paths and replicates. Prediction: within [0.8, 1.2]
  for both base models (base models: 0.32–0.76).
- **P2 (screen calls, primary).** The full harmonic → soft-mode screen pipeline re-run with each
  fine-tuned model on the three systems at 100/300/600/900 K, scored against the labels.
  Prediction: the BaTiO₃ and KNbO₃ 300 K mis-calls are corrected (2 systems × 2 models); the
  CsSnBr₃ 300 K and 600 K mis-calls persist; KNbO₃ 600 K persists (it persists on PBE).
- **P3 (SSCHA criterion, primary).** Converged SSCHA (the grid recipe, start A) on BaTiO₃ at 100 K
  with the fine-tuned MACE-MP-0, seed-0 replicate. Prediction: still called stable against the
  unstable label, because a curvature read at the symmetric reference is local (§3.3).
- **S1 (secondary).** Force and energy RMSE against PBE on the held-out C3a points and C3b
  configurations, base against fine-tuned.
- **S2 (secondary, forgetting).** Harmonic calls of the six stable controls: predicted unchanged.

## Analysis

Descriptive: counts per (system, model, T) with all three replicates shown; no significance test
(three systems cannot support one). A prediction is supported, refuted or unresolved as defined
above; all three outcomes are reported in the paper and the response letter whichever way they
fall. No result is dropped. If a fine-tune fails to train (divergence, NaN), it is re-run once
with the same settings and a new seed, and the failure is recorded.

## Deviations

Recorded 2026-10-03, after the tooling was built and BEFORE any training or test data was computed
(no ft_ pw.x job had run; no model had been fine-tuned).

- **D1 Contact redraw.** The C2 generator with |ω| for imaginary modes and its 0.5 THz floor gives
  physically impossible draws for very soft spectra: CsSnBr₃ at 300/600 K (both base models;
  closest atom pair down to 0.28 of its equilibrium distance, up to 13 of 14 draws below 0.75) and
  KNbO₃/CHGNet at 600 K (down to 0.40). PBE on overlapping atoms is not a training label for the
  surface the screen reads. Draws whose closest pair falls below **0.6** of its equilibrium
  distance are redrawn (`configs --min-pair-ratio 0.6`); the acceptance fraction per set is
  recorded in manifest.json (8.5 % for CsSnBr₃/CHGNet at 600 K, the lowest). The training set is
  therefore milder than the literal draws for those sets; this is stated in the paper.
- **D2 Epoch budget.** "Each tool's documented fine-tuning defaults" gave 6 epochs (MACE naive
  fine-tuning) and 5 (CHGNet notebook), about 324 optimiser steps for MACE on this dataset. The
  budget is fixed at **30 epochs for both**, with checkpoint selection on the validation split only;
  every other setting is the documented one (scripts/finetune/*.json record both values).
- **D3 Implementation details (consistent with the design, listed for completeness).** MACE keeps
  all foundation elements (`--foundation_model_elements=True`) so that S2 can run; one energy
  offset per system from training configurations only; stress not trained (no stress labels);
  training cells are each base model's own relaxed lattice, as the held-out cells are; the
  undisplaced cells are excluded from training (they coincide with held-out Q = 0 frames); S1 on
  C3b covers BaTiO₃/MACE-sampled configurations only, the only C3b set of a test system.
