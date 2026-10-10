# Revision 2026-09-10 — estimator-noise treatment and the association restatement

Status: **APPLIED 2026-09-10.** All three changes are now in `manuscript.md`; the ESI
addition is `paper/supplementary.md` §S1.2; both incidental bugs are addressed (the
`analysis.summary()` fix is committed, the `agi_bcc` exclusion is now stated explicitly in
§3.2 rather than silently applied). DOCX rebuilt. `scripts/verify_claims.py` still passes
27/27 — the revision withdrew an *interpretation*, not any measured number.

Retained as the rationale record for the changes; see the git history for the diffs.

Source of numbers: `scripts/estimator_noise.py` → `results/estimator_noise.json`
(zero compute; re-reads the deposited ledger; numpy/pandas only, no scipy added).

## Why this revision exists

Dyna-Mat (arXiv:2607.03433) §II.4 found the harmonic-derived κ_SRME correlates only weakly with
finite-temperature observables and attributed that to κ_SRME being a **noisy estimator** —
explicitly *"its sensitivity to the finite displacement used in phonon calculations"* — rather
than to a physical harmonic/finite-T distinction. A referee will apply that directly: *the CHGNet
reordering is estimator noise.* Answering it surfaced two further problems in our own text.

Three changes, in descending order of importance:

1. **§3.2's association claim is restated.** McNemar's exact test measures marginal homogeneity,
   not association; our prose attaches "significantly" to the association reading. Under a
   system-clustered permutation test the association is *not* significant at any temperature.
2. **CHGNet's 0.789 is banded** and the *ordering* is promoted to the claim, because 0.789 holds
   only for tol ∈ [0.05, 0.20].
3. **A measured harmonic noise floor is added**, which answers the Dyna-Mat objection with data
   already in the ledger.

---

## Change 1 — the association restatement (the one that matters)

### What is wrong

`manuscript.md:327-330` currently reads:

> At T = 300 K the relationship reverses and becomes significant: φ = −0.129 with McNemar exact
> p = 0.007, driven by 17 harmonically-correct units that are mis-called at finite temperature
> against only 4 the other way. Harmonic correctness therefore does not transfer, and at the
> higher temperature it is significantly *anti*-associated with finite-temperature correctness on
> this set.

Two defects. (a) **The p-value tests a different hypothesis than the sentence claims.** McNemar's
exact test on the discordant cells tests whether the two layers have equal error rates (marginal
homogeneity). It says nothing about whether being harmonically correct *predicts* being
finite-temperature correct. (b) **The 75 pairs are clustered** — 5 models × 15 systems, each system
contributing 5 correlated rows — and are treated as independent.

Re-tested with a system-clustered permutation null for φ (10,000 permutations of whole systems'
finite-temperature rows, preserving within-system structure; seed 0):

| T (K) | φ | clustered perm p (2-sided) | clustered bootstrap CI95 | both-incorrect cell |
|---|---|---|---|---|
| 100 | +0.1487 | 0.1247 | [−0.082, +0.486] | 1 |
| 300 | **−0.1285** | **0.3051** | [−0.195, −0.057] | **0** |
| 600 | −0.1579 | 0.0831 | [−0.237, −0.077] | 0 |
| 900 | −0.0984 | 0.3584 | [−0.161, −0.039] | 0 |

**No temperature reaches conventional significance for anti-association.** The bootstrap interval
excluding zero is weaker than it appears: it characterises the precision of a point estimate, it
does not test the no-association null. And φ's sign rides on a structurally empty both-incorrect
cell at 300/600/900 K. The model-level Spearman (ρ = −0.667 at 300 K) is computed on n = 5 models
and cannot be significant by construction.

### What survives, and is still a good result

The **marginal** claim is genuinely significant and is the one the data carries: at 300 K, 17
harmonically-correct units are mis-called at finite temperature against only 4 the other way,
McNemar exact p = 0.007. That is exactly "harmonic accuracy does not transfer" — which is the
paper's title claim. Only the stronger *anti-association* reading has to go.

### Proposed replacement — §3.2, replacing `manuscript.md:324-330`

> The matched non-bcc, non-borderline comparison (n = 75 system×model pairs, clustered as five
> models over fifteen systems) is temperature-dependent, and we separate two questions that a
> single 2×2 table invites conflating. The first is whether the two layers are equally hard, a
> question of marginal homogeneity that McNemar's exact test addresses. At T = 300 K they are not:
> 17 harmonically-correct units are mis-called at finite temperature against only 4 the other way,
> McNemar exact p = 0.007. Harmonic correctness therefore does not transfer, and this is the
> significant result. The second question is whether harmonic and finite-temperature correctness
> are *associated*, which McNemar does not test. There the effect is negative but not significant:
> φ = −0.129 at 300 K, with a system-clustered permutation p = 0.31 over 10,000 permutations of
> whole systems' finite-temperature rows, and a system-clustered bootstrap 95% interval of
> [−0.195, −0.057] — an interval that reflects the precision of the point estimate rather than a
> test of the null, and whose sign rides on a structurally empty both-incorrect cell. At T = 100 K
> the association is weakly positive and likewise not significant (φ = +0.149, clustered
> permutation p = 0.12, McNemar exact p = 0.73, matched per-model rank correlation ρ = +0.65).
> With five models and fifteen systems this design cannot resolve the sign of the association, and
> we do not claim it. The transferable finding is the failure of transfer itself.

### Proposed replacement — abstract, replacing `manuscript.md:26-27`

> …while MatterSim is harmonically perfect and only mid-table; on the matched set harmonic
> correctness does not transfer — 17 harmonically-correct units are mis-called at 300 K against
> only 4 the other way (McNemar exact p = 0.007) — although the residual negative association is
> not itself significant once pairs are clustered by system.

### Proposed replacement — intro H2 bullet, replacing `manuscript.md:77-80`

> (The pre-registered form was "harmonic accuracy is uncorrelated with finite-temperature
> accuracy". The matched-set outcome is that harmonic correctness does not transfer —
> significantly so at 300 K, as a difference in layer difficulty — while the direction of any
> residual association is unresolved at this sample size, so the pre-registered form is neither
> cleanly confirmed nor cleanly rejected; see §3.2.)

---

## Change 2 — band CHGNet's harmonic accuracy, claim the ordering

Per-model harmonic accuracy across the imaginary-frequency tolerance (n = 19 scored systems):

| model | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| chgnet | 0.684 | **0.789** | **0.789** | **0.789** | 0.895 | 0.895 |
| mace_mp0 | 0.737 | 0.895 | 0.895 | 0.895 | 0.895 | 0.895 |
| mattersim | 0.684 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| orb_v2 | 0.632 | 0.895 | 0.895 | 0.842 | 0.842 | 0.526 |
| sevennet0 | 0.684 | 1.000 | 1.000 | 1.000 | 0.947 | 0.895 |

CHGNet is uniquely-worst **only for tol ∈ [0.05, 0.20]**. At 0.30 it ties MACE-MP-0 and ORB-v2
becomes uniquely worst. Since `manuscript.md:250-253` already concedes that two of CHGNet's four
harmonic errors are finite-displacement noise, a referee can turn that concession against the
"worst harmonically" phrasing.

The ordering, however, is **never reversed**: CHGNet sits strictly below MatterSim at every
tolerance in [0.05, 0.50], and at tol = 0 the two merely *tie* at 0.684.

### Proposed addition — §3.2, after the CHGNet sentence at `manuscript.md:333-336`

> CHGNet's harmonic accuracy is tolerance-dependent: 0.789 across the plateau tol ∈ [0.05, 0.20]
> used throughout, rising to 0.895 at tol = 0.30, where two marginal false-unstables (CeO₂ at
> −0.267 THz and NaCl at −0.238 THz) flip back and ORB-v2 becomes the uniquely worst model. We
> therefore state the ordering rather than the number as the claim: CHGNet sits strictly below
> MatterSim harmonically at every tolerance in [0.05, 0.50] and above it at finite temperature. At
> tol = 0 all five models collapse to 0.63–0.74 as the Γ acoustic numerical zeros flood the
> false-unstable count, and the comparison is degenerate.

---

## Change 3 — the measured harmonic noise floor (the Dyna-Mat answer)

The v1 and v2 harmonic generations are the *same algorithm at the same settings* — 0.01 Å
displacements, 2×2×2 supercells, 12×12×12 meshes — re-measured in independently pinned
environments (`mlip_dynstab/__init__.py:23-24`). That is 100 paired (system, model)
re-measurements of the estimator, already deposited.

| model | n | median \|Δ\| (THz) | p95 \|Δ\| | max \|Δ\| | call flips |
|---|---|---|---|---|---|
| chgnet | 20 | 5.7e-05 | 3.6e-03 | **0.00755** | **0** |
| mattersim | 20 | 5.4e-05 | 5.4e-04 | **0.00057** | **0** |
| sevennet0 | 20 | 1.7e-05 | 4.8e-04 | 0.00102 | 0 |
| mace_mp0 | 20 | 1.5e-12 | 4.2e-01 | 0.53575 | 0 |
| orb_v2 | 20 | 4.3e-02 | 1.9e+00 | 2.01071 | **2** |

### Proposed addition — new paragraph in §3.2, before the CHGNet demonstration

> Because the reordering rests on harmonic calls, we bound the harmonic estimator's own
> reproducibility noise directly rather than assuming it is small. The v1 and v2 harmonic
> generations (§2.3) are the same algorithm at the same settings — 0.01 Å displacements, 2×2×2
> supercells, 12×12×12 meshes — re-measured in independently pinned environments, giving 100
> paired (system, model) re-measurements. The spread is strongly model-dependent and must not be
> pooled: CHGNet's largest deviation is 0.0076 THz and MatterSim's is 0.00057 THz, with zero
> stability-call flips across all forty of their re-measurements, whereas ORB-v2's reaches 2.01 THz
> and flips two calls, one of them the KTaO₃ unit already excluded as borderline. The two models
> carrying the reordering are therefore separated by two to four orders of magnitude more than the
> estimator's own noise, and what noise exists is localised to the float32 direct-force model that
> §3.1 already identifies as the weakest. This bounds environment and library nondeterminism at
> fixed displacement amplitude; it does not probe sensitivity to the amplitude itself, which would
> require re-measurement on a displacement grid.

### ESI addition

A fuller version belongs in the ESI: the full per-model table above, the two flipped units named
(`ktao3_cubic`, excluded as borderline; `hf_bcc` at −0.0891 → −0.1937 THz), and the honest caveat
that v1/v2 share code path, displacement and a seedless algorithm, making this a **lower bound**
on estimator noise.

---

## Still open — the one thing this does not cover

Dyna-Mat names **finite displacement** specifically, and Change 3 holds `disp` fixed at 0.01 Å. The
on-target answer is a displacement sweep, which needs compute:

- `disp` is hardcoded at `harmonic.py:40` and is **not in the unit hash** (`cli.py:54`). It must be
  added to the `settings` dict with a `METHOD_VERSION` bump, or `has_unit()` silently skips every
  re-run unit.
- Cost: 95 scored units × 4 amplitudes ≈ 400 units at 0.31–9.70 s each. Suggested grid
  0.005 / 0.01 / 0.02 / 0.03 Å. A couple of dollars on a rented box; there is no local GPU.
- Verdict: **worth doing if the schedule allows.** Without it the rebuttal is strong but
  incomplete on the precise axis named. With it, the objection closes completely.

## Two incidental bugs found

1. **`analysis.summary()` (`analysis.py:401-402`) calls `load_ledger()` instead of
   `load_canonical()`**, so it reports `n_rows = 1809` and double- or triple-counts every table.
   Nothing in the paper pipeline uses it — `verify_claims.py:46` and `make_figures.py:25` both wrap
   correctly — but `python -m mlip_dynstab.analysis` hands a reviewer corrupted numbers. For a
   paper whose selling point is audit discipline this is worth the one-line fix.
2. **The matched set's `str.contains("bcc")` also drops the superionic `agi_bcc`**, which is why
   n = 15 systems rather than 16. Almost certainly unintended. Changing it would move the published
   φ/McNemar numbers, so either keep it and state the exclusion explicitly, or fix it and re-run —
   but do not leave it undocumented.
