# Revision plan — August 2026

Companion to `tasks/audit-2026-08-16.md`. That file records what is broken. This one records what
the paper should *say* instead. Every quotation below was retrieved and read in this session, not
recalled.

---

## 1. The literature settles §3.3, and mostly in our favour

### 1.1 Bianco, Errea, Paulatto, Calandra & Mauri, *Phys. Rev. B* **96**, 014111 (2017)
arXiv:1703.03212 — *"Second-order structural phase transitions, free energy curvature, and
temperature-dependent anharmonic phonons in the SCHA: theory and stochastic implementation."*

- **"By definition, the SCHA matrix Φ is positive-definite"** — required for the trial Gaussian
  ρ to be normalisable.
- **"D_ab^(s) is positive-definite, thus it is impossible to observe any softening in its
  eigenvalues."**
- The free-energy Hessian, their Eq. (27), is the correct object:
  ∂²F/∂R∂R = **Φ** + **Φ⁽³⁾Λ**[**1** − **Φ⁽⁴⁾Λ**]⁻¹**Φ⁽³⁾**
- Dropping Φ⁽⁴⁾ (the "bubble") is valid only "in the limiting case where the absolute values of Ξ's
  eigenvalues are much smaller than one."
- Demonstrated on **ferroelectric transitions in rock-salt crystals such as SnTe or GeTe**.

### 1.2 Monacelli, *Phys. Rev. B* **112**, 014109 (2025)
arXiv:2407.03090 — *"Simulating the anharmonic phonon spectrum in critical systems."*
Benchmarked on **CsSnI₃ — a system in our own curated set — and PbTe.**

- **"looking for sign changes in eigenvalues of Φ(S) and Φ(T) is trivially wrong, as it is easy to
  prove that both Φ(S) and Φ(T) are positive definite and, when properly converged, never display
  negative eigenvalues"**
- The bubble approximation **"breaks down in 2D materials and the cubic phase of metal halide
  perovskites, producing qualitatively incorrect results."**
- **"the fourth order Φ⁽⁴⁾ in these systems plays a significant role around the tetragonal-to-cubic
  phase transition, and the complete RPA-like resummation is needed to determine the phase
  stability."**
- **"SCHA free energy Hessian can have imaginary frequencies ... SCHA can be identified directly
  from the high-symmetry phase by looking for the crossing point when a phonon band changes from
  imaginary to real."**

### 1.3 What this kills, and what it hands us

**Dies.** Three sentences in §3.3 cannot survive a referee who knows this literature:

1. *"`ForcePositiveDefinite` at initialisation erases it (+2.88 THz)"* — presented as a defect. It
   is the definition. Φ must be positive-definite or the theory has no normalisable trial state.
2. *"the SCHA auxiliary matrix then stays at +2.89 THz **because at low temperature its narrow
   Gaussian width never samples the anharmonic double well**"* — the mechanism is simply wrong. Φ
   stays positive because it is **provably impossible** for it to soften, at any temperature and any
   width. Reading the auxiliary matrix for instability is, verbatim, "trivially wrong."
3. The **"trap" / "the gold standard is itself a methodological trap"** framing, including the title
   of §3.3 and the corresponding sentences in the Abstract, Discussion and Conclusions.

**Survives, and is now corroborated rather than merely asserted.** Our sentence *"Enabling the
fourth-order term (`include_v4=True`) is the only route that could recover the instability"* is
**correct**, and Monacelli says so independently: the complete RPA-like resummation is needed to
determine phase stability in exactly these systems. (An earlier reviewer note claiming the SSCHA
documentation calls Φ⁽⁴⁾ negligible was checked and is **wrong** — do not act on it.)

**Handed to us.** Monacelli's last quote validates our entire experimental design: the free-energy
Hessian *can* go imaginary and the instability *is* meant to be found from the high-symmetry phase.
So the design is right; only the v4 truncation is at fault. That converts our result from a
contradiction of the SSCHA literature into a confirmation and extension of it.

### 1.4 The reframed finding

> Monacelli predicts, from theory, that the bubble approximation breaks down for the cubic phase of
> metal halide perovskites. We measure that breakdown at benchmark scale, show it is not confined to
> halides — it holds for oxide perovskites and, in a numerically clean form with zero stochastic
> blow-ups, for cubic fluorites — quantify it across five foundation MLIPs, and show that a
> single-mode quantum SCHA screen costing a fraction of the compute does not fail in that regime.

That is a narrower claim than "SSCHA is a trap" and a far stronger one, because it is a
*quantitative confirmation of a published theoretical prediction* plus an extension to two chemistry
families the prediction did not cover. It also changes the referee pool from hostile to friendly.

### 1.5 Required citation repair

The manuscript uses `get_free_energy_hessian` and **never cites the paper that derived it.** Add:

- Bianco, Errea, Paulatto, Calandra & Mauri, *Phys. Rev. B* **96**, 014111 (2017),
  doi:10.1103/PhysRevB.96.014111 — cite in §2.5 for the Hessian and in §3.3 for positive-definiteness.
- Monacelli, *Phys. Rev. B* **112**, 014109 (2025), doi:10.1103/8611-5k5v (arXiv:2407.03090) — cite
  in §3.3 for the bubble breakdown and the RPA requirement.
- Yang, Yin, Ao & Li, *Phys. Chem. Chem. Phys.* **28**, 4459–4469 (2026),
  doi:10.1039/D5CP04693A (arXiv:2510.18178) — cite in §1 as prior finite-T MLIP benchmarking.
- *Are Foundational Atomistic Models Reliable for Finite-Temperature Molecular Dynamics?*,
  *J. Phys. Chem. C* (2025), doi:10.1021/acs.jpcc.5c07541 — cite in §1; it is on PbTiO₃, one of our
  own systems.

---

## 2. Novelty framing

**Delete** (§1, ~line 49): *"Whether foundation MLIPs reproduce this harmonic-to-finite-temperature
stabilisation is essentially un-benchmarked"*. Falsified by both 2025 papers above.

**Replace with** the honest, surviving delta:

> Two 2025 studies have since probed finite-temperature reliability directly: four MACE foundation
> variants were benchmarked for the dynamic stability of halide double perovskites, and the
> ferroelectric–paraelectric transition of PbTiO₃ was used to show a disconnect between static
> accuracy and dynamic reliability. Both are confined to a single model family or a single system.
> What remains untested is whether the finding is architecture-general and chemistry-general: no
> study has classified finite-temperature dynamic stability across multiple independent MLIP
> architectures and across the distinct anharmonic families — displacive and antiferrodistortive
> oxide perovskites, halide perovskites, entropy-stabilised bcc refractory metals, and cubic
> fluorites — under a common free-energy criterion and a temperature ladder.

**Title.** "Blind spot" asserts nobody is looking, which is now false in print. Options, in my order
of preference:

1. *Finite-temperature dynamic stability separates foundation machine-learning interatomic
   potentials that harmonic benchmarks rank equally* — leads with the H2 result, makes a
   falsifiable claim, no priority assertion.
2. *A cheap quantum soft-mode free-energy screen outperforms stochastic SCHA for displacive
   instabilities in foundation-MLIP stability screening* — leads with the cost/accuracy inversion,
   which is the most quotable result.
3. *Harmonic accuracy does not certify a foundation machine-learning interatomic potential for
   finite-temperature dynamic stability* — safest, closest to what the data actually support.

This is the author's call. My recommendation is **1**, with **3** as the conservative fallback.

---

## 3. Drafted replacement for the §3.3 root-cause paragraph

> **Root cause: the bubble approximation, not the reference structure.** A controlled diagnostic on
> cubic BaTiO₃ (MACE-MP-0, 100 K) isolates the mechanism, and the SCHA literature identifies it
> precisely. The auxiliary SCHA matrix **Φ** is positive-definite by construction — a normalisable
> Gaussian trial state requires it — so it can never soften, and searching its eigenvalues for an
> instability is meaningless.[Bianco2017, Monacelli2025] The object that *can* detect a displacive
> transition from a fixed high-symmetry reference is the free-energy Hessian,
> ∂²F/∂**R**∂**R** = **Φ** + **Φ**⁽³⁾**Λ**[**1** − **Φ**⁽⁴⁾**Λ**]⁻¹**Φ**⁽³⁾, which is derived for
> exactly this purpose and demonstrated on the ferroelectric transitions of SnTe and GeTe.[Bianco2017]
> Our diagnostic shows the failure enters at the truncation: with the fourth-order term dropped —
> the bubble approximation, and the `python-sscha` default — the Hessian returns +2.87 THz for a
> mode whose harmonic frequency is −5.6 THz. This is the regime in which the bubble is known to
> fail: it "breaks down in ... the cubic phase of metal halide perovskites, producing qualitatively
> incorrect results," and there Φ⁽⁴⁾ "plays a significant role ... and the complete RPA-like
> resummation is needed to determine the phase stability."[Monacelli2025] Our contribution is to
> measure that breakdown at benchmark scale and to show it is not confined to halides: it holds for
> oxide perovskites, and for cubic fluorites it holds in a numerically clean form with zero
> stochastic blow-ups (§3.3, fluorites), which separates a methodological limit from a numerical
> one. Enabling `include_v4=True` is therefore the physically correct escalation, consistent with
> the RPA requirement above; it did not complete in 18 min on a single unit here, which is the
> practical barrier at screening scale.

**Consequential edits elsewhere:** the Abstract's "is itself a methodological trap", the §3.3
heading "a trap for perovskites", and the Discussion's "the standard escalation ('if in doubt, run
SSCHA') fails" all need the same treatment — the failure is the *default truncation*, not SSCHA.

---

## 3a. RESULTS OF THE MULTI-MODE RE-RUN (2026-08-16, complete)

The screen's criterion changed from "the globally softest commensurate mode" to **"every distinct
imaginary commensurate mode; the phase is unstable if ANY condenses."** Full grid re-run: 5 models
× 20 systems × 4 T = 400 units, 0 errors, in one pinned environment per model with `pip freeze`
locks committed (`envs/lock-*-2026-08-16.txt`). Reproduce with `scripts/analyze_v4.py`.

Recall on genuinely-unstable units, scored against the temperature-resolved ground truth:

| family | legacy | multi-mode | n |
|---|---|---|---|
| FE oxide perovskite | 0.77 | **0.53** | 30 |
| AFD (SrTiO₃) | 0.20 | **0.60** | 5 |
| halide perovskite | 0.72 | **0.92** | 25 |
| cubic fluorite | 1.00 | 1.00 | 20 |
| controls (correct-stable) | 1.000 | **1.000** | 120 |
| **overall accuracy, all 20 systems, all T** | 0.740 | **0.728** | 400 |

Read this correctly. Overall accuracy is **unchanged** (0.740 → 0.728); what changed is its
*composition*. The screen got substantially better on the antiferrodistortive and halide
instabilities and worse on the FE oxides, because it is now looking at the modes that actually
drive those transitions instead of whichever mode happened to survive an inverted mask. The
headline is not "the fix improved the screen" — it is **"the fix moved the screen's accuracy onto
the physically correct modes."**

Controls remain perfect: **zero** imaginary commensurate modes are found for any of the six
controls under any of the five models, so the screen never invents an instability.

**The screen-vs-SSCHA contrast survives but narrows sharply: 0.53 vs 0.30 on the FE-oxide set,**
where the published claim was 0.77 vs 0.23. It is still a real inversion — the cheap screen beats
the expensive default-truncation SSCHA in the displacive regime — but it is now a factor of 1.8,
not 3.3, and the abstract must say so.

### The new SrTiO₃ validation gate is a genuine physics test

The screen now resolves three distinct imaginary modes in cubic SrTiO₃ and reports which condenses:

```
MACE-MP-0, T=100 K   q =  Γ    ; R(½½½) ; (0,½,½)
                  harm = -2.365 ; -2.089 ; -0.817  THz
                    Q₀ =  0.000 ;  0.140 ;  0.000  Å      -> only R condenses -> UNSTABLE
MACE-MP-0, T=300 K  Q₀ =  0.000 ;  0.000 ;  0.000  Å      -> STABLE
```

The Γ ferroelectric mode correctly refuses to condense — that is quantum paraelectricity, and it is
the physically right answer — while the R-point antiferrodistortive tilt does, bracketing the
experimental T_c = 105 K. **3 of 5 models (MACE-MP-0, MatterSim, SevenNet-0) reproduce this; CHGNet
never condenses the tilt, and ORB-v2 does not even select R.** That is a per-model gate carrying far
more information than the old single-model "hardens −2.6 → +1.8 THz" sentence, which was an artefact
of the inverted mask forcing the screen off Γ.

### H2's supporting statement is now falsified and must be rewritten

Per-model recall on the genuinely-unstable non-bcc set, T ≤ 300 K:

| model | recall | control acc |
|---|---|---|
| SevenNet-0 | **0.875** | 1.0 |
| CHGNet | 0.812 | 1.0 |
| MACE-MP-0 | 0.750 | 1.0 |
| MatterSim | 0.750 | 1.0 |
| ORB-v2 | 0.688 | 1.0 |

§3.2 currently asserts *"MatterSim and SevenNet-0 are harmonically perfect yet sit behind MACE-MP-0
and CHGNet on finite-temperature displacive stability."* On the corrected data **SevenNet-0 is the
finite-temperature leader** while also being a harmonic leader, so that sentence is false as
written. H2's weaker form ("necessary but not sufficient") still stands on ORB-v2 and MatterSim, but
the specific inversion claim goes.

### The mode cap cannot change any conclusion

`max_modes = 24` binds on 20 of 400 rows (MatterSim bcc-Zr/Hf find 41 imaginary modes, ORB bcc-Ti
32, CHGNet CsSnBr₃ 29). **All 20 capped rows are already called UNSTABLE, and the minimum number of
condensing modes among them is 3.** Since one condensing mode suffices for an unstable call, the cap
could only ever create a false-*stable*, which requires all 24 screened modes to be non-condensing.
No capped row is anywhere near that boundary, so the truncation is provably inconsequential here.
`max_modes` is now part of the unit hash and the cache key, so a tighter-cap unit can never be
silently reused as a looser-cap one.

## 4. Propagation — DONE

All six figures regenerated from the corrected ledger and every dependent number updated in the
manuscript, supplementary, cover letter and `.zenodo.json`. Summary of what moved:

| quantity | published | now |
|---|---|---|
| screen recall, FE oxides, T ≤ 300 K | 0.77 | **0.53** |
| SSCHA recall, same set | 0.23 | **0.30** |
| bcc method agreement, Spearman ρ | 0.78 | **0.74** |
| bcc sign agreement | 0.64 | **0.62** |
| " excl. ORB-v2 | 0.78 / ρ 0.63 | **0.69 / ρ 0.74** |
| H3 consensus error rate | 0.167 | **0.200** |
| H3 split-vote / unanimous error | 0.39 / 0.07 | **0.57 / 0.09** |
| H3 enrichment | 5.5× | **6.6×** |
| H3 AUC (vote split / freq σ) | 0.75 / 0.52 | **0.76 / 0.55** |
| finite-T leader | CHGNet, MACE (0.933) | **SevenNet-0 (0.900)** |
| finite-T worst | ORB-v2 (0.700) | **ORB-v2 (0.800)** |

A guard was added so this class of error cannot recur silently: `mlip_dynstab.analysis.canonical`
selects one generation of each method's rows, and `scripts/make_figures.py` routes through it.
Without it, `df[df.method == "softmode"]` matches both the legacy and multi-mode grids and
double-counts every unit — 800 rows where there are 400 measurements.

Two claims did **not** survive propagation and were rewritten rather than re-fitted:
- **H2's supporting example.** "The harmonic leaders are not the finite-temperature leaders" is
  false now that SevenNet-0 leads both. The replacement uses CHGNet, which is worst harmonically
  (0.789) and second best at finite temperature (0.867) — a cleaner demonstration of the same
  point, and one that does not depend on which model happens to lead.
- **The T\* multi-anchor ordering.** SrTiO₃ < BaTiO₃/KNbO₃ still holds, but PbTiO₃ — the highest
  T_c of the four — is predicted to stabilise at 100 K by three of five models. That is an ordering
  failure, not a scale error, so T* is demoted from "validation" to "diagnostic" and the SrTiO₃
  gate plus control performance carries the licensing argument instead.

Also corrected: the six SSCHA numerical blow-ups are **not** all float32 — four are ORB-v2 on
SrTiO₃ and two are MatterSim on PbTiO₃, so the precision attribution in the published text was
wrong. And the §3.5 soft-mode convergence runs are now marked **superseded**, since they were
produced by the pre-correction single-mode selection and have not been repeated at 3×3×3.

## 5. Added to the compute queue by this reading

One `include_v4=True` run driven to completion on a single BaTiO₃ unit would convert the root-cause
paragraph's central claim from literature-supported inference to direct measurement. Wall-clock-cap
it and report the cap honestly if it does not converge.
