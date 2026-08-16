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

## 4. Blocked on the softmode re-run

These cannot be written until the v2 grid lands, because every number changes:

- screen recall on the FE set (was 0.77), the screen-vs-SSCHA contrast, and Fig. 5
- the method-agreement ρ and sign agreement on bcc (was 0.78 / 0.64), and Fig. 4
- all T* multi-anchor values, §3.2's per-model finite-T table, and Fig. 2
- H3's AUC and enrichment, and Fig. 6
- §3.5's finite-size claims for BaTiO₃ and SrTiO₃

The corrected **SSCHA-side** numbers from audit D1 are stable and can be written now:
FE-perovskite recall 0.23 → **0.30**, fluorite 0.17 → **0.23**, bcc unchanged (0 of 75 rows).

## 5. Added to the compute queue by this reading

One `include_v4=True` run driven to completion on a single BaTiO₃ unit would convert the root-cause
paragraph's central claim from literature-supported inference to direct measurement. Wall-clock-cap
it and report the cap honestly if it does not converge.
