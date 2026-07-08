# Suggested reviewers — *Digital Discovery*

The manuscript sits at the intersection of (a) anharmonic lattice dynamics / finite-temperature
phonon methodology (SSCHA, TDEP) and (b) foundation machine-learning interatomic potentials (MACE,
CHGNet, ORB, SevenNet, MatterSim). A balanced panel should cover both, and should weight toward (a)
since the five MLIP teams in (b) are the evaluated parties, not neutral reviewers of them. None of
the names below are collaborators or recent co-authors of the author (sole independent author,
Purdue University; no known conflicts). **Affiliations reflect the author's best knowledge and may
be stale — verify current institution before submitting to the portal.**

## Tier 1 — strongest fit: anharmonic finite-temperature methodology, no stake in any tested MLIP

1. **Lorenzo Monacelli** — Sapienza University of Rome, Italy.
   Lead developer of the SSCHA code (ref. 4) and co-author of the SSCHA validation study this paper
   cites directly (ref. 14, on ferroelectric/anharmonic materials). Best placed to judge whether the
   paper's SSCHA usage and its "SSCHA can false-stabilise displacive instabilities" claim are sound,
   since he has no stake in which foundation MLIP comes out ahead.
   *Field: anharmonic lattice dynamics, SSCHA methodology.*

2. **Ion Errea** — University of the Basque Country (UPV/EHU) & DIPC, San Sebastián, Spain.
   Co-developer of SSCHA, with a long track record applying it to systems that are harmonically
   unstable but anharmonically stabilised (hydrides, perovskites). Directly relevant to the
   bcc-metal and ferroelectric-perovskite test cases.
   *Field: anharmonic lattice dynamics, SSCHA methodology.*

3. **Olle Hellman** — Linköping University, Sweden.
   Developer of the Temperature Dependent Effective Potential (TDEP) method (ref. 5), the leading
   alternative to SSCHA for finite-temperature phonons. A methodology reviewer from outside the
   SSCHA lineage itself, useful for checking whether the paper's framing of "the" finite-temperature
   gold standard is fair.
   *Field: anharmonic lattice dynamics, alternative (TDEP) methodology.*

## Tier 2 — foundation-MLIP benchmarking generalists (not owning any of the five tested models)

4. **Kamal Choudhary** — NIST (National Institute of Standards and Technology), USA.
   Maintains JARVIS-DFT and has published broadly on benchmarking foundation/universal MLIPs across
   many tasks. A generalist check on whether the benchmark design and the false-stable-rate framing
   are fair to the field, without a specific model to defend.
   *Field: MLIP benchmarking, high-throughput DFT.*

5. **Volker Deringer** — University of Oxford, UK.
   Works on validating machine-learning interatomic potentials (GAP lineage) against real materials
   physics, including cases where MLIPs get subtle structural/dynamical effects wrong. Good check on
   whether the paper's soft-mode screen is a reasonable diagnostic or an artifact of the chosen
   models.
   *Field: MLIP validation and methodology.*

## Tier 3 — developers of the specific benchmarks/models this paper evaluates (relevant but conflicted; defer to editor)

6. **Authors of "Universal MLIPs are ready for phonons"** (A. Loew, M. A. L. Marques et al., ref. 1)
   — Martin Luther University Halle-Wittenberg, Germany. This is the harmonic-only benchmark whose
   scope limitation motivates the paper. Deepest knowledge of the harmonic PES-softening result the
   paper builds on and complicates, but reviewing a paper that argues their benchmark is
   insufficient is a direct conflict of interest — suggest including only if the editor is
   comfortable with it.
   *Field: MLIP phonon benchmarking.*

7. **A developer from one of the five evaluated MLIP teams** — e.g. Ilyes Batatia / Gábor Csányi
   (MACE, Cambridge), Bowen Deng / Gerbrand Ceder (CHGNet, Berkeley), Yutack Park / Seungwoo Hwang
   (SevenNet, KAIST), or the MatterSim (Microsoft Research) or Orb (Orbital Materials) teams. Each
   has the deepest possible knowledge of their own model's training data and known failure modes,
   but is evaluating a paper that reports a false-stable rate or a finite-temperature shortfall for
   their own model — a direct conflict. Not recommended as a primary reviewer; note only in case the
   editor wants a rebuttal-informed technical check.
   *Field: foundation MLIP development.*

## Recommended selection

A typical 3-reviewer request: **Monacelli** or **Errea** (SSCHA methodology), **Hellman**
(independent finite-T methodology check), and **Choudhary** or **Deringer** (MLIP-benchmarking
generalist). Avoid stacking the panel with Tier 3 names; at most one, and only at the editor's
discretion.

## Opposed reviewers

None. (Sole independent author; no competitive or conflicting parties to exclude.)
