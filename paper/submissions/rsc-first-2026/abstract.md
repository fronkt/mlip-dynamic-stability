# RSC FIRST 2026: AI in Chemistry — abstract submission

**Venue:** RSC FIRST 2026: AI in Chemistry, Xiamen, China, 19–21 November 2026
**Session:** Parallel Session 01 — Intelligent Computation
**Presentation type:** Oral (fallback: poster)
**Deadline:** 15 September 2026
**Portal:** Conference Personal Dashboard → Submit

---

## Title

Finite-temperature dynamic stability separates foundation machine-learning interatomic
potentials that harmonic benchmarks rank equally

## Author

Frank Cai
Purdue University, West Lafayette, Indiana, USA
frankyc11223@gmail.com
ORCID 0009-0003-0041-1459

## Abstract (269 words)

Foundation machine-learning interatomic potentials (MLIPs) are now routine, cheap stand-ins for
DFT in high-throughput materials screening, yet they are benchmarked almost exclusively against
harmonic phonons at 0 K. Dynamic stability is physically a finite-temperature property, and a
technologically central class of materials — cubic perovskites, bcc refractory metals and cubic
fluorites — is harmonically unstable but thermally stabilised by anharmonicity. Ranking models on
0 K phonons therefore risks ranking them on the wrong quantity.

We test five foundation MLIPs (MACE-MP-0, CHGNet, ORB-v2, SevenNet-0, MatterSim) in exactly this
regime, using a curated set of systems whose finite-temperature behaviour is documented in the
experimental and DFT literature, so that no new DFT ground truth is required. We introduce a cheap
quantum self-consistent-harmonic "soft-mode free-energy" screen that evaluates *every* imaginary
commensurate mode and calls the high-symmetry phase unstable if any of them condenses, and we
cross-validate it against gold-standard multi-mode stochastic SCHA (SSCHA).

Three results follow. First, harmonic accuracy does not predict finite-temperature accuracy in
either direction: the worst model harmonically is the second best at finite temperature, and a
harmonically perfect model is only mid-table. Second, screening every imaginary mode rather than
the softest one is essential, because the deepest mode need not be the one that condenses — in
SrTiO3 the Γ ferroelectric mode is deeper than the R-point tilt yet is quantum-suppressed, and
only the tilt drives the 105 K transition. Third, MLIP-driven SSCHA at its default bubble
truncation systematically false-stabilises deep displacive instabilities and can diverge
numerically; we measure this predicted failure at benchmark scale. Cross-model disagreement
supplies a cheap, practical guardrail for flagging the resulting unreliable calls.

## Notes

- Base text adapted from the 288-word SIMBIOCHEM condensation
  (`paper/workshop-simbiochem/main.tex`), reframed for a chemistry rather than ML audience:
  leads with the screening stake, drops the generative-CSP motivation, and promotes the SSCHA
  bubble-truncation result to a headline finding (it is the result this committee will find most
  interesting — Csányi, Jun Cheng, Chao Zhang and Mathieu Salanne all build or use MLIPs).
- Non-anonymous, unlike the SIMBIOCHEM version. Contact email is frankyc11223@gmail.com,
  matching the author block actually used in the Digital Discovery manuscript
  (`paper/manuscript.md:5`) and the ORCID record. (An earlier draft of this file used a
  `@purdue.edu` address and claimed it matched the DD submission; it does not.)
- No abstract word limit is published on the conference site; 269 words is comfortably inside what is conventional for an
  RSC conference abstract. Trim candidates if a limit turns out to be lower: the SrTiO3 clause
  in result two, then the final guardrail sentence.
- The official template must be downloaded from the Abstract Submission page and this text pasted
  into it; the conference requires the template, not free-form text.
