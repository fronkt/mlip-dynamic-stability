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
Purdue University, West Lafayette, Indiana 47907, USA
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
SrTiO₃ the Γ ferroelectric mode is deeper than the R-point tilt yet is quantum-suppressed, and
only the tilt drives the 105 K transition. Third, MLIP-driven SSCHA at its default bubble
truncation systematically false-stabilises deep displacive instabilities and can diverge
numerically; we measure this predicted failure at benchmark scale. Cross-model disagreement
supplies a cheap, practical guardrail for flagging the resulting unreliable calls.

## Keywords

machine-learning interatomic potentials; dynamic stability; anharmonicity; stochastic
self-consistent harmonic approximation; high-throughput screening

## References

1. A. Loew, D. Sun, H.-C. Wang, S. Botti and M. A. L. Marques, *npj Comput. Mater.*, 2025, **11**, 178.
2. L. Monacelli, R. Bianco, M. Cherubini, M. Calandra, I. Errea and F. Mauri, *J. Phys.: Condens. Matter*, 2021, **33**, 363001.
3. T. Tadano and S. Tsuneyuki, *Phys. Rev. B*, 2015, **92**, 054301.
4. L. Monacelli, *Phys. Rev. B*, 2025, **112**, 014109.

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
- **Word limit RESOLVED (2026-09-10): the official template states "300 words maximum."** An
  earlier note in this file said no limit was published; that was wrong — the limit lives in the
  template, not on the web page. At 269 words the abstract fits with 31 words of headroom. Trim
  candidates if it ever needs to shrink: the SrTiO₃ clause in result two, then the final guardrail
  sentence.
- **Template obtained.** The Abstract Submission page links it via a redirect
  (`/en/web/jump/36624?mid=3190186&nid=148374` →
  `https://files.sciconf.cn/public/2117/54869/202604/2026042208335886319410752.docx`). It requires:
  Times New Roman throughout, a **Keywords** line and a **References** list in RSC style
  (`A. Name, B. Name and C. Name, Journal Title, 2000, 35, 3523`) — both of which this draft
  originally lacked and which are now supplied above. Figures/images are permitted.
- **Submission-ready file: `abstract-rsc-first-2026.docx`** in this directory, generated into the
  official template by `build_abstract_docx.py` (re-runnable; it reads this file as the single
  source of truth, so edit the Markdown and rebuild rather than editing the DOCX).
- Submission route: Conference Personal Dashboard at https://www.rscfirst.org.cn/en/user/login/ →
  Submit. Requires Frank's login; presentation type selected in the portal, not in the file.
- Affiliation carries postal code 47907 (Purdue West Lafayette main campus), which the manuscript's
  author block omits; the template asks for a postal address. No department is named, matching
  `paper/manuscript.md:5`. Change if you would rather list Engineering Technology.
