# Cover letter

Frank Cai
Purdue University, West Lafayette, Indiana, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

To the Editors, *Digital Discovery* (Editor-in-Chief: Prof. Alán Aspuru-Guzik)

Dear Editors,

I am pleased to submit the manuscript **"Finite-temperature dynamic stability is a blind spot of
foundation machine-learning interatomic potentials"** for consideration as a Paper in *Digital
Discovery*.

Foundation (universal) machine-learning interatomic potentials (MLIPs) — MACE-MP-0, CHGNet, ORB,
SevenNet, MatterSim — are now routine, cheap stand-ins for DFT in high-throughput pipelines,
including the dynamical-stability filtering of generative crystal-structure-prediction (CSP)
outputs. Yet they are benchmarked almost exclusively against **harmonic (0 K) phonons**, while
dynamic stability is physically a **finite-temperature** property. An entire technologically
central class of materials — cubic perovskites, bcc refractory metals, cubic fluorites — is
*harmonically unstable but thermally stabilised by anharmonicity*, exactly the regime the
harmonic-only oracle (used by recent benchmarks such as PhononBench and by CSP screening) gets
wrong. Whether foundation MLIPs reproduce this harmonic → finite-temperature stabilisation has
been essentially un-benchmarked. This work closes that gap.

Our contributions, and why they should interest the *Digital Discovery* readership:

- **A finite-temperature dynamic-stability benchmark** of five foundation MLIPs on a curated,
  literature-grounded set (no new DFT required), with a fully reproducible, resumable per-unit
  results ledger and open code.
- **A cheap, validated screen.** We introduce a single-mode quantum self-consistent-harmonic
  ("soft-mode free energy") screen that resolves the harmonic → finite-T stabilisation at
  sub-second CPU cost per temperature, validated against experimental transition temperatures.
- **A cautionary methodological result of direct practical value.** The "gold-standard"
  multi-mode stochastic SCHA, when driven by an MLIP in the way a practitioner would realistically
  deploy it, is itself a trap: it is clean for martensitic bcc metals but *systematically
  false-stabilises* deep displacive (ferroelectric) instabilities — the regime that dominates
  generative-CSP outputs — so the cheap screen is the more reliable indicator there. This inverts
  the usual cost/accuracy intuition and reframes how such cross-checks should be used.
- **An actionable guardrail.** Inter-model (ensemble) vote disagreement flags unreliable calls
  (AUC 0.75), whereas the continuous frequency spread does not — a cheap, deployable rule.

This is a data-driven, evaluation-methodology contribution about the reliability of ML surrogates
for materials discovery, which is squarely within the scope of *Digital Discovery* and directly
relevant to anyone using foundation MLIPs to screen generated structures.

I confirm that this manuscript is original, has not been published previously, and is not under
consideration for publication elsewhere. It has a single author with no conflicts of interest to
declare. All code and the per-unit results ledger are openly available
(https://github.com/fronkt/mlip-dynamic-stability; archived at Zenodo,
https://doi.org/10.5281/zenodo.20805824), so every reported number regenerates from the deposited
artifacts.

Thank you for considering this work.

Sincerely,

Frank Cai
Purdue University
