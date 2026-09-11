# Compute runbook — the three RSC-revision items that need a rented box

Everything else in the revision is done from the deposited ledger. These three are not, and all
three need the same box, so they should be run in one session.

**Why not locally.** This laptop has 8 CPUs and no GPU. Only `mace` and `chgnet` import here;
`mattersim`, `sevenn` and `orb_models` are absent, and **`sscha` / `cellconstructor` are absent**,
which rules out anything in item 2. A CHGNet-only displacement sweep was started locally
(item 3, partial).

**Budget.** ~$20–60 agreed. See [[feedback_vast_workflow]] conventions: cap workers on many-core
boxes, install torch from the cu128 index for a 5090, and check `Inet down` ≥ 200 Mbit/s before
pulling checkpoints (~1.5 GB of model weights across five models).

**Do not use ACCESS CHE260157 for any of this.** That allocation is STS-scope only.

---

## Item 1 — Force-level ensemble uncertainty (Referee 2)

*The on-target half of R2's second point. §3.4 already answers the frequency-level version; this
is the force-level version they actually asked for.*

**What to compute.** For each displaced configuration along the cached E(Q) maps, evaluate the
forces under all five models and record the cross-model spread. Then ask the same question §3.4
asks of the frequency spread: does it predict that the consensus stability call is wrong?

**Inputs already deposited.** `results/cache/softmode_v3*.json` holds 222 cached E(Q) maps; the
displaced structures are regenerable from the stored mode pattern and Q grid without redoing the
q-search.

**Output.** A per-(system, model, T) force-spread column, then the same clustered AUC treatment
as §3.4 via `mlip_dynstab.stats.cluster_permutation_auc`. Expect to add a row to Table S9 and a
sentence to §3.4.

**Watch for.** ORB-v2 is float32 and non-conservative — its force spread is not comparable to the
others' and must be reported split, as everything else now is.

## Item 2 — Four-seed stochastic test beyond bcc-Zr (Referee 1, item 4)

*Explicitly requested: "expanded beyond bcc-Zr to include representative displacive systems
(BaTiO3 or fluorites)".*

**What to run.** `scripts/sscha_repro.py` on BaTiO₃ and one fluorite (ZrO₂), four independent
seeds each, at 100 K, mirroring the existing bcc-Zr / MACE-MP-0 study which gave
+1.798 ± 0.001 THz.

**Requires.** `python-sscha` + `cellconstructor` + one model env. Note the v3 gotcha recorded in
the Data Availability section: `ase>=3.23` removed the `Phonons` attributes cellconstructor's ASE
bridge consumed, so the harmonic initialiser must be built from phonopy full force constants
(this is already what `run_sscha_v2.py` does — reuse that path, do not write a new one).

**Expected outcome, and the honest framing either way.** The bcc study found stochastic noise
2–3 orders below the cross-model margins. On BaTiO₃ the relevant units are the ones that
false-stabilise, so the question is whether +2.87 THz is seed-stable. If it is, the
false-stabilisation is deterministic and the truncation account is strengthened. **If it is not,
that is a finding against the paper and must be reported as such** — it would mean part of the
perovskite SSCHA behaviour is sampling noise rather than truncation.

**Then.** Update ESI §S2.4 and Table S11's "what the harness did not retain" note, and close the
R1.4 line in the response letter.

## Item 3 — Displacement-amplitude sweep (not requested; closes a self-declared gap)

**Status: partially running locally** (CHGNet only, `scripts/run_disp_sweep.py --models chgnet
--device cpu`, log in the session scratchpad). Finish the other four models on the box.

    python scripts/run_disp_sweep.py --device cuda        # all five; skips what is done

**Grid.** 0.005 / 0.02 / 0.03 Å over the 19 scored systems. The production 0.01 Å is already
deposited and the script refuses it as a sweep value.

**The trap, already handled — do not "fix" it again.** Non-production amplitudes are recorded
under method `harmonic_dispsweep`, never `harmonic`, because sweep rows carry the same
`method_version` as production rows and would otherwise pass through `analysis.canonical()` and
inflate the denominator of every harmonic rate in the paper. The production amplitude is
deliberately **absent** from the unit hash so that all 100 deposited hashes stay valid; this was
verified, not assumed.

**Analysis.** Report per model: max |Δ| in the soft-mode frequency across amplitudes, and any
stability-call flips. The claim to test is the one ESI §S1.2 currently cannot make — that the
§3.2 reordering is not an artifact of the displacement amplitude. CHGNet and MatterSim are the
two models that carry it, so those two matter most.

---

## On return

- [ ] `python scripts/verify_claims.py` — 29 assertions, must stay green
- [ ] `python scripts/stats_hardening.py --n-perm 10000`
- [ ] `python scripts/build_esi_tables.py` (then `--check` in CI)
- [ ] Fold results into ESI §S1.2 / §S2.4 / Tables S9 and S11
- [ ] Close the three `[PENDING]` items in `paper/response/response_to_referees.md`
- [ ] Mint a new Zenodo version; the manuscript cites the concept DOI so the reference survives
