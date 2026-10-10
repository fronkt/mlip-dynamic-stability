# Compute runbook: RSC revision Phase 1 (C1-C6) on one rented box

Rewritten 2026-09-26 for the phased plan (`tasks/todo.md`, Phase 1). The previous three-item
runbook is kept verbatim in `tasks/compute-runbook-rsc-revision-archive-2026-09-26.md`.

Everything in Phase 1 runs on **one** vast.ai box, driven by two scripts:

- `scripts/box/setup_rsc_box.sh` rebuilds the five pinned model envs from
  `envs/lock-<model>-2026-08-17.txt` (the envs the August re-measurement ran in), adds SSCHA, Quantum
  ESPRESSO and SSSP 1.3, smoke-tests every model and prints a readiness table. It is idempotent.
- `scripts/box/run_rsc_revision.sh run` runs every stage in order with a timeout and a stage
  marker, so a relaunch resumes. `status`, `check`, `pack` and `pull-list` are its other
  subcommands.

**Do not use ACCESS CHE260157 for any of this.** That allocation is STS-scope only.

---

## 1. What the run answers

| Item | Referee point | Stage(s) on the box | Output (under `results/revision/`) | Fold-in (todo Phase 4) |
|---|---|---|---|---|
| **C1** SSCHA seeds + diagnostics | R1.4: sample sizes, gradient history, stopping criteria, Hessian uncertainty; four-seed test beyond bcc-Zr | `sscha_seeds_mace`, `sscha_seeds_mattersim` (`sscha_seed_study.py --preset revision`) | `sscha_seeds/<unit>.json` (per-seed histories, populations, stopping test that fired, bootstrap spread of the lowest Hessian eigenvalue, ledger-reproduction check), `sscha_seeds/configs/*.extxyz` | F1: §3.3, §3.5, §S2.4, Table S11, new seed table |
| **C1b** include_v4 | Our own mechanism test for T3 ("the truncation is what goes wrong") | `sscha_v4` (last in the GPU chain; the v4 child runs on threaded OpenBLAS, see §8) | `v4` block inside `sscha_seeds/batio3_cubic_mace_mp0_100K_sc222.json` | F1 |
| **C2** force-level ensemble uncertainty | R2.2 | `fs_fc_*`, `fs_configs`, `fs_forces_*` (five models + MACE small/large + MatterSim 1M) | `force_spread/{fc,configs,forces}/`; `summary.json` comes from the local `analyze` | F2: §3.4, §4, Table S9 |
| **C3a** PBE along the soft-mode coordinates | R1.1 | `dft_plan`, `dft_geom_*` (MatterSim first), `dft_mlipeval_*`, `dft_qe_inputs`, QE pass `^a_` | `dft/geom/` (+ `geom/checks/`: a path that misses its cached map is written and flagged, `geom` exits 4, and the stage counts as done with a `[flag]` line), `dft/mlip/`, `dft/qe/a_*/` | F2: new §3.x paragraph + ESI section + table |
| **C3b** PBE forces on SSCHA-sampled configurations | R1.2 (MLIP out-of-distribution error, or method failure?) | `dft_c3b_inputs`, `dft_mlipeval_c3b_*`, QE pass `^b_`; written only once every C1 unit's seeds are final (see §6) | `dft/geom_c3b/`, `dft/qe/b_*/` | F2 |
| **C4** displacement sweep | closes ESI §S1.2's self-declared gap | **done before the box**: 57 units x 5 models are committed (CHGNet 1d93e25, the other four a27e1e3). On the box only `c4_coverage_<model>`: a `run_disp_sweep.py --dry-run` per env that must find 57/57 deposited units, i.e. the rebuilt envs report the deposited model versions | nothing (log lines only). `RUN_C4=1` restores the real sweep and its ledger writes | F3: Table S13 (already folded) |
| **C5** bcc finite size in the current envs | R1.4 / §3.5 | inside the SSCHA preset (zr_bcc 3x3x3, MACE-MP-0 and MatterSim, 100 and 300 K) | `sscha_seeds/zr_bcc_*_sc333.json` | F3: §3.5 finite-size sentence |
| **C6** pull, destroy | | `pack` | one tarball | |

Nothing on the box writes `results/ledger.parquet`/`.jsonl` unless `RUN_C4=1` is set.

## 2. Honest framing: state the reading before the number exists

Report whichever way each result falls. The paper's claims move to fit the data, never the reverse.

- **C1.** If the lowest free-energy Hessian eigenvalue of BaTiO3/ZrO2 at 100 K is seed-stable
  (spread well below the cross-model margins), the false-stabilisation is deterministic and the
  truncation account is strengthened. **If it is not, that is a finding against the paper**: part
  of the SSCHA behaviour is sampling noise, and it goes in the text as such. Seed 0 must reproduce
  the deposited ledger value; the JSON records whether it does. If it does not, stop and find out
  why before interpreting the other seeds.
- **C1b.** Three outcomes, each reportable: include_v4 recovers the instability (direct support for
  "the truncation is what goes wrong"); it stays positive (the truncation is not the whole story);
  it hits the 3 h cap (say so with the thread count and BLAS, and claim nothing). A v4 number is
  read only if `v4_false_roundtrip_ok` is true: the child recomputes include_v4=False on the
  reloaded ensemble under OpenBLAS and must match the parent's reference-BLAS value within 1e-3 THz.
- **C2.** The decision rule is pre-registered in `scripts/force_spread.py` (docstring, 2026-09-26):
  "flags untrustworthy calls" only if the clustered-bootstrap 95% CI of the primary AUC lies wholly
  above 0.5 **and** the four-model AUC is above 0.5; a CI wholly below 0.5 is "anti-informative";
  anything else is "not shown". Do not add a post-hoc primary.
- **C3a.** If PBE puts no double well where the screen found one (or one where it found none), the
  screen's call on that unit is not supported by first principles, and the text says so for that
  unit.
- **C3b.** MLIP force error on SSCHA-sampled configurations much larger than on the near-equilibrium
  rattled baseline supports "MLIP out-of-distribution"; comparable errors point at the method. Either
  answers R1.2.
- **C5.** Every stability-call flip is listed. A sign change of the bcc margin at 3x3x3 is
  reported, not averaged away.

## 3. Box choice

Hard requirements, and why:

| Requirement | Why |
|---|---|
| GPU compute capability 7.5-9.0 (Turing/Ampere/Ada/Hopper), **not Blackwell** (no RTX 5090, B200, RTX PRO 6000) | mace/chgnet/sevennet locks pin torch 2.6.0+cu124, which has no Blackwell kernels. setup stops rather than swap torch. |
| Host driver >= 580 (vast `cuda_vers >= 13.0`) | orb/mattersim locks pin torch 2.13.0, a CUDA 13.0 build. |
| cgroup quota >= 30 cores (vast `cpu_cores_effective`); **prefer >= 48** | QE is the long pole and gets (quota - 8) cores. `nproc` lies on vast; setup prints the real quota. |
| >= 150 GB disk, >= 64 GB RAM, >= 12 GB GPU RAM | five envs + uv cache + QE ~40 GB; up to 56 QE ranks of 40-atom PAW cells (~0.5 GB each) beside the C1b child, whose v4 tensor and three 14400 x 14400 work matrices take ~8-10 GB. |
| Inet down >= 200 Mbit/s | the env builds pull ~10 GB of wheels. |

Search (field names checked against `vastai search offers --help`):

```bash
export VAST_API_KEY=$(cat ~/.vast_api_key)
vastai search offers 'num_gpus=1 compute_cap>=750 compute_cap<1000 cuda_vers>=13.0 cpu_cores_effective>=48 cpu_ram>=64 gpu_ram>=12 disk_space>=150 inet_down>=200 reliability>0.98' -o 'dph'
```

Snapshot 2026-09-27: 64 offers met the 30-core version, 25 the 48-core one, from $0.27/h
(RTX 3060 12 GB, 64 vCPU, old Xeon E5-2697A v4) to $0.40-0.47/h (RTX 4090, 64 vCPU EPYC). An RTX
4090 or 3090 with 64 vCPU on EPYC at about $0.45/h is the sweet spot: QE then runs 7 jobs x 8 ranks.

Image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (Ubuntu 22.04). The envs do not use the
image's Python, since uv brings its own, so the image only has to boot with the NVIDIA driver visible.

## 4. Budget

- **GPU chain** (C3 prep, C1+C5, C3b prep, C2, the C4 coverage check, C1b): unmeasured; estimated
  10-17 h.
- **QE** (from `dft_reference.py plan`, c0 = 8 core-s per k-point *assumed*, k points counted
  with time-reversal only, so an upper bound on k):

  | pass | jobs | modelled core-h (upper) | at 56 cores | at 24 cores |
  |---|---|---|---|---|
  | `a_` C3a deciding + reference paths (R1.1) | 230 | <= 2630 | <= 47 h | <= 110 h |
  | `b_` C3b SSCHA configs (R1.2) | 68 | <= 1174 | <= 21 h | <= 49 h |
  | `ax_` extra modes (optional) | 290 | <= 998 | <= 18 h | <= 42 h |

  The 40-atom SrTiO3 paths (6x6x6 k-mesh) are 1207 of the 2630 a_ core-h.
- At ~$0.45/h, a box living <= 90 h costs <= ~$40. **Credit was $11.63 on 09-26: top up to at
  least $50 before renting.** vast stops an instance when the balance reaches zero, mid-QE.
- **Decision point, about 6-10 h into QE.** On the box run
  `cd /root/mlip-dynamic-stability && /root/env-mace/bin/python scripts/dft_reference.py analyze`
  (it prints the measured c0 and "remaining <= N core-h"), then hours = N / (quota - 8). If a_ + b_
  will not finish by about 2 Oct (Phase 4 fold-in), apply the levers in this order:
  1. `touch /root/logs/qe.no_extras`: the driver skips the ax_ pass (read when each pass is due).
  2. A second CPU-heavy box running only `qe_queue.sh --filter '^b_'` over a copy of
     `results/revision/dft/qe/` (the queue is directory-based; merge the job directories on return).
  3. The k-spacing (0.15 1/A, 2*pi included) and C3b `--per-unit` are scientific settings of
     `dft_reference.py`. Change them only with a convergence check and a line in the ESI.

## 5. Launch

Local steps run in git-bash from the repo root. One ssh at a time, never faster than one connection
every ~8 s (vast's gateway throttles).

0. **Push first.** The box clones `rsc-figure-fixes` from GitHub, so every script it runs must be
   committed and pushed: `scripts/box/{setup_rsc_box.sh,run_rsc_revision.sh,qe_queue.sh}`,
   `scripts/{sscha_seed_study.py,force_spread.py,dft_reference.py}` and this runbook. Each Python
   study's local CPU smoke test must have passed.
1. Rent (after the credit top-up):
   ```bash
   vastai create instance <OFFER_ID> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 150 --ssh --direct
   vastai show instances-v1 --raw     # ALWAYS before any retry: a "failed" create can still exist
   ```
   Note `ssh_host`/`ssh_port` (or the direct IP/port).
2. Wait for sshd with a bounded, gentle loop. If it never answers, destroy and rent elsewhere:
   ```bash
   for i in $(seq 1 10); do ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -p PORT root@HOST true && break; sleep 20; done
   ```
3. Ship the setup script and start it (one scp, one ssh). **No private key is needed**: the repo is
   public and the box never pushes, so setup clones over https.
   ```bash
   scp -P PORT scripts/box/setup_rsc_box.sh root@HOST:/root/
   ssh -p PORT root@HOST 'sed -i "s/\r$//" /root/setup_rsc_box.sh && bash -n /root/setup_rsc_box.sh && tmux new-session -d -s setup "bash /root/setup_rsc_box.sh"'
   ```
   Stripping `\r` matters: a CRLF script dies silently inside tmux.
4. Poll every ~5 min until the readiness table prints (about 30-45 min):
   ```bash
   ssh -p PORT root@HOST 'tail -n 25 /root/setup.log'
   ```
   - `STOP:` lines (wrong GPU arch or driver): destroy the box now and rent another.
   - `cpu quota < 30`: destroy and re-rent.
   - `is not at origin/rsc-figure-fixes`: the box's checkout has local changes that block the
     fast-forward. Move them aside by hand (never `git reset`, never `git checkout --` a dirty
     file) and rerun; building envs on stale scripts would waste the hour.
   - `NOT READY`: fix the failure (the log says which step), rerun the same tmux command; it resumes.
   - The readiness table's `system_libblas` must be the reference build (`.../blas/libblas.so.3`)
     and `openblas_v4_dir` must not be `missing` (C1b depends on it); `env-mace.json` and
     `env-mattersim.json` record `schamodules_linkage`.
5. Pre-flight, then launch the orchestrator **once**:
   ```bash
   ssh -p PORT root@HOST 'R=/root/mlip-dynamic-stability/scripts/box/run_rsc_revision.sh; bash $R check && tmux new-session -d -s rsc "bash $R run"'
   ```
   `check` verifies setup readiness, the GPU, pw.x, SSSP, CRLF, and that every flag and subcommand the
   orchestrator uses appears in each study script's `--help`. `run` repeats it and refuses to start
   if it fails.

## 6. While it runs

- **Status** (every few hours): `ssh -p PORT root@HOST 'bash /root/mlip-dynamic-stability/scripts/box/run_rsc_revision.sh status'`.
  It shows stage markers, errors since the last start, QE done/total per prefix with a naive ETA,
  GPU utilisation and the CPU split.
- **Pull the GPU results as soon as the log says `GPU chain finished`** (C1, C1b, C2, C4, C5 are then
  complete), and after that about every 12 h. Boxes die; a pull is cheap:
  ```bash
  ssh -p PORT root@HOST 'bash /root/mlip-dynamic-stability/scripts/box/run_rsc_revision.sh pack'
  scp -P PORT 'root@HOST:/root/pull/rsc-pull-<stamp>.tar.gz*' .
  md5sum -c rsc-pull-<stamp>.tar.gz.md5 && tar -xzf rsc-pull-<stamp>.tar.gz --force-local
  ```
  The tarball holds `results/revision/**` (box logs under `results/revision/box_logs/`, the setup
  provenance under `box_logs/setup/`), and `results/ledger.box.{parquet,jsonl}` once C4 has run. It
  never overwrites the local ledger. QE scratch is excluded.
- **Relaunch after a crash, a reboot or a timed-out stage.** A live QE pass keeps the `rsc` tmux
  session alive after the orchestrator exits, so `new-session -s rsc` would fail as a duplicate:
  ```bash
  ssh -p PORT root@HOST 'R=/root/mlip-dynamic-stability/scripts/box/run_rsc_revision.sh; tmux has-session -t rsc 2>/dev/null && tmux new-window -t rsc -n run "bash $R run" || tmux new-session -d -s rsc "bash $R run"'
  ```
  Finished stages are skipped and the studies skip finished units; a running QE pass is left alone
  and the new driver waits for it. If the previous orchestrator is still alive (it waits on QE
  after the GPU chain), stop it first by PID (below); its QE pass keeps running in its own window.
  To force a stage, delete `/root/logs/stages/<stage>.done`.
- **C3b waits for final seeds.** `c3b-inputs` keeps the first selection it writes for a unit
  (`--force` would orphan PBE jobs), so the orchestrator writes C3b only when no C1 unit has seeds
  still to run (a failed seed is final: the study skips it without `--retry-failed`; a unit not
  started at all is added on a later launch). After a seed stage times out, the log says
  `C3b held back` and the b_ pass does not run; relaunch once to finish the seeds, and C3b and the
  b_ pass follow on that launch. An extxyz unit-check failure refuses C3b outright (inspect first).
- **C1b threads and BLAS.** C1b gets 8 threads while any QE pass is running or still due, and
  (quota - 2) only once the QE driver has finished. Its child process runs on threaded OpenBLAS
  (`LD_LIBRARY_PATH` to `openblas-pthread`; `V4_BLAS=reference` turns that off), so the dense
  14400 x 14400 LAPACK/BLAS step takes minutes, not hours. A timed-out v4 still exits 0, so the
  `sscha_v4` stage is marked done; a second attempt needs that marker deleted and a larger
  `V4_TIMEOUT_S` (the sscha script skips a v4 that timed out at the same cap; the stage cap follows
  it), and that is a decision to record.
- **Stopping things.** Never `pkill -f` a pattern (it matches the tmux server's argv and kills
  everything under it). By PID only:
  ```bash
  kill -TERM $(pgrep -P $(cat /root/logs/qe_queue.pid))                 # the QE pass (the queue kills its mpiruns)
  O=$(cat /root/logs/orchestrator.pid); pkill -TERM -P $O; kill -TERM $O  # the orchestrator, its running stage and the QE driver
  ```
- **Destroy the box the moment it is idle** (all done and pulled, or blocked on a human decision
  overnight): `vastai destroy instance <id> -y`, then `vastai show instances-v1 --raw` to confirm.

## 7. On return (local)

- [ ] Final `pack` -> `scp` -> `md5sum -c` -> extract. Then **destroy the box** and confirm.
- [ ] Read the `c4_coverage_*` lines in `box_logs/orchestrator.log`: all five at 57/57 means the box
      envs reproduce the deposited model versions. A shortfall is version drift to explain, not
      work to fill in.
- [ ] Ledger merge, only if the run was launched with `RUN_C4=1` and `results/ledger.box.parquet`
      arrived. The box may add only `harmonic_dispsweep` rows; append them through the project's
      own writer, never by copying the file over:
      ```python
      import json
      import pandas as pd
      from mlip_dynstab import ledger
      loc = pd.read_parquet("results/ledger.parquet"); box = pd.read_parquet("results/ledger.box.parquet")
      new = box[~box.uhash.isin(loc.uhash)]
      assert set(new.method) <= {"harmonic_dispsweep"}, set(new.method)
      shared = box[box.uhash.isin(loc.uhash)].set_index("uhash").sort_index()
      old = loc.set_index("uhash").loc[shared.index]
      prod = shared.method != "harmonic_dispsweep"               # production rows must be untouched
      for col in ("method", "model_version", "pred_stable", "min_freq_thz"):
          assert shared.loc[prod, col].astype(str).equals(old.loc[prod, col].astype(str)), col
      # sweep rows on both sides (same uhash, so never appended twice); if the box recomputed any,
      # their agreement is a free box-vs-local reproducibility check
      rec = ~prod
      print("sweep rows on both sides:", int(rec.sum()), " max |dfreq| THz:",
            float((shared.loc[rec, "min_freq_thz"] - old.loc[rec, "min_freq_thz"]).abs().max()) if rec.any() else 0)
      print(new.groupby(["model", "disp_ang"]).size())          # expect 19 per (model, amplitude)
      for row in json.loads(new.to_json(orient="records")):     # native types; NaN -> None, dropped
          ledger.record({k: v for k, v in row.items() if v is not None})
      ```
      Then delete `results/ledger.box.*`.
- [ ] Local analysis stages: `python scripts/sscha_seed_study.py --summarize`,
      `python scripts/force_spread.py --stage analyze`, `python scripts/dft_reference.py analyze`.
- [ ] Apply section 2's readings; fold in per todo Phase 4 (F1-F3).
- [ ] `python scripts/verify_claims.py` with an assertion for every new number (F4);
      `python scripts/stats_hardening.py --n-perm 10000` if the ledger changed;
      `python scripts/build_esi_tables.py` then `python scripts/build_esi_tables.py --check`.
- [ ] Provenance: `results/revision/box_logs/setup/readiness.json` and `freeze-<env>.txt` record the
      box envs (python, torch, CUDA, every pin). The DAS names them. Promote a freeze to
      `envs/lock-<model>-<date>.txt` only if it differs from the August lock on a science package.
- [ ] Tick C1-C6 in `tasks/todo.md`; add anything surprising to `tasks/lessons.md`.

## 8. Known traps

- The rebuilt envs use uv-managed Python 3.11/3.12 without the August conda base, so ~70 conda
  tooling packages are absent and their unpinned deps resolve fresh. Every *pinned* version is
  reproduced exactly: `compare_lock.py` in setup refuses the env if torch, numpy, scipy, ase,
  phonopy, spglib, e3nn, the model package or SSCHA differ.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set everywhere, as it was in August: mace-torch 0.3.16
  cannot load its checkpoint under torch >= 2.6 without it.
- **BLAS.** SSCHA is built against the system reference BLAS (`libblas-dev`), as in June/August, and
  the C1/C5 seeds run on it. Setup also installs the OpenBLAS runtime (`libopenblas0-pthread`, no
  `.pc` file, so meson still links the reference build) and pins Debian's `libblas.so.3` /
  `liblapack.so.3` alternatives back to the reference build, because installing OpenBLAS would
  otherwise switch every SSCHA process to it. Only the C1b include_v4 child gets OpenBLAS, by
  `LD_LIBRARY_PATH`: `get_odd_straight_with_v4` inverts and multiplies 14400 x 14400 matrices
  (dgetrf/dgetri + three dgemm, ~2e13 flop for 2x2x2 BaTiO3), which single-threaded reference BLAS
  needs hours for, about the 3 h cap, whatever the OpenMP thread count. The library changes, not
  the algorithm, and the child's v4=False round trip checks the swap numerically.
- The SSCHA build runs with a system-only PATH and no `PKG_CONFIG_PATH`, so the pytorch image's
  `/opt/conda/bin` (a conda pkg-config, BLAS or mpicc) cannot leak into the link.
- julia (`WITH_JULIA=1`) accelerates only python-sscha's Fourier-space gradient, not `get_v4` or
  the v4 linear algebra (checked in the 1.6.1 source), and importing it switches every Ensemble in
  that env onto a different code path. That is why it lives in its own env (`/root/env-mace-jl`);
  the orchestrator uses it only if `V4_PY` points there. The plan's "try the julia backend" for
  C1b is answered by this reading, not by a run.
- `dft_reference.py geom` exits 4 when a regenerated path misses its cached map (seen locally for
  BaTiO3 under MACE and CHGNet). The structures are written and flagged, so the orchestrator
  counts the stage as done with a `[flag]` line instead of holding back all of C3a; exit 3
  (deferred: MatterSim's pattern missing) still stops C3a.
- The QE driver waits for the C3b verdict before the b_ pass. If a_ drains first, QE idles until C1
  finishes (hours, not days). After the GPU chain, QE keeps its (quota - 8) cores for the passes
  already running; the 8 left over go to C1b, and after that sit idle.
- With `RUN_C4=1`, `run_disp_sweep.py` exits 0 even when units fail; the orchestrator reads its
  last `done:` line and leaves the stage unmarked if anything failed.
