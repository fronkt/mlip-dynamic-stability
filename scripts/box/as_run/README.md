# Scripts as actually run on the RSC-revision box (vast.ai 52872517, 2026-09-27)

The reviewed orchestrator (`scripts/box/run_rsc_revision.sh`, `setup_rsc_box.sh`) was NOT used end to end;
the box was driven stage by stage with these scripts, copied back verbatim from `/root/` on the box.
Box: RTX 3090, cgroup quota 61.4 vCPU (cgroup v1: `/sys/fs/cgroup/cpu/cpu.cfs_quota_us`), Ubuntu 22.04,
image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, repo cloned over https at /root/mlip-dynamic-stability.

| Script | What it did | Stage |
|---|---|---|
| box_base.sh | apt (gfortran, BLAS/LAPACK), uv, micromamba + conda-forge QE 7.5 at /root/qe | setup |
| env_build.sh `<model> <py>` | /root/env-<model> from envs/lock-<model>-2026-08-17.txt (conda `@ file://` lines rewritten to `==ver` or dropped; torch cu124 index) | setup |
| model_smoke.py | forces on rattled Si for every model/checkpoint; pulls checkpoints | setup |
| qe_test.sh | 5-atom BaTiO3 SCF smoke test (13 s on 16 ranks) | setup |
| run_c4.sh | displacement sweep, four models, sequential (one ledger writer) | C4 |
| run_c2.sh | force_spread fc -> configs -> forces (5 models + 3 committee ckpts) -> analyze | C2 |
| run_c1.sh `<env> <threads>` | sscha_seed_study --preset revision (C1 + C5) | C1/C5 |
| run_c3a.sh | dft_reference geom/mlip-eval (5 envs) -> qe-inputs (k 0.25/0.15 1/A) -> qe_queue '^a_' | C3a |
| run_c3b.sh | dft_reference c3b-inputs + mlip-eval -> waits for C3a -> qe_queue '^b_' | C3b |
| c1c_smoke.sh, c1c_run.sh `<env> <A|B>` | converged SSCHA (--preset converged), four processes | C1c |
| c1c_bto_b.sh, c1c_bto_b1.sh | BaTiO3 start-B retries (0.3 THz failed on a complex-dyn assertion; 1.0 THz ran) | C1c |

SSSP 1.3.0 PBE efficiency came from the Materials Cloud API route
`https://archive.materialscloud.org/api/records/rcyfm-68h65/files/<file>/content` (the old
`record/file?...&parent_id=19` URL returns "Page not found").

## 2026-10-08 runners (built, not launched): E3 replicates + E5 MACE 30 epochs (GPU), pbe-lattice B/C (QE)

| File | What |
|---|---|
| gpu_e3_e5_2026-10-08.sh `run\|watchdog` | GPU box: 5 envs, checks, E3 (130 jobs + 1.0 THz retries) beside E5 (3 MACE fine-tunes at 30 epochs, 4 evaluations, P3, summary), E3 summary, QUEUE_DONE, 60 min grace, self-stop; watchdog 30 min idle / 20 h cap |
| e3_select.py, e3_units_2026-10-08.tsv | the pre-registered selection (replace=False, Deviation 2026-10-08): 53 disagreeing + 12 random = 65 units |
| e3_summarize.py | the prereg's reported quantities -> results/revision/e3_replicates/summary.{json,csv} |
| qe_pl_2026-10-08.sh `run\|watchdog` | QE box: QE 7.5 + SSSP, env-pl, smoke SCF, phase B (9 jobs, hash-checked), phase C generated from B on the box (30-60 jobs), QUEUE_DONE, 60 min grace, self-stop; watchdog 30 min idle / 6 h cap |
| pl_phaseB_pwin_2026-10-08.sha256, qe_smoke_bto.in | phase-B input hashes from the local generation; the QE smoke input |

Launch (one line each; `P`/`H` = the box's ssh port and host):

    scp -P $P scripts/box/as_run/gpu_e3_e5_2026-10-08.sh $H:/root/ && ssh -p $P $H "tmux new -d -s gpu 'bash /root/gpu_e3_e5_2026-10-08.sh run; echo' && tmux new -d -s watchdog 'bash /root/gpu_e3_e5_2026-10-08.sh watchdog; echo'"
    scp -P $P scripts/box/as_run/qe_pl_2026-10-08.sh $H:/root/ && ssh -p $P $H "tmux new -d -s qe 'bash /root/qe_pl_2026-10-08.sh run; echo' && tmux new -d -s watchdog 'bash /root/qe_pl_2026-10-08.sh watchdog; echo'"

Progress: `ssh -p $P $H 'tail -5 /root/logs/gpu_runner.log; tail -3 /root/e3/events.log; wc -l /root/e3/jobs.tsv'`
(QE: `tail -5 /root/logs/qe_pl_runner.log /root/logs/qe_pl_B.log /root/logs/qe_pl_C.log`).

Pull + merge, GPU (Git Bash, laptop; only after QUEUE_DONE):

    RAW=/c/Users/frank/mlip-rsc-revision-raw; cd /c/Users/frank/mlip-dynamic-stability
    ssh -p $P $H 'test -e /root/QUEUE_DONE && cat /root/e5_done && head -3 /root/logs/e3_summary.log' || echo "NOT DONE - do not pull"
    ssh -p $P $H 'cd /root/mlip-dynamic-stability && tar czf - $(ls -d results/revision/sscha_converged_grid_startB* results/revision/sscha_converged_seed10 results/revision/e3_replicates results/revision/finetune_mace30) -C /root logs e3' > $RAW/gpu_e3_e5_2026-10-08.tar.gz
    gzip -t $RAW/gpu_e3_e5_2026-10-08.tar.gz && tar tzf $RAW/gpu_e3_e5_2026-10-08.tar.gz | wc -l && ssh -p $P $H 'touch /root/pulled'
    tar xzf $RAW/gpu_e3_e5_2026-10-08.tar.gz --exclude='*/work/*' --exclude='*/locks/*' --exclude='*.model' --exclude='*.pt' --exclude='*.pth.tar' results/revision
    mkdir -p results/revision/finetune_mace30/logs && tar xzf $RAW/gpu_e3_e5_2026-10-08.tar.gz --wildcards --strip-components=1 -C results/revision/finetune_mace30/logs 'logs/e5_*.log'
    python scripts/box/as_run/e3_summarize.py          # status counts: any 'running' = killed at the 3 h cap, report it as such
    git status --short results/revision | grep -E '\.(model|pt|pth\.tar)$' && echo "WEIGHTS STAGED - stop"
    git add results/revision/sscha_converged_grid_startB results/revision/sscha_converged_seed10 results/revision/e3_replicates results/revision/finetune_mace30
    [ -d results/revision/sscha_converged_grid_startB1 ] && git add results/revision/sscha_converged_grid_startB1
    git commit -m "E3 replicates (65 units: start B + seed 10) and E5 MACE-MP-0 30-epoch re-run from the 10-08 GPU box" && git pull --rebase && git push

Pull + merge, QE (only after QUEUE_DONE; merges a job only if JOB DONE, SCF converged and pw.in.ran == pw.in,
and refuses one whose pw.in differs from a pw.in already in the repo):

    RAW=/c/Users/frank/mlip-rsc-revision-raw; REPO=/c/Users/frank/mlip-dynamic-stability; cd $REPO
    ssh -p $P $H 'test -e /root/QUEUE_DONE && grep -E "QE_B_EXIT|QE_C_EXIT|PHASE_C" /root/logs/qe_pl_runner.log' || echo "NOT DONE - do not pull"
    ssh -p $P $H 'cd /root/mlip-dynamic-stability && tar czf - results/revision/dft_checks/qe/pl_*_fd* results/revision/dft_checks/qe/pl_*_prof* results/revision/dft_checks/pbe_lattice -C /root logs' > $RAW/qe_pl_2026-10-08.tar.gz
    gzip -t $RAW/qe_pl_2026-10-08.tar.gz && ssh -p $P $H 'touch /root/pulled'
    X=$RAW/qe_pl_extract_2026-10-08; mkdir -p $X && tar xzf $RAW/qe_pl_2026-10-08.tar.gz -C $X
    ok=0; bad=0; for d in $X/results/revision/dft_checks/qe/pl_*_fd* $X/results/revision/dft_checks/qe/pl_*_prof*; do j=${d##*/}; t=$REPO/results/revision/dft_checks/qe/$j
      if grep -q 'JOB DONE' $d/pw.out && grep -q 'convergence has been achieved' $d/pw.out && cmp -s $d/pw.in $d/pw.in.ran && { [ ! -f $t/pw.in ] || cmp -s $d/pw.in $t/pw.in; }; then
        mkdir -p $t && cp $d/pw.in $d/pw.in.ran $d/pw.out $d/job.json $t/ && ok=$((ok+1)); else echo "NOT MERGED $j"; bad=$((bad+1)); fi; done; echo "merged $ok, not merged $bad"
    cp -r $X/results/revision/dft_checks/pbe_lattice results/revision/dft_checks/
    python scripts/dft_checks.py analyze-checks --workers 1        # Windows: the process pool dies, use 1
    git add results/revision/dft_checks/qe/pl_* results/revision/dft_checks/pbe_lattice results/revision/dft_checks/*.csv results/revision/dft_checks/*.json
    git commit -m "E2 pbe-lattice phases B+C from the 10-08 QE box (JOB DONE, SCF converged, pw.in.ran == pw.in) + analyze-checks" && git pull --rebase && git push
