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
