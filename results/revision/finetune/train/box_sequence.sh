#!/bin/bash
# Pre-registered fine-tuning trial: tasks/preregistration-finetune-2026-10-03.md
# Settings: scripts/finetune/mace_medium_finetune.json, scripts/finetune/chgnet_finetune.json
# Run the STEPS in order (each is resumable).  This file is a runbook, not one unattended job:
# step 1 is ~1.2k core-hours of pw.x, the rest ~4 GPU-hours.
cd /root/mlip-dynamic-stability
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { echo "===== $* $(date -u)"; "$@"; rc=$?; [ $rc -ne 0 ] && echo "[rc=$rc] $*"; return 0; }
PY_MACE=/root/env-mace/bin/python; PY_CHG=/root/env-chgnet/bin/python

# STEP 0. inputs (generated locally, NOT committed): results/revision/finetune/{manifest.json,configs/,fc/}
#         and results/revision/dft/qe/ft_*/ (246 dirs). Copy them to the box (rsync) or commit them first, then:
ls results/revision/dft/qe | grep -c '^ft_'          # expect 246

# STEP 1. PBE single points (qe_queue.sh unchanged; 246 jobs, ~1.2k core-h: ~21 h at 56 cores, ~29 h at 40)
export ESPRESSO_PSEUDO=/root/sssp   # the directory that holds the UPF files (it was /root/sssp/upf for C3a/C3b)
bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir $ESPRESSO_PSEUDO -r 8 -n 5 --filter '^ft_' \
    --log /root/logs/qe_ft.log < /dev/null

# STEP 2. training / validation files (needs the base model: one run per env)
run $PY_MACE -u scripts/finetune_trial.py dataset --model mace_mp0 --device cuda
run $PY_CHG  -u scripts/finetune_trial.py dataset --model chgnet   --device cuda

# STEP 3. fine-tunes: MACE-MP-0 medium x seeds 0,1,2 (~0.3 h each on the 3090, float64), CHGNet x 3 (~0.1 h each)
run $PY_MACE -m mace.cli.run_train --name=ft_mace_mp0_seed0 --seed=0 --work_dir=results/revision/finetune/models/mace_mp0/seed0 --train_file=results/revision/finetune/dataset/mace_mp0/train.extxyz --valid_file=results/revision/finetune/dataset/mace_mp0/valid.extxyz --device=cuda --foundation_model=medium --multiheads_finetuning=False --energy_weight=10.0 --forces_weight=10.0 --E0s=estimated --lr=0.001 --weight_decay=0.0 --scaling=rms_forces_scaling --batch_size=2 --max_num_epochs=6 --ema --ema_decay=0.999 --amsgrad --clip_grad=1.0 --default_dtype=float64 --foundation_model_elements=True --energy_key=REF_energy --forces_key=REF_forces --save_cpu
run $PY_MACE -m mace.cli.run_train --name=ft_mace_mp0_seed1 --seed=1 --work_dir=results/revision/finetune/models/mace_mp0/seed1 --train_file=results/revision/finetune/dataset/mace_mp0/train.extxyz --valid_file=results/revision/finetune/dataset/mace_mp0/valid.extxyz --device=cuda --foundation_model=medium --multiheads_finetuning=False --energy_weight=10.0 --forces_weight=10.0 --E0s=estimated --lr=0.001 --weight_decay=0.0 --scaling=rms_forces_scaling --batch_size=2 --max_num_epochs=6 --ema --ema_decay=0.999 --amsgrad --clip_grad=1.0 --default_dtype=float64 --foundation_model_elements=True --energy_key=REF_energy --forces_key=REF_forces --save_cpu
run $PY_MACE -m mace.cli.run_train --name=ft_mace_mp0_seed2 --seed=2 --work_dir=results/revision/finetune/models/mace_mp0/seed2 --train_file=results/revision/finetune/dataset/mace_mp0/train.extxyz --valid_file=results/revision/finetune/dataset/mace_mp0/valid.extxyz --device=cuda --foundation_model=medium --multiheads_finetuning=False --energy_weight=10.0 --forces_weight=10.0 --E0s=estimated --lr=0.001 --weight_decay=0.0 --scaling=rms_forces_scaling --batch_size=2 --max_num_epochs=6 --ema --ema_decay=0.999 --amsgrad --clip_grad=1.0 --default_dtype=float64 --foundation_model_elements=True --energy_key=REF_energy --forces_key=REF_forces --save_cpu
run $PY_CHG scripts/finetune/chgnet_finetune_run.py --dataset-dir=results/revision/finetune/dataset/chgnet --seed=0 --out-dir=results/revision/finetune/models/chgnet/seed0 --device=cuda
run $PY_CHG scripts/finetune/chgnet_finetune_run.py --dataset-dir=results/revision/finetune/dataset/chgnet --seed=1 --out-dir=results/revision/finetune/models/chgnet/seed1 --device=cuda
run $PY_CHG scripts/finetune/chgnet_finetune_run.py --dataset-dir=results/revision/finetune/dataset/chgnet --seed=2 --out-dir=results/revision/finetune/models/chgnet/seed2 --device=cuda
# A fine-tune that ends in NaN is re-run once with the same settings and the next unused seed
# (pre-registration, 'Analysis') and the failure recorded; e.g. CHGNet seed 3:
# run $PY_CHG scripts/finetune/chgnet_finetune_run.py --dataset-dir=results/revision/finetune/dataset/chgnet --seed=3 --out-dir=results/revision/finetune/models/chgnet/seed3 --device=cuda

# STEP 4. evaluation: base (the foundation model, same code path) and each replicate;
#         parts c3a (P1, S1), c3b (S1), p2 (P2 screen pipeline), s2 (controls); ~0.3 h per model/tag
run $PY_MACE -u scripts/finetune_trial.py evaluate --model mace_mp0 --tag base --part all --device cuda --workers 8
run $PY_MACE -u scripts/finetune_trial.py evaluate --model mace_mp0 --tag seed0 --part all --device cuda --workers 8
run $PY_MACE -u scripts/finetune_trial.py evaluate --model mace_mp0 --tag seed1 --part all --device cuda --workers 8
run $PY_MACE -u scripts/finetune_trial.py evaluate --model mace_mp0 --tag seed2 --part all --device cuda --workers 8
run $PY_CHG -u scripts/finetune_trial.py evaluate --model chgnet --tag base --part all --device cuda --workers 8
run $PY_CHG -u scripts/finetune_trial.py evaluate --model chgnet --tag seed0 --part all --device cuda --workers 8
run $PY_CHG -u scripts/finetune_trial.py evaluate --model chgnet --tag seed1 --part all --device cuda --workers 8
run $PY_CHG -u scripts/finetune_trial.py evaluate --model chgnet --tag seed2 --part all --device cuda --workers 8

# STEP 5. P3: converged SSCHA, BaTiO3 100 K, seed-0 fine-tuned MACE-MP-0 (start A, grid recipe; ~0.5 h)
MLIP_DYNSTAB_CKPT_MACE_MP0=$(pwd)/results/revision/finetune/models/mace_mp0/seed0/ft_mace_mp0_seed0.model \
  $PY_MACE -u scripts/sscha_seed_study.py --preset grid --model mace_mp0 --system batio3_cubic --T 100 \
  --start A --device cuda --out-dir results/revision/finetune/sscha_converged_grid 2>&1 | tee /root/logs/ft_p3_sscha.log

# STEP 6. summary (any env, no GPU) -> results/revision/finetune/summary.json
run /root/env-mace/bin/python scripts/finetune_trial.py evaluate --part summary
# then pull results/revision/finetune/{dataset,models,eval,p2cache,sscha_converged_grid,summary.json} and
# results/revision/dft/qe/ft_*/pw.out back; models/ are not committed unless you decide to (MACE ~45 MB each).
