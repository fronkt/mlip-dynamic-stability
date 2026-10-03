#!/bin/bash
# P3: converged SSCHA (the grid recipe, start A) on BaTiO3 100 K with the seed-0 fine-tuned MACE-MP-0.
# scripts/sscha_seed_study.py is NOT edited: it loads its calculator through
# mlip_dynstab.calculators.get_calculator, which honours MLIP_DYNSTAB_CKPT_MACE_MP0 (unset in production).
# Writes only under --out-dir (never the production grid directory, never the ledger).
cd /root/mlip-dynamic-stability
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
MLIP_DYNSTAB_CKPT_MACE_MP0=$(pwd)/results/revision/finetune/models/mace_mp0/seed0/ft_mace_mp0_seed0.model \
  /root/env-mace/bin/python -u scripts/sscha_seed_study.py --preset grid --model mace_mp0 \
  --system batio3_cubic --T 100 --start A --device cuda --out-dir results/revision/finetune/sscha_converged_grid \
  2>&1 | tee /root/logs/ft_p3_sscha.log
# estimated 0.5 h on a free GPU (cap 2.2 h: the relaxation cap is 7200 s)
# result: results/revision/finetune/sscha_converged_grid/batio3_cubic_mace_mp0_100K_sc222_startA.json  -> run.hessian.dynamically_stable
# summarise: python scripts/finetune_trial.py evaluate --part summary
