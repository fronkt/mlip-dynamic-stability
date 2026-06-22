#!/bin/bash
# Stage-4 revision compute queue (referee M5 finite-size convergence + m6 SSCHA reproducibility).
# Waits for the in-flight fluorite grid to clear so it does not contend for the GPU, then runs a
# small, targeted set of jobs. Convergence units go to the ledger (distinct supercell hash);
# the reproducibility study prints to the log only.
cd /root/mlip-dynamic-stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f run_overnight.sh >/dev/null 2>&1; do sleep 60; done
echo "GRID CLEAR; starting revision compute $(date)"

declare -A PY=(
  [mace_mp0]=/venv/main/bin/python
  [sevennet0]=/root/env-sevennet/bin/python
  [mattersim]=/root/env-mattersim/bin/python
  [chgnet]=/root/env-chgnet/bin/python
  [orb_v2]=/root/env-orb/bin/python
)
MACE=${PY[mace_mp0]}

run() { echo "===== $* $(date) ====="; timeout 1800 "$@" || echo "[error] $* (timeout-or-fail)"; }

# ---- M5a: softmode finite-size sign robustness (perovskite 2x2x2 -> 3x3x3; bcc 6x6x6 -> 3x3x3) ----
run $MACE -u -m mlip_dynstab.cli --system srtio3_cubic --model mace_mp0 --method softmode --T 100 --supercell 3 3 3 --force
run $MACE -u -m mlip_dynstab.cli --system batio3_cubic --model mace_mp0 --method softmode --T 100 --supercell 3 3 3 --force
run $MACE -u -m mlip_dynstab.cli --system zr_bcc       --model mace_mp0 --method softmode --T 100 --supercell 3 3 3 --force

# ---- M5b: SSCHA finite-size sign robustness ----
# bcc-Zr (27-atom, cheap): does the dynamic-stabilization margin survive 2x2x2 -> 3x3x3?
run $MACE -u -m mlip_dynstab.cli --system zr_bcc --model mace_mp0 --method sscha --T 100 --supercell 3 3 3 --force
run $MACE -u -m mlip_dynstab.cli --system zr_bcc --model mace_mp0 --method sscha --T 300 --supercell 3 3 3 --force
# SrTiO3 (135-atom, slow): does the SSCHA false-stable PERSIST at a larger cell? (critical for M4)
run $MACE -u -m mlip_dynstab.cli --system srtio3_cubic --model mace_mp0 --method sscha --T 100 --supercell 3 3 3 --force

# ---- m6: SSCHA stochastic reproducibility (bcc-Zr/mace/100K, 4 seeds) ----
echo "===== sscha_repro zr_bcc mace 100 $(date) ====="
timeout 1800 $MACE -u scripts/sscha_repro.py zr_bcc mace_mp0 100 0,1,2,3 || echo "[error] repro"

echo "REVISION_COMPUTE_DONE $(date)"
