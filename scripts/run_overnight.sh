#!/bin/bash
# Overnight SSCHA cross-validation: gold-standard multi-mode SSCHA on the perovskite + fluorite
# families (where the fast single-mode `softmode` method is the primary result), at the SAME
# T-ladder as softmode so the two methods can be compared row-for-row. Resumable: recorded
# (system,model,T) units skip. Waits for the in-flight bcc Ti/Hf grid first to avoid GPU thrash.
cd /root/mlip-dynamic-stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f run_sscha_tihf.sh >/dev/null 2>&1; do sleep 60; done
echo "TIHF grid clear; starting overnight SSCHA $(date)"

declare -A PY=(
  [mace_mp0]=/venv/main/bin/python
  [sevennet0]=/root/env-sevennet/bin/python
  [mattersim]=/root/env-mattersim/bin/python
  [chgnet]=/root/env-chgnet/bin/python
  [orb_v2]=/root/env-orb/bin/python
)
# 5 perovskites spanning Tc 105-763 K + the 2 fluorites; same ladder as the softmode grid.
SYSTEMS="srtio3_cubic batio3_cubic pbtio3_cubic knbo3_cubic cssni3_cubic zro2_cubic hfo2_cubic"
for S in $SYSTEMS; do
  for M in mattersim sevennet0 mace_mp0 chgnet orb_v2; do
    for T in 100 300 600 900; do
      echo "===== $M $S T=$T $(date) ====="
      # per-run cap: 40-atom perovskite SSCHA can hit minimizer step-collapse; kill thrashers
      # at 7 min so the queue keeps moving (skipped runs retry later, resumable).
      timeout 420 ${PY[$M]} -u -m mlip_dynstab.cli --system "$S" --model "$M" --method sscha --T "$T" \
        || echo "[error] $M/$S/T=$T (timeout-or-fail)"
    done
  done
done
echo "OVERNIGHT_DONE $(date)"
