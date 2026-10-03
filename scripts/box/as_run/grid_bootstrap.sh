#!/bin/bash
# Converged-SSCHA grid box (vast.ai, 2026-10-03): base layer without QE, repo clone, the five
# pinned envs in parallel (env_build.sh), then a forces smoke test per env. Logs in /root/logs/.
set -x
mkdir -p /root/logs
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq gfortran libblas-dev liblapack-dev build-essential tmux rsync git curl bzip2 pkg-config > /root/apt.log 2>&1
echo "APT_DONE rc=$?"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv python install 3.11 3.12
cd /root
[ -d mlip-dynamic-stability ] || git clone -b rsc-figure-fixes https://github.com/fronkt/mlip-dynamic-stability.git
cd /root/mlip-dynamic-stability && git log --oneline -1
for spec in "mace 3.11" "chgnet 3.11" "sevennet 3.11" "orb 3.12" "mattersim 3.12"; do
  set -- $spec
  bash scripts/box/as_run/env_build.sh "$1" "$2" > /root/logs/env_$1.log 2>&1 &
done
wait
grep -h "ENVCHECK\|SSCHA_OK\|ENV_DONE" /root/logs/env_*.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
for pair in "mace mace_mp0" "chgnet chgnet" "sevennet sevennet0" "orb orb_v2" "mattersim mattersim"; do
  set -- $pair
  /root/env-$1/bin/python scripts/box/as_run/model_smoke.py "$2" 2>&1 | grep -E "SMOKE|Error" | tail -2
done
echo "BOOTSTRAP_DONE $(date -u)"
