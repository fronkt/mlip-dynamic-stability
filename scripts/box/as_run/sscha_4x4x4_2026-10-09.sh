#!/bin/bash
# GPU box, 2026-10-09: 4x4x4 converged SSCHA for SrTiO3 and one fluorite (Referee 3.2).
#   Pre-registration: tasks/preregistration-4x4x4-2026-10-09.md. Five units, start A, the grid's
#   recipe with --supercell 4 4 4, --unit-timeout 28800, outer cap 9 h each, all five in parallel.
#   Output: results/revision/sscha_converged_sc444/<tag>_startA.json
# Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel on an RTX 3090/4090-class GPU (NOT Blackwell).
# Ask for >= 32 vCPU (cgroup quota), >= 96 GB RAM, >= 24 GB VRAM.
#
# Launch (after scp of this file to /root/):
#   tmux new -d -s gpu 'bash /root/sscha_4x4x4_2026-10-09.sh run; echo'
#   tmux new -d -s watchdog 'bash /root/sscha_4x4x4_2026-10-09.sh watchdog; echo'
#
# run:      bootstrap (apt, uv, clone, four pinned envs, model smoke) -> recipe check (sc444 recipe ==
#           the grid's) -> /root/setup_done -> five units in parallel -> summary -> /root/QUEUE_DONE ->
#           grace (60 min, `touch /root/pulled` ends it) -> stops THIS instance. Any exit stops it (trap).
# watchdog: stops the instance if the runner's tmux session is gone; if no python worker has run for
#           30 min after setup; if the grace window overran by 15 min; and at the 12 h cap.

set -u
R=/root/mlip-dynamic-stability
LOG=/root/logs/runner.log
OUT=results/revision/sscha_converged_sc444
CAP_S=$(( 12 * 3600 ))
GRACE_MIN=60
JOB_CAP=9h
mkdir -p /root/logs

# tag  system  model  T  env   (the pre-registered five)
UNITS="srtio3_cubic chgnet 100 chgnet
srtio3_cubic mace_mp0 100 mace
srtio3_cubic mattersim 100 mattersim
srtio3_cubic sevennet0 100 sevennet
hfo2_cubic mace_mp0 600 mace"

self_stop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "SELF_STOP ($1) $(date -u) instance=$id" >> "$LOG"
  curl -s -w " http=%{http_code}\n" -X PUT -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" -d '{"state":"stopped"}' \
    "https://console.vast.ai/api/v0/instances/$id/" >> "$LOG" 2>&1
}

cgroup_cores() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    [ "$q" != "max" ] && { echo $(( q / p )); return; }
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -gt 0 ] && { echo $(( q / p )); return; }
  fi
  nproc
}

fail() { echo "SMOKE_FAIL: $* $(date -u)"; exit 1; }

case "${1:-}" in
  run)
    trap 'self_stop "runner exited"' EXIT
    exec >> "$LOG" 2>&1
    echo "RUN start $(date -u)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq gfortran libblas-dev liblapack-dev build-essential tmux rsync git curl \
      bzip2 pkg-config > /root/logs/apt.log 2>&1
    echo "APT_DONE rc=$?"
    command -v uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    uv python install 3.11 3.12
    cd /root || exit 1
    [ -d "$R" ] || git clone -b rsc-figure-fixes https://github.com/fronkt/mlip-dynamic-stability.git
    cd "$R" || exit 1
    git pull --ff-only || echo "NOTE: git pull failed; running the checked-out commit"
    git log --oneline -1
    for spec in "mace 3.11" "chgnet 3.11" "sevennet 3.11" "mattersim 3.12"; do
      set -- $spec
      [ -x "/root/env-$1/bin/python" ] || bash scripts/box/as_run/env_build.sh "$1" "$2" > "/root/logs/env_$1.log" 2>&1 &
    done
    wait
    grep -h "ENVCHECK\|SSCHA_OK\|ENV_DONE" /root/logs/env_*.log
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    for pair in "mace_mp0 mace" "chgnet chgnet" "sevennet0 sevennet" "mattersim mattersim"; do
      set -- $pair
      out=$("/root/env-$2/bin/python" scripts/box/as_run/model_smoke.py "$1" 2>&1 | grep -E "SMOKE|Error" | tail -2)
      echo "$out"
      echo "$out" | grep -q SMOKE || fail "model smoke $1"
      "/root/env-$2/bin/python" -c "import sscha, cellconstructor" || fail "sscha import in env-$2"
    done
    echo "BOOTSTRAP_DONE $(date -u)"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || fail "no GPU"
    free -g | head -2
    /root/env-chgnet/bin/python - <<'PY' || fail "recipe check"
import importlib.util, json, sys
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("sss", "scripts/sscha_seed_study.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
g = json.load(open("results/revision/sscha_converged_grid/srtio3_cubic_mace_mp0_100K_sc222_startA.json"))["recipe"]
a = ["--converge", "--system", "srtio3_cubic", "--model", "mace_mp0", "--T", "100", "--supercell", "4", "4", "4",
     "--n-boot", "10", "--unit-timeout", "28800", "--start", "A"]
r = m._clean(m.converged_recipe(m.parse_args(a)))
assert r == g, f"sc444 recipe differs from the grid's: {[k for k in r if r[k] != g.get(k)]}"
print("RECIPE_OK (sc444 == grid)")
PY
    touch /root/setup_done
    echo "SETUP_DONE $(date -u)"

    cores=$(cgroup_cores); thr=$(( cores / 5 )); [ "$thr" -lt 2 ] && thr=2; [ "$thr" -gt 8 ] && thr=8
    export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    printf 'utc_end\ttag\trc\twall_s\n' > /root/logs/jobs.tsv
    while read -r system model T env; do
      tag="${system}_${model}_${T}K_sc444"
      (
        t0=$(date +%s)
        timeout --kill-after=120 "$JOB_CAP" "/root/env-$env/bin/python" -u scripts/sscha_seed_study.py \
          --converge --system "$system" --model "$model" --T "$T" --supercell 4 4 4 --n-boot 10 \
          --unit-timeout 28800 --device cuda --start A --out-dir "$OUT" \
          > "/root/logs/$tag.log" 2>&1 < /dev/null
        rc=$?
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$tag" "$rc" "$(( $(date +%s) - t0 ))" >> /root/logs/jobs.tsv
      ) &
      echo "START $tag pid $! threads $thr $(date -u)"
      sleep 20
    done <<< "$UNITS"
    wait
    echo "ALL_UNITS_DONE $(date -u)"
    cat /root/logs/jobs.tsv
    mkdir -p "$OUT" && cp /root/logs/jobs.tsv "$OUT/jobs.tsv"
    /root/env-chgnet/bin/python - <<'PY' > /root/logs/summary.log 2>&1
import json, glob
for f in sorted(glob.glob("results/revision/sscha_converged_sc444/*_startA.json")):
    d = json.load(open(f)); r = d.get("run") or {}; c = d.get("converged") or d.get("result") or {}
    print(f.split("/")[-1], r.get("status"), json.dumps({k: c.get(k) for k in list(c)[:12]}, default=str)[:400])
PY
    cat /root/logs/summary.log
    touch /root/QUEUE_DONE
    echo "QUEUE_DONE $(date -u)"
    for i in $(seq 1 "$GRACE_MIN"); do [ -e /root/pulled ] && break; sleep 60; done
    echo "GRACE_END pulled=$([ -e /root/pulled ] && echo yes || echo no) $(date -u)"
    ;;
  watchdog)
    echo "WATCHDOG start $(date -u)" >> "$LOG"
    t0=$(date +%s); idle=0
    while true; do
      sleep 300
      now=$(date +%s)
      if ! tmux has-session -t gpu 2> /dev/null; then self_stop "runner session gone"; exit 0; fi
      if [ $(( now - t0 )) -ge "$CAP_S" ]; then self_stop "12 h cap"; exit 0; fi
      if [ -e /root/QUEUE_DONE ]; then
        if [ $(( now - $(stat -c %Y /root/QUEUE_DONE) )) -ge $(( (GRACE_MIN + 15) * 60 )) ]; then
          self_stop "grace overran"; exit 0
        fi
        continue
      fi
      [ -e /root/setup_done ] || continue
      if pgrep -f "^/root/env-[a-z]+/bin/python" > /dev/null; then idle=0; else idle=$(( idle + 1 )); fi
      if [ "$idle" -ge 6 ]; then self_stop "no worker for 30 min"; exit 0; fi
    done
    ;;
  *) echo "usage: $0 run|watchdog" >&2; exit 2 ;;
esac
