#!/bin/bash
# GPU box, 2026-10-08: E3 pre-registered SSCHA replicates + E5 MACE-MP-0 re-run at 30 epochs.
#   E3: tasks/preregistration-repeats-2026-10-03.md (Deviation 2026-10-08): 65 units
#       (scripts/box/as_run/e3_units_2026-10-08.tsv) x {start B, start A --conv-seed 10} = 130 jobs
#       (+ one 1.0 THz retry of start B per complex-dynamical-matrix failure).
#   E5: tasks/preregistration-finetune-2026-10-03.md D4: MACE-MP-0 fine-tunes regenerated at 30
#       epochs, base + 3 replicate evaluations (c3a incl. the ax_ points, c3b, p2, s2), P3, summary,
#       all under results/revision/finetune_mace30/ (the 6-epoch results/revision/finetune/ is untouched).
# Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel on an RTX 3090/4090-class GPU (NOT Blackwell:
# the pinned torch has no sm_120 kernels). Ask for >= 32 vCPU (cgroup quota), >= 24 GB VRAM.
#
# Launch (after scp of this file to /root/):
#   tmux new -d -s gpu 'bash /root/gpu_e3_e5_2026-10-08.sh run; echo'
#   tmux new -d -s watchdog 'bash /root/gpu_e3_e5_2026-10-08.sh watchdog; echo'
#
# run:      bootstrap (apt, uv, clone, the five pinned envs, model smoke) -> checks (selection
#           re-derived and equal to the committed TSV, recipe identical to the grid's, one dry-run
#           replicate, E5 dataset byte-identical, regenerated MACE commands at 30 epochs, GPU) ->
#           /root/setup_done -> E5 chain in the background + the E3 scheduler (3/3/3/3/2 workers per
#           model, each queue longest unit first; slots a finished model frees go to ORB-v2 first,
#           then MACE-MP-0, up to 5 per model) -> E3 summary -> /root/QUEUE_DONE -> grace (60 min, `touch /root/pulled`
#           ends it) -> stops THIS instance. Any exit of the runner stops the instance (trap).
# watchdog: stops the instance if the runner's tmux session is gone; if no worker process has run
#           for 30 min after setup; if the grace window overran by 15 min; and at the 20 h cap.
# Stop = PUT state=stopped with the container's own CONTAINER_API_KEY (verified on box 1,
# 2026-10-04). Stop keeps the disk; destroy only after the pull.

set -u
SELF=$(readlink -f "$0")
R=/root/mlip-dynamic-stability
LOG=/root/logs/gpu_runner.log
E3=/root/e3
O=results/revision/finetune_mace30
CAP_S=$(( 20 * 3600 ))          # watchdog hard cap
GRACE_MIN=60
JOB_CAP=3h                      # outer wall cap per E3 job (the grid's longest unit: 134 min)
mkdir -p /root/logs /root/logs/e3 "$E3"

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

declare -A ENV=([mace_mp0]=mace [chgnet]=chgnet [sevennet0]=sevennet [mattersim]=mattersim [orb_v2]=orb)
MODELS_OTHER="mace_mp0 chgnet sevennet0 mattersim"

fail() { echo "SMOKE_FAIL: $* $(date -u)"; exit 1; }

# ------------------------------------------------------------------ E3: one job ----
json_field() {   # <python> <json> -> "status<TAB>error"
  "$1" - "$2" <<'PY'
import json, sys
try:
    r = (json.load(open(sys.argv[1])) or {}).get("run") or {}
    print(f"{r.get('status')}\t{str(r.get('error') or '')[:200]}")
except Exception as exc:
    print(f"unreadable\t{type(exc).__name__}: {exc}")
PY
}

run_job() {      # <kind B|B1|S10> <tag> <system> <model> <T> <sc> <env>
  local kind=$1 tag=$2 system=$3 model=$4 T=$5 sc=$6 env=$7 extra dir start t0 rc st
  case "$kind" in
    B)   dir=results/revision/sscha_converged_grid_startB;  start=B; extra=(--start B) ;;
    B1)  dir=results/revision/sscha_converged_grid_startB1; start=B; extra=(--start B --start-b-thz 1.0) ;;
    S10) dir=results/revision/sscha_converged_seed10;       start=A; extra=(--start A --conv-seed 10) ;;
  esac
  t0=$(date +%s)
  echo "$(date -u +%FT%TZ) START $kind $tag (worker $$)" >> "$E3/events.log"
  # shellcheck disable=SC2086
  timeout --kill-after=120 "$JOB_CAP" "/root/env-$env/bin/python" -u scripts/sscha_seed_study.py \
    --converge --system "$system" --model "$model" --T "$T" --supercell $sc --n-boot 10 \
    --unit-timeout 7200 --device cuda "${extra[@]}" --out-dir "$dir" \
    >> "/root/logs/e3/${kind}_${tag}.log" 2>&1 < /dev/null
  rc=$?
  st=$(json_field "/root/env-$env/bin/python" "$dir/${tag}_start${start}.json")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$kind" "$tag" "$rc" \
    "$(( $(date +%s) - t0 ))" "$st" >> "$E3/jobs.tsv"
  echo "$(date -u +%FT%TZ) END $kind $tag rc=$rc ${st%%$'\t'*}" >> "$E3/events.log"
  if [ "$kind" = B ] && [ "${st%%$'\t'*}" = failed ] && [[ "$st" == *"dynamical matrix is complex"* ]]; then
    echo "$(date -u +%FT%TZ) B_RETRY_1THZ $tag" >> "$E3/events.log"
    run_job B1 "$tag" "$system" "$model" "$T" "$sc" "$env"
  fi
}

worker() {       # <model> <k>: claims jobs of its model's queue until none is left
  local model=$1 k=$2 kind tag system m T sc env
  echo $$ > "$E3/pids/$model.$k"
  cd "$R" || exit 1
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  while IFS=$'\t' read -r kind tag system m T sc env <&3; do
    mkdir "$E3/claim/${kind}_$tag" 2> /dev/null || continue
    run_job "$kind" "$tag" "$system" "$m" "$T" "$sc" "$env"
  done 3< "$E3/queue_$model.txt"
  echo "$(date -u +%FT%TZ) WORKER_EXIT $model.$k" >> "$E3/events.log"
}

unclaimed() {
  local n=0 kind tag rest
  while IFS=$'\t' read -r kind tag rest; do [ -d "$E3/claim/${kind}_$tag" ] || n=$(( n + 1 )); done \
    < "$E3/queue_$1.txt"
  echo "$n"
}

live() {
  local n=0 f
  for f in "$E3/pids/$1".*; do
    [ -e "$f" ] || continue
    kill -0 "$(cat "$f")" 2> /dev/null && n=$(( n + 1 ))
  done
  echo "$n"
}

NEXTK=0
spawn() {
  NEXTK=$(( NEXTK + 1 ))
  setsid bash "$SELF" worker "$1" "$NEXTK" > /dev/null 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) spawn $1 k=$NEXTK" >> "$E3/events.log"
  sleep 8                                  # let it write its pid file and load the model
}

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -dc 0-9; }

e3_schedule() {  # <base workers per model> <ORB-v2 base> <max workers per model>
  # Base targets as the grid (3/3/3/3/2). Slots a finished model frees go to the models that still
  # have jobs, ORB-v2 first (its units are the slowest and set the wall time otherwise), then
  # MACE-MP-0 (the next longest queue), each up to <max>, the total never above the starting total,
  # and an extra worker only while the GPU has >= 4 GB free.
  local tgt=$1 orb0=$2 maxper=$3 total m u l busy left base free
  total=$(( 4 * tgt + orb0 ))
  while true; do
    busy=0; left=0
    for m in orb_v2 $MODELS_OTHER; do
      base=$tgt; [ "$m" = orb_v2 ] && base=$orb0
      u=$(unclaimed "$m"); l=$(live "$m"); left=$(( left + u ))
      while [ "$u" -gt 0 ] && [ "$l" -lt "$base" ]; do spawn "$m"; l=$(( l + 1 )); u=$(( u - 1 )); done
      busy=$(( busy + l ))
    done
    for m in orb_v2 mace_mp0 chgnet mattersim sevennet0; do
      u=$(unclaimed "$m"); l=$(live "$m")
      while [ "$u" -gt 0 ] && [ "$l" -lt "$maxper" ] && [ "$busy" -lt "$total" ]; do
        free=$(gpu_free_mb); [ "${free:-0}" -ge 4000 ] || break 2
        spawn "$m"; l=$(( l + 1 )); u=$(( u - 1 )); busy=$(( busy + 1 ))
      done
    done
    if [ "$left" -eq 0 ] && [ "$busy" -eq 0 ]; then
      echo "E3_QUEUE_EMPTY $(date -u)"; return 0
    fi
    sleep 60
  done
}

# ------------------------------------------------------------------ E5 chain ----
e5_chain() {
  cd "$R" || exit 1
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  local PY_MACE=/root/env-mace/bin/python line k rc=0
  echo "E5 start $(date -u)"
  # base evaluation needs no fine-tune: run it beside the three trainings
  $PY_MACE -u scripts/finetune_trial.py evaluate --out "$O" --model mace_mp0 --tag base --part all \
    --device cuda --workers 4 > /root/logs/e5_eval_base.log 2>&1 < /dev/null &
  local base_pid=$!
  k=0
  while IFS= read -r line; do
    ( eval "${line#run }" ) > "/root/logs/e5_train_seed$k.log" 2>&1 < /dev/null &
    echo "E5 train seed$k pid $! $(date -u)"
    k=$(( k + 1 ))
  done < <(grep -E '^run \$PY_MACE -m mace\.cli\.run_train ' "$O/train/box_sequence.sh")
  wait
  echo "E5 trainings + base eval done $(date -u)"
  for k in 0 1 2; do
    f=$O/models/mace_mp0/seed$k/ft_mace_mp0_seed$k.model
    if [ -s "$f" ]; then echo "E5 model seed$k ok $(sha256sum "$f")"
    else echo "E5 model seed$k MISSING (see /root/logs/e5_train_seed$k.log; NaN -> next seed is a human decision)"; rc=1; fi
    grep -E "Epoch [0-9]+|nan" "/root/logs/e5_train_seed$k.log" | tail -2
  done
  [ -s "$O/eval/c3a_mace_mp0_base.json" ] || { echo "E5 base eval missing"; rc=1; }
  for k in 0 1 2; do
    [ -s "$O/models/mace_mp0/seed$k/ft_mace_mp0_seed$k.model" ] || continue
    $PY_MACE -u scripts/finetune_trial.py evaluate --out "$O" --model mace_mp0 --tag "seed$k" \
      --part all --device cuda --workers 4 > "/root/logs/e5_eval_seed$k.log" 2>&1 < /dev/null \
      || { echo "E5 eval seed$k rc=$?"; rc=1; }
  done
  if [ -s "$O/models/mace_mp0/seed0/ft_mace_mp0_seed0.model" ]; then
    MLIP_DYNSTAB_CKPT_MACE_MP0=$R/$O/models/mace_mp0/seed0/ft_mace_mp0_seed0.model \
      timeout --kill-after=120 3h $PY_MACE -u scripts/sscha_seed_study.py --preset grid \
      --model mace_mp0 --system batio3_cubic --T 100 --start A --device cuda \
      --out-dir "$O/sscha_converged_grid" > /root/logs/e5_p3_sscha.log 2>&1 < /dev/null \
      || { echo "E5 P3 rc=$?"; rc=1; }
  fi
  $PY_MACE scripts/finetune_trial.py evaluate --part summary --out "$O" > /root/logs/e5_summary.log 2>&1 \
    || { echo "E5 summary rc=$?"; rc=1; }
  tail -3 /root/logs/e5_summary.log
  ( cd "$O/models" 2> /dev/null && find . -name '*.model' -o -name '*.pt' | sort | xargs -r sha256sum ) \
    > "$O/models/WEIGHTS.sha256" 2> /dev/null
  echo "E5_DONE rc=$rc $(date -u)"
  echo "$rc" > /root/e5_done
}

# ------------------------------------------------------------------ modes ----
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
    for spec in "mace 3.11" "chgnet 3.11" "sevennet 3.11" "orb 3.12" "mattersim 3.12"; do
      set -- $spec
      [ -x "/root/env-$1/bin/python" ] || bash scripts/box/as_run/env_build.sh "$1" "$2" > "/root/logs/env_$1.log" 2>&1 &
    done
    wait
    grep -h "ENVCHECK\|SSCHA_OK\|ENV_DONE" /root/logs/env_*.log
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    for m in mace_mp0 chgnet sevennet0 orb_v2 mattersim; do
      out=$("/root/env-${ENV[$m]}/bin/python" scripts/box/as_run/model_smoke.py "$m" 2>&1 | grep -E "SMOKE|Error" | tail -2)
      echo "$out"
      echo "$out" | grep -q SMOKE || fail "model smoke $m"
      "/root/env-${ENV[$m]}/bin/python" -c "import sscha, cellconstructor" || fail "sscha import in env-${ENV[$m]}"
    done
    echo "BOOTSTRAP_DONE $(date -u)"

    # ---- checks before anything is computed
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || fail "no GPU"
    CORES=$(cgroup_cores); echo "CORES $CORES (cgroup quota)"
    PYC=/root/env-chgnet/bin/python
    $PYC scripts/box/as_run/e3_select.py --out "$E3/units.tsv" || fail "e3_select"
    cmp "$E3/units.tsv" scripts/box/as_run/e3_units_2026-10-08.tsv || fail "selection differs from the committed TSV"
    $PYC - <<'PY' || fail "recipe check"
import importlib.util, json, sys
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("sss", "scripts/sscha_seed_study.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
g = json.load(open("results/revision/sscha_converged_grid/batio3_cubic_mace_mp0_100K_sc222_startA.json"))["recipe"]
base = ["--converge", "--system", "batio3_cubic", "--model", "mace_mp0", "--T", "100", "--n-boot", "10", "--unit-timeout", "7200"]
rb = m._clean(m.converged_recipe(m.parse_args(base + ["--start", "B"])))
rs = m._clean(m.converged_recipe(m.parse_args(base + ["--start", "A", "--conv-seed", "10"])))
r1 = m._clean(m.converged_recipe(m.parse_args(base + ["--start", "B", "--start-b-thz", "1.0"])))
assert rb == g, "start-B recipe differs from the grid's"
assert {k for k in rs if rs[k] != g.get(k)} == {"seed"} and rs["seed"] == 10, "seed-10 recipe"
assert {k for k in r1 if r1[k] != g.get(k)} == {"start_b_thz"}, "1.0 THz recipe"
print("RECIPE_OK (B == grid; seed10 differs only in seed; B1 only in start_b_thz)")
PY
    timeout 900 $PYC -u scripts/sscha_seed_study.py --converge --system batio3_cubic --model chgnet \
      --T 100 --supercell 2 2 2 --start B --n-boot 10 --unit-timeout 7200 --device cpu --dry-run \
      --out-dir "$E3/dryrun" > /root/logs/e3_dryrun.log 2>&1 || fail "E3 dry run (see /root/logs/e3_dryrun.log)"
    grep -E "^\[batio3" /root/logs/e3_dryrun.log | tail -2
    for d in sscha_converged_grid_startB sscha_converged_grid_startB1 sscha_converged_seed10; do
      if ls "results/revision/$d"/*_start[AB].json > /dev/null 2>&1; then
        echo "NOTE: results/revision/$d already holds JSONs (a relaunch?); finished ones are skipped"
      fi
    done
    n=$(ls -d results/revision/dft/qe/ax_*/ 2> /dev/null | wc -l)
    echo "ax_ job dirs in the tree: $n (E5 c3a covers them)"
    [ "$n" -eq 533 ] || fail "need the 533 ax_ outputs for the E5 held-out set"
    mkdir -p "$O/dataset"
    cp -r results/revision/finetune/dataset/mace_mp0 "$O/dataset/"
    for f in train.extxyz valid.extxyz labelled_all.extxyz dataset.json; do
      cmp "results/revision/finetune/dataset/mace_mp0/$f" "$O/dataset/mace_mp0/$f" || fail "dataset copy $f"
    done
    cp results/revision/finetune/manifest.json "$O/"
    /root/env-mace/bin/python scripts/finetune_trial.py train --emit --out "$O" > /root/logs/e5_emit.log 2>&1 \
      || fail "train --emit"
    nm=$(grep -cE '^run \$PY_MACE -m mace\.cli\.run_train .*--max_num_epochs=30( |$)' "$O/train/box_sequence.sh")
    nb=$(grep -E '^run \$PY_MACE -m mace\.cli\.run_train ' "$O/train/box_sequence.sh" | grep -cv -- '--max_num_epochs=30')
    echo "E5 MACE commands at 30 epochs: $nm (other: $nb)"
    [ "$nm" -eq 3 ] && [ "$nb" -eq 0 ] || fail "regenerated MACE commands are not 3 x 30 epochs"
    grep -q "finetune_mace30/dataset/mace_mp0/train.extxyz" "$O/train/box_sequence.sh" || fail "E5 paths"

    # ---- queues: per model, longest grid unit first, B then seed-10 of each unit
    rm -rf "$E3/claim" "$E3/pids"; mkdir -p "$E3/claim" "$E3/pids"
    for m in $MODELS_OTHER orb_v2; do
      awk -F'\t' -v m="$m" '!/^#/ && $3 == m {
          printf "B\t%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, $4, $5, $6
          printf "S10\t%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, $4, $5, $6 }' "$E3/units.tsv" > "$E3/queue_$m.txt"
      echo "queue $m: $(wc -l < "$E3/queue_$m.txt") jobs"
    done
    [ -f "$E3/jobs.tsv" ] || printf 'utc_end\tkind\ttag\trc\twall_s\tstatus\terror\n' > "$E3/jobs.tsv"
    touch /root/setup_done
    echo "SETUP_DONE $(date -u)"

    setsid bash "$SELF" e5 > /root/logs/e5.log 2>&1 < /dev/null &
    if [ "$CORES" -ge 32 ]; then e3_schedule 3 2 5; else e3_schedule 2 2 4; fi
    for i in $(seq 1 600); do [ -e /root/e5_done ] && break; sleep 60; done
    echo "E5 status: $(cat /root/e5_done 2> /dev/null || echo 'not done after 10 h wait')"
    cp "$E3/jobs.tsv" "$E3/events.log" /root/logs/ 2> /dev/null
    mkdir -p results/revision/e3_replicates
    cp "$E3/jobs.tsv" results/revision/e3_replicates/jobs.tsv
    /root/env-chgnet/bin/python scripts/box/as_run/e3_summarize.py > /root/logs/e3_summary.log 2>&1
    head -3 /root/logs/e3_summary.log
    touch /root/QUEUE_DONE
    echo "QUEUE_DONE $(date -u)"
    for i in $(seq 1 "$GRACE_MIN"); do [ -e /root/pulled ] && break; sleep 60; done
    echo "GRACE_END pulled=$([ -e /root/pulled ] && echo yes || echo no) $(date -u)"
    ;;
  worker) worker "$2" "$3" ;;
  e5) e5_chain ;;
  watchdog)
    echo "WATCHDOG start $(date -u)" >> "$LOG"
    t0=$(date +%s); idle=0
    while true; do
      sleep 300
      now=$(date +%s)
      if ! tmux has-session -t gpu 2> /dev/null; then self_stop "runner session gone"; exit 0; fi
      if [ $(( now - t0 )) -ge "$CAP_S" ]; then self_stop "20 h cap"; exit 0; fi
      if [ -e /root/QUEUE_DONE ]; then
        if [ $(( now - $(stat -c %Y /root/QUEUE_DONE) )) -ge $(( (GRACE_MIN + 15) * 60 )) ]; then
          self_stop "grace overran"; exit 0
        fi
        continue
      fi
      [ -e /root/setup_done ] || continue
      # anchored: matches the worker/E5 python processes, never this watchdog's own command line
      if pgrep -f "^/root/env-[a-z]+/bin/python" > /dev/null; then idle=0; else idle=$(( idle + 1 )); fi
      if [ "$idle" -ge 6 ]; then self_stop "no worker for 30 min"; exit 0; fi
    done
    ;;
  *) echo "usage: $0 run|watchdog   (worker <model> <k> and e5 are internal)" >&2; exit 2 ;;
esac
