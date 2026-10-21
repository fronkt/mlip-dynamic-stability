#!/bin/bash
# GPU box, 2026-10-22: R2 multi-mode SSCHA centroid scan + the R2 lattice sweep.
#   Pre-registration: tasks/preregistration-centroid-2026-10-21.md (arms 1-4, gates, decision rule).
#   Code: scripts/sscha_centroid_scan.py -> results/revision/r2_centroid/ (one JSON per arm, gate/,
#         v4.json, summary.json); scripts/lattice_sweep.py (separate, post hoc) ->
#         results/revision/r2_lattice/<model>.json, one model env after another.
#   BEFORE LAUNCH: both scripts must be committed and pushed on rsc-figure-fixes (the box clones it).
# Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel on an RTX 3090/4090-class GPU (NOT Blackwell:
# the pinned torch has no sm_120 kernels). Ask for >= 36 vCPU (cgroup quota: the v4 child wants ~24
# threads beside three SSCHA arms and the lattice sweep), >= 64 GB RAM, >= 24 GB VRAM, >= 80 GB disk.
#
# Launch (after scp of this file to /root/). PRICE_PER_HR = this instance's $/h and CREDIT_USD = the
# account balance now; they only feed the cost and run-out time printed first (and a refusal to start
# when the credit covers less than 1.2 x the expected hours):
#   tmux new -d -s gpu 'PRICE_PER_HR=0.45 CREDIT_USD=30 bash /root/r2_centroid_2026-10-22.sh run; echo'
#   tmux new -d -s watchdog 'bash /root/r2_centroid_2026-10-22.sh watchdog; echo'
#
# run:      economics -> bootstrap (apt, uv, clone, the five pinned envs, model smoke) -> setup_done ->
#           selftest -> gate (patterns, M-band gate, atom mapping; plumbing failure stops the box) ->
#           smoke (three tiny SSCHA arms + a 300 s v4 in r2_centroid/_smoke, then --smoke-check; any
#           failure stops the box) -> SSCHA arms on the GPU, at most MAX_GPU_ARMS at once, in the order
#           m_scan Q0=0, free, gamma, then m_scan 0.075, 0.04, 0.10 (ARM_CAP each) -> v4 on the CPU as
#           soon as Q0=0 has saved its Hessian ensemble (V4_CAP) -> lattice sweep, five models in
#           sequence, beside the arms -> wait (QUEUE_CAP from setup_done: what still runs is killed) ->
#           summary -> /root/QUEUE_DONE (written last) -> grace (60 min; `touch /root/pulled` ends it)
#           -> stops THIS instance. Any exit of the runner stops the instance (trap).
# watchdog: stops the instance if the runner's tmux session is gone; if setup has not finished after
#           SETUP_CAP_MIN; if no python worker has run for 30 min after setup; if the grace window
#           overran by 15 min; and at the hard cap.
# Stop = PUT state=stopped with the container's own CONTAINER_API_KEY (tasks/lessons.md 2026-10-04;
# verified on box 1). Stop keeps the disk for the pull; destroy only after it.
#
# Progress: ssh -p $P $H 'tail -8 /root/logs/r2_runner.log; cat /root/logs/jobs.tsv'
#
# Pull + merge (Git Bash, laptop; ONLY after QUEUE_DONE, never while a job runs):
#   RAW=/c/Users/frank/mlip-rsc-revision-raw; cd /c/Users/frank/mlip-dynamic-stability
#   ssh -p $P $H 'test -e /root/QUEUE_DONE && tail -14 /root/logs/r2_summary.log' || echo "NOT DONE - do not pull"
#   ssh -p $P $H 'cd /root/mlip-dynamic-stability && tar czf - results/revision/r2_centroid results/revision/r2_lattice results/revision/r2_qe_lattice -C /root logs' > $RAW/r2_centroid_2026-10-22.tar.gz
#   gzip -t $RAW/r2_centroid_2026-10-22.tar.gz && tar tzf $RAW/r2_centroid_2026-10-22.tar.gz | wc -l && ssh -p $P $H 'touch /root/pulled'
#   tar xzf $RAW/r2_centroid_2026-10-22.tar.gz --exclude='*/work/*' --exclude='*/locks/*' --exclude='*/_smoke/*' --exclude='*/_dryrun/*' results/revision
#   grep -l '"status": "running"' results/revision/r2_centroid/*.json && echo "AN ARM WAS KILLED AT A CAP - report it as such"
#   python scripts/sscha_centroid_scan.py --summarize      # rebuilds summary.json locally (pure JSON, no sscha)
#   python scripts/qe_lattice_check.py analyze              # -> results/revision/r2_qe_lattice/summary.json
#   git status --short results/revision/r2_centroid results/revision/r2_lattice
#   git add results/revision/r2_centroid/*.json results/revision/r2_centroid/jobs.tsv results/revision/r2_centroid/summary.log results/revision/r2_centroid/gate results/revision/r2_lattice
#   git commit -m "R2 centroid scan (prereg 2026-10-21) and lattice sweep from the 10-22 GPU box" && git pull --rebase && git push
# The tarball keeps the work/ dirs (Hessian ensembles, dyn files) and the smoke run; they stay in $RAW.

set -u
R=/root/mlip-dynamic-stability
LOG=/root/logs/r2_runner.log
OUT=results/revision/r2_centroid
SMOKE=$OUT/_smoke
LAT=results/revision/r2_lattice
PY=/root/env-mace/bin/python
CS=scripts/sscha_centroid_scan.py
PRICE_PER_HR=${PRICE_PER_HR:-}
CREDIT_USD=${CREDIT_USD:-}
EXPECTED_H=6.5                  # setup ~0.8 h + gate/smoke ~0.4 h + v4 path (Q0=0 ensemble ~0.9 h, v4 ~4.1 h at 16 threads)
CAP_H=8                         # watchdog hard cap (credit was $4.27 on 2026-10-21; keep the worst case under it)
CAP_S=$(( CAP_H * 3600 ))
QUEUE_CAP_S=$(( 6 * 3600 ))     # global queue cap, from setup_done
SETUP_CAP_MIN=90
GRACE_MIN=60
MAX_GPU_ARMS=3
ARM_CAP=2.5h                    # outer cap per SSCHA arm (relax cap + Hessian ensemble + bootstrap)
ARM_RELAX_CAP_S=6600            # --unit-timeout: the relaxation's own wall cap inside ARM_CAP
V4_CAP_S=$(( 5 * 3600 ))        # outer cap of the v4 process; the child gets this minus 900 s (16 threads: ~4.1 h)
LAT_CAP_S=$(( 90 * 60 ))        # per lattice-sweep model
LAT_THR=2
QE_THR=4                        # pw.x MPI ranks for the R2 lattice check (scripts/qe_lattice_check.py)
QE_CAP_S=$(( 150 * 60 ))        # whole QE chain (install + 6 vc-relax)
M_GATE_FAIL=${M_GATE_FAIL:-run} # run | skip: the M arms if ONLY the M-band gate fails (prereg: not interpreted)
ARMS=("m_scan 0" "free 0.075" "gamma 0" "m_scan 0.075" "m_scan 0.04" "m_scan 0.1")
LAT_MODELS="mace_mp0 chgnet orb_v2 sevennet0 mattersim"
declare -A ENV=([mace_mp0]=mace [chgnet]=chgnet [sevennet0]=sevennet [mattersim]=mattersim [orb_v2]=orb)
mkdir -p /root/logs

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

economics() {
  echo "ECON expected ${EXPECTED_H} h; hard cap ${CAP_H} h; queue cap $(( QUEUE_CAP_S / 3600 )) h after setup"
  if [ -z "$PRICE_PER_HR" ]; then
    echo "ECON PRICE_PER_HR not set: cost and run-out time UNKNOWN (set both at launch)"
    return 0
  fi
  awk -v p="$PRICE_PER_HR" -v e="$EXPECTED_H" -v c="$CAP_H" \
    'BEGIN { printf "ECON at %.3f $/h: expected $%.2f, hard cap $%.2f\n", p, p * e, p * c }'
  if [ -z "$CREDIT_USD" ]; then
    echo "ECON CREDIT_USD not set: run-out time UNKNOWN"
    return 0
  fi
  local s
  s=$(awk -v c="$CREDIT_USD" -v p="$PRICE_PER_HR" 'BEGIN { printf "%d", 3600 * c / p }')
  echo "ECON credit \$$CREDIT_USD at this price alone runs out $(date -u -d "@$(( $(date +%s) + s ))" +%FT%TZ)" \
    "(in $(( s / 3600 )) h $(( s % 3600 / 60 )) min)"
  [ "$s" -lt "$CAP_S" ] && echo "ECON WARNING: the credit runs out BEFORE the ${CAP_H} h hard cap"
  if awk -v s="$s" -v e="$EXPECTED_H" 'BEGIN { exit !(s < 1.2 * e * 3600) }'; then
    fail "credit covers less than 1.2 x the expected ${EXPECTED_H} h; top up first"
  fi
}

run_py() {      # <timeout> <threads> <log> <script args...>: one centroid-scan process in env-mace
  local cap=$1 thr=$2 log=$3
  shift 3
  env OMP_NUM_THREADS="$thr" MKL_NUM_THREADS="$thr" OPENBLAS_NUM_THREADS="$thr" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    timeout --kill-after=120 "$cap" "$PY" -u "$CS" "$@" >> "$log" 2>&1 < /dev/null
}

job_row() {     # <name> <rc> <t0>
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$(( $(date +%s) - $3 ))" >> /root/logs/jobs.tsv
}

remaining_s() { echo $(( DEADLINE - $(date +%s) )); }

launch_arm() {  # <arm> <q0>
  local arm=$1 q0=$2 name="arm_${1}_$2"
  (
    t0=$(date +%s)
    run_py "$ARM_CAP" "$ARM_THR" "/root/logs/$name.log" --arm "$arm" --q0 "$q0" --out-dir "$OUT" \
      --device cuda --threads "$ARM_THR" --unit-timeout "$ARM_RELAX_CAP_S"
    job_row "$name" $? "$t0"
  ) &
  ARM_PIDS+=("$!")
  echo "START $name pid $! threads $ARM_THR $(date -u)"
}

live_arms() {
  local n=0 p
  for p in "${ARM_PIDS[@]}"; do kill -0 "$p" 2> /dev/null && n=$(( n + 1 )); done
  echo "$n"
}

v4_waiter() {   # waits for the Q0 = 0 Hessian ensemble, then the v4 arm on the CPU
  local marker=$OUT/work/m_scan_Q0.0000/ENSEMBLE_SAVED.json t0 cap rc
  until [ -e "$marker" ]; do
    if awk -F'\t' '$2 == "arm_m_scan_0"' /root/logs/jobs.tsv | grep -q .; then
      [ -e "$marker" ] && break
      echo "V4_SKIPPED: the Q0 = 0 arm ended without a saved Hessian ensemble $(date -u)"
      return 0
    fi
    if [ "$(remaining_s)" -le 0 ]; then echo "V4_SKIPPED: queue cap $(date -u)"; return 0; fi
    sleep 60
  done
  cap=$(remaining_s); [ "$cap" -gt "$V4_CAP_S" ] && cap=$V4_CAP_S
  if [ "$cap" -lt 3600 ]; then echo "V4_SKIPPED: only ${cap} s left before the queue cap"; return 0; fi
  echo "V4_START $(date -u) threads $V4_THR outer cap ${cap} s child cap $(( cap - 900 )) s"
  t0=$(date +%s)
  run_py "${cap}s" "$V4_THR" /root/logs/v4.log --arm v4 --out-dir "$OUT" --threads "$V4_THR" \
    --v4-timeout "$(( cap - 900 ))"
  rc=$?
  job_row v4 "$rc" "$t0"
  echo "V4_END rc=$rc $(date -u)"
  grep -E "^\[v4\]" /root/logs/v4.log | tail -2
}

qe_chain() {    # R2 comment 2: is our PBE lattice converged? QE 7.5 + SSSP 1.3 as phase A, 6 vc-relax
  local d t0 rc
  timeout "${QE_CAP_S}s" bash -c '
    set -u
    bash scripts/box/as_run/qe_setup_2026-10-03.sh > /root/logs/qe_setup.log 2>&1
    grep -q QE_SETUP_DONE /root/logs/qe_setup.log || { echo "QE_SETUP_FAILED"; exit 1; }
    export PATH=/root/qe/bin:$PATH ESPRESSO_PSEUDO=/root/sssp OMP_NUM_THREADS=1
    for d in results/revision/r2_qe_lattice/qe/r2l_*/; do
      [ -f "$d/pw.out" ] && grep -q "JOB DONE" "$d/pw.out" && { echo "QE_SKIP $d (done)"; continue; }
      t0=$(date +%s)
      ( cd "$d" && mpirun --allow-run-as-root --bind-to none -np '"$QE_THR"' pw.x -nk 2 -in pw.in \
          > pw.out 2> pw.err < /dev/null )
      echo "QE_JOB $d rc=$? $(( $(date +%s) - t0 )) s $(date -u)"
      rm -rf "$d/scratch"
    done
    echo "QE_DONE $(date -u)"
  '
  rc=$?
  echo "QE_CHAIN_END rc=$rc $(date -u)"
}

lattice_chain() {  # one model env after another; skipped cleanly if the script is not in the clone
  local m e t0 rc cap dev
  if [ ! -f scripts/lattice_sweep.py ]; then
    echo "LATTICE_MISSING: scripts/lattice_sweep.py is not in the clone; lattice sweep skipped $(date -u)"
    return 0
  fi
  mkdir -p "$LAT"
  for m in $LAT_MODELS; do
    e=${ENV[$m]}
    if [ ! -x "/root/env-$e/bin/python" ] || grep -qx "$m" /root/logs/smoke_failed_models 2> /dev/null; then
      echo "LATTICE_SKIP $m: env-$e missing or its model smoke failed"
      continue
    fi
    cap=$(remaining_s); [ "$cap" -gt "$LAT_CAP_S" ] && cap=$LAT_CAP_S
    if [ "$cap" -lt 600 ]; then echo "LATTICE_DEADLINE: $m and later skipped $(date -u)"; return 0; fi
    dev=()
    "/root/env-$e/bin/python" scripts/lattice_sweep.py --help 2> /dev/null | grep -q -- "--device" \
      && dev=(--device cuda)
    t0=$(date +%s)
    env OMP_NUM_THREADS=$LAT_THR MKL_NUM_THREADS=$LAT_THR OPENBLAS_NUM_THREADS=$LAT_THR \
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      timeout --kill-after=120 "${cap}s" "/root/env-$e/bin/python" -u scripts/lattice_sweep.py \
      --model "$m" --out "$LAT/$m.json" --precision production "${dev[@]}" > "/root/logs/lattice_$m.log" 2>&1 < /dev/null
    rc=$?
    job_row "lattice_$m" "$rc" "$t0"
    echo "LATTICE $m rc=$rc $(( $(date +%s) - t0 )) s $(date -u)"
  done
  echo "LATTICE_DONE $(date -u)"
}

case "${1:-}" in
  run)
    trap 'self_stop "runner exited"' EXIT
    exec >> "$LOG" 2>&1
    echo "RUN start $(date -u)"
    rm -f /root/QUEUE_DONE /root/pulled /root/setup_done
    economics

    # ---- bootstrap (idempotent: a restarted box skips what exists)
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
    [ -f "$CS" ] || fail "$CS is not in the clone: commit and push it first"
    for spec in "mace 3.11" "chgnet 3.11" "sevennet 3.11" "orb 3.12" "mattersim 3.12"; do
      read -r e v <<< "$spec"
      [ -x "/root/env-$e/bin/python" ] || bash scripts/box/as_run/env_build.sh "$e" "$v" > "/root/logs/env_$e.log" 2>&1 &
    done
    wait
    grep -h "ENVCHECK\|SSCHA_OK\|ENV_DONE" /root/logs/env_*.log
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    : > /root/logs/smoke_failed_models
    for m in mace_mp0 chgnet sevennet0 orb_v2 mattersim; do
      out=$("/root/env-${ENV[$m]}/bin/python" scripts/box/as_run/model_smoke.py "$m" 2>&1 | grep -E "SMOKE|Error" | tail -2)
      echo "$out"
      if ! echo "$out" | grep -q SMOKE; then
        [ "$m" = mace_mp0 ] && fail "model smoke mace_mp0"
        echo "$m" >> /root/logs/smoke_failed_models
        echo "NOTE: model smoke $m failed; its lattice sweep is skipped"
      fi
    done
    "$PY" -c "import sscha, cellconstructor" || fail "sscha import in env-mace"
    echo "BOOTSTRAP_DONE $(date -u)"
    touch /root/setup_done
    DEADLINE=$(( $(stat -c %Y /root/setup_done) + QUEUE_CAP_S ))
    echo "SETUP_DONE $(date -u); queue cap at $(date -u -d "@$DEADLINE" +%FT%TZ)"

    # ---- checks before anything is computed
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || fail "no GPU"
    CORES=$(cgroup_cores)
    if [ "$CORES" -ge 44 ]; then ARM_THR=4; else ARM_THR=2; fi
    V4_THR=$(( CORES - MAX_GPU_ARMS * ARM_THR - LAT_THR - QE_THR - 2 )); [ "$V4_THR" -lt 8 ] && V4_THR=8
    echo "CORES $CORES (cgroup quota): $MAX_GPU_ARMS arms x $ARM_THR threads, v4 $V4_THR, lattice $LAT_THR, QE $QE_THR"
    [ "$CORES" -ge 24 ] || fail "need >= 24 cores for arms + v4 + QE (have $CORES)"
    qe_chain > /root/logs/qe_chain.log 2>&1 &
    QE_PID=$!
    echo "QE chain pid $QE_PID $(date -u)"
    free -g | head -2
    [ -f /root/logs/jobs.tsv ] || printf 'utc_end\tjob\trc\twall_s\n' > /root/logs/jobs.tsv
    "$PY" "$CS" --selftest > /root/logs/selftest.log 2>&1 || fail "selftest (see /root/logs/selftest.log)"
    tail -1 /root/logs/selftest.log

    # ---- gate: patterns, the M-band gate, the mapping onto the cellconstructor supercell
    t0=$(date +%s)
    run_py 40m 8 /root/logs/gate.log --arm gate --device cuda --out-dir "$OUT" --threads 8
    GATE_RC=$?
    job_row gate "$GATE_RC" "$t0"
    grep -E "^\[gate" /root/logs/gate.log | tail -3
    case "$GATE_RC" in
      0) echo "GATE_OK $(date -u)" ;;
      3) echo "GATE_M_FAIL $(date -u): the M band missed -5.404 +/- 0.02 THz; the M arms are not" \
              "interpreted (prereg); M_GATE_FAIL=$M_GATE_FAIL" ;;
      *) fail "gate plumbing (rc $GATE_RC; see /root/logs/gate.log and $OUT/gate/gate.json)" ;;
    esac

    # ---- smoke: the python-sscha behaviour the scan relies on, on a tiny recipe, before 5 GPU-hours
    if [ -e /root/smoke_passed ]; then
      echo "SMOKE already passed on this box; skipped"
    else
      rm -rf "$SMOKE"
      SMK=(--out-dir "$SMOKE" --gate-dir "$OUT/gate" --conv-n-configs 100 --conv-max-pop 2
           --conv-n-hessian 100 --conv-steps-per-pop 30 --n-boot 2 --unit-timeout 600 --device cuda)
      t0=$(date +%s)
      for spec in "m_scan 0" "m_scan 0.075" "free 0.075"; do
        read -r a q <<< "$spec"
        run_py 30m "$ARM_THR" "/root/logs/smoke_${a}_$q.log" --arm "$a" --q0 "$q" --threads "$ARM_THR" "${SMK[@]}" &
      done
      wait
      run_py 15m "$V4_THR" /root/logs/smoke_v4.log --arm v4 --v4-n 16 --v4-timeout 300 \
        --threads "$V4_THR" "${SMK[@]}"
      "$PY" "$CS" --smoke-check "${SMK[@]}" > /root/logs/smoke_check.log 2>&1
      rc=$?
      job_row smoke "$rc" "$t0"
      cat /root/logs/smoke_check.log
      [ "$rc" -eq 0 ] || fail "smoke check (see /root/logs/smoke_*.log)"
      touch /root/smoke_passed
      echo "SMOKE_PASSED $(date -u)"
    fi

    # ---- production: SSCHA arms on the GPU (slots), v4 on the CPU, lattice sweep beside them
    ARM_PIDS=()
    v4_waiter > /root/logs/v4_waiter.log 2>&1 &
    V4_PID=$!
    LAT_PID=
    for spec in "${ARMS[@]}"; do
      read -r arm q0 <<< "$spec"
      if [ "$GATE_RC" -eq 3 ] && [ "$M_GATE_FAIL" = skip ] \
         && { [ "$arm" = free ] || { [ "$arm" = m_scan ] && [ "$q0" != 0 ]; }; }; then
        echo "SKIP arm_${arm}_$q0 (M-band gate failed, M_GATE_FAIL=skip)"
        continue
      fi
      while [ "$(live_arms)" -ge "$MAX_GPU_ARMS" ] && [ "$(remaining_s)" -gt 0 ]; do sleep 30; done
      if [ "$(remaining_s)" -le 0 ]; then echo "QUEUE_CAP: arm_${arm}_$q0 and later not started"; break; fi
      launch_arm "$arm" "$q0"
      sleep 20
      if [ -z "$LAT_PID" ] && [ "${#ARM_PIDS[@]}" -ge "$MAX_GPU_ARMS" ]; then
        lattice_chain > /root/logs/lattice_chain.log 2>&1 &
        LAT_PID=$!
        echo "LATTICE chain pid $LAT_PID $(date -u)"
      fi
    done
    if [ -z "$LAT_PID" ]; then
      lattice_chain > /root/logs/lattice_chain.log 2>&1 &
      LAT_PID=$!
    fi
    while :; do
      alive=0
      for p in "${ARM_PIDS[@]}" "$V4_PID" "$LAT_PID" "$QE_PID"; do kill -0 "$p" 2> /dev/null && alive=$(( alive + 1 )); done
      [ "$alive" -eq 0 ] && break
      if [ "$(remaining_s)" -le 0 ]; then
        echo "QUEUE_CAP_HIT $(date -u): stopping what still runs"
        pkill -TERM -f "scripts/sscha_centroid_scan.py|scripts/lattice_sweep.py|pw.x"
        sleep 90
        pkill -KILL -f "scripts/sscha_centroid_scan.py|scripts/lattice_sweep.py|pw.x"
        break
      fi
      sleep 60
    done
    wait
    echo "ALL_JOBS_DONE $(date -u)"
    cat /root/logs/v4_waiter.log /root/logs/lattice_chain.log /root/logs/qe_chain.log
    cat /root/logs/jobs.tsv

    # ---- summary, then the end marker (last), then the pull window
    "$PY" "$CS" --summarize --out-dir "$OUT" > /root/logs/r2_summary.log 2>&1
    tail -14 /root/logs/r2_summary.log
    cp /root/logs/jobs.tsv "$OUT/jobs.tsv"
    cp /root/logs/r2_summary.log "$OUT/summary.log"
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
      if [ $(( now - t0 )) -ge "$CAP_S" ]; then self_stop "${CAP_H} h cap"; exit 0; fi
      if [ -e /root/QUEUE_DONE ]; then
        if [ $(( now - $(stat -c %Y /root/QUEUE_DONE) )) -ge $(( (GRACE_MIN + 15) * 60 )) ]; then
          self_stop "grace overran"; exit 0
        fi
        continue
      fi
      if [ ! -e /root/setup_done ]; then
        if [ $(( now - t0 )) -ge $(( SETUP_CAP_MIN * 60 )) ]; then self_stop "setup not done after ${SETUP_CAP_MIN} min"; exit 0; fi
        continue
      fi
      # anchored: the env pythons (arms, v4 and its spawned child, lattice), never this watchdog
      if pgrep -f "^/root/env-[a-z]+/bin/python|pw.x|micromamba" > /dev/null; then idle=0; else idle=$(( idle + 1 )); fi
      if [ "$idle" -ge 6 ]; then self_stop "no worker for 30 min"; exit 0; fi
    done
    ;;
  *) echo "usage: $0 run|watchdog" >&2; exit 2 ;;
esac
