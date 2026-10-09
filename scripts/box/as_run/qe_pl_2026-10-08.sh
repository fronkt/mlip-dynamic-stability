#!/bin/bash
# CPU/QE box, 2026-10-08: E2 pbe-lattice phases B and C (scripts/dft_checks.py), the only E2 stages
# never run. Phase A (PBE vc-relax, 3 jobs) is committed; B needs A, C needs B's force constants,
# so the box runs B, generates C from B's outputs, then runs C.
#   B: 9 finite-displacement 40-atom supercells at the PBE lattice (3 per system: BaTiO3, KNbO3,
#      CsSnBr3), inputs byte-checked against scripts/box/as_run/pl_phaseB_pwin_2026-10-08.sha256.
#   C: frozen-mode profiles along the PBE eigenvector at the deciding q and the softest q,
#      10 amplitudes each, 30-60 jobs of 5-10 atoms (fewer where the softest mode is the deciding one).
# Any CPU image with curl + bzip2 (e.g. ubuntu 22.04); >= 32 cores (cgroup quota), >= 64 GB RAM.
#
# Launch (after scp of this file to /root/):
#   tmux new -d -s qe 'bash /root/qe_pl_2026-10-08.sh run; echo'
#   tmux new -d -s watchdog 'bash /root/qe_pl_2026-10-08.sh watchdog; echo'
#
# run:      apt + uv + clone -> QE 7.5 (conda-forge, micromamba) + SSSP 1.3 efficiency (md5-checked,
#           scripts/box/as_run/qe_setup_2026-10-03.sh) -> /root/env-pl (numpy/ase/phonopy/pymatgen,
#           the versions that generated the reference hashes) -> QE smoke SCF -> phase B inputs +
#           hash check -> /root/setup_done -> B queue (oxides, then CsSnBr3 at -k 1) -> phase C
#           inputs from B -> C queue -> /root/QUEUE_DONE -> grace (60 min, `touch /root/pulled`
#           ends it) -> stops THIS instance. Any exit of the runner stops the instance (trap).
# watchdog: stops the instance if the runner's tmux session is gone; if no pw.x / input generator
#           has run for 30 min after setup; if the grace window overran by 15 min; and at the 6 h cap.

set -u
R=/root/mlip-dynamic-stability
LOG=/root/logs/qe_pl_runner.log
CK=results/revision/dft_checks
CAP_S=$(( 6 * 3600 ))
GRACE_MIN=60
PY=/root/env-pl/bin/python
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

mem_gb() {
  local m
  if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
    m=$(cat /sys/fs/cgroup/memory.max)
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    m=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
  fi
  local t=$(( $(sed -n 's/^MemTotal: *\([0-9]*\) kB/\1/p' /proc/meminfo) * 1024 ))
  [ -n "${m:-}" ] && [ "$m" -lt "$t" ] && t=$m
  echo $(( t / 1024 / 1024 / 1024 ))
}

fail() { echo "SMOKE_FAIL: $* $(date -u)"; exit 1; }

Q() { bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir /root/sssp --root "$CK/qe" "$@" < /dev/null; }

# done = JOB DONE, SCF converged, and pw.in.ran == pw.in (pw.x ran THIS input)
n_done() {
  local n=0 d
  for d in $CK/qe/$1; do
    [ -f "$d/pw.in" ] || continue
    [ -f "$d/pw.out" ] && grep -q 'JOB DONE' "$d/pw.out" && cmp -s "$d/pw.in" "$d/pw.in.ran" \
      && grep -q 'convergence has been achieved' "$d/pw.out" && n=$(( n + 1 ))
  done
  echo "$n"
}
n_jobs() { local n=0 d; for d in $CK/qe/$1; do [ -f "$d/pw.in" ] && n=$(( n + 1 )); done; echo "$n"; }

case "${1:-}" in
  run)
    trap 'self_stop "runner exited"' EXIT
    exec >> "$LOG" 2>&1
    echo "RUN start $(date -u)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq tmux rsync git curl bzip2 ca-certificates python3 > /root/logs/apt.log 2>&1
    echo "APT_DONE rc=$?"
    command -v uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    cd /root || exit 1
    [ -d "$R" ] || git clone -b rsc-figure-fixes https://github.com/fronkt/mlip-dynamic-stability.git
    cd "$R" || exit 1
    git pull --ff-only || echo "NOTE: git pull failed; running the checked-out commit"
    git log --oneline -1
    bash scripts/box/as_run/qe_setup_2026-10-03.sh > /root/logs/qe_setup.log 2>&1
    grep -E "QE_DONE|SSSP_CHECK|QE_SETUP_DONE|FAIL" /root/logs/qe_setup.log
    grep -q "SSSP_CHECK ok" /root/logs/qe_setup.log && [ -x /root/qe/bin/pw.x ] || fail "QE/SSSP setup"
    if [ ! -x "$PY" ]; then
      uv python install 3.12
      uv venv --python 3.12 /root/env-pl
      uv pip install --python "$PY" numpy==2.3.5 scipy==1.17.1 spglib==2.7.0 pyyaml==6.0.3 \
        ase==3.28.0 phonopy==3.5.1 pymatgen==2026.5.4 pymatgen-core==2026.10.2 \
        > /root/logs/env_pl.log 2>&1 || fail "env-pl"
      uv pip install --python "$PY" --no-deps -e "$R" >> /root/logs/env_pl.log 2>&1 || fail "env-pl repo"
    fi
    $PY -c "import numpy, ase, phonopy, pymatgen.core, mlip_dynstab; print('ENV_PL', numpy.__version__, ase.__version__, phonopy.__version__)" \
      || fail "env-pl import"

    CORES=$(cgroup_cores); MEM=$(mem_gb)
    echo "CORES $CORES (cgroup quota)  MEM ${MEM} GB"
    [ "$CORES" -ge 8 ] || fail "need >= 8 cores, have $CORES"
    # QE smoke: 5-atom cubic BaTiO3 SCF, the production pseudos, 8 ranks
    mkdir -p /root/qesmoke && cp scripts/box/as_run/qe_smoke_bto.in /root/qesmoke/pw.in
    ( cd /root/qesmoke && export PATH=/root/qe/bin:$PATH LD_LIBRARY_PATH=/root/qe/lib OMP_NUM_THREADS=1 \
        ESPRESSO_PSEUDO=/root/sssp OMPI_MCA_btl_vader_single_copy_mechanism=none \
      && timeout 900 mpirun --allow-run-as-root --bind-to none -np 8 pw.x -nk 4 -in pw.in > pw.out 2> pw.err < /dev/null )
    grep -E "^!|JOB DONE|convergence has been achieved" /root/qesmoke/pw.out
    grep -q "JOB DONE" /root/qesmoke/pw.out && grep -q "convergence has been achieved" /root/qesmoke/pw.out \
      || fail "QE smoke SCF (see /root/qesmoke)"

    # phase A must be committed and converged; B inputs are generated here and hash-checked
    for s in batio3_cubic knbo3_cubic cssnbr3_cubic; do
      grep -q "JOB DONE" "$CK/qe/pl_${s}_vcrelax/pw.out" || fail "phase A $s missing"
    done
    $PY scripts/dft_checks.py pbe-lattice --phase B --sssp-dir /root/sssp > /root/logs/pl_gen_B.log 2>&1
    rc=$?; tail -5 /root/logs/pl_gen_B.log
    [ "$rc" -eq 0 ] || fail "phase B generation rc=$rc"
    nb=$(n_jobs "pl_*_fd*"); echo "phase B jobs: $nb"
    [ "$nb" -eq 9 ] || fail "expected 9 phase-B jobs, have $nb"
    if ( cd "$CK/qe" && grep -v '^#' "$R/scripts/box/as_run/pl_phaseB_pwin_2026-10-08.sha256" | sha256sum -c --quiet ); then
      echo "PHASE_B_INPUTS byte-identical to the 2026-10-08 local generation"
    else
      echo "WARNING: phase-B pw.in differ from the 2026-10-08 local generation (env drift?); running the box's"
    fi
    touch /root/setup_done
    echo "SETUP_DONE $(date -u)"

    # ---- B: oxides (6 jobs, 2 pools), then CsSnBr3 (3 jobs, 1 pool: k-pools duplicate memory)
    r1=$(( CORES / 6 )); [ "$r1" -gt 16 ] && r1=16; [ "$r1" -lt 1 ] && r1=1
    r2=$(( CORES / 3 )); [ "$r2" -gt 24 ] && r2=24; [ "$r2" -lt 1 ] && r2=1
    n1=6; n2=3
    if [ "$MEM" -lt 48 ]; then n1=3; n2=1; r1=$(( CORES / 3 )); [ "$r1" -gt 16 ] && r1=16; r2=$CORES; [ "$r2" -gt 24 ] && r2=24; fi
    echo "B: oxides -r $r1 -n $n1 -k 2, CsSnBr3 -r $r2 -n $n2 -k 1"
    for pass in 1 2; do
      Q --filter '^pl_(batio3|knbo3)_cubic_fd[0-9]+$' -r "$r1" -n "$n1" -k 2 --log /root/logs/qe_pl_B.log
    done
    for pass in 1 2; do
      Q --filter '^pl_cssnbr3_cubic_fd[0-9]+$' -r "$r2" -n "$n2" -k 1 --log /root/logs/qe_pl_B.log
    done
    echo "QE_B_EXIT done $(n_done "pl_*_fd*")/9 $(date -u)"

    # ---- C: needs every B job; generated here from B's force constants
    $PY scripts/dft_checks.py pbe-lattice --phase C --sssp-dir /root/sssp > /root/logs/pl_gen_C.log 2>&1
    rc=$?; tail -12 /root/logs/pl_gen_C.log
    nc=$(n_jobs "pl_*_prof*"); echo "phase C generation rc=$rc, jobs: $nc"
    if [ "$nc" -gt 0 ]; then
      for pass in 1 2; do
        Q --filter '^pl_.*_prof(soft)?_i[0-9]+$' -r 8 --log /root/logs/qe_pl_C.log
      done
    else
      echo "PHASE_C_NOT_RUN (generation rc=$rc; a system whose B jobs are not all ok is DEFERRED)"
    fi
    echo "QE_C_EXIT done $(n_done "pl_*_prof*")/$nc $(date -u)"
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
      if ! tmux has-session -t qe 2> /dev/null; then self_stop "runner session gone"; exit 0; fi
      if [ $(( now - t0 )) -ge "$CAP_S" ]; then self_stop "6 h cap"; exit 0; fi
      if [ -e /root/QUEUE_DONE ]; then
        if [ $(( now - $(stat -c %Y /root/QUEUE_DONE) )) -ge $(( (GRACE_MIN + 15) * 60 )) ]; then
          self_stop "grace overran"; exit 0
        fi
        continue
      fi
      [ -e /root/setup_done ] || continue
      if pgrep -x pw.x > /dev/null || pgrep -f "^/root/env-pl/bin/python" > /dev/null; then idle=0
      else idle=$(( idle + 1 )); fi
      if [ "$idle" -ge 6 ]; then self_stop "no pw.x for 30 min"; exit 0; fi
    done
    ;;
  *) echo "usage: $0 run|watchdog" >&2; exit 2 ;;
esac
