#!/usr/bin/env bash
# Provision one rented vast.ai box for the RSC-revision compute (tasks/todo.md Phase 1, C1-C6).
#
# The August re-measurement ran in five pinned environments (envs/lock-<model>-2026-08-17.txt) and
# the revision numbers are only comparable to the deposited ones if they run in the same five. So
# every env is rebuilt from its lock, torch included, and the script STOPS rather than substitute a
# torch build when the GPU cannot run the pinned one: torch 2.6.0+cu124 (mace/chgnet/sevennet) has
# no Blackwell kernels, and torch 2.13.0 (orb/mattersim) is a CUDA 13.0 build that needs driver
# >= 580 and compute capability >= 7.5.
#
# Idempotent: each step checks whether it is already done, so a dropped ssh session or a half-built
# box is fixed by running the script again. Everything is logged to /root/setup.log; provenance
# (pins, freezes, per-env smoke JSON, readiness.json) goes to /root/logs/setup/, which the pull
# brings home. /root/setup.ready is written only when every check passes.
#
#   bash setup_rsc_box.sh                  # everything
#   WITH_JULIA=1 bash setup_rsc_box.sh     # also build /root/env-mace-jl (read the julia note first)
#   ALLOW_GPU=1 bash setup_rsc_box.sh      # skip the nvidia-smi gate (the per-env CUDA test still runs)
#
# Clones over ssh when /root/.ssh/id_ed25519 exists, otherwise over https: the repo is public and
# the box never pushes, so the private key does not need to leave the laptop.

set -uo pipefail

REPO_URL=${REPO_URL:-git@github.com:fronkt/mlip-dynamic-stability.git}
REPO_HTTPS=${REPO_HTTPS:-https://github.com/fronkt/mlip-dynamic-stability.git}
BRANCH=${BRANCH:-rsc-figure-fixes}
REPO=${REPO:-/root/mlip-dynamic-stability}
LOCK_DATE=${LOCK_DATE:-2026-08-17}
QE_PREFIX=${QE_PREFIX:-/root/qe}
QE_SPEC=${QE_SPEC:-qe=7.5}           # the conda-forge build that ran cleanly on vast before
SSSP_DIR=${SSSP_DIR:-/root/sssp}
# The locks pin CellConstructor + python-sscha in all five envs, but only these two run SSCHA in
# the revision (C1, C1b, C5); finite_t.py imports them lazily, so the other three never touch them.
SSCHA_ENVS=${SSCHA_ENVS:-"mace mattersim"}
WITH_JULIA=${WITH_JULIA:-0}
LOG=/root/setup.log
PROV=/root/logs/setup

TORCH_CU124_INDEX=https://download.pytorch.org/whl/cu124
# SSSP 1.3.0 PBE efficiency, Materials Cloud Archive doi 10.24435/materialscloud:f3-ym
# (record rcyfm-68h65). md5s are the archive's own, checked 2026-09-26.
SSSP_API=https://archive.materialscloud.org/api/records/rcyfm-68h65/files
SSSP_TAR=SSSP_1.3.0_PBE_efficiency.tar.gz
SSSP_TAR_MD5=a58f1b3373f330179fd0832c48bb9a52
SSSP_JSON=SSSP_1.3.0_PBE_efficiency.json
SSSP_JSON_MD5=3153c4b20fc90a44fba0236627525644
SSSP_ELEMENTS="Sr Ba Ti O Zr"

# name | lock | python | torch source | model key | checkpoints to fetch ("-" = the default only)
# Python versions are the August ones: 3.11 for the cu124 envs, 3.12 for orb/mattersim.
ENV_SPECS=(
  "mace      mace      3.11 cu124 mace_mp0  small,medium,large"
  "chgnet    chgnet    3.11 cu124 chgnet    -"
  "sevennet  sevennet  3.11 cu124 sevennet0 -"
  "orb       orb       3.12 pypi  orb_v2    -"
  "mattersim mattersim 3.12 pypi  mattersim mattersim-v1.0.0-1m,mattersim-v1.0.0-5m"
)

export PATH=/root/.local/bin:/root/.juliaup/bin:$PATH
export UV_PYTHON_PREFERENCE=only-managed   # never build an env on the image's conda python
export UV_HTTP_TIMEOUT=600                 # the cu124/cu130 wheel stacks are ~3 GB per env
# mace-torch 0.3.16 calls torch.load without weights_only; torch >= 2.6 then refuses the pickled
# model. The August box had this set, which is how the pinned envs loaded MACE.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PROV"
exec > >(tee -a "$LOG") 2>&1

say()  { echo "[$(date -u +%FT%TZ)] $*"; }
warn() { say "[warn] $*"; }
die()  { say "[FATAL] $*"; rm -f /root/setup.ready; exit 1; }
fact() { printf '%s\t%s\n' "$1" "$2" >> "$PROV/facts.tsv"; }

cpu_quota() {
  # vast containers expose every host core to nproc; the cgroup quota is what we may use.
  local q p
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    [[ $q != max ]] && { echo $(( q / p )); return; }
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    (( q > 0 )) && { echo $(( q / p )); return; }
  fi
  env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc   # GNU nproc obeys both variables
}

# ---------------------------------------------------------------------------------------------
preflight() {
  say "== preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this container sees no GPU"
  local q
  q=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>&1 | head -n 1)
  say "GPU: $q"
  local name cap drv mem
  IFS=',' read -r name cap drv mem <<< "$q"
  cap=${cap// /}; drv=${drv// /}; name=${name## }
  fact gpu "$name"; fact gpu_compute_cap "$cap"; fact driver "$drv"; fact gpu_mem "${mem## }"
  if [[ ${ALLOW_GPU:-0} != 1 ]]; then
    [[ $cap =~ ^[0-9]+[.][0-9]+$ && $drv =~ ^[0-9]+ ]] || die "could not read compute capability/driver from nvidia-smi ('$q'); ALLOW_GPU=1 skips this gate"
    local maj=${cap%%.*} min=${cap##*.} dmaj=${drv%%.*}
    (( maj >= 10 )) && die "STOP: $name is compute capability $cap (Blackwell). The mace/chgnet/sevennet locks pin torch 2.6.0+cu124, which has no kernels for it, and this script will not swap in another torch. Destroy the box and rent an Ampere/Ada/Hopper GPU (runbook: box choice)."
    (( maj * 10 + min < 75 )) && die "STOP: $name is compute capability $cap; the orb/mattersim locks pin torch 2.13.0 (CUDA 13.0), which needs >= 7.5. Rent a newer GPU."
    (( dmaj < 580 )) && die "STOP: driver $drv; torch 2.13.0 (CUDA 13.0, orb/mattersim locks) needs >= 580. Rent a host whose max CUDA is >= 13.0."
  fi
  local quota; quota=$(cpu_quota)
  say "CPU: cgroup quota $quota cores (nproc says $(nproc)); RAM $(awk '/MemTotal/{printf "%d GB", $2/1048576}' /proc/meminfo)"
  fact cpu_quota "$quota"; fact nproc "$(nproc)"
  (( quota >= 30 )) || warn "cpu quota $quota < 30: QE will be slow; the runbook says destroy and re-rent"
  local ram; ram=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
  fact ram_gb "$ram"
  (( ${ram:-0} >= 60 )) || warn "RAM ${ram} GB: (quota - 8) QE ranks of a 40-atom PAW cell want ~1-2 GB each"
  local free; free=$(df -BG --output=avail /root | tail -n 1 | tr -dc 0-9)
  free=${free:-0}
  say "disk: ${free} GB free under /root"; fact disk_free_gb "$free"
  (( free >= 100 )) || warn "only ${free} GB free: five envs + uv cache + QE need ~40 GB, QE scratch more"
}

net_check() {
  # After apt: a fresh image may not have curl, and a missing tool must not read as 0 Mbit/s.
  local bps mbit
  # curl still prints the average speed when --max-time cuts the 100 MB transfer short
  bps=$(curl -s -o /dev/null -w '%{speed_download}' --max-time 15 https://cachefly.cachefly.net/100mb.test)
  [[ $bps =~ ^[0-9.]+$ ]] || bps=0
  mbit=$(awk -v b="$bps" 'BEGIN{printf "%d", b*8/1e6}')
  say "download: ~${mbit} Mbit/s"; fact inet_down_mbit "$mbit"
  (( mbit >= 200 )) || warn "download ${mbit} Mbit/s < 200: env builds pull ~10 GB of wheels"
}

apt_setup() {
  say "== apt"
  # pkg-config lets meson find BLAS/LAPACK the ordinary way; the rest is the plan's list.
  # libopenblas0-pthread is the runtime library only (no .pc file, so meson still builds SSCHA
  # against the reference BLAS); blas_setup keeps it off the system default and the orchestrator
  # hands it to the C1b include_v4 child alone.
  local pkgs="gfortran libblas-dev liblapack-dev libopenblas0-pthread build-essential pkg-config tmux rsync git curl bzip2 ca-certificates openssh-client"
  if dpkg -s $pkgs >/dev/null 2>&1; then say "apt packages present"; return 0; fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y && apt-get install -y --no-install-recommends $pkgs || die "apt install failed"
}

blas_setup() {
  # Installing OpenBLAS makes Debian's alternatives point libblas.so.3/liblapack.so.3 at it (it
  # outranks the reference build). The SSCHA seeds (C1, C5) must run on the reference BLAS the
  # June/August SSCHA ran on, so the system default is pinned back; only the include_v4 child gets
  # OpenBLAS, through LD_LIBRARY_PATH (run_rsc_revision.sh v4_attempt). Re-pinned on every run.
  local ma lib
  ma=$(gcc -print-multiarch 2>/dev/null); ma=${ma:-x86_64-linux-gnu}
  for lib in blas lapack; do
    if [[ -f /usr/lib/$ma/$lib/lib$lib.so.3 ]]; then
      update-alternatives --set "lib$lib.so.3-$ma" "/usr/lib/$ma/$lib/lib$lib.so.3" >/dev/null \
        || warn "could not pin lib$lib.so.3 to the reference build"
    fi
    fact "system_lib${lib}" "$(readlink -f "/usr/lib/$ma/lib$lib.so.3" 2>/dev/null)"
  done
  OPENBLAS_V4_DIR=/usr/lib/$ma/openblas-pthread
  if [[ -f $OPENBLAS_V4_DIR/libblas.so.3 && -f $OPENBLAS_V4_DIR/liblapack.so.3 ]]; then
    fact openblas_v4_dir "$OPENBLAS_V4_DIR"
  else
    warn "no OpenBLAS at $OPENBLAS_V4_DIR: C1b would run on the reference BLAS and likely hit its cap"
    fact openblas_v4_dir "missing"
  fi
  say "system BLAS: $(readlink -f "/usr/lib/$ma/libblas.so.3"); C1b OpenBLAS: $OPENBLAS_V4_DIR"
}

uv_setup() {
  say "== uv + python"
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed"; }
  uv --version || die "uv not on PATH"
  uv python install 3.11 3.12 || die "uv python install failed"
  HELPER_PY=$(uv python find 3.12) || die "no managed python 3.12"
  fact uv "$(uv --version)"
}

repo_setup() {
  say "== repo"
  export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"
  git config --global --add safe.directory "$REPO"
  if [[ -d $REPO/.git ]]; then
    # Never reset: once the orchestrator has run, results and the ledger live in this checkout.
    # But never go on from a stale checkout either: the envs would take an hour to build and then
    # run old scripts (check only catches missing files, not old ones).
    { git -C "$REPO" fetch origin "$BRANCH" && git -C "$REPO" checkout "$BRANCH" \
        && git -C "$REPO" merge --ff-only "origin/$BRANCH"; } \
      || warn "could not fast-forward $REPO"
    if [[ $(git -C "$REPO" rev-parse HEAD) != "$(git -C "$REPO" rev-parse "origin/$BRANCH" 2>/dev/null)" ]]; then
      git -C "$REPO" status --short | head -n 20
      [[ ${ALLOW_STALE_REPO:-0} == 1 ]] \
        || die "$REPO is not at origin/$BRANCH (local changes above block the fast-forward). Move them aside by hand, never reset; ALLOW_STALE_REPO=1 goes on regardless"
      warn "ALLOW_STALE_REPO=1: going on at $(git -C "$REPO" rev-parse --short HEAD)"
    fi
  elif [[ -f /root/.ssh/id_ed25519 ]]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO" \
      || { warn "ssh clone failed; trying https"; git clone --branch "$BRANCH" "$REPO_HTTPS" "$REPO"; } \
      || die "clone failed"
  else
    # The repo is public (checked 2026-09-26) and the box never pushes, so no key is needed.
    git clone --branch "$BRANCH" "$REPO_HTTPS" "$REPO" || die "clone failed"
  fi
  REPO_HEAD=$(git -C "$REPO" rev-parse --short HEAD)
  say "repo $REPO at $REPO_HEAD ($BRANCH)"; fact repo_head "$REPO_HEAD"
  [[ -f $REPO/envs/lock-mace-$LOCK_DATE.txt ]] || die "envs/lock-*-$LOCK_DATE.txt not in the checkout"
}

# ---------------------------------------------------------------------------------------------
# A pip-freeze lock cannot be fed to an installer verbatim:
#  - "-e git+..." is this repo; it is installed editable from the checkout instead.
#  - the cu124 envs were venvs over a conda base, so ~70 lines are "name @ file:///home/conda/..."
#    with no version. A wheel path still names its version (numpy 2.2.2): that becomes a pin. The
#    rest are conda's own tooling (conda, libmambapy, ...) or plain deps the resolver re-supplies.
#  - CellConstructor/python-sscha are built afterwards, against the env's own numpy.
# The pins are then passed as both requirements and overrides: every listed version is exact, and
# declared-dependency clashes frozen into the lock (mattersim's torchaudio 2.11 beside torch 2.13)
# do not stop the resolve. All five resolve for linux x86_64 (uv pip compile, 2026-09-26).
lock_pins() {
  awk '
    /^[[:space:]]*(#|$)/ { next }
    /^-e / { next }
    / @ file:/ {
      if ($0 ~ /[.]whl/) { n = split($3, p, "/"); split(p[n], w, "-"); print $1 "==" w[2] }
      next
    }
    tolower($0) ~ /^(cellconstructor|python-sscha)==/ { next }
    { print }
  ' "$1"
}

write_helpers() {
  # Small Python helpers, written once per run; kept free of backslashes on purpose.
  cat > "$PROV/compare_lock.py" <<'PY'
"""Compare an env's freeze against its lock pins; exit 2 if a science-relevant package differs."""
import sys


def norm(name):
    return name.strip().lower().replace("_", "-").replace(".", "-")


def pins(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith(("#", "-e")) or " @ " in line or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        out[norm(name)] = ver.strip()
    return out


want, have = pins(sys.argv[1]), pins(sys.argv[2])
critical = {norm(x) for x in sys.argv[3].split(",") if x}
diff = [(n, v, have.get(n)) for n, v in sorted(want.items()) if have.get(n) != v]
for n, v, h in diff:
    flag = "CRITICAL" if n in critical else "minor"
    print(f"  [{flag}] lock {n}=={v}  env {h}")
print(f"  {len(want) - len(diff)}/{len(want)} lock pins reproduced exactly")
sys.exit(2 if any(n in critical for n, _, _ in diff) else 0)
PY
  cat > "$PROV/verify_env.py" <<'PY'
"""CUDA + model smoke test for one env; writes a JSON record and fetches every checkpoint listed.

Exit 3 means torch cannot run on this GPU at all (the case where setup must stop rather than
swap torch); exit 1 means the model or SSCHA check failed."""
import datetime
import json
import os
import platform
import socket
import sys
import time
from importlib.metadata import PackageNotFoundError, version

env, model, ckpt_arg, want_sscha, out = sys.argv[1:6]
ckpts = [c for c in ckpt_arg.split(",") if c and c != "-"] or [None]
device = os.environ.get("VERIFY_DEVICE", "cuda")


def ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


rec = {
    "script": "setup_rsc_box.sh/verify_env.py", "env": env, "model": model,
    "python": platform.python_version(), "executable": sys.executable,
    "host": socket.gethostname(),
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "env_vars": {k: os.environ.get(k) for k in ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "PYTORCH_CUDA_ALLOC_CONF")},
    "versions": {p: ver(p) for p in (
        "numpy", "scipy", "ase", "phonopy", "spglib", "torch", "e3nn", "mace-torch", "chgnet",
        "orb-models", "sevenn", "mattersim", "CellConstructor", "python-sscha", "julia",
        "mlip-dynamic-stability")},
}
import torch  # noqa: E402

rec["torch_cuda"] = torch.version.cuda
rec["cuda_available"] = bool(torch.cuda.is_available())
cuda_ok = False
if device == "cuda":
    try:
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["capability"] = list(torch.cuda.get_device_capability(0))
        rec["arch_list"] = torch.cuda.get_arch_list()
        x = torch.ones(64, 64, device="cuda")
        rec["cuda_matmul"] = float((x @ x).sum().item())   # "no kernel image" surfaces here
        cuda_ok = True
    except Exception as exc:                                # noqa: BLE001
        rec["cuda_error"] = f"{type(exc).__name__}: {exc}"
else:
    cuda_ok = True

rec["smoke"] = []
if cuda_ok:
    import numpy as np
    from mlip_dynstab.calculators import get_calculator
    from mlip_dynstab.systems import build_atoms, get_spec
    for ck in ckpts:
        row, t0 = {"ckpt": ck}, time.time()
        try:
            handle = get_calculator(model, device=device, **({} if ck is None else {"model": ck}))
            atoms = build_atoms(get_spec("si_diamond"))
            atoms.rattle(stdev=0.01, seed=0)
            atoms.calc = handle.calc
            forces = atoms.get_forces()
            row.update(version=handle.version, max_abs_force_eV_A=float(np.abs(forces).max()),
                       ok=bool(np.isfinite(forces).all()))
        except Exception as exc:                            # noqa: BLE001
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.time() - t0, 1)
        rec["smoke"].append(row)

rec["sscha_ok"] = None
if want_sscha == "1":
    try:
        import cellconstructor  # noqa: F401
        import sscha.Ensemble
        import SCHAModules
        rec["sscha_ok"] = True
        rec["sscha_julia_ext"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
    except Exception as exc:                                # noqa: BLE001
        rec["sscha_ok"] = False
        rec["sscha_error"] = f"{type(exc).__name__}: {exc}"
    try:
        # Which BLAS/LAPACK/OpenMP/MPI the Fortran extension resolves to on this box (the C1
        # seeds must run on the reference BLAS; include_v4 gets OpenBLAS by LD_LIBRARY_PATH).
        import subprocess
        so = getattr(SCHAModules, "__file__", "")
        ldd = subprocess.run(["ldd", so], capture_output=True, text=True).stdout.splitlines()
        rec["schamodules_linkage"] = [ln.strip() for ln in ldd
                                      if any(k in ln for k in ("blas", "lapack", "gomp", "mkl", "mpi"))]
    except Exception as exc:                                # noqa: BLE001
        rec["schamodules_linkage"] = f"not recorded: {type(exc).__name__}: {exc}"

rec["ok"] = cuda_ok and all(r["ok"] for r in rec["smoke"]) and rec["sscha_ok"] is not False
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1)
for r in rec["smoke"]:
    print(f"  smoke {env}: {r}")
if not cuda_ok:
    print(f"  CUDA FAILED: {rec.get('cuda_error')}")
    sys.exit(3)
sys.exit(0 if rec["ok"] else 1)
PY
  cat > "$PROV/sssp_check.py" <<'PY'
"""Verify that every element's UPF named by the SSSP json is present with the json's md5."""
import hashlib
import json
import os
import sys

d, jname, elements = sys.argv[1], sys.argv[2], sys.argv[3].split()
table = json.load(open(os.path.join(d, jname), encoding="utf-8"))
bad = 0
for el in elements:
    ent = table[el]
    path = os.path.join(d, ent["filename"])
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
    ok = md5 == ent["md5"]
    bad += not ok
    print(f"  {el:2s} {ent['filename']:34s} ecutwfc {ent['cutoff_wfc']:5.1f} ecutrho "
          f"{ent['cutoff_rho']:6.1f}  {'ok' if ok else 'MISSING/BAD md5'}")
sys.exit(1 if bad else 0)
PY
  cat > "$PROV/readiness.py" <<'PY'
"""Assemble /root/logs/setup/readiness.json and print the readiness table."""
import glob
import json
import os
import sys

prov = sys.argv[1]
TAB, NL = chr(9), chr(10)
facts = {}
for line in open(os.path.join(prov, "facts.tsv"), encoding="utf-8"):
    if TAB in line:
        k, v = line.rstrip(NL).split(TAB, 1)
        facts[k] = v
envs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(prov, "env-*.json")))]
# env-mace-jl (WITH_JULIA=1) is optional; nothing in the orchestrator uses it unless told to.
required = [e for e in envs if not e["env"].endswith("-jl")]
print()
print(f"{'env':10s} {'python':8s} {'torch':16s} {'cuda':5s} {'model pkg':26s} {'sscha':7s} {'ckpts':7s} ok")
for e in envs:
    v = e["versions"]
    name = {"mace_mp0": "mace-torch", "chgnet": "chgnet", "orb_v2": "orb-models",
            "sevennet0": "sevenn", "mattersim": "mattersim"}[e["model"]]
    pkg = f"{name}={v.get(name)}"
    sscha = "-" if e.get("sscha_ok") is None else ("yes" if e["sscha_ok"] else "FAIL")
    good = sum(1 for r in e["smoke"] if r.get("ok"))
    print(f"{e['env']:10s} {e['python']:8s} {str(v.get('torch')):16s} {str(e.get('torch_cuda')):5s} "
          f"{pkg:26s} {sscha:7s} {good}/{len(e['smoke']):<5d} {'OK' if e['ok'] else 'FAIL'}")
print()
for k in ("gpu", "gpu_compute_cap", "driver", "cpu_quota", "disk_free_gb", "inet_down_mbit",
          "repo_head", "qe_package", "qe_version", "mpi", "qe_selftest", "sssp_source", "sssp_dir",
          "sssp_check", "system_libblas", "system_liblapack", "openblas_v4_dir", "julia_env"):
    print(f"  {k:16s} {facts.get(k, '-')}")
ready = (len(required) == 5 and all(e["ok"] for e in required)
         and facts.get("qe_selftest") == "ok" and facts.get("sssp_check") == "ok"
         and facts.get("failed_steps", "") == "")
json.dump({"ready": ready, "facts": facts, "envs": envs},
          open(os.path.join(prov, "readiness.json"), "w", encoding="utf-8"), indent=1)
print()
print("READY" if ready else "NOT READY: " + (facts.get("failed_steps") or "see the table above"))
sys.exit(0 if ready else 1)
PY
}

build_env() {
  local name=$1 lockname=$2 pyver=$3 source=$4 model=$5 ckpts=$6
  local env=/root/env-$name py=/root/env-$name/bin/python
  local lock=$REPO/envs/lock-$lockname-$LOCK_DATE.txt pins=$PROV/pins-$name.txt
  local want_sscha=0
  [[ " $SSCHA_ENVS " == *" $lockname "* ]] && want_sscha=1
  say "== env-$name (lock $lockname, python $pyver, torch from $source, sscha $want_sscha)"
  lock_pins "$lock" > "$pins"
  local idx=()
  [[ $source == cu124 ]] && idx=(--extra-index-url "$TORCH_CU124_INDEX" --index-strategy unsafe-best-match)
  local sig; sig="$(md5sum < "$lock" | cut -c1-32) sscha=$want_sscha"
  if [[ -f $env/.rsc-ready && $(cat "$env/.rsc-ready") == "$sig" ]]; then
    say "env-$name already built from this lock"
  else
    rm -rf "$env"    # a partial env from an interrupted build; only this script creates these
    uv venv --python "$pyver" "$env" || return 1
    uv pip install --python "$py" "${idx[@]}" -r "$pins" --overrides "$pins" || return 1
    if (( want_sscha )); then
      # --no-build-isolation so the Fortran extensions compile against this env's numpy, not
      # whatever numpy an isolated build would fetch. python-sscha's build needs cellconstructor.
      # The env's bin goes first on PATH so meson-python finds this env's meson and ninja, and the
      # rest of PATH is the system's alone: the pytorch image puts /opt/conda/bin first, and a
      # conda pkg-config or mpicc there would link SSCHA to a conda BLAS or to MPI instead of the
      # apt reference BLAS (verify_env.py records the linkage either way).
      local uvb bpath="$env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
      uvb=$(command -v uv)
      uv pip install --python "$py" "${idx[@]}" -c "$pins" meson meson-python ninja cython || return 1
      env -u PKG_CONFIG_PATH PATH="$bpath" "$uvb" pip install --python "$py" --no-build-isolation --overrides "$pins" CellConstructor==1.6.2 || return 1
      env -u PKG_CONFIG_PATH PATH="$bpath" "$uvb" pip install --python "$py" --no-build-isolation --overrides "$pins" python-sscha==1.6.1 || return 1
    fi
    uv pip install --python "$py" "${idx[@]}" --overrides "$pins" -e "$REPO" || return 1
    echo "$sig" > "$env/.rsc-ready"
  fi
  uv pip freeze --python "$py" > "$PROV/freeze-$name.txt"
  local want=$PROV/want-$name.txt
  cp "$pins" "$want"
  (( want_sscha )) && printf '%s\n' "CellConstructor==1.6.2" "python-sscha==1.6.1" >> "$want"
  local critical="torch,numpy,scipy,ase,phonopy,spglib,e3nn,mace-torch,chgnet,orb-models,sevenn,mattersim,CellConstructor,python-sscha"
  "$py" "$PROV/compare_lock.py" "$want" "$PROV/freeze-$name.txt" "$critical" \
    || { say "[error] env-$name does not reproduce its lock on a science-relevant package"; return 1; }
  "$py" "$PROV/verify_env.py" "$name" "$model" "$ckpts" "$want_sscha" "$PROV/env-$name.json"
  local rc=$?
  if (( rc == 3 )); then
    die "STOP: torch in env-$name cannot run on this GPU (see $PROV/env-$name.json, cuda_error). The lock pins this torch; not substituting one. Destroy the box and rent a GPU the runbook allows."
  fi
  return $rc
}

julia_env() {
  # python-sscha 1.6.1 uses julia only for the Fourier-space gradient in Ensemble
  # (fourier_gradient.jl). The include_v4 Hessian is SCHAModules.get_v4, Fortran + OpenMP over
  # n_mode^4, which julia does not touch, so this cannot speed up C1b; threads can. And importing
  # julia flips every Ensemble in that env onto the Fourier path, so it lives in its own env and
  # never in env-mace, where the C1 seeds must follow the June/August code path.
  (( WITH_JULIA == 1 )) || return 0
  say "== julia (WITH_JULIA=1) -> /root/env-mace-jl"
  command -v julia >/dev/null || { curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel lts || return 1; }
  export PATH=/root/.juliaup/bin:$PATH
  fact julia "$(julia --version 2>&1)"
  build_env mace-jl mace 3.11 cu124 mace_mp0 - || return 1
  local py=/root/env-mace-jl/bin/python
  uv pip install --python "$py" julia==0.6.2 || return 1
  "$py" -c "import julia; julia.install()" || return 1
  "$py" -c "import sys, sscha.Ensemble as E; print('julia ext', E.__JULIA_EXT__); sys.exit(0 if E.__JULIA_EXT__ else 1)"
}

# ---------------------------------------------------------------------------------------------
qe_setup() {
  say "== Quantum ESPRESSO (conda-forge via micromamba; the Ubuntu apt build aborts on input read)"
  export MAMBA_ROOT_PREFIX=/root/micromamba
  local mm=/root/micromamba/bin/micromamba
  if [[ ! -x $mm ]]; then
    mkdir -p /root/micromamba
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /root/micromamba bin/micromamba || return 1
  fi
  if [[ ! -x $QE_PREFIX/bin/pw.x ]]; then
    # openmpi named explicitly: conda-forge can otherwise solve qe against MPICH, and the
    # OpenMPI 5 build is the one that has run cleanly on vast.
    "$mm" create -y -p "$QE_PREFIX" -c conda-forge "$QE_SPEC" openmpi \
      || { warn "$QE_SPEC did not solve; taking the current conda-forge qe (version recorded)"; rm -rf "$QE_PREFIX"; "$mm" create -y -p "$QE_PREFIX" -c conda-forge qe openmpi; } \
      || return 1
  fi
  "$mm" list -p "$QE_PREFIX" > "$PROV/qe-packages.txt" 2>&1
  local v; v=$(awk '$1=="qe"{print $2" ("$3")"}' "$PROV/qe-packages.txt")
  say "QE package: ${v:-unknown}"; fact qe_package "${v:-unknown}"
  # conda-forge solves qe against OpenMPI or MPICH; only OpenMPI takes these flags.
  if [[ -x $QE_PREFIX/bin/ompi_info ]]; then MPI_FLAGS="--allow-run-as-root --bind-to none"; fact mpi openmpi
  else MPI_FLAGS=""; fact mpi "not openmpi (mpich?)"; fi
  [[ -x $QE_PREFIX/bin/pw.x ]]
}

sssp_setup() {
  say "== SSSP 1.3.0 PBE efficiency -> $SSSP_DIR"
  mkdir -p "$SSSP_DIR"
  if [[ -f $SSSP_DIR/$SSSP_JSON ]] && "$HELPER_PY" "$PROV/sssp_check.py" "$SSSP_DIR" "$SSSP_JSON" "$SSSP_ELEMENTS"; then
    SSSP_SOURCE=$(cat "$SSSP_DIR/.source" 2>/dev/null || echo "materialscloud SSSP 1.3.0 PBE efficiency")
  else
    local tmp=/root/sssp_download; mkdir -p "$tmp"
    curl -fL --retry 5 -o "$tmp/$SSSP_JSON" "$SSSP_API/$SSSP_JSON/content"
    [[ $(md5sum < "$tmp/$SSSP_JSON" | cut -c1-32) == "$SSSP_JSON_MD5" ]] \
      || { fact sssp_check "could not fetch the 1.3.0 json"; return 1; }
    cp "$tmp/$SSSP_JSON" "$SSSP_DIR/"   # dft_reference.py reads cutoffs and filenames from it
    if curl -fL --retry 5 -o "$tmp/$SSSP_TAR" "$SSSP_API/$SSSP_TAR/content" \
       && [[ $(md5sum < "$tmp/$SSSP_TAR" | cut -c1-32) == "$SSSP_TAR_MD5" ]]; then
      tar -xzf "$tmp/$SSSP_TAR" -C "$SSSP_DIR"
      SSSP_SOURCE="Materials Cloud SSSP 1.3.0 PBE efficiency, doi 10.24435/materialscloud:f3-ym, tar md5 $SSSP_TAR_MD5"
    else
      # The apt package ships some SSSP release under its own layout. Its files are used only
      # where they are byte-identical to 1.3.0's (the md5 check below), so the PBE numbers are
      # the same either way; anything else leaves the box NOT READY.
      warn "Materials Cloud tarball failed; trying the apt SSSP files against the 1.3.0 md5s"
      DEBIAN_FRONTEND=noninteractive apt-get install -y quantum-espresso-data-sssp || return 1
      local f src
      for f in $("$HELPER_PY" -c "import json, sys; t = json.load(open(sys.argv[1])); print(' '.join(t[e]['filename'] for e in sys.argv[2].split()))" "$SSSP_DIR/$SSSP_JSON" "$SSSP_ELEMENTS"); do
        src=$(find /usr/share/espresso/pseudo -maxdepth 1 -iname "$f" | head -n 1)
        [[ -n $src ]] && cp "$src" "$SSSP_DIR/$f"
      done
      SSSP_SOURCE="apt quantum-espresso-data-sssp $(dpkg-query -W -f='${Version}' quantum-espresso-data-sssp), files md5-identical to SSSP 1.3.0 PBE efficiency"
    fi
    echo "$SSSP_SOURCE" > "$SSSP_DIR/.source"
  fi
  "$HELPER_PY" "$PROV/sssp_check.py" "$SSSP_DIR" "$SSSP_JSON" "$SSSP_ELEMENTS" || { fact sssp_check "md5 mismatch"; return 1; }
  fact sssp_source "$SSSP_SOURCE"; fact sssp_dir "$SSSP_DIR"; fact sssp_check ok
}

qe_selftest() {
  # One bcc-Zr SCF on two ranks with the SSSP Zr pseudopotential: proves pw.x, MPI and the
  # pseudo dir end to end before the orchestrator commits a day of CPU to them.
  say "== QE self-test"
  local d=/root/qe_selftest upf="" f
  for f in "$SSSP_DIR"/*; do [[ ${f##*/} =~ ^[Zz][Rr][._-] ]] && { upf=${f##*/}; break; }; done
  [[ -n $upf ]] || { fact qe_selftest "no Zr UPF"; return 1; }
  rm -rf "$d"; mkdir -p "$d"
  cat > "$d/pw.in" <<EOF
&control
  calculation = 'scf', prefix = 'zr', outdir = './tmp', pseudo_dir = '$SSSP_DIR', tprnfor = .true.
/
&system
  ibrav = 3, celldm(1) = 6.746, nat = 1, ntyp = 1, ecutwfc = 30, ecutrho = 240,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02
/
&electrons
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
Zr 91.224 $upf
ATOMIC_POSITIONS crystal
Zr 0.0 0.0 0.0
K_POINTS automatic
8 8 8 0 0 0
EOF
  # vader single-copy (CMA) needs ptrace, which docker usually denies; qe_queue.sh sets the same.
  ( cd "$d" && export PATH="$QE_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$QE_PREFIX/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 \
      OMPI_MCA_btl_vader_single_copy_mechanism=none \
      && timeout 600 mpirun ${MPI_FLAGS:-} -np 2 pw.x -in pw.in > pw.out 2>&1 < /dev/null )
  local header; header=$(grep -m1 'Program PWSCF' "$d/pw.out" | sed 's/^ *//')
  fact qe_version "${header:-unknown}"
  if grep -q 'JOB DONE' "$d/pw.out" && grep -q 'convergence has been achieved' "$d/pw.out"; then
    say "QE self-test ok: $header; $(grep -m1 '^!' "$d/pw.out" | sed 's/^ *//')"
    fact qe_selftest ok
  else
    say "[error] QE self-test failed; tail of $d/pw.out:"; tail -n 20 "$d/pw.out"
    fact qe_selftest failed; return 1
  fi
}

write_env_file() {
  cat > /root/rsc_env.sh <<EOF
# Written by setup_rsc_box.sh $(date -u +%FT%TZ); sourced by run_rsc_revision.sh.
export PATH=/root/.local/bin:\$PATH
export QE_PREFIX=$QE_PREFIX
export SSSP_DIR=$SSSP_DIR
export SSSP_SOURCE="${SSSP_SOURCE:-unknown}"
export QE_MPI_FLAGS="${MPI_FLAGS:-}"
export OPENBLAS_V4_DIR=${OPENBLAS_V4_DIR:-/usr/lib/x86_64-linux-gnu/openblas-pthread}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
}

# ---------------------------------------------------------------------------------------------
main() {
  local t0=$SECONDS failed=() spec name lockname pyver source model ckpts
  rm -f /root/setup.ready "$PROV/facts.tsv" "$PROV"/env-*.json
  say "setup_rsc_box.sh start on $(hostname)"
  fact host "$(hostname)"; fact started_utc "$(date -u +%FT%TZ)"
  preflight
  apt_setup
  blas_setup
  net_check
  uv_setup
  repo_setup
  write_helpers
  for spec in "${ENV_SPECS[@]}"; do
    read -r name lockname pyver source model ckpts <<< "$spec"
    build_env "$name" "$lockname" "$pyver" "$source" "$model" "$ckpts" || { say "[error] env-$name failed"; failed+=("env-$name"); }
  done
  if (( WITH_JULIA == 1 )); then
    julia_env && fact julia_env ok || { warn "julia env failed; optional, C1b runs in env-mace without it"; fact julia_env failed; }
  fi
  qe_setup || { say "[error] QE install failed"; failed+=("qe"); }
  sssp_setup || { say "[error] SSSP setup failed"; failed+=("sssp"); }
  qe_selftest || failed+=("qe-selftest")
  write_env_file
  fact failed_steps "${failed[*]}"
  fact setup_seconds "$(( SECONDS - t0 ))"
  if "$HELPER_PY" "$PROV/readiness.py" "$PROV"; then
    date -u +%FT%TZ > /root/setup.ready
    say "setup finished in $(( (SECONDS - t0) / 60 )) min. Next: bash $REPO/scripts/box/run_rsc_revision.sh check"
  else
    say "setup finished NOT READY after $(( (SECONDS - t0) / 60 )) min; fix the failures above and rerun (idempotent)"
    exit 1
  fi
}

main "$@"
