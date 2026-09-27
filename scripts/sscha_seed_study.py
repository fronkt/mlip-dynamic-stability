"""SSCHA seed study with persisted convergence diagnostics (revision items C1, C1b, C5).

Referee 1 (R1.4) asked for what the paper's SSCHA numbers never carried: sample sizes, gradient
history, stopping criteria and an uncertainty on the free-energy Hessian, and for the four-seed
reproducibility test to reach beyond bcc-Zr (where MACE-MP-0 has no instability to lose) to the
displacive systems. The old per-seed values were printed to a log and never deposited. This
script re-runs the production recipe once per seed and writes everything to one JSON per unit,
seed by seed, so a dead box costs at most the seed in flight.

Fidelity to production. Each seed replays ``finite_t.compute_finite_t_sscha`` step for step: the
same relax, the same phonopy initialiser (``_cc_dyn_from_phonopy``, imported, not copied),
ForcePositiveDefinite + Symmetrize, ``np.random.seed(seed)`` at the same point, the same
minimiser settings (read from the production function's own signature defaults, so they cannot
drift from what the ledger was computed with), the same dedicated Hessian ensemble and the same
|omega|-based acoustic mask. Everything added is read-only and draws nothing from the global
RNG: a post-step hook that copies the minimiser's histories, diagonalisations of matrices that
already exist, and a bootstrap on a private ``np.random.Generator``. Seed 0 is therefore the
production computation, and the JSON records whether it reproduces the deposited ledger value.

What python-sscha 1.6.1 actually does, read from its sdist (the line numbers cited here and
below are that source's), and why the per-step and per-population records exist:
  * ``Relax.relax`` re-enters ``SSCHA_Minimizer.init(delete_previous_data=False)`` for every
    population (Relax.py:386), so the free-energy, gradient and Kong-Liu histories accumulate
    across populations; init() clears them only on request (SchaMinimizer.py:1127-1134).
  * ``max_ka`` is compared with that cumulative history, ``len(self.__fe__) > self.max_ka``
    (SchaMinimizer.py:1370). It caps the total number of steps, not the steps per population as
    the comment in finite_t.py assumes.
  * a population that stops unconverged restores the dyn from before its last step
    (SchaMinimizer.py:1389-1397). With the cumulative cap, a population that starts past the
    cap takes one step and then discards it.
  * convergence needs the dyn gradient AND the structure gradient below meaningful_factor times
    their stochastic errors (gradi_op "all", SchaMinimizer.py:1622-1651) or below abs_conv_thr
    = 1e-8. With meaningful_factor = 1e-4 the gradient has to sit four decades under its own
    error.
These are readings of the code, not measurements. The box run confirms or refutes each one:
every step is tagged with its population and with the stopping test that fired, and the dyn at
the end of every population is reloaded and compared with the one before it.

Outputs (under results/revision/sscha_seeds/ unless --out/--out-dir says otherwise):
  <system>_<model>_<T>K_sc<abc>.json     one per unit, rewritten after every seed
  configs/<unit>_seed<k>.extxyz           a seeded subsample of the final Hessian ensemble with
                                          MLIP energy (eV) and forces (eV/A), for the C3b PBE
                                          force benchmark; C1 units only (c3b_unit), everything
                                          else under configs_not_c3b/ so C3b does not queue it
  work/<unit>_seed<k>/                    dyn_pop<p>_* per population, dyn_harmonic_*,
                                          dyn_start_fpd_*, dyn_hessian_*, and the Hessian
                                          ensemble (save_bin) that the include_v4 child reloads

Usage (from the repo root, inside the model's env on the GPU box):
    python scripts/sscha_seed_study.py --system batio3_cubic --model mace_mp0 --T 100
    python scripts/sscha_seed_study.py --preset revision --list
    python scripts/sscha_seed_study.py --preset revision            # the units this env can run
    python scripts/sscha_seed_study.py --preset revision --include-v4
    python scripts/sscha_seed_study.py --summarize
    python scripts/sscha_seed_study.py --system batio3_cubic --model mace_mp0 --T 100 --dry-run --device cpu --out-dir <scratch>
Resumable: a seed whose status is "ok" is skipped, and so is a "failed" one unless
--retry-failed is given. A seed left "running" by a dead box is recomputed.

Converged mode (C1c; the seed-study paths above are untouched). The box runs showed that no
production relaxation converges, and that python-sscha 1.6.1's convergence test compares the
gradient with a placeholder error of ones (Ensemble.py:2657), not its stochastic error. This mode
relaxes to genuine convergence (real error, per-population step cap, Kong-Liu guard; the reasons
and source lines are in the comment block above CONV_PRESET) from two starts per unit, A = the
production ForcePositiveDefinite start and B = imaginary modes set to a small positive
frequency, and records per population and per step what the minimiser did. Outputs under
results/revision/sscha_converged/: <unit>_start<A|B>.json (flushed at every population end),
<unit>_AB.json (start comparison), work/<unit>_start<S>/ (dyn_pop*, dyn files, Hessian ensemble).
    python scripts/sscha_seed_study.py --preset converged --list
    python scripts/sscha_seed_study.py --preset converged                # both starts, this env
    python scripts/sscha_seed_study.py --converge --system batio3_cubic --model mace_mp0 --T 100 --start B
    python scripts/sscha_seed_study.py --converge --summarize
    python scripts/sscha_seed_study.py --preset converged --dry-run --device cpu --out-dir <scratch>
Resumable per (unit, start) with the seed study's rules; --unit-timeout caps each relaxation
(between minimiser steps only: wrap every box process in `timeout` for true hangs). A flock per
(unit, start) under <out>/locks/ keeps two processes off the same start; a killed run's partial
history moves to previous_runs when it is recomputed.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

# The file actually run ("scripts/sscha_seed_study.py" under its own name), so a copy run beside
# it (e.g. while the committed one is in use on the box) records its own name with its sha256.
SCRIPT = Path(__file__).resolve().relative_to(REPO).as_posix()
SCHEMA = "sscha_seed_study/v1"
STUDY_DIR = REPO / "results" / "revision" / "sscha_seeds"
LEDGER_TOL_THZ = 0.02
# Rebuilding the same ensemble (bootstrap identity replicate, v4 save/load round trip) may only
# differ by floating-point noise; 1e-3 THz is far above that and far below any seed spread.
REBUILD_TOL_THZ = 1e-3
C_THZ_PER_CM = 2.99792458e-2            # finite_t.py:1029 applies this to CC.Units.RY_TO_CM
RY_TO_CM_CC162 = 109737.36034769034314   # cellconstructor 1.6.2 Units.py:18; --dry-run only

# Salts for the private generators. They keep the bootstrap draw and the saved-config draw
# independent of each other and of the SSCHA's own global-RNG stream.
BOOT_SALT, CFG_SALT, MOCK_SALT = 20260926, 20260927, 20260928

MODEL_DIST = {"mace_mp0": "mace-torch", "chgnet": "chgnet", "orb_v2": "orb-models",
              "orb_v3": "orb-models", "sevennet0": "sevenn", "mattersim": "mattersim"}
MODEL_IMPORT = {"mace_mp0": "mace", "chgnet": "chgnet", "orb_v2": "orb_models",
                "orb_v3": "orb_models", "sevennet0": "sevenn", "mattersim": "mattersim"}
MODEL_ENV = {"mace_mp0": "mace", "chgnet": "chgnet", "orb_v2": "orb", "orb_v3": "orb",
             "sevennet0": "sevennet", "mattersim": "mattersim"}

RECIPE_KEYS = ("n_configs", "max_pop", "n_hessian", "disp", "relax", "fmax",
               "root_representation", "min_step_dyn", "meaningful_factor", "max_ka",
               "imag_tol_thz")

# --preset revision. One process can only run the models its env provides, so the preset runs
# whichever of these units the current env can import and says which it skipped.
# c3b=True marks the units whose saved configurations are C3b input (see c3b_unit()).
PRESETS = {
    "revision": [
        # C1 (R1.4d/e): four seeds on units whose SSCHA answer is itself in question.
        dict(system="batio3_cubic", model="mace_mp0", T=100.0, supercell=(2, 2, 2),
             seeds=(0, 1, 2, 3), v4=True, c3b=True, item="C1 + C1b",
             why="Table S2 false-stable: the harmonic soft mode is erased by "
                 "ForcePositiveDefinite; also the include_v4 target"),
        dict(system="zro2_cubic", model="mace_mp0", T=100.0, supercell=(2, 2, 2),
             seeds=(0, 1, 2, 3), c3b=True, item="C1", why="fluorite false-stable"),
        dict(system="zr_bcc", model="mattersim", T=50.0, supercell=(2, 2, 2),
             seeds=(0, 1, 2, 3), c3b=True, item="C1",
             why="a bcc unit that HAS a harmonic instability (the old seed test used "
                 "MACE-MP-0 Zr, which has none)"),
        dict(system="srtio3_cubic", model="mace_mp0", T=600.0, supercell=(2, 2, 2),
             seeds=(0, 1, 2, 3), c3b=True, item="C1", why="high-T false-unstable runaway"),
        # C5: bcc finite size. 3x3x3 in the current envs, seed 0, against the canonical 2x2x2
        # ledger rows (recorded in each JSON under ledger.other_supercells).
        dict(system="zr_bcc", model="mattersim", T=100.0, supercell=(3, 3, 3), seeds=(0,),
             item="C5", why="finite size, 3x3x3 vs ledger 2x2x2"),
        dict(system="zr_bcc", model="mattersim", T=300.0, supercell=(3, 3, 3), seeds=(0,),
             item="C5", why="finite size, 3x3x3 vs ledger 2x2x2"),
        dict(system="zr_bcc", model="mace_mp0", T=100.0, supercell=(3, 3, 3), seeds=(0,),
             item="C5", why="finite size, 3x3x3 vs ledger 2x2x2"),
        dict(system="zr_bcc", model="mace_mp0", T=300.0, supercell=(3, 3, 3), seeds=(0,),
             item="C5", why="finite size, 3x3x3 vs ledger 2x2x2"),
    ],
}


# ------------------------------------------------------------------ small utilities ----

def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p) -> str:
    """Repo-relative POSIX path when possible, so the JSON reads the same on every machine."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def from_rel(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else REPO / p


def t_tag(T: float) -> str:
    return f"{float(T):g}"


def unit_tag(system: str, model: str, T: float, supercell) -> str:
    return f"{system}_{model}_{t_tag(T)}K_sc{''.join(str(int(x)) for x in supercell)}"


def c3b_unit(system: str, model: str, T: float, supercell) -> bool:
    """Whether this unit's saved configurations are C3b input, i.e. go to configs/.

    ``dft_reference.py c3b-inputs`` reads every extxyz under configs/ and groups by (system,
    model, T) alone, while its job budget covers only the four 2x2x2 C1 units (C1_UNITS,
    SSCHA_SUPERCELL there). A C5 3x3x3 Zr unit (or any manual unit) written there would add
    unplanned metallic 27-atom PBE jobs, and qe_queue.sh runs smallest first, so they would
    run ahead of the 40-atom perovskite configurations that R1.2 turns on. Such units keep
    their configurations, under configs_not_c3b/."""
    key = (system, model, float(T), tuple(int(x) for x in supercell))
    return any(u.get("c3b") and key == (u["system"], u["model"], float(u["T"]),
                                        tuple(int(x) for x in u["supercell"]))
               for units in PRESETS.values() for u in units)


def _clean(o):
    """JSON-safe copy: numpy scalars/arrays to Python, NaN/inf to null, paths to strings."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.complexfloating):
        return _clean(float(np.real(o)))
    if isinstance(o, Path):
        return o.as_posix()
    return o


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(_clean(obj), indent=1), encoding="utf-8")
    for attempt in range(10):            # os.replace can race a reader on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def load_json(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar(x):
    """One float from a minimiser history entry. The julia branch stores the dyn-gradient
    error as an array (SchaMinimizer.py:597-600); its norm is recorded then."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 0:
        return float(np.real(a))
    return float(np.linalg.norm(np.real(a)))


@contextlib.contextmanager
def chdir(path: Path):
    """Relax writes dyn_pop<p>_* into the CWD (Relax.py:407-408), and the ensemble writes
    error_* files there on failure; keep them in the seed's work directory."""
    old = os.getcwd()
    Path(path).mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def sha256(path: Path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------- provenance ----

def git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def git_dirty(paths):
    """Uncommitted edits under the production code: a reproduction claim needs this empty."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return None


def _dist_version(name):
    import importlib.metadata as md
    for n in (name, name.lower()):
        try:
            return md.version(n)
        except Exception:
            continue
    return None


def package_versions(model: str) -> dict:
    names = ["numpy", "scipy", "ase", "phonopy", "spglib", "torch", "pandas", "pyarrow",
             "pymatgen", "python-sscha", "CellConstructor", "julia"]
    if model in MODEL_DIST:
        names.append(MODEL_DIST[model])
    return {n: _dist_version(n) for n in names}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_check(model: str, installed: dict) -> dict:
    """Compare the installed versions with envs/lock-<env>-<date>.txt, the pinned env the
    ledger rows were measured in. A reproduction that fails in a different env is not the
    same claim as one that fails in the pinned env."""
    env = MODEL_ENV.get(model)
    locks = sorted((REPO / "envs").glob(f"lock-{env}-*.txt")) if env else []
    if not locks:
        return {"lock_file": None}
    lock = locks[-1]
    pinned = {}
    for line in lock.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*(\S+)", line)
        if m:
            pinned[_norm(m.group(1))] = m.group(2)
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*@\s*(\S+)", line)
        if m:
            base = m.group(2).rsplit("/", 1)[-1]
            v = re.search(r"^[A-Za-z0-9_.]+?-([0-9][^-]*)-", base)
            pinned[_norm(m.group(1))] = v.group(1) if v else m.group(2)
    rows, ok = {}, True
    for name, have in installed.items():
        want = pinned.get(_norm(name))
        if want is None and have is None:
            continue
        match = (want == have) if (want is not None and have is not None) else None
        ok = ok and (match is not False)
        rows[name] = {"pinned": want, "installed": have, "match": match}
    return {"lock_file": rel(lock), "packages": rows, "all_pinned_match": ok}


def cpu_budget() -> dict:
    """CPUs this container may use. On vast.ai, nproc and os.cpu_count() report the host;
    the cgroup quota is the real budget (tasks/todo.md, Phase 1)."""
    info = {"os_cpu_count": os.cpu_count(), "affinity": None, "cgroup_quota": None}
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        f = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if f and f[0] != "max":
            info["cgroup_quota"] = float(f[0]) / float(f[1])
    except Exception:
        pass
    if info["cgroup_quota"] is None:
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                info["cgroup_quota"] = q / per
        except Exception:
            pass
    cands = [x for x in (info["cgroup_quota"], info["affinity"], info["os_cpu_count"]) if x]
    info["usable"] = max(1, int(min(cands))) if cands else 1
    return info


def schamodules_linkage() -> dict:
    """Which BLAS/LAPACK/OpenMP the python-sscha Fortran extension links. include_v4 ends in a
    dgetrf/dgetri inversion and three dgemm calls on (3N)^2 x (3N)^2 matrices
    (get_odd_straight_with_v4.f90:118-151): a single-threaded reference BLAS and a threaded
    OpenBLAS differ by more than the v4 time cap."""
    try:
        import SCHAModules
        so = getattr(SCHAModules, "__file__", None)
        out = {"file": so}
        if so and sys.platform.startswith("linux"):
            r = subprocess.run(["ldd", so], capture_output=True, text=True, timeout=30)
            out["ldd"] = [ln.strip() for ln in r.stdout.splitlines()
                          if re.search(r"blas|lapack|gomp|mkl|openblas|gfortran", ln)]
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def provenance(model: str, dry_run: bool, device: str) -> dict:
    pk = package_versions(model)
    out = {"script": SCRIPT, "script_sha256": sha256(Path(__file__)),
           "finite_t_sha256": sha256(REPO / "mlip_dynstab" / "finite_t.py"),
           "git_head": git_head(),
           "git_dirty_production_code": git_dirty(["mlip_dynstab"]),
           "host": socket.gethostname(), "utc": utc_now(),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": pk, "env_lock": lock_check(model, pk),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "cpu": cpu_budget(), "device": device, "dry_run": bool(dry_run)}
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                out["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return out


# -------------------------------------------------------------- production recipe ----

def production_recipe() -> dict:
    """The SSCHA settings every deposited sscha ledger row was computed with. cli.run_unit
    passes only the supercell, so the signature defaults ARE the recipe; reading them here
    means this study cannot silently run a different one."""
    import inspect
    from mlip_dynstab.finite_t import compute_finite_t_sscha
    params = inspect.signature(compute_finite_t_sscha).parameters
    return {k: params[k].default for k in RECIPE_KEYS}


def ledger_lookup(system: str, model: str, T: float, supercell) -> dict:
    """The canonical (current-generation) sscha ledger value for this unit. Read-only."""
    try:
        from mlip_dynstab.analysis import load_canonical
        df = load_canonical()
    except Exception as exc:
        return {"found": False, "error": f"{type(exc).__name__}: {exc}"}
    d = df[(df["method"] == "sscha") & (df["system"] == system) & (df["model"] == model)]
    d = d[np.isclose(d["temperature_K"].astype(float), float(T))]
    sc = d["supercell"].map(lambda v: tuple(int(x) for x in v) if v is not None else None)
    want = tuple(int(x) for x in supercell)
    same = d[sc.map(lambda s: s == want)]
    other = d[sc.map(lambda s: s != want)]
    out = {"source": "mlip_dynstab.analysis.load_canonical(), method == 'sscha'",
           "found": bool(len(same)), "n_rows": int(len(same))}
    if len(same):
        r = same.iloc[-1]
        out.update(min_eff_freq_thz=float(r["min_eff_freq_thz"]), uhash=str(r["uhash"]),
                   model_version=str(r.get("model_version")),
                   method_version=r.get("method_version"),
                   lowest6_thz=[float(x) for x in (r.get("ft_lowest6_thz") if
                                                   r.get("ft_lowest6_thz") is not None else [])],
                   wall_s=r.get("wall_s"))
        if len(same) > 1:
            out["all_values_thz"] = [float(x) for x in same["min_eff_freq_thz"]]
    out["other_supercells"] = [
        {"supercell": list(s), "min_eff_freq_thz": float(v), "uhash": str(h)}
        for s, v, h in zip(sc[other.index], other["min_eff_freq_thz"], other["uhash"])]
    return out


def ledger_comparison(ledger: dict, value: float, tol: float, model_version: str) -> dict:
    if not ledger.get("found"):
        return {"ledger_thz": None, "this_thz": value, "abs_diff_thz": None, "tol_thz": tol,
                "reproduces_ledger": None,
                "note": "no canonical sscha ledger row for this unit and supercell"}
    diff = abs(float(value) - ledger["min_eff_freq_thz"])
    return {"ledger_thz": ledger["min_eff_freq_thz"], "this_thz": value, "abs_diff_thz": diff,
            "tol_thz": tol, "reproduces_ledger": bool(diff <= tol),
            "ledger_uhash": ledger.get("uhash"),
            "model_version_ledger": ledger.get("model_version"),
            "model_version_now": model_version,
            "model_version_matches": ledger.get("model_version") == model_version}


# ----------------------------------------------------------- frequency diagnostics ----

def nonacoustic_idx(w) -> np.ndarray:
    """Indices of the non-acoustic modes, exactly as finite_t.py:1074-1075 selects them: the
    three acoustic modes are the three NEAREST ZERO in magnitude, never the three most
    negative (that mistake discarded real instabilities in the v1 grid)."""
    w = np.asarray(w, dtype=float)
    if w.size <= 3:
        return np.arange(w.size)
    return np.delete(np.arange(w.size), np.argsort(np.abs(w))[:3])


def freq_summary(w_ry, ry_to_thz: float, imag_tol: float) -> dict:
    w = np.real(np.asarray(w_ry)).astype(float) * ry_to_thz
    idx = nonacoustic_idx(w)
    ac = np.delete(w, idx)
    return {"min_nonac_thz": float(w[idx].min()),
            "lowest6_thz": [float(x) for x in np.sort(w)[:6]],
            "n_imag_nonac": int(np.sum(w[idx] < imag_tol)),
            "acoustic_max_abs_thz": float(np.max(np.abs(ac))) if ac.size else None}


def dyn_freqs(dyn):
    w, pols = dyn.DiagonalizeSupercell()
    return np.real(np.asarray(w)).astype(float), np.asarray(pols)


def dyn_rel_change(a, b):
    """||Phi_a - Phi_b||_F / ||Phi_b||_F summed over the commensurate q grid (q matched by
    coordinate). Zero means the matrix did not move."""
    qb = [np.asarray(q, dtype=float) for q in b.q_tot]
    num = den = 0.0
    for ia, q in enumerate(a.q_tot):
        j = next((j for j, q2 in enumerate(qb) if np.allclose(np.asarray(q, float), q2,
                                                              atol=1e-6)), None)
        if j is None:
            return None
        A, B = np.asarray(a.dynmats[ia]), np.asarray(b.dynmats[j])
        num += float(np.sum(np.abs(A - B) ** 2))
        den += float(np.sum(np.abs(B) ** 2))
    return math.sqrt(num / den) if den > 0 else None


def freq_along_softmode(w_h, pols_h, w_x, pols_x, ry_to_thz: float,
                        deg_tol_thz: float = 1e-3) -> dict:
    """The frequency matrix X assigns to the Hessian's softest non-acoustic mode.

    Rayleigh quotient of X's mass-scaled supercell matrix, averaged over the Hessian mode's
    degenerate subspace (trace/k does not depend on which eigenvectors the diagonaliser picked
    inside a degenerate star). With X = the ForcePositiveDefinite start this says whether the
    Hessian just echoes it; with X = the harmonic dyn it says what that mode was before its
    sign was flipped.
    """
    idx = nonacoustic_idx(w_h)
    wmin = w_h[idx].min()
    sub = [int(i) for i in idx if abs(w_h[i] - wmin) <= deg_tol_thz / ry_to_thz]
    E = pols_h[:, sub]
    D = (pols_x * (np.sign(w_x) * w_x ** 2)) @ np.conj(pols_x).T
    q = float(np.real(np.trace(np.conj(E).T @ D @ E))) / len(sub)
    ortho = float(np.max(np.abs(np.conj(pols_x).T @ pols_x - np.eye(pols_x.shape[1]))))
    return {"thz": math.copysign(math.sqrt(abs(q)), q) * ry_to_thz, "degeneracy": len(sub),
            "pols_orthonormality_err": ortho}


def spread(vals) -> dict:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"n": 0, "values": []}
    a = np.asarray(v)
    return {"n": len(v), "values": v, "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(v) > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "range": float(a.max() - a.min())}


def boot_stats(vals, imag_tol: float) -> dict:
    a = np.asarray([float(x) for x in vals])
    if a.size == 0:
        return {"n_ok": 0}
    return {"n_ok": int(a.size), "mean_thz": float(a.mean()),
            "std_thz": float(a.std(ddof=1)) if a.size > 1 else None,
            "p2_5_thz": float(np.percentile(a, 2.5)), "p97_5_thz": float(np.percentile(a, 97.5)),
            "min_thz": float(a.min()), "max_thz": float(a.max()),
            "frac_unstable": float(np.mean(a < imag_tol))}


# ---------------------------------------------------------- minimiser step recorder ----

class StepRecorder:
    """Read-only ``custom_function_post`` hook for ``SSCHA_Minimizer.run``.

    run() calls it after every step, once the step's free energy and Kong-Liu size are appended
    and the stopping tests have run (SchaMinimizer.py:1313-1380). It copies the minimiser's own
    histories, tags them with ``minim.population`` (set before each run, Relax.py:385) and
    re-evaluates the stopping tests exactly as check_stop() and run() do. It draws no random
    numbers and writes nothing back, so the trajectory is the production trajectory.
    """

    def __init__(self, ry_to_thz: float):
        self.ry_to_thz = ry_to_thz
        self.steps: list[dict] = []
        self.pop_start: dict[int, float] = {}
        self.errors: list[str] = []

    def _min_thz(self, w):
        w = np.real(np.asarray(w)).astype(float)
        return float(w[nonacoustic_idx(w)].min() * self.ry_to_thz)

    def post(self, minim):
        try:
            fe, fe_err = getattr(minim, "__fe__"), getattr(minim, "__fe_err__")
            gc, gc_err = getattr(minim, "__gc__"), getattr(minim, "__gc_err__")
            gw, gw_err = getattr(minim, "__gw__"), getattr(minim, "__gw_err__")
            kl_hist = getattr(minim, "__KL__")
            ens = minim.ensemble
            pop = int(minim.population)
            if pop not in self.pop_start:
                # Relax sets ensemble.dyn_0 = minim.dyn at the start of each population
                # (Relax.py:369), so w_0 is the population's starting auxiliary dyn.
                self.pop_start[pop] = self._min_thz(ens.w_0)
            n = float(ens.N)
            kl = float(np.real(kl_hist[-1]))
            max_ka = int(minim.max_ka)
            self.steps.append({
                "pop": pop, "n_fe_cumulative": len(fe),
                "fe_Ry": _scalar(fe[-1]), "fe_err_Ry": _scalar(fe_err[-1]),
                "gc": _scalar(gc[-1]), "gc_err": _scalar(gc_err[-1]),
                "gw": _scalar(gw[-1]), "gw_err": _scalar(gw_err[-1]),
                "kl_eff": kl, "kl_ratio": kl / n,
                "aux_min_nonac_thz": self._min_thz(ens.current_w),
                "line_step": _scalar(getattr(minim.minimizer, "step", None)),
                "stop_converged": bool(minim.is_converged()),
                "stop_kong_liu": bool(kl / n < minim.kong_liu_ratio
                                      and minim.minimizer.is_new_direction()),
                "stop_max_ka": bool(max_ka > 0 and len(fe) > max_ka),
            })
        except Exception as exc:          # never let a diagnostic kill the production run
            self.errors.append(f"{type(exc).__name__}: {exc}")


def population_table(steps: list, pop_start: dict, pop_end: dict) -> list:
    """One row per population: steps taken, which stopping test ended it, and how far the
    auxiliary dyn actually moved (from the dyn Relax saved at the population's end)."""
    rows = []
    for p in sorted({s["pop"] for s in steps} | set(pop_start) | set(pop_end)):
        st = [s for s in steps if s["pop"] == p]
        last = st[-1] if st else None
        reason = None
        if last is not None:
            reason = ("converged" if last["stop_converged"] else
                      "kong_liu" if last["stop_kong_liu"] else
                      "max_ka_cumulative" if last["stop_max_ka"] else "other")
        # run() restores the pre-step dyn when it stops unconverged (SchaMinimizer.py:1391-1397),
        # so a Kong-Liu or max_ka stop discards the last recorded step. "other" is an
        # imaginary-frequency stop, and there the count is ambiguous: check_stop's
        # (SchaMinimizer.py:1663-1671) discards the last recorded step, while run()'s break
        # (:1307-1310) discards a step the hook never saw. It is left null, not guessed.
        kept = (len(st) if reason == "converged" else
                max(len(st) - 1, 0) if reason in ("kong_liu", "max_ka_cumulative") else None)
        rows.append({
            "pop": p, "n_steps": len(st),
            "n_steps_kept": kept,
            "stop_reason": reason,
            "fe_first_Ry": st[0]["fe_Ry"] if st else None,
            "fe_last_Ry": last["fe_Ry"] if last else None,
            "fe_err_last_Ry": last["fe_err_Ry"] if last else None,
            "gc_last": last["gc"] if last else None,
            "gc_err_last": last["gc_err"] if last else None,
            "kl_ratio_last": last["kl_ratio"] if last else None,
            "start_aux_min_nonac_thz": pop_start.get(p),
            **pop_end.get(p, {}),
        })
    return rows


def population_end_dyns(work: Path, n_pops: int, nqirr: int, dyn_start, ry: float,
                        tol: float) -> dict:
    """Reload dyn_pop<p>_ (saved after each population, Relax.py:407-408) and measure what
    each population changed relative to the previous one's end (the start dyn for p = 1)."""
    import cellconstructor.Phonons as CCP
    out, prev = {}, dyn_start
    for p in range(1, n_pops + 1):
        prefix = work / f"dyn_pop{p}_"
        if not Path(str(prefix) + "1").exists():
            out[p] = {"end_dyn_file": None}
            continue
        d = CCP.Phonons(str(prefix), int(nqirr))
        out[p] = {"end_aux_min_nonac_thz": freq_summary(dyn_freqs(d)[0], ry, tol)["min_nonac_thz"],
                  "rel_change_vs_prev_end": dyn_rel_change(d, prev),
                  "end_dyn_file": rel(Path(str(prefix) + "1"))}
        prev = d
    return out


# ------------------------------------------------------ Hessian-ensemble uncertainty ----

def antithetic_pairs_ok(ens) -> bool:
    """generate(evenodd=True) stores each draw next to its mirror image, (+u, -u) at (2k, 2k+1)
    (Ensemble.py:1389-1399). Verified on the data, not assumed."""
    u = np.asarray(ens.u_disps)
    if u.shape[0] % 2:
        return False
    scale = max(1.0, float(np.max(np.abs(u))))
    return bool(np.max(np.abs(u[0::2] + u[1::2])) < 1e-8 * scale)


def resample_ensemble(ens, idx):
    """``Ensemble.split`` (Ensemble.py:3938-3979) with an index array in place of its boolean
    mask, so a configuration can be drawn more than once. split() cannot express a draw with
    replacement; everything else (a fresh Ensemble on the same generating dyn, energies and
    forces copied, update_weights to the current dyn) is split() line for line. The Hessian
    reads only the real-space forces, displacements and weights this rebuilds
    (Ensemble.py:3714-3727)."""
    idx = np.asarray(idx, dtype=int)
    sub = type(ens)(ens.dyn_0, ens.T0, ens.dyn_0.GetSupercell())
    sub.init_from_structures([ens.structures[i] for i in idx])
    sub.force_computed[:] = np.asarray(ens.force_computed)[idx]
    sub.stress_computed[:] = np.asarray(ens.stress_computed)[idx]
    sub.energies[:] = np.asarray(ens.energies)[idx]
    sub.forces[:, :, :] = np.asarray(ens.forces)[idx, :, :]
    sub.has_stress = ens.has_stress
    sub.ignore_small_w = ens.ignore_small_w
    if ens.has_stress:
        sub.stresses[:, :, :] = np.asarray(ens.stresses)[idx, :, :]
    sub.update_weights(ens.current_dyn, ens.current_T)
    return sub


def hessian_uncertainty(he, n_boot: int, n_splits: int, rng, ry: float, tol: float,
                        full_min: float) -> dict:
    """Spread of the lowest non-acoustic free-energy-Hessian frequency due to the finite
    Hessian ensemble (R1.4c). The antithetic pair, not the configuration, is the resampling
    unit: the pairing cancels odd-order noise, and resampling single configurations would
    break the cancellation and overstate the spread. Falls back to K disjoint pair-preserving
    splits through the library's own split() if a bootstrap replicate cannot be built.

    resample_ensemble() re-implements split() and was only exercised against stubs before the
    box, so replicate zero is the identity draw (every configuration once, in order): it must
    return the full-ensemble value, or the bootstrap measures some other estimator and the
    library's split() is used instead."""
    out = {"full_ensemble_min_nonac_thz": full_min, "n_configs": int(he.N),
           "antithetic_pairs_verified": antithetic_pairs_ok(he)}
    paired = out["antithetic_pairs_verified"]
    vals, walls = [], []
    try:
        t = time.time()
        h = resample_ensemble(he, np.arange(he.N)).get_free_energy_hessian(include_v4=False)
        ident = freq_summary(dyn_freqs(h)[0], ry, tol)["min_nonac_thz"]
        d = abs(ident - full_min)
        out["identity_replicate"] = {"min_nonac_thz": ident, "abs_diff_thz": d,
                                     "tol_thz": REBUILD_TOL_THZ, "passed": bool(d <= REBUILD_TOL_THZ),
                                     "wall_s": time.time() - t}
        if d > REBUILD_TOL_THZ:
            raise RuntimeError(f"identity replicate gives {ident:+.6f} THz, the full ensemble "
                               f"{full_min:+.6f} THz: resample_ensemble is not split()")
        for _ in range(int(n_boot)):
            if paired:
                k = he.N // 2
                p = rng.integers(0, k, k)
                idx = np.stack([2 * p, 2 * p + 1], axis=1).ravel()
            else:
                idx = rng.integers(0, he.N, he.N)
            t = time.time()
            h = resample_ensemble(he, idx).get_free_energy_hessian(include_v4=False)
            vals.append(freq_summary(dyn_freqs(h)[0], ry, tol)["min_nonac_thz"])
            walls.append(time.time() - t)
        out.update(method="bootstrap_antithetic_pairs" if paired else "bootstrap_configs",
                   B=int(n_boot), values_thz=vals, **boot_stats(vals, tol),
                   wall_s_per_replicate=float(np.mean(walls)) if walls else None)
        return out
    except Exception as exc:
        out["bootstrap_error"] = f"{type(exc).__name__}: {exc}"
        out["bootstrap_traceback"] = traceback.format_exc()
        out["bootstrap_values_before_error"] = vals
    try:
        units = rng.permutation(he.N // 2 if paired else he.N)
        vals = []
        for g in np.array_split(units, int(n_splits)):
            mask = np.zeros(he.N, dtype=bool)
            if paired:
                mask[2 * g] = True
                mask[2 * g + 1] = True
            else:
                mask[g] = True
            h = he.split(mask).get_free_energy_hessian(include_v4=False)
            vals.append(freq_summary(dyn_freqs(h)[0], ry, tol)["min_nonac_thz"])
        st = boot_stats(vals, tol)
        out.update(method="disjoint_splits", K=int(n_splits), values_thz=vals, **st,
                   std_full_estimate_thz=(st["std_thz"] / math.sqrt(n_splits)
                                          if st.get("std_thz") is not None else None),
                   note="bootstrap failed; each split holds N/K configurations, so the "
                        "spread of the full-ensemble estimate is taken as std/sqrt(K)")
    except Exception as exc:
        out["method"] = "none"
        out["split_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ------------------------------------------------------------- C3b config subsample ----

def pick_configs(n: int, k: int, rng, paired: bool) -> list:
    """k configurations without replacement, at most one per antithetic pair: a mirror image
    adds no new geometry to a force benchmark."""
    if paired:
        npair = n // 2
        k = min(int(k), npair)
        pairs = rng.choice(npair, size=k, replace=False)
        member = rng.integers(0, 2, size=k)
        return sorted(int(x) for x in 2 * pairs + member)
    return sorted(int(x) for x in rng.choice(n, size=min(int(k), n), replace=False))


def save_configs(path: Path, frames_in: list, rydberg_ev: float, meta: dict, calc,
                 tol_e: float = 2e-3, tol_f: float = 2e-3) -> dict:
    """Write the C3b subsample as extxyz in eV and eV/A, then prove the units.

    python-sscha stores energies in Ry and forces in Ry/A, positions in Angstrom, having
    divided the calculator's eV by ``Rydberg = ase create_units('2006')['Ry']``
    (Ensemble.py:110-112, 4152-4153). Converting back must use that same constant, not
    cellconstructor's CODATA-2018 RY_TO_EV (Units.py:23); the two differ at 1e-7. The check
    re-reads the written file and re-evaluates the MLIP at the saved positions: a Ry/eV or
    Bohr/A slip shows up as a factor of 13.6 or 1.89, far outside the tolerance.
    """
    import ase.io
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    if not frames_in:
        return {"path": None, "n_saved": 0, "indices": [],
                "unit_check": {"passed": None, "skipped": "--save-configs 0"}}
    frames = []
    for f in frames_in:
        a = Atoms(symbols=f["symbols"], positions=f["xat_ang"], cell=f["cell_ang"], pbc=True)
        a.calc = SinglePointCalculator(a, energy=float(f["energy_ry"]) * rydberg_ev,
                                       forces=np.asarray(f["forces_ry_per_ang"]) * rydberg_ev)
        a.info.update(meta)
        a.info["index"] = int(f["index"])
        a.info["pair"] = int(f["index"]) // 2
        a.arrays["u_disp"] = np.asarray(f["u_disp_ang"], dtype=float)
        frames.append(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(path), frames, format="extxyz")

    back = ase.io.read(str(path), index=0)
    if back.calc is not None:
        e_saved, f_saved = back.get_potential_energy(), back.get_forces()
    else:
        e_saved, f_saved = float(back.info["energy"]), np.asarray(back.arrays["forces"])
    chk = back.copy()
    chk.calc = calc
    e_new, f_new = float(chk.get_potential_energy()), np.asarray(chk.get_forces())
    de, df = abs(e_new - float(e_saved)), float(np.max(np.abs(f_new - f_saved)))
    return {"path": rel(path), "n_saved": len(frames),
            "indices": [int(f["index"]) for f in frames_in],
            "units": {"positions": "Angstrom", "energy": "eV", "forces": "eV/Angstrom",
                      "rydberg_ev_used": rydberg_ev},
            "unit_check": {"index": int(frames_in[0]["index"]) if frames_in else None,
                           "energy_saved_ev": float(e_saved), "energy_reeval_ev": e_new,
                           "abs_dE_ev": de, "max_abs_dF_ev_per_ang": df,
                           "tol_e_ev": tol_e, "tol_f_ev_per_ang": tol_f,
                           "passed": bool(de <= tol_e and df <= tol_f)}}


def ensemble_frames(ens, idx) -> list:
    """Copy the chosen configurations out of a CellConstructor ensemble, in its own units."""
    nat = np.asarray(ens.xats).shape[1]
    u = np.asarray(ens.u_disps).reshape(ens.N, nat, 3)
    frames = []
    for i in idx:
        s = ens.structures[i]
        frames.append({"index": int(i), "symbols": list(s.atoms),
                       "cell_ang": np.asarray(s.unit_cell, dtype=float).copy(),
                       "xat_ang": np.asarray(ens.xats[i], dtype=float).copy(),
                       "energy_ry": float(np.asarray(ens.energies)[i]),
                       "forces_ry_per_ang": np.asarray(ens.forces[i], dtype=float).copy(),
                       "u_disp_ang": u[i].copy()})
    return frames


# --------------------------------------------------------------------- one seed ----

def run_seed_sscha(ctx: dict, seed: int, sp: dict, out: dict) -> None:
    """One production SSCHA computation plus diagnostics; fills ``out`` as it goes so a
    failure still leaves whatever was measured before it."""
    import warnings
    warnings.filterwarnings("ignore")                  # as compute_finite_t_sscha does
    import cellconstructor as CC
    import cellconstructor.Phonons  # noqa: F401
    import sscha
    import sscha.Ensemble
    import sscha.SchaMinimizer
    import sscha.Relax
    from mlip_dynstab.finite_t import _cc_dyn_from_phonopy
    from mlip_dynstab.harmonic import _relax

    R, calc = ctx["recipe"], ctx["calc"]
    T, sc, tol = float(ctx["T"]), tuple(int(x) for x in ctx["supercell"]), R["imag_tol_thz"]
    ry = float(CC.Units.RY_TO_CM * C_THZ_PER_CM)
    work = sp["work"]
    work.mkdir(parents=True, exist_ok=True)
    tm = out.setdefault("timings_s", {})
    out["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
    t_all = t = time.time()

    # --- finite_t.py:1031-1040, with read-only diagnostics interleaved -------------------
    prim = ctx["atoms"].copy()
    prim.calc = calc
    if R["relax"]:
        prim = _relax(prim, fmax=R["fmax"])
    out["relaxed_cellpar"] = [float(x) for x in prim.cell.cellpar()]
    dyn = _cc_dyn_from_phonopy(prim, calc, sc, R["disp"])
    tm["relax_and_harmonic_fc"] = time.time() - t
    dyn_harm = dyn.Copy()
    w_harm, p_harm = dyn_freqs(dyn_harm)
    out["harmonic_pre_fpd"] = {**freq_summary(w_harm, ry, tol),
                               "note": "phonopy full FCs at disp, on the supercell's "
                                       "commensurate q grid (not the 12^3 harmonic-layer mesh)"}
    dyn.ForcePositiveDefinite()
    dyn.Symmetrize()
    dyn_start = dyn.Copy()
    w_start, p_start = dyn_freqs(dyn_start)
    out["start_post_fpd"] = freq_summary(w_start, ry, tol)
    out["dyn_meta"] = {"nqirr": int(dyn.nqirr), "nq": len(dyn.q_tot),
                       "nat_prim": int(dyn.structure.N_atoms),
                       "nat_supercell": int(dyn.structure.N_atoms * len(dyn.q_tot))}

    # --- finite_t.py:1042-1058 --------------------------------------------------------------
    np.random.seed(seed)
    ens = sscha.Ensemble.Ensemble(dyn.Copy(), T, supercell=dyn.GetSupercell())
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ens, root_representation=R["root_representation"])
    minim.min_step_dyn = R["min_step_dyn"]
    minim.meaningful_factor = R["meaningful_factor"]
    minim.max_ka = R["max_ka"]
    relaxer = sscha.Relax.SSCHA(minim, ase_calculator=calc, N_configs=R["n_configs"],
                                max_pop=R["max_pop"], save_ensemble=False)
    rec = StepRecorder(ry)
    relaxer.setup_custom_functions(custom_function_post=rec.post)
    out["_recorder"] = rec
    out["stopping_criteria"] = {
        "meaningful_factor": float(minim.meaningful_factor),
        "abs_conv_thr": float(minim.abs_conv_thr),
        "kong_liu_ratio": float(minim.kong_liu_ratio),
        "gradi_op": minim.gradi_op, "minim_struct": bool(minim.minim_struct),
        "max_ka": int(minim.max_ka), "max_pop": int(relaxer.max_pop),
        "n_configs_per_population": int(relaxer.N_configs),
        "root_representation": minim.root_representation,
        "min_step_dyn": float(minim.min_step_dyn), "precond_dyn": bool(minim.precond_dyn),
        "use_julia": bool(minim.use_julia),
        "semantics": {
            "converged": "gc < gc_err*meaningful_factor (or < abs_conv_thr) AND the same for "
                         "gw, gradi_op 'all' (SchaMinimizer.py:1622-1651)",
            "population_end": "converged, or KL/N < kong_liu_ratio on a new line-search "
                              "direction (SchaMinimizer.py:1654-1661), or max_ka",
            "max_ka": "compared with the CUMULATIVE step history across populations "
                      "(SchaMinimizer.py:1370; histories kept by Relax.py:386)",
            "restore": "an unconverged population discards its last step "
                       "(SchaMinimizer.py:1389-1397)"}}
    t = time.time()
    with chdir(work):
        converged = bool(relaxer.relax(get_stress=False))    # production ignores this value
    tm["sscha_relax"] = time.time() - t
    final_dyn = relaxer.minim.dyn
    n_pops = int(relaxer.start_pop) - 1                     # Relax.py:412,419; starts at 1

    # --- finite_t.py:1061-1064 --------------------------------------------------------------
    t = time.time()
    with chdir(work):
        he = sscha.Ensemble.Ensemble(final_dyn, T, supercell=final_dyn.GetSupercell())
        he.generate(R["n_hessian"])
        he.get_energy_forces(calc, compute_stress=False)
    tm["hessian_ensemble"] = time.time() - t
    # Snapshot and persist before the Hessian: get_free_energy_hessian converts the arrays to
    # Hartree and back in place (Ensemble.py:3669, 3818).
    paired = antithetic_pairs_ok(he)
    cfg_idx = pick_configs(he.N, ctx["save_configs"], np.random.default_rng([CFG_SALT, seed]),
                           paired)
    frames = ensemble_frames(he, cfg_idx)
    he.save_bin(str(sp["ensemble"]), 1)
    out["ensemble_dir"] = rel(sp["ensemble"])
    t = time.time()
    hess = he.get_free_energy_hessian(include_v4=False)
    tm["hessian"] = time.time() - t

    w_hess, p_hess = dyn_freqs(hess)
    out["hessian"] = freq_summary(w_hess, ry, tol)
    out["hessian"]["dynamically_stable"] = bool(out["hessian"]["min_nonac_thz"] >= tol)
    w_aux, p_aux = dyn_freqs(final_dyn)
    out["final_aux"] = freq_summary(w_aux, ry, tol)

    # --- diagnostics: echo of the start, per-population movement, library histories -------
    out["echo"] = {
        "rel_change_final_aux_vs_start": dyn_rel_change(final_dyn, dyn_start),
        "rel_change_hessian_vs_final_aux": dyn_rel_change(hess, final_dyn),
        "rel_change_hessian_vs_start": dyn_rel_change(hess, dyn_start),
        "harmonic_along_hessian_softmode": freq_along_softmode(w_hess, p_hess, w_harm, p_harm, ry),
        "start_along_hessian_softmode": freq_along_softmode(w_hess, p_hess, w_start, p_start, ry),
        "final_aux_along_hessian_softmode": freq_along_softmode(w_hess, p_hess, w_aux, p_aux, ry),
    }
    pop_end = population_end_dyns(work, n_pops, final_dyn.nqirr, dyn_start, ry, tol)
    raw = {k: [_scalar(x) for x in getattr(relaxer.minim, f"__{k}__")]
           for k in ("fe", "fe_err", "gc", "gc_err", "gw", "gw_err", "KL")}
    raw["good_kasteps"] = [int(i) for i in getattr(relaxer.minim, "__good_kasteps__")]
    pops = population_table(rec.steps, rec.pop_start, pop_end)
    kept = [r["n_steps_kept"] for r in pops]
    out["relax"] = {
        "converged": converged, "n_populations": n_pops,
        # n_populations counts ensembles drawn; with the cumulative max_ka most of them may take
        # one step and discard it, so the number that actually moved the dyn is kept apart.
        "n_populations_moving_dyn": (None if None in kept else sum(1 for k in kept if k > 0)),
        "n_steps_kept_total": (None if None in kept else int(sum(kept))),
        "n_steps_total": len(rec.steps),
        "n_force_evaluations": n_pops * int(R["n_configs"]) + int(he.N),
        "populations": pops,
        "steps": rec.steps, "hook_errors": rec.errors,
        "units": {"fe_Ry": "Ry per primitive cell (SchaMinimizer.get_free_energy)",
                  "gc": "norm of the preconditioned dyn gradient, python-sscha internal units",
                  "gw": "norm of the structure gradient, python-sscha internal units",
                  "kl_eff": "Kong-Liu effective sample size; kl_ratio = kl_eff / n_configs",
                  "alignment": "per step, fe/fe_err and kl are evaluated AFTER the step and "
                               "gc/gw BEFORE it (minimization_step computes the gradient, then "
                               "moves). The raw __fe__/__KL__ histories carry an extra leading "
                               "entry from the first init() (SchaMinimizer.py:1150-1156), and "
                               "one more after any population that ended in run()'s "
                               "imaginary-frequency break, so index them by the steps' "
                               "n_fe_cumulative, not by position."},
        "minimizer_raw_histories": raw,
    }
    out.pop("_recorder", None)

    for name, d in (("dyn_harmonic_", dyn_harm), ("dyn_start_fpd_", dyn_start),
                    ("dyn_hessian_", hess)):
        d.save_qe(str(work / name))

    t = time.time()
    out["bootstrap"] = hessian_uncertainty(he, ctx["n_boot"], ctx["n_splits"],
                                           np.random.default_rng([BOOT_SALT, seed]), ry, tol,
                                           out["hessian"]["min_nonac_thz"])
    tm["bootstrap"] = time.time() - t

    meta = {"system": ctx["system"], "model": ctx["model"], "T": float(T), "seed": int(seed),
            "supercell": "x".join(str(x) for x in sc),
            "source": "SSCHA free-energy-Hessian ensemble (final auxiliary dyn)"}
    out["configs"] = save_configs(sp["configs"], frames, float(sscha.Ensemble.Rydberg), meta,
                                  calc)
    out["configs"]["antithetic_pairs_verified"] = paired
    out["configs"]["rng"] = f"default_rng([{CFG_SALT}, seed])"
    tm["total"] = time.time() - t_all


def run_seed_mock(ctx: dict, seed: int, sp: dict, out: dict) -> None:
    """--dry-run stand-in for run_seed_sscha: identical output schema, no sscha or
    cellconstructor. The relax and the MLIP energy/force calls are real, so the extxyz unit
    check is a genuine test; every SSCHA number is synthetic and marked so."""
    from ase.units import create_units
    from mlip_dynstab.harmonic import _relax
    R, calc, tol = ctx["recipe"], ctx["calc"], ctx["recipe"]["imag_tol_thz"]
    rng = np.random.default_rng([MOCK_SALT, seed])
    tm = out.setdefault("timings_s", {})
    t_all = time.time()
    out["MOCK"] = "dry-run: every SSCHA number in this entry is synthetic"
    prim = ctx["atoms"].copy()
    prim.calc = calc
    if R["relax"] and not ctx.get("calc_is_fallback"):
        prim = _relax(prim, fmax=R["fmax"])
    out["relaxed_cellpar"] = [float(x) for x in prim.cell.cellpar()]
    base = ctx["ledger"].get("min_eff_freq_thz") or 1.0
    hmin = float(base + 0.01 * seed)
    blk = lambda v: {"min_nonac_thz": v, "lowest6_thz": [0.0, 0.0, 0.0, v, v, v],  # noqa: E731
                     "n_imag_nonac": int(v < tol), "acoustic_max_abs_thz": 0.0}
    out["harmonic_pre_fpd"] = blk(-abs(base) - 1.0)
    out["start_post_fpd"] = blk(abs(base) + 1.0)
    out["final_aux"] = blk(hmin + 0.02)
    out["hessian"] = {**blk(hmin), "dynamically_stable": bool(hmin >= tol)}
    steps = [{"pop": p, "n_fe_cumulative": 1 + i + 3 * (p - 1), "fe_Ry": -1e-3 * i,
              "fe_err_Ry": 1e-5, "gc": 1e-3 / (i + 1), "gc_err": 1e-4, "gw": 0.0, "gw_err": 0.0,
              "kl_eff": 200.0, "kl_ratio": 0.78, "aux_min_nonac_thz": hmin + 0.02,
              "line_step": 0.5, "stop_converged": False, "stop_kong_liu": False,
              "stop_max_ka": i == 2} for p in (1, 2) for i in range(3)]
    pop_start = {1: abs(base) + 1.0, 2: hmin + 0.02}
    pop_end = {p: {"end_aux_min_nonac_thz": hmin + 0.02, "rel_change_vs_prev_end": 0.0,
                   "end_dyn_file": None} for p in (1, 2)}
    pops = population_table(steps, pop_start, pop_end)
    out["relax"] = {"converged": False, "n_populations": 2,
                    "n_populations_moving_dyn": sum(1 for r in pops if r["n_steps_kept"]),
                    "n_steps_kept_total": sum(r["n_steps_kept"] for r in pops),
                    "n_steps_total": len(steps), "populations": pops,
                    "steps": steps, "hook_errors": []}
    out["bootstrap"] = {"method": "mock", "B": int(ctx["n_boot"]),
                        "full_ensemble_min_nonac_thz": hmin,
                        **boot_stats(rng.normal(hmin, 0.02, int(ctx["n_boot"])), tol)}

    sc_atoms = prim.repeat(tuple(int(x) for x in ctx["supercell"]))
    nat = len(sc_atoms)
    npair = max(int(ctx["save_configs"]), 2)
    u = rng.normal(0.0, 0.03, size=(npair, nat, 3))
    disp = np.empty((2 * npair, nat, 3))
    disp[0::2], disp[1::2] = u, -u
    rydberg = float(create_units("2006")["Ry"])            # the constant sscha.Ensemble uses
    frames_all = []
    t = time.time()
    for i in range(2 * npair):
        a = sc_atoms.copy()
        a.positions = sc_atoms.positions + disp[i]
        a.calc = calc
        frames_all.append({"index": i, "symbols": a.get_chemical_symbols(),
                           "cell_ang": a.cell.array.copy(), "xat_ang": a.positions.copy(),
                           "energy_ry": float(a.get_potential_energy()) / rydberg,
                           "forces_ry_per_ang": a.get_forces() / rydberg,
                           "u_disp_ang": disp[i].copy()})
    tm["mock_mlip_calls"] = time.time() - t
    idx = pick_configs(2 * npair, ctx["save_configs"], np.random.default_rng([CFG_SALT, seed]),
                       True)
    meta = {"system": ctx["system"], "model": ctx["model"], "T": float(ctx["T"]),
            "seed": int(seed), "supercell": "x".join(str(int(x)) for x in ctx["supercell"]),
            "source": "DRY-RUN mock (Gaussian displacements, not an SSCHA ensemble)"}
    out["configs"] = save_configs(sp["configs"], [frames_all[i] for i in idx], rydberg, meta,
                                  calc)
    sp["ensemble"].mkdir(parents=True, exist_ok=True)
    np.save(sp["ensemble"] / "mock_xats.npy", np.array([f["xat_ang"] for f in frames_all]))
    out["ensemble_dir"] = rel(sp["ensemble"])
    tm["total"] = time.time() - t_all


# ------------------------------------------------------------ include_v4 (C1b) ----

def _peak_rss_kb(who: str):
    try:
        import resource
        k = resource.RUSAGE_CHILDREN if who == "children" else resource.RUSAGE_SELF
        return int(resource.getrusage(k).ru_maxrss)       # kB on Linux
    except Exception:
        return None


def _die_with_parent(parent_pid: int) -> bool:
    """Linux: have the kernel SIGKILL this process when its parent dies. The runbook's timeout
    signals the whole process group, but a parent killed by pid would otherwise leave a
    many-thread, multi-GB v4 child competing with the QE queue until the box is destroyed."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        import signal
        ok = ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0) == 0
        if os.getppid() != parent_pid:           # the parent died before prctl took effect
            os._exit(1)
        return bool(ok)
    except Exception:
        return False


def v4_child(ens_dir: str, T: float, imag_tol: float, parent_min_thz: float, out_json: str,
             mock_sleep_s=None, parent_pid=None) -> None:
    """include_v4=True on a saved Hessian ensemble, in a spawned process so the parent can kill
    it at the time cap. It first recomputes include_v4=False on the reloaded ensemble: that
    number must match the parent's, which proves the save/load round trip before the v4 one
    is read. python-sscha 1.6.1 has no julia path for the Hessian (the v4 tensor comes from
    SCHAModules.get_v4, Ensemble.py:3752-3777), so julia cannot speed this up."""
    pdeath = _die_with_parent(int(parent_pid)) if parent_pid else False
    res = {"pid": os.getpid(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
           "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
           "dies_with_parent": pdeath, "stage": "start", "utc_start": utc_now()}
    dump = lambda: atomic_write_json(Path(out_json), res)    # noqa: E731
    dump()
    try:
        if mock_sleep_s is not None:        # --dry-run: exercise spawn, timeout and kill only
            res.update(mock=True, stage="v4_false_done",
                       include_v4_false={"min_nonac_thz": parent_min_thz, "abs_diff_vs_parent_thz": 0.0})
            dump()
            time.sleep(float(mock_sleep_s))
            res.update(stage="done", include_v4_true={"min_nonac_thz": parent_min_thz,
                                                      "wall_s": float(mock_sleep_s)})
            dump()
            return
        import warnings
        warnings.filterwarnings("ignore")
        import cellconstructor as CC
        import cellconstructor.Phonons  # noqa: F401
        import sscha
        import sscha.Ensemble
        ry = float(CC.Units.RY_TO_CM * C_THZ_PER_CM)
        res["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
        res["schamodules"] = schamodules_linkage()
        ens = sscha.Ensemble.load_ensemble_bin(str(ens_dir), 1, float(T))
        res["n_configs"] = int(ens.N)
        t = time.time()
        h = ens.get_free_energy_hessian(include_v4=False)
        v = freq_summary(dyn_freqs(h)[0], ry, imag_tol)
        res["include_v4_false"] = {**v, "wall_s": time.time() - t,
                                   "abs_diff_vs_parent_thz": abs(v["min_nonac_thz"] - parent_min_thz)}
        res["stage"] = "v4_false_done"
        dump()
        t = time.time()
        h4 = ens.get_free_energy_hessian(include_v4=True)
        v4 = freq_summary(dyn_freqs(h4)[0], ry, imag_tol)
        res["include_v4_true"] = {**v4, "wall_s": time.time() - t,
                                  "dynamically_stable": bool(v4["min_nonac_thz"] >= imag_tol)}
        h4.save_qe(str(Path(ens_dir).parent / "dyn_hessian_v4_"))
        res.update(stage="done", utc_end=utc_now(), peak_rss_self_kb=_peak_rss_kb("self"))
        dump()
    except BaseException as exc:
        res.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                   peak_rss_self_kb=_peak_rss_kb("self"))
        dump()
        raise SystemExit(1)


def run_v4(doc: dict, timeout_s: float, threads: int, dry_run: bool, mock_sleep: float) -> dict:
    s0 = (doc.get("seeds") or {}).get("0")
    if not s0 or s0.get("status") != "ok" or not s0.get("ensemble_dir"):
        return {"status": "skipped", "reason": "seed 0 has no completed, saved Hessian ensemble"}
    ens_dir = from_rel(s0["ensemble_dir"])
    out_json = ens_dir.parent / "v4_child.json"
    if out_json.exists():
        out_json.rename(out_json.with_name(f"v4_child.stale-{int(time.time())}.json"))
    tol = doc["recipe"]["imag_tol_thz"]
    parent = float(s0["hessian"]["min_nonac_thz"])
    ctx = multiprocessing.get_context("spawn")
    # The spawned child inherits these. All three: an OpenBLAS or MKL build reads its own
    # variable first, and a stale one from the calling stage would override OMP_NUM_THREADS.
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in thread_vars}
    for k in thread_vars:
        os.environ[k] = str(int(threads))
    try:
        p = ctx.Process(target=v4_child, name="sscha-v4",
                        args=(str(ens_dir), float(doc["unit"]["T"]), tol, parent, str(out_json),
                              mock_sleep if dry_run else None, os.getpid()))
        t0, utc0 = time.time(), utc_now()
        p.start()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(f"[v4] child pid {p.pid}, {threads} OpenMP threads, cap {timeout_s:.0f} s", flush=True)
    p.join(timeout_s)
    killed = False
    if p.is_alive():
        p.terminate()
        p.join(30)
        if p.is_alive():
            p.kill()
            p.join()
        killed = True
    child = load_json(out_json) or {}
    # Classify from what the child wrote, not from the race at the deadline: a child that
    # completes between join(timeout) and terminate() exits 0 with a full result (seen on a
    # loaded machine), and that result is a finished v4 Hessian.
    if child.get("stage") == "done" and p.exitcode == 0:
        status = "finished"
    elif killed:
        status = "timeout"
    else:
        status = "failed"
    v4t = child.get("include_v4_true") or {}
    v4f = child.get("include_v4_false") or {}
    rt = v4f.get("abs_diff_vs_parent_thz")
    return {"status": status, "seed": 0, "timeout_s": float(timeout_s),
            "killed_by_parent": killed,
            "wall_s": time.time() - t0, "utc_start": utc0, "utc_end": utc_now(),
            "exitcode": p.exitcode, "omp_threads": int(threads),
            "v4_min_nonac_thz": v4t.get("min_nonac_thz"),
            "v4_dynamically_stable": v4t.get("dynamically_stable"),
            "v4_false_roundtrip_thz": v4f.get("min_nonac_thz"),
            "v4_false_parent_thz": parent,
            # the v4 number is about the saved ensemble only if the reload reproduced v4=False
            "v4_false_roundtrip_ok": (None if rt is None else bool(rt <= REBUILD_TOL_THZ)),
            "roundtrip_tol_thz": REBUILD_TOL_THZ,
            "peak_rss_children_kb": _peak_rss_kb("children"),
            "ensemble_dir": rel(ens_dir), "child_json": rel(out_json), "child": child,
            "dry_run": bool(dry_run)}


# ---------------------------------------------------------------------- one unit ----

def unit_summary(doc: dict) -> dict:
    seeds = doc.get("seeds") or {}
    ok = {int(k): v for k, v in seeds.items() if v.get("status") == "ok"}
    order = sorted(ok)

    def col(*keys):
        vals = []
        for s in order:
            v = ok[s]
            for k in keys:
                v = v.get(k) if isinstance(v, dict) else None
            vals.append(v)
        return vals

    hess = col("hessian", "min_nonac_thz")
    boot = col("bootstrap", "std_thz")
    tol = (doc.get("recipe") or {}).get("imag_tol_thz", -0.1)
    boot_ok = [b for b in boot if b is not None]
    hs = spread(hess)
    s0 = ok.get(0, {})
    return {
        "seeds_ok": order,
        "seeds_failed": sorted(int(k) for k, v in seeds.items() if v.get("status") == "failed"),
        "seeds_running": sorted(int(k) for k, v in seeds.items() if v.get("status") == "running"),
        "hessian_min_nonac_thz": hs,
        "start_post_fpd_min_nonac_thz": spread(col("start_post_fpd", "min_nonac_thz")),
        "final_aux_min_nonac_thz": spread(col("final_aux", "min_nonac_thz")),
        "harmonic_pre_fpd_min_nonac_thz": spread(col("harmonic_pre_fpd", "min_nonac_thz")),
        "bootstrap_std_thz": boot,
        "seed_std_over_median_boot_std": (hs["std"] / float(np.median(boot_ok))
                                          if hs.get("std") is not None and boot_ok
                                          and float(np.median(boot_ok)) > 0 else None),
        "calls_stable": [bool(h >= tol) for h in hess if h is not None],
        "all_seeds_same_call": (len({bool(h >= tol) for h in hess if h is not None}) <= 1
                                if hess else None),
        "n_populations": col("relax", "n_populations"),
        "n_populations_moving_dyn": col("relax", "n_populations_moving_dyn"),
        "n_steps_kept_total": col("relax", "n_steps_kept_total"),
        "converged": col("relax", "converged"),
        "reproduces_ledger": (s0.get("ledger_comparison") or {}).get("reproduces_ledger"),
        "config_unit_checks_passed": col("configs", "unit_check", "passed"),
        "bootstrap_method": col("bootstrap", "method"),
        "bootstrap_identity_passed": col("bootstrap", "identity_replicate", "passed"),
        "julia_ext_available": col("julia_ext_available"),
        "model_versions": sorted({str(v) for v in col("provenance", "model_version")}),
        "v4_status": (doc.get("v4") or {}).get("status"),
        "v4_min_nonac_thz": (doc.get("v4") or {}).get("v4_min_nonac_thz"),
        "v4_false_roundtrip_ok": (doc.get("v4") or {}).get("v4_false_roundtrip_ok"),
    }


def unit_paths(tag: str, args, c3b: bool = True) -> dict:
    if args.out:
        js = Path(args.out).resolve()
        base = js.parent
    else:
        base = Path(args.out_dir).resolve() if args.out_dir else (
            STUDY_DIR / "_dryrun" if args.dry_run else STUDY_DIR)
        js = base / f"{tag}.json"
    return {"json": js, "configs": base / ("configs" if c3b else "configs_not_c3b"),
            "work": base / "work", "c3b_input": bool(c3b)}


def seed_paths(up: dict, tag: str, seed: int) -> dict:
    work = up["work"] / f"{tag}_seed{seed}"
    return {"work": work, "ensemble": work / "hessian_ensemble",
            "configs": up["configs"] / f"{tag}_seed{seed}.extxyz"}


def update_doc(path: Path, mutate) -> dict:
    """Read-merge-write, so two processes writing different seeds of one unit cannot erase
    each other's entries."""
    doc = load_json(path)
    mutate(doc)
    doc["summary"] = unit_summary(doc)
    doc["last_written_utc"] = utc_now()
    atomic_write_json(path, doc)
    return doc


def load_calculator(model: str, device: str, dry_run: bool, atoms):
    """The production calculator. Under --dry-run a model this env lacks falls back to a
    Lennard-Jones stand-in sized to the cell, so the pipeline still runs end to end."""
    from mlip_dynstab.calculators import get_calculator
    try:
        h = get_calculator(model, device=device)
        return h.calc, h.version, False
    except Exception as exc:
        if not dry_run:
            raise
        from ase.calculators.lj import LennardJones
        d = atoms.get_all_distances(mic=True)
        dmin = float(np.min(d[d > 0])) if np.any(d > 0) else 2.5
        sigma = dmin / 2 ** (1 / 6)
        print(f"[dry-run] {model} unavailable here ({type(exc).__name__}); using a "
              f"Lennard-Jones stand-in (sigma {sigma:.3f} A)")
        return LennardJones(sigma=sigma, epsilon=0.1, rc=2.5 * sigma), \
            f"LennardJones-dry-run(sigma={sigma:.4f})", True


def run_unit(unit: dict, args) -> dict:
    """All requested seeds of one (system, model, T, supercell). Returns flags for the exit
    code. Only 'ok' seeds (and 'failed' ones, unless --retry-failed) are skipped."""
    from mlip_dynstab.systems import get_spec, build_atoms
    system, model, T = unit["system"], unit["model"], float(unit["T"])
    sc = tuple(int(x) for x in unit["supercell"])
    seeds = [int(s) for s in unit["seeds"]]
    tag = unit_tag(system, model, T, sc)
    up = unit_paths(tag, args, c3b_unit(system, model, T, sc))
    recipe = production_recipe()
    spec = get_spec(system)                    # KeyError if the registry lacks the system
    flags = {"failed": [], "unit_check_failed": [], "refused": [], "json": up["json"]}

    # Cheap refusals first, before any model or sscha import.
    existing = load_json(up["json"])
    ident = {"system": system, "model": model, "T": T, "supercell": list(sc)}
    if existing is not None:
        if bool(existing.get("dry_run")) != bool(args.dry_run):
            raise SystemExit(f"{up['json']} was written with dry_run={existing.get('dry_run')}; "
                             "refusing to mix mocked and real seeds. Move it aside.")
        if existing.get("unit") != ident:
            raise SystemExit(f"{up['json']} holds {existing.get('unit')}, not {ident}.")
        if existing.get("recipe") != _clean(recipe):
            raise SystemExit(f"{up['json']} was computed with recipe {existing.get('recipe')}; "
                             f"production is now {recipe}. Refusing to mix; move it aside.")
        done = {int(k) for k, v in (existing.get("seeds") or {}).items()
                if v.get("status") == "ok"
                or (v.get("status") == "failed" and not args.retry_failed)}
        if set(seeds) <= done:
            print(f"[skip] {tag}: seeds {seeds} already recorded")
            return flags
    if not args.dry_run:
        ok, why = env_can_run(model)           # fail before loading a model, not after
        if not ok:
            raise SystemExit(f"{tag}: {why}")
    atoms = build_atoms(spec)
    # Model first, sscha second: the order cli.run_unit -> compute_finite_t_sscha loads the
    # torch and Fortran/OpenMP runtimes in.
    calc, model_version, fallback = load_calculator(model, args.device, args.dry_run, atoms)
    led = ledger_lookup(system, model, T, sc)
    prov = provenance(model, args.dry_run, args.device)
    if not args.dry_run:
        import sscha.Ensemble
        prov["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
        prov["julia_error"] = str(getattr(sscha.Ensemble, "__JULIA_ERROR__", ""))[:500]
        prov["schamodules"] = schamodules_linkage()
    prov["model_version"] = model_version

    # One unit's seeds must share one code path. An importable julia extension switches the
    # minimiser to the Fourier-space gradient (SchaMinimizer.py:232-235, 350-354), a different
    # trajectory from the ledger's, and the runbook's V4_PY may point at such an env; a different
    # model version is a different PES. Seeds that would mix with the unit's 'ok' seeds are
    # refused (not marked failed), so the right env can still compute them later.
    env_sig = {"model_version": model_version,
               "julia_ext_available": prov.get("julia_ext_available")}
    clash = {}
    for k, v in ((existing or {}).get("seeds") or {}).items():
        theirs = {"model_version": (v.get("provenance") or {}).get("model_version"),
                  "julia_ext_available": v.get("julia_ext_available")}
        if v.get("status") == "ok" and theirs != env_sig:
            clash[int(k)] = theirs

    if existing is None:
        atomic_write_json(up["json"], {
            "schema": SCHEMA, "study": "C1/C1b/C5 SSCHA seed study (R1.4)",
            "dry_run": bool(args.dry_run), "unit": ident, "unit_tag": tag,
            "item": unit.get("item"), "why": unit.get("why"),
            "recipe": recipe, "recipe_source": "inspect.signature(finite_t.compute_finite_t_sscha)",
            "settings": {"n_boot": int(args.n_boot), "n_splits_fallback": int(args.n_splits),
                         "save_configs": int(args.save_configs), "ledger_tol_thz": float(args.ledger_tol),
                         "device": args.device, "bootstrap_unit": "antithetic pair",
                         "bootstrap_rng": f"default_rng([{BOOT_SALT}, seed])",
                         "configs_rng": f"default_rng([{CFG_SALT}, seed])"},
            "provenance_created": prov, "seeds": {}, "v4": None,
            "paths": {"configs": rel(up["configs"]), "work": rel(up["work"]),
                      "configs_are_c3b_input": up["c3b_input"]}})

    def _meta(doc):
        doc["ledger"] = led
        doc["provenance_last_run"] = prov
        doc["calculator_fallback"] = bool(fallback)
    update_doc(up["json"], _meta)

    ctx = {"system": system, "model": model, "T": T, "supercell": sc, "atoms": atoms,
           "calc": calc, "recipe": recipe, "ledger": led, "n_boot": int(args.n_boot),
           "n_splits": int(args.n_splits), "save_configs": int(args.save_configs),
           "calc_is_fallback": fallback}
    for seed in seeds:
        prev = (load_json(up["json"])["seeds"] or {}).get(str(seed))
        if prev and prev.get("status") == "ok":
            print(f"[skip] {tag} seed {seed}: already ok")
            continue
        if prev and prev.get("status") == "failed" and not args.retry_failed:
            print(f"[skip] {tag} seed {seed}: failed before ({prev.get('error')}); "
                  "--retry-failed to rerun")
            continue
        if clash:
            flags["refused"].append((tag, seed))
            print(f"[REFUSED] {tag} seed {seed}: this env is {env_sig}, but seeds "
                  f"{sorted(clash)} were computed with {list(clash.values())[0]}; run it "
                  "from the env those seeds used", flush=True)
            continue
        sp = seed_paths(up, tag, seed)
        if sp["work"].exists():                # a dead or failed attempt: keep it, aside
            sp["work"].rename(sp["work"].with_name(f"{sp['work'].name}.stale-{int(time.time())}"))
        entry = {"seed": seed, "status": "running", "utc_start": utc_now(),
                 "host": socket.gethostname(), "work_dir": rel(sp["work"]),
                 # per seed: a resumed run may pass different options than the unit's creator
                 "settings": {"n_boot": int(args.n_boot), "n_splits_fallback": int(args.n_splits),
                              "save_configs": int(args.save_configs),
                              "ledger_tol_thz": float(args.ledger_tol), "device": args.device,
                              "configs_are_c3b_input": up["c3b_input"]}}
        update_doc(up["json"], lambda d: d["seeds"].__setitem__(str(seed), entry))
        print(f"===== {tag} seed {seed} {utc_now()} =====", flush=True)
        t0 = time.time()
        try:
            (run_seed_mock if args.dry_run else run_seed_sscha)(ctx, seed, sp, entry)
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
            rec = entry.get("_recorder")
            if rec is not None:
                entry["relax_partial"] = {"steps": rec.steps, "hook_errors": rec.errors,
                                          "pop_start_aux_min_nonac_thz": rec.pop_start}
            flags["failed"].append((tag, seed))
            print(f"[FAIL] {tag} seed {seed}: {entry['error']}", flush=True)
        entry.pop("_recorder", None)
        entry["wall_s"] = time.time() - t0
        entry["utc_end"] = utc_now()
        entry["provenance"] = {k: prov.get(k) for k in ("git_head", "host", "packages",
                                                         "model_version", "device",
                                                         "omp_num_threads")}
        if entry["status"] == "ok" and seed == 0:
            entry["ledger_comparison"] = ledger_comparison(
                led, entry["hessian"]["min_nonac_thz"], float(args.ledger_tol), model_version)
        uc = ((entry.get("configs") or {}).get("unit_check") or {})
        if entry["status"] == "ok" and uc.get("passed") is False:
            flags["unit_check_failed"].append((tag, seed))
            print(f"[UNIT CHECK FAILED] {tag} seed {seed}: {uc}", flush=True)
        update_doc(up["json"], lambda d: d["seeds"].__setitem__(str(seed), entry))
        if entry["status"] == "ok":
            b = entry.get("bootstrap") or {}
            lc = entry.get("ledger_comparison") or {}
            print(f"[seed {seed}] harm {entry['harmonic_pre_fpd']['min_nonac_thz']:+.3f} | "
                  f"start {entry['start_post_fpd']['min_nonac_thz']:+.3f} | "
                  f"aux {entry['final_aux']['min_nonac_thz']:+.3f} | "
                  f"hess {entry['hessian']['min_nonac_thz']:+.3f} THz "
                  f"(boot sd {b.get('std_thz')}) | npop {entry['relax']['n_populations']}"
                  f"/{entry['relax'].get('n_populations_moving_dyn')} moved "
                  f"conv {entry['relax']['converged']} | repro {lc.get('reproduces_ledger')} "
                  f"| {entry['wall_s']:.0f}s", flush=True)
    return flags


def maybe_run_v4(json_path: Path, args) -> None:
    doc = load_json(json_path)
    prev = doc.get("v4") or {}
    if prev.get("status") == "finished":
        print(f"[skip] v4 already finished for {doc['unit_tag']}")
        return
    if prev.get("status") == "timeout" and prev.get("timeout_s", 0) >= args.v4_timeout:
        print(f"[skip] v4 timed out before at >= {args.v4_timeout:.0f} s; raise --v4-timeout")
        return
    if prev.get("status") == "failed" and not args.retry_failed:
        print("[skip] v4 failed before; --retry-failed to rerun")
        return
    threads = args.v4_threads or cpu_budget()["usable"]
    try:
        block = run_v4(doc, float(args.v4_timeout), int(threads), args.dry_run,
                       float(args.dry_run_v4_sleep))
    except Exception as exc:                   # the seeds are already saved; record and go on
        block = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc(), "timeout_s": float(args.v4_timeout)}
    update_doc(json_path, lambda d: d.__setitem__("v4", block))
    print(f"[v4] {doc['unit_tag']}: {block['status']} after {block.get('wall_s', 0):.0f} s; "
          f"v4 min {block.get('v4_min_nonac_thz')} THz; v4=False round trip "
          f"{block.get('v4_false_roundtrip_thz')} vs parent {block.get('v4_false_parent_thz')}",
          flush=True)


# ---------------------------------------------------------------------- summarize ----

def summarize(directory: Path) -> int:
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        print(f"no unit JSONs in {directory}")
        return 1

    def f(x, w=7, p=3):
        return f"{x:+{w}.{p}f}" if isinstance(x, (int, float)) and x is not None else f"{'-':>{w}}"

    def npop(s):
        a, b = _dig(s, "relax", "n_populations"), _dig(s, "relax", "n_populations_moving_dyn")
        return "-" if a is None else f"{a}/{'?' if b is None else b}"

    # npop is "drawn/moved": ensembles generated / populations whose kept steps moved the dyn
    hdr = (f"{'unit':40s} {'seed':>4s} {'status':>7s} {'harm':>7s} {'start':>7s} {'aux':>7s} "
           f"{'hess':>7s} {'bootsd':>7s} {'npop':>5s} {'conv':>5s} {'repro':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for fp in files:
        doc = load_json(fp)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            continue
        tag = doc.get("unit_tag", fp.stem) + (" [DRY]" if doc.get("dry_run") else "")
        for k in sorted(doc.get("seeds") or {}, key=int):
            s = doc["seeds"][k]
            g = lambda *ks: _dig(s, *ks)  # noqa: E731
            repro = g("ledger_comparison", "reproduces_ledger")
            bsd = g("bootstrap", "std_thz")
            print(f"{tag:40s} {k:>4s} {s.get('status', '?'):>7s} "
                  f"{f(g('harmonic_pre_fpd', 'min_nonac_thz'))} "
                  f"{f(g('start_post_fpd', 'min_nonac_thz'))} {f(g('final_aux', 'min_nonac_thz'))} "
                  f"{f(g('hessian', 'min_nonac_thz'))} "
                  f"{(f'{bsd:7.3f}' if isinstance(bsd, (int, float)) else '      -')} "
                  f"{npop(s):>5s} "
                  f"{str(g('relax', 'converged')) if g('relax', 'converged') is not None else '-':>5s} "
                  f"{('-' if repro is None else str(repro)):>5s}")
        sm = doc.get("summary") or {}
        hs = sm.get("hessian_min_nonac_thz") or {}
        led = doc.get("ledger") or {}
        line = f"    seeds ok {sm.get('seeds_ok')}"
        if hs.get("n"):
            line += (f": Hessian min mean {hs['mean']:+.3f}"
                     + (f" sd {hs['std']:.3f}" if hs.get("std") is not None else "")
                     + f" range [{hs['min']:+.3f}, {hs['max']:+.3f}] THz")
        if led.get("found"):
            line += f"; ledger {led['min_eff_freq_thz']:+.3f}"
        for o in led.get("other_supercells") or []:
            line += f"; ledger sc{''.join(map(str, o['supercell']))} {o['min_eff_freq_thz']:+.3f}"
        if sm.get("seed_std_over_median_boot_std") is not None:
            line += f"; seed sd / boot sd {sm['seed_std_over_median_boot_std']:.2f}"
        if False in (sm.get("bootstrap_identity_passed") or []):
            line += "; BOOTSTRAP IDENTITY CHECK FAILED (split fallback used)"
        if len(sm.get("model_versions") or []) > 1:
            line += f"; MIXED MODEL VERSIONS {sm['model_versions']}"
        print(line)
        v4 = doc.get("v4")
        if v4:
            print(f"    v4: {v4.get('status')} after {v4.get('wall_s', 0):.0f} s "
                  f"(cap {v4.get('timeout_s', 0):.0f}); v4 min {v4.get('v4_min_nonac_thz')}; "
                  f"v4=False round trip {v4.get('v4_false_roundtrip_thz')} "
                  f"vs {v4.get('v4_false_parent_thz')} (ok {v4.get('v4_false_roundtrip_ok')})")
    return 0


def _dig(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


# ======================================================= C1c: converged SSCHA mode ====
#
# The seed study showed that no production run converges: the cumulative max_ka = 20 stops
# population 1 after 19 kept steps and lets populations 2-8 take one discarded step each, so the
# final Hessian sits within ~0.2 THz of the ForcePositiveDefinite start. This mode asks whether
# "SSCHA calls the deep displacive wells stable" survives a relaxation that actually converges.
# Read from the python-sscha 1.6.1 sdist (line numbers are that source's):
#
#  * The convergence test is gc < meaningful_factor * gc_err (and the same for gw), gradi_op
#    "all" (SchaMinimizer.py:1622-1651); the docstring calls meaningful_factor "the ration
#    between the gradient and its error below which the minimization is considered to be
#    converged" (:132-134), default 0.2 (:113).
#  * Without julia that "error" is a placeholder. minimization_step takes the dyn gradient from
#    get_preconditioned_gradient_parallel (:357-364), which computes the serial gradient AND its
#    stochastic error and then returns `gradient, np.zeros_like(gradient) + 1` (Ensemble.py:
#    2641-2657). After the division by sqrt(QE_nsym * prod(supercell)) (:456) gc_err is the
#    constant 3*nat/sqrt(nsym) = 2.165 (BaTiO3), 1.299 (ZrO2), 0.433 (Zr) in every step of every
#    box run, whatever the ensemble. meaningful_factor has therefore been an absolute threshold
#    on gc (1e-4 * 2.165 = 2.2e-4 for BaTiO3), not a noise test; that is why an uncapped
#    population chased the gradient for 1000+ steps.
#  * The discarded error is a real one: the serial routine's average_error_weight is the
#    standard error of the reweighted mean (SCHAModules/module_stochastic.f90:237-284, 1/sqrt(N)
#    delta-method ratio estimator). RealErrorEnsemble hands exactly that back to the minimiser;
#    the gradient that drives the step is still the library's parallel one, so the trajectory is
#    unchanged and only the stopping test gains its documented meaning. Caveat on its scale: the
#    per-element real-space error matrix is Fourier transformed like a force-constant matrix
#    (GetDynQFromFCSupercell(..., fc2=grad_err), Ensemble.py:2786-2803; Phonons.py:4586-4587), so
#    the nq supercell images add coherently at Gamma and cancel at q != 0. Its norm is about
#    sqrt(nq) times a q-resolved error norm, and after the /sqrt(nsym*nq) of :456 the pure-noise
#    value of gc/gc_err depends on the unit's symmetry (roughly sqrt(nsym * d_sym / d_total),
#    O(1-2) here). It is a noise scale that falls as 1/sqrt(N_eff), which is what the stopping
#    test needs, but gc/gc_err is not a calibrated z-score; the fresh-ensemble check below is.
#  * check_stop tests convergence BEFORE the Kong-Liu stop (:1648 vs :1656), after every
#    evaluation including rejected line-search trials (run() :1360-1364; Minimizer.run_step
#    :231-251 retries along the old direction), and the ratio-estimator error grows as the
#    weights degenerate. A trial point with KL/N ~ 0.03 (seen at 600 K) could pass with a real
#    error. The guard: at a gradient evaluated where KL/N < converge_min_kl_ratio the returned
#    error is zero, so no convergence can be declared there. 0.7 is the FAQ's Hessian condition
#    ("the SSCHA minimization must end with a gradient that can be decreased indefinitely
#    without decreasing the KL below 0.7 /0.8", UserGuide/faq.rst:214).
#  * max_ka is compared with the cumulative history (:1370; Relax re-enters init with
#    delete_previous_data=False, Relax.py:386). The per-population cap is set from the pre-step
#    hook at each population's first step: max_ka = len(__fe__) + cap - 1 allows exactly `cap`
#    steps (each step appends one __fe__, :1315). A capped population keeps all but its last
#    step (:1389-1397) and the next population is drawn from there.
#  * Kong-Liu 0.5 is the library default and the FAQ's "good and safe value" (faq.rst:301); the
#    line search caps each accepted direction at a 10% KL drop (Minimizer.py:61, 242), so a
#    population ends after ~6-7 accepted directions (~50-70 evaluations on the box logs).
#  * "when your calculation stops because it converged (not because it runs out of
#    iterations), then it should be well converged" (faq.rst:196); "The free energy hessian
#    requires much more configurations than the SCHA minimization" (faq.rst:214); the gradient
#    error scales as 1/sqrt(N) (faq.rst:284).
#
# Convergence evidence recorded per (unit, start): the library's own stop reason with the real
# error; KL at the converged gradient; an independent test on the FRESH Hessian ensemble drawn
# at the final dyn (KL = 1); and start independence. The fresh test is calibrated on the data,
# not on the library's error normalisation: the symmetrised gradient is recomputed on K disjoint
# antithetic-pair splits, whose spread gives its pure-noise expectation, R = |g|^2 / tr Cov(g)
# is ~1 at the exact SCHA minimum, and at a dyn that carries the last population's own sampling
# error it is ~1 + n_hessian / (KL * n_configs); much larger means a systematic gradient remains.
# A converged population keeps the step taken from its converged gradient (no restore when
# converged, SchaMinimizer.py:1391), so the final dyn is one step past that gradient; kl_ratio
# after that step is recorded. Start A is the production start (ForcePositiveDefinite,
# imaginary omega -> |omega|, far above the instability); start B replaces each imaginary
# non-acoustic mode by a small positive frequency (start_b_thz) and leaves every other mode as
# ForcePositiveDefinite builds it. A converged SSCHA reaches the same auxiliary dyn and Hessian
# from both. B draws from global seed base+1, A from base: shared random numbers would make the
# two Hessians agree beyond what their bootstrap sd (which assumes independent noise) allows.

CONV_PRESET = "converged"
CONV_SCHEMA = "sscha_converged/v1"
CONV_DIR = REPO / "results" / "revision" / "sscha_converged"
CONV_STARTS = ("A", "B")
CONV_SEED = 0
CONV_MEANINGFUL = 0.2        # SchaMinimizer.py:113, with the REAL error (RealErrorEnsemble)
CONV_KL_RATIO = 0.5          # SchaMinimizer.py:113; faq.rst:301
CONV_MIN_KL = 0.7            # faq.rst:214; KL/N required at a gradient that may converge
CONV_DEFAULTS = {"n_configs": 1000, "max_pop": 30, "n_hessian": 2000, "max_steps_per_pop": 400,
                 "start_b_thz": 0.3, "unit_timeout_s": 7200.0}
CONV_SALT = 20260929
FRESH_SPLITS = 8             # disjoint antithetic-pair splits for the fresh-gradient noise estimate
# R / R_expected at or below this: the fresh gradient is consistent with a converged dyn. Under
# that hypothesis R / R_expected is roughly chi^2_d / d with d the number of symmetric gradient
# components (a handful for bcc Zr, tens for the perovskites); 2.5 is its ~97% point at d = 5.
FRESH_R_FACTOR_OK = 2.5
AB_Z_TOL = 3.0               # |Hessian min A - B| within 3 combined bootstrap sd: start-independent
AB_ABS_TOL_THZ = 0.05   # THz; see conv_compare
IMAG_REPLACE_TOL_THZ = 1e-3  # start B replaces only modes below -1e-3 THz (not numerical zeros)
FLUSH_EVERY_STEPS = 50

CONV_SEMANTICS = {
    "converged": "gc < meaningful_factor * gc_err AND gw < meaningful_factor * gw_err (or < "
                 "abs_conv_thr), gradi_op 'all' (SchaMinimizer.py:1622-1651). gc_err is the "
                 "library's serial stochastic error (module_stochastic.f90:270-274), not the "
                 "parallel wrapper's placeholder (Ensemble.py:2657); it is set to zero where "
                 "KL/N at the gradient is below converge_min_kl_ratio, so convergence cannot be "
                 "declared at a statistically invalid point. Its scale is the Fourier transform "
                 "of the real-space error matrix (coherent at Gamma), divided by "
                 "sqrt(nsym*nq) (:456): a 1/sqrt(N_eff) noise scale, not a calibrated z-score. "
                 "The structures here have no free Wyckoff parameter, so gw is zero by symmetry "
                 "and the test reduces to gc",
    "converged_dyn": "a converged population is not restored (:1391), so the final dyn is one "
                     "step past the gradient that passed; kl_ratio_at_end is the KL after it",
    "population_end": "converged, KL/N < kong_liu_ratio on a new direction (:1654-1661), or "
                      "max_steps_per_pop steps (max_ka reset per population by the pre hook)",
    "relax_end": "a converged population (Relax.py:410-411), max_pop populations, or the "
                 "wall-clock cap (the post hook ends the current population after one more step "
                 "and sets max_pop so no new population is drawn, Relax.py:412-416)",
    "restore": "an unconverged population discards its last step (SchaMinimizer.py:1389-1397)",
}

CONVERGED_UNITS = [
    dict(system="batio3_cubic", model="mace_mp0", T=100.0, supercell=(2, 2, 2), item="C1c",
         why="Table S2 false-stable: harmonic Gamma mode -6.3 THz; production SSCHA stayed at "
             "the ForcePositiveDefinite start"),
    dict(system="zro2_cubic", model="mace_mp0", T=100.0, supercell=(2, 2, 2), item="C1c",
         why="fluorite false-stable: harmonic -5.5 THz; production SSCHA stayed at the start"),
    dict(system="srtio3_cubic", model="mace_mp0", T=100.0, supercell=(2, 2, 2), item="C1c",
         why="positive control: labelled unstable at 100 K (AFD R mode, transition 105 K); a "
             "converged SSCHA should call it unstable"),
    dict(system="zr_bcc", model="mattersim", T=50.0, supercell=(2, 2, 2), item="C1c",
         why="bcc with a harmonic instability (-2.0 THz), low T"),
    dict(system="zr_bcc", model="mattersim", T=300.0, supercell=(2, 2, 2), item="C1c",
         why="bcc at 300 K on the ledger's 2x2x2 cell"),
    dict(system="zr_bcc", model="mattersim", T=300.0, supercell=(3, 3, 3), item="C1c",
         why="bcc at 300 K, 3x3x3: the production 3x3x3 run gave -2.1 THz against +1.95 on 2x2x2"),
]


def conv_base_dir(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir).resolve()
    return CONV_DIR / "_dryrun" if args.dry_run else CONV_DIR


def conv_paths(base: Path, tag: str, start: str) -> dict:
    work = Path(base) / "work" / f"{tag}_start{start}"
    return {"json": Path(base) / f"{tag}_start{start}.json", "work": work,
            "ensemble": work / "hessian_ensemble", "compare": Path(base) / f"{tag}_AB.json"}


def converged_recipe(args) -> dict:
    """The production recipe (relax, disp, fmax, root2, min_step_dyn, imag_tol) with the
    relaxation and Hessian settings of the converged mode on top. Stored in every JSON; a run
    with a different recipe refuses to write into a file made with another one."""
    r = production_recipe()
    r.update(n_configs=int(args.conv_n_configs), max_pop=int(args.conv_max_pop),
             n_hessian=int(args.conv_n_hessian), meaningful_factor=CONV_MEANINGFUL,
             max_ka="per_population", max_steps_per_pop=int(args.conv_steps_per_pop),
             kong_liu_ratio=CONV_KL_RATIO, converge_min_kl_ratio=CONV_MIN_KL,
             gradient_error="serial_stochastic_error", start_b_thz=float(args.start_b_thz),
             # julia's get_fourier_gradient (SchaMinimizer.py:350-354) would bypass the real-error
             # subclass and its KL guard; the box envs have no julia, and this keeps it so.
             use_julia=False,
             seed=CONV_SEED, seed_rule="np.random.seed(seed + index of start), A=0, B=1",
             fresh_splits=FRESH_SPLITS)
    return r


def conv_global_seed(R: dict, start: str) -> int:
    return int(R["seed"]) + CONV_STARTS.index(start)


def real_error_ensemble_class(base, log: list, min_kl_ratio: float):
    """A subclass of ``base`` (sscha.Ensemble.Ensemble) whose get_preconditioned_gradient_parallel
    returns the library's parallel gradient together with the library's serial stochastic error
    instead of the placeholder array of ones (Ensemble.py:2657).

    The serial routine is the one the parallel wrapper already calls per chunk and whose error it
    discards (Ensemble.py:2641); on one process the chunk is the whole ensemble, so the two
    gradients agree to rounding and the relative difference is logged at every call. Where KL/N at
    the gradient is below ``min_kl_ratio`` the returned error is zero (no convergence can be
    declared there); a failure of the serial call also returns zero, never the placeholder.
    Ensemble fixes its attribute set (Ensemble.py:276-304), so the diagnostics go to ``log``."""

    class RealErrorEnsemble(base):
        def get_preconditioned_gradient_parallel(self, *args, timer=None, **kwargs):
            t0 = time.time()
            grad, placeholder = base.get_preconditioned_gradient_parallel(self, *args, timer=timer,
                                                                          **kwargs)
            rec = {"call": len(log), "t_parallel_s": time.time() - t0,
                   "placeholder_raw_norm": float(np.linalg.norm(np.asarray(placeholder)))}
            try:
                t1 = time.time()
                sub = args[0] if args else kwargs.get("subtract_sscha", True)
                kw = {k: v for k, v in kwargs.items() if k not in ("subtract_sscha", "return_error")}
                g_ser, err = self.get_preconditioned_gradient(sub, True, *args[2:], **kw)
                err = np.array(err, dtype=np.complex128)
                g = np.asarray(grad)
                gn = float(np.linalg.norm(g))
                dn = float(np.linalg.norm(np.asarray(g_ser) - g))
                kl = float(self.get_effective_sample_size()) / float(self.N)
                rec.update(t_serial_s=time.time() - t1, kl_ratio_at_gradient=kl,
                           err_raw_norm=float(np.linalg.norm(err)),
                           grad_rel_diff_parallel_vs_serial=(dn / gn if gn > 0 else dn),
                           guarded=bool(kl < min_kl_ratio))
                if rec["guarded"]:
                    err[...] = 0.0
                log.append(rec)
                return grad, err
            except Exception as exc:
                rec.update(error=f"{type(exc).__name__}: {exc}", guarded=True)
                log.append(rec)
                return grad, np.zeros(np.shape(placeholder), dtype=np.complex128)

    RealErrorEnsemble.__name__ = RealErrorEnsemble.__qualname__ = f"RealError{base.__name__}"
    return RealErrorEnsemble


class PopulationController:
    """``custom_function_pre`` / post companion that turns the cumulative max_ka into a
    per-population cap and enforces a wall-clock cap on the relaxation.

    pre(): at a population's first step (Relax sets minim.population before init and run,
    Relax.py:385-388) it sets max_ka = len(__fe__) + cap - 1, and flushes the previous
    population's record. post(): past the deadline it sets max_ka = len(__fe__) (run() stops
    after one more step, SchaMinimizer.py:1370) and relaxer.max_pop = population (Relax draws no
    further population, Relax.py:412-416), so a capped run ends with the dyn and history it had.
    It never touches the dyn, the ensemble or the RNG."""

    def __init__(self, relaxer, steps_per_pop: int, deadline=None, on_flush=None,
                 flush_every: int = FLUSH_EVERY_STEPS):
        self.relaxer = relaxer
        self.cap = max(1, int(steps_per_pop))
        self.deadline = deadline
        self.on_flush = on_flush
        self.flush_every = int(flush_every)
        self.t0 = time.time()
        self.cur_pop = None
        self.pops: dict[int, dict] = {}
        self.capped = None
        self.errors: list[str] = []
        self._since_flush = 0

    def state(self) -> dict:
        return {"steps_per_pop": self.cap, "pops": self.pops, "capped": self.capped,
                "deadline_after_s": (None if self.deadline is None else self.deadline - self.t0),
                "errors": self.errors}

    def _flush(self, reason: str):
        self._since_flush = 0
        if self.on_flush is not None:
            try:
                self.on_flush(reason)
            except Exception as exc:
                self.errors.append(f"flush: {type(exc).__name__}: {exc}")

    def pre(self, minim):
        try:
            pop = int(minim.population)
            if pop == self.cur_pop:
                return
            prev, self.cur_pop = self.cur_pop, pop
            n0 = len(getattr(minim, "__fe__"))
            if self.capped is None:
                minim.max_ka = n0 + self.cap - 1
            self.pops[pop] = {"n_fe_at_start": n0, "max_ka_set": int(minim.max_ka),
                              "t_first_step_s": time.time() - self.t0}
            if prev is not None:
                self._flush(f"end of population {prev}")
        except Exception as exc:
            self.errors.append(f"pre: {type(exc).__name__}: {exc}")

    def post(self, minim):
        try:
            now = time.time()
            row = self.pops.get(self.cur_pop)
            if row is not None:
                row["t_last_step_s"] = now - self.t0
            if self.capped is None and self.deadline is not None and now > self.deadline:
                n = len(getattr(minim, "__fe__"))
                minim.max_ka = n
                self.relaxer.max_pop = int(minim.population)
                self.capped = {"pop": int(minim.population), "n_fe": n, "t_s": now - self.t0}
                print(f"[wall cap] relaxation stopped in population {minim.population} after "
                      f"{now - self.t0:.0f} s", flush=True)
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._flush("periodic")
        except Exception as exc:
            self.errors.append(f"post: {type(exc).__name__}: {exc}")


class ConvRecorder(StepRecorder):
    """StepRecorder plus, per step, the real gradient error from RealErrorEnsemble's log, scaled
    exactly as minimization_step scales the error it is handed (/ sqrt(QE_nsym * prod(supercell)),
    SchaMinimizer.py:456, with QE_nsym stored as minim.N_symmetries, :337), the KL at the
    gradient, whether the guard fired, and the wall time. Read-only like its parent."""

    def __init__(self, ry_to_thz: float, grad_log: list):
        super().__init__(ry_to_thz)
        self.grad_log = grad_log
        self._seen = 0
        self.t0 = time.time()

    def post(self, minim):
        n = len(self.steps)
        super().post(minim)
        new = self.grad_log[self._seen:]
        self._seen = len(self.grad_log)
        if len(self.steps) == n:
            return
        try:
            st = self.steps[-1]
            st["t_s"] = time.time() - self.t0
            st["n_gradient_calls"] = len(new)
            if new:
                g = new[-1]
                div = math.sqrt(float(getattr(minim, "N_symmetries", 1) or 1)
                                * float(np.prod(np.asarray(minim.ensemble.supercell))))
                real = g.get("err_raw_norm")
                st["gc_err_real"] = None if real is None else real / div
                st["gc_err_placeholder"] = g["placeholder_raw_norm"] / div
                st["gc_over_err_real"] = (st["gc"] / st["gc_err_real"]
                                          if st.get("gc_err_real") else None)
                st["kl_ratio_at_gradient"] = g.get("kl_ratio_at_gradient")
                st["convergence_guarded"] = g.get("guarded")
                st["grad_rel_diff_parallel_vs_serial"] = g.get("grad_rel_diff_parallel_vs_serial")
                st["t_gradient_s"] = (g.get("t_parallel_s") or 0.0) + (g.get("t_serial_s") or 0.0)
                if g.get("error"):
                    st["gradient_error_failure"] = g["error"]
        except Exception as exc:
            self.errors.append(f"conv: {type(exc).__name__}: {exc}")


def wire_conv_hooks(relaxer, ry: float, grad_log: list, R: dict, ctx: dict, out: dict):
    """Attach recorder and controller to the relaxer. The deadline counts from here, i.e. it caps
    the relaxation; the Hessian ensemble and bootstrap after it are bounded by n_hessian/n_boot."""
    rec = ConvRecorder(ry, grad_log)
    timeout = float(ctx.get("unit_timeout") or 0.0)
    deadline = (time.time() + timeout) if timeout > 0 else None

    def flush(reason):
        out["relax_partial"] = {"reason": reason, "utc": utc_now(), "n_steps": len(rec.steps),
                                "steps": rec.steps, "pop_start_aux_min_nonac_thz": rec.pop_start,
                                "controller": ctrl.state(), "hook_errors": rec.errors}
        ctx["flush"]()

    ctrl = PopulationController(relaxer, R["max_steps_per_pop"], deadline, flush)

    def post(minim):
        n = len(rec.steps)
        rec.post(minim)
        ctrl.post(minim)
        # A stop test fired: this is the population's last step, so write it now rather than
        # at the next population's first step, which comes after N_configs force calls.
        st = rec.steps[-1] if len(rec.steps) > n else {}
        if st.get("stop_converged") or st.get("stop_kong_liu") or st.get("stop_max_ka"):
            ctrl._flush(f"end of population {st.get('pop')} (stop test fired)")

    relaxer.setup_custom_functions(custom_function_pre=ctrl.pre, custom_function_post=post)
    out["_recorder"], out["_controller"] = rec, ctrl
    return rec, ctrl


def conv_stopping_criteria(minim, relaxer, R: dict) -> dict:
    julia = bool(getattr(minim, "use_julia", False))
    return {"meaningful_factor": float(minim.meaningful_factor),
            "abs_conv_thr": float(minim.abs_conv_thr),
            "kong_liu_ratio": float(minim.kong_liu_ratio),
            "converge_min_kl_ratio": float(R["converge_min_kl_ratio"]),
            "gradi_op": minim.gradi_op, "minim_struct": bool(minim.minim_struct),
            "max_steps_per_pop": int(R["max_steps_per_pop"]), "max_pop": int(relaxer.max_pop),
            "n_configs_per_population": int(relaxer.N_configs),
            "root_representation": getattr(minim, "root_representation", None),
            "min_step_dyn": getattr(minim, "min_step_dyn", None),
            "precond_dyn": getattr(minim, "precond_dyn", None), "use_julia": julia,
            "gradient_error_source": ("julia get_fourier_gradient error (native; the guard is "
                                      "not applied)" if julia else
                                      "serial get_preconditioned_gradient error via "
                                      "RealErrorEnsemble"),
            "semantics": CONV_SEMANTICS}


def conv_population_table(steps: list, pop_start: dict, pop_end: dict, ctrl) -> list:
    """population_table plus the converged-mode columns: the real error and gc/err at the last
    step, KL at that gradient, whether the population ended converged, which cap ended it, and
    its wall times (minimisation, and the gap before it = ensemble + forces + finalize)."""
    rows = population_table(steps, pop_start, pop_end)
    prev_end = 0.0
    for r in rows:
        p = r["pop"]
        st = [s for s in steps if s["pop"] == p]
        last = st[-1] if st else {}
        c = ctrl.pops.get(p, {})
        if r["stop_reason"] == "max_ka_cumulative":
            r["stop_reason"] = ("wall_cap" if ctrl.capped and ctrl.capped["pop"] == p
                                else "per_population_step_cap")
        r["converged_at_end"] = r["stop_reason"] == "converged"
        t1, t2 = c.get("t_first_step_s"), c.get("t_last_step_s")
        r.update(gc_err_real_last=last.get("gc_err_real"),
                 gc_over_err_real_last=last.get("gc_over_err_real"),
                 kl_ratio_at_gradient_last=last.get("kl_ratio_at_gradient"),
                 guarded_last=last.get("convergence_guarded"),
                 gw_last=last.get("gw"), gw_err_last=last.get("gw_err"),
                 n_fe_at_start=c.get("n_fe_at_start"), max_ka_set=c.get("max_ka_set"),
                 t_first_step_s=t1, t_last_step_s=t2,
                 minimization_wall_s=(t2 - t1 if t1 is not None and t2 is not None else None),
                 ensemble_and_overhead_before_s=(t1 - prev_end if t1 is not None else None))
        if t2 is not None:
            prev_end = t2
    return rows


def conv_relax_block(rec, ctrl, pop_end: dict, converged: bool, n_pops: int, R: dict,
                     grad_log: list) -> dict:
    pops = conv_population_table(rec.steps, rec.pop_start, pop_end, ctrl)
    last = rec.steps[-1] if rec.steps else {}
    reason = "converged" if converged else ("wall_cap" if ctrl.capped else "max_pop")
    diffs = [g["grad_rel_diff_parallel_vs_serial"] for g in grad_log
             if g.get("grad_rel_diff_parallel_vs_serial") is not None]
    kl_end = last.get("kl_ratio_at_gradient", last.get("kl_ratio"))
    return {
        "converged": bool(converged), "stop_reason": reason,
        "not_converged_within_cap": reason == "wall_cap",
        "n_populations": int(n_pops), "n_steps_total": len(rec.steps),
        "n_force_evaluations_relax": int(n_pops) * int(R["n_configs"]),
        "kl_ratio_at_end": last.get("kl_ratio"), "kl_ratio_at_gradient_end": kl_end,
        "gc_over_err_real_end": last.get("gc_over_err_real"),
        "faq_hessian_kl_ok": (bool(kl_end is not None and kl_end >= CONV_MIN_KL)
                              if converged else None),
        "gradient_checks": {
            "n_calls": len(grad_log),
            "n_guarded": sum(1 for g in grad_log if g.get("guarded")),
            "n_errors": sum(1 for g in grad_log if g.get("error")),
            "max_rel_diff_parallel_vs_serial": max(diffs) if diffs else None,
            "mean_t_parallel_s": (float(np.mean([g["t_parallel_s"] for g in grad_log]))
                                  if grad_log else None),
            "mean_t_serial_s": (float(np.mean([g["t_serial_s"] for g in grad_log
                                               if "t_serial_s" in g]))
                                if any("t_serial_s" in g for g in grad_log) else None)},
        "wall_cap": ctrl.capped, "controller": ctrl.state(),
        "populations": pops, "steps": rec.steps, "hook_errors": rec.errors,
        "units": {"gc / gc_err": "as the minimiser compares them (SchaMinimizer.py:596-600, "
                                 "456); gc_err is 0 where the KL guard fired",
                  "gc_err_real": "the serial stochastic error, same scaling, guard ignored",
                  "gc_err_placeholder": "what the library's own test would have used",
                  "kl_ratio": "KL/N after the step; kl_ratio_at_gradient: before it",
                  "fe_Ry": "Ry per primitive cell"},
    }


def soften_imaginary_modes(dyn, w_small_ry: float, ry_to_thz: float):
    """Start B: ForcePositiveDefinite's construction (cellconstructor Phonons.py:1576-1607: per q,
    DyagDinQ, then Phi' = sqrt(M_a M_b) sum_mu w_mu^2 e_mu^a e_mu^b*), except that each imaginary
    non-acoustic mode gets w_small^2 instead of |w|^2. Real modes, and the three acoustic modes at
    Gamma (the three nearest zero, as nonacoustic_idx), are rebuilt exactly as there; modes
    within IMAG_REPLACE_TOL_THZ of zero are left to that rule too. The caller symmetrizes, as
    production does after ForcePositiveDefinite. Returns (new dyn, list of replaced modes)."""
    out = dyn.Copy()
    nat = out.structure.N_atoms
    mass1 = np.zeros(3 * nat)
    for i in range(nat):
        mass1[3 * i: 3 * i + 3] = out.structure.masses[out.structure.atoms[i]]
    msq = np.sqrt(np.outer(mass1, mass1))
    replaced = []
    for iq in range(len(out.dynmats)):
        w, pols = out.DyagDinQ(iq)
        w = np.real(np.asarray(w)).astype(float)
        q = np.asarray(out.q_tot[iq], dtype=float)
        acoustic = (set(int(i) for i in np.argsort(np.abs(w))[:3])
                    if float(np.sqrt(q.dot(q))) < 1e-6 else set())
        w2 = w ** 2
        for i in range(len(w)):
            if i not in acoustic and w[i] * ry_to_thz < -IMAG_REPLACE_TOL_THZ:
                w2[i] = w_small_ry ** 2
                replaced.append({"iq": iq, "q": q.tolist(), "mode": i,
                                 "harmonic_thz": float(w[i] * ry_to_thz),
                                 "start_thz": float(w_small_ry * ry_to_thz)})
        out.dynmats[iq] = np.einsum("i, ji, ki", w2, pols, np.conj(pols)) * msq
        # At q = -q + G the dynamical matrix is real, but DyagDinQ may return complex
        # eigenvectors inside a degenerate subspace; if the replacement does not act uniformly
        # on that subspace the rebuilt matrix picks up an imaginary part and cellconstructor's
        # DiagonalizeSupercell asserts (seen on BaTiO3 start B). Keep it real where the input was.
        if np.max(np.abs(np.imag(dyn.dynmats[iq]))) < 1e-12:
            out.dynmats[iq] = np.real(out.dynmats[iq]) + 0j
    return out, replaced


def build_start(dyn_harm, start: str, start_b_thz: float, ry: float):
    """(start dyn, record). A: the production start (finite_t.py:1039-1040). B: see
    soften_imaginary_modes. Both symmetrized afterwards, as production does."""
    if start == "A":
        d = dyn_harm.Copy()
        d.ForcePositiveDefinite()
        d.Symmetrize()
        return d, {"start": "A", "rule": "ForcePositiveDefinite (imaginary w -> |w|) + Symmetrize",
                   "source": "finite_t.py:1039-1040 (the production start)"}
    d, replaced = soften_imaginary_modes(dyn_harm, float(start_b_thz) / ry, ry)
    d.Symmetrize()
    return d, {"start": "B",
               "rule": f"ForcePositiveDefinite construction with every imaginary non-acoustic mode "
                       f"set to +{start_b_thz:g} THz instead of |w|, + Symmetrize",
               "start_b_thz": float(start_b_thz), "replaced_modes": replaced,
               "n_replaced": len(replaced), "identical_to_A": not replaced}


def split_noise_statistic(g_full, parts) -> dict:
    """R = |g|^2 / tr Cov(g) from K disjoint splits. ``parts`` is [(n_k, g_k)]: the gradient on
    each split and its size in resampling units (antithetic pairs). With rho = 1 the gradient is
    an average over configurations followed by linear maps (Fourier transform, SymmetrizeFCQ), so
    g = sum_k (n_k/N) g_k (checked: lin_rel_err) and tr Cov(g) = sum_k n_k |g_k - g|^2 /
    ((K-1) N). Pure numpy, so it is tested without sscha."""
    g_full = np.asarray(g_full)
    n = np.array([float(p[0]) for p in parts])
    G = np.array([np.asarray(p[1]) for p in parts])
    k, N = len(parts), float(n.sum())
    gbar = np.tensordot(n / N, G, axes=1)
    gnorm2 = float(np.sum(np.abs(g_full) ** 2))
    lin = float(np.sqrt(np.sum(np.abs(gbar - g_full) ** 2) / gnorm2)) if gnorm2 > 0 else None
    dev2 = np.array([float(np.sum(np.abs(G[i] - gbar) ** 2)) for i in range(k)])
    tr_cov = float(np.sum(n * dev2) / ((k - 1) * N)) if k > 1 else None
    r = gnorm2 / tr_cov if tr_cov else None
    return {"K": int(k), "split_sizes": [int(x) for x in n], "lin_rel_err": lin,
            "gnorm2": gnorm2, "tr_cov": tr_cov, "R": r,
            "noise_ratio": None if r is None else math.sqrt(r)}


def fresh_gradient_check(he, dyn, rng, n_relax: int, kl_relax_end, n_splits: int = FRESH_SPLITS) -> dict:
    """The gradient on the fresh Hessian ensemble drawn AT the final auxiliary dyn (KL = 1), with
    the symmetrization of minimization_step (non-spglib branch, SchaMinimizer.py:330-337, 449-451,
    596). Unlike the last population's reweighted gradient, which the minimiser drove down on that
    very ensemble, this one is independent.

    Two numbers. gc_over_err uses the library's error and its /sqrt(nsym*nq) (:456); its
    pure-noise value depends on the unit's symmetry and supercell (see the C1c comment block), so
    it is recorded, not judged. The judged one is calibrated on the data: the same symmetrised
    gradient on ``n_splits`` disjoint antithetic-pair splits (Ensemble.split, the library's own)
    gives tr Cov(g), and R = |g|^2 / tr Cov(g) (split_noise_statistic). At the exact SCHA minimum
    E[R] = 1; the relaxed dyn carries the last population's sampling error, which adds about
    n_hessian / (KL * n_configs), so R_expected = 1 + that, and R / R_expected well above 1 means
    a systematic gradient the relaxation did not remove. Reads the ensemble only; run it before
    get_free_energy_hessian, which converts the ensemble's arrays in place (Ensemble.py:3669,
    3818)."""
    import cellconstructor as CC
    import cellconstructor.symmetries  # noqa: F401
    t = time.time()
    qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
    qe_sym.SetupQPoint(verbose=False)
    nsym = int(qe_sym.QE_nsym)

    def sym_grad(ens, with_error):
        res = ens.get_preconditioned_gradient(True, with_error, preconditioned=1)
        g, e = res if with_error else (res, None)
        g = np.array(g, dtype=np.complex128)
        qe_sym.SymmetrizeFCQ(g, dyn.q_stars, asr="custom")
        return g, (None if e is None else np.array(e, dtype=np.complex128))

    g, e = sym_grad(he, True)
    e /= np.sqrt(nsym * np.prod(np.asarray(he.supercell)))
    gc = float(np.sqrt(np.sum(np.abs(g) ** 2)))
    gc_err = float(np.sqrt(np.sum(np.abs(e) ** 2)))
    out = {"gc": gc, "gc_err": gc_err, "gc_over_err": gc / gc_err if gc_err > 0 else None,
           "gc_over_err_note": "library error normalisation; not a calibrated z-score",
           "kl_ratio": float(he.get_effective_sample_size()) / float(he.N),
           "n_configs": int(he.N), "nsym": nsym}
    try:
        paired = antithetic_pairs_ok(he)
        units = rng.permutation(he.N // 2 if paired else he.N)
        parts = []
        for grp in np.array_split(units, int(n_splits)):
            mask = np.zeros(he.N, dtype=bool)
            if paired:
                mask[2 * grp] = True
                mask[2 * grp + 1] = True
            else:
                mask[grp] = True
            parts.append((len(grp), sym_grad(he.split(mask), False)[0]))
        st = split_noise_statistic(g, parts)
        kl = min(1.0, max(CONV_MIN_KL, float(kl_relax_end))) if kl_relax_end else 1.0
        r_exp = 1.0 + float(he.N) / (kl * float(n_relax))
        rr = st["R"] / r_exp if st["R"] is not None else None
        out["split_null"] = {**st, "antithetic_pairs_verified": paired,
                             "R_expected_if_converged": r_exp,
                             "R_over_expected": rr, "threshold": FRESH_R_FACTOR_OK}
        out["consistent_with_minimum"] = (None if rr is None or (st["lin_rel_err"] or 0) > 1e-6
                                          else bool(rr <= FRESH_R_FACTOR_OK))
    except Exception as exc:
        out["split_null"] = {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()}
        out["consistent_with_minimum"] = None
    out["wall_s"] = time.time() - t
    return out


def run_converged_sscha(ctx: dict, start: str, sp: dict, out: dict) -> None:
    """One converged SSCHA relaxation from start A or B, the Hessian at its end, diagnostics.
    Fills ``out`` as it goes and flushes it to disk at every population end."""
    import warnings
    warnings.filterwarnings("ignore")                  # as compute_finite_t_sscha does
    import cellconstructor as CC
    import cellconstructor.Phonons  # noqa: F401
    import sscha
    import sscha.Ensemble
    import sscha.SchaMinimizer
    import sscha.Relax
    from mlip_dynstab.finite_t import _cc_dyn_from_phonopy
    from mlip_dynstab.harmonic import _relax

    R, calc = ctx["recipe"], ctx["calc"]
    T, sc, tol = float(ctx["T"]), tuple(int(x) for x in ctx["supercell"]), R["imag_tol_thz"]
    ry = float(CC.Units.RY_TO_CM * C_THZ_PER_CM)
    work = sp["work"]
    work.mkdir(parents=True, exist_ok=True)
    tm = out.setdefault("timings_s", {})
    out["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
    t_all = t = time.time()

    prim = ctx["atoms"].copy()
    prim.calc = calc
    if R["relax"]:
        prim = _relax(prim, fmax=R["fmax"])
    out["relaxed_cellpar"] = [float(x) for x in prim.cell.cellpar()]
    dyn_harm = _cc_dyn_from_phonopy(prim, calc, sc, R["disp"])
    tm["relax_and_harmonic_fc"] = time.time() - t
    w_harm, p_harm = dyn_freqs(dyn_harm)
    out["harmonic_pre_fpd"] = freq_summary(w_harm, ry, tol)
    dyn, out["start_construction"] = build_start(dyn_harm, start, R["start_b_thz"], ry)
    dyn_start = dyn.Copy()
    w_start, p_start = dyn_freqs(dyn_start)
    out["start"] = freq_summary(w_start, ry, tol)
    out["dyn_meta"] = {"nqirr": int(dyn.nqirr), "nq": len(dyn.q_tot),
                       "nat_prim": int(dyn.structure.N_atoms),
                       "nat_supercell": int(dyn.structure.N_atoms * len(dyn.q_tot))}
    ctx["flush"]()

    out["global_seed"] = conv_global_seed(R, start)
    np.random.seed(out["global_seed"])
    grad_log: list = []
    Ens = real_error_ensemble_class(sscha.Ensemble.Ensemble, grad_log, R["converge_min_kl_ratio"])
    ens = Ens(dyn.Copy(), T, supercell=dyn.GetSupercell())
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ens, root_representation=R["root_representation"])
    minim.use_julia = bool(R["use_julia"])           # False: the gradient goes through Ens
    minim.min_step_dyn = R["min_step_dyn"]
    minim.meaningful_factor = R["meaningful_factor"]
    minim.kong_liu_ratio = R["kong_liu_ratio"]
    # Backstop only: the controller replaces it at every population's first step.
    minim.max_ka = int(R["max_steps_per_pop"]) * int(R["max_pop"]) + 1
    relaxer = sscha.Relax.SSCHA(minim, ase_calculator=calc, N_configs=R["n_configs"],
                                max_pop=R["max_pop"], save_ensemble=False)
    rec, ctrl = wire_conv_hooks(relaxer, ry, grad_log, R, ctx, out)
    out["stopping_criteria"] = conv_stopping_criteria(minim, relaxer, R)
    t = time.time()
    with chdir(work):
        converged = bool(relaxer.relax(get_stress=False))
    tm["sscha_relax"] = time.time() - t
    final_dyn = relaxer.minim.dyn
    n_pops = int(relaxer.start_pop) - 1                     # Relax.py:412,419; starts at 1
    pop_end = population_end_dyns(work, n_pops, final_dyn.nqirr, dyn_start, ry, tol)
    out["relax"] = conv_relax_block(rec, ctrl, pop_end, converged, n_pops, R, grad_log)
    raw = {k: [_scalar(x) for x in getattr(relaxer.minim, f"__{k}__")]
           for k in ("fe", "fe_err", "gc", "gc_err", "gw", "gw_err", "KL")}
    raw["good_kasteps"] = [int(i) for i in getattr(relaxer.minim, "__good_kasteps__")]
    out["relax"]["minimizer_raw_histories"] = raw
    out.pop("relax_partial", None)
    w_aux, p_aux = dyn_freqs(final_dyn)
    out["final_aux"] = freq_summary(w_aux, ry, tol)
    final_dyn.save_qe(str(work / "dyn_final_aux_"))
    ctx["flush"]()

    t = time.time()
    with chdir(work):
        he = sscha.Ensemble.Ensemble(final_dyn, T, supercell=final_dyn.GetSupercell())
        he.generate(int(R["n_hessian"]))
        he.get_energy_forces(calc, compute_stress=False)
    tm["hessian_ensemble"] = time.time() - t
    try:
        # KL of the gradient that converged (>= converge_min_kl_ratio by the guard). An
        # unconverged run's last gradient may be a rejected low-KL trial point, which would
        # inflate R_expected, so it is judged as if KL = 1 (the strictest reference).
        kl_ref = out["relax"].get("kl_ratio_at_gradient_end") if converged else 1.0
        out["fresh_gradient_check"] = fresh_gradient_check(
            he, final_dyn, np.random.default_rng([CONV_SALT, out["global_seed"], ord(start), 1]),
            int(R["n_configs"]), kl_ref, int(R["fresh_splits"]))
    except Exception as exc:                  # a diagnostic; never costs the Hessian
        out["fresh_gradient_check"] = {"error": f"{type(exc).__name__}: {exc}",
                                       "traceback": traceback.format_exc()}
    he.save_bin(str(sp["ensemble"]), 1)
    out["ensemble_dir"] = rel(sp["ensemble"])
    t = time.time()
    hess = he.get_free_energy_hessian(include_v4=False)
    tm["hessian"] = time.time() - t
    w_hess, p_hess = dyn_freqs(hess)
    out["hessian"] = freq_summary(w_hess, ry, tol)
    out["hessian"]["dynamically_stable"] = bool(out["hessian"]["min_nonac_thz"] >= tol)
    out["echo"] = {
        "rel_change_final_aux_vs_start": dyn_rel_change(final_dyn, dyn_start),
        "rel_change_hessian_vs_final_aux": dyn_rel_change(hess, final_dyn),
        "rel_change_hessian_vs_start": dyn_rel_change(hess, dyn_start),
        "harmonic_along_hessian_softmode": freq_along_softmode(w_hess, p_hess, w_harm, p_harm, ry),
        "start_along_hessian_softmode": freq_along_softmode(w_hess, p_hess, w_start, p_start, ry),
        "final_aux_along_hessian_softmode": freq_along_softmode(w_hess, p_hess, w_aux, p_aux, ry),
    }
    files = {}
    for name, d in (("dyn_harmonic_", dyn_harm), (f"dyn_start{start}_", dyn_start),
                    ("dyn_hessian_", hess)):
        d.save_qe(str(work / name))
    for name in ("dyn_harmonic_", f"dyn_start{start}_", "dyn_final_aux_", "dyn_hessian_"):
        files[name] = rel(work / f"{name}1")
    out["dyn_files"] = files
    ctx["flush"]()

    t = time.time()
    out["bootstrap"] = hessian_uncertainty(
        he, ctx["n_boot"], ctx["n_splits"],
        np.random.default_rng([BOOT_SALT, CONV_SALT, out["global_seed"], ord(start)]), ry, tol,
        out["hessian"]["min_nonac_thz"])
    tm["bootstrap"] = time.time() - t
    tm["total"] = time.time() - t_all


class _DryGradEnsemble:
    """--dry-run base for RealErrorEnsemble: sscha.Ensemble.Ensemble's two gradient entry points
    with the library's return conventions (the parallel one returns the ones placeholder,
    Ensemble.py:2657; the serial one returns (grad, err), :2830-2831) and its fixed attribute set
    (:276-304), so a subclass that set an instance attribute fails here as it would there. The
    gradient is a synthetic 'distance to the minimum'."""

    def __init__(self, n, w_ry, dist, err_elem, target_ry):
        self.N = int(n)
        self.w_0 = np.asarray(w_ry, dtype=float).copy()
        self.current_w = self.w_0.copy()
        self.target_ry = np.asarray(target_ry, dtype=float).copy()
        self.start_ry = self.w_0.copy()
        self.kl = float(n)
        self.dist = float(dist)
        self.dist0 = float(dist)
        self.err_elem = float(err_elem)
        self.supercell = np.array([2, 2, 2])
        self.shape = (8, 15, 15)
        self.__total_attributes__ = list(self.__dict__)
        self.fixed_attributes = True

    def __setattr__(self, name, value):
        if "fixed_attributes" in self.__dict__ and name not in self.__total_attributes__:
            raise AttributeError(f"'{name}' is not a member of '{type(self).__name__}'")
        object.__setattr__(self, name, value)

    def get_effective_sample_size(self):
        return self.kl

    def get_preconditioned_gradient(self, subtract_sscha=True, return_error=False,
                                    use_ups_supercell=True, preconditioned=1, fast_grad=False,
                                    verbose=True, timer=None):
        g = np.full(self.shape, self.dist / math.sqrt(float(np.prod(self.shape))),
                    dtype=np.complex128)
        e = np.full(self.shape, self.err_elem, dtype=np.complex128)
        return (g, e) if return_error else g

    def get_preconditioned_gradient_parallel(self, *args, timer=None, **kwargs):
        g, _ = self.get_preconditioned_gradient(*args, timer=timer, **kwargs)
        return g, np.zeros_like(g) + 1


class _DryLine:
    def __init__(self):
        self.step = 0.5

    def is_new_direction(self):
        return True


class _DryMinim:
    """--dry-run stand-in for SSCHA_Minimizer: the attributes the hooks read and run()'s control
    flow (SchaMinimizer.py:1282-1398): pre hook; gradient from the ensemble, error scaled as :456,
    histories as :587-600; fe/KL appended (:1313-1332); check_stop (:1601-1661); max_ka (:1370);
    post hook; restore on an unconverged stop (:1391-1397)."""

    def __init__(self, ens, shrink=0.93, kl_decay=0.94, sleep_s=0.0):
        for k in ("fe", "fe_err", "gc", "gc_err", "gw", "gw_err", "KL", "good_kasteps"):
            setattr(self, f"__{k}__", [])
        self.ensemble = ens
        self.population = 0
        self.max_ka = -1
        self.kong_liu_ratio = 0.5
        self.meaningful_factor = 0.2
        self.abs_conv_thr = 1e-8
        self.gradi_op = "all"
        self.minim_struct = True
        self.N_symmetries = 48
        self.use_julia = False
        self.minimizer = _DryLine()
        self.shrink, self.kl_decay, self.sleep_s = float(shrink), float(kl_decay), float(sleep_s)
        self._conv = False

    def h(self, k):
        return getattr(self, f"__{k}__")

    def is_converged(self):
        return self._conv

    def init(self):
        if len(self.h("fe")) == len(self.h("gc")):            # SchaMinimizer.py:1150-1156
            self.h("fe").append(-1.0)
            self.h("fe_err").append(1e-5)
            self.h("KL").append(self.ensemble.kl)
        self._conv = False

    def _move(self, factor):
        ens = self.ensemble
        ens.dist *= factor
        frac = ens.dist / ens.dist0 if ens.dist0 > 0 else 0.0
        ens.current_w = ens.target_ry + frac * (ens.start_ry - ens.target_ry)

    def run(self, custom_function_pre=None, custom_function_post=None,
            custom_function_gradient=None):
        self._conv = False
        running = True
        while running:
            if custom_function_pre is not None:
                custom_function_pre(self)
            ens = self.ensemble
            g, e = ens.get_preconditioned_gradient_parallel(True, True, preconditioned=1)
            e = e / np.sqrt(self.N_symmetries * np.prod(ens.supercell))
            self.h("gc").append(float(np.sqrt(np.sum(np.abs(g) ** 2))))
            self.h("gc_err").append(float(np.sqrt(np.sum(np.abs(e) ** 2))))
            self.h("gw").append(0.0)
            self.h("gw_err").append(1e-5)
            self._move(self.shrink)
            ens.kl *= self.kl_decay
            self.h("fe").append(-1.0 - 1e-3 * (1.0 - ens.dist))
            self.h("fe_err").append(1e-5)
            self.h("KL").append(ens.kl)
            running = not self._check_stop()
            if len(self.h("fe")) > self.max_ka and self.max_ka > 0:
                running = False
            if custom_function_post is not None:
                custom_function_post(self)
            if self.sleep_s:
                time.sleep(self.sleep_s)
        if not self._conv:
            self._move(1.0 / self.shrink)

    def _check_stop(self):
        gc, gce = self.h("gc")[-1], self.h("gc_err")[-1]
        gw, gwe = self.h("gw")[-1], self.h("gw_err")[-1]
        gc_ok = gc < gce * self.meaningful_factor or gc < self.abs_conv_thr
        gw_ok = gw < gwe * self.meaningful_factor or gw < self.abs_conv_thr
        if gc_ok and gw_ok:
            self._conv = True
            return True
        ens = self.ensemble
        return bool(ens.kl / float(ens.N) < self.kong_liu_ratio
                    and self.minimizer.is_new_direction())


class _DryRelax:
    """--dry-run stand-in for sscha.Relax.SSCHA: setup_custom_functions and relax()'s population
    loop (Relax.py:284-295, 364-421): new ensemble at the current dyn, population set, init, run,
    stop on convergence or when pop > max_pop (read afresh every population)."""

    def __init__(self, minim, N_configs, max_pop):
        self.minim = minim
        self.N_configs = int(N_configs)
        self.max_pop = int(max_pop)
        self.start_pop = 1
        self._pre = self._post = None

    def setup_custom_functions(self, custom_function_pre=None, custom_function_gradient=None,
                               custom_function_post=None):
        self._pre, self._post = custom_function_pre, custom_function_post

    def relax(self, get_stress=False):
        pop, running = self.start_pop, True
        while running:
            ens = self.minim.ensemble
            ens.w_0 = ens.current_w.copy()
            ens.kl = float(ens.N)
            self.minim.population = pop
            self.minim.init()
            self.minim.run(custom_function_pre=self._pre, custom_function_post=self._post)
            running = not self.minim.is_converged()
            pop += 1
            if pop > self.max_pop:
                running = False
        self.start_pop = pop
        return self.minim.is_converged()


def run_converged_mock(ctx: dict, start: str, sp: dict, out: dict) -> None:
    """--dry-run stand-in for run_converged_sscha: same schema, no sscha or cellconstructor.
    The relax (unless the calculator is the LJ fallback) is real; the SSCHA is _DryRelax /
    _DryMinim on a synthetic gradient, but the hooks, RealErrorEnsemble and its guard, the
    per-population and wall caps, the population table and every JSON flush are the real code."""
    from mlip_dynstab.harmonic import _relax
    R, calc, tol = ctx["recipe"], ctx["calc"], ctx["recipe"]["imag_tol_thz"]
    ry = RY_TO_CM_CC162 * C_THZ_PER_CM
    rng = np.random.default_rng([MOCK_SALT, CONV_SALT, ord(start)])
    tm = out.setdefault("timings_s", {})
    t_all = time.time()
    out["MOCK"] = ("dry-run: every SSCHA number here is synthetic; the hooks, the real-error "
                   "subclass and its KL guard, the caps and the JSON flushes are the real code")
    prim = ctx["atoms"].copy()
    prim.calc = calc
    if R["relax"] and not ctx.get("calc_is_fallback"):
        prim = _relax(prim, fmax=R["fmax"])
    out["relaxed_cellpar"] = [float(x) for x in prim.cell.cellpar()]
    base = ctx["ledger"].get("min_eff_freq_thz") or 1.0
    harm = -abs(base) - 1.0
    start_thz = abs(harm) if start == "A" else float(R["start_b_thz"])
    target = abs(base) * 0.5 + 0.2

    def spectrum(v):
        return np.array([0.0, 0.0, 0.0] + [v] * 3 + [v + 1.0] * 9) / ry

    out["harmonic_pre_fpd"] = freq_summary(spectrum(harm), ry, tol)
    out["start_construction"] = {"start": start, "MOCK": True,
                                 "n_replaced": 0 if start == "A" else 3}
    out["start"] = freq_summary(spectrum(start_thz), ry, tol)
    out["dyn_meta"] = {"nqirr": 4, "nq": 8, "nat_prim": 5, "nat_supercell": 40}
    ctx["flush"]()

    grad_log: list = []
    Ens = real_error_ensemble_class(_DryGradEnsemble, grad_log, R["converge_min_kl_ratio"])
    ens = Ens(R["n_configs"], spectrum(start_thz), 1.0 if start == "A" else 0.3, 0.01,
              spectrum(target))
    out["global_seed"] = conv_global_seed(R, start)
    minim = _DryMinim(ens, sleep_s=float(ctx.get("mock_step_sleep") or 0.0))
    minim.use_julia = bool(R["use_julia"])
    minim.meaningful_factor = R["meaningful_factor"]
    minim.kong_liu_ratio = R["kong_liu_ratio"]
    minim.max_ka = int(R["max_steps_per_pop"]) * int(R["max_pop"]) + 1
    relaxer = _DryRelax(minim, R["n_configs"], R["max_pop"])
    rec, ctrl = wire_conv_hooks(relaxer, ry, grad_log, R, ctx, out)
    out["stopping_criteria"] = conv_stopping_criteria(minim, relaxer, R)
    t = time.time()
    converged = bool(relaxer.relax(get_stress=False))
    tm["sscha_relax"] = time.time() - t
    n_pops = int(relaxer.start_pop) - 1
    pop_end = {p: {"end_aux_min_nonac_thz": None, "rel_change_vs_prev_end": None,
                   "end_dyn_file": None} for p in range(1, n_pops + 1)}
    out["relax"] = conv_relax_block(rec, ctrl, pop_end, converged, n_pops, R, grad_log)
    out["relax"]["minimizer_raw_histories"] = {k: list(minim.h(k)) for k in
                                               ("fe", "fe_err", "gc", "gc_err", "KL")}
    out.pop("relax_partial", None)
    out["final_aux"] = freq_summary(ens.current_w, ry, tol)
    hmin = float(base) + (0.004 if start == "B" else 0.0)
    out["hessian"] = {**freq_summary(spectrum(hmin), ry, tol),
                      "dynamically_stable": bool(hmin >= tol)}
    r_exp = 1.0 + int(R["n_hessian"]) / float(R["n_configs"])
    out["fresh_gradient_check"] = {"MOCK": True, "gc_over_err": 1.1, "kl_ratio": 1.0,
                                   "split_null": {"R": 1.2 * r_exp, "R_expected_if_converged": r_exp,
                                                  "R_over_expected": 1.2,
                                                  "threshold": FRESH_R_FACTOR_OK},
                                   "consistent_with_minimum": True}
    out["echo"] = {"MOCK": True}
    out["bootstrap"] = {"method": "mock", "B": int(ctx["n_boot"]),
                        "full_ensemble_min_nonac_thz": hmin,
                        **boot_stats(rng.normal(hmin, 0.01, int(ctx["n_boot"])), tol)}
    tm["total"] = time.time() - t_all


def _start_digest(doc) -> dict:
    r = (doc or {}).get("run") or {}

    def g(*k):
        return _dig(r, *k)

    # hessian_uncertainty's disjoint-split fallback reports the spread of N/K-size ensembles as
    # std_thz; the full ensemble's sd is std_full_estimate_thz there.
    boot_sd = (g("bootstrap", "std_full_estimate_thz")
               if g("bootstrap", "method") == "disjoint_splits" else g("bootstrap", "std_thz"))
    return {"status": r.get("status"), "converged": g("relax", "converged"),
            "stop_reason": g("relax", "stop_reason"), "n_populations": g("relax", "n_populations"),
            "harmonic_min_thz": g("harmonic_pre_fpd", "min_nonac_thz"),
            "start_min_thz": g("start", "min_nonac_thz"),
            "final_aux_min_thz": g("final_aux", "min_nonac_thz"),
            "final_aux_lowest6_thz": g("final_aux", "lowest6_thz"),
            "hessian_min_thz": g("hessian", "min_nonac_thz"),
            "hessian_lowest6_thz": g("hessian", "lowest6_thz"),
            "boot_sd_thz": boot_sd, "boot_method": g("bootstrap", "method"),
            "fresh_gc_over_err": g("fresh_gradient_check", "gc_over_err"),
            "fresh_R_over_expected": g("fresh_gradient_check", "split_null", "R_over_expected"),
            "fresh_consistent": g("fresh_gradient_check", "consistent_with_minimum"),
            "global_seed": r.get("global_seed"),
            "kl_ratio_at_gradient_end": g("relax", "kl_ratio_at_gradient_end"),
            "model_version": g("provenance", "model_version"),
            "julia_ext_available": r.get("julia_ext_available"), "wall_s": r.get("wall_s"),
            "nqirr": g("dyn_meta", "nqirr"), "work_dir": r.get("work_dir")}


def conv_compare(base: Path, tag: str, imag_tol: float) -> dict:
    """Start A against start B for one unit, written to <tag>_AB.json (a derived file, rebuilt
    whenever a start finishes or --summarize runs). Start-independent: |Hessian min A - B| within
    AB_Z_TOL combined bootstrap sd. The bootstrap covers the Hessian ensemble only, not the noise
    of the relaxed dyn, so a borderline z is read with that in mind."""
    docs = {s: load_json(conv_paths(base, tag, s)["json"]) for s in CONV_STARTS}
    dA, dB = _start_digest(docs["A"]), _start_digest(docs["B"])
    out = {"schema": CONV_SCHEMA + "/compare", "unit_tag": tag, "utc": utc_now(),
           "A": dA, "B": dB, "both_ok": dA["status"] == "ok" and dB["status"] == "ok"}
    if not out["both_ok"]:
        out["verdict"] = "incomplete"
    else:
        hA, hB = dA["hessian_min_thz"], dB["hessian_min_thz"]
        comb = math.sqrt((dA["boot_sd_thz"] or 0.0) ** 2 + (dB["boot_sd_thz"] or 0.0) ** 2)
        d = abs(hA - hB)
        # Absolute floor: on bcc the Hessian minimum does not depend on the configurations at all
        # (bootstrap sd ~1e-16 THz), so the z test alone calls a 0.002 THz A/B difference
        # "start-dependent". Differences below AB_ABS_TOL_THZ are agreement at any sd.
        indep = (bool(d <= max(AB_Z_TOL * comb, AB_ABS_TOL_THZ)) if comb > 0
                 else (bool(d <= AB_ABS_TOL_THZ) or None))
        conv = bool(dA["converged"] and dB["converged"])
        out.update(both_converged=conv, hessian_min_diff_thz=d, combined_boot_sd_thz=comb,
                   z=(d / comb if comb > 0 else None), z_tol=AB_Z_TOL, abs_tol_thz=AB_ABS_TOL_THZ,
                   z_note="combined_boot_sd covers the two Hessian ensembles only (independent "
                          "seeds), not the sampling noise of the two relaxed dyns, so z "
                          "overstates start dependence; read dyn_level and final_aux_min_diff "
                          "with it",
                   both_fresh_consistent=bool(dA["fresh_consistent"] and dB["fresh_consistent"]),
                   hessian_start_independent=indep,
                   same_stability_call=bool((hA >= imag_tol) == (hB >= imag_tol)),
                   final_aux_min_diff_thz=abs(dA["final_aux_min_thz"] - dB["final_aux_min_thz"]),
                   same_env=bool(dA["model_version"] == dB["model_version"]
                                 and dA["julia_ext_available"] == dB["julia_ext_available"]))
        if not conv:
            v = "not converged (A: {}, B: {})".format(dA["stop_reason"], dB["stop_reason"])
        elif indep is None:
            v = "converged; start independence untested (no bootstrap sd)"
        elif indep and out["same_stability_call"]:
            v = "converged and start-independent"
        else:
            v = "converged but START-DEPENDENT"
        if conv and not out["both_fresh_consistent"]:
            v += "; fresh-ensemble gradient NOT shown consistent with a minimum for " + ",".join(
                s for s, x in (("A", dA), ("B", dB)) if not x["fresh_consistent"])
        out["verdict"] = v
        try:
            import cellconstructor.Phonons as CCP
            lv = {}
            for name in ("dyn_final_aux_", "dyn_hessian_"):
                a, b = (CCP.Phonons(str(from_rel(x["work_dir"]) / name), int(x["nqirr"]))
                        for x in (dA, dB))
                lv[f"rel_diff_{name.strip('_')}"] = dyn_rel_change(a, b)
            out["dyn_level"] = lv
        except Exception as exc:
            out["dyn_level"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    atomic_write_json(conv_paths(base, tag, "A")["compare"], out)
    return out


def run_converged_unit(unit: dict, args) -> dict:
    """Starts A and/or B of one unit, one JSON per (unit, start). An 'ok' start is skipped, and
    so is a 'failed' one unless --retry-failed; a 'running' one is recomputed if its process is
    dead (dead box, timeout) and skipped as busy if another live process holds its lock."""
    from mlip_dynstab.systems import get_spec, build_atoms
    system, model, T = unit["system"], unit["model"], float(unit["T"])
    sc = tuple(int(x) for x in unit["supercell"])
    tag = unit_tag(system, model, T, sc)
    base = conv_base_dir(args)
    recipe = converged_recipe(args)
    spec = get_spec(system)
    ident = {"system": system, "model": model, "T": T, "supercell": list(sc)}
    flags = {"failed": [], "done": [], "busy": []}

    todo = []
    for s in unit["starts"]:
        ex = load_json(conv_paths(base, tag, s)["json"])
        if ex is not None:
            where = conv_paths(base, tag, s)["json"]
            if bool(ex.get("dry_run")) != bool(args.dry_run):
                raise SystemExit(f"{where} was written with dry_run={ex.get('dry_run')}; "
                                 "refusing to mix mocked and real runs. Move it aside.")
            if ex.get("unit") != ident or ex.get("start") != s:
                raise SystemExit(f"{where} holds {ex.get('unit')} start {ex.get('start')}.")
            if ex.get("recipe") != _clean(recipe):
                raise SystemExit(f"{where} was computed with recipe {ex.get('recipe')}; this run "
                                 f"is {recipe}. Refusing to mix; move it aside.")
            st = (ex.get("run") or {}).get("status")
            if st == "ok" or (st == "failed" and not args.retry_failed):
                print(f"[skip] {tag} start {s}: {st}"
                      + ("" if st == "ok" else " (--retry-failed to rerun)"))
                continue
        todo.append(s)
    if not todo:
        conv_compare(base, tag, recipe["imag_tol_thz"])
        return flags
    if not args.dry_run:
        ok, why = env_can_run(model)
        if not ok:
            raise SystemExit(f"{tag}: {why}")
    atoms = build_atoms(spec)
    calc, model_version, fallback = load_calculator(model, args.device, args.dry_run, atoms)
    led = ledger_lookup(system, model, T, sc)
    prov = provenance(model, args.dry_run, args.device)
    if not args.dry_run:
        import sscha.Ensemble
        prov["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
        prov["julia_error"] = str(getattr(sscha.Ensemble, "__JULIA_ERROR__", ""))[:500]
        prov["schamodules"] = schamodules_linkage()
    prov["model_version"] = model_version

    for s in todo:
        sp = conv_paths(base, tag, s)
        with start_lock(base / "locks" / f"{tag}_start{s}.lock") as held:
            if not held:
                print(f"[busy] {tag} start {s}: another live process is running it; skipped",
                      flush=True)
                flags["busy"].append((tag, s))
                continue
            # Re-read under the lock: another process may have finished it since the scan above.
            st = ((load_json(sp["json"]) or {}).get("run") or {}).get("status")
            if st == "ok" or (st == "failed" and not args.retry_failed):
                print(f"[skip] {tag} start {s}: {st} (finished by another process)")
                continue
            _run_converged_start(unit, args, s, sp, ctx_base=dict(
                system=system, model=model, T=T, supercell=sc, atoms=atoms, calc=calc,
                recipe=recipe, ledger=led, calc_is_fallback=fallback, ident=ident, tag=tag,
                prov=prov, model_version=model_version), flags=flags)
    cmp_ = conv_compare(base, tag, recipe["imag_tol_thz"])
    print(f"[{tag}] A vs B: {cmp_.get('verdict')}"
          + (f" (dHess {cmp_['hessian_min_diff_thz']:.3f} THz, z {cmp_.get('z')})"
             if cmp_.get("both_ok") else ""), flush=True)
    return flags


@contextlib.contextmanager
def start_lock(path: Path):
    """Advisory lock on one (unit, start), so two processes (say --preset converged and
    --start B, launched by hand) cannot both run it: the second would rename the first's work
    directory from under it and overwrite its JSON. flock is released by the kernel when its
    holder dies, so a killed run leaves no stale lock and a relaunch resumes normally. Yields
    True if held, False if a live process holds it. Without fcntl (Windows dry runs) no lock is
    taken; the box is Linux."""
    try:
        import fcntl
    except ImportError:
        yield True
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "utc": utc_now()}))
        fh.flush()
        yield True
    finally:
        fh.close()


def _run_converged_start(unit: dict, args, s: str, sp: dict, ctx_base: dict, flags: dict) -> None:
    """One (unit, start), called with its lock held: JSON set-up, the run, the final record."""
    tag, ident, prov = ctx_base["tag"], ctx_base["ident"], ctx_base["prov"]
    recipe, led, model_version = ctx_base["recipe"], ctx_base["ledger"], ctx_base["model_version"]
    doc = load_json(sp["json"]) or {
        "schema": CONV_SCHEMA, "study": "C1c converged SSCHA, start A vs B (R1.4)",
        "dry_run": bool(args.dry_run), "unit": ident, "unit_tag": tag, "start": s,
        "item": unit.get("item"), "why": unit.get("why"), "recipe": recipe,
        "recipe_source": "production_recipe() + converged_recipe() (" + SCRIPT + ")",
        "provenance_created": prov, "previous_runs": []}
    doc.update(ledger=led, provenance_last_run=prov,
               calculator_fallback=bool(ctx_base["calc_is_fallback"]))
    stale = None
    if sp["work"].exists():                    # a dead or failed attempt: keep it, aside
        stale = sp["work"].with_name(f"{sp['work'].name}.stale-{int(time.time())}")
        sp["work"].rename(stale)
    if doc.get("run"):
        # The whole previous run, including the relax_partial of a killed or timed-out one: its
        # step history exists nowhere else. Its work dir now lives under the stale name.
        old = dict(doc["run"])
        old["work_dir_moved_to"] = rel(stale) if stale is not None else None
        doc.setdefault("previous_runs", []).append(old)
    entry = {"start": s, "status": "running", "utc_start": utc_now(),
             "host": socket.gethostname(), "pid": os.getpid(), "work_dir": rel(sp["work"]),
             "settings": {"n_boot": int(args.n_boot), "n_splits_fallback": int(args.n_splits),
                          "unit_timeout_s": float(args.unit_timeout), "device": args.device}}
    doc["run"] = entry

    def flush(doc=doc, entry=entry, path=sp["json"]):
        snap = dict(doc)
        snap["run"] = {k: v for k, v in entry.items() if not str(k).startswith("_")}
        snap["last_written_utc"] = utc_now()
        atomic_write_json(path, snap)

    flush()
    ctx = {k: ctx_base[k] for k in ("system", "model", "T", "supercell", "atoms", "calc",
                                    "recipe", "ledger", "calc_is_fallback")}
    ctx.update(n_boot=int(args.n_boot), n_splits=int(args.n_splits),
               unit_timeout=float(args.unit_timeout), flush=flush,
               mock_step_sleep=float(args.dry_run_step_sleep))
    print(f"===== {tag} start {s} {utc_now()} =====", flush=True)
    t0 = time.time()
    try:
        (run_converged_mock if args.dry_run else run_converged_sscha)(ctx, s, sp, entry)
        entry["status"] = "ok"
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        rec, ctrl = entry.get("_recorder"), entry.get("_controller")
        if rec is not None:
            entry["relax_partial"] = {"reason": "exception", "steps": rec.steps,
                                      "hook_errors": rec.errors,
                                      "pop_start_aux_min_nonac_thz": rec.pop_start,
                                      "controller": ctrl.state() if ctrl else None}
        flags["failed"].append((tag, s))
        print(f"[FAIL] {tag} start {s}: {entry['error']}", flush=True)
    entry.pop("_recorder", None)
    entry.pop("_controller", None)
    entry["wall_s"] = time.time() - t0
    entry["utc_end"] = utc_now()
    entry["provenance"] = {k: prov.get(k) for k in ("git_head", "host", "packages",
                                                     "model_version", "device",
                                                     "omp_num_threads")}
    if entry["status"] == "ok":
        lc = ledger_comparison(led, entry["hessian"]["min_nonac_thz"], float(args.ledger_tol),
                               model_version)
        lc["note"] = ("informational: the converged recipe is not the production recipe, so "
                      "no reproduction of the ledger is expected")
        entry["ledger_reference"] = lc
    flush()
    if entry["status"] == "ok":
        flags["done"].append((tag, s))
        rx, b = entry["relax"], entry.get("bootstrap") or {}
        fr = _dig(entry, "fresh_gradient_check", "split_null", "R_over_expected")
        print(f"[{tag} {s}] harm {entry['harmonic_pre_fpd']['min_nonac_thz']:+.3f} | "
              f"start {entry['start']['min_nonac_thz']:+.3f} | "
              f"aux {entry['final_aux']['min_nonac_thz']:+.3f} | "
              f"hess {entry['hessian']['min_nonac_thz']:+.3f} THz (boot sd {b.get('std_thz')}) "
              f"| npop {rx['n_populations']} conv {rx['converged']} stop {rx['stop_reason']} "
              f"| fresh R/Rexp {fr} | {entry['wall_s']:.0f}s", flush=True)


def summarize_converged(directory: Path) -> int:
    files = sorted(Path(directory).glob("*_start[AB].json"))
    if not files:
        print(f"no converged-mode JSONs in {directory}")
        return 1

    def f(x, w=7, p=3):
        return f"{x:+{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"

    # fresh = R / R_expected of the fresh-ensemble gradient (fresh_gradient_check); ~1 when converged
    hdr = (f"{'unit':34s} {'st':>2s} {'status':>7s} {'harm':>7s} {'start':>7s} {'aux':>7s} "
           f"{'hess':>7s} {'bootsd':>6s} {'npop':>4s} {'stop':>9s} {'KLend':>5s} {'fresh':>5s} "
           f"{'wall':>6s}")
    print(hdr)
    print("-" * len(hdr))
    tags = {}
    for fp in files:
        doc = load_json(fp)
        if not isinstance(doc, dict) or doc.get("schema") != CONV_SCHEMA:
            continue
        tags[doc["unit_tag"]] = (doc.get("recipe") or {}).get("imag_tol_thz", -0.1)
        d = _start_digest(doc)

        def num(x, fmt, w):
            return format(x, fmt) if isinstance(x, (int, float)) else "-".rjust(w)

        name = doc["unit_tag"] + (" [DRY]" if doc.get("dry_run") else "")
        npop = "-" if d["n_populations"] is None else str(d["n_populations"])
        print(f"{name:34s} {doc['start']:>2s} {str(d['status']):>7s} "
              f"{f(d['harmonic_min_thz'])} {f(d['start_min_thz'])} "
              f"{f(d['final_aux_min_thz'])} {f(d['hessian_min_thz'])} "
              f"{num(d['boot_sd_thz'], '6.3f', 6)} {npop:>4s} {str(d['stop_reason'] or '-'):>9s} "
              f"{num(d['kl_ratio_at_gradient_end'], '5.2f', 5)} "
              f"{num(d['fresh_R_over_expected'], '5.2f', 5)} {num(d['wall_s'], '6.0f', 6)}")
    for tag, itol in sorted(tags.items()):
        c = conv_compare(Path(directory), tag, itol)
        extra = (f": dHess {c['hessian_min_diff_thz']:.3f} THz, z {c['z']}, dAux "
                 f"{c['final_aux_min_diff_thz']:.3f} THz, dyn {c.get('dyn_level')}"
                 if c.get("both_ok") else "")
        print(f"  {tag}: {c['verdict']}{extra}")
    return 0


def main_converged(args) -> int:
    """--preset converged, or --converge for one unit. Never touches the seed-study paths."""
    if args.out:
        raise SystemExit("--out names one seed-study JSON; converged mode uses --out-dir")
    base = conv_base_dir(args)
    if args.summarize:
        return summarize_converged(base)
    starts = (args.start,) if args.start else CONV_STARTS
    if args.preset == CONV_PRESET:
        units = [dict(u, starts=starts) for u in CONVERGED_UNITS
                 if (not args.model or u["model"] == args.model)
                 and (not args.system or u["system"] == args.system)]
    else:
        if not (args.system and args.model and args.T is not None):
            raise SystemExit("--converge needs --system, --model and --T (or --preset converged)")
        units = [dict(system=args.system, model=args.model, T=args.T,
                      supercell=tuple(args.supercell), starts=starts, item="manual",
                      why="manual invocation")]
    if args.list:
        print(f"converged mode: {len(units)} units x starts {','.join(starts)} -> {rel(base)}")
        for u in units:
            sc = " ".join(map(str, u["supercell"]))
            print(f"  [{u['item']}] {unit_tag(u['system'], u['model'], u['T'], u['supercell'])}"
                  f"  env={MODEL_ENV[u['model']]}  -- {u['why']}")
            for s in u["starts"]:
                print(f"      python {SCRIPT} --converge --system {u['system']} --model "
                      f"{u['model']} --T {t_tag(u['T'])} --supercell {sc} --start {s}")
        return 0
    runnable = []
    for u in units:
        # A single unit reaches run_converged_unit, which refuses loudly in the wrong env.
        ok, why = ((True, "") if args.dry_run or args.preset != CONV_PRESET
                   else env_can_run(u["model"]))
        if ok:
            runnable.append(u)
        else:
            print(f"[env] skipping {unit_tag(u['system'], u['model'], u['T'], u['supercell'])}: "
                  f"{why}")
    failed, busy = [], []
    for u in runnable:
        tag = unit_tag(u["system"], u["model"], u["T"], u["supercell"])
        try:
            fl = run_converged_unit(u, args)
            failed += fl["failed"]
            busy += fl["busy"]
        except (Exception, SystemExit) as exc:
            if args.preset != CONV_PRESET:
                raise
            print(f"[UNIT FAILED] {tag}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                  flush=True)
            failed.append((tag, "unit"))
    if busy:
        print(f"\n{len(busy)} start(s) left to the live process that holds them: {busy}.")
    if failed:
        print(f"\n{len(failed)} start(s)/unit(s) failed: {failed}. Tracebacks are in the JSONs "
              "(or above, for a whole unit).")
    return 2 if failed else 0


# --------------------------------------------------------------------------- CLI ----

def env_can_run(model: str) -> tuple[bool, str]:
    import importlib.util as iu
    mod = MODEL_IMPORT.get(model)
    if mod is None or iu.find_spec(mod) is None:
        return False, f"backend '{mod}' not importable"
    for m in ("sscha", "cellconstructor"):
        if iu.find_spec(m) is None:
            return False, f"'{m}' not importable"
    return True, ""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="SSCHA seed study with persisted convergence diagnostics (C1/C1b/C5).")
    ap.add_argument("--system")
    ap.add_argument("--model")
    ap.add_argument("--T", type=float, dest="T")
    ap.add_argument("--seeds", default="0,1,2,3", help="comma-separated (default 0,1,2,3)")
    ap.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=30,
                    help="bootstrap replicates of the Hessian ensemble (default 30)")
    ap.add_argument("--n-splits", type=int, default=4,
                    help="disjoint splits if a bootstrap replicate cannot be built (default 4)")
    ap.add_argument("--save-configs", type=int, default=16,
                    help="Hessian-ensemble configurations saved per seed for C3b (default 16)")
    ap.add_argument("--include-v4", action="store_true",
                    help="after the seeds, include_v4=True on seed 0's ensemble in a child "
                         "process (C1b). In --preset mode only units marked v4 get it.")
    ap.add_argument("--v4-timeout", type=float, default=10800.0,
                    help="hard cap on the include_v4 child, seconds (default 10800)")
    ap.add_argument("--v4-threads", type=int, default=None,
                    help="OpenMP threads for the v4 child (default: the cgroup CPU quota)")
    ap.add_argument("--out", default=None, help="unit JSON path (single-unit mode)")
    ap.add_argument("--out-dir", default=None,
                    help=f"study directory (default {rel(STUDY_DIR)}; --dry-run: "
                         f"{rel(STUDY_DIR / '_dryrun')})")
    ap.add_argument("--preset", choices=sorted([*PRESETS, CONV_PRESET]),
                    help=f"'{CONV_PRESET}' runs the C1c converged-SSCHA units (starts A and B)")
    ap.add_argument("--list", action="store_true", help="with --preset: print the plan and exit")
    ap.add_argument("--summarize", action="store_true",
                    help="print a table of every unit JSON in the study directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="no sscha/cellconstructor import; SSCHA numbers mocked, relax and MLIP "
                         "calls real (a Lennard-Jones stand-in if the model is not installed)")
    ap.add_argument("--dry-run-v4-sleep", type=float, default=3.0, help=argparse.SUPPRESS)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--ledger-tol", type=float, default=LEDGER_TOL_THZ)
    cg = ap.add_argument_group("converged mode (C1c)")
    cg.add_argument("--converge", action="store_true",
                    help="converged SSCHA for one unit (--system/--model/--T/--supercell), "
                         f"written under {rel(CONV_DIR)}")
    cg.add_argument("--start", choices=list(CONV_STARTS), default=None,
                    help="A = production ForcePositiveDefinite start, B = imaginary modes set to "
                         "--start-b-thz (default: both)")
    cg.add_argument("--unit-timeout", type=float, default=CONV_DEFAULTS["unit_timeout_s"],
                    help="wall-clock cap on each (unit, start) relaxation, seconds; <= 0 disables "
                         f"(default {CONV_DEFAULTS['unit_timeout_s']:g})")
    cg.add_argument("--start-b-thz", type=float, default=CONV_DEFAULTS["start_b_thz"])
    cg.add_argument("--conv-n-configs", type=int, default=CONV_DEFAULTS["n_configs"])
    cg.add_argument("--conv-max-pop", type=int, default=CONV_DEFAULTS["max_pop"])
    cg.add_argument("--conv-n-hessian", type=int, default=CONV_DEFAULTS["n_hessian"])
    cg.add_argument("--conv-steps-per-pop", type=int, default=CONV_DEFAULTS["max_steps_per_pop"])
    cg.add_argument("--dry-run-step-sleep", type=float, default=0.0, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.converge or args.preset == CONV_PRESET:
        return main_converged(args)
    if args.summarize:
        d = (Path(args.out).resolve().parent if args.out else
             Path(args.out_dir).resolve() if args.out_dir else
             STUDY_DIR / "_dryrun" if args.dry_run else STUDY_DIR)
        return summarize(d)

    if args.preset:
        if args.out:
            raise SystemExit("--out names one unit's JSON; use --out-dir with --preset")
        units = [u for u in PRESETS[args.preset]
                 if (not args.model or u["model"] == args.model)
                 and (not args.system or u["system"] == args.system)]
        if args.list:
            print(f"preset '{args.preset}': {len(units)} units")
            for u in units:
                sc = " ".join(map(str, u["supercell"]))
                print(f"  [{u['item']}] {unit_tag(u['system'], u['model'], u['T'], u['supercell'])}"
                      f"  seeds {','.join(map(str, u['seeds']))}{'  +v4' if u.get('v4') else ''}"
                      f"  env={MODEL_ENV[u['model']]}  -- {u['why']}")
                print(f"      python {SCRIPT} --system {u['system']} --model {u['model']} "
                      f"--T {t_tag(u['T'])} --supercell {sc} "
                      f"--seeds {','.join(map(str, u['seeds']))}"
                      f"{' --include-v4' if u.get('v4') else ''}")
            return 0
        runnable = []
        for u in units:
            ok, why = (True, "") if args.dry_run else env_can_run(u["model"])
            if ok:
                runnable.append(u)
            else:
                print(f"[env] skipping {unit_tag(u['system'], u['model'], u['T'], u['supercell'])}: {why}")
    else:
        if not (args.system and args.model and args.T is not None):
            raise SystemExit("need --system, --model and --T (or --preset / --summarize)")
        runnable = [dict(system=args.system, model=args.model, T=args.T,
                         supercell=tuple(args.supercell),
                         seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
                         v4=args.include_v4, item="manual", why="manual invocation")]

    failed, unit_bad, refused, v4_jobs = [], [], [], []
    for u in runnable:
        tag = unit_tag(u["system"], u["model"], u["T"], u["supercell"])
        try:
            flags = run_unit(u, args)
        except (Exception, SystemExit) as exc:
            # Unattended box: one unit's refusal (mixed dry-run, recipe or identity) or crash
            # must not cost the preset's other units. A single-unit run still fails loudly.
            if not args.preset:
                raise
            print(f"[UNIT FAILED] {tag}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                  flush=True)
            failed.append((tag, "unit"))
            continue
        failed += flags["failed"]
        unit_bad += flags["unit_check_failed"]
        refused += flags["refused"]
        if args.include_v4 and u.get("v4"):
            v4_jobs.append(flags["json"])
    # v4 last: a multi-hour CPU job must not hold the GPU queue behind it
    for js in v4_jobs:
        maybe_run_v4(js, args)

    if failed:
        print(f"\n{len(failed)} seed(s)/unit(s) failed: {failed}. Tracebacks are in the unit "
              "JSONs (or above, for a whole unit).")
    if refused:
        print(f"\n{len(refused)} seed(s) refused to mix envs: {refused}.")
    if unit_bad:
        print(f"\nextxyz unit check FAILED for {unit_bad}: do not use those configs for C3b.")
        return 3
    return 2 if (failed or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
