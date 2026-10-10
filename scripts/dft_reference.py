"""PBE reference energies and forces on the exact structures the MLIPs were asked about.

Referee 1 asks two things the paper cannot answer from MLIP data alone.

* R1.1 (plan item C3a): is the double well the soft-mode screen integrates real, or an MLIP
  artefact? Answered by PBE single points along the screen's own soft-mode coordinates for
  SrTiO3, BaTiO3, bcc-Zr and fluorite ZrO2, with all five MLIPs re-evaluated on the identical
  geometries, and the single-mode screen re-solved on the PBE-fitted potential.
* R1.2 (plan item C3b): when the SSCHA false-stabilises, or runs away at high T, is the method
  failing or is the MLIP failing on thermally sampled configurations it never trained on?
  Answered by MLIP-vs-PBE force and energy errors on configurations drawn from the C1 ensembles,
  set against the same errors on a near-equilibrium (rattled) baseline.

The script never re-derives geometry in a second code path. It regenerates the screen's paths
with the production functions in ``mlip_dynstab.finite_t`` and checks each E(Q) against the
cached map that the ledger was computed from. Only then are the structures handed to pw.x. A
cached map stores q, band, dim, m_eff and dE but no eigenvector, so the check is the only
evidence that the regenerated coordinate is the one the screen used. A failed check is recorded
and the structures are still written; they then carry the flag.

Two things the check taught, both handled here rather than papered over:

* Inside a degenerate eigenspace (the Gamma T1u triplet, the R25 tilt triplet, X doublets) the
  eigenvector LAPACK returns is arbitrary, so a regenerated pattern can differ from the
  production one while being equally "correct" (MACE BaTiO3 Gamma: 16.8 vs 20.9 meV deep,
  m_eff 41 vs 53 amu). When a map fails, the direction reproducing the cached E(Q) is searched
  for in the degenerate span (``_recover_pattern``); it is adopted when it beats LAPACK's on
  (pass, fit-window pass, max diff), and the check is re-run on the production sampler. A
  shifted harmonic frequency does not stop the search: CHGNet BaTiO3 Gamma regenerates
  0.24 THz off the cache yet its cached map is found in the span to 0.6 meV, so the potential
  along the coordinate is unchanged at the sampled amplitudes. Those production patterns are
  generic directions with no point symmetry left, which is also why they are the costly pw.x
  jobs.
* A frozen mode repeats in a cell smaller than the modulated supercell (the 40-atom R-tilt
  cell in 10 atoms). pw.x runs each C3a path in whichever cell is cheaper by the cost model;
  the energy is scaled by the image count and each supercell atom takes its image's force, so
  the analysis still sees the identical supercell structures the MLIPs were evaluated on.

Stages, in order. Each is resumable and skips outputs that already exist.

  plan        any env, no MLIP. C3a paths, atoms and k-meshes from the caches and ledger alone.
  geom        per model env. Relax, FCs at the production FC supercell, regenerate every cached
              mode's pattern and E(Q), check against the cache, write the displaced structures.
              Run it for MatterSim FIRST. A harmonically stable model (MACE/CHGNet/ORB/SevenNet
              bcc-Zr, ORB SrTiO3) gets a reference path along MatterSim's deciding pattern at
              its own relaxed lattice, and only the MatterSim env can make that pattern.
  c3b-inputs  any env, after C1. Pick SSCHA configs, add the undisplaced supercell and a rattled
              baseline, write structures and pw.x inputs.
  mlip-eval   per model env. This model's energy and forces on every structure any model wrote.
  qe-inputs   any env. pw.x inputs for the C3a paths.
  (box)       scripts/box/qe_queue.sh runs pw.x over <out>/qe/*/pw.in.
  analyze     local. PBE vs MLIP curves, PBE-backed screen calls, C3b errors, measured cost.

Box order (python per model env as in scripts/run_revision_compute.sh):

    $PY_mattersim scripts/dft_reference.py geom --model mattersim            # FIRST
    for m in mace_mp0 chgnet sevennet0 orb_v2; do $PY_m scripts/dft_reference.py geom --model $m; done
    python scripts/dft_reference.py c3b-inputs --sssp-dir /root/sssp          # after C1
    for m in (all five); do $PY_m scripts/dft_reference.py mlip-eval --model $m; done
    python scripts/dft_reference.py qe-inputs --sssp-dir /root/sssp
    bash scripts/box/qe_queue.sh --qe-prefix ... --pseudo-dir /root/sssp --filter '^a_' -r 8 -k 8
    bash scripts/box/qe_queue.sh ... --filter '^b_' -r 16 -k 4               # then '^ax_'
    python scripts/dft_reference.py analyze                                  # local, after pull

geom exits 3 when a stable model's reference path waited on the MatterSim pattern (re-run it)
and 4 when a map failed its check (the summary says which, and whether it was a deciding mode).
k-point convention: see KSPACING_CONVENTION (|b| includes 2*pi). Cost figures printed by plan,
qe-inputs and c3b-inputs rest on an ASSUMED per-k-point cost (--c0-core-s); analyze measures it.

Outputs go under results/revision/dft/ (``--out-root`` moves them; the smoke test uses it).
Nothing here writes to the ledger.

Three follow-up checks on the persistent PBE screen errors (convergence, functional, PBE
lattice / eigenvector) are the stages ``conv-inputs``, ``xc-inputs``, ``pbe-lattice`` and
``analyze-checks``, implemented in scripts/dft_checks.py (its docstring has the acceptance
criterion, the approximations and the box commands). They write results/revision/dft_checks/
and only read results/revision/dft/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np                                           # noqa: E402

from mlip_dynstab import DEFAULT_IMAG_TOL_THZ, SOFTMODE_EQMAP_VERSION   # noqa: E402
from mlip_dynstab.systems import build_atoms, get_spec      # noqa: E402

SCRIPT = "scripts/dft_reference.py"
OUT_DEFAULT = REPO / "results" / "revision" / "dft"
SSCHA_DIR_DEFAULT = REPO / "results" / "revision" / "sscha_seeds"
CACHE_DIR = REPO / "results" / "cache"

SYSTEMS = ("srtio3_cubic", "batio3_cubic", "zr_bcc", "zro2_cubic", "knbo3_cubic", "cssnbr3_cubic")
MODELS = ("mace_mp0", "chgnet", "orb_v2", "sevennet0", "mattersim")
# The model whose deciding pattern defines the reference coordinate for models that have no
# instability of their own. MatterSim has a harmonic instability in every
# target system.
REF_MODEL = "mattersim"

# Production screen settings: compute_finite_t_softmode defaults and cli.run_unit (max_modes=24).
Q_MAX, N_PTS, FMAX, DISP, MAX_MODES = 0.45, 10, 1e-3, 0.01, 24
T_SCREEN = (50.0, 100.0, 300.0, 600.0, 900.0)

METALLIC_KLASS = {"bcc-metal"}
METALLIC_EXTRA = {"cu_fcc"}

# C1 units (tasks/todo.md) for the plan's C3b job count only; c3b-inputs reads what C1 wrote.
C1_UNITS = (("batio3_cubic", "mace_mp0", 100.0), ("zro2_cubic", "mace_mp0", 100.0),
            ("zr_bcc", "mattersim", 50.0), ("srtio3_cubic", "mace_mp0", 600.0))
SSCHA_SUPERCELL = (2, 2, 2)

KSPACING_CONVENTION = ("n_i = max(1, ceil(|b_i| / spacing)) with b = 2*pi*inv(cell).T, i.e. |b_i| "
                       "includes 2*pi (the VASP KSPACING convention); Gamma-centred, unshifted")

MODEL_ENV = {"mace_mp0": "mace", "chgnet": "chgnet", "orb_v2": "orb", "sevennet0": "sevennet",
             "mattersim": "mattersim"}
_PKGS = ("numpy", "scipy", "ase", "phonopy", "spglib", "pandas", "torch", "mace-torch", "chgnet",
         "orb-models", "sevenn", "mattersim", "python-sscha", "cellconstructor")


# ------------------------------------------------------------------ provenance ----

def _git_head():
    try:
        h = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        if not h:
            return None
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return None


def _versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in _PKGS:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _lock_check(model, versions):
    """Compare the running env with envs/lock-<env>-*.txt (the latest dated one).

    Only the packages this study depends on are compared; a mismatch is recorded, not fatal,
    because the local smoke test runs outside the pinned envs by design."""
    env = MODEL_ENV.get(model)
    locks = sorted((REPO / "envs").glob(f"lock-{env}-*.txt")) if env else []
    if not locks:
        return {"lock": None}
    pinned = {}
    for line in locks[-1].read_text(encoding="utf-8").splitlines():
        if "==" in line:
            n, v = line.split("==", 1)
            pinned[n.strip().lower()] = v.strip()
    mism = {p: {"installed": v, "lock": pinned[p.lower()]} for p, v in versions.items()
            if v and p.lower() in pinned and pinned[p.lower()] != v}
    return {"lock": locks[-1].relative_to(REPO).as_posix(), "mismatches": mism}


def provenance(args, stage, model=None, handle=None):
    vers = _versions()
    prov = {"script": SCRIPT, "stage": stage, "git_head": _git_head(), "versions": vers,
            "python": sys.version.split()[0], "platform": platform.platform(),
            "host": socket.gethostname(),
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "settings": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items() if k != "func"},
            "screen_settings": {"q_max_A": Q_MAX, "n_pts": N_PTS, "relax_fmax": FMAX,
                                "fc_disp_A": DISP, "max_modes": MAX_MODES,
                                "imag_tol_thz": DEFAULT_IMAG_TOL_THZ,
                                "eqmap_version": SOFTMODE_EQMAP_VERSION}}
    if model:
        prov["model"] = model
        prov["env_lock"] = _lock_check(model, vers)
    if handle is not None:
        prov["model_version"] = handle.version
        prov["device"] = handle.device
    return prov


# ---------------------------------------------------------------------- helpers ----

def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _write_frames(path, frames):
    from ase.io import write
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.extxyz")
    write(tmp, frames, format="extxyz")
    os.replace(tmp, path)


def _read_frames(path):
    from ase.io import read
    return read(path, index=":", format="extxyz")


def _frac(x):
    f = Fraction(float(x)).limit_denominator(12)
    if f == 0:
        return "0"
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}d{f.denominator}"


def qtag(q, band):
    """Filename-safe q label: fractions with 'd' for '/', components joined by '-'."""
    return "-".join(_frac(x) for x in q) + f"-b{int(band)}"


def path_stem(system, model, q, band, ref_model=None):
    s = f"{system}_{model}_q{qtag(q, band)}"
    return s + (f"-ref{ref_model}" if ref_model else "")


def _tstr(T):
    return f"{float(T):g}"


def is_metal(system):
    return get_spec(system).klass in METALLIC_KLASS or system in METALLIC_EXTRA


def fc_supercell_for(system):
    """The FC supercell the production screen used (cli.run_unit): 6x6x6 for bcc, else 2x2x2."""
    return (6, 6, 6) if get_spec(system).klass == "bcc-metal" else (2, 2, 2)


def cache_file(system, model):
    sc = fc_supercell_for(system)
    return CACHE_DIR / (f"softmode_v{SOFTMODE_EQMAP_VERSION}m{MAX_MODES}_{system}_{model}"
                        f"_sc{''.join(map(str, sc))}.json")


def load_cache(system, model):
    p = cache_file(system, model)
    if not p.exists():
        raise SystemExit(f"missing cached map {p}")
    c = _load(p)
    if [int(x) for x in c["fc_supercell"]] != list(fc_supercell_for(system)):
        raise SystemExit(f"{p.name}: fc_supercell {c['fc_supercell']} is not the production "
                         f"{fc_supercell_for(system)}")
    return p, c


_LEDGER = None


def _ledger_rows(system, model):
    """Canonical softmode rows, lowest T first. Read through analysis.canonical(), never raw."""
    global _LEDGER
    if _LEDGER is None:
        from mlip_dynstab.analysis import load_canonical
        _LEDGER = load_canonical()
    df = _LEDGER
    d = df[(df.method == "softmode") & (df.system == system) & (df.model == model)]
    return d.sort_values("temperature_K")


def _cache_decision(cache, T):
    """(mode index, stable) exactly as compute_finite_t_softmode decides from a cached map."""
    from mlip_dynstab.finite_t import _solve_scha, _sym_curvature_freq
    solved = []
    for i, e in enumerate(cache["modes"]):
        if e["band"] < 0:
            solved.append({"i": i, "curv": float(e["harm_thz"]), "Q0": 0.0, "stable": True})
            continue
        eff, Q0, st = _solve_scha(e["a"], e["b"], e["c"], e["m_eff"], T)
        curv = _sym_curvature_freq(e["a"], e["b"], e["c"], e["m_eff"], T)
        solved.append({"i": i, "curv": eff if curv is None else curv, "Q0": Q0,
                       "stable": bool(st)})
    cond = [s for s in solved if not s["stable"]]
    d = max(cond, key=lambda s: s["Q0"]) if cond else min(solved, key=lambda s: s["curv"])
    return d["i"], not cond


def _match_mode(cache, q_str, harm):
    """Cache index of the ledger's deciding mode: same q (to the ledger's 4 decimals) and the
    nearest harmonic frequency, which separates degenerate-looking bands at one q."""
    try:
        q = [float(x) for x in str(q_str).split(",")]
    except ValueError:
        return None
    best = None
    for i, e in enumerate(cache["modes"]):
        if max(abs(a - b) for a, b in zip(e["q"], q)) < 1e-3:
            d = abs(float(e["harm_thz"]) - float(harm))
            if best is None or d < best[1]:
                best = (i, d)
    return best[0] if best and best[1] < 1e-3 else None


_DECIDING = {}


def deciding(system, model, cache):
    """Which cached modes decide the screen's call, and at which ladder T.

    Re-derived from the cache with the production solve and cross-checked against the ledger's
    ft_decide_* columns. The ledger wins a disagreement (it is what the paper reports), and the
    disagreement is recorded rather than resolved silently. Memoised: one solve is ~1 s and
    MatterSim's bcc-Zr map has 24 modes, re-read for every model that borrows its pattern."""
    if (system, model) not in _DECIDING:
        _DECIDING[(system, model)] = _deciding(system, model, cache)
    return _DECIDING[(system, model)]


def _deciding(system, model, cache):
    rows = _ledger_rows(system, model)
    by_T = {}
    for _, r in rows.iterrows():
        T = float(r["temperature_K"])
        i_cache, st_cache = _cache_decision(cache, T)
        i_led = _match_mode(cache, r.get("ft_decide_q"), r.get("ft_decide_harm_thz"))
        by_T[_tstr(T)] = {"T": T, "idx": i_led if i_led is not None else i_cache,
                          "idx_cache_solve": i_cache, "idx_ledger": i_led,
                          "pred_stable_ledger": bool(r["pred_stable"]),
                          "stable_cache_solve": bool(st_cache),
                          "agree": bool(i_led == i_cache and bool(r["pred_stable"]) == st_cache)}
    if not by_T:
        raise SystemExit(f"no canonical softmode rows for {system}/{model}")
    lowest = min(by_T.values(), key=lambda d: d["T"])
    return {"by_T": by_T, "lowest_T": lowest["T"], "idx_lowest": lowest["idx"],
            "idx_all": sorted({d["idx"] for d in by_T.values()}),
            "all_agree": all(d["agree"] for d in by_T.values())}


def _n_prim(system):
    return len(build_atoms(get_spec(system)))


def _kmesh(cell, spacing):
    rec = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, float)).T
    return [max(1, int(math.ceil(np.linalg.norm(b) / spacing - 1e-9))) for b in rec]


def _nk_tr_upper(mesh):
    """Irreducible k count with time reversal only: the upper bound for a structure with no
    point symmetry. Points that are their own inverse are the 2^d TRIM-like points of the mesh."""
    n = int(np.prod(mesh))
    self_inv = int(np.prod([2 if m % 2 == 0 else 1 for m in mesh]))
    return (n + self_inv) // 2


def _nk_irr(atoms, mesh):
    """Irreducible k count under the structure's own symmetry (spglib, symprec 1e-5 A, close to
    pw.x's default tolerance); the cost figure that matters, since a single frozen mode keeps
    part of the point group. None if spglib is unavailable."""
    try:
        import spglib
        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        mapping, _ = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0],
                                                   is_time_reversal=True, symprec=1e-5)
        return int(len(np.unique(mapping)))
    except Exception:
        return None


def cost_core_h(nat, nk, c0):
    """Modelled pw.x cost: c0 core-s per k-point per SCF for a 5-atom cell, x (nat/5)^2.2
    (bands and FFT grid both grow with nat; orthogonalisation starts to bite by 40 atoms)."""
    return c0 * nk * (nat / 5.0) ** 2.2 / 3600.0


def _omega_thz(a, m_eff):
    """Harmonic frequency of V = a Q^2 for mass m_eff, signed like the screen's frequencies."""
    from mlip_dynstab.finite_t import _W_TO_OMEGA2
    om2 = 2.0 * a * _W_TO_OMEGA2 / m_eff
    return float(np.sign(om2) * np.sqrt(abs(om2)) / (2 * np.pi) / 1e12)


def _fit_window(dE):
    """The well+barrier window _fit_double_well fits (mirrors it; used for like-for-like RMSE)."""
    dE = np.asarray(dE, float)
    emin = float(dE.min())
    return dE <= emin + max(0.06, 5.0 * abs(min(emin, 0.0)))


# ----------------------------------------------------------- path descriptors ----

def _same_mode(t, d):
    """Symmetry copy: same model, cell size, harmonic frequency, effective mass and depth."""
    return (t["system"] == d["system"] and t["path_model"] == d["path_model"]
            and t["pattern_model"] == d["pattern_model"] and t["n_atoms"] == d["n_atoms"]
            and abs(t["harm_thz"] - d["harm_thz"]) < 0.01
            and abs(t["m_eff"] - d["m_eff"]) <= 0.01 * max(t["m_eff"], 1e-9)
            and abs(t["depth_meV"] - d["depth_meV"]) <= max(0.1, 0.01 * t["depth_meV"]))


def select_paths(descs, cap):
    """Which geometry paths get PBE. Deciding and reference paths always. Any other screened
    mode only if its modulated cell has <= cap atoms and it is not a symmetry copy of a path
    already taken for that model (q and -q, or the degenerate partners of one branch)."""
    order = {"decide": 0, "ref": 1, "mode": 2}
    taken, skipped = [], []
    for d in sorted(descs, key=lambda d: (d["system"], d["path_model"], order[d["role"]],
                                          d["n_atoms"], d["stem"])):
        if d["role"] == "mode":
            if d["n_atoms"] > cap:
                skipped.append({"stem": d["stem"], "why": f"{d['n_atoms']} atoms > cap {cap}"})
                continue
            dup = next((t for t in taken if _same_mode(t, d)), None)
            if dup is not None:
                skipped.append({"stem": d["stem"], "why": f"symmetry copy of {dup['stem']}"})
                continue
        taken.append(d)
    return taken, skipped


def _descs_from_caches(systems, models):
    """Path descriptors from the caches and ledger alone (the plan stage; no MLIP needed)."""
    descs = []
    for system in systems:
        n_prim = _n_prim(system)
        for model in models:
            _, cache = load_cache(system, model)
            dec = deciding(system, model, cache)
            if all(e["band"] < 0 for e in cache["modes"]):
                _, rc = load_cache(system, REF_MODEL)
                rdec = deciding(system, REF_MODEL, rc)
                e = rc["modes"][rdec["idx_lowest"]]
                if e["band"] < 0:
                    continue
                descs.append(dict(system=system, path_model=model, pattern_model=REF_MODEL,
                                  role="ref", stem=path_stem(system, model, e["q"], e["band"],
                                                             REF_MODEL),
                                  q=e["q"], band=e["band"], dim=e["dim"],
                                  n_atoms=int(np.prod(e["dim"])) * n_prim,
                                  harm_thz=float(e["harm_thz"]), m_eff=float(e["m_eff"]),
                                  depth_meV=float(e["well_depth_meV"])))
                continue
            for i, e in enumerate(cache["modes"]):
                descs.append(dict(system=system, path_model=model, pattern_model=model,
                                  role="decide" if i in dec["idx_all"] else "mode",
                                  stem=path_stem(system, model, e["q"], e["band"]),
                                  q=e["q"], band=e["band"], dim=e["dim"],
                                  n_atoms=int(np.prod(e["dim"])) * n_prim,
                                  harm_thz=float(e["harm_thz"]), m_eff=float(e["m_eff"]),
                                  depth_meV=float(e["well_depth_meV"])))
    return descs


# ------------------------------------------------------------------- plan stage ----

def _generic_path_frames(system, q, dim, amp=0.1, seed=0):
    """(Q=0 cell, one displaced cell) for a GENERIC frozen mode at q: random complex
    polarisations per basis atom with the Bloch phase of q. It has the translational period
    of any real mode at q but no point symmetry, which is what the screen's patterns in a
    degenerate space turn out to have; so its pw.x cost is the realistic upper case."""
    from ase.build import make_supercell
    prim = build_atoms(get_spec(system))
    sc = make_supercell(prim, np.diag([int(x) for x in dim]))
    f = sc.get_scaled_positions() @ np.diag([float(x) for x in dim])   # prim fractional
    R = np.floor(f + 1e-6)
    fp = prim.get_scaled_positions()
    basis = [int(np.argmin(np.linalg.norm((fi - Ri) - fp - np.round((fi - Ri) - fp), axis=1)))
             for fi, Ri in zip(f, R)]
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(len(prim), 3)) + 1j * rng.normal(size=(len(prim), 3))
    u = np.real(e[basis] * np.exp(2j * np.pi * (R @ np.asarray(q, float)))[:, None])
    disp = sc.copy()
    disp.set_positions(sc.get_positions() + amp * u / np.abs(u).max())
    return sc, disp


def stage_plan(args):
    """C3a/C3b job counts and a cost figure from the caches alone, before anything is run.

    The k-mesh uses the registry lattice constant, not the relaxed one (at most one k-point per
    direction different). The core-hour figure is a model, not a measurement: per-k-point SCF
    cost c0 for a 5-atom cell, scaled as (n_atoms/5)^2.2 and multiplied by the time-reversal-only
    irreducible k count, which is the upper bound for a structure with no point symmetry.
    ``analyze`` replaces c0 with the value measured on the box."""
    descs = _descs_from_caches(args.systems, args.models)
    taken, skipped = select_paths(descs, args.extra_max_atoms)

    def cost(nat, mesh):
        return cost_core_h(nat, _nk_tr_upper(mesh), args.c0_core_s)

    rows = []
    for d in taken:
        spec = get_spec(d["system"])
        prim = build_atoms(spec)
        cell = np.diag(d["dim"]) @ prim.cell.array
        sp = args.kspacing_metal if is_metal(d["system"]) else args.kspacing_insulator
        mesh = _kmesh(cell, sp)
        # Realistic figure: a generic mode at this q in the cell qe-inputs would pick (9
        # displaced frames), plus the undistorted frame in the same cell (full symmetry).
        sc0, sc1 = _generic_path_frames(d["system"], d["q"], d["dim"])
        lat, reps, _ = _pick_cell(sc1, d["system"], args)
        f0, f1, mesh_run = sc0, sc1, mesh
        if lat is not None:
            f0 = _reduce_frame(sc0, lat, reps)[0]
            f1 = _reduce_frame(sc1, lat, reps)[0]
            mesh_run = _kmesh(lat, sp)
        nk1 = _nk_irr(f1, mesh_run) or _nk_tr_upper(mesh_run)
        nk0 = _nk_irr(f0, mesh_run) or _nk_tr_upper(mesh_run)
        c_gen = ((N_PTS - 1) * cost_core_h(len(f1), nk1, args.c0_core_s)
                 + cost_core_h(len(f0), nk0, args.c0_core_s))
        rows.append(dict(d, kmesh=mesh, nk_tr_upper=_nk_tr_upper(mesh), n_jobs=N_PTS,
                         core_h_upper=N_PTS * cost(d["n_atoms"], mesh),
                         run_atoms=len(f1), run_kmesh=mesh_run, nk_irr_generic=nk1,
                         core_h_generic=c_gen))
    c3b = []
    for system, model, T in C1_UNITS:
        prim = build_atoms(get_spec(system))
        cell = np.diag(SSCHA_SUPERCELL) @ prim.cell.array
        nat = len(prim) * int(np.prod(SSCHA_SUPERCELL))
        sp = args.kspacing_metal if is_metal(system) else args.kspacing_insulator
        mesh = _kmesh(cell, sp)
        n_jobs = args.per_unit + args.n_baseline      # the undisplaced ref is symmetric, cheap
        c3b.append(dict(system=system, model=model, T=T, n_atoms=nat, kmesh=mesh,
                        nk_tr_upper=_nk_tr_upper(mesh), n_jobs=n_jobs + 1,
                        core_h_upper=n_jobs * cost(nat, mesh)))

    print(f"C3a: {len(taken)} paths x {N_PTS} Q points = {len(taken) * N_PTS} pw.x jobs "
          f"(extra-mode cap {args.extra_max_atoms} atoms; {len(skipped)} screened modes left out)")
    print(f"{'system':13s} {'model':10s} {'role':6s} {'q':18s} {'nat':>4s} {'kmesh':>10s} "
          f"{'nk<=':>5s} {'core-h<=':>9s} {'run nat':>7s} {'run k':>10s} {'nk':>5s} "
          f"{'core-h':>7s}")
    for r in rows:
        q = ",".join(_frac(x) for x in r["q"])
        print(f"{r['system']:13s} {r['path_model']:10s} {r['role']:6s} {q:18s} {r['n_atoms']:4d} "
              f"{'x'.join(map(str, r['kmesh'])):>10s} {r['nk_tr_upper']:5d} "
              f"{r['core_h_upper']:9.1f} {r['run_atoms']:7d} "
              f"{'x'.join(map(str, r['run_kmesh'])):>10s} {r['nk_irr_generic']:5d} "
              f"{r['core_h_generic']:7.1f}")
    print(f"C3b: {sum(r['n_jobs'] for r in c3b)} pw.x jobs "
          f"({args.per_unit} configs + {args.n_baseline} baseline + 1 undisplaced per unit)")
    for r in c3b:
        print(f"  {r['system']:13s} {r['model']:10s} T={r['T']:g} nat={r['n_atoms']} "
              f"k={'x'.join(map(str, r['kmesh']))} nk<={r['nk_tr_upper']} "
              f"core-h<={r['core_h_upper']:.0f}")
    a_h = sum(r["core_h_upper"] for r in rows)
    a_g = sum(r["core_h_generic"] for r in rows)
    a_d = sum(r["core_h_generic"] for r in rows if r["role"] != "mode")
    b_h = sum(r["core_h_upper"] for r in c3b)
    print(f"core-h, ASSUMED c0 = {args.c0_core_s} core-s per k-point per SCF for a 5-atom cell:")
    print(f"  C3a as run (smallest cell, generic mode): {a_g:.0f} "
          f"(deciding + reference paths {a_d:.0f}, extra modes {a_g - a_d:.0f}); "
          f"full supercells, no symmetry: {a_h:.0f}")
    print(f"  C3b (no symmetry, 40-atom perovskite cells): {b_h:.0f}")
    _dump(Path(args.out_root) / "plan.json",
          {"provenance": provenance(args, "plan"), "c3a": rows, "c3a_skipped": skipped,
           "c3b": c3b, "core_h_upper": {"c3a": a_h, "c3b": b_h},
           "core_h_c3a_as_run": {"all": a_g, "deciding_and_reference": a_d},
           "cost_model": {"c0_core_s_per_k_5atoms": args.c0_core_s, "exponent": 2.2,
                          "k_count": "time-reversal-only upper bound",
                          "kspacing_convention": KSPACING_CONVENTION}})


# ------------------------------------------------------------------- geom stage ----

def _freqs_at(ph, q):
    ph.run_qpoints([list(q)], with_eigenvectors=False)
    return np.array(ph.qpoints.frequencies)[0]


def _energy(atoms, calc):
    from ase import Atoms
    b = Atoms(symbols=atoms.get_chemical_symbols(), positions=atoms.get_positions(),
              cell=atoms.get_cell(), pbc=True)
    b.calc = calc
    return float(b.get_potential_energy())


def _compare_map(Qs, dE, e):
    """Regenerated E(Q) vs the cached map: pass iff max |diff| <= max(1 meV, 1% of the depth).
    The diff restricted to the fit window is reported too, because the steep repulsive wall
    at large Q amplifies any pattern difference far more than the well does."""
    Qc, dEc = np.asarray(e["Qs"], float), np.asarray(e["dE"], float)
    if len(Qc) != len(Qs) or not np.allclose(Qc, Qs, atol=1e-9):
        return {"pass": False, "reason": "Q grid differs from the cache"}
    diff = np.abs(np.asarray(dE) - dEc) * 1000.0
    tol = max(1.0, 0.01 * float(e["well_depth_meV"]))
    win = _fit_window(dEc)
    return {"pass": bool(diff.max() <= tol), "max_abs_diff_meV": float(diff.max()),
            "tol_meV": tol, "max_abs_diff_window_meV": float(diff[win].max()),
            "pass_window": bool(diff[win].max() <= tol),
            "diff_meV": diff.tolist(), "depth_regen_meV": float(-min(np.min(dE), 0.0) * 1000),
            "depth_cache_meV": float(e["well_depth_meV"])}


def _check_rank(c):
    """Order two map checks: full pass, then fit-window pass, then the smaller max diff."""
    return (bool(c.get("pass")), bool(c.get("pass_window")),
            -float(c.get("max_abs_diff_meV", np.inf)))


def _dE_along(base, calc, u, Qs, E0):
    from ase import Atoms
    pos0 = base.get_positions()
    out = []
    for Q in Qs:
        a = Atoms(symbols=base.get_chemical_symbols(), positions=pos0 + Q * u,
                  cell=base.get_cell(), pbc=True)
        a.calc = calc
        out.append(a.get_potential_energy() - E0)
    return np.array(out)


def _degenerate_basis(ph, q, band, dim, tol_thz=0.05):
    """Orthonormal real basis of every pattern the screen's modulation could have produced at
    (q, band): the real span of Re and Im of the modulations of all bands degenerate with it.
    Production took Re(modulation of the eigenvector LAPACK returned); inside a degenerate
    eigenspace, or for a complex eigenvector's global phase, that choice is arbitrary and can
    differ between machines and library builds, so it lies somewhere in this span.

    "Degenerate" is taken loosely (0.05 THz): a float32 model's relaxation leaves the cubic
    cell very slightly broken, which splits a T1u triplet by ~0.01 THz (CHGNet BaTiO3: -6.135,
    -6.134, -6.122) and makes the split directions machine-dependent too."""
    freqs = _freqs_at(ph, q)
    bands = [k for k in range(len(freqs)) if abs(freqs[k] - freqs[int(band)]) < tol_thz]
    raw = []
    for k in bands:
        ph.run_modulations(dimension=[int(x) for x in dim],
                           phonon_modes=[[list(q), int(k), 1.0, 0.0]])
        mods, _ = ph.get_modulations_and_supercell()
        M = np.asarray(mods[0])
        raw += [M.real.ravel(), M.imag.ravel()]
    basis = []
    for v in raw:
        w = v - sum((v @ b) * b for b in basis) if basis else v.copy()
        if np.linalg.norm(w) > 1e-6 * max(np.linalg.norm(v), 1e-300):
            basis.append(w / np.linalg.norm(w))
    return bands, np.array(basis)


def _sphere(angles, n):
    x = np.ones(n)
    for i, a in enumerate(angles):
        x[i] *= np.cos(a)
        x[i + 1:] *= np.sin(a)
    return x


def _angles(x):
    x = np.asarray(x, float)
    n = len(x)
    a = np.zeros(n - 1)
    for i in range(n - 2):
        r = np.linalg.norm(x[i:])
        a[i] = np.arccos(np.clip(x[i] / r, -1.0, 1.0)) if r > 0 else 0.0
    a[n - 2] = np.arctan2(x[n - 1], x[n - 2])
    return a


def _recover_pattern(base, calc, ph, e, dim, E0, tol_meV, seed=0):
    """Find the direction in the degenerate span whose E(Q) reproduces the cached map.

    Three filters, cheapest first, because each energy is an MLIP single point. (1) The
    effective mass: m_eff = sum m|u|^2 under the max-component normalisation varies over the
    span and costs nothing, so a dense sweep of the span's unit sphere is cut to the directions
    within 3% of the cached m_eff. (2) Two energies (the cached minimum and the next Q) rank
    those. (3) A least-squares polish on all Q points from the best starts, stopping at the
    first that passes. Any direction related to the production one by a point operation of the
    undistorted cell has the same E(Q) under every symmetric Hamiltonian, PBE included, so
    matching all ten energies and m_eff pins the coordinate to the production direction's
    symmetry orbit, which is all C3a needs. Returns (u or None, info).

    The WHOLE sphere is swept, not the half with u ~ -u: the screen samples Q >= 0 only, and
    at a q with 3q = G (bcc (1/3,2/3,0), the MatterSim Zr mode) a cubic invariant makes E(Q)
    along u and -u different, so folding the sphere can discard the production direction.
    The m_eff shell is probed at its closest members AND at an even subsample of the rest,
    because the closest 40 can bunch on one arc of the shell."""
    from scipy.optimize import least_squares
    bands, B = _degenerate_basis(ph, e["q"], e["band"], dim)
    n = len(B)
    info = {"bands": bands, "subspace_dim": int(n)}
    if n < 2:
        info["why"] = "non-degenerate real mode: only +/-u exists, nothing to search"
        return None, info
    Qs, dEc = np.asarray(e["Qs"], float), np.asarray(e["dE"], float)
    rest = np.arange(1, len(Qs))
    i_min = int(np.argmin(dEc)) if dEc.min() < 0 else 1
    probe = np.array(sorted({max(i_min, 1), min(max(i_min, 1) + 1, len(Qs) - 1)}))
    masses = base.get_masses()[:, None]
    nat = len(base)
    n_eval = [0]

    def pattern(x):
        v = (x @ B).reshape(nat, 3)
        return v / np.abs(v).max()

    def resid(x, sel):
        n_eval[0] += len(sel)
        return _dE_along(base, calc, pattern(x), Qs[sel], E0) - dEc[sel]

    if n == 2:
        t = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
        X = np.stack([np.cos(t), np.sin(t)], axis=1)
    else:
        X = np.random.default_rng(seed).normal(size=(8000 * (n - 1), n))
        X /= np.linalg.norm(X, axis=1)[:, None]
    m_eff = np.array([float(np.sum(masses * pattern(x) ** 2)) for x in X])
    rel = np.abs(m_eff - e["m_eff"]) / e["m_eff"]
    keep = np.where(rel < 0.03)[0]
    if len(keep) < 8:
        keep = np.argsort(rel)[:8]
    if len(keep) > 48:
        by_rel = keep[np.argsort(rel[keep])]
        tail = np.sort(by_rel[24:])
        # X is in random (n > 2) or angular (n == 2) order, so a stride over the remainder in
        # that order spreads the second half of the probes along the whole shell
        tail = tail[np.linspace(0, len(tail) - 1, 24).round().astype(int)]
        keep = np.concatenate([by_rel[:24], tail])
    score = np.array([np.sqrt(np.mean(resid(X[i], probe) ** 2)) for i in keep])
    best = None
    for i in keep[np.argsort(score)][:4]:
        r = least_squares(lambda a: resid(_sphere(a, n), rest), _angles(X[i]), diff_step=1e-4,
                          xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=40)
        err = float(np.abs(r.fun).max()) * 1000
        if best is None or err < best[0]:
            best = (err, r.x)
        if err <= tol_meV:
            break
    err, a = best
    u = pattern(_sphere(a, n))
    info.update({"coeffs": _sphere(a, n).tolist(), "max_abs_resid_meV": err,
                 "n_candidates_meff": int(len(keep)), "n_single_points": int(n_eval[0]),
                 "m_eff_found": float(np.sum(masses * u ** 2))})
    return u, info


def _path_frames(base, u, Qs, E, meta):
    from ase import Atoms
    pos0 = base.get_positions()
    frames = []
    for i, (Q, En) in enumerate(zip(Qs, E)):
        a = Atoms(symbols=base.get_chemical_symbols(), positions=pos0 + Q * u,
                  cell=base.get_cell(), pbc=True)
        a.info.update(meta)
        a.info.update({"Q": float(Q), "index": i, "mlip_energy_eV": float(En)})
        a.arrays["mode_u"] = np.asarray(u, float)
        frames.append(a)
    return frames


def _pattern_record(stem, system, model, pattern_model, role, q, band, dim, base, u, m_eff,
                    prim, E0, Qs, dE, harm, e_cache, check, extra, prov):
    return {"stem": stem, "system": system, "path_model": model, "pattern_model": pattern_model,
            "role": role, "q": [float(x) for x in q], "band": int(band),
            "dim": [int(x) for x in dim], "n_atoms": len(base), "m_eff": float(m_eff),
            "harm_thz": float(harm), "depth_meV": float(-min(np.min(dE), 0.0) * 1000),
            "u": np.asarray(u).tolist(), "base_symbols": base.get_chemical_symbols(),
            "base_positions": base.get_positions().tolist(),
            "base_cell": base.cell.array.tolist(), "prim_cell": prim.cell.array.tolist(),
            "E0_eV": float(E0), "Qs": [float(x) for x in Qs], "dE_eV": np.asarray(dE).tolist(),
            "cache": ({k: e_cache[k] for k in ("a", "b", "c", "harm_thz", "m_eff",
                                               "well_depth_meV", "dE")} if e_cache else None),
            "check": check, **extra, "geom_file": f"geom/{stem}.extxyz", "provenance": prov}


def stage_geom(args):
    from ase import Atoms
    from mlip_dynstab.calculators import get_calculator
    from mlip_dynstab.finite_t import (_harmonic_phonon, _imaginary_commensurate_modes,
                                       _mode_pattern, _sample_well, _softest_mesh_mode)
    from mlip_dynstab.harmonic import _relax

    out = Path(args.out_root)
    handle = get_calculator(args.model, device=args.device)
    calc = handle.calc
    prov = provenance(args, "geom", args.model, handle)
    status = 0
    tally = []                                    # (stem, role, n_atoms, outcome)
    for system in args.systems:
        model = args.model
        cpath, cache = load_cache(system, model)
        dec = deciding(system, model, cache)
        stable_model = all(e["band"] < 0 for e in cache["modes"])
        check_path = out / "geom" / "checks" / f"{system}_{model}.json"
        record = _load(check_path) if check_path.exists() else {"paths": {}}

        # Plan this (system, model)'s paths before paying for a relaxation.
        todo = []
        if stable_model:
            _, rcache = load_cache(system, REF_MODEL)
            rdec = deciding(system, REF_MODEL, rcache)
            re_ = rcache["modes"][rdec["idx_lowest"]]
            if re_["band"] < 0:
                print(f"[geom] {system}/{model}: {REF_MODEL} is also stable; no reference path")
                continue
            ref_stem = path_stem(system, REF_MODEL, re_["q"], re_["band"])
            stem = path_stem(system, model, re_["q"], re_["band"], REF_MODEL)
            todo.append({"kind": "ref", "stem": stem, "ref_stem": ref_stem, "e": re_})
        else:
            for i, e in enumerate(cache["modes"]):
                role = "decide" if i in dec["idx_all"] else "mode"
                todo.append({"kind": "own", "i": i, "role": role, "e": e,
                             "stem": path_stem(system, model, e["q"], e["band"])})
        pending = [t for t in todo if args.force
                   or not (out / "geom" / f"{t['stem']}.extxyz").exists()
                   or not (out / "geom" / "patterns" / f"{t['stem']}.json").exists()]
        if not pending:
            print(f"[geom] {system}/{model}: all {len(todo)} paths present, skipped")
            continue
        missing_ref = [t for t in pending if t["kind"] == "ref" and not
                       (out / "geom" / "patterns" / f"{t['ref_stem']}.json").exists()]
        if missing_ref:
            print(f"[geom] DEFERRED {system}/{model}: harmonically stable, needs the {REF_MODEL} "
                  f"pattern {missing_ref[0]['ref_stem']}.json -- run geom --model {REF_MODEL} "
                  f"first, then re-run this", flush=True)
            record.update({"system": system, "model": model, "status": "deferred",
                           "deferred_on": missing_ref[0]["ref_stem"], "provenance": prov})
            _dump(check_path, record)
            status = 3
            continue

        t0 = time.time()
        spec = get_spec(system)
        fc_sc = tuple(int(x) for x in cache["fc_supercell"])
        prim = build_atoms(spec)
        prim.calc = calc
        prim = _relax(prim, fmax=FMAX)
        ph = _harmonic_phonon(prim, calc, fc_sc, DISP)
        regen, n_imag_regen = _imaginary_commensurate_modes(ph, fc_sc, DEFAULT_IMAG_TOL_THZ,
                                                            MAX_MODES)
        mode_list = {"n_imag_total_regen": int(n_imag_regen),
                     "n_imag_total_cache": int(cache["n_imag_total"]),
                     "regen": [{"q": m["q"], "band": m["band"], "harm_thz": m["harm_thz"]}
                               for m in regen],
                     "cache": [{"q": e["q"], "band": e["band"], "harm_thz": e["harm_thz"]}
                               for e in cache["modes"]]}
        if stable_model:
            f0, qsoft, *_ = _softest_mesh_mode(ph, fc_sc)
            mode_list["stable_check"] = {
                "softest_regen_thz": float(f0), "softest_cache_thz": cache["modes"][0]["harm_thz"],
                "q_regen": [float(x) for x in qsoft], "q_cache": cache["modes"][0]["q"],
                "pass": bool(n_imag_regen == 0
                             and abs(f0 - cache["modes"][0]["harm_thz"]) < 0.05)}
        max_den = max(fc_sc)

        for t in pending:
            e = t["e"]
            if t["kind"] == "own":
                dim, base, u, m_eff = _mode_pattern(ph, e["q"], e["band"], max_den)
                pattern_model, role, ecache = model, t["role"], e
                extra = {"decide_T": sorted(v["T"] for v in dec["by_T"].values()
                                            if v["idx"] == t["i"]),
                         "decide_lowest_T": bool(t["i"] == dec["idx_lowest"]),
                         "cache_index": t["i"]}
            else:
                # The reference coordinate: MatterSim's pattern carried to this model's relaxed
                # lattice. Fractional coordinates are kept (set_cell scale_atoms) and the
                # Cartesian unit pattern u is kept, so Q means the same amplitude in A.
                pat = _load(out / "geom" / "patterns" / f"{t['ref_stem']}.json")
                base0 = Atoms(symbols=pat["base_symbols"], positions=pat["base_positions"],
                              cell=pat["base_cell"], pbc=True)
                tmat = np.linalg.solve(np.array(pat["prim_cell"]), prim.cell.array)
                base = base0.copy()
                base.set_cell(np.array(pat["base_cell"]) @ tmat, scale_atoms=True)
                u, m_eff, dim = np.array(pat["u"]), float(pat["m_eff"]), pat["dim"]
                pattern_model, role, ecache = REF_MODEL, "ref", None
                own = _freqs_at(ph, e["q"])
                extra = {"ref_pattern_stem": t["ref_stem"],
                         "lattice_transform": tmat.tolist(),
                         "lattice_scale": float(np.cbrt(np.linalg.det(tmat))),
                         "lattice_transform_offdiag_max": float(np.abs(
                             tmat - np.diag(np.diag(tmat))).max()),
                         "this_model_freqs_at_q_thz": own.tolist(),
                         "ref_pattern_check": pat.get("check", {}).get("pass"),
                         "ref_pattern_standin": bool(pat.get("standin", False))}
            E0 = _energy(base, calc)
            Qs, dE = _sample_well(base, calc, u, Q_MAX, N_PTS)
            harm = float(_freqs_at(ph, e["q"])[int(e["band"])]) if t["kind"] == "own" \
                else float(e["harm_thz"])
            if t["kind"] == "own":
                check = _compare_map(Qs, dE, e)
                # A different harmonic frequency is recorded as drift but does NOT stop the
                # search. The frequency comes from 0.01 A finite-displacement FCs; E(Q) is
                # sampled at 0.005-0.45 A. Locally CHGNet BaTiO3 Gamma sits 0.24 THz off the
                # cache, yet a direction in its degenerate span reproduces the cached map to
                # 0.6 meV. Only the map comparison says whether the potential along the
                # coordinate changed.
                drift = abs(harm - float(e["harm_thz"])) > args.harm_tol
                if (not check["pass"] and "tol_meV" in check
                        and not args.no_recover
                        and (role == "decide" or len(base) <= args.recover_max_atoms)):
                    u_r, rinfo = _recover_pattern(base, calc, ph, e, dim, E0, check["tol_meV"])
                    if u_r is not None:
                        Qs2, dE2 = _sample_well(base, calc, u_r, Q_MAX, N_PTS)
                        check2 = _compare_map(Qs2, dE2, e)
                        # copies: whichever check survives gets rinfo as its "recovery", and
                        # holding the same dict here would make the record circular (json.dump
                        # raises on it, killing geom after the extxyz is already written)
                        rinfo["phonopy_pattern_check"] = dict(check)
                        rinfo["recovered_check"] = dict(check2)
                        # Adopted if it beats the phonopy pattern on (pass, fit-window pass,
                        # max diff), in that order: a window pass outranks a smaller max diff
                        # because the window sets the screen's a, b, c. Locally CHGNet BaTiO3 X
                        # comes back at 4.5 meV (window 1.1) against LAPACK's 10.4 eV: not a
                        # pass, but the screen's coordinate on a marginally different cell, so
                        # it is kept (flagged FAIL). u is saved, so an adopted search result
                        # is as reproducible as LAPACK's choice.
                        if _check_rank(check2) > _check_rank(check):
                            u, Qs, dE, check = u_r, Qs2, dE2, check2
                            m_eff = float(np.sum(base.get_masses()[:, None] * u ** 2))
                            rinfo["adopted"] = True
                        else:
                            rinfo["adopted"] = False
                    check["recovery"] = rinfo
                check["m_eff_regen"], check["m_eff_cache"] = float(m_eff), float(e["m_eff"])
                check["m_eff_rel_diff"] = float(abs(m_eff - e["m_eff"]) / e["m_eff"])
                check["harm_regen_thz"], check["harm_cache_thz"] = harm, float(e["harm_thz"])
                check["harm_drift"] = bool(drift)
                check["dim_regen"], check["dim_cache"] = [int(x) for x in dim], e["dim"]
                check["dim_match"] = [int(x) for x in dim] == [int(x) for x in e["dim"]]
                if not check["pass"] or not check["dim_match"]:
                    if drift:
                        print(f"[geom] !!! DRIFT {t['stem']}: harmonic {harm:.4f} THz here vs "
                              f"{float(e['harm_thz']):.4f} in the cache", flush=True)
                    print(f"[geom] !!! CACHE MISMATCH {t['stem']}: max|dE diff| "
                          f"{check.get('max_abs_diff_meV', float('nan')):.3f} meV "
                          f"(tol {check.get('tol_meV', float('nan')):.2f}; window "
                          f"{check.get('max_abs_diff_window_meV', float('nan')):.3f}); "
                          f"m_eff {m_eff:.2f} vs {e['m_eff']:.2f}; structures still written, "
                          f"flagged", flush=True)
                    status = max(status, 4)
            else:
                check = {"pass": None, "note": "reference path: no cached map for this "
                                               "coordinate under this model",
                         "stable_model_check": mode_list.get("stable_check")}
            meta = {"system": system, "path_model": model, "pattern_model": pattern_model,
                    "mlip_model": model, "role": role, "stem": t["stem"],
                    "q": np.array(e["q"], float), "band": int(e["band"]),
                    "dim": np.array(dim, int), "m_eff": float(m_eff),
                    "check_pass": "none" if check["pass"] is None else str(bool(check["pass"]))}
            frames = _path_frames(base, u, Qs, E0 + np.asarray(dE), meta)
            _write_frames(out / "geom" / f"{t['stem']}.extxyz", frames)
            _dump(out / "geom" / "patterns" / f"{t['stem']}.json",
                  _pattern_record(t["stem"], system, model, pattern_model, role, e["q"],
                                  e["band"], dim, base, u, m_eff, prim, E0, Qs, dE, harm,
                                  ecache, check, extra, prov))
            record["paths"][t["stem"]] = {"role": role, "n_atoms": len(base), "check": check,
                                          **{k: v for k, v in extra.items()
                                             if k != "lattice_transform"}}
            tag = "PASS" if check["pass"] else ("REF " if check["pass"] is None else "FAIL")
            if check.get("recovery", {}).get("adopted"):
                tag += " (recovered in the degenerate span)"
            tally.append((t["stem"], role, len(base), tag.split()[0]))
            dmax = check.get("max_abs_diff_meV")
            print(f"[geom] {tag} {t['stem']:48s} nat={len(base):3d} "
                  f"depth={-min(np.min(dE), 0) * 1000:8.2f} meV"
                  + (f"  max|diff|={dmax:.4f} meV" if dmax is not None else ""), flush=True)

        record.update({"system": system, "model": model, "status": "done",
                       "cache_file": cpath.relative_to(REPO).as_posix(),
                       "cache_sha256": _sha256(cpath), "fc_supercell": list(fc_sc),
                       "relaxed_prim_cell": prim.cell.array.tolist(),
                       "harmonically_stable_in_cache": stable_model, "deciding": dec,
                       "mode_list": mode_list, "wall_s": round(time.time() - t0, 1),
                       "provenance": prov})
        if not dec["all_agree"]:
            print(f"[geom] !!! {system}/{model}: ledger deciding mode disagrees with the cache "
                  f"re-solve at some T (recorded; ledger used)", flush=True)
        _dump(check_path, record)
    if tally:
        fails = [t for t in tally if t[3] == "FAIL"]
        print(f"[geom] summary {args.model}: {sum(t[3] == 'PASS' for t in tally)} pass, "
              f"{sum(t[3] == 'REF' for t in tally)} reference paths, {len(fails)} fail "
              f"({sum(t[1] == 'decide' for t in fails)} of them deciding modes)")
        for t in fails:
            print(f"[geom]   FAIL {t[0]} role={t[1]} nat={t[2]}")
    return status


# -------------------------------------------------------------- mlip-eval stage ----

def _geom_files(out):
    files = sorted((out / "geom").glob("*.extxyz")) + sorted((out / "geom_c3b").glob("*.extxyz"))
    return [f for f in files if not f.name.endswith(".tmp.extxyz")]


def stage_mlip_eval(args):
    from ase import Atoms
    from mlip_dynstab.calculators import get_calculator
    out = Path(args.out_root)
    handle = get_calculator(args.model, device=args.device)
    prov = provenance(args, "mlip-eval", args.model, handle)
    n_done = n_skip = 0
    for gf in _geom_files(out):
        dest = out / "mlip" / args.model / gf.parent.name / (gf.stem + ".json")
        sha = _sha256(gf)
        if dest.exists() and not args.force and _load(dest).get("geom_sha256") == sha:
            n_skip += 1
            continue
        frames = _read_frames(gf)
        t0 = time.time()
        E, F = [], []
        for a in frames:
            b = Atoms(symbols=a.get_chemical_symbols(), positions=a.get_positions(),
                      cell=a.get_cell(), pbc=True)
            b.calc = handle.calc
            E.append(float(b.get_potential_energy()))
            F.append(np.asarray(b.get_forces(), float))
        # Consistency with the energies/forces the producing stage attached, where this model
        # produced them: catches an env or checkpoint drift between stages.
        cons = {}
        own_E = [(i, float(a.info["mlip_energy_eV"])) for i, a in enumerate(frames)
                 if a.info.get("mlip_model") == args.model and "mlip_energy_eV" in a.info]
        if own_E:
            cons["max_abs_dE_vs_geom_meV"] = max(abs(E[i] - e) for i, e in own_E) * 1000
        src = [(i, a) for i, a in enumerate(frames) if a.info.get("owner_model") == args.model
               and a.info.get("role") == "sscha"]
        if src:
            dE_src = [abs(E[i] - float(a.info["src_mlip_energy_eV"])) for i, a in src
                      if np.isfinite(float(a.info.get("src_mlip_energy_eV", np.nan)))]
            dF_src = [float(np.abs(F[i] - a.arrays["src_mlip_forces"]).max()) for i, a in src
                      if "src_mlip_forces" in a.arrays]
            if dE_src:
                cons["max_abs_dE_vs_sscha_meV"] = max(dE_src) * 1000
            if dF_src:
                cons["max_abs_dF_vs_sscha_eV_A"] = max(dF_src)
        _dump(dest, {"provenance": prov, "model": args.model, "model_version": handle.version,
                     "geom_file": _rel(gf, out), "geom_sha256": sha, "n_frames": len(frames),
                     "n_atoms": [len(a) for a in frames], "energies_eV": E,
                     "forces_eV_A": [f.tolist() for f in F], "consistency": cons,
                     "wall_s": round(time.time() - t0, 2)})
        n_done += 1
        flag = "  ".join(f"{k}={v:.3g}" for k, v in cons.items())
        print(f"[mlip-eval] {args.model} {gf.parent.name}/{gf.name} ({len(frames)} frames) {flag}",
              flush=True)
    print(f"[mlip-eval] {args.model}: {n_done} evaluated, {n_skip} already current")


# ------------------------------------------------------------- QE input writer ----

def load_sssp(args):
    """Per-element pseudopotential file and recommended cutoffs from the SSSP json.

    SSSP 1.3 keys are cutoff_wfc / cutoff_rho (Ry); the 1.0/1.1 layout (cutoff, dual) is also
    accepted so an older json cannot silently yield wrong cutoffs."""
    if args.sssp_json:
        jp = Path(args.sssp_json)
    else:
        if not args.sssp_dir:
            raise SystemExit("give --sssp-dir (or --sssp-json)")
        c = sorted(Path(args.sssp_dir).glob("*.json"))
        eff = [p for p in c if "efficiency" in p.name.lower()]
        c = eff or c
        if len(c) != 1:
            raise SystemExit(f"expected one SSSP json in {args.sssp_dir}, found "
                             f"{[p.name for p in c]}; pass --sssp-json")
        jp = c[0]
    raw = _load(jp)
    table = {}
    for el, d in raw.items():
        wfc = d.get("cutoff_wfc", d.get("cutoff"))
        rho = d.get("cutoff_rho")
        if rho is None and wfc is not None and "dual" in d:
            rho = float(wfc) * float(d["dual"])
        if wfc is None or rho is None or "filename" not in d:
            continue
        table[el] = {"filename": d["filename"], "ecutwfc": float(wfc), "ecutrho": float(rho),
                     "md5": d.get("md5"), "family": d.get("pseudopotential")}
    pdir = Path(args.sssp_dir) if args.sssp_dir else jp.parent
    return {"json": jp.name, "sha256": _sha256(jp), "table": table, "dir": pdir}


def _fortran(x):
    return f"{x:.3e}".replace("e", "d")


def _structure_sha(atoms):
    frac = np.round(np.mod(np.round(atoms.get_scaled_positions(wrap=False), 6), 1.0), 6) + 0.0
    blob = json.dumps([atoms.get_chemical_symbols(), np.round(atoms.cell.array, 6).tolist(),
                       frac.tolist()], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def pw_input(atoms, job, system, sssp, args):
    from ase.data import atomic_masses, atomic_numbers
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    missing = [s for s in species if s not in sssp["table"]]
    if missing:
        raise SystemExit(f"SSSP json {sssp['json']} has no entry for {missing}")
    ecutwfc = max(sssp["table"][s]["ecutwfc"] for s in species)
    ecutrho = max(sssp["table"][s]["ecutrho"] for s in species)
    metal = is_metal(system)
    spacing = args.kspacing_metal if metal else args.kspacing_insulator
    degauss = args.degauss_metal if metal else args.degauss_insulator
    mesh = _kmesh(atoms.cell.array, spacing)
    nat = len(atoms)
    conv = 1e-10 * nat
    L = ["&CONTROL",
         f"  ! job {job}, written by {SCRIPT}",
         "  ! pseudo_dir is taken from $ESPRESSO_PSEUDO (exported by scripts/box/qe_queue.sh)",
         "  calculation = 'scf'", "  prefix = 'pwscf'", "  outdir = './scratch'",
         "  disk_io = 'none'", "  tprnfor = .true.", "  tstress = .false.",
         "  verbosity = 'low'", "/",
         "&SYSTEM", "  ibrav = 0", f"  nat = {nat}", f"  ntyp = {len(species)}",
         f"  ecutwfc = {ecutwfc:.1f}", f"  ecutrho = {ecutrho:.1f}",
         "  occupations = 'smearing'", "  smearing = 'mv'", f"  degauss = {degauss}", "/",
         "&ELECTRONS", f"  conv_thr = {_fortran(conv)}", "  mixing_beta = 0.4",
         "  electron_maxstep = 200", "/", "ATOMIC_SPECIES"]
    L += [f"{s} {atomic_masses[atomic_numbers[s]]:.4f} {sssp['table'][s]['filename']}"
          for s in species]
    L += ["CELL_PARAMETERS angstrom"]
    L += ["  " + " ".join(f"{x:.10f}" for x in v) for v in atoms.cell.array]
    L += ["ATOMIC_POSITIONS angstrom"]
    L += [f"{s} " + " ".join(f"{x:.10f}" for x in p)
          for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
    L += ["K_POINTS automatic", f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0", ""]
    settings = {"is_metal": metal, "smearing": "mv", "degauss_Ry": degauss,
                "kspacing_inv_A": spacing, "kspacing_convention": KSPACING_CONVENTION,
                "kmesh": mesh, "nk_full": int(np.prod(mesh)), "nk_tr_upper": _nk_tr_upper(mesh),
                "nk_irr_spglib": _nk_irr(atoms, mesh),
                "ecutwfc_Ry": ecutwfc, "ecutrho_Ry": ecutrho, "conv_thr_Ry": conv,
                "mixing_beta": 0.4, "disk_io": "none", "functional": "PBE (from the SSSP PBE set)",
                "pseudos": {s: sssp["table"][s] for s in species},
                "sssp_json": sssp["json"], "sssp_sha256": sssp["sha256"]}
    return "\n".join(L), settings


class JobWriter:
    """One directory per pw.x job under <out>/qe/<job>/ (pw.in + job.json).

    Identical structures (the undisplaced Q=0 cell shared by two paths of one model) become one
    job with several members, found by a structure hash that also scans jobs written earlier,
    so a later run attaches to the existing job instead of renaming it. A pw.in is never
    overwritten under a finished pw.out unless --force: that would silently orphan a result."""

    def __init__(self, out, sssp, args, prov):
        self.qe = out / "qe"
        self.sssp, self.args, self.prov = sssp, args, prov
        self.by_sha, self.meta = {}, {}
        for jj in sorted(self.qe.glob("*/job.json")):
            m = _load(jj)
            self.by_sha[m["structure_sha"]] = m["job"]
            self.meta[m["job"]] = m
        self.n_new = self.n_alias = self.n_same = self.n_refused = 0

    def would_refuse(self, job, atoms, system):
        """True if add() would refuse this job: a finished pw.out under a pw.in that the
        current settings would change, without --force. qe-inputs asks for every frame of a
        path first, because refusing one frame while rewriting the rest leaves a path whose
        E(Q)-E(0) mixes two k-meshes or cutoffs."""
        sha = _structure_sha(atoms)
        if self.args.force or (sha in self.by_sha and self.by_sha[sha] != job):
            return False
        d = self.qe / job
        if not ((d / "pw.in").exists() and (d / "pw.out").exists()):
            return False
        text, _ = pw_input(atoms, job, system, self.sssp, self.args)
        return (d / "pw.in").read_text(encoding="utf-8") != text

    def add(self, job, atoms, system, member):
        sha = _structure_sha(atoms)
        if sha in self.by_sha and self.by_sha[sha] != job:
            host = self.by_sha[sha]
            mem = self.meta[host]["members"]
            if not any(m["geom_file"] == member["geom_file"] and m["frame"] == member["frame"]
                       for m in mem):
                mem.append(member)
                _dump(self.qe / host / "job.json", self.meta[host])
            self.n_alias += 1
            return host
        text, settings = pw_input(atoms, job, system, self.sssp, self.args)
        d = self.qe / job
        pw = d / "pw.in"
        if pw.exists():
            old = pw.read_text(encoding="utf-8")
            if old == text:
                self.n_same += 1
                m = self.meta.get(job) or {}
                if m and not any(x["geom_file"] == member["geom_file"]
                                 and x["frame"] == member["frame"] for x in m["members"]):
                    m["members"].append(member)
                    _dump(d / "job.json", m)
                if m:
                    return job
            elif (d / "pw.out").exists() and not self.args.force:
                print(f"[qe] REFUSED {job}: pw.in would change under an existing pw.out "
                      f"(delete the job dir or pass --force)")
                self.n_refused += 1
                return job
            else:
                # --force with a changed input: the old output answers another question. Left
                # in place, its 'JOB DONE' would make the queue skip the job and analyze would
                # read it (the structure check cannot see a k-mesh or cutoff change).
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                for old_f in ("pw.out", "pw.err", "pw.in.ran"):
                    if (d / old_f).exists():
                        os.replace(d / old_f, d / f"{old_f}.superseded-{stamp}")
        d.mkdir(parents=True, exist_ok=True)
        with open(pw, "w", encoding="utf-8", newline="\n") as fh:   # LF: pw.x runs on Linux
            fh.write(text)
        for k in [k for k, v in self.by_sha.items() if v == job and k != sha]:
            del self.by_sha[k]                     # the job's old structure no longer lives here
        meta = {"job": job, "system": system, "structure_sha": sha, "n_atoms": len(atoms),
                "species": sorted(set(atoms.get_chemical_symbols())),
                "members": [member], "qe": settings,
                "pw_in_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "provenance": self.prov}
        _dump(d / "job.json", meta)
        self.by_sha[sha] = job
        self.meta[job] = meta
        self.n_new += 1
        return job

    def report(self, label):
        print(f"[{label}] jobs: {self.n_new} written, {self.n_same} unchanged, "
              f"{self.n_alias} identical structures attached to an existing job, "
              f"{self.n_refused} refused")


def _reduced_lattices(frame, symprec=1e-4):
    """Candidate smaller lattices the modulated supercell is periodic in (maybe none).

    A frozen R-point mode in a 2x2x2 perovskite repeats in a 10-atom cell whatever direction it
    takes in the R25 triplet; an M mode in 10 atoms, the bcc (1/3,2/3,0) mode in 3. spglib with
    no_idealize keeps the Cartesian frame; a result that is not an integer sublattice of the
    supercell (a rotation, or a spurious cell) is rejected. Two bases are offered, spglib's
    and its Niggli-reduced form, because the k-spacing rule over-samples a skewed cell and the
    cheaper one is not always the more compact-looking one."""
    try:
        import spglib
        from ase.build.niggli import niggli_reduce_cell
        res = spglib.standardize_cell((frame.cell.array, frame.get_scaled_positions(),
                                       frame.get_atomic_numbers()),
                                      to_primitive=True, no_idealize=True, symprec=symprec)
    except Exception:
        return []
    if res is None or len(res[2]) >= len(frame):
        return []
    lat = np.asarray(res[0], float)
    n = frame.cell.array @ np.linalg.inv(lat)
    if np.abs(n - np.rint(n)).max() > 1e-6:
        return []
    out = [lat]
    op = np.asarray(niggli_reduce_cell(lat)[1])    # its own cell is re-oriented; use the op
    if np.allclose(op, np.rint(op)) and round(abs(np.linalg.det(op))) == 1:
        out.append(np.rint(op) @ lat)
    # pw.x wants right-handed CELL_PARAMETERS; -a3 spans the same lattice
    return [c if np.linalg.det(c) > 0 else c * np.array([[1.0], [1.0], [-1.0]]) for c in out]


def _pick_cell(frame, system, args):
    """(lattice or None, reps, costs): the cheapest periodic cell for pw.x by modelled cost
    (spglib-irreducible k at the spacing rule x n_atoms^2.2), None meaning the full supercell."""
    sp = args.kspacing_metal if is_metal(system) else args.kspacing_insulator
    mesh = _kmesh(frame.cell.array, sp)
    best = (cost_core_h(len(frame), _nk_irr(frame, mesh) or _nk_tr_upper(mesh), 1.0), None, None)
    costs = {"full": best[0]}
    if args.no_reduce_cell:
        return None, None, costs
    for k, lat in enumerate(_reduced_lattices(frame)):
        r = _reduce_frame(frame, lat)
        if r is None:
            continue
        m = _kmesh(lat, sp)
        c = cost_core_h(len(r[0]), _nk_irr(r[0], m) or _nk_tr_upper(m), 1.0)
        costs[f"reduced{k}_{len(r[0])}at"] = c
        if c < best[0]:
            best = (c, lat, r[2])
    return best[1], best[2], costs


def _reduce_frame(frame, lat, reps=None, tol=1e-5):
    """(reduced Atoms, map supercell atom -> reduced atom, representative atoms) or None.

    Every supercell atom must sit on an image of its representative, same species, within tol
    A; otherwise the frame is not periodic in lat and None comes back. Passing the path's reps
    keeps the reduced atom order identical along the path, so forces map back the same way."""
    from ase import Atoms
    n_img = int(round(abs(np.linalg.det(frame.cell.array) / np.linalg.det(lat))))
    if n_img < 2 or len(frame) % n_img:
        return None
    inv = np.linalg.inv(lat)
    frac = frame.get_positions() @ inv
    z = frame.get_atomic_numbers()

    def dist(i, j):
        d = frac[i] - frac[j]
        d -= np.round(d)
        return float(np.linalg.norm(d @ lat))

    if reps is None:
        reps = []
        for i in range(len(frame)):
            if not any(z[r] == z[i] and dist(r, i) < tol for r in reps):
                reps.append(i)
    if len(reps) * n_img != len(frame):
        return None
    mp = np.full(len(frame), -1)
    for i in range(len(frame)):
        hit = [k for k, r in enumerate(reps) if z[r] == z[i] and dist(r, i) < tol]
        if len(hit) != 1:
            return None
        mp[i] = hit[0]
    if np.bincount(mp, minlength=len(reps)).min() != n_img:
        return None
    f = frac[reps] - np.floor(frac[reps])
    red = Atoms(numbers=z[reps], scaled_positions=f, cell=lat, pbc=True)
    return red, mp.tolist(), [int(r) for r in reps]


def _cost_report(w, c0, label):
    """Modelled cost of every job under <out>/qe, by prefix, with the spglib-irreducible k count
    (the TR-only bound in brackets). c0 is an assumption until analyze measures it."""
    groups = {}
    for m in w.meta.values():
        q = m["qe"]
        nk = q.get("nk_irr_spglib") or q["nk_tr_upper"]
        g = groups.setdefault(m["job"].split("_", 1)[0], [0, 0.0, 0.0, 0])
        g[0] += 1
        g[1] += cost_core_h(m["n_atoms"], nk, c0)
        g[2] += cost_core_h(m["n_atoms"], q["nk_tr_upper"], c0)
        g[3] = max(g[3], m["n_atoms"])
    for p, (n, h, hu, nmax) in sorted(groups.items()):
        print(f"[{label}] {p}_*: {n} jobs, <= {nmax} atoms, modelled {h:.0f} core-h "
              f"({hu:.0f} with no symmetry), c0 = {c0} core-s/k-point (ASSUMED)")


def _check_pseudos(sssp, w):
    """UPF files named by the written jobs that are not in the SSSP directory."""
    used = {v["filename"] for m in w.meta.values() for v in m["qe"]["pseudos"].values()}
    return sorted(f for f in used if not (sssp["dir"] / f).exists())


def stage_qe_inputs(args):
    out = Path(args.out_root)
    sssp = load_sssp(args)
    prov = provenance(args, "qe-inputs")
    pats = []
    for p in sorted((out / "geom" / "patterns").glob("*.json")):
        d = _load(p)
        if d["system"] not in args.systems:
            continue
        if not (out / d["geom_file"]).exists():
            print(f"[qe-inputs] skip {d['stem']}: pattern without a geometry file "
                  f"(a {d['pattern_model']} pattern consumed only by reference paths?)")
            continue
        pats.append(d)
    taken, skipped = select_paths(pats, args.extra_max_atoms)
    if args.only_decide:
        taken = [d for d in taken if d["role"] in ("decide", "ref")]
    w = JobWriter(out, sssp, args, prov)
    n_red = []
    for d in taken:
        frames = _read_frames(out / d["geom_file"])
        # 'a_' = deciding and reference paths (the R1.1 answer); 'ax_' = optional extra modes,
        # so the queue can run the answer first: --filter '^a_' then '^b_' then '^ax_'.
        prefix = "ax" if d["role"] == "mode" else "a"
        # One cell per path, chosen on the most displaced frame (its symmetry is the path's;
        # Q=0 has more and is periodic in the same lattice), and one atom order, so E(Q)-E(0)
        # is taken between identical k-point sets.
        lat, reps, costs = _pick_cell(frames[-1], d["system"], args)
        reduced = None
        if lat is not None:
            reduced = [_reduce_frame(a, lat, reps) for a in frames]
            if any(r is None for r in reduced):
                # All frames of a path must share one cell (E(Q)-E(0) on one k set), so a
                # single non-periodic frame sends the whole path back to its full supercell.
                print(f"[qe-inputs] WARNING {d['stem']}: frame(s) "
                      f"{[i for i, r in enumerate(reduced) if r is None]} not periodic in the "
                      f"reduced cell; this path runs in its full {len(frames[0])}-atom cell")
                lat, reduced = None, None
        run_atoms = [a if reduced is None else reduced[i][0] for i, a in enumerate(frames)]
        blocked = [i for i, a in enumerate(run_atoms)
                   if w.would_refuse(f"{prefix}_{d['stem']}_i{i:02d}", a, d["system"])]
        if blocked:
            print(f"[qe] REFUSED path {d['stem']}: frame(s) {blocked} have a finished pw.out "
                  f"under different settings; the whole path is left as it was (mixing two "
                  f"k-meshes or cutoffs in one E(Q) is worse than either). --force redoes it.")
            w.n_refused += len(frames)
            continue
        for i, a in enumerate(frames):
            job = f"{prefix}_{d['stem']}_i{i:02d}"
            member = {"study": "c3a", "geom_file": d["geom_file"], "frame": i,
                      "stem": d["stem"], "role": d["role"], "path_model": d["path_model"],
                      "Q": float(a.info["Q"]), "reduction": None}
            atoms = a
            if reduced is not None:
                r = reduced[i]
                atoms = r[0]
                member["reduction"] = {"n_images": len(a) // len(atoms), "map": r[1],
                                       "reps": r[2], "lattice": lat.tolist()}
            w.add(job, atoms, d["system"], member)
        if lat is not None:
            n_red.append(f"{d['stem']}: {len(frames[0])} -> {len(reps)} atoms, modelled cost "
                         f"x{min(costs.values()) / costs['full']:.2f}")
    for line in n_red:
        print(f"[qe-inputs] reduced cell {line}")
    w.report("qe-inputs")
    _cost_report(w, args.c0_core_s, "qe-inputs")
    miss = _check_pseudos(sssp, w)
    print(f"[qe-inputs] {len(taken)} paths -> pw.x inputs; {len(skipped)} modes left out "
          f"(cap {args.extra_max_atoms} atoms or symmetry copies)")
    if miss:
        print(f"[qe-inputs] WARNING: UPF files not found in {sssp['dir']}: {miss[:6]}"
              f"{' ...' if len(miss) > 6 else ''} (the queue checks again before running)")
    _dump(out / "qe" / "c3a_selection.json",
          {"provenance": prov, "taken": [d["stem"] for d in taken], "skipped": skipped})


# ------------------------------------------------------------ c3b-inputs stage ----

def _cfg_meta(a, f, k):
    info = a.info

    def get(*keys):
        for key in keys:
            if key in info:
                return info[key]
        return None
    system, model = get("system"), get("model")
    T = get("T", "temperature_K")
    if system is None or model is None or T is None:
        raise SystemExit(f"{f.name} frame {k}: info must carry system, model and T "
                         f"(scripts/sscha_seed_study.py contract); got {sorted(info)}")
    E = None
    if a.calc is not None:
        E = a.calc.results.get("energy")
    if E is None:
        E = get("energy", "mlip_energy", "mlip_energy_eV")
    F = None
    if a.calc is not None and "forces" in a.calc.results:
        F = np.asarray(a.calc.results["forces"], float)
    elif "forces" in a.arrays:
        F = np.asarray(a.arrays["forces"], float)
    elif "mlip_forces" in a.arrays:
        F = np.asarray(a.arrays["mlip_forces"], float)
    seed = get("seed")
    idx = get("index")
    return {"system": str(system), "model": str(model), "T": float(T),
            "seed": 0 if seed is None else seed, "index": k if idx is None else int(idx),
            "energy": None if E is None else float(E), "forces": F,
            "src_file": f.name, "src_frame": k}


def _select_configs(items, n):
    """Up to n configs spread evenly over seeds and over each seed's index range, taken
    round-robin across seeds so a truncated set still spans every seed.

    sscha_seed_study.py saves antithetic pairs (u and -u, info 'pair'). Their MLIP errors share
    the even-order part, so one member of each pair is taken before any partner: a PBE job
    buys more independent information on a new pair than on the mirror image of one taken."""
    by_seed = {}
    for meta, a in items:
        by_seed.setdefault(meta["seed"], []).append((meta, a))
    seeds = sorted(by_seed, key=str)
    quota = int(math.ceil(n / max(len(seeds), 1)))
    picks = {}
    def spread(lst, k):
        idx = np.unique(np.linspace(0, len(lst) - 1, k).round().astype(int)) if k else []
        return [lst[i] for i in idx]

    for s in seeds:
        lst = sorted(by_seed[s], key=lambda t: t[0]["index"])
        k = min(quota, len(lst))
        if all("pair" in a.info for _, a in lst):
            first, rest, seen = [], [], set()
            for t in lst:
                (rest if int(t[1].info["pair"]) in seen else first).append(t)
                seen.add(int(t[1].info["pair"]))
            picks[s] = spread(first, min(k, len(first))) + spread(rest, max(0, k - len(first)))
        else:
            picks[s] = spread(lst, k)
    out, r = [], 0
    while len(out) < n and any(r < len(picks[s]) for s in seeds):
        for s in seeds:
            if r < len(picks[s]) and len(out) < n:
                out.append(picks[s][r])
        r += 1
    return out


def _ideal_supercell(system, cfg):
    """The undisplaced supercell on the config's own cell. The four prototypes have no free
    internal coordinates, so the relaxed ideal structure is the prototype scaled onto the
    cell the SSCHA ran with. Atom order does not matter: only its energy is used."""
    from ase.build import make_supercell
    prim = build_atoms(get_spec(system))
    n_rep = len(cfg) / len(prim)
    M = cfg.cell.array @ np.linalg.inv(prim.cell.array)
    s = (abs(np.linalg.det(M)) / n_rep) ** (1.0 / 3.0)
    P = np.rint(M / s).astype(int)
    resid = float(np.abs(M / s - P).max())
    if resid > 0.02 or round(abs(np.linalg.det(P))) != round(n_rep):
        raise SystemExit(f"{system}: config cell is not a scaled supercell of the prototype "
                         f"(residual {resid:.3g}, P={P.tolist()})")
    sc = make_supercell(prim, P)
    sc.set_cell(cfg.cell.array, scale_atoms=True)
    if sorted(sc.get_chemical_symbols()) != sorted(cfg.get_chemical_symbols()):
        raise SystemExit(f"{system}: config species differ from the prototype supercell")
    return sc, {"P": P.tolist(), "lattice_scale": float(s), "residual": resid}


def _displacements(cfg, ideal):
    """Per-atom displacement of cfg from its nearest same-species ideal site (minimum image)."""
    cell = cfg.cell.array
    fi = ideal.get_scaled_positions(wrap=True)
    fc = cfg.get_scaled_positions(wrap=True)
    si = np.array(ideal.get_chemical_symbols())
    sc = np.array(cfg.get_chemical_symbols())
    used, disp = set(), np.zeros((len(cfg), 3))
    for j in range(len(cfg)):
        cand = np.where(si == sc[j])[0]
        d = fc[j] - fi[cand]
        d -= np.round(d)
        cart = d @ cell
        k = int(np.argmin(np.linalg.norm(cart, axis=1)))
        used.add(int(cand[k]))
        disp[j] = cart[k]
    disp -= disp.mean(axis=0)          # rigid translation carries no energy
    return disp, len(used) == len(cfg)


def stage_c3b_inputs(args):
    from ase import Atoms
    out = Path(args.out_root)
    cfg_dir = Path(args.sscha_dir) / "configs"
    files = sorted(cfg_dir.glob("*.extxyz"))
    if not files:
        raise SystemExit(f"no SSCHA configs under {cfg_dir}: run C1 "
                         f"(scripts/sscha_seed_study.py) first")
    prov = provenance(args, "c3b-inputs")
    groups = {}
    for f in files:
        for k, a in enumerate(_read_frames(f)):
            m = _cfg_meta(a, f, k)
            groups.setdefault((m["system"], m["model"], m["T"]), []).append((m, a))
    gdir = out / "geom_c3b"
    ideal_by = {}
    for (system, model, T), items in sorted(groups.items()):
        dest = gdir / f"{system}_{model}_T{_tstr(T)}.extxyz"
        seeds = sorted({str(m["seed"]) for m, _ in items})
        if dest.exists() and not args.force:
            n_prev = len(_read_frames(dest))
            print(f"[c3b] {dest.name} kept ({n_prev} configs; {len(items)} now available over "
                  f"{len(seeds)} seeds). --force reselects and orphans existing jobs.")
            ideal_by.setdefault((system, model), items[0][1])
            continue
        chosen = _select_configs(items, args.per_unit)
        ideal, fitinfo = _ideal_supercell(system, chosen[0][1])
        ideal_by[(system, model)] = chosen[0][1]
        frames = []
        for m, a in chosen:
            if abs(np.linalg.det(a.cell.array) - np.linalg.det(ideal.cell.array)) > 1e-6 * \
                    abs(np.linalg.det(ideal.cell.array)):
                raise SystemExit(f"{system}/{model}/T={T}: configs do not share one cell")
            disp, bij = _displacements(a, ideal)
            dev = None
            if "u_disp" in a.arrays:
                # the seed study's own displacement from the SSCHA centroid: exact, so it
                # replaces the site mapping, and it checks the prototype reference
                dev = float(np.abs((disp - disp.mean(axis=0))
                                   - (a.arrays["u_disp"] - a.arrays["u_disp"].mean(axis=0))
                                   ).max())
                disp = np.asarray(a.arrays["u_disp"], float)
            b = Atoms(symbols=a.get_chemical_symbols(), positions=a.get_positions(),
                      cell=a.cell.array, pbc=True)
            b.info.update({"system": system, "owner_model": model, "T": float(T),
                           "seed": m["seed"], "index": int(m["index"]), "role": "sscha",
                           "src_file": m["src_file"], "src_frame": int(m["src_frame"]),
                           "u_rms_A": float(np.sqrt((disp ** 2).sum(axis=1).mean())),
                           "u_max_A": float(np.linalg.norm(disp, axis=1).max()),
                           "site_map_bijective": str(bool(bij)),
                           "centroid_vs_prototype_A": -1.0 if dev is None else dev,
                           "pair": int(a.info.get("pair", -1)),
                           "src_mlip_energy_eV": float("nan") if m["energy"] is None
                           else m["energy"]})
            if m["forces"] is not None:
                b.arrays["src_mlip_forces"] = m["forces"]
            frames.append(b)
        _write_frames(dest, frames)
        if len(seeds) < 4 or len(chosen) < args.per_unit:
            print(f"[c3b] NOTE {system}/{model}/T={T:g}: {len(chosen)} configs from "
                  f"{len(seeds)} seeds (target {args.per_unit} over 4)")
        print(f"[c3b] {dest.name}: {len(frames)} configs, seeds {seeds}, "
              f"u_rms {np.mean([f.info['u_rms_A'] for f in frames]):.3f} A, "
              f"cell fit {fitinfo}")

    for (system, model), cfg in sorted(ideal_by.items()):
        dest = gdir / f"{system}_{model}_ref.extxyz"
        if dest.exists() and not args.force:
            continue
        ideal, fitinfo = _ideal_supercell(system, cfg)
        frames = []
        a = Atoms(symbols=ideal.get_chemical_symbols(), positions=ideal.get_positions(),
                  cell=ideal.cell.array, pbc=True)
        a.info.update({"system": system, "owner_model": model, "role": "ref", "index": -1,
                       "u_rms_A": 0.0, "u_max_A": 0.0})
        frames.append(a)
        for k in range(args.n_baseline):
            b = a.copy()
            b.info = dict(a.info)
            seed = args.rattle_seed + k
            b.rattle(stdev=args.rattle_stdev, seed=seed)
            disp, _ = _displacements(b, ideal)
            b.info.update({"role": "baseline", "index": k, "rattle_seed": seed,
                           "rattle_stdev_A": args.rattle_stdev,
                           "u_rms_A": float(np.sqrt((disp ** 2).sum(axis=1).mean())),
                           "u_max_A": float(np.linalg.norm(disp, axis=1).max())})
            frames.append(b)
        _write_frames(dest, frames)
        print(f"[c3b] {dest.name}: undisplaced + {args.n_baseline} rattled "
              f"(stdev {args.rattle_stdev} A), cell fit {fitinfo}")

    sssp = load_sssp(args)
    w = JobWriter(out, sssp, args, prov)
    for gf in sorted(gdir.glob("*.extxyz")):
        if gf.name.endswith(".tmp.extxyz"):
            continue
        for i, a in enumerate(_read_frames(gf)):
            system, model, role = a.info["system"], a.info["owner_model"], a.info["role"]
            if role == "sscha":
                job = (f"b_{system}_{model}_T{_tstr(a.info['T'])}_s{a.info['seed']}"
                       f"_n{int(a.info['index']):04d}")
            elif role == "ref":
                job = f"b_{system}_{model}_ref"
            else:
                job = f"b_{system}_{model}_base{int(a.info['index'])}"
            w.add(job, a, system, {"study": "c3b", "geom_file": _rel(gf, out), "frame": i,
                                   "role": role, "owner_model": model,
                                   "T": float(a.info["T"]) if "T" in a.info else None})
    w.report("c3b-inputs")
    _cost_report(w, args.c0_core_s, "c3b-inputs")
    miss = _check_pseudos(sssp, w)
    if miss:
        print(f"[c3b-inputs] WARNING: UPF files not found in {sssp['dir']}: {miss[:6]}")


# ---------------------------------------------------------------- analyze stage ----

def _qe_seconds(s):
    tok = re.findall(r"([\d.]+)\s*([dhms])", s)
    mult = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    return sum(float(v) * mult[u] for v, u in tok) if tok else None


def parse_pw_out(path):
    """Energy (eV; with smearing this is QE's '!' line, the variational free energy, which the
    forces are consistent with), forces (eV/A), convergence, and the cost figures."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    r = {"job_done": "JOB DONE" in txt,
         "converged": ("convergence has been achieved" in txt
                       and "convergence NOT achieved" not in txt)}
    m = re.search(r"Program PWSCF\s+v\.?\s*(\S+)", txt)
    r["qe_version"] = m.group(1) if m else None
    m = re.search(r"Number of MPI processes:\s+(\d+)", txt)
    r["nproc"] = int(m.group(1)) if m else None
    m = re.findall(r"number of k points=\s*(\d+)", txt)
    r["nk_irr"] = int(m[-1]) if m else None
    m = re.search(r"PWSCF\s*:\s*(.+?)CPU\s+(.+?)WALL", txt)
    r["wall_s"] = _qe_seconds(m.group(2)) if m else None
    m = re.findall(r"convergence has been achieved in\s+(\d+)\s+iterations", txt)
    r["n_scf"] = int(m[-1]) if m else None
    if r["job_done"] and r["converged"]:
        from ase.io import read
        try:
            a = read(path, format="espresso-out", index=-1)
            r["energy_eV"] = float(a.get_potential_energy())
            r["forces"] = np.asarray(a.get_forces(), float)
            r["atoms"] = a
        except Exception as exc:              # one odd file must not sink the whole analysis
            r["parse_error"] = f"{type(exc).__name__}: {exc}"
    return r


def _pos_mismatch(parsed, ref):
    if parsed.get_chemical_symbols() != ref.get_chemical_symbols():
        return float("inf")
    d = parsed.get_scaled_positions(wrap=False) - ref.get_scaled_positions(wrap=False)
    d -= np.round(d)
    cell_d = float(np.abs(parsed.cell.array - ref.cell.array).max())
    return max(float(np.linalg.norm(d @ ref.cell.array, axis=1).max()), cell_d)


def _curve_metrics(Qs, dE, m_eff):
    from mlip_dynstab.finite_t import _fit_double_well
    Qs, dE = np.asarray(Qs, float), np.asarray(dE, float)
    a, b, c = _fit_double_well(Qs, dE)
    i = int(np.argmin(dE))
    return {"depth_meV": float(-min(dE.min(), 0.0) * 1000), "Q_min_A": float(Qs[i]),
            "a": a, "b": b, "c": c, "bounded": bool(c > 0 or (b > 0 and c >= 0)),
            "omega_harm_thz": _omega_thz(a, m_eff)}


def _screen_call(task):
    """One single-mode screen solve; module-level so a process pool can run it (~1 s each)."""
    from mlip_dynstab.finite_t import _solve_scha, _sym_curvature_freq
    a, b, c, m_eff, T = task
    eff, Q0, st = _solve_scha(a, b, c, m_eff, T)
    curv = _sym_curvature_freq(a, b, c, m_eff, T)
    return {"stable": bool(st), "Q0_A": float(Q0), "eff_thz": float(eff),
            "curv_thz": None if curv is None else float(curv)}


def _solve_all(tasks, workers):
    if workers <= 1 or len(tasks) < 8:
        return [_screen_call(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_screen_call, tasks, chunksize=4))


def _load_mlip(out, gf, flags):
    """Every model's energies/forces on geometry file gf, dropping any computed on a different
    version of the file (a regenerated geometry with stale MLIP numbers would pass silently)."""
    res, sha = {}, _sha256(gf)
    for m in MODELS:
        p = out / "mlip" / m / gf.parent.name / f"{gf.stem}.json"
        if not p.exists():
            continue
        d = _load(p)
        if d["geom_sha256"] != sha:
            flags.append(f"{gf.name}: {m} numbers are for another version of the file "
                         f"(re-run mlip-eval --model {m}); ignored")
            continue
        res[m] = {"E": np.asarray(d["energies_eV"], float),
                  "F": [np.asarray(f, float) for f in d["forces_eV_A"]]}
    return res


def _ledger_call(system, model, T):
    try:
        rows = _ledger_rows(system, model)
    except Exception:
        return None
    r = rows[np.isclose(rows["temperature_K"].astype(float), float(T))]
    return None if r.empty else bool(r.iloc[0]["pred_stable"])


def _agreement(unit_calls):
    """Call agreement counts over the ladder rows, per system and pooled, with and without
    ORB-v2. Descriptive only: the rows are 4 systems x 5 models x 4 T, units within a system
    share one PES family and one experimental label, so the system is the cluster and a pooled
    test over rows would count one system's verdict up to 20 times. Also note the PBE curves
    sit at each MLIP's own relaxed lattice, so 'PBE-backed' differs by model partly through
    the lattice, not only through the coordinate."""
    pairs = {"pbe_vs_gt": ("pbe_backed_stable", "gt_stable"),
             "mlip_vs_gt": ("mlip_pred_stable_ledger", "gt_stable"),
             "pbe_vs_mlip": ("pbe_backed_stable", "mlip_pred_stable_ledger")}
    rows = [r for r in unit_calls if r["mlip_pred_stable_ledger"] is not None]
    out = {"note": "descriptive; clustered by system; ladder T only", "n_rows": len(rows)}
    for label, keep in (("all_models", lambda r: True),
                        ("without_orb_v2", lambda r: r["model"] != "orb_v2")):
        sub = [r for r in rows if keep(r)]
        block = {}
        for name, (a, b) in pairs.items():
            per = {}
            for s in sorted({r["system"] for r in sub}):
                rs = [r for r in sub if r["system"] == s]
                per[s] = {"agree": sum(bool(r[a]) == bool(r[b]) for r in rs), "n": len(rs)}
            block[name] = {"per_system": per,
                           "pooled_agree": sum(v["agree"] for v in per.values()),
                           "pooled_n": sum(v["n"] for v in per.values()),
                           "systems_all_agree": sum(v["agree"] == v["n"] for v in per.values()),
                           "n_systems": len(per)}
        out[label] = block
    return out


def stage_analyze(args):
    from mlip_dynstab.cli import _finite_t_gt
    out = Path(args.out_root)
    prov = provenance(args, "analyze")
    flags = []

    # ---- jobs and their pw.x results
    # A job serves a geometry frame only while that frame (in the job's reduced cell, if any)
    # still hashes to the structure the job was written for. Re-selected C3b configs or a
    # regenerated path would otherwise leave old jobs pointing at frames that now hold other
    # structures, and whichever job.json sorted last would silently win.
    jobs, member_of, geom_cache = {}, {}, {}
    for jj in sorted((out / "qe").glob("*/job.json")):
        m = _load(jj)
        jobs[m["job"]] = m
        for mem in m["members"]:
            gf, k = out / mem["geom_file"], int(mem["frame"])
            if gf not in geom_cache:
                geom_cache[gf] = _read_frames(gf) if gf.exists() else []
            if k >= len(geom_cache[gf]):
                flags.append(f"{m['job']}: member {mem['geom_file']}#{k} no longer exists")
                continue
            a, red = geom_cache[gf][k], mem.get("reduction")
            if red:
                r_ = _reduce_frame(a, np.array(red["lattice"]), red["reps"])
                a = r_[0] if r_ is not None else None
            if a is None or _structure_sha(a) != m["structure_sha"]:
                flags.append(f"{m['job']}: {mem['geom_file']}#{k} now holds a different "
                             f"structure; this job no longer serves it")
                continue
            member_of[(mem["geom_file"], k)] = (m["job"], mem)
    served = {}                                   # job -> first member it still serves
    for (gfile, k), (jname, mem) in member_of.items():
        served.setdefault(jname, mem)
    results, job_rows = {}, []
    for name, m in jobs.items():
        po = out / "qe" / name / "pw.out"
        row = {"job": name, "n_atoms": m["n_atoms"], "kmesh": "x".join(map(str, m["qe"]["kmesh"])),
               "nk_tr_upper": m["qe"]["nk_tr_upper"],
               "nk_irr_spglib": m["qe"].get("nk_irr_spglib"),
               "status": "pending" if name in served else "orphaned"}
        if po.exists():
            r = parse_pw_out(po)
            row.update({k: r.get(k) for k in ("qe_version", "nproc", "nk_irr", "wall_s",
                                              "n_scf")})
            # qe_queue.sh copies pw.in to pw.in.ran as it launches: an output whose input has
            # since changed (qe-inputs --force) answers another question.
            ran = out / "qe" / name / "pw.in.ran"
            pin = out / "qe" / name / "pw.in"
            row["input_matches_run"] = (None if not ran.exists()
                                        else ran.read_bytes() == pin.read_bytes())
            if row["input_matches_run"] is None:
                flags.append(f"{name}: no pw.in.ran, so the input that produced pw.out is "
                             f"unverified (kept)")
            if not r["job_done"]:
                row["status"] = "incomplete"
            elif row["input_matches_run"] is False:
                row["status"] = "stale_output"
                flags.append(f"{name}: pw.out was produced by a different pw.in, excluded")
            elif not r["converged"]:
                row["status"] = "scf_not_converged"
                flags.append(f"{name}: SCF not converged, excluded")
            elif "parse_error" in r:
                row["status"] = "parse_error"
                flags.append(f"{name}: ase could not parse pw.out ({r['parse_error']}), excluded")
            elif name not in served:
                row["status"] = "orphaned"
            else:
                mem = served[name]
                gf = out / mem["geom_file"]
                if gf not in geom_cache:
                    geom_cache[gf] = _read_frames(gf)
                frame = geom_cache[gf][int(mem["frame"])]
                red = mem.get("reduction")
                if red:
                    frame = _reduce_frame(frame, np.array(red["lattice"]), red["reps"])[0]
                mis = _pos_mismatch(r["atoms"], frame)
                row["pos_mismatch_A"] = mis
                if mis > args.pos_tol:
                    row["status"] = "structure_mismatch"
                    flags.append(f"{name}: pw.out structure differs from its geometry by "
                                 f"{mis:.2e} A, excluded")
                else:
                    row["status"] = "ok"
                    results[name] = r
            if row.get("wall_s") and row.get("nproc"):
                row["core_h"] = row["wall_s"] * row["nproc"] / 3600.0
        job_rows.append(row)

    def dft(geom_rel, frame):
        """PBE energy/forces of one geometry frame, carried back from its reduced cell (energy
        x images; each supercell atom takes its representative's force)."""
        hit = member_of.get((geom_rel, frame))
        if not hit or hit[0] not in results:
            return None
        r, red = results[hit[0]], hit[1].get("reduction")
        if not red:
            return r
        return {"energy_eV": r["energy_eV"] * red["n_images"],
                "forces": r["forces"][np.asarray(red["map"])]}

    def dft_settings(geom_rel, frame):
        """What must be identical for two pw.x energies to be subtracted: k-mesh, cutoffs,
        smearing, pseudopotentials and the cell pw.x ran in. Energies are only ever differenced
        within a path or against the unit's own reference, and a re-run of qe-inputs with other
        settings can leave jobs of both kinds side by side."""
        hit = member_of.get((geom_rel, frame))
        if not hit:
            return None
        q, red = jobs[hit[0]]["qe"], hit[1].get("reduction")
        return json.dumps([q["kmesh"], q["ecutwfc_Ry"], q["ecutrho_Ry"], q["smearing"],
                           q["degauss_Ry"], sorted(v["filename"] for v in q["pseudos"].values()),
                           None if not red else np.round(red["lattice"], 6).tolist()])

    # ---- C3a: E(Q) along the screen's coordinates
    c3a_paths, c3a_calls, c3a_curves = [], [], []
    for p in sorted((out / "geom" / "patterns").glob("*.json")):
        pat = _load(p)
        gf = out / pat["geom_file"]
        if not gf.exists():
            continue
        system, pm = pat["system"], pat["path_model"]
        frames = geom_cache.get(gf) or _read_frames(gf)
        Qs = np.array([float(a.info["Q"]) for a in frames])
        have = [i for i in range(len(frames)) if dft(pat["geom_file"], i) is not None]
        entry = {"stem": pat["stem"], "system": system, "path_model": pm,
                 "pattern_model": pat["pattern_model"], "role": pat["role"],
                 "q": pat["q"], "band": pat["band"], "dim": pat["dim"],
                 "n_atoms": pat["n_atoms"], "m_eff": pat["m_eff"],
                 "decide_T": pat.get("decide_T", []),
                 "cache_check_pass": pat["check"].get("pass"),
                 "cache_check_max_diff_meV": pat["check"].get("max_abs_diff_meV"),
                 "cache_check_window_diff_meV": pat["check"].get("max_abs_diff_window_meV"),
                 "n_Q": len(frames), "n_Q_dft": len(have)}
        if 0 not in have or len(have) < 4:
            entry["status"] = "pending" if not have else "insufficient_dft"
            c3a_paths.append(entry)
            continue
        if len({dft_settings(pat["geom_file"], i) for i in have}) > 1:
            entry["status"] = "mixed_settings"
            flags.append(f"{pat['stem']}: its pw.x jobs differ in k-mesh/cutoffs/cell, so "
                         f"E(Q)-E(0) is not defined; excluded (qe-inputs --force redoes it)")
            c3a_paths.append(entry)
            continue
        entry["status"] = "complete" if len(have) == len(frames) else "partial"
        idx = np.array(have)
        Ep = np.array([dft(pat["geom_file"], i)["energy_eV"] for i in have])
        dEp = Ep - Ep[0]
        Qd = Qs[idx]
        curves = {"pbe": dEp}
        for m, d in _load_mlip(out, gf, flags).items():
            curves[m] = d["E"][idx] - d["E"][idx][0]
        win = _fit_window(dEp)
        metrics = {}
        for name, dE in curves.items():
            mt = _curve_metrics(Qd, dE, pat["m_eff"])
            if name != "pbe":
                diff = (dE - dEp) * 1000
                mt.update({"rmse_vs_pbe_window_meV": float(np.sqrt(np.mean(diff[win] ** 2))),
                           "max_abs_vs_pbe_meV": float(np.abs(diff).max()),
                           "max_abs_vs_pbe_meV_per_atom": float(np.abs(diff).max()
                                                                / pat["n_atoms"])})
            metrics[name] = mt
            c3a_curves.append({"stem": pat["stem"], "curve": name,
                               "Q_A": Qd.tolist(), "dE_meV": (dE * 1000).tolist()})
        entry["curves"] = metrics
        if pat.get("cache"):
            cc = pat["cache"]
            # the cached map's own m_eff: when the regenerated pattern missed the cache its
            # m_eff differs (CHGNet BaTiO3 Gamma locally: 41.7 vs 64.8 amu), and the cached fit
            # solved with the wrong mass would not reproduce the ledger's call
            entry["cache_fit"] = {"a": cc["a"], "b": cc["b"], "c": cc["c"],
                                  "m_eff": cc["m_eff"], "depth_meV": cc["well_depth_meV"]}
        c3a_paths.append(entry)

    # Screen calls on every fitted curve (PBE, each MLIP, the cached production fit) at every
    # T, solved in one batch: each solve is ~1 s and there are hundreds.
    tasks, keys = [], []
    for e in c3a_paths:
        if "curves" not in e:
            continue
        fits = dict(e["curves"])
        if e.get("cache_fit"):
            fits["cache"] = e["cache_fit"]
        for T in T_SCREEN:
            for name, mt in fits.items():
                tasks.append((mt["a"], mt["b"], mt["c"], mt.get("m_eff", e["m_eff"]), T))
                keys.append((e["stem"], T, name))
    solved = dict(zip(keys, _solve_all(tasks, args.workers)))
    for e in c3a_paths:
        if "curves" not in e:
            continue
        spec = get_spec(e["system"])
        pm = e["path_model"]
        ladder = {float(t) for t in _ledger_rows(e["system"], pm)["temperature_K"]}
        names = list(e["curves"]) + (["cache"] if e.get("cache_fit") else [])
        e["calls"] = []
        for T in T_SCREEN:
            row = {"stem": e["stem"], "system": e["system"], "path_model": pm,
                   "role": e["role"], "T": T, "in_ladder": T in ladder,
                   "gt_stable": bool(_finite_t_gt(spec, T)),
                   "ledger_unit_pred_stable": _ledger_call(e["system"], pm, T)}
            for name in names:
                cl = solved[(e["stem"], T, name)]
                row.update({f"{name}_stable": cl["stable"], f"{name}_Q0_A": cl["Q0_A"],
                            f"{name}_curv_thz": cl["curv_thz"]})
            # Self-check of this pipeline against the paper: where this path decided the
            # ledger's call, the cached fit re-solved here must give the ledger's verdict.
            if e.get("cache_fit") and T in e.get("decide_T", []):
                row["cache_reproduces_ledger"] = (row["cache_stable"]
                                                  == row["ledger_unit_pred_stable"])
                if not row["cache_reproduces_ledger"]:
                    flags.append(f"{e['stem']} T={T:g}: cached fit re-solved gives "
                                 f"stable={row['cache_stable']}, ledger says "
                                 f"{row['ledger_unit_pred_stable']}")
            e["calls"].append(row)
        c3a_calls.extend(e["calls"])

    # Unit-level PBE-backed call: the screen's rule (unstable if ANY mapped mode condenses),
    # applied to the modes that have PBE. Coverage says how much of the screen that is.
    unit_calls = []
    for system in SYSTEMS:
        spec = get_spec(system)
        for model in MODELS:
            own = [e for e in c3a_paths if e["system"] == system and e["path_model"] == model
                   and e.get("status") in ("complete", "partial")]
            if not own:
                continue
            n_screened = int(load_cache(system, model)[1]["n_screened"])
            for T in T_SCREEN:
                crow = [c for e in own for c in e["calls"] if c["T"] == T]
                same = [c.get(f"{model}_stable") for c in crow]
                unit_calls.append({
                    "system": system, "model": model, "T": T,
                    "gt_stable": bool(_finite_t_gt(spec, T)),
                    "mlip_pred_stable_ledger": _ledger_call(system, model, T),
                    "pbe_backed_stable": all(c["pbe_stable"] for c in crow),
                    # the model's own call re-fitted on the PBE-mapped Q points (None if this
                    # model's energies on those structures are missing)
                    "mlip_same_paths_stable": None if None in same else all(same),
                    "n_paths_pbe": len(own), "n_screened": n_screened,
                    "kind": "reference_coordinate" if all(e["role"] == "ref" for e in own)
                    else "own_modes",
                    # does the mode that decided the ledger's call at this T have PBE?
                    "deciding_mode_has_pbe": any(T in e.get("decide_T", []) for e in own),
                    # a path whose regenerated map missed the cache is not the screen's own
                    # coordinate; the unit call then rests partly on a substitute
                    "all_paths_match_cache": all(e.get("cache_check_pass") in (True, None)
                                                 for e in own)})
    c3a_agreement = _agreement(unit_calls)

    # ---- C3b: forces and energies on SSCHA configs vs a near-equilibrium baseline.
    # Energy errors are taken relative to the undisplaced supercell on the same cell, in each
    # code separately; the reference is <system>_<owner>_ref.extxyz frame 0, baselines 1..n.
    c3b_cfg, c3b_units, mixed = [], [], set()

    def score(frames, grel, first, ml, ref_ml, ref_dft, ref_set, system, owner, T, label):
        for m in MODELS:
            if m not in ml:
                continue
            per = []
            for k, a in enumerate(frames[first:], start=first):
                r = dft(grel, k)
                if r is None:
                    continue
                Fp = r["forces"]
                dF = ml[m]["F"][k] - Fp
                row = {"system": system, "owner_model": owner, "T": T, "set": label,
                       "eval_model": m, "is_owner": m == owner, "seed": a.info.get("seed", ""),
                       "index": int(a.info["index"]), "n_atoms": len(a),
                       "u_rms_A": float(a.info.get("u_rms_A", np.nan)),
                       "force_rmse": float(np.sqrt(np.mean(dF ** 2))),
                       "force_mae": float(np.mean(np.abs(dF))),
                       "rms_f_pbe": float(np.sqrt(np.mean(Fp ** 2))),
                       "_ss": float(np.sum(dF ** 2)), "_sp": float(np.sum(Fp ** 2)),
                       "_n": dF.size}
                same_set = dft_settings(grel, k) == ref_set
                if ref_dft is not None and not same_set and (grel, k) not in mixed:
                    mixed.add((grel, k))
                    flags.append(f"{grel}#{k}: pw.x settings differ from the unit's "
                                 f"undisplaced reference; no energy error for it")
                if ref_dft is not None and same_set and m in ref_ml:
                    dEp = r["energy_eV"] - ref_dft["energy_eV"]
                    dEm = ml[m]["E"][k] - ref_ml[m]["E"][0]
                    row["e_err_meV_per_atom"] = float((dEm - dEp) / len(a) * 1000)
                    row["dE_pbe_meV_per_atom"] = float(dEp / len(a) * 1000)
                per.append(row)
            if not per:
                continue
            c3b_cfg.extend(per)
            ss, sp = sum(x["_ss"] for x in per), sum(x["_sp"] for x in per)
            nc = sum(x["_n"] for x in per)
            ee = [x["e_err_meV_per_atom"] for x in per if "e_err_meV_per_atom" in x]
            c3b_units.append({
                "system": system, "owner_model": owner, "T": T, "set": label,
                "eval_model": m, "is_owner": m == owner, "n_configs": len(per),
                "u_rms_mean_A": float(np.nanmean([x["u_rms_A"] for x in per])),
                "force_rmse": float(np.sqrt(ss / nc)),
                "force_mae": float(np.mean([x["force_mae"] for x in per])),
                "rms_f_pbe": float(np.sqrt(sp / nc)),
                "rel_force_rmse": float(np.sqrt(ss / sp)) if sp > 0 else None,
                "e_err_mean_meV_per_atom": float(np.mean(ee)) if ee else None,
                "e_err_rms_meV_per_atom": float(np.sqrt(np.mean(np.square(ee)))) if ee else None,
                "e_err_maxabs_meV_per_atom": float(np.max(np.abs(ee))) if ee else None,
                "energy_reference_has_pbe": ref_dft is not None})

    refs = {}
    for gf in sorted((out / "geom_c3b").glob("*_ref.extxyz")):
        frames = _read_frames(gf)
        system, owner = frames[0].info["system"], frames[0].info["owner_model"]
        grel = _rel(gf, out)
        ml = _load_mlip(out, gf, flags)
        refs[(system, owner)] = (ml, dft(grel, 0), dft_settings(grel, 0))
        score(frames, grel, 1, ml, ml, dft(grel, 0), dft_settings(grel, 0), system, owner,
              None, "baseline")
    for gf in sorted((out / "geom_c3b").glob("*.extxyz")):
        if gf.name.endswith((".tmp.extxyz", "_ref.extxyz")):
            continue
        frames = _read_frames(gf)
        system, owner = frames[0].info["system"], frames[0].info["owner_model"]
        ref_ml, ref_dft, ref_set = refs.get((system, owner), ({}, None, None))
        score(frames, _rel(gf, out), 0, _load_mlip(out, gf, flags), ref_ml, ref_dft, ref_set,
              system, owner, float(frames[0].info["T"]), "sscha")
    for u in c3b_units:
        base = next((b for b in c3b_units if b["set"] == "baseline" and u["set"] == "sscha"
                     and (b["system"], b["owner_model"], b["eval_model"])
                     == (u["system"], u["owner_model"], u["eval_model"])), None)
        if base and base["force_rmse"] > 0:
            u["force_rmse_over_baseline"] = u["force_rmse"] / base["force_rmse"]
            u["rel_force_rmse_baseline"] = base["rel_force_rmse"]

    # ---- cost: measured c0 and the remaining work at that rate
    meas = [r for r in job_rows if r.get("core_h") and r.get("nk_irr")]
    c0 = [r["core_h"] / cost_core_h(r["n_atoms"], r["nk_irr"], 1.0) for r in meas]
    c0_med = float(np.median(c0)) if c0 else None
    pend = [r for r in job_rows if r["status"] in ("pending", "incomplete", "stale_output")]
    cost = {"n_jobs": len(job_rows),
            "qe_versions": sorted({r["qe_version"] for r in job_rows if r.get("qe_version")}),
            "by_status": {s: sum(1 for r in job_rows if r["status"] == s)
                          for s in sorted({r["status"] for r in job_rows})},
            "core_h_spent": float(sum(r.get("core_h") or 0 for r in job_rows)),
            "c0_core_s_measured_median": c0_med, "n_measured": len(meas),
            "core_h_remaining_model": (float(sum(
                cost_core_h(r["n_atoms"], r["nk_irr_spglib"] or r["nk_tr_upper"], c0_med)
                for r in pend)) if c0_med else None)}

    summary = {"provenance": prov, "flags": flags, "cost": cost,
               "c3a": {"paths": c3a_paths, "unit_calls": unit_calls,
                       "agreement": c3a_agreement},
               "c3b": {"units": c3b_units},
               "conventions": {
                   "energies": "every energy is relative to the same structure set's own "
                               "reference in the same code (Q=0 of the path, or the undisplaced "
                               "supercell); absolute energies are never compared across codes",
                   "path_energy_normalisation": "per modulated cell, as the screen's E(Q) maps",
                   "force_rmse": "over all Cartesian components of all configs, eV/A",
                   "rel_force_rmse": "RMSE / RMS PBE force component",
                   "pbe_energy": "QE '!' total energy (smeared free energy, consistent with "
                                 "the forces)",
                   "gt": "cli._finite_t_gt: stable iff T >= spec transition_T_K"}}
    _dump(out / "summary.json", summary)

    def write_csv(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys and not isinstance(r[k], (dict, list)):
                    keys.append(k)
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    flat = []
    for e in c3a_paths:
        for name, mt in (e.get("curves") or {}).items():
            flat.append({**{k: e[k] for k in ("stem", "system", "path_model", "pattern_model",
                                              "role", "band", "n_atoms", "m_eff", "status",
                                              "n_Q_dft", "cache_check_pass",
                                              "cache_check_max_diff_meV",
                                              "cache_check_window_diff_meV")},
                         "q": ",".join(_frac(x) for x in e["q"]), "curve": name, **mt})
    write_csv("c3a_paths.csv", flat)
    write_csv("c3a_calls.csv", c3a_calls)
    write_csv("c3a_unit_calls.csv", unit_calls)
    write_csv("c3b_units.csv", c3b_units)
    write_csv("c3b_configs.csv", [{k: v for k, v in r.items() if not k.startswith("_")}
                                  for r in c3b_cfg])
    write_csv("qe_jobs.csv", job_rows)
    _dump(out / "c3a_curves.json", {"provenance": prov, "curves": c3a_curves})

    done = sum(1 for e in c3a_paths if e.get("status") in ("complete", "partial"))
    print(f"[analyze] jobs {cost['by_status']}; C3a paths with PBE {done}/{len(c3a_paths)}; "
          f"C3b unit rows {len(c3b_units)}; flags {len(flags)}")
    for f in flags[:20]:
        print(f"  FLAG {f}")
    if c0_med:
        print(f"[analyze] measured c0 {c0_med:.2f} core-s per k-point (5-atom equiv.); "
              f"remaining ~{cost['core_h_remaining_model']:.0f} core-h at that rate")


# ------------------------------------------------------------------------- CLI ----

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p, model=False):
        p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
        p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
        if model:
            p.add_argument("--model", required=True, choices=list(MODELS))
            p.add_argument("--device", default="cuda")
        p.add_argument("--force", action="store_true", help="redo work whose output exists")

    def qe_opts(p):
        p.add_argument("--sssp-dir", default=None,
                       help="directory holding the SSSP 1.3 PBE efficiency json (and UPFs)")
        p.add_argument("--sssp-json", default=None)
        p.add_argument("--kspacing-insulator", type=float, default=0.15,
                       help="1/A, 2*pi included (see KSPACING_CONVENTION)")
        p.add_argument("--kspacing-metal", type=float, default=0.10)
        p.add_argument("--degauss-insulator", type=float, default=0.005, help="Ry")
        p.add_argument("--degauss-metal", type=float, default=0.02, help="Ry, Marzari-Vanderbilt")
        p.add_argument("--c0-core-s", type=float, default=8.0,
                       help="ASSUMED core-s per k-point per SCF (5-atom cell) for the cost print")

    p = sub.add_parser("plan", help="C3a/C3b job list and cost from the caches (no MLIP)")
    common(p)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--extra-max-atoms", type=int, default=12)
    p.add_argument("--kspacing-insulator", type=float, default=0.15)
    p.add_argument("--kspacing-metal", type=float, default=0.10)
    p.add_argument("--per-unit", type=int, default=12)
    p.add_argument("--n-baseline", type=int, default=4)
    p.add_argument("--c0-core-s", type=float, default=8.0,
                   help="ASSUMED core-seconds per k-point per SCF for a 5-atom cell")
    p.add_argument("--no-reduce-cell", action="store_true")
    p.set_defaults(func=stage_plan)

    p = sub.add_parser("geom", help="regenerate + verify the screen's paths (model env)")
    common(p, model=True)
    p.add_argument("--no-recover", action="store_true",
                   help="do not search the degenerate span when a map fails its check")
    p.add_argument("--recover-max-atoms", type=int, default=12,
                   help="search only cells this small (deciding modes are always searched). "
                        "Default = qe-inputs' --extra-max-atoms: a larger non-deciding path "
                        "never gets PBE, while MatterSim bcc-Zr (run first, under a 3 h stage "
                        "cap) screens 24 modes and each degenerate one costs thousands of "
                        "single points to search")
    p.add_argument("--harm-tol", type=float, default=0.02,
                   help="THz; a larger regenerated-vs-cached harmonic difference is reported "
                        "as drift (the degenerate span is still searched)")
    p.set_defaults(func=stage_geom)

    p = sub.add_parser("mlip-eval", help="this model on every structure (model env)")
    common(p, model=True)
    p.set_defaults(func=stage_mlip_eval)

    p = sub.add_parser("qe-inputs", help="pw.x inputs for the C3a paths")
    common(p)
    qe_opts(p)
    p.add_argument("--extra-max-atoms", type=int, default=12,
                   help="non-deciding modes get PBE only if their cell is this small")
    p.add_argument("--only-decide", action="store_true",
                   help="deciding and reference paths only")
    p.add_argument("--no-reduce-cell", action="store_true",
                   help="run pw.x on the full modulated supercell, not its smallest periodic "
                        "cell")
    p.set_defaults(func=stage_qe_inputs)

    p = sub.add_parser("c3b-inputs", help="SSCHA configs + baseline -> structures + pw.x inputs")
    common(p)
    qe_opts(p)
    p.add_argument("--sscha-dir", type=Path, default=SSCHA_DIR_DEFAULT)
    p.add_argument("--per-unit", type=int, default=12)
    p.add_argument("--n-baseline", type=int, default=4)
    p.add_argument("--rattle-stdev", type=float, default=0.02)
    p.add_argument("--rattle-seed", type=int, default=20260926)
    p.set_defaults(func=stage_c3b_inputs)

    p = sub.add_parser("analyze", help="PBE vs MLIP (local)")
    common(p)
    p.add_argument("--pos-tol", type=float, default=1e-3,
                   help="max A between the pw.out structure and its geometry")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="processes for the screen solves")
    p.set_defaults(func=stage_analyze)

    # Numerical / functional / lattice checks on the C3a curves (conv-inputs, xc-inputs,
    # pbe-lattice, analyze-checks) live in scripts/dft_checks.py. A failure to load that module
    # must not take the production stages down.
    sys.modules.setdefault("dft_reference", sys.modules[__name__])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import dft_checks
        dft_checks.register(sub)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[dft_reference] check stages unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    args = ap.parse_args(argv)
    args.out_root = Path(args.out_root)
    rc = args.func(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
