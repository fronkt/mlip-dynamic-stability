"""POST HOC (R2 major revision, not pre-registered): lattice sweep of the single-mode soft-mode screen.

Referee 2 asks why a sub-percent lattice change moves the perovskite soft-mode well so much, and
whether each MLIP's volume bias is systematic. The paper has PBE wells at two lattices only (PBE's
own and MACE's), which confound the lattice with the eigenvector, and same-model lattice
diagnostics for the MACE fine-tunes (scripts/finetune_lattice_diag.py). This script gives one base
model a CURVE per system: well depth, quartic fit coefficients and the finite-T screen call versus
the lattice parameter, at fixed internal (cubic) positions.

Per system (one pinned model env per run):
 a. Relax the cubic cell at P = 0 with the production relaxation (harmonic._relax: FIRE +
    FrechetCellFilter, fmax 1e-3, 300 steps) from the registry prototype -> a0 = V**(1/3), with
    the residual pressure and force recorded.
 b. At a = a0 (1 + s/100) for s in S_PCT, and at the PBE lattice where one exists
    (results/revision/dft_checks/pl_lattice.csv), relaxed fractional positions held fixed and the
    cell scaled isotropically:
      - pressure from the calculator stress;
      - the production screen, verbatim: compute_finite_t_softmode(relax=False) on the scaled
        cell (2x2x2 force constants at 0.01 A, every distinct imaginary commensurate mode up to 24,
        10-point E(Q), _fit_double_well, single-mode quantum SCHA at each T; the phase is unstable
        if ANY mode condenses), with n_curv_blind and a signed SCHA margin (below);
      - for every mode the s = 0 screen maps ("tracked" modes), two depth measures:
          own   = that mode AT THIS LATTICE: the s = 0 pattern projected (mass-weighted) onto the
                  degenerate eigenspace at the same q that it overlaps most. Projecting fixes the
                  arbitrary basis phonopy returns inside a degenerate band (Gamma T1u, X doublets),
                  so own - fixed is the physical eigenvector change, not a basis rotation;
                  ``overlap`` records the cosine with the s = 0 pattern.
          fixed = the s = 0 pattern itself, in the s = 0 modulated cell scaled with the lattice.
        Lattice and eigenvector effects are then separable: fixed varies only the lattice.
      - the softest screened mode at this lattice (q, band recorded), as the screen maps it.
      Depths, meV per modulated cell: ``depth_sampled`` (production: -min over the 10 amplitudes),
      ``depth_fit`` (minimum of the fitted kappa2 Q^2 + kappa4 Q^4 + kappa6 Q^6) and
      ``depth_refined`` (bounded Brent minimisation of E(Q) inside the sampled bracket; the
      headline, because the sampled minimum jumps between grid points as the lattice moves).
 c. Derived: d ln(depth)/d ln a by central differences (s = +-0.25 and +-0.5) and by a log-linear
    fit over s in [-0.5, +0.5]; the same for |kappa2|, kappa4 and |omega_harm|; the two-point slope
    between a0 and the PBE lattice; and per temperature the lattice a* where the call flips: every
    bracket whose call differs is bisected with screen-only evaluations (``flip_screens``) to
    --flip-tol-pct of a0 (default 0.02 %), and a* is the final bracket's midpoint, quoted as %
    from a0 and from the PBE lattice.

SCHA margin (recorded, not used for a*): for each screened mode, min F(Q0) over the displaced
centroids minus min F(Q0) over the centroids the production solver counts as symmetric
(Q0 <= 1.5 grid steps of the _solve_scha grid), in meV; >= 0 exactly when the production argmin
rule says stable, and the phase margin is the minimum over modes. Agreement with the production
call is checked at every point (``margin_call_mismatch``). It is not interpolated: on the stable
side it saturates at the curvature shoulder (~+0.3 meV) while condensation makes it fall steeply
on the unstable side, so a linear interpolation is biased toward the stable end.

Reference mode (the headline series): the s = 0 mode that decides the most temperatures (ties: the
highest T, then the deeper well), the same rule as dft_checks._model_deciding_path.

Precision: ``--precision auto`` (default) forces float64 where the calculator allows it. MACE-MP-0
is float64 in production already; ORB is loaded with precision="float64" (production used the
loader default) when the loader accepts it; CHGNet hard-codes float32 (chgnet.graph TORCH_DTYPE)
and SevenNet/MatterSim expose no dtype switch, so they run as in production. ``--precision
production`` loads every model exactly as mlip_dynstab.calculators.get_calculator does. The
parameter dtype and the returned force dtype are recorded either way.

At s = 0 the run is compared with the production ledger (results/ledger.jsonl, softmode at the
current METHOD_VERSION) and, when present, the production E(Q) cache (read only); see
``ledger_check``. Nothing is written to the ledger or results/cache: the E(Q) maps of this sweep
live in a throw-away temp dir and in the output JSON.

    python scripts/lattice_sweep.py --model mace_mp0 --out results/revision/r2_lattice/mace_mp0.json \
        [--device cuda|cpu] [--systems batio3_cubic,knbo3_cubic,srtio3_cubic] \
        [--s-pct=-1.5,-1,-0.75,-0.5,-0.25,0,0.25,0.5,1] [--temps 100,300,600] \
        [--precision auto|production] [--no-pbe] [--flip-tol-pct 0.02] [--threads N] [--dry-run]

Resumable: the JSON is rewritten atomically after every lattice point; a rerun skips finished
points and reuses the stored relaxed cell and s = 0 patterns, and refuses to mix settings,
precision or dry-run state. s = 0 is always run (it defines the tracked patterns). ``--dry-run``
uses a Lennard-Jones stand-in instead of the model, to test the plumbing in seconds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

SCHEMA = "r2_lattice_sweep/1"
SCRIPT = "scripts/lattice_sweep.py"
SYSTEMS = ("batio3_cubic", "knbo3_cubic", "srtio3_cubic")
S_PCT = (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
TEMPS = (100.0, 300.0, 600.0)
SUPERCELL = (2, 2, 2)          # production FC cell for the perovskites (cli.run_unit)
MAX_MODES = 24                 # production cap (cli.run_unit default)
DEG_TOL_THZ = 5e-3             # bands closer than this at one q are one degenerate set
BRENT_XATOL = 1e-5             # A, depth refinement
FIT_WINDOW = (-0.5, 0.5)       # % window of the log-linear slope fit
PBE_CSV = REPO / "results" / "revision" / "dft_checks" / "pl_lattice.csv"
PBE_SOURCE = "PBE vc-relax (this check)"
LEDGER = REPO / "results" / "ledger.jsonl"
MODEL_ENV = {"mace_mp0": "mace", "chgnet": "chgnet", "orb_v2": "orb", "orb_v3": "orb",
             "sevennet0": "sevennet", "mattersim": "mattersim"}
MODEL_PKG = {"mace_mp0": "mace-torch", "chgnet": "chgnet", "orb_v2": "orb-models",
             "orb_v3": "orb-models", "sevennet0": "sevenn", "mattersim": "mattersim"}
PKGS = ("numpy", "scipy", "ase", "phonopy", "torch", "e3nn", "pymatgen", "torch-geometric",
        "pyyaml")
# s = 0 vs production ledger: what counts as a reproduction
TOL = {"a_A": 5e-4, "harm_thz": 0.02, "depth_rel": 0.02, "depth_abs_meV": 0.5,
       "curv_thz": 0.02}


# --------------------------------------------------------------------------- utilities ----

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
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


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label(s_pct: float) -> str:
    return f"s{s_pct:+.2f}"


def _fin(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _atoms_rec(at) -> dict:
    return {"symbols": at.get_chemical_symbols(), "cell": at.cell.array.tolist(),
            "scaled_positions": at.get_scaled_positions(wrap=False).tolist()}


def _atoms_from(rec):
    import ase
    return ase.Atoms(symbols=rec["symbols"], cell=rec["cell"],
                     scaled_positions=rec["scaled_positions"], pbc=True)


def _scaled(at, factor: float):
    x = at.copy()
    x.calc = None
    x.set_cell(at.cell.array * factor, scale_atoms=True)
    return x


def _a_of(at) -> float:
    return float(np.cbrt(at.get_volume()))


def _qkey(q):
    return tuple(sorted(round(abs(c), 6) for c in q))


def production_params() -> dict:
    """Read the production defaults off the functions themselves so they cannot drift."""
    from mlip_dynstab.finite_t import compute_finite_t_softmode, _solve_scha
    from mlip_dynstab.harmonic import _relax
    from mlip_dynstab import METHOD_VERSION, SOFTMODE_EQMAP_VERSION
    p = inspect.signature(compute_finite_t_softmode).parameters
    s = inspect.signature(_solve_scha).parameters
    r = inspect.signature(_relax).parameters
    return {"supercell": list(SUPERCELL), "max_modes": MAX_MODES, "disp_A": p["disp"].default,
            "q_max_A": p["q_max"].default, "n_pts": p["n_pts"].default,
            "imag_tol_thz": p["imag_tol_thz"].default, "relax_fmax": p["fmax"].default,
            "relax_steps": r["steps"].default, "scha_q_box": s["q_box"].default,
            "scha_nq": s["nq"].default, "softmode_method_version": METHOD_VERSION["softmode"],
            "softmode_eqmap_version": SOFTMODE_EQMAP_VERSION, "deg_tol_thz": DEG_TOL_THZ,
            "brent_xatol_A": BRENT_XATOL}


# --------------------------------------------------------------------- provenance ----

def _git() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    try:
        head = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"git_head": head or None, "git_dirty_tracked": dirty,
                "git_branch": run("rev-parse", "--abbrev-ref", "HEAD") or None}
    except Exception as exc:                       # no git on the box is not fatal
        return {"git_head": None, "git_error": repr(exc)}


def _versions(model: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in PKGS + ((MODEL_PKG[model],) if model in MODEL_PKG else ()):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _lock_check(model: str, versions: dict) -> dict:
    """Running env vs envs/lock-<env>-<latest date>.txt. Recorded, never fatal: the local check
    runs outside the pinned envs on purpose."""
    env = MODEL_ENV.get(model)
    locks = sorted((REPO / "envs").glob(f"lock-{env}-*.txt")) if env else []
    if not locks:
        return {"lock": None}
    pinned = {}
    for line in locks[-1].read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            k, v = line.split("==", 1)
            pinned[k.strip().lower()] = v.strip()
    mism = {p: {"running": v, "pinned": pinned[p.lower()]}
            for p, v in versions.items()
            if p.lower() in pinned and v is not None and v != pinned[p.lower()]}
    return {"lock": locks[-1].name, "mismatches": mism, "matches_lock": not mism}


def _torch_info(calc) -> dict:
    out = {}
    try:
        import torch
    except ImportError:
        return {"torch": None}
    out["torch_threads"] = torch.get_num_threads()
    out["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    seen = set()

    def walk(o, depth):
        if o is None or depth > 3 or id(o) in seen:
            return None
        seen.add(id(o))
        if isinstance(o, torch.nn.Module):
            for p in o.parameters():
                return str(p.dtype)
            return None
        if isinstance(o, (list, tuple)):
            for x in o:
                r = walk(x, depth + 1)
                if r:
                    return r
            return None
        for name in ("models", "model", "potential"):
            if hasattr(o, name):
                r = walk(getattr(o, name), depth + 1)
                if r:
                    return r
        return None

    out["param_dtype"] = walk(calc, 0)
    return out


def _checkpoint_id(model: str, calc, version: str) -> dict:
    out = {"version_string": version}
    try:
        if model == "mace_mp0":
            from mace.calculators.foundations_models import download_mace_mp_checkpoint
            ck = version.split("ckpt=")[-1].split(";")[0] if "ckpt=" in version else "medium"
            path = download_mace_mp_checkpoint(ck)       # cached: no download
            out.update(file=os.path.basename(path), sha256=_sha256(path))
        elif model == "chgnet":
            out["chgnet_model_version"] = getattr(calc.model, "version", None)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


# --------------------------------------------------------------------- calculator ----

def load_calculator(model: str, device: str, precision: str, dry_run: bool, atoms):
    """(calc, version, dtype_note). The production loader unless --precision auto can raise a
    model to float64 that production ran lower (ORB)."""
    if dry_run:
        from ase.calculators.lj import LennardJones
        d = atoms.get_all_distances(mic=True)
        sigma = float(np.min(d[d > 0])) / 2 ** (1 / 6)
        return (LennardJones(sigma=sigma, epsilon=0.1, rc=2.5 * sigma),
                f"LennardJones-dry-run(sigma={sigma:.4f})", "dry-run stand-in, float64")
    if precision == "auto" and model in ("orb_v2", "orb_v3"):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        from mlip_dynstab.calculators import _pkg_version
        loader = pretrained.orb_v2 if model == "orb_v2" else pretrained.orb_v3_conservative_inf_omat
        if "precision" not in inspect.signature(loader).parameters:
            raise SystemExit(f"{loader.__name__} takes no precision argument in this orb-models; "
                             "rerun with --precision production")
        net, adapter = loader(device=device, precision="float64")
        calc = ORBCalculator(net, adapter, device=device)
        return (calc, f"orb-models={_pkg_version('orb-models')};{model};precision=float64",
                "float64 forced via the orb loader (production: loader default float32-high)")
    from mlip_dynstab.calculators import get_calculator
    h = get_calculator(model, device=device)
    note = {"mace_mp0": "float64 (production default_dtype)",
            "chgnet": "float32: chgnet hard-codes TORCH_DTYPE, no float64 switch",
            "sevennet0": "production precision: no dtype switch in SevenNetCalculator",
            "mattersim": "production precision: no dtype switch in MatterSimCalculator"}
    return h.calc, h.version, note.get(model, "production loader")


# ------------------------------------------------------------------ physics helpers ----

def _pressure(at, calc) -> dict:
    from ase import units
    x = at.copy()
    x.calc = calc
    e = float(x.get_potential_energy())
    st = np.asarray(x.get_stress(voigt=True), float)
    return {"pressure_GPa": float(-st[:3].mean() / units.GPa),
            "stress_voigt_GPa": (st / units.GPa).tolist(), "energy_eV_per_cell": e,
            "n_atoms_cell": len(x)}


def _spectrum(ph, supercell):
    """Frequencies on the FC-commensurate q set, as the production q-search builds it."""
    n = [int(x) for x in supercell]
    qs = [[i / n[0], j / n[1], k / n[2]]
          for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]
    ph.run_qpoints(qs, with_eigenvectors=False)
    return qs, np.array(ph.qpoints.frequencies)


def _q_index(qs, q) -> int:
    for i, qq in enumerate(qs):
        if max(abs(a - b) for a, b in zip(qq, q)) < 1e-9:
            return i
    raise ValueError(f"q {q} is not on the commensurate set")


def _softest_commensurate(qs, freqs):
    """(q, band, f) of the softest commensurate mode, acoustic at Gamma masked by |omega| (the
    production rule of _softest_mesh_mode / _imaginary_commensurate_modes)."""
    fr = freqs.copy()
    for qi, q in enumerate(qs):
        if max(abs(c) for c in q) < 1e-8:
            fr[qi, np.argsort(np.abs(fr[qi]))[:3]] = np.inf
    iq, ib = np.unravel_index(int(np.argmin(fr)), fr.shape)
    return list(qs[iq]), int(ib), float(freqs[iq, ib])


def _fit_meta(Qs, dE, a, b, c) -> dict:
    """Depth of the fitted polynomial, and the fit window size (mirrors _fit_double_well's
    window rule; recorded because a window or quartic/sextic switch moves kappa2/kappa4)."""
    Qs = np.asarray(Qs, float)
    dE = np.asarray(dE, float)
    emin = float(dE.min())
    window = max(0.06, 5.0 * abs(min(emin, 0.0)))
    n_fit = int(max((dE <= emin + window).sum(), 4))
    Q = np.linspace(0.0, float(Qs.max()), 9001)
    V = a * Q ** 2 + b * Q ** 4 + c * Q ** 6
    i = int(np.argmin(V))
    return {"fit_model": "sextic" if c != 0.0 else "quartic", "n_fit_pts": n_fit,
            "depth_fit_meV": float(-min(V[i], 0.0) * 1000), "Q_fit_A": float(Q[i])}


def _refine(base, calc, u, Qs, dE) -> dict:
    """Bounded Brent minimisation of E(Q) inside the bracket of the sampled minimum."""
    from scipy.optimize import minimize_scalar
    Qs = np.asarray(Qs, float)
    dE = np.asarray(dE, float)
    i = int(np.argmin(dE))
    if dE[i] >= 0.0:
        return {"depth_refined_meV": 0.0, "Q_refined_A": 0.0, "min_at_edge": False,
                "n_refine_evals": 0}
    if i == len(Qs) - 1:                 # minimum beyond q_max: the edge value is a lower bound
        return {"depth_refined_meV": float(-dE[i] * 1000), "Q_refined_A": float(Qs[i]),
                "min_at_edge": True, "n_refine_evals": 0}
    b0 = base.copy()
    b0.calc = calc
    E0 = b0.get_potential_energy()
    pos0 = b0.get_positions()

    def f(Q):
        x = base.copy()
        x.set_positions(pos0 + Q * u)
        x.calc = calc
        return x.get_potential_energy() - E0

    r = minimize_scalar(f, bounds=(float(Qs[i - 1]), float(Qs[i + 1])), method="bounded",
                        options={"xatol": BRENT_XATOL})
    better = float(r.fun) < float(dE[i])
    return {"depth_refined_meV": float(-min(float(r.fun), float(dE[i])) * 1000),
            "Q_refined_A": float(r.x if better else Qs[i]), "min_at_edge": False,
            "n_refine_evals": int(r.nfev) + 1}


def _well(base, calc, u, P) -> dict:
    """Production E(Q) sampling + fit along pattern u in cell base, plus the refined minimum."""
    from mlip_dynstab.finite_t import _sample_well, _fit_double_well
    Qs, dE = _sample_well(base, calc, u, P["q_max_A"], P["n_pts"])
    a, b, c = _fit_double_well(Qs, dE)
    i = int(np.argmin(dE))
    rec = {"kappa2": a, "kappa4": b, "kappa6": c,
           "depth_sampled_meV": float(-min(dE.min(), 0.0) * 1000), "Q_sampled_A": float(Qs[i]),
           **_fit_meta(Qs, dE, a, b, c), **_refine(base, calc, u, Qs, dE),
           "Qs": Qs.tolist(), "dE_eV": dE.tolist()}
    rec["n_energy_evals"] = len(Qs) + 1 + rec["n_refine_evals"]
    return rec


def _own_pattern(ph, q, u_ref, symbols_ref, max_den) -> dict:
    """The s = 0 pattern followed to this lattice: project u_ref (mass-weighted) onto each
    degenerate eigenspace at q and keep the one it overlaps most. Real and imaginary parts of the
    phonopy modulations span each set, so the arbitrary basis phonopy returns inside a degenerate
    band drops out."""
    import ase
    from fractions import Fraction
    ph.run_qpoints([list(q)], with_eigenvectors=False)
    f = np.array(ph.qpoints.frequencies)[0]
    nb = len(f)
    dim = [Fraction(x).limit_denominator(max_den).denominator if abs(x) > 1e-9 else 1 for x in q]
    ph.run_modulations(dimension=dim, phonon_modes=[[list(q), b, 1.0, 0.0] for b in range(nb)])
    mods, sc_ph = ph.get_modulations_and_supercell()
    base = ase.Atoms(symbols=sc_ph.symbols, scaled_positions=sc_ph.scaled_positions,
                     cell=sc_ph.cell, pbc=True)
    if base.get_chemical_symbols() != list(symbols_ref):
        raise RuntimeError(f"modulated cell at q={q} differs in atom order from the s=0 cell")
    w = np.repeat(np.sqrt(base.get_masses()), 3)
    x_ref = w * np.asarray(u_ref, float).ravel()
    order = np.argsort(f)
    groups, cur = [], [int(order[0])]
    for b in order[1:]:
        if abs(f[b] - f[cur[0]]) < DEG_TOL_THZ:
            cur.append(int(b))
        else:
            groups.append(cur)
            cur = [int(b)]
    groups.append(cur)
    best = None
    for g in groups:
        cols = []
        for b in g:
            m = np.asarray(mods[b]).ravel()
            cols += [w * m.real, w * m.imag]
        B = np.stack(cols, axis=1)
        U, s, _ = np.linalg.svd(B, full_matrices=False)
        U = U[:, s > 1e-8 * s.max()] if s.max() > 0 else U[:, :0]
        p = U @ (U.T @ x_ref)
        weight = float(p @ p / (x_ref @ x_ref))
        if best is None or weight > best[0]:
            best = (weight, g, p)
    weight, g, p = best
    u = (p / w).reshape(-1, 3)
    u = u / np.abs(u).max()
    if (w * u.ravel()) @ x_ref < 0:
        u = -u
    m_eff = float(np.sum(base.get_masses()[:, None] * u ** 2))
    return {"base": base, "u": u, "m_eff": m_eff, "bands": sorted(g),
            "freq_thz": float(np.mean(f[g])), "n_degenerate": len(g),
            "overlap_with_s0": float(math.sqrt(max(weight, 0.0)))}


def _scha_margins(modes, T, P):
    """Per-mode signed SCHA margin (meV): min F over displaced centroids - min F over the
    centroids _solve_scha counts as symmetric. >= 0 <=> the production argmin says stable."""
    from mlip_dynstab.finite_t import _scha_branch
    Q0s = np.linspace(0.0, P["scha_q_box"], P["scha_nq"])
    inner = Q0s <= 1.5 * (Q0s[1] - Q0s[0])
    out = []
    for e in modes:
        if e["band"] < 0:
            out.append(float("inf"))
            continue
        F = np.array([_scha_branch(q0, e["a"], e["b"], e["c"], e["m_eff"], T)[0] for q0 in Q0s])
        fi, fo = F[inner][np.isfinite(F[inner])], F[~inner][np.isfinite(F[~inner])]
        if fo.size == 0:
            out.append(float("inf"))
        elif fi.size == 0:
            out.append(float("-inf"))
        else:
            out.append(float((fo.min() - fi.min()) * 1000))
    return out


# ---------------------------------------------------------------------- one lattice ----

def run_screen(at, calc, temps, P, workdir: Path):
    """The production screen on this cell (relax=False), maps in a throw-away cache."""
    from mlip_dynstab.finite_t import compute_finite_t_softmode
    cache = workdir / "screen.json"
    if cache.exists():
        cache.unlink()
    calls, t_map, t_solve = {}, 0.0, 0.0
    maps = None
    for T in temps:
        t0 = time.time()
        res = compute_finite_t_softmode(at, calc, T, supercell=SUPERCELL,
                                        max_modes=P["max_modes"], relax=False,
                                        cache_path=str(cache))
        if maps is None:
            t_map = time.time() - t0
            maps = json.loads(cache.read_text())
        else:
            t_solve += time.time() - t0
        row = res.as_row()
        modes = maps["modes"]
        dec = [i for i, m in enumerate(modes)
               if m["harm_thz"] == row["ft_decide_harm_thz"]
               and ",".join(f"{x:.4f}" for x in m["q"]) == row["ft_decide_q"]]
        margins = _scha_margins(modes, T, P)
        stable_str = row["ft_modes_stable"].split(";")
        mism = sum(1 for mg, st in zip(margins, stable_str) if (mg >= 0) != (st == "1"))
        phase = min(margins)
        calls[str(int(T))] = {
            "T_K": float(T), "pred_stable": bool(res.dynamically_stable),
            "min_eff_freq_thz": float(row["min_eff_freq_thz"]),
            "n_condensed": int(row["ft_n_condensed"]), "n_curv_blind": int(row["ft_n_curv_blind"]),
            "decide_index": dec[0] if dec else None, "decide_q": row["ft_decide_q"],
            "decide_dim": row["ft_decide_dim"], "decide_harm_thz": float(row["ft_decide_harm_thz"]),
            "decide_well_depth_meV": float(row["ft_well_depth_meV"]),
            "order_param_Q_ang": float(row["ft_order_param_Q_ang"]),
            "modes_curv_thz": [float(x) for x in row["ft_modes_curv_thz"].split(";")],
            "modes_Q0_ang": [float(x) for x in row["ft_modes_Q0_ang"].split(";")],
            "modes_stable": [s == "1" for s in stable_str],
            "modes_margin_meV": margins, "phase_margin_meV": phase,
            "phase_margin_inf": not math.isfinite(phase),
            "margin_call_mismatch": mism + int((phase >= 0) != bool(res.dynamically_stable))}
    screen = {"n_imag_total": int(maps["n_imag_total"]), "n_screened": int(maps["n_screened"]),
              "modes": []}
    for i, m in enumerate(maps["modes"]):
        e = {"index": i, "q": m["q"], "band": m["band"], "dim": m["dim"], "m_eff": m["m_eff"],
             "harm_thz": m["harm_thz"], "kappa2": m["a"], "kappa4": m["b"], "kappa6": m["c"],
             "depth_sampled_meV": m["well_depth_meV"], "Qs": m["Qs"], "dE_eV": m["dE"]}
        if m["band"] >= 0:
            e.update(_fit_meta(m["Qs"], m["dE"], m["a"], m["b"], m["c"]))
        screen["modes"].append(e)
    return screen, maps, calls, t_map, t_solve


def build_refs(ph, maps, calls, P):
    """Tracked patterns from the s = 0 screen (every mapped mode) and the reference index."""
    from mlip_dynstab.finite_t import _mode_pattern
    max_den = max(SUPERCELL)
    modes = [m for m in maps["modes"] if m["band"] >= 0]
    placeholder = not modes
    if placeholder:                       # harmonically stable at s=0: follow the softest mode
        qs, freqs = _spectrum(ph, SUPERCELL)
        q, band, f0 = _softest_commensurate(qs, freqs)
        modes = [{"q": q, "band": band, "harm_thz": f0, "m_eff": None, "well_depth_meV": 0.0}]
    refs = []
    for k, m in enumerate(modes):
        dim, base, u, m_eff = _mode_pattern(ph, m["q"], m["band"], max_den)
        decide_T = [c["T_K"] for c in calls.values()
                    if not placeholder and c["decide_index"] == k]
        refs.append({"k": k, "q": list(m["q"]), "band": int(m["band"]), "dim": list(dim),
                     "qkey": list(_qkey(m["q"])), "harm_thz_s0": float(m["harm_thz"]),
                     "m_eff": m_eff, "m_eff_screen": m.get("m_eff"),
                     "depth_sampled_s0_meV": float(m["well_depth_meV"]), "decide_T": decide_T,
                     "base": _atoms_rec(base), "u": u.tolist()})
    key = lambda r: (len(r["decide_T"]), max(r["decide_T"] or [0.0]), r["depth_sampled_s0_meV"])
    ref_index = max(range(len(refs)), key=lambda i: key(refs[i]))
    return {"modes": refs, "reference_index": ref_index, "harmonically_stable_s0": placeholder,
            "rule": "decides most temperatures; ties: highest T, then deeper sampled well "
                    "(dft_checks._model_deciding_path)"}


def run_point(label, kind, s_pct, factor, relaxed, calc, refs, P, temps, workdir):
    from mlip_dynstab.finite_t import _harmonic_phonon, _mode_pattern
    tw = {}
    t_all = time.time()
    at = _scaled(relaxed, factor)
    t0 = time.time()
    pres = _pressure(at, calc)
    tw["pressure"] = time.time() - t0
    screen, maps, calls, tw["screen_maps"], tw["screen_solve"] = run_screen(at, calc, temps, P,
                                                                              workdir)
    t0 = time.time()
    ph = _harmonic_phonon(at, calc, SUPERCELL, P["disp_A"])
    tw["phonons"] = time.time() - t0
    n_fd = len(ph.supercells_with_displacements)
    new_refs = None
    if refs is None:                      # this is s = 0: the tracked patterns come from here
        new_refs = refs = build_refs(ph, maps, calls, P)
    qs, freqs = _spectrum(ph, SUPERCELL)
    max_den = max(SUPERCELL)
    t0 = time.time()
    tracked = []
    for r in refs["modes"]:
        u_ref = np.asarray(r["u"], float)
        base_ref = _atoms_from(r["base"])
        own = _own_pattern(ph, r["q"], u_ref, base_ref.get_chemical_symbols(), max_den)
        own_w = _well(own["base"], calc, own["u"], P)
        own_w.update({k: own[k] for k in ("bands", "freq_thz", "n_degenerate",
                                          "overlap_with_s0", "m_eff")})
        fixed_w = _well(_scaled(base_ref, factor), calc, u_ref, P)
        fixed_w["m_eff"] = r["m_eff"]
        tracked.append({"k": r["k"], "q": r["q"], "band": r["band"],
                        "harm_thz": float(freqs[_q_index(qs, r["q"]), r["band"]]),
                        "own": own_w, "fixed": fixed_w})
    tw["tracked"] = time.time() - t0
    t0 = time.time()
    m0 = screen["modes"][0]                # the screen sorts most-imaginary first
    soft = {"index": 0, "q": m0["q"], "band": m0["band"], "dim": m0["dim"],
            "harm_thz": m0["harm_thz"], "qkey": list(_qkey(m0["q"]))}
    if m0["band"] >= 0:
        dim, base, u, _ = _mode_pattern(ph, m0["q"], m0["band"], max_den)
        soft.update({k: m0[k] for k in ("kappa2", "kappa4", "kappa6", "depth_sampled_meV",
                                        "fit_model", "depth_fit_meV", "Q_fit_A")})
        soft.update(_refine(base, calc, u, m0["Qs"], m0["dE_eV"]))
    else:
        soft.update({"kappa2": 0.0, "kappa4": 0.0, "kappa6": 0.0, "depth_sampled_meV": 0.0,
                     "depth_fit_meV": 0.0, "depth_refined_meV": 0.0, "harmonically_stable": True})
    tw["softest_refine"] = time.time() - t0
    tw["total"] = time.time() - t_all
    n_e = sum(t["own"]["n_energy_evals"] + t["fixed"]["n_energy_evals"] for t in tracked)
    rec = {"label": label, "kind": kind, "s_pct": float(s_pct), "scale": float(factor),
           "a_A": _a_of(at), "cell": at.cell.array.tolist(), **pres,
           "calls": calls, "screen": screen, "softest_own": soft, "tracked": tracked,
           "min_commensurate_thz": _softest_commensurate(qs, freqs)[2],
           "wall_s": tw, "n_fd_supercells_per_phonon_run": n_fd,
           "n_energy_evals_tracked": n_e,
           "utc_done": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return rec, new_refs


# ------------------------------------------------------------------------- derived ----

def _slopes(pts, get, sign):
    """d ln(sign*y)/d ln a: central differences at s = +-0.25, +-0.5 and a log-linear fit over
    FIT_WINDOW. None where the quantity is missing or has the wrong sign (log undefined)."""
    by_s = {round(p["s_pct"], 6): p for p in pts}

    def val(p):
        y = get(p)
        return math.log(sign * y) if _fin(y) and sign * y > 0 else None

    out = {}
    for h in (0.25, 0.5):
        lo, hi = by_s.get(-h), by_s.get(h)
        if lo is None or hi is None:
            out[f"central_h{h}"] = None
            continue
        yl, yh = val(lo), val(hi)
        out[f"central_h{h}"] = (None if yl is None or yh is None else
                                (yh - yl) / (math.log(hi["a_A"]) - math.log(lo["a_A"])))
    win = [p for s, p in sorted(by_s.items()) if FIT_WINDOW[0] - 1e-9 <= s <= FIT_WINDOW[1] + 1e-9]
    ys = [val(p) for p in win]
    if len(win) >= 3 and all(y is not None for y in ys):
        x = np.log([p["a_A"] for p in win])
        y = np.array(ys)
        A = np.stack([x, np.ones_like(x)], axis=1)
        (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - (m * x + c)
        sxx = float(((x - x.mean()) ** 2).sum())
        se = math.sqrt(float(res @ res) / (len(x) - 2) / sxx) if len(x) > 2 else None
        r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum()) if len(x) > 1 else None
        out["loglin_fit"] = {"slope": float(m), "stderr": se, "r2": r2, "n": len(x),
                             "s_window_pct": list(FIT_WINDOW)}
    else:
        out["loglin_fit"] = None
    return out


def _two_point(p0, p1, get, sign):
    y0, y1 = get(p0), get(p1)
    if not (_fin(y0) and _fin(y1) and sign * y0 > 0 and sign * y1 > 0):
        return None
    return (math.log(sign * y0) - math.log(sign * y1)) / (math.log(p0["a_A"]) - math.log(p1["a_A"]))


QUANTITIES = (("depth_refined_meV", 1), ("depth_sampled_meV", 1), ("depth_fit_meV", 1),
              ("kappa2", -1), ("kappa4", 1))


def derived(sysrec, temps) -> dict:
    pts = sysrec["points"]
    a0 = sysrec["relax"]["a0_A"]
    grid = sorted([p for p in pts.values() if p["kind"] == "grid"], key=lambda p: p["s_pct"])
    pbe = pts.get("pbe")
    s0 = pts.get(_label(0.0))
    refs = sysrec.get("refs")
    out = {"series": {}, "notes": []}
    if refs is None or s0 is None:
        return out
    getters = {}
    for r in refs["modes"]:
        k = r["k"]
        getters[f"own[{k}]"] = (lambda kk: lambda p: p["tracked"][kk]["own"])(k)
        getters[f"fixed[{k}]"] = (lambda kk: lambda p: p["tracked"][kk]["fixed"])(k)
    getters["softest_own"] = lambda p: p["softest_own"]
    series = {}
    for name, src in getters.items():
        blk = {}
        for q, sign in QUANTITIES:
            g = (lambda qq, ss: lambda p: src(p).get(qq))(q, sign)
            blk[q] = {**_slopes(grid, g, sign),
                      "pbe_two_point": _two_point(s0, pbe, g, sign) if pbe else None}
        if name.startswith("own["):
            k = int(name[4:-1])
            g = lambda p, kk=k: p["tracked"][kk]["harm_thz"]
            blk["harm_thz_abs"] = {**_slopes(grid, g, -1),
                                   "pbe_two_point": _two_point(s0, pbe, g, -1) if pbe else None}
            ov = [p["tracked"][k]["own"]["overlap_with_s0"] for p in grid]
            blk["min_overlap_with_s0"] = min(ov) if ov else None
        if name == "softest_own":
            win = [p for p in grid if FIT_WINDOW[0] - 1e-9 <= p["s_pct"] <= FIT_WINDOW[1] + 1e-9]
            keys = {tuple(p["softest_own"]["qkey"]) for p in win}
            blk["softest_q_star_constant_in_window"] = len(keys) <= 1
        fm = {src(p).get("fit_model") for p in grid}
        blk["fit_models_on_grid"] = sorted(x for x in fm if x)
        series[name] = blk
    out["series"] = series
    k = refs["reference_index"]
    own, fixed = series[f"own[{k}]"], series[f"fixed[{k}]"]
    out["reference"] = {
        "k": k, "q": refs["modes"][k]["q"], "band": refs["modes"][k]["band"],
        "series_own": f"own[{k}]", "series_fixed": f"fixed[{k}]",
        "headline_dlnDepth_dlna": {"own": own["depth_refined_meV"],
                                   "fixed": fixed["depth_refined_meV"]},
        "dln_abs_kappa2_dlna_own": own["kappa2"], "dln_kappa4_dlna_own": own["kappa4"],
        "dln_abs_omega_harm_dlna": own["harm_thz_abs"]}
    # depth curves for a quick read
    out["depth_curve_reference"] = [
        {"label": p["label"], "s_pct": p["s_pct"], "a_A": p["a_A"], "P_GPa": p["pressure_GPa"],
         "own_depth_refined_meV": p["tracked"][k]["own"]["depth_refined_meV"],
         "fixed_depth_refined_meV": p["tracked"][k]["fixed"]["depth_refined_meV"],
         "own_depth_sampled_meV": p["tracked"][k]["own"]["depth_sampled_meV"],
         "own_kappa2": p["tracked"][k]["own"]["kappa2"],
         "own_kappa4": p["tracked"][k]["own"]["kappa4"],
         "harm_thz": p["tracked"][k]["harm_thz"],
         "overlap_with_s0": p["tracked"][k]["own"]["overlap_with_s0"],
         "ln_own_over_fixed_depth": (
             math.log(p["tracked"][k]["own"]["depth_refined_meV"]
                      / p["tracked"][k]["fixed"]["depth_refined_meV"])
             if (p["tracked"][k]["own"]["depth_refined_meV"] or 0) > 0
             and (p["tracked"][k]["fixed"]["depth_refined_meV"] or 0) > 0 else None),
         "softest_q": p["softest_own"]["q"], "softest_depth_refined_meV":
             p["softest_own"].get("depth_refined_meV"),
         "calls": {T: c["pred_stable"] for T, c in p["calls"].items()},
         "n_curv_blind": {T: c["n_curv_blind"] for T, c in p["calls"].items()}}
        for p in sorted(pts.values(), key=lambda p: p["a_A"])]
    # Call flips per temperature over every evaluated lattice (grid, PBE and the screen-only
    # bisection points of refine_flips), sorted by a. a* is the midpoint of the final bracket.
    # It is NOT interpolated in the SCHA margin: on the stable side the margin saturates at the
    # curvature shoulder (~+0.3 meV) while condensation makes it drop steeply on the unstable
    # side, so a linear interpolation is biased toward the stable end.
    a_pbe = pbe["a_A"] if pbe else sysrec.get("a_pbe_A")
    main_seq = sorted(pts.values(), key=lambda p: p["a_A"])
    seq = sorted(list(pts.values()) + list(sysrec.get("flip_screens", {}).values()),
                 key=lambda p: p["a_A"])
    flips = {}
    for T in [str(int(t)) for t in temps]:
        ent = {"calls_by_label": {p["label"]: p["calls"][T]["pred_stable"] for p in main_seq},
               "call_at_a0": s0["calls"][T]["pred_stable"],
               "call_at_pbe": pbe["calls"][T]["pred_stable"] if pbe else None, "flips": []}
        for p1, p2 in zip(seq, seq[1:]):
            c1, c2 = p1["calls"][T], p2["calls"][T]
            if c1["pred_stable"] == c2["pred_stable"]:
                continue
            a_star = 0.5 * (p1["a_A"] + p2["a_A"])
            ent["flips"].append({
                "a_star_A": a_star, "pct_from_a0": 100 * (a_star / a0 - 1),
                "pct_from_pbe": 100 * (a_star / a_pbe - 1) if a_pbe else None,
                "halfwidth_pct_of_a0": 50 * (p2["a_A"] - p1["a_A"]) / a0,
                "bracket": [p1["label"], p2["label"]], "bracket_a_A": [p1["a_A"], p2["a_A"]],
                "bracket_phase_margin_meV": [c1["phase_margin_meV"], c2["phase_margin_meV"]],
                "unstable_side": "larger a" if not c2["pred_stable"] else "smaller a",
                "condensing_q_on_unstable_side": (c2 if not c2["pred_stable"] else c1)["decide_q"],
                "interpolation": "midpoint of the final (bisected) bracket"})
        ent["n_flips"] = len(ent["flips"])
        ent["swept_a_range_A"] = [main_seq[0]["a_A"], main_seq[-1]["a_A"]]
        flips[T] = ent
    out["call_flips"] = flips
    mism = sum(c["margin_call_mismatch"] for p in seq for c in p["calls"].values())
    out["margin_call_mismatch_total"] = mism
    if mism:
        out["notes"].append("SCHA margin sign disagrees with the production call somewhere")
    return out


def refine_flips(rec, relaxed, calc, temps, P, work, tol_pct, save, max_evals=40):
    """Bisect every lattice bracket whose call differs at some T until it is narrower than
    tol_pct of a0, with the production screen only (no tracked maps). One screen gives the
    calls at every T, so a midpoint can split several temperatures' brackets at once."""
    fs = rec.setdefault("flip_screens", {})
    n = 0
    while n < max_evals:
        seq = sorted([(p["scale"], p) for p in list(rec["points"].values()) + list(fs.values())],
                     key=lambda t: t[0])
        todo = None
        for (s1, p1), (s2, p2) in zip(seq, seq[1:]):
            if (s2 - s1) * 100 > tol_pct and any(
                    p1["calls"][T]["pred_stable"] != p2["calls"][T]["pred_stable"]
                    for T in p1["calls"]):
                todo = 0.5 * (s1 + s2)
                break
        if todo is None:
            return n
        label = f"b{(todo - 1) * 100:+.5f}"
        if label in fs:                                # resolution limit of the label
            return n
        t0 = time.time()
        at = _scaled(relaxed, todo)
        screen, _, calls, _, _ = run_screen(at, calc, temps, P, work)
        fs[label] = {"label": label, "kind": "bisect", "s_pct": (todo - 1) * 100,
                     "scale": todo, "a_A": _a_of(at), **_pressure(at, calc), "calls": calls,
                     "screen_modes": [{k: m.get(k) for k in ("q", "band", "harm_thz", "kappa2",
                                                            "kappa4", "kappa6",
                                                            "depth_sampled_meV")}
                                      for m in screen["modes"]],
                     "wall_s": time.time() - t0}
        print(f"    {label} a={fs[label]['a_A']:.5f} "
              + " ".join(f"{T}K:{'S' if c['pred_stable'] else 'U'}" for T, c in calls.items())
              + f"  [{fs[label]['wall_s']:.1f} s]", flush=True)
        save()
        n += 1
    return n


# ------------------------------------------------------------------ ledger check ----

def ledger_check(system, model, model_version, s0, a0, refs) -> dict:
    from mlip_dynstab import METHOD_VERSION, SOFTMODE_EQMAP_VERSION
    out = {"tolerances": TOL, "items": {}}
    ok = []
    # lattice vs the deposited relaxed lattice table
    a_dep = None
    if PBE_CSV.exists():
        for row in csv.DictReader(PBE_CSV.open(encoding="utf-8")):
            if row["system"] == system and row["source"] == f"{model} relaxed":
                a_dep = float(row["a_A"])
    if a_dep is not None:
        d = a0 - a_dep
        out["items"]["a0_vs_pl_lattice_csv"] = {"ours": a0, "deposited": a_dep, "diff": d,
                                                "pass": abs(d) <= TOL["a_A"]}
        ok.append(abs(d) <= TOL["a_A"])
    rows = {}
    if LEDGER.exists():
        for line in LEDGER.open(encoding="utf-8"):
            r = json.loads(line)
            if (r.get("system") == system and r.get("model") == model
                    and r.get("method") == "softmode"
                    and r.get("method_version") == METHOD_VERSION["softmode"]):
                rows[str(int(r["temperature_K"]))] = r          # last occurrence wins
    # Inside a degenerate band (Gamma T1u, X doublet) phonopy's eigenvector is arbitrary in the
    # span and the arm of a star is picked by round-off, so the depth/curvature of those modes
    # can differ between installs while the screen is unchanged (dft_checks docstring: MACE
    # BaTiO3 Gamma 16.8 vs 20.9 meV). The pass test therefore uses only gauge-invariant
    # quantities: the call, the deciding q-STAR and its harmonic frequency, depth and curvature
    # when that mode is non-degenerate, every mode's harmonic frequency, and the curvature of
    # every non-degenerate mode. Degenerate-mode curvatures and n_curv_blind are reported.
    ndeg = [t["own"]["n_degenerate"] for t in s0["tracked"]]
    star = lambda s: _qkey(float(x) for x in s.split(","))
    for T, c in s0["calls"].items():
        r = rows.get(T)
        if r is None:
            out["items"][f"T{T}"] = {"ledger_row": None}
            continue
        led_curv = [float(x) for x in r["ft_modes_curv_thz"].split(";")]
        led_harm = [float(x) for x in r["ft_modes_harm_thz"].split(";")]
        harm = [m["harm_thz"] for m in s0["screen"]["modes"]]
        same_n = len(led_curv) == len(c["modes_curv_thz"]) == len(ndeg)
        dd = c["decide_well_depth_meV"] - r["ft_well_depth_meV"]
        dh = c["decide_harm_thz"] - r["ft_decide_harm_thz"]
        di = c["decide_index"]
        dec_nondeg = di is not None and di < len(ndeg) and ndeg[di] == 1
        if same_n:
            dharm = max(abs(a - b) for a, b in zip(harm, led_harm))
            dcurv = [abs(a - b) for a, b in zip(c["modes_curv_thz"], led_curv)]
            dcurv_nd = max([d for d, n in zip(dcurv, ndeg) if n == 1] or [0.0])
        else:
            dharm, dcurv, dcurv_nd = None, None, None
        tol3 = 1e-3                                  # ledger mode strings carry 3 decimals
        it = {"uhash": r["uhash"], "ledger_model_version": r["model_version"],
              "pred_stable": [c["pred_stable"], bool(r["pred_stable"])],
              "decide_q": [c["decide_q"], r["ft_decide_q"]],
              "decide_star_equal": star(c["decide_q"]) == star(r["ft_decide_q"]),
              "decide_mode_nondegenerate": dec_nondeg,
              "decide_harm_thz": [c["decide_harm_thz"], r["ft_decide_harm_thz"]],
              "decide_well_depth_meV": [c["decide_well_depth_meV"], r["ft_well_depth_meV"]],
              "modes_harm_thz": [harm, led_harm], "max_abs_dharm_thz": dharm,
              "modes_curv_thz": [c["modes_curv_thz"], led_curv], "abs_dcurv_thz": dcurv,
              "modes_n_degenerate": ndeg,
              "modes_Q0_ang": [c["modes_Q0_ang"], [float(x) for x in
                                                   r["ft_modes_Q0_ang"].split(";")]],
              "n_curv_blind": [c["n_curv_blind"], r["ft_n_curv_blind"]],
              "min_eff_freq_thz": [c["min_eff_freq_thz"], r["min_eff_freq_thz"]]}
        p = (c["pred_stable"] == bool(r["pred_stable"]) and it["decide_star_equal"]
             and abs(dh) <= TOL["harm_thz"] and same_n and dharm <= TOL["harm_thz"] + tol3
             and dcurv_nd <= TOL["curv_thz"] + tol3
             and (not dec_nondeg or abs(dd) <= max(TOL["depth_abs_meV"],
                                                   TOL["depth_rel"] * r["ft_well_depth_meV"])))
        it["pass"] = bool(p)
        it["n_curv_blind_equal"] = c["n_curv_blind"] == r["ft_n_curv_blind"]
        it["gauge_note"] = ("degenerate modes differ only through the arbitrary basis inside "
                            "the band; not counted" if same_n and any(
                                d > TOL["curv_thz"] + tol3 for d, n in zip(dcurv, ndeg) if n > 1)
                            else None)
        out["items"][f"T{T}"] = it
        ok.append(bool(p))
    # production E(Q) cache, read only (gitignored, so usually only on the machine that made it)
    pc = REPO / "results" / "cache" / (f"softmode_v{SOFTMODE_EQMAP_VERSION}m{MAX_MODES}_{system}"
                                       f"_{model}_sc{''.join(map(str, SUPERCELL))}.json")
    if pc.exists():
        prod = json.loads(pc.read_text())
        ours = s0["screen"]["modes"]
        cmp = []
        for i, (m, o) in enumerate(zip(prod["modes"], ours)):
            cmp.append({"index": i, "q": [o["q"], m["q"]], "band": [o["band"], m["band"]],
                        "star_equal": _qkey(o["q"]) == _qkey(m["q"]),
                        "n_degenerate": ndeg[i] if i < len(ndeg) else None,
                        "harm_thz": [o["harm_thz"], m["harm_thz"]],
                        "depth_meV": [o["depth_sampled_meV"], m["well_depth_meV"]],
                        "abc": [[o["kappa2"], o["kappa4"], o["kappa6"]], [m["a"], m["b"], m["c"]]],
                        "max_abs_dE_meV": float(np.max(np.abs(np.subtract(o["dE_eV"], m["dE"]))))
                        * 1000 if len(o["dE_eV"]) == len(m["dE"]) else None})
        out["production_cache"] = {"file": pc.name, "n_modes": [len(ours), len(prod["modes"])],
                                   "modes": cmp}
    out["model_version_ours"] = model_version
    out["pass"] = bool(ok) and all(ok)
    return out


# --------------------------------------------------------------------------- main ----

def _pbe_lattices() -> dict:
    out = {}
    if PBE_CSV.exists():
        for row in csv.DictReader(PBE_CSV.open(encoding="utf-8")):
            if row["source"] == PBE_SOURCE:
                out[row["system"]] = float(row["a_A"])
    return out


def relax_record(system, calc, P) -> dict:
    from mlip_dynstab.harmonic import _relax
    from mlip_dynstab.systems import build_atoms, get_spec
    t0 = time.time()
    start = build_atoms(get_spec(system))
    at = start.copy()
    at.calc = calc
    at = _relax(at, fmax=P["relax_fmax"])
    f = np.asarray(at.get_forces(), float)
    d = at.get_scaled_positions(wrap=False) - start.get_scaled_positions(wrap=False)
    d -= np.round(d)
    cp = at.cell.cellpar()
    rec = {"a0_A": _a_of(at), "a_start_A": _a_of(start), "cellpar": cp.tolist(),
           "max_cell_anisotropy_A": float(cp[:3].max() - cp[:3].min()),
           "max_angle_dev_deg": float(np.abs(cp[3:] - 90.0).max()),
           "max_frac_dev_from_ideal": float(np.abs(d).max()),
           "max_force_eV_A": float(np.linalg.norm(f, axis=1).max()),
           **_pressure(at, calc), "wall_s": time.time() - t0, "atoms": _atoms_rec(at)}
    rec["residual_pressure_GPa"] = rec.pop("pressure_GPa")
    return rec


def main() -> None:
    from mlip_dynstab import SUPPORTED_MODELS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--s-pct", default=",".join(str(s) for s in S_PCT),
                    help="lattice offsets in %% of a0 (s = 0 is always added)")
    ap.add_argument("--temps", default=",".join(str(int(t)) for t in TEMPS))
    ap.add_argument("--precision", choices=("auto", "production"), default="auto")
    ap.add_argument("--no-pbe", action="store_true", help="skip the PBE-lattice point")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--flip-tol-pct", type=float, default=0.02,
                    help="bisect call-flip brackets to this width in %% of a0 (0: off)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Lennard-Jones stand-in instead of the model (plumbing test)")
    args = ap.parse_args()

    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    s_list = sorted({0.0, *(float(x) for x in args.s_pct.split(",") if x.strip())},
                    key=lambda s: (abs(s), s))
    temps = [float(t) for t in args.temps.split(",") if t.strip()]
    P = production_params()
    settings = {"production": P, "temps_K": temps, "precision": args.precision,
                "fit_window_pct": list(FIT_WINDOW)}
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path

    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
    if out is not None:
        for k, v in (("schema", SCHEMA), ("model", args.model), ("dry_run", bool(args.dry_run))):
            if out.get(k) != v:
                raise SystemExit(f"{out_path}: {k}={out.get(k)!r}, this run {v!r}; refusing to mix. "
                                 "Move the file aside.")
        if _clean(out.get("settings")) != _clean(settings):
            raise SystemExit(f"{out_path}: settings differ from this run; refusing to mix.\n"
                             f"file: {out.get('settings')}\nrun:  {settings}")
    else:
        out = {"schema": SCHEMA, "script": SCRIPT, "model": args.model,
               "dry_run": bool(args.dry_run),
               "note": "POST HOC (R2 major revision), not pre-registered. Fixed internal "
                       "(cubic) positions, isotropic cell scaling; production screen code "
                       "(compute_finite_t_softmode, relax=False) at every lattice.",
               "settings": settings, "sessions": [], "systems": {}}

    from mlip_dynstab.systems import build_atoms, get_spec
    probe = build_atoms(get_spec(systems[0]))
    calc, version, dtype_note = load_calculator(args.model, args.device, args.precision,
                                                args.dry_run, probe)
    if out.get("model_version") not in (None, version):
        raise SystemExit(f"{out_path}: model_version {out['model_version']!r} != {version!r}")
    out["model_version"] = version
    probe.calc = calc
    forces_dtype = str(np.asarray(probe.get_forces()).dtype)   # what the calculator returns
    versions = _versions(args.model)
    session = {"index": len(out["sessions"]), "utc_start": datetime.now(timezone.utc)
               .isoformat(timespec="seconds"), "host": socket.gethostname(),
               "platform": platform.platform(), "python": platform.python_version(),
               "argv": sys.argv[1:], "device": args.device, "script_sha256": _sha256(__file__),
               **_git(), "packages": versions, "lock_check": _lock_check(args.model, versions),
               "checkpoint": ({"version_string": version} if args.dry_run
                              else _checkpoint_id(args.model, calc, version)),
               "dtype_policy": dtype_note, "forces_dtype": forces_dtype, **_torch_info(calc)}
    out["sessions"].append(session)
    atomic_write_json(out_path, out)
    pbe_a = {} if args.no_pbe else _pbe_lattices()
    print(f"[lattice_sweep] {args.model} ({version}) on {args.device}; dtype: {dtype_note}; "
          f"param dtype {session.get('param_dtype')}", flush=True)

    with tempfile.TemporaryDirectory(prefix="latsweep_") as tmpd:
        work = Path(tmpd)
        for system in systems:
            rec = out["systems"].setdefault(system, {"points": {}})
            if "relax" not in rec:
                rec["relax"] = relax_record(system, calc, P)
                rec["relax"]["session"] = session["index"]
                atomic_write_json(out_path, out)
            relaxed = _atoms_from(rec["relax"]["atoms"])
            a0 = rec["relax"]["a0_A"]
            print(f"  {system}: a0 = {a0:.5f} A, residual P = "
                  f"{rec['relax']['residual_pressure_GPa']:+.4f} GPa", flush=True)
            targets = [(_label(s), "grid", s, 1.0 + s / 100.0) for s in s_list]
            if system in pbe_a:
                rec["a_pbe_A"] = pbe_a[system]
                rec["s_pbe_pct"] = 100 * (pbe_a[system] / a0 - 1)
                targets.append(("pbe", "pbe", rec["s_pbe_pct"], pbe_a[system] / a0))
                targets.sort(key=lambda t: (t[0] != _label(0.0), abs(t[2]), t[2]))
            for label, kind, s, factor in targets:
                if label in rec["points"]:
                    continue
                pt, new_refs = run_point(label, kind, s, factor, relaxed, calc, rec.get("refs"),
                                         P, temps, work)
                pt["session"] = session["index"]
                if new_refs is not None:
                    rec["refs"] = new_refs
                rec["points"][label] = pt
                k = rec["refs"]["reference_index"]
                tr = pt["tracked"][k]
                calls = " ".join(f"{T}K:{'S' if c['pred_stable'] else 'U'}"
                                 for T, c in pt["calls"].items())
                print(f"    {label:7s} a={pt['a_A']:.5f} P={pt['pressure_GPa']:+7.3f} GPa  "
                      f"ref q={tr['q']} harm={tr['harm_thz']:+.3f}  depth own/fixed "
                      f"{tr['own']['depth_refined_meV']:.2f}/{tr['fixed']['depth_refined_meV']:.2f}"
                      f" meV  {calls}  [{pt['wall_s']['total']:.1f} s]", flush=True)
                if label == _label(0.0) and not args.dry_run:
                    rec["ledger_check"] = ledger_check(system, args.model, version, pt, a0,
                                                       rec["refs"])
                    print(f"    ledger check at s=0: pass={rec['ledger_check']['pass']}",
                          flush=True)
                rec["derived"] = derived(rec, temps)
                atomic_write_json(out_path, out)
            if args.flip_tol_pct > 0 and rec.get("refs") is not None:
                rec["flip_tol_pct"] = args.flip_tol_pct
                refine_flips(rec, relaxed, calc, temps, P, work, args.flip_tol_pct,
                             lambda: atomic_write_json(out_path, out))
            if not args.dry_run and _label(0.0) in rec["points"]:      # refresh on resume too
                rec["ledger_check"] = ledger_check(system, args.model, version,
                                                   rec["points"][_label(0.0)], a0, rec["refs"])
            rec["derived"] = derived(rec, temps)
            atomic_write_json(out_path, out)
    session["utc_end"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(out_path, out)
    print(f"[lattice_sweep] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
