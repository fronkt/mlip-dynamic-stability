"""Multi-mode SSCHA free energy along the condensing soft mode (R2, second-round referee).

Pre-registration: tasks/preregistration-centroid-2026-10-21.md (fixed; this file implements it).
One unit: BaTiO3 cubic / MACE-MP-0 medium / 100 K, the 2x2x2 supercell (40 atoms) treated as a
Gamma-only cell, the converged grid recipe (sscha_seed_study.converged_recipe, imported, not copied).

Arms (one JSON each, under results/revision/r2_centroid/ unless --out-dir says otherwise):
  gate     gate/gate.json + gate/pattern.npz. The screen's M pattern (softest band at q=(1/2,0,1/2),
           gate -5.404 +/- 0.02 THz), the Gamma T1u [111] pattern and its static E(Q) minimum, both
           mapped onto the cellconstructor supercell atom order. Every other arm LOADS pattern.npz, so
           all arms share one bit-identical pattern. Exit 0 = all checks pass, 3 = only the M-band gate
           failed (the prereg's "not interpreted" case), 1 = plumbing failure.
  m_scan   fixed centroid (minim_struct False) at cubic + Q0 * u_M, Q0 in {0, 0.04, 0.075, 0.10} A:
           m_scan_Q<Q0>.json
  free     free centroid (minim_struct True) from Q0 = 0.075: free_Q0.0750.json; the final centroid is
           projected back onto the pattern, Q_final = <u, R - R0> / <u, u>
  gamma    fixed centroid at the static-E(Q) minimum along the Gamma T1u [111] pattern: gamma.json
  v4       include_v4 False vs True on a 512-configuration subsample (antithetic pairs kept, fixed
           seed) of the Q0 = 0 arm's Hessian ensemble, in a spawned CPU process with a time cap
           (sscha_seed_study.v4_child / run_v4): v4.json
  --summarize   summary.json: the prereg decision rule applied mechanically (decide(), a pure function
                with a self-test: --selftest), the gates, the secondary arms, the v4 shift.
  --smoke-check  pass/fail of the short box smoke runs (see the launcher).

How each piece is built, and why
  * Reference: the grid's relax (production_recipe relax/fmax), the harmonic dyn from
    finite_t._cc_dyn_from_phonopy at the production disp (0.03 A) on 2x2x2, start A from
    sscha_seed_study.build_start (ForcePositiveDefinite + Symmetrize on the q-space dyn), then
    Phonons.GenerateSupercellDyn -> one 40-atom Gamma-only dyn (supercell (1,1,1)). The conversion is
    checked: the 120 supercell frequencies of the q-space and Gamma-only dyns must agree to 1e-3 THz.
  * Pattern: the screen's own force constants, finite_t._harmonic_phonon at the screen's disp (read
    from compute_finite_t_softmode's signature, 0.01 A), NOT the SSCHA's disp-0.03 dyn: the gate value
    -5.404 THz is the screen's (the disp-0.03 dyn puts the Gamma mode at -6.30 THz, the screen -6.41).
    Softest band at q = (1/2, 0, 1/2), phonopy run_modulations directly in [2,2,2], real part, max
    |Cartesian component| = 1 (as finite_t._mode_pattern), so Q0 is in the screen's units. The gate also
    builds the screen's 2x1x2 pattern with _mode_pattern itself and checks that the 2x2x2 one is it,
    tiled (and M_eff against the ledger's 238.02 amu).
  * Mapping: phonopy supercell atoms -> cellconstructor supercell atoms by fractional position (same
    lattice, tolerance 1e-4 in fractional coordinates, one-to-one asserted, species checked). Checked
    independently of the mapping code: the finite-difference curvature of the MLIP energy along the
    MAPPED pattern in the cellconstructor-ordered cell must give the band frequency, and the mapped
    pattern must lie in the imaginary subspace of the cellconstructor harmonic dyn.
  * Gamma [111]: the soft optical triplet at Gamma (acoustic masked by |omega|) is triply degenerate,
    so the [111] member is fixed basis-independently: the projector onto the triplet's eigenspace,
    E E^dagger, applied to a unit Ti displacement along [111] (mass-weighted coordinates). That is the
    triplet member that transforms like (x+y+z) whatever basis the diagonaliser returned; the
    projected weight is logged. Static E(Q) on 0..0.30 A (step 0.005) then +/-0.0075 A around the
    coarse minimum (step 0.00025); the fine-grid argmin is the amplitude (a parabola vertex is logged).
  * Fixed centroid: SSCHA_Minimizer.minim_struct = False and gradi_op = "gc" (the structure gradient
    is non-zero by construction at a displaced fixed centroid; the prereg's converged stop is the dyn
    gradient over its real error). The final centroid must equal the start to 1e-8 A (minimum image),
    else the arm fails. Free arm: minim_struct True, gradi_op "all" (the grid's).
  * Free energy (primary observable, prereg deviation D4): F of the fresh n_hessian ensemble drawn at the
    final matrix (KL = 1, independent of the minimisation). Also recorded, as a secondary verdict:
    after the relaxation the last population is reweighted to the
    final dyn (Ensemble.update_weights) and Ensemble.get_free_energy(return_error=True) gives F and its
    stochastic error. The minimiser's last history entry and the fresh Hessian ensemble's F (KL = 1)
    are recorded beside it. With supercell (1,1,1) F is per 40-atom cell; the grid's F (supercell
    2x2x2) is per 5-atom primitive cell, so its 1.8e-5 Ry error is ~1.45e-4 Ry (2 meV) per 40 atoms.
  * Seeds: one independent global seed per arm (ARM_SEED_OFFSET), so the two runs of the decision
    rule have the independent errors its sigma = sqrt(s1^2 + s2^2) assumes.
  * Everything else is the grid's code, imported: RealErrorEnsemble (real serial gradient error +
    Kong-Liu guard), the per-population step cap and wall cap (PopulationController), the step
    recorder, the population table, the fresh-gradient check, the Hessian ensemble of n_hessian, the
    antithetic-pair bootstrap, the include_v4 child and its round-trip check.

python-sscha 1.6.1 / cellconstructor 1.6.2 behaviour this file relies on and the box smoke must show
(none can be run on the Windows PC, which has neither): (1) minim_struct False leaves the centroid
untouched (checked to 1e-8 A); (2) the minimiser and the fresh check set up symmetries on a 40-atom
Gamma-only supercell (QE_Symmetry; the count is logged as n_symmetries; --neglect-symmetries is the
fallback if it raises); (3) GenerateSupercellDyn returns a Gamma-only Phonons with masses (conversion
check); (4) Ensemble.get_free_energy divides by prod(supercell), so F is per 40 atoms (smoke: F(Q0=0)/8
against the grid's per-cell F); (5) the free arm's centroid moves.

Usage (repo root; on the box inside /root/env-mace):
    python scripts/sscha_centroid_scan.py --selftest
    python scripts/sscha_centroid_scan.py --arm gate --device cuda
    python scripts/sscha_centroid_scan.py --arm m_scan --q0 0.075 --device cuda --unit-timeout 6600
    python scripts/sscha_centroid_scan.py --arm free --device cuda
    python scripts/sscha_centroid_scan.py --arm gamma --device cuda
    python scripts/sscha_centroid_scan.py --arm v4 --threads 24 --v4-timeout 13500
    python scripts/sscha_centroid_scan.py --summarize
    python scripts/sscha_centroid_scan.py --arm m_scan --q0 0.075 --dry-run --device cpu --out-dir <scratch>
Resumable like the grid: an 'ok' arm is skipped, a 'failed' one unless --retry-failed, a 'running' one
whose process is gone is recomputed; a JSON made with another recipe or dry-run mode is refused.
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import sscha_seed_study as sss  # noqa: E402  (definitions only; its main is guarded)

SCRIPT = Path(__file__).resolve().relative_to(REPO).as_posix()
SCHEMA = "sscha_centroid/v1"
OUT_DIR = REPO / "results" / "revision" / "r2_centroid"
PREREG = "tasks/preregistration-centroid-2026-10-21.md"
GRID_JSON = (REPO / "results" / "revision" / "sscha_converged_grid"
             / "batio3_cubic_mace_mp0_100K_sc222_startA.json")

# ---- fixed by the pre-registration -------------------------------------------------------------
SYSTEM, MODEL, T_K, SC = "batio3_cubic", "mace_mp0", 100.0, (2, 2, 2)
M_Q = (0.5, 0.0, 0.5)
M_GATE_THZ, M_GATE_TOL_THZ = -5.404, 0.02
M_EFF_LEDGER_AMU = 238.02486351343032            # ledger ft_m_eff_amu, 2x1x2 cell (screen v4)
LEDGER_V = (-14.25817012328368, 1080.6343492399399, 5314.012038511375)   # eV per 2x1x2 cell
SCAN_Q0 = (0.0, 0.04, 0.075, 0.10)
MAIN_Q0 = 0.075
FREE_Q0 = 0.075
HESS_REF_THZ, HESS_REF_SD_THZ, HESS_GATE_NSD = 1.98, 0.055, 3.0
DECIDE_NSIGMA = 2.0
V4_N = 512

# ---- implementation constants -------------------------------------------------------------------
RY_EV = 13.605693122994            # CODATA 2018, only to print Ry differences in meV
THZ_PER_SQRT_EV_A2_AMU = 15.633302  # sqrt(eV / (A^2 amu)) / (2 pi) in THz (phonopy's VaspToTHz)
MAP_TOL_FRAC = 1e-4
FIXED_TOL_ANG = 1e-8
R0_TOL_ANG = 1e-5                  # an arm's own relaxed reference against the gate's
CELL_TOL_ANG = 1e-6
DEG_TOL_THZ = 1e-3
CONVERSION_TOL_THZ = 1e-3
FD_H_ANG = 0.005
FD_TOL_THZ = 0.15                  # FD curvature (h = 0.005, disp-free) vs the disp-0.01 band frequency
SOFT_WEIGHT_MIN = 0.95
GAMMA_COARSE = (0.0, 0.30, 0.005)
GAMMA_FINE_HALF, GAMMA_FINE_STEP = 0.0075, 0.00025
M_STATIC_Q = (0.025, 0.04, 0.075, 0.10)
CENTROID_SALT, V4_SALT = 20261021, 20261022
ARM_SEED_OFFSET = {"m_scan_Q0.0000": 0, "m_scan_Q0.0400": 1, "m_scan_Q0.0750": 2,
                   "m_scan_Q0.1000": 3, "free_Q0.0750": 4, "gamma": 5}
ARMS = ("gate", "m_scan", "free", "gamma", "v4")
SSCHA_ARMS = ("m_scan", "free", "gamma")
DEFAULT_UNIT_TIMEOUT_S = 6600.0    # relaxation cap; the box wraps each arm in a 2.5 h `timeout`
DEFAULT_V4_TIMEOUT_S = 13500.0     # child cap; the box wraps the v4 arm in a 4 h `timeout`
# --dry-run only: a synthetic scenario so --summarize exercises a verdict. F(Q0=0) per 40 atoms is the
# grid's per-cell F times 8; the error is the grid's per-cell error times 8.
MOCK_FE0_RY, MOCK_FE_ERR_RY, MOCK_FREE_Q = -23.3789, 1.45e-4, 0.080

utc_now, rel, load_json = sss.utc_now, sss.rel, sss.load_json
atomic_write_json, _clean, _dig = sss.atomic_write_json, sss._clean, sss._dig


# ================================================================ pure helpers (self-tested) ====

def screen_disp() -> float:
    """The screen's finite-displacement amplitude, from its own signature (cannot drift)."""
    from mlip_dynstab.finite_t import compute_finite_t_softmode
    return float(inspect.signature(compute_finite_t_softmode).parameters["disp"].default)


def ledger_v(q: float) -> float:
    """The screen's fitted static well, eV per 2x1x2 cell."""
    a, b, c = LEDGER_V
    return a * q ** 2 + b * q ** 4 + c * q ** 6


def normalize_max(u):
    u = np.asarray(u, dtype=float)
    mx = float(np.abs(u).max())
    return u / mx if mx > 0 else u


def frac_of(coords, cell):
    """Fractional coordinates; cell rows are the lattice vectors (positions = frac @ cell)."""
    return np.linalg.solve(np.asarray(cell, float).T, np.asarray(coords, float).T).T


def min_image(d, cell):
    """Cartesian displacement(s) reduced to the minimum image of the cell."""
    cell = np.asarray(cell, float)
    f = frac_of(np.atleast_2d(d), cell)
    f -= np.round(f)
    return (f @ cell).reshape(np.shape(d))


def map_atoms(frac_src, sym_src, frac_dst, sym_dst, tol=MAP_TOL_FRAC) -> list:
    """perm such that destination atom i is source atom perm[i]: same species, fractional
    positions equal modulo 1 within ``tol`` on every component (both sets in the SAME lattice).
    Raises unless the map is one-to-one and every match is unique."""
    fs = np.asarray(frac_src, float) % 1.0
    fd = np.asarray(frac_dst, float) % 1.0
    if fs.shape != fd.shape or len(sym_src) != len(sym_dst):
        raise RuntimeError(f"atom sets differ in size: {fs.shape} vs {fd.shape}")
    perm = []
    for i in range(len(fd)):
        d = fs - fd[i]
        d -= np.round(d)
        dist = np.max(np.abs(d), axis=1)
        hits = np.flatnonzero(dist <= tol)
        if len(hits) != 1:
            raise RuntimeError(f"destination atom {i} ({sym_dst[i]} at {fd[i]}) matches {len(hits)} "
                               f"source atoms within {tol} (closest {float(dist.min()):.2e})")
        j = int(hits[0])
        if str(sym_src[j]) != str(sym_dst[i]):
            raise RuntimeError(f"destination atom {i} is {sym_dst[i]}, its position match is "
                               f"{sym_src[j]}")
        perm.append(j)
    if len(set(perm)) != len(perm):
        raise RuntimeError("atom map is not one-to-one")
    return perm


def assign_to_prim(sc_frac, sc_sym, dim, prim_frac, prim_sym, tol=MAP_TOL_FRAC) -> list:
    """For every atom of a diagonal ``dim`` supercell, the primitive atom it is an image of. Each
    primitive atom must receive exactly prod(dim) images."""
    red = (np.asarray(sc_frac, float) * np.asarray(dim, float)) % 1.0
    pf = np.asarray(prim_frac, float) % 1.0
    out = []
    for i in range(len(red)):
        d = pf - red[i]
        d -= np.round(d)
        dist = np.max(np.abs(d), axis=1)
        hits = np.flatnonzero(dist <= tol)
        if len(hits) != 1 or str(prim_sym[int(hits[0])]) != str(sc_sym[i]):
            raise RuntimeError(f"supercell atom {i} has no unique primitive image")
        out.append(int(hits[0]))
    counts = np.bincount(out, minlength=len(pf))
    if not np.all(counts == int(np.prod(dim))):
        raise RuntimeError(f"primitive images are not {int(np.prod(dim))} each: {counts.tolist()}")
    return out


def pattern_projection(R, R0, cell, u) -> dict:
    """Q = <u, d> / <u, u> with d = R - R0 (minimum image), and what lies off the pattern."""
    d = min_image(np.asarray(R, float) - np.asarray(R0, float), cell)
    u = np.asarray(u, float)
    uu = float(np.sum(u * u))
    q = float(np.sum(u * d)) / uu if uu > 0 else None
    res = d - (q or 0.0) * u
    return {"Q_ang": q, "u_dot_u": uu, "residual_norm_ang": float(np.linalg.norm(res))}


def gamma_111_vector(eig_deg, masses, ti_index: int) -> tuple:
    """The [111] member of a degenerate polar triplet, independent of the basis the diagonaliser
    returned: P t with P = E E^dagger the projector onto the triplet (columns of ``eig_deg``,
    mass-weighted eigenvectors, 3n x k) and t a unit displacement of atom ``ti_index`` along [111]
    (mass weighting of a single-atom vector does not change its direction). Returns (u (n x 3)
    displacements normalised to max |component| 1 with the Ti component positive, projected weight
    |P t|^2, relative imaginary part)."""
    E = np.asarray(eig_deg, dtype=np.complex128)
    n = E.shape[0] // 3
    t = np.zeros(3 * n, dtype=np.complex128)
    t[3 * ti_index:3 * ti_index + 3] = 1.0 / math.sqrt(3.0)
    v = E @ (np.conj(E).T @ t)
    weight = float(np.real(np.vdot(v, v)))
    nv = float(np.linalg.norm(v))
    imag_rel = float(np.linalg.norm(np.imag(v)) / nv) if nv > 0 else None
    u = np.real(v).reshape(n, 3) / np.sqrt(np.asarray(masses, float))[:, None]
    if float(np.sum(u[ti_index])) < 0:
        u = -u
    return normalize_max(u), weight, imag_rel


def decide(f0, f_main, scan, gates, nsigma: float = DECIDE_NSIGMA) -> dict:
    """The pre-registered decision rule, mechanically. Pure: numbers in, verdict out.

    f0, f_main: {"fe_meV", "err_meV", "converged"} for Q0 = 0 and Q0 = 0.075 (meV per 40-atom cell,
    any common origin), or None. scan: the same with "q0_ang" for the fixed-centroid M points (Q0 = 0
    and 0.075 may be included; they are skipped). gates: {name: bool or None}; anything not True
    means the arms are "not interpreted" (prereg, Gates).

      NOT_EVALUABLE  a gate did not pass, or Q0 = 0 / 0.075 is missing, not converged or has no error
                     ("only arms that reach the converged stop enter a comparison")
      SUPPORTED      dF = F(0.075) - F(0) < -nsigma * sigma, sigma = sqrt(s0^2 + s1^2)
      NOT_SUPPORTED  dF > +nsigma * sigma and no converged scan point below F(0) - nsigma * sigma_pair
      INCONCLUSIVE   otherwise
    Unconverged scan points do not enter the comparison; they are listed in scan_points_excluded."""
    out = {"rule": f"dF_M = F({MAIN_Q0}) - F(0); SUPPORTED if dF < -{nsigma:g} sigma; NOT_SUPPORTED "
                   f"if dF > +{nsigma:g} sigma and no converged scan point below F(0) - {nsigma:g} "
                   "sigma; otherwise INCONCLUSIVE; NOT_EVALUABLE if a gate failed or an arm of the "
                   "contrast did not converge",
           "nsigma": float(nsigma), "reasons": []}
    reasons = out["reasons"]
    bad = sorted(k for k, v in (gates or {}).items() if v is not True)
    if bad:
        reasons.append("gate(s) not passed or not evaluated: " + ", ".join(bad))

    def usable(name, f):
        if f is None:
            reasons.append(f"{name}: no result")
            return False
        if not f.get("converged"):
            reasons.append(f"{name}: not converged")
            return False
        e = f.get("err_meV")
        if e is None or not math.isfinite(float(e)) or float(e) <= 0 or f.get("fe_meV") is None:
            reasons.append(f"{name}: no free energy or no stochastic error")
            return False
        return True

    ok0, ok1 = usable("Q0=0", f0), usable(f"Q0={MAIN_Q0}", f_main)
    if bad or not (ok0 and ok1):
        out["verdict"] = "NOT_EVALUABLE"
        return out
    dF = float(f_main["fe_meV"]) - float(f0["fe_meV"])
    s = math.hypot(float(f0["err_meV"]), float(f_main["err_meV"]))
    points, excluded = [], []
    for p in scan or []:
        q = float(p["q0_ang"])
        if abs(q) < 1e-9 or abs(q - MAIN_Q0) < 1e-9:
            continue
        e = p.get("err_meV")
        if (not p.get("converged") or p.get("fe_meV") is None or e is None
                or not math.isfinite(float(e)) or float(e) <= 0):
            excluded.append(q)
            continue
        dp = float(p["fe_meV"]) - float(f0["fe_meV"])
        sp = math.hypot(float(p["err_meV"]), float(f0["err_meV"]))
        points.append({"q0_ang": q, "dF_meV": dp, "sigma_meV": sp, "z": dp / sp,
                       "below_F0_minus_nsigma": bool(dp < -nsigma * sp)})
    out.update(dF_meV=dF, sigma_meV=s, z=dF / s, scan_points=points,
               scan_points_excluded=excluded)
    if dF < -nsigma * s:
        out["verdict"] = "SUPPORTED"
    elif dF > nsigma * s and not any(p["below_F0_minus_nsigma"] for p in points):
        out["verdict"] = "NOT_SUPPORTED"
    else:
        out["verdict"] = "INCONCLUSIVE"
        if dF > nsigma * s:
            reasons.append("dF > +nsigma sigma, but a scan point lies below F(0) - nsigma sigma")
    return out


# ============================================================================ small plumbing ====

def arm_tag(arm: str, q0) -> str:
    if arm == "gamma":
        return "gamma"
    return f"{arm}_Q{float(q0):.4f}"


def arm_seed_offset(tag: str) -> int:
    if tag in ARM_SEED_OFFSET:
        return ARM_SEED_OFFSET[tag]
    return 100 + int(round(float(tag.rsplit("_Q", 1)[-1]) * 1e4)) % 1000   # --allow-other-q0 only


def unit_ident(arm: str, q0) -> dict:
    return {"system": SYSTEM, "model": MODEL, "T": T_K, "supercell": list(SC),
            "cell_form": "gamma_only", "arm": arm,
            "q0_ang": None if q0 is None else float(q0)}


def out_base(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir).resolve()
    return OUT_DIR / "_dryrun" if args.dry_run else OUT_DIR


def gate_dir_of(args, base: Path) -> Path:
    return Path(args.gate_dir).resolve() if args.gate_dir else base / "gate"


def arm_paths(base: Path, tag: str) -> dict:
    work = Path(base) / "work" / tag
    return {"json": Path(base) / f"{tag}.json", "work": work,
            "ensemble": work / "hessian_ensemble", "marker": work / "ENSEMBLE_SAVED.json"}


def centroid_recipe(args) -> dict:
    """The grid's converged recipe (sscha_seed_study.converged_recipe) plus what this study fixes.
    Stored in every JSON; an arm refuses to write into a file made with another recipe."""
    r = sss.converged_recipe(args)
    r.update(seed_rule="np.random.seed(seed + ARM_SEED_OFFSET[arm tag]); one stream per arm",
             cell_form="gamma_only 2x2x2 (40 atoms, supercell (1,1,1)) via Phonons.GenerateSupercellDyn"
                       " of start A",
             pattern=f"screen FCs (finite_t._harmonic_phonon, disp {screen_disp():g}); q (0.5, 0, 0.5) "
                     "softest band; phonopy run_modulations [2,2,2]; real part; max |component| = 1",
             gamma_pattern="soft T1u triplet at Gamma, projector onto Ti [111]; static E(Q) argmin",
             fixed_minim_struct=False, fixed_gradi_op="gc", free_minim_struct=True,
             free_gradi_op="all", neglect_symmetries=bool(args.neglect_symmetries))
    return r


def apply_threads(n) -> dict:
    if not n:
        return {"threads": None}
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(int(n))
    info = {"threads": int(n), "threadpoolctl": False, "torch": None}
    try:
        from threadpoolctl import threadpool_limits
        globals()["_TPL"] = threadpool_limits(int(n))     # keep the limiter alive
        info["threadpoolctl"] = True
    except Exception:
        pass
    if "torch" in sys.modules:
        try:
            sys.modules["torch"].set_num_threads(int(n))
            info["torch"] = int(n)
        except Exception:
            pass
    return info


def provenance(args) -> dict:
    p = sss.provenance(MODEL, args.dry_run, args.device)
    p.update(centroid_script=SCRIPT, centroid_script_sha256=sss.sha256(Path(__file__)),
             threads=args.threads, prereg=PREREG, prereg_sha256=sss.sha256(REPO / PREREG))
    if not args.dry_run:
        try:
            import sscha.Ensemble
            p["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
            p["schamodules"] = sss.schamodules_linkage()
        except Exception as exc:
            p["sscha_import_error"] = f"{type(exc).__name__}: {exc}"
    return p


def _flog(x, fmt="+.4f"):
    return "-" if x is None else format(x, fmt)


# ================================================================== reference + calculator ====

def gamma_only(dyn):
    """(40-atom Gamma-only Phonons, how). GenerateSupercellDyn first; the manual route (the same
    construction: generate_supercell + GetRealSpaceFC) only if it is missing or returns no Phonons."""
    import cellconstructor.Phonons as CCP
    sc = tuple(int(x) for x in dyn.GetSupercell())
    try:
        d = dyn.GenerateSupercellDyn(sc)
        if isinstance(d, CCP.Phonons):
            return d, "Phonons.GenerateSupercellDyn"
    except (AttributeError, TypeError):
        pass
    st = dyn.structure.generate_supercell(sc)
    d = CCP.Phonons(st, 1)
    d.dynmats[0] = np.asarray(dyn.GetRealSpaceFC(sc), dtype=np.complex128)
    d.q_tot = [np.zeros(3)]
    d.q_stars = [[np.zeros(3)]]
    return d, "manual: structure.generate_supercell + GetRealSpaceFC"


def build_reference(atoms, calc, R: dict, ry, dry_run: bool, relax: bool = True) -> dict:
    """Relaxed cubic primitive cell and, unless dry_run, the cellconstructor dyns: harmonic and
    start A in q-space (the grid's), both as 40-atom Gamma-only dyns, with the conversion check.
    Dry run: an ASE prim.repeat(2,2,2) stands in for the cellconstructor supercell (cell-major atom
    order, unlike phonopy's atom-major one), so the mapper still has a real permutation to find."""
    from mlip_dynstab.harmonic import _relax
    prim = atoms.copy()
    prim.calc = calc
    if R["relax"] and relax:
        prim = _relax(prim, fmax=R["fmax"])
    out = {"prim": prim, "relaxed_cellpar": [float(x) for x in prim.cell.cellpar()]}
    if dry_run:
        sc = prim.repeat(SC)
        out.update(cc_symbols=sc.get_chemical_symbols(), r0=sc.get_positions().copy(),
                   cell=np.asarray(sc.cell.array, float).copy(),
                   order="ASE repeat (dry-run stand-in for the cellconstructor supercell)")
        return out
    from mlip_dynstab.finite_t import _cc_dyn_from_phonopy
    tol = R["imag_tol_thz"]
    dyn_harm = _cc_dyn_from_phonopy(prim, calc, SC, R["disp"])
    dyn_start, start_rec = sss.build_start(dyn_harm, "A", R["start_b_thz"], ry)
    harm_sc, how_h = gamma_only(dyn_harm)
    start_sc, how_s = gamma_only(dyn_start)
    conv = {}
    for name, a, b in (("harmonic", dyn_harm, harm_sc), ("startA", dyn_start, start_sc)):
        wa = np.sort(sss.dyn_freqs(a)[0]) * ry
        wb = np.sort(sss.dyn_freqs(b)[0]) * ry
        conv[name] = float(np.max(np.abs(wa - wb))) if wa.shape == wb.shape else None
    ok = all(v is not None and v <= CONVERSION_TOL_THZ for v in conv.values())
    st = start_sc.structure
    out.update(dyn_harm=dyn_harm, dyn_start=dyn_start, dyn_harm_sc=harm_sc, dyn_start_sc=start_sc,
               start_record=start_rec,
               harmonic_q=sss.freq_summary(sss.dyn_freqs(dyn_harm)[0], ry, tol),
               gamma_only={"method": how_s, "method_harmonic": how_h,
                           "nat": int(st.N_atoms), "nq": len(start_sc.q_tot),
                           "supercell": [int(x) for x in start_sc.GetSupercell()],
                           "max_abs_freq_diff_thz": conv, "tol_thz": CONVERSION_TOL_THZ,
                           "conversion_ok": bool(ok)},
               cc_symbols=[str(s) for s in st.atoms], r0=np.asarray(st.coords, float).copy(),
               cell=np.asarray(st.unit_cell, float).copy(),
               order="cellconstructor Gamma-only supercell (GenerateSupercellDyn)")
    if not ok:
        raise RuntimeError(f"Gamma-only conversion changed the spectrum: {conv} THz")
    return out


def sc_atoms(symbols, positions, cell):
    from ase import Atoms
    return Atoms(symbols=list(symbols), positions=np.asarray(positions, float),
                 cell=np.asarray(cell, float), pbc=True)


def static_scan(atoms, calc, u, qs) -> np.ndarray:
    """Static MLIP energy along a pattern, eV per cell relative to Q = 0."""
    pos0 = atoms.get_positions()
    a = atoms.copy()
    a.calc = calc
    e0 = float(a.get_potential_energy())
    out = []
    for q in qs:
        b = atoms.copy()
        b.set_positions(pos0 + float(q) * np.asarray(u, float))
        b.calc = calc
        out.append(float(b.get_potential_energy()) - e0)
    return np.asarray(out)


def fd_curvature(atoms, calc, u, h=FD_H_ANG) -> dict:
    """Central-difference curvature of the MLIP energy along u, as a frequency with the pattern's
    effective mass (ASE masses, as the screen's M_eff). Independent of how u was mapped."""
    e = static_scan(atoms, calc, u, (-h, h))
    k = float(e[0] + e[1]) / h ** 2                       # E(0) = 0 by construction
    m = float(np.sum(atoms.get_masses()[:, None] * np.asarray(u, float) ** 2))
    f = math.copysign(math.sqrt(abs(k) / m), k) * THZ_PER_SQRT_EV_A2_AMU if m > 0 else None
    return {"h_ang": h, "k_ev_per_ang2": k, "m_eff_amu": m, "freq_thz": f}


def check_reference_vs_pattern(ref: dict, pat: dict) -> dict:
    """An arm's own relaxed reference must be the gate's (same atom order, positions, cell)."""
    if list(ref["cc_symbols"]) != list(pat["symbols"]):
        raise RuntimeError("supercell atom order differs from the gate's pattern file")
    dcell = float(np.max(np.abs(np.asarray(ref["cell"]) - np.asarray(pat["cell"]))))
    dr = float(np.max(np.abs(min_image(np.asarray(ref["r0"]) - np.asarray(pat["r0"]), ref["cell"]))))
    rec = {"pattern_file": pat["path"], "pattern_sha256": pat["sha256"], "max_abs_cell_diff_ang": dcell,
           "max_abs_r0_diff_ang": dr, "tol_cell_ang": CELL_TOL_ANG, "tol_r0_ang": R0_TOL_ANG,
           "q_gamma_ang": pat["q_gamma"], "m_band_gate_passed": pat["m_band_passed"]}
    if dcell > CELL_TOL_ANG or dr > R0_TOL_ANG:
        raise RuntimeError(f"relaxed reference differs from the gate's: cell {dcell:.2e} A, "
                           f"positions {dr:.2e} A")
    return rec


def centroid_record(R, R0, cell, pat, R_start=None) -> dict:
    R = np.asarray(R, float)
    d = min_image(R - np.asarray(R0, float), cell)
    pm = pattern_projection(R, R0, cell, pat["u_m"])
    pg = pattern_projection(R, R0, cell, pat["u_g"])
    sym = list(pat["symbols"])
    mean = {s: np.mean(d[[i for i, x in enumerate(sym) if x == s]], axis=0).tolist()
            for s in dict.fromkeys(sym)}
    rec = {"max_abs_component_from_cubic_ang": float(np.max(np.abs(d))),
           "max_atom_disp_from_cubic_ang": float(np.max(np.linalg.norm(d, axis=1))),
           "Q_M_ang": pm["Q_ang"], "residual_off_M_ang": pm["residual_norm_ang"],
           "Q_G111_ang": pg["Q_ang"], "mean_disp_by_species_ang": mean}
    if R_start is not None:
        ds = min_image(R - np.asarray(R_start, float), cell)
        rec["max_abs_move_from_start_ang"] = float(np.max(np.abs(ds)))
    return rec


def load_pattern(gdir: Path, dry_run: bool) -> dict:
    g = load_json(gdir / "gate.json")
    if not g or g.get("status") != "ok":
        raise SystemExit(f"{gdir / 'gate.json'}: no finished gate; run --arm gate first")
    if bool(g.get("dry_run")) != bool(dry_run):
        raise SystemExit(f"{gdir / 'gate.json'} was written with dry_run={g.get('dry_run')}; refusing "
                         "to mix mocked and real runs")
    if not (g.get("checks") or {}).get("plumbing_ok"):
        raise SystemExit(f"{gdir / 'gate.json'}: the gate's plumbing checks failed; see the file")
    p = gdir / "pattern.npz"
    sha = sss.sha256(p)
    if sha != g.get("pattern_sha256"):
        raise SystemExit(f"{p} does not match the sha256 in gate.json")
    z = np.load(p, allow_pickle=False)
    return {"u_m": np.asarray(z["u_m"], float), "u_g": np.asarray(z["u_g"], float),
            "r0": np.asarray(z["r0"], float), "cell": np.asarray(z["cell"], float),
            "symbols": [str(s) for s in z["symbols"]], "q_gamma": float(z["q_gamma"]),
            "path": rel(p), "sha256": sha, "m_band_passed": bool(g["checks"].get("m_band"))}


# ============================================================================== gate arm ====

def m_pattern_screen(ph) -> dict:
    """The screen's M pattern and the M-band gate."""
    from mlip_dynstab.finite_t import _mode_pattern
    ph.run_qpoints([list(M_Q)], with_eigenvectors=False)
    fr = np.asarray(ph.qpoints.frequencies, float)[0]
    ib = int(np.argmin(fr))
    f = float(fr[ib])
    band = {"q": list(M_Q), "band": ib, "freqs_at_q_thz": fr.tolist(), "freq_thz": f,
            "degeneracy": int(np.sum(np.abs(fr - f) <= DEG_TOL_THZ)),
            "target_thz": M_GATE_THZ, "tol_thz": M_GATE_TOL_THZ,
            "abs_diff_thz": abs(f - M_GATE_THZ), "passed": bool(abs(f - M_GATE_THZ) <= M_GATE_TOL_THZ)}
    ph.run_modulations(dimension=list(SC), phonon_modes=[[list(M_Q), ib, 1.0, 0.0]])
    mods, sc_ph = ph.get_modulations_and_supercell()
    m = np.asarray(mods[0])
    u = normalize_max(np.real(m))
    sc = {"symbols": [str(s) for s in sc_ph.symbols],
          "frac": np.asarray(sc_ph.scaled_positions, float) % 1.0,
          "cell": np.asarray(sc_ph.cell, float)}
    # the screen's own minimal cell, built by the screen's own function
    dim, base, u212, meff212 = _mode_pattern(ph, list(M_Q), ib, 2)
    cart = sc["frac"] @ sc["cell"]
    f212 = frac_of(cart, np.asarray(base.cell.array, float)) % 1.0
    assign = []
    for i in range(len(f212)):
        d = (np.asarray(base.get_scaled_positions(), float) % 1.0) - f212[i]
        d -= np.round(d)
        assign.append(int(np.argmin(np.max(np.abs(d), axis=1))))
    tiled = np.asarray(u212, float)[assign]
    s = 1.0 if float(np.sum(tiled * u)) >= 0 else -1.0
    diff = float(np.max(np.abs(u - s * tiled)))
    pat = {"dim_modulation": list(SC), "screen_dim": [int(x) for x in dim],
           "imag_over_real": float(np.abs(np.imag(m)).max() / max(np.abs(np.real(m)).max(), 1e-300)),
           "max_abs_diff_vs_screen_cell_tiled": diff, "sign_vs_screen_cell": s,
           "consistent_with_screen_cell": bool(diff < 1e-8),
           "m_eff_screen_cell_amu": float(meff212), "m_eff_ledger_amu": M_EFF_LEDGER_AMU,
           "m_eff_rel_diff": abs(float(meff212) - M_EFF_LEDGER_AMU) / M_EFF_LEDGER_AMU,
           "u_dot_u": float(np.sum(u * u))}
    return {"band": band, "pattern": pat, "u_ph": u, "sc": sc}


def gamma_pattern_screen(ph, sc: dict) -> dict:
    """Soft T1u triplet at Gamma -> [111] member -> tiled onto the modulation supercell."""
    ph.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    fr = np.asarray(ph.qpoints.frequencies, float)[0]
    ev = np.asarray(ph.qpoints.eigenvectors)[0]
    masked = fr.copy()
    masked[np.argsort(np.abs(fr))[:3]] = np.inf          # acoustic: the three nearest zero
    fmin = float(np.min(masked))
    deg = [int(i) for i in np.flatnonzero(np.abs(masked - fmin) <= DEG_TOL_THZ)]
    prim = ph.primitive
    psym = [str(s) for s in prim.symbols]
    ti = psym.index("Ti")
    u_prim, w, imag_rel = gamma_111_vector(ev[:, deg], np.asarray(prim.masses, float), ti)
    img = assign_to_prim(sc["frac"], sc["symbols"], SC, np.asarray(prim.scaled_positions, float),
                         psym)
    u = normalize_max(u_prim[img])
    return {"freqs_thz": fr.tolist(), "soft_thz": fmin, "bands": deg, "degeneracy": len(deg),
            "ti_projection_weight": w, "projector_imag_rel": imag_rel,
            "u_prim": u_prim.tolist(), "u_dot_u": float(np.sum(u * u)), "u_ph": u}


def gamma_amplitude(atoms, calc, u) -> dict:
    lo, hi, st = GAMMA_COARSE
    qc = np.arange(lo, hi + 1e-12, st)
    ec = static_scan(atoms, calc, u, qc)
    i = int(np.argmin(ec))
    q0 = float(qc[i])
    qf = np.arange(q0 - GAMMA_FINE_HALF, q0 + GAMMA_FINE_HALF + 1e-12, GAMMA_FINE_STEP)
    qf = qf[qf >= 0.0]
    ef = static_scan(atoms, calc, u, qf)
    j = int(np.argmin(ef))
    qmin = float(qf[j])
    sel = np.abs(qf - qmin) <= 0.002 + 1e-12
    vertex = None
    if sel.sum() >= 3 and 0 < j < len(qf) - 1:
        c = np.polyfit(qf[sel], ef[sel], 2)
        vertex = float(-c[1] / (2 * c[0])) if c[0] > 0 else None
    return {"q_amplitude_ang": qmin, "rule": "argmin of the fine grid", "q_parabola_vertex_ang": vertex,
            "dE_min_meV_per_40": float(ef[j]) * 1e3, "no_static_well": bool(i == 0),
            "coarse": {"Q_ang": qc.tolist(), "dE_meV": (ec * 1e3).tolist()},
            "fine": {"Q_ang": qf.tolist(), "dE_meV": (ef * 1e3).tolist()}}


def cc_pattern_checks(ref: dict, u_m, u_g, ry: float, tol: float) -> dict:
    """The mapped patterns against the cellconstructor harmonic dyn (disp 0.03, Gamma-only, its own
    atom order): Rayleigh frequency and the weight in its imaginary subspace."""
    dyn = ref["dyn_harm_sc"]
    w, pols = sss.dyn_freqs(dyn)
    st = dyn.structure
    m = np.array([float(st.masses[s]) for s in st.atoms])
    D = (pols * (np.sign(w) * w ** 2)) @ np.conj(pols).T
    out = {}
    for name, u in (("M", u_m), ("Gamma111", u_g)):
        e = (np.sqrt(m)[:, None] * np.asarray(u, float)).ravel().astype(np.complex128)
        e /= np.linalg.norm(e)
        q = float(np.real(np.conj(e) @ D @ e))
        c = np.conj(pols).T @ e
        wt = float(np.sum(np.abs(c[(w * ry) < tol]) ** 2))
        out[name] = {"rayleigh_thz": math.copysign(math.sqrt(abs(q)), q) * ry,
                     "weight_in_imaginary_subspace": wt,
                     "ok": bool(q < 0 and wt >= SOFT_WEIGHT_MIN)}
    out["min_weight"] = SOFT_WEIGHT_MIN
    out["ok"] = bool(out["M"]["ok"] and out["Gamma111"]["ok"])
    return out


def compute_gate(doc: dict, args, atoms, calc, fallback: bool, gdir: Path) -> None:
    R = doc["recipe"]
    tm = doc.setdefault("timings_s", {})
    t = time.time()
    ry = None
    if not args.dry_run:
        import cellconstructor as CC
        ry = float(CC.Units.RY_TO_CM * sss.C_THZ_PER_CM)
    ref = build_reference(atoms, calc, R, ry, args.dry_run, relax=not fallback)
    tm["reference"] = time.time() - t
    doc["relaxed_cellpar"] = ref["relaxed_cellpar"]
    doc["reference_order"] = ref["order"]
    if not args.dry_run:
        doc["gamma_only"] = ref["gamma_only"]
        doc["harmonic_q_disp_production"] = ref["harmonic_q"]
    from mlip_dynstab.finite_t import _harmonic_phonon
    t = time.time()
    ph = _harmonic_phonon(ref["prim"], calc, SC, screen_disp())
    tm["screen_fc"] = time.time() - t
    mp = m_pattern_screen(ph)
    doc["m_band"] = mp["band"]
    doc["m_pattern"] = mp["pattern"]
    doc["m_pattern"]["screen_disp_ang"] = screen_disp()
    gp = gamma_pattern_screen(ph, mp["sc"])
    sc = mp["sc"]
    if np.max(np.abs(sc["cell"] - ref["cell"])) > CELL_TOL_ANG:
        raise RuntimeError("phonopy and reference supercell lattices differ")
    perm = map_atoms(sc["frac"], sc["symbols"], frac_of(ref["r0"], ref["cell"]), ref["cc_symbols"])
    u_m, u_g = mp["u_ph"][perm], gp["u_ph"][perm]
    d = np.asarray(sc["frac"])[perm] - (frac_of(ref["r0"], ref["cell"]) % 1.0)
    d -= np.round(d)
    doc["mapping"] = {"n_atoms": len(perm), "tol_frac": MAP_TOL_FRAC, "one_to_one": True,
                      "max_frac_residual": float(np.max(np.abs(d))), "destination_order": ref["order"],
                      "perm_dst_to_phonopy": perm,
                      "is_identity": bool(perm == list(range(len(perm))))}
    cell40 = sc_atoms(ref["cc_symbols"], ref["r0"], ref["cell"])
    t = time.time()
    fdm = fd_curvature(cell40, calc, u_m)
    fdm["band_thz"] = mp["band"]["freq_thz"]
    fdm["passed"] = bool(fdm["freq_thz"] is not None and fdm["freq_thz"] < 0
                         and abs(fdm["freq_thz"] - fdm["band_thz"]) <= FD_TOL_THZ)
    doc["m_pattern"]["fd_check_mapped_order"] = fdm
    es = static_scan(cell40, calc, u_m, M_STATIC_Q)
    doc["m_pattern"]["static_energy_mapped_order"] = [
        {"Q_ang": q, "dE_meV_per_40": float(e) * 1e3, "ledger_2V_meV": 2e3 * ledger_v(q),
         "diff_meV": float(e) * 1e3 - 2e3 * ledger_v(q)} for q, e in zip(M_STATIC_Q, es)]
    fdg = fd_curvature(cell40, calc, u_g)
    fdg["band_thz"] = gp["soft_thz"]
    fdg["passed"] = bool(fdg["freq_thz"] is not None and fdg["freq_thz"] < 0
                         and abs(fdg["freq_thz"] - fdg["band_thz"]) <= FD_TOL_THZ)
    amp = gamma_amplitude(cell40, calc, u_g)
    tm["energy_scans"] = time.time() - t
    gp.pop("u_ph")
    doc["gamma_pattern"] = {**gp, "fd_check_mapped_order": fdg, "static_scan": amp}
    print(f"[gate] M band {mp['band']['freq_thz']:+.4f} THz (gate {M_GATE_THZ} +/- {M_GATE_TOL_THZ}: "
          f"{'PASS' if mp['band']['passed'] else 'FAIL'}), degeneracy {mp['band']['degeneracy']}, "
          f"FD along the mapped pattern {fdm['freq_thz']:+.4f}; Gamma T1u {gp['soft_thz']:+.4f} "
          f"(x{gp['degeneracy']}), [111] amplitude {amp['q_amplitude_ang']:.5f} A "
          f"({amp['dE_min_meV_per_40']:+.2f} meV / 40 atoms)", flush=True)
    if args.dry_run:
        doc["cc_checks"] = {"skipped": "dry run: no cellconstructor"}
        cc_ok = True
    else:
        doc["cc_checks"] = cc_pattern_checks(ref, u_m, u_g, ry, R["imag_tol_thz"])
        cc_ok = doc["cc_checks"]["ok"]
    grid = load_json(GRID_JSON)
    mine = _clean(R)
    if grid is None:
        doc["recipe_vs_grid"] = {"grid_json": rel(GRID_JSON), "matches": None, "note": "grid JSON absent"}
    else:
        g = grid.get("recipe") or {}
        diff = [k for k in g if k != "seed_rule" and mine.get(k) != g[k]]
        doc["recipe_vs_grid"] = {"grid_json": rel(GRID_JSON), "differs": diff,
                                 "ignored": ["seed_rule"], "matches": not diff}
    gdir.mkdir(parents=True, exist_ok=True)
    p = gdir / "pattern.npz"
    tmp = gdir / f"pattern.tmp{os.getpid()}.npz"
    np.savez(tmp, u_m=u_m, u_g=u_g, r0=ref["r0"], cell=ref["cell"],
             symbols=np.array(ref["cc_symbols"], dtype="U3"), q_gamma=np.float64(amp["q_amplitude_ang"]),
             perm=np.asarray(perm, dtype=np.int64), u_m_phonopy=mp["u_ph"],
             r0_phonopy=np.asarray(sc["frac"]) @ np.asarray(sc["cell"]),
             symbols_phonopy=np.array(sc["symbols"], dtype="U3"))
    os.replace(tmp, p)
    doc["pattern_file"] = rel(p)
    doc["pattern_sha256"] = sss.sha256(p)
    checks = {"m_band": mp["band"]["passed"],
              "m_band_nondegenerate": mp["band"]["degeneracy"] == 1,
              "screen_cell_consistent": mp["pattern"]["consistent_with_screen_cell"],
              "m_eff_matches_ledger": mp["pattern"]["m_eff_rel_diff"] < 1e-3,
              "fd_m_mapped": fdm["passed"], "gamma_triplet": gp["degeneracy"] == 3,
              "fd_gamma_mapped": fdg["passed"], "gamma_static_well": not amp["no_static_well"],
              "cc_dyn_patterns": cc_ok,
              "gamma_only_conversion": (True if args.dry_run else ref["gamma_only"]["conversion_ok"]),
              "recipe_matches_grid": doc["recipe_vs_grid"]["matches"] is not False}
    checks["plumbing_ok"] = all(v for k, v in checks.items() if k not in ("m_band",))
    doc["checks"] = checks


def gate_exit_code(doc) -> int:
    if (doc or {}).get("status") != "ok" or not (doc.get("checks") or {}).get("plumbing_ok"):
        return 1
    return 0 if doc["checks"].get("m_band") else 3


def run_gate(args) -> int:
    base = out_base(args)
    gdir = gate_dir_of(args, base)
    gpath = gdir / "gate.json"
    recipe = centroid_recipe(args)
    ex = load_json(gpath)
    if ex is not None:
        if bool(ex.get("dry_run")) != bool(args.dry_run):
            raise SystemExit(f"{gpath} was written with dry_run={ex.get('dry_run')}; move it aside")
        if ex.get("recipe") != _clean(recipe):
            raise SystemExit(f"{gpath} was computed with another recipe; move it aside")
        if ex.get("status") == "ok" and (gdir / "pattern.npz").exists():
            print(f"[skip] gate: done ({rel(gpath)}); checks {ex.get('checks')}", flush=True)
            return gate_exit_code(ex)
        if ex.get("status") == "failed" and not args.retry_failed:
            print(f"[skip] gate failed before ({ex.get('error')}); --retry-failed", flush=True)
            return 1
    with sss.start_lock(base / "locks" / "gate.lock") as held:
        if not held:
            print("[busy] gate: a live process holds it", flush=True)
            return 4
        from mlip_dynstab.systems import get_spec, build_atoms
        if not args.dry_run:
            ok, why = sss.env_can_run(MODEL)
            if not ok:
                raise SystemExit(f"gate: {why}")
        atoms = build_atoms(get_spec(SYSTEM))
        calc, model_version, fallback = sss.load_calculator(MODEL, args.device, args.dry_run, atoms)
        thr = apply_threads(args.threads)
        prov = provenance(args)
        prov["model_version"] = model_version
        prov["thread_limits"] = thr
        doc = {"schema": SCHEMA + "/gate", "study": "R2 centroid scan: patterns and gates",
               "prereg": PREREG, "dry_run": bool(args.dry_run), "unit": unit_ident("gate", None),
               "recipe": recipe, "status": "running", "utc_start": utc_now(),
               "host": socket.gethostname(), "pid": os.getpid(), "provenance": prov,
               "calculator_fallback": bool(fallback),
               "previous": ex if ex and ex.get("status") != "ok" else None}
        atomic_write_json(gpath, doc)
        t0 = time.time()
        try:
            compute_gate(doc, args, atoms, calc, fallback, gdir)
            doc["status"] = "ok"
        except Exception as exc:
            doc["status"] = "failed"
            doc["error"] = f"{type(exc).__name__}: {exc}"
            doc["traceback"] = traceback.format_exc()
            print(f"[gate FAIL] {doc['error']}", flush=True)
        doc["wall_s"] = time.time() - t0
        doc["utc_end"] = utc_now()
        atomic_write_json(gpath, doc)
    print(f"[gate] status {doc['status']}; checks {doc.get('checks')}", flush=True)
    return gate_exit_code(doc)


# ======================================================================= SSCHA arms (real) ====

def final_free_energy(minim, final_dyn, T: float) -> dict:
    """F at the final dyn: the last population reweighted to it, Ensemble.get_free_energy with its
    stochastic error. The minimiser's last history entry is kept beside it (evaluated after the last
    step; a restored, unconverged population's last entry belongs to the discarded step)."""
    fe_h = [sss._scalar(x) for x in getattr(minim, "__fe__")]
    fe_eh = [sss._scalar(x) for x in getattr(minim, "__fe_err__")]
    ens = minim.ensemble
    try:
        ens.update_weights(final_dyn, T)
        fe, err = ens.get_free_energy(return_error=True)
        fe, err = float(np.real(fe)), float(np.real(err))
        kl = float(ens.get_effective_sample_size()) / float(ens.N)
        source = ("last population reweighted to the final dyn; Ensemble.get_free_energy("
                  "return_error=True)")
        fallback = None
    except Exception as exc:              # never lose the arm's primary observable to an API surprise
        fe, err, kl = (fe_h[-1] if fe_h else None), (fe_eh[-1] if fe_eh else None), None
        source = "FALLBACK: the minimiser's last history entry (__fe__, __fe_err__)"
        fallback = f"{type(exc).__name__}: {exc}"
    return {"fe_Ry": fe, "fe_err_Ry": err, "kl_ratio": kl, "n_configs": int(ens.N),
            "source": source, "fallback_reason": fallback,
            "history_last_fe_Ry": fe_h[-1] if fe_h else None,
            "history_last_fe_err_Ry": fe_eh[-1] if fe_eh else None,
            "history_minus_final_Ry": (fe_h[-1] - fe) if (fe_h and fe is not None) else None,
            "per_cell_atoms": int(final_dyn.structure.N_atoms),
            "ensemble_supercell": [int(x) for x in np.asarray(ens.supercell).ravel()],
            "normalisation": "Ensemble.get_free_energy divides by prod(supercell) = 1 here: Ry per "
                             "40-atom Gamma-only cell (the grid's values are per 5-atom cell)"}


def run_arm_sscha(ctx: dict, sp: dict, out: dict) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    import cellconstructor as CC
    import cellconstructor.Phonons  # noqa: F401
    import sscha
    import sscha.Ensemble
    import sscha.SchaMinimizer
    import sscha.Relax

    R, calc, T, arm, pat = ctx["recipe"], ctx["calc"], float(ctx["T"]), ctx["arm"], ctx["pattern"]
    tol = R["imag_tol_thz"]
    ry = float(CC.Units.RY_TO_CM * sss.C_THZ_PER_CM)
    work = sp["work"]
    work.mkdir(parents=True, exist_ok=True)
    tm = out.setdefault("timings_s", {})
    out["julia_ext_available"] = bool(getattr(sscha.Ensemble, "__JULIA_EXT__", False))
    t_all = t = time.time()

    ref = build_reference(ctx["atoms"], calc, R, ry, False)
    tm["relax_and_harmonic_fc"] = time.time() - t
    out["relaxed_cellpar"] = ref["relaxed_cellpar"]
    out["harmonic_pre_fpd"] = ref["harmonic_q"]
    out["start_construction"] = ref["start_record"]
    out["gamma_only"] = ref["gamma_only"]
    out["pattern_source"] = check_reference_vs_pattern(ref, pat)
    q0 = float(ctx["q0"])
    u = pat["u_g"] if arm == "gamma" else pat["u_m"]
    dyn0, dyn_harm_sc = ref["dyn_start_sc"], ref["dyn_harm_sc"]
    R0, cell = np.asarray(ref["r0"], float), np.asarray(ref["cell"], float)
    dyn = dyn0.Copy()
    dyn.structure.coords[:, :] = R0 + q0 * u
    R_start = np.asarray(dyn.structure.coords, float).copy()
    out["centroid_start"] = {"q0_ang": q0, "pattern": "gamma111" if arm == "gamma" else "M",
                             **centroid_record(R_start, R0, cell, pat)}
    w_harm, p_harm = sss.dyn_freqs(dyn_harm_sc)
    dyn_start = dyn.Copy()
    w_start, p_start = sss.dyn_freqs(dyn_start)
    out["harmonic_gamma_only"] = sss.freq_summary(w_harm, ry, tol)
    out["start"] = sss.freq_summary(w_start, ry, tol)
    out["dyn_meta"] = {"nqirr": int(dyn.nqirr), "nq": len(dyn.q_tot), "nat_prim": len(ref["prim"]),
                       "nat_supercell": int(dyn.structure.N_atoms), "cell_form": "gamma_only"}
    ctx["flush"]()

    out["global_seed"] = int(R["seed"]) + arm_seed_offset(ctx["tag"])
    np.random.seed(out["global_seed"])
    grad_log: list = []
    Ens = sss.real_error_ensemble_class(sscha.Ensemble.Ensemble, grad_log, R["converge_min_kl_ratio"])
    ens = Ens(dyn.Copy(), T, supercell=dyn.GetSupercell())
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ens, root_representation=R["root_representation"])
    minim.use_julia = bool(R["use_julia"])
    minim.min_step_dyn = R["min_step_dyn"]
    minim.meaningful_factor = R["meaningful_factor"]
    minim.kong_liu_ratio = R["kong_liu_ratio"]
    minim.max_ka = int(R["max_steps_per_pop"]) * int(R["max_pop"]) + 1     # backstop; see the grid
    free = arm == "free"
    minim.minim_struct = bool(R["free_minim_struct"] if free else R["fixed_minim_struct"])
    minim.gradi_op = R["free_gradi_op"] if free else R["fixed_gradi_op"]
    if R["neglect_symmetries"]:
        if not hasattr(minim, "neglect_symmetries"):
            raise RuntimeError("--neglect-symmetries: SSCHA_Minimizer has no neglect_symmetries")
        minim.neglect_symmetries = True
    relaxer = sscha.Relax.SSCHA(minim, ase_calculator=calc, N_configs=R["n_configs"],
                                max_pop=R["max_pop"], save_ensemble=False)

    # centroid probe at every flush (population ends and every 50 steps), before the JSON write
    base_flush = ctx["flush"]
    track = out.setdefault("centroid_track", [])

    def flush():
        try:
            Rn = np.asarray(relaxer.minim.dyn.structure.coords, float)
            c = centroid_record(Rn, R0, cell, pat, R_start)
            track.append({"utc": utc_now(), "pop": int(getattr(relaxer.minim, "population", -1)),
                          "Q_M_ang": c["Q_M_ang"], "Q_G111_ang": c["Q_G111_ang"],
                          "max_abs_move_from_start_ang": c["max_abs_move_from_start_ang"]})
        except Exception as exc:
            track.append({"utc": utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        base_flush()

    ctx["flush"] = flush
    rec, ctrl = sss.wire_conv_hooks(relaxer, ry, grad_log, R, ctx, out)
    out["stopping_criteria"] = sss.conv_stopping_criteria(minim, relaxer, R)
    out["stopping_criteria"]["neglect_symmetries"] = bool(getattr(minim, "neglect_symmetries", False))
    t = time.time()
    with sss.chdir(work):
        converged = bool(relaxer.relax(get_stress=False))
    tm["sscha_relax"] = time.time() - t
    final_dyn = relaxer.minim.dyn
    n_pops = int(relaxer.start_pop) - 1
    pop_end = sss.population_end_dyns(work, n_pops, final_dyn.nqirr, dyn_start, ry, tol)
    out["relax"] = sss.conv_relax_block(rec, ctrl, pop_end, converged, n_pops, R, grad_log)
    raw = {k: [sss._scalar(x) for x in getattr(relaxer.minim, f"__{k}__")]
           for k in ("fe", "fe_err", "gc", "gc_err", "gw", "gw_err", "KL")}
    raw["good_kasteps"] = [int(i) for i in getattr(relaxer.minim, "__good_kasteps__")]
    out["relax"]["minimizer_raw_histories"] = raw
    out["relax"]["n_symmetries"] = getattr(relaxer.minim, "N_symmetries", None)
    out.pop("relax_partial", None)

    R_fin = np.asarray(final_dyn.structure.coords, float)
    out["centroid_final"] = centroid_record(R_fin, R0, cell, pat, R_start)
    if not free:
        mv = out["centroid_final"]["max_abs_move_from_start_ang"]
        out["fixed_centroid_check"] = {"max_abs_move_ang": mv, "tol_ang": FIXED_TOL_ANG,
                                       "passed": bool(mv <= FIXED_TOL_ANG)}
    t = time.time()
    out["free_energy"] = final_free_energy(relaxer.minim, final_dyn, T)
    tm["free_energy"] = time.time() - t
    w_aux, p_aux = sss.dyn_freqs(final_dyn)
    out["final_aux"] = sss.freq_summary(w_aux, ry, tol)
    final_dyn.save_qe(str(work / "dyn_final_aux_"))
    ctx["flush"]()
    if not free and not out["fixed_centroid_check"]["passed"]:
        raise RuntimeError(f"fixed centroid moved by {out['fixed_centroid_check']['max_abs_move_ang']:.3e}"
                           " A with minim_struct False")

    t = time.time()
    with sss.chdir(work):
        he = sscha.Ensemble.Ensemble(final_dyn, T, supercell=final_dyn.GetSupercell())
        he.generate(int(R["n_hessian"]))
        he.get_energy_forces(calc, compute_stress=False)
    tm["hessian_ensemble"] = time.time() - t
    try:                                       # read-only on he; before the in-place Hessian
        fe, err = he.get_free_energy(return_error=True)
        out["free_energy"]["fresh_ensemble"] = {"fe_Ry": float(np.real(fe)),
                                                "fe_err_Ry": float(np.real(err)),
                                                "n_configs": int(he.N), "kl_ratio": 1.0}
    except Exception as exc:
        out["free_energy"]["fresh_ensemble"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        kl_ref = out["relax"].get("kl_ratio_at_gradient_end") if converged else 1.0
        out["fresh_gradient_check"] = sss.fresh_gradient_check(
            he, final_dyn, np.random.default_rng([sss.CONV_SALT, CENTROID_SALT, out["global_seed"], 1]),
            int(R["n_configs"]), kl_ref, int(R["fresh_splits"]))
    except Exception as exc:
        out["fresh_gradient_check"] = {"error": f"{type(exc).__name__}: {exc}",
                                       "traceback": traceback.format_exc()}
    he.save_bin(str(sp["ensemble"]), 1)
    out["ensemble_dir"] = rel(sp["ensemble"])
    atomic_write_json(sp["marker"], {"utc": utc_now(), "n_configs": int(he.N),
                                     "antithetic_pairs_verified": sss.antithetic_pairs_ok(he),
                                     "ensemble_dir": rel(sp["ensemble"]), "T": T})
    ctx["flush"]()
    t = time.time()
    hess = he.get_free_energy_hessian(include_v4=False)
    tm["hessian"] = time.time() - t
    w_hess, p_hess = sss.dyn_freqs(hess)
    out["hessian"] = sss.freq_summary(w_hess, ry, tol)
    out["hessian"]["dynamically_stable"] = bool(out["hessian"]["min_nonac_thz"] >= tol)
    out["hessian"]["note"] = ("d2F/dR2 at this centroid with Phi relaxed; at a fixed displaced "
                              "centroid F is not stationary in R")
    out["echo"] = {
        "rel_change_final_aux_vs_start": sss.dyn_rel_change(final_dyn, dyn_start),
        "rel_change_hessian_vs_final_aux": sss.dyn_rel_change(hess, final_dyn),
        "harmonic_along_hessian_softmode": sss.freq_along_softmode(w_hess, p_hess, w_harm, p_harm, ry),
        "start_along_hessian_softmode": sss.freq_along_softmode(w_hess, p_hess, w_start, p_start, ry),
        "final_aux_along_hessian_softmode": sss.freq_along_softmode(w_hess, p_hess, w_aux, p_aux, ry)}
    files = {}
    for name, d in (("dyn_harmonic_", ref["dyn_harm"]), ("dyn_start_gamma_", dyn_start),
                    ("dyn_hessian_", hess)):
        d.save_qe(str(work / name))
    for name in ("dyn_harmonic_", "dyn_start_gamma_", "dyn_final_aux_", "dyn_hessian_"):
        files[name] = rel(work / f"{name}1")
    out["dyn_files"] = files
    ctx["flush"]()
    t = time.time()
    out["bootstrap"] = sss.hessian_uncertainty(
        he, ctx["n_boot"], ctx["n_splits"],
        np.random.default_rng([sss.BOOT_SALT, CENTROID_SALT, out["global_seed"]]), ry, tol,
        out["hessian"]["min_nonac_thz"])
    tm["bootstrap"] = time.time() - t
    tm["total"] = time.time() - t_all


def mock_rel_free_energy_meV(arm: str, q: float) -> float:
    """--dry-run scenario only: a quartic with its minimum, -22 meV per 40 atoms, at 0.075 A (the
    single-mode estimate in the prereg), so the summary exercises the SUPPORTED branch."""
    if arm == "free":
        return -23.0
    if arm == "gamma":
        return -5.0
    x = (float(q) / MAIN_Q0) ** 2
    return -22.0 * x * (2.0 - x)


def run_arm_mock(ctx: dict, sp: dict, out: dict) -> None:
    """--dry-run stand-in for run_arm_sscha: same schema, no sscha or cellconstructor. The relax,
    the pattern consistency check and the centroid bookkeeping are real; the SSCHA is the grid's
    run_converged_mock (its hooks, real-error subclass, caps and flushes are the real code); every
    free energy, frequency and centroid move is synthetic and marked so."""
    R, arm, pat = ctx["recipe"], ctx["arm"], ctx["pattern"]
    t = time.time()
    ref = build_reference(ctx["atoms"], ctx["calc"], R, None, True, relax=not ctx["calc_is_fallback"])
    out["pattern_source"] = check_reference_vs_pattern(ref, pat)
    out.setdefault("timings_s", {})["reference"] = time.time() - t
    # run_converged_mock relaxes unless the calculator is a fallback; the prim is relaxed already
    mctx = dict(ctx, atoms=ref["prim"], calc_is_fallback=True,
                ledger={"min_eff_freq_thz": HESS_REF_THZ})
    sss.run_converged_mock(mctx, "A", sp, out)
    out["MOCK"] = ("dry-run: every SSCHA number, free energy and centroid move here is synthetic; the "
                   "relax, the pattern check, the hooks, caps, flushes and the JSON layout are real")
    out["gamma_only"] = {"MOCK": True, "conversion_ok": True}
    q0 = float(ctx["q0"])
    u = pat["u_g"] if arm == "gamma" else pat["u_m"]
    R0, cell = np.asarray(ref["r0"], float), np.asarray(ref["cell"], float)
    R_start = R0 + q0 * u
    R_fin = R0 + MOCK_FREE_Q * pat["u_m"] if arm == "free" else R_start.copy()
    out["centroid_start"] = {"q0_ang": q0, "pattern": "gamma111" if arm == "gamma" else "M",
                             **centroid_record(R_start, R0, cell, pat)}
    out["centroid_final"] = centroid_record(R_fin, R0, cell, pat, R_start)
    if arm != "free":
        mv = out["centroid_final"]["max_abs_move_from_start_ang"]
        out["fixed_centroid_check"] = {"max_abs_move_ang": mv, "tol_ang": FIXED_TOL_ANG,
                                       "passed": bool(mv <= FIXED_TOL_ANG)}
    out["global_seed"] = int(R["seed"]) + arm_seed_offset(ctx["tag"])
    rng = np.random.default_rng([sss.MOCK_SALT, CENTROID_SALT, out["global_seed"]])
    fe = MOCK_FE0_RY + mock_rel_free_energy_meV(arm, q0) / 1e3 / RY_EV + rng.normal(0, MOCK_FE_ERR_RY)
    out["free_energy"] = {"MOCK": True, "fe_Ry": float(fe), "fe_err_Ry": MOCK_FE_ERR_RY,
                          "kl_ratio": 0.97, "per_cell_atoms": 40,
                          "source": "dry-run scenario (mock_rel_free_energy_meV)",
                          "fresh_ensemble": {"MOCK": True, "fe_Ry": float(fe), "fe_err_Ry": 6e-5}}
    out["relax"]["n_symmetries"] = None
    sp["ensemble"].mkdir(parents=True, exist_ok=True)
    np.save(sp["ensemble"] / "mock_r_start.npy", R_start)
    out["ensemble_dir"] = rel(sp["ensemble"])
    atomic_write_json(sp["marker"], {"utc": utc_now(), "MOCK": True, "n_configs": int(R["n_hessian"]),
                                     "ensemble_dir": rel(sp["ensemble"]), "T": T_K})
    ctx["flush"]()


# ====================================================================== SSCHA arm driver ====

def resolve_q0(args):
    if args.arm == "gamma":
        return None
    q0 = args.q0 if args.q0 is not None else (FREE_Q0 if args.arm == "free" else None)
    if q0 is None:
        raise SystemExit("--arm m_scan needs --q0 (one of " + ", ".join(f"{q:g}" for q in SCAN_Q0) + ")")
    allowed = SCAN_Q0 if args.arm == "m_scan" else (FREE_Q0,)
    if not any(abs(float(q0) - a) < 1e-9 for a in allowed) and not args.allow_other_q0:
        raise SystemExit(f"--q0 {q0:g} is not pre-registered for --arm {args.arm} ({allowed}); "
                         "--allow-other-q0 for a test run in its own --out-dir")
    return float(q0)


def arm_should_run(doc, ident: dict, recipe: dict, args, under_lock: bool) -> tuple:
    if doc is None:
        return True, "new"
    where = doc.get("arm_tag", "?")
    if bool(doc.get("dry_run")) != bool(args.dry_run):
        raise SystemExit(f"{where} was written with dry_run={doc.get('dry_run')}; refusing to mix "
                         "mocked and real runs. Move it aside.")
    if doc.get("unit") != _clean(ident):
        raise SystemExit(f"{where}: the JSON holds {doc.get('unit')}, not {ident}")
    if doc.get("recipe") != _clean(recipe):
        raise SystemExit(f"{where} was computed with recipe {doc.get('recipe')}; this run is "
                         f"{recipe}. Refusing to mix; move it aside.")
    run = doc.get("run") or {}
    st = run.get("status")
    if st == "ok":
        return False, "ok"
    if st == "failed":
        if args.retry_failed:
            return True, "retrying a failure (--retry-failed)"
        return False, f"failed ({run.get('error')}); --retry-failed to rerun"
    if st == "running" and not under_lock:
        return True, "running: try the lock"
    return True, f"status {st}; its process is gone (the lock is free): recomputing"


def run_arm(args) -> int:
    base = out_base(args)
    q0 = resolve_q0(args)
    tag = arm_tag(args.arm, q0)
    recipe = centroid_recipe(args)
    ident = unit_ident(args.arm, q0)
    sp = arm_paths(base, tag)
    go, why = arm_should_run(load_json(sp["json"]), ident, recipe, args, False)
    if not go:
        print(f"[skip] {tag}: {why}", flush=True)
        return 0
    with sss.start_lock(base / "locks" / f"{tag}.lock") as held:
        if not held:
            print(f"[busy] {tag}: a live process holds it", flush=True)
            return 4
        ex = load_json(sp["json"])
        go, why = arm_should_run(ex, ident, recipe, args, True)
        if not go:
            print(f"[skip] {tag}: {why}", flush=True)
            return 0
        pat = load_pattern(gate_dir_of(args, base), args.dry_run)
        if q0 is None:
            q0 = pat["q_gamma"]
        if not args.dry_run:
            ok, msg = sss.env_can_run(MODEL)
            if not ok:
                raise SystemExit(f"{tag}: {msg}")
        from mlip_dynstab.systems import get_spec, build_atoms
        atoms = build_atoms(get_spec(SYSTEM))
        calc, model_version, fallback = sss.load_calculator(MODEL, args.device, args.dry_run, atoms)
        thr = apply_threads(args.threads)
        prov = provenance(args)
        prov["model_version"] = model_version
        prov["thread_limits"] = thr
        doc = ex or {"schema": SCHEMA, "study": "R2 multi-mode SSCHA centroid scan", "prereg": PREREG,
                     "dry_run": bool(args.dry_run), "unit": ident, "arm_tag": tag, "arm": args.arm,
                     "recipe": recipe,
                     "recipe_source": "sscha_seed_study.converged_recipe() + centroid_recipe() ("
                                      + SCRIPT + ")",
                     "provenance_created": prov, "previous_runs": []}
        doc.update(provenance_last_run=prov, calculator_fallback=bool(fallback))
        stale = None
        if sp["work"].exists():
            stale = sp["work"].with_name(f"{sp['work'].name}.stale-{int(time.time())}")
            sp["work"].rename(stale)
        if doc.get("run"):
            old = dict(doc["run"])
            old["work_dir_moved_to"] = rel(stale) if stale is not None else None
            doc.setdefault("previous_runs", []).append(old)
        entry = {"arm": args.arm, "q0_ang": q0, "status": "running", "utc_start": utc_now(),
                 "host": socket.gethostname(), "pid": os.getpid(), "work_dir": rel(sp["work"]),
                 "settings": {"n_boot": int(args.n_boot), "n_splits_fallback": int(args.n_splits),
                              "unit_timeout_s": float(args.unit_timeout), "device": args.device,
                              "gate_dir": rel(gate_dir_of(args, base)), "threads": args.threads}}
        doc["run"] = entry

        def flush(doc=doc, entry=entry, path=sp["json"]):
            snap = dict(doc)
            snap["run"] = {k: v for k, v in entry.items() if not str(k).startswith("_")}
            snap["last_written_utc"] = utc_now()
            atomic_write_json(path, snap)

        flush()
        ctx = {"recipe": recipe, "calc": calc, "atoms": atoms, "T": T_K, "arm": args.arm, "q0": q0,
               "tag": tag, "pattern": pat, "n_boot": int(args.n_boot), "n_splits": int(args.n_splits),
               "unit_timeout": float(args.unit_timeout), "flush": flush,
               "calc_is_fallback": fallback, "mock_step_sleep": float(args.dry_run_step_sleep)}
        print(f"===== {tag} (Q0 {q0:.5f} A) {utc_now()} =====", flush=True)
        t0 = time.time()
        try:
            (run_arm_mock if args.dry_run else run_arm_sscha)(ctx, sp, entry)
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
            rec, ctrl = entry.get("_recorder"), entry.get("_controller")
            if rec is not None and "relax" not in entry:
                entry["relax_partial"] = {"reason": "exception", "steps": rec.steps,
                                          "hook_errors": rec.errors,
                                          "pop_start_aux_min_nonac_thz": rec.pop_start,
                                          "controller": ctrl.state() if ctrl else None}
            print(f"[FAIL] {tag}: {entry['error']}", flush=True)
        entry.pop("_recorder", None)
        entry.pop("_controller", None)
        entry["wall_s"] = time.time() - t0
        entry["utc_end"] = utc_now()
        entry["provenance"] = {k: prov.get(k) for k in ("git_head", "host", "packages", "model_version",
                                                         "device", "omp_num_threads", "thread_limits")}
        flush()
    if entry["status"] != "ok":
        return 2
    fe, rx = entry.get("free_energy") or {}, entry.get("relax") or {}
    print(f"[{tag}] F = {_flog(fe.get('fe_Ry'), '.7f')} +/- {_flog(fe.get('fe_err_Ry'), '.2e')} Ry per "
          f"40 atoms ({(fe.get('fe_err_Ry') or 0) * RY_EV * 1e3:.2f} meV) | conv {rx.get('converged')} stop "
          f"{rx.get('stop_reason')} npop {rx.get('n_populations')} | hess "
          f"{_dig(entry, 'hessian', 'min_nonac_thz')} THz | Q_M start "
          f"{_dig(entry, 'centroid_start', 'Q_M_ang')} final {_dig(entry, 'centroid_final', 'Q_M_ang')} "
          f"| {entry['wall_s']:.0f} s", flush=True)
    return 0


# ================================================================================ v4 arm ====

def run_v4_arm(args) -> int:
    base = out_base(args)
    recipe = centroid_recipe(args)
    src_tag = arm_tag("m_scan", 0.0)
    src_p = arm_paths(base, src_tag)
    src = load_json(src_p["json"])
    run0 = (src or {}).get("run") or {}
    if not run0.get("ensemble_dir") or not src_p["marker"].exists():
        raise SystemExit(f"{rel(src_p['json'])}: no saved Hessian ensemble yet (marker "
                         f"{rel(src_p['marker'])}); run --arm m_scan --q0 0 first")
    if bool(src.get("dry_run")) != bool(args.dry_run):
        raise SystemExit("the Q0 = 0 arm was written in the other dry-run mode")
    if src.get("recipe") != _clean(recipe):
        raise SystemExit("the Q0 = 0 arm was computed with another recipe; refusing to pair")
    vjson = base / "v4.json"
    vwork = base / "work" / "v4"
    sub_dir = vwork / "subsample_ensemble"
    prev = load_json(vjson)
    if prev is not None:
        if bool(prev.get("dry_run")) != bool(args.dry_run) or prev.get("recipe") != _clean(recipe):
            raise SystemExit(f"{rel(vjson)} was made in another mode or with another recipe")
        st = (prev.get("v4") or {}).get("status")
        if st == "finished":
            print("[skip] v4 already finished", flush=True)
            return 0
        if st == "timeout" and (prev.get("v4") or {}).get("timeout_s", 0) >= args.v4_timeout:
            print(f"[skip] v4 timed out before at >= {args.v4_timeout:.0f} s; raise --v4-timeout",
                  flush=True)
            return 0
        if st == "failed" and not args.retry_failed:
            print("[skip] v4 failed before; --retry-failed to rerun", flush=True)
            return 0
    with sss.start_lock(base / "locks" / "v4.lock") as held:
        if not held:
            print("[busy] v4: a live process holds it", flush=True)
            return 4
        thr = int(args.threads or sss.cpu_budget()["usable"])
        T = float(T_K)
        tol = recipe["imag_tol_thz"]
        doc = {"schema": SCHEMA + "/v4", "study": "R2 arm 4: include_v4 True vs False",
               "prereg": PREREG, "dry_run": bool(args.dry_run), "recipe": recipe,
               "source_arm": src_tag, "source_json": rel(src_p["json"]),
               "source_ensemble_dir": run0["ensemble_dir"],
               "full_ensemble_v2_min_thz": _dig(run0, "hessian", "min_nonac_thz"),
               "settings": {"n_subsample": int(args.v4_n), "timeout_s": float(args.v4_timeout),
                            "threads": thr, "n_boot": int(args.n_boot)},
               "provenance": provenance(args), "utc_start": utc_now(),
               "previous": prev.get("v4") if prev else None,
               "v4": {"status": "running"}}
        atomic_write_json(vjson, doc)
        if vwork.exists():
            vwork.rename(vwork.with_name(f"v4.stale-{int(time.time())}"))
        vwork.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            rng = np.random.default_rng([CENTROID_SALT, V4_SALT])
            if args.dry_run:
                n_full = int(recipe["n_hessian"])
                paired = True
                pairs = np.sort(rng.choice(n_full // 2, int(args.v4_n) // 2, replace=False))
                idx = np.stack([2 * pairs, 2 * pairs + 1], axis=1).ravel()
                sub_dir.mkdir(parents=True, exist_ok=True)
                np.save(sub_dir / "mock_indices.npy", idx)
                v2 = float(doc["full_ensemble_v2_min_thz"] or HESS_REF_THZ) + 0.003
                v2_sum = {"MOCK": True, "min_nonac_thz": v2}
                boot = {"method": "mock", "std_thz": 0.06, "full_ensemble_min_nonac_thz": v2}
            else:
                import warnings
                warnings.filterwarnings("ignore")
                import cellconstructor as CC
                import sscha.Ensemble
                ry = float(CC.Units.RY_TO_CM * sss.C_THZ_PER_CM)
                full = sscha.Ensemble.load_ensemble_bin(str(sss.from_rel(run0["ensemble_dir"])), 1, T)
                n_full = int(full.N)
                paired = sss.antithetic_pairs_ok(full)
                k = int(args.v4_n)
                if paired:
                    pairs = np.sort(rng.choice(n_full // 2, k // 2, replace=False))
                    idx = np.stack([2 * pairs, 2 * pairs + 1], axis=1).ravel()
                else:
                    idx = np.sort(rng.choice(n_full, k, replace=False))
                sub = sss.resample_ensemble(full, idx)
                sub.save_bin(str(sub_dir), 1)
                t = time.time()
                h = sub.get_free_energy_hessian(include_v4=False)
                v2_sum = {**sss.freq_summary(sss.dyn_freqs(h)[0], ry, tol), "wall_s": time.time() - t}
                v2 = v2_sum["min_nonac_thz"]
                boot = sss.hessian_uncertainty(sub, int(args.n_boot), int(args.n_splits),
                                               np.random.default_rng([sss.BOOT_SALT, V4_SALT]), ry,
                                               tol, v2)
            doc["subsample"] = {"n_full": n_full, "n": int(len(idx)), "antithetic_pairs": bool(paired),
                                "without_replacement": True, "indices": [int(i) for i in idx],
                                "rng": f"default_rng([{CENTROID_SALT}, {V4_SALT}])",
                                "dir": rel(sub_dir)}
            doc["v2_subsample"] = v2_sum
            doc["v2_subsample_bootstrap"] = boot
            atomic_write_json(vjson, doc)
            # sscha_seed_study.run_v4 takes a seed-study unit document; this is the minimal one that
            # points it at the subsample (seed "0" is its own key, not a seed of this study).
            fake = {"seeds": {"0": {"status": "ok", "ensemble_dir": rel(sub_dir),
                                    "hessian": {"min_nonac_thz": v2}}},
                    "recipe": {"imag_tol_thz": tol}, "unit": {"T": T}}
            block = sss.run_v4(fake, float(args.v4_timeout), thr, args.dry_run,
                               float(args.dry_run_v4_sleep))
            v4m = block.get("v4_min_nonac_thz")
            sd = boot.get("std_full_estimate_thz") if boot.get("method") == "disjoint_splits" \
                else boot.get("std_thz")
            block.update(
                v2_subsample_min_thz=v2, v2_subsample_boot_sd_thz=sd,
                shift_thz=(None if v4m is None else float(v4m) - float(v2)),
                same_sign=(None if v4m is None else bool((float(v4m) >= 0) == (float(v2) >= 0))),
                same_stability_call=(None if v4m is None else bool((float(v4m) >= tol) == (float(v2) >= tol))))
            doc["v4"] = block
        except Exception as exc:
            doc["v4"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(), "timeout_s": float(args.v4_timeout)}
        doc["wall_s"] = time.time() - t0
        doc["utc_end"] = utc_now()
        atomic_write_json(vjson, doc)
    b = doc["v4"]
    print(f"[v4] {b.get('status')}: v2 (subsample) {b.get('v2_subsample_min_thz')} THz, v4 "
          f"{b.get('v4_min_nonac_thz')} THz, shift {b.get('shift_thz')}, same sign {b.get('same_sign')}, "
          f"round trip ok {b.get('v4_false_roundtrip_ok')}, {doc['wall_s']:.0f} s", flush=True)
    return 0 if b.get("status") in ("finished", "timeout") else 2


# ============================================================================= summarize ====

def arm_digest(doc) -> dict:
    r = (doc or {}).get("run") or {}
    fe = r.get("free_energy") or {}
    klg = _dig(r, "relax", "kl_ratio_at_gradient_end")
    conv = bool(r.get("status") == "ok" and _dig(r, "relax", "converged"))
    kl_ok = None if klg is None else bool(klg >= sss.CONV_MIN_KL)
    boot_sd = (_dig(r, "bootstrap", "std_full_estimate_thz")
               if _dig(r, "bootstrap", "method") == "disjoint_splits" else _dig(r, "bootstrap", "std_thz"))
    return {"arm": (doc or {}).get("arm"), "q0_ang": r.get("q0_ang"), "status": r.get("status"),
            "dry_run": (doc or {}).get("dry_run"), "error": r.get("error"),
            "converged": bool(conv and kl_ok is not False), "library_converged": conv,
            "kl_ratio_at_gradient_end": klg, "stop_reason": _dig(r, "relax", "stop_reason"),
            "n_populations": _dig(r, "relax", "n_populations"),
            "n_symmetries": _dig(r, "relax", "n_symmetries"),
            "fe_Ry": fe.get("fe_Ry"), "fe_err_Ry": fe.get("fe_err_Ry"),
            "fe_history_last_Ry": fe.get("history_last_fe_Ry"),
            "fe_fresh_Ry": _dig(fe, "fresh_ensemble", "fe_Ry"),
            "fe_fresh_err_Ry": _dig(fe, "fresh_ensemble", "fe_err_Ry"),
            "hessian_min_thz": _dig(r, "hessian", "min_nonac_thz"), "boot_sd_thz": boot_sd,
            "final_aux_min_thz": _dig(r, "final_aux", "min_nonac_thz"),
            "Q_M_start_ang": _dig(r, "centroid_start", "Q_M_ang"),
            "Q_M_final_ang": _dig(r, "centroid_final", "Q_M_ang"),
            "Q_G111_final_ang": _dig(r, "centroid_final", "Q_G111_ang"),
            "residual_off_M_final_ang": _dig(r, "centroid_final", "residual_off_M_ang"),
            "max_move_from_start_ang": _dig(r, "centroid_final", "max_abs_move_from_start_ang"),
            "fixed_centroid_held": _dig(r, "fixed_centroid_check", "passed"),
            "fresh_R_over_expected": _dig(r, "fresh_gradient_check", "split_null", "R_over_expected"),
            "pattern_sha256": _dig(r, "pattern_source", "pattern_sha256"),
            "wall_s": r.get("wall_s"), "utc_end": r.get("utc_end")}


def summarize(args) -> int:
    base = out_base(args)
    gate = load_json(gate_dir_of(args, base) / "gate.json")
    docs = {}
    for fp in sorted(base.glob("*.json")):
        d = load_json(fp)
        if isinstance(d, dict) and d.get("schema") == SCHEMA and d.get("arm_tag"):
            docs[d["arm_tag"]] = d
    dig = {t: arm_digest(d) for t, d in docs.items()}
    ref_tag = arm_tag("m_scan", 0.0)
    ref_recipe = (docs.get(ref_tag) or {}).get("recipe")
    recipe_ok = {t: (d.get("recipe") == ref_recipe) for t, d in docs.items()}
    for t in dig:
        dig[t]["recipe_matches_q0_arm"] = recipe_ok[t]
    shas = {x["pattern_sha256"] for x in dig.values() if x.get("pattern_sha256")}
    gate_sha = (gate or {}).get("pattern_sha256")

    def f_of(tag, src="fresh"):
        """Free energy relative to Q0 = 0. Primary (prereg deviation D4, 2026-10-21): the fresh
        n_hessian ensemble drawn at the final matrix (KL = 1, independent of the minimisation);
        secondary: the last population reweighted to the final matrix."""
        k, ke = ("fe_fresh_Ry", "fe_fresh_err_Ry") if src == "fresh" else ("fe_Ry", "fe_err_Ry")
        x = dig.get(tag)
        f0 = dig.get(ref_tag)
        if x is None or f0 is None or x[k] is None or f0[k] is None:
            return None
        return {"fe_meV": (x[k] - f0[k]) * RY_EV * 1e3,
                "err_meV": None if x[ke] is None else x[ke] * RY_EV * 1e3,
                "converged": bool(x["converged"] and recipe_ok.get(tag)), "q0_ang": x["q0_ang"]}

    h0 = (dig.get(ref_tag) or {}).get("hessian_min_thz")
    sd0 = (dig.get(ref_tag) or {}).get("boot_sd_thz")
    q0h = None if h0 is None else bool(abs(h0 - HESS_REF_THZ) <= HESS_GATE_NSD * HESS_REF_SD_THZ)
    # The prereg's two gates plus two plumbing conditions without which no comparison means anything
    # (the gate's own checks, one pattern file for every arm). A fixed arm whose centroid moved has
    # already failed (run_arm_sscha raises), so it drops out as unconverged; listed below for reading.
    gates = {"m_band": _dig(gate, "checks", "m_band") if (gate or {}).get("status") == "ok" else None,
             "q0_hessian": q0h,
             "gate_plumbing": _dig(gate, "checks", "plumbing_ok"),
             "one_pattern_for_all_arms": bool(shas == {gate_sha}) if shas else None}
    fixed_ok = [x["fixed_centroid_held"] for x in dig.values()
                if x["arm"] in ("m_scan", "gamma") and x["status"] == "ok"]
    scan = [f for f in (f_of(arm_tag("m_scan", q)) for q in SCAN_Q0) if f is not None]
    primary = decide(f_of(ref_tag), f_of(arm_tag("m_scan", MAIN_Q0)), scan, gates)
    scan_rw = [f for f in (f_of(arm_tag("m_scan", q), "reweighted") for q in SCAN_Q0) if f is not None]
    secondary_rw = decide(f_of(ref_tag, "reweighted"), f_of(arm_tag("m_scan", MAIN_Q0), "reweighted"),
                          scan_rw, gates)
    table = []
    for q in SCAN_Q0:
        t = arm_tag("m_scan", q)
        f = f_of(t)
        x = dig.get(t) or {}
        table.append({"q0_ang": q, "status": x.get("status"), "converged": x.get("converged"),
                      "dF_meV_per_40": None if f is None else f["fe_meV"],
                      "err_meV": None if f is None else f["err_meV"],
                      "static_2V_ledger_meV": 2e3 * ledger_v(q),
                      "hessian_min_thz": x.get("hessian_min_thz"), "stop_reason": x.get("stop_reason")})
    free_t = arm_tag("free", FREE_Q0)
    ff, fg = f_of(free_t), f_of("gamma")
    v4doc = load_json(base / "v4.json") or {}
    v4 = v4doc.get("v4") or {}
    grid = load_json(GRID_JSON)
    norm = None
    if grid and dig.get(ref_tag, {}).get("fe_Ry") is not None:
        st = ((grid.get("run") or {}).get("relax") or {}).get("steps") or [{}]
        gfe, gerr = st[-1].get("fe_Ry"), st[-1].get("fe_err_Ry")
        if gfe is not None:
            x = dig[ref_tag]
            dd = (x["fe_Ry"] - 8.0 * gfe) * RY_EV * 1e3
            ss = math.hypot((x["fe_err_Ry"] or 0) * RY_EV * 1e3, 8 * (gerr or 0) * RY_EV * 1e3)
            norm = {"grid_fe_per_5atom_Ry": gfe, "grid_fe_err_per_5atom_Ry": gerr,
                    "this_fe_per_40atom_Ry": x["fe_Ry"], "diff_meV_per_40": dd, "sigma_meV": ss,
                    "z": dd / ss if ss > 0 else None,
                    "note": "F(Q0=0) Gamma-only vs 8 x the grid's q-space F of the same unit; the "
                            "Gamma-only cell has fewer symmetry constraints, so a small negative "
                            "difference is allowed; a factor-8 normalisation slip is ~20 Ry"}
    summary = {
        "schema": SCHEMA + "/summary", "utc": utc_now(), "prereg": PREREG,
        "dry_run_any": any(bool(d.get("dry_run")) for d in docs.values()) or bool((gate or {}).get("dry_run")),
        "units": {"F": "meV per 40-atom 2x2x2 cell, relative to F(Q0 = 0) of the fixed-centroid arm",
                  "Ry_to_eV": RY_EV},
        "gates": {"values": gates,
                  "fixed_centroids_held_ok_arms": (all(v is True for v in fixed_ok) if fixed_ok
                                                   else None),
                  "m_band": {k: _dig(gate, "m_band", k) for k in ("freq_thz", "target_thz", "tol_thz",
                                                                    "abs_diff_thz", "passed")},
                  "q0_hessian": {"hessian_min_thz": h0, "boot_sd_thz": sd0, "ref_thz": HESS_REF_THZ,
                                 "ref_sd_thz": HESS_REF_SD_THZ, "n_sd": HESS_GATE_NSD,
                                 "z_ref_sd": None if h0 is None else (h0 - HESS_REF_THZ) / HESS_REF_SD_THZ,
                                 "z_combined_sd": (None if h0 is None or sd0 is None else
                                                   (h0 - HESS_REF_THZ) / math.hypot(sd0, HESS_REF_SD_THZ)),
                                 "rule": "within 3 SD of +1.98 THz, SD = the prereg's 0.055 THz",
                                 "passed": q0h},
                  "gate_checks": (gate or {}).get("checks")},
        "primary": primary,
        "primary_free_energy_source": "fresh n_hessian ensemble at the final matrix (prereg deviation D4)",
        "secondary_reweighted_last_population": secondary_rw,
        "m_scan_table": table,
        "free_arm": {"converged": (dig.get(free_t) or {}).get("converged"),
                     "dF_meV_per_40": None if ff is None else ff["fe_meV"],
                     "err_meV": None if ff is None else ff["err_meV"],
                     "Q_M_start_ang": (dig.get(free_t) or {}).get("Q_M_start_ang"),
                     "Q_M_final_ang": (dig.get(free_t) or {}).get("Q_M_final_ang"),
                     "residual_off_M_final_ang": (dig.get(free_t) or {}).get("residual_off_M_final_ang"),
                     "Q_G111_final_ang": (dig.get(free_t) or {}).get("Q_G111_final_ang"),
                     "note": "secondary; reported as measured (prereg)"},
        "gamma_arm": {"converged": (dig.get("gamma") or {}).get("converged"),
                      "q_amplitude_ang": (dig.get("gamma") or {}).get("q0_ang"),
                      "dF_meV_per_40": None if fg is None else fg["fe_meV"],
                      "err_meV": None if fg is None else fg["err_meV"],
                      "static_dE_min_meV_per_40": _dig(gate, "gamma_pattern", "static_scan",
                                                       "dE_min_meV_per_40"),
                      "note": "secondary; reported as measured (prereg)"},
        "v4": {k: v4.get(k) for k in ("status", "v2_subsample_min_thz", "v2_subsample_boot_sd_thz",
                                      "v4_min_nonac_thz", "shift_thz", "same_sign",
                                      "same_stability_call", "v4_false_roundtrip_ok", "wall_s",
                                      "omp_threads", "timeout_s")},
        "v4_full_ensemble_v2_min_thz": v4doc.get("full_ensemble_v2_min_thz"),
        "normalisation_check_vs_grid": norm,
        "arms": dig}
    summary["v4"]["rule"] = ("does not change the sign: v4 and v2 lowest non-acoustic frequencies have "
                             "the same sign; shift reported with the subsample's bootstrap SD")
    out = base / "summary.json"
    atomic_write_json(out, summary)
    print(f"R2 centroid scan summary ({rel(base)}){' [DRY RUN]' if summary['dry_run_any'] else ''}")
    print(f"  gates: {gates}")
    print(f"  {'Q0':>6s} {'status':>7s} {'conv':>5s} {'dF meV/40':>10s} {'err':>6s} {'2V_ledger':>9s} "
          f"{'hess':>7s}")
    for r in table:
        print(f"  {r['q0_ang']:6.3f} {str(r['status']):>7s} {str(r['converged']):>5s} "
              f"{_flog(r['dF_meV_per_40'], '+10.2f'):>10s} {_flog(r['err_meV'], '6.2f'):>6s} "
              f"{r['static_2V_ledger_meV']:+9.1f} {_flog(r['hessian_min_thz'], '+7.3f'):>7s}")
    print(f"  PRIMARY: {primary['verdict']}  dF_M {_flog(primary.get('dF_meV'), '+.2f')} meV, sigma "
          f"{_flog(primary.get('sigma_meV'), '.2f')} meV; reasons {primary['reasons']}")
    print(f"  free arm: {summary['free_arm']}")
    print(f"  gamma arm: {summary['gamma_arm']}")
    print(f"  v4: {summary['v4']}")
    print(f"  normalisation vs grid: {norm}")
    print(f"wrote {rel(out)}")
    return 0


def smoke_check(args) -> int:
    """Pass/fail of the box smoke run (a tiny recipe in its own --out-dir): the python-sscha
    behaviour this file relies on, checked on the real library before the production arms."""
    base = out_base(args)
    res = []

    def add(name, ok, info=None):
        res.append((name, bool(ok), info))

    docs = [d for d in (load_json(fp) for fp in sorted(base.glob("*.json")))
            if isinstance(d, dict) and d.get("schema") == SCHEMA and d.get("arm_tag")]
    add("at least one smoke arm", bool(docs))
    for d in docs:
        tag, r = d["arm_tag"], d.get("run") or {}
        add(f"{tag}: status ok", r.get("status") == "ok", r.get("error"))
        fe = r.get("free_energy") or {}
        add(f"{tag}: F and its error from Ensemble.get_free_energy (no fallback)",
            fe.get("fe_Ry") is not None and (fe.get("fe_err_Ry") or 0) > 0
            and fe.get("fallback_reason") is None,
            {k: fe.get(k) for k in ("fe_Ry", "fe_err_Ry", "history_minus_final_Ry", "per_cell_atoms",
                                    "ensemble_supercell", "fallback_reason")})
        add(f"{tag}: Gamma-only conversion", _dig(r, "gamma_only", "conversion_ok") is True,
            _dig(r, "gamma_only", "max_abs_freq_diff_thz"))
        add(f"{tag}: symmetries set up", _dig(r, "relax", "n_symmetries") is not None,
            _dig(r, "relax", "n_symmetries"))
        gc = _dig(r, "relax", "gradient_checks") or {}
        add(f"{tag}: real gradient error used", (gc.get("n_calls") or 0) > 0 and gc.get("n_errors") == 0,
            gc)
        add(f"{tag}: Hessian computed", _dig(r, "hessian", "min_nonac_thz") is not None,
            _dig(r, "hessian", "min_nonac_thz"))
        ms = _dig(r, "stopping_criteria", "minim_struct")
        if d.get("arm") == "free":
            mv = _dig(r, "centroid_final", "max_abs_move_from_start_ang")
            add(f"{tag}: minim_struct True and the centroid moved", ms is True and (mv or 0) > 1e-6,
                {"move_ang": mv, "Q_M_final": _dig(r, "centroid_final", "Q_M_ang")})
        else:
            add(f"{tag}: minim_struct False and the centroid held to 1e-8 A",
                ms is False and _dig(r, "fixed_centroid_check", "passed") is True,
                _dig(r, "fixed_centroid_check"))
    x = next((d for d in docs if d["arm_tag"] == arm_tag("m_scan", 0.0)), None)
    grid = load_json(GRID_JSON)
    if x is not None and grid is not None:
        fe = _dig(x, "run", "free_energy", "fe_Ry")
        st = ((grid.get("run") or {}).get("relax") or {}).get("steps") or [{}]
        g = st[-1].get("fe_Ry")
        ok = fe is not None and g is not None and abs(fe / 8.0 - g) < 2e-3
        add("F(Q0=0) is per 40 atoms (F/8 within 2e-3 Ry of the grid's per-cell F)", ok,
            {"fe_over_8": None if fe is None else fe / 8.0, "grid_per_cell": g})
    v = load_json(base / "v4.json")
    if v is not None:
        b = v.get("v4") or {}
        add("v4: subsample saved, reloaded and v2 round trip within 1e-3 THz",
            b.get("status") in ("finished", "timeout") and b.get("v4_false_roundtrip_ok") is True,
            {k: b.get(k) for k in ("status", "v4_false_roundtrip_thz", "v4_false_parent_thz", "error")})
    for name, ok, info in res:
        print(f"{'SMOKE_OK  ' if ok else 'SMOKE_FAIL'} {name}" + ("" if info is None else f"  {info}"))
    bad = [n for n, ok, _ in res if not ok]
    print(f"SMOKE {'PASSED' if not bad else 'FAILED'}: {len(res) - len(bad)}/{len(res)}")
    return 0 if not bad else 1


# ============================================================================== self-test ====

def selftest() -> int:
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    def raises(fn):
        try:
            fn()
        except RuntimeError:
            return True
        return False

    F = lambda fe, err, conv=True, q=None: {"fe_meV": fe, "err_meV": err, "converged": conv,  # noqa: E731
                                            "q0_ang": q}
    g_ok = {"m_band": True, "q0_hessian": True}
    scan_hi = [F(3.0, 2.0, q=0.04), F(10.0, 2.0, q=0.075), F(20.0, 2.0, q=0.10)]
    check("decide: dF = -10, errors 2, 2 -> SUPPORTED",
          decide(F(0, 2), F(-10, 2), [], g_ok)["verdict"] == "SUPPORTED")
    check("decide: dF = +10, scan above F(0) -> NOT_SUPPORTED",
          decide(F(0, 2), F(10, 2), scan_hi, g_ok)["verdict"] == "NOT_SUPPORTED")
    check("decide: dF = +10 but Q0 = 0.04 at -10 -> INCONCLUSIVE",
          decide(F(0, 2), F(10, 2), [F(-10, 2, q=0.04)], g_ok)["verdict"] == "INCONCLUSIVE")
    check("decide: |dF| < 2 sigma -> INCONCLUSIVE",
          decide(F(0, 2), F(-3, 2), [], g_ok)["verdict"] == "INCONCLUSIVE")
    s = math.hypot(2, 2)
    check("decide: dF exactly -2 sigma is not < -2 sigma -> INCONCLUSIVE",
          decide(F(0, 2), F(-2 * s, 2), [], g_ok)["verdict"] == "INCONCLUSIVE")
    check("decide: Q0 = 0.075 not converged -> NOT_EVALUABLE",
          decide(F(0, 2), F(-10, 2, conv=False), [], g_ok)["verdict"] == "NOT_EVALUABLE")
    check("decide: Q0 = 0 missing -> NOT_EVALUABLE",
          decide(None, F(-10, 2), [], g_ok)["verdict"] == "NOT_EVALUABLE")
    check("decide: M-band gate failed -> NOT_EVALUABLE",
          decide(F(0, 2), F(-10, 2), [], {"m_band": False, "q0_hessian": True})["verdict"]
          == "NOT_EVALUABLE")
    check("decide: gate not evaluated (None) -> NOT_EVALUABLE",
          decide(F(0, 2), F(-10, 2), [], {"m_band": True, "q0_hessian": None})["verdict"]
          == "NOT_EVALUABLE")
    d = decide(F(0, 2), F(10, 2), [F(-10, 2, conv=False, q=0.04)], g_ok)
    check("decide: an unconverged scan point is excluded, not counted",
          d["verdict"] == "NOT_SUPPORTED" and d["scan_points_excluded"] == [0.04])
    check("decide: zero error -> NOT_EVALUABLE",
          decide(F(0, 0), F(-10, 0), [], g_ok)["verdict"] == "NOT_EVALUABLE")
    check("decide: the sign of dF follows F(0.075) - F(0)",
          abs(decide(F(5, 1), F(-5, 1), [], g_ok)["dF_meV"] + 10) < 1e-12)

    # mapper: a 2x2x2 perovskite in phonopy's atom-major order and in ASE repeat's cell-major order
    pf = np.array([[0, 0, 0], [.5, .5, .5], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    ps = ["Ba", "Ti", "O", "O", "O"]
    cells = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    am_f = np.array([(pf[a] + c) / 2 for a in range(5) for c in cells])
    am_s = [ps[a] for a in range(5) for _ in cells]
    cm_f = np.array([(pf[a] + c) / 2 for c in cells for a in range(5)])
    cm_s = [ps[a] for _ in cells for a in range(5)]
    perm = map_atoms(am_f, am_s, cm_f, cm_s)
    check("map_atoms: two orderings, src[perm] == dst",
          np.allclose(am_f[perm], cm_f) and [am_s[j] for j in perm] == cm_s
          and perm != list(range(40)))
    check("map_atoms: 5e-5 noise accepted", map_atoms(am_f, am_s, cm_f + 5e-5, cm_s) == perm)
    check("map_atoms: 5e-4 offset refused", raises(lambda: map_atoms(am_f, am_s, cm_f + 5e-4, cm_s)))
    check("map_atoms: wrap-around (1 - 1e-6 == 0) accepted",
          map_atoms(am_f, am_s, (cm_f - 1e-6) % 1.0, cm_s) == perm)
    bad_s = list(cm_s)
    bad_s[0] = "Ti"
    check("map_atoms: species mismatch refused", raises(lambda: map_atoms(am_f, am_s, cm_f, bad_s)))
    dup = cm_f.copy()
    dup[1] = dup[0]
    check("map_atoms: duplicated position refused", raises(lambda: map_atoms(am_f, am_s, dup, cm_s)))
    img = assign_to_prim(cm_f, cm_s, (2, 2, 2), pf, ps)
    check("assign_to_prim: every supercell atom to its primitive atom",
          img == [a for _ in cells for a in range(5)])

    cell = np.diag([8.0, 8.0, 8.0])
    check("min_image: 7.9 A across the boundary is -0.1 A",
          np.allclose(min_image(np.array([7.9, 0.0, -7.95]), cell), [-0.1, 0.0, 0.05]))
    rng = np.random.default_rng(0)
    u = normalize_max(rng.normal(size=(40, 3)))
    r0 = cm_f @ cell
    w = rng.normal(size=(40, 3))
    w -= np.sum(w * u) / np.sum(u * u) * u
    pr = pattern_projection((r0 + 0.07 * u + 1e-3 * w) % 8.0, r0, cell, u)
    check("pattern_projection: Q = 0.07 recovered through wrapping and an orthogonal part",
          abs(pr["Q_ang"] - 0.07) < 1e-12 and abs(pr["residual_norm_ang"] - 1e-3 * np.linalg.norm(w)) < 1e-9)

    # gamma_111_vector on a synthetic triplet: polar x, y, z modes rotated inside the subspace
    m = np.array([137.3, 47.9, 16.0, 16.0, 16.0])
    base = []
    for ax in range(3):
        v = np.zeros((5, 3))
        v[:, ax] = [0.1, 0.8, -0.3, -0.3, -0.4]
        v = (v * np.sqrt(m)[:, None]).ravel()
        base.append(v / np.linalg.norm(v))
    base = np.array(base).T
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    uu, wt, im = gamma_111_vector(base @ q, m, 1)
    check("gamma_111_vector: basis-independent [111] member, Ti positive",
          np.allclose(uu[:, 0], uu[:, 1]) and np.allclose(uu[:, 1], uu[:, 2]) and uu[1, 0] > 0
          and abs(np.abs(uu).max() - 1) < 1e-12 and im < 1e-10 and 0 < wt <= 1)
    print(f"selftest: {'PASSED' if not fails else 'FAILED'} ({len(fails)} failure(s))")
    return 0 if not fails else 1


# =================================================================================== CLI ====

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="R2 multi-mode SSCHA centroid scan (" + PREREG + ")")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--q0", type=float, default=None,
                    help="centroid amplitude along the M pattern, A (m_scan: 0, 0.04, 0.075, 0.1; free: "
                         "0.075; ignored for gamma, whose amplitude comes from the gate)")
    ap.add_argument("--allow-other-q0", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", default=None,
                    help=f"default {rel(OUT_DIR)} ({rel(OUT_DIR / '_dryrun')} with --dry-run)")
    ap.add_argument("--gate-dir", default=None, help="pattern/gate directory (default <out-dir>/gate)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="no sscha/cellconstructor import; SSCHA numbers mocked; relax, phonopy, the "
                         "patterns, the mapper and the MLIP energies are real")
    ap.add_argument("--unit-timeout", type=float, default=DEFAULT_UNIT_TIMEOUT_S,
                    help="wall cap on the relaxation, s (default %(default)g); wrap the process too")
    ap.add_argument("--threads", type=int, default=None,
                    help="BLAS/OpenMP/torch threads of this process; for --arm v4 the child's threads "
                         "(default: the cgroup CPU quota)")
    ap.add_argument("--n-boot", type=int, default=sss.GRID_DEFAULTS["n_boot"])
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--neglect-symmetries", action="store_true",
                    help="fallback if python-sscha cannot set up symmetries on the Gamma-only cell "
                         "(changes the recipe: own --out-dir)")
    ap.add_argument("--v4-n", type=int, default=V4_N, help="subsample size for --arm v4 (default 512)")
    ap.add_argument("--v4-timeout", type=float, default=DEFAULT_V4_TIMEOUT_S)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke-check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    cg = ap.add_argument_group("recipe (the grid's defaults; change only for a smoke run in its own --out-dir)")
    cg.add_argument("--conv-n-configs", type=int, default=sss.CONV_DEFAULTS["n_configs"])
    cg.add_argument("--conv-max-pop", type=int, default=sss.CONV_DEFAULTS["max_pop"])
    cg.add_argument("--conv-n-hessian", type=int, default=sss.CONV_DEFAULTS["n_hessian"])
    cg.add_argument("--conv-steps-per-pop", type=int, default=sss.CONV_DEFAULTS["max_steps_per_pop"])
    cg.add_argument("--conv-seed", type=int, default=sss.CONV_SEED)
    cg.add_argument("--start-b-thz", type=float, default=sss.CONV_DEFAULTS["start_b_thz"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--dry-run-step-sleep", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--dry-run-v4-sleep", type=float, default=3.0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not (args.arm or args.summarize or args.smoke_check or args.selftest):
        ap.error("one of --arm, --summarize, --smoke-check, --selftest")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.summarize:
        return summarize(args)
    if args.smoke_check:
        return smoke_check(args)
    if args.arm == "gate":
        return run_gate(args)
    if args.arm == "v4":
        return run_v4_arm(args)
    return run_arm(args)


if __name__ == "__main__":
    sys.exit(main())
