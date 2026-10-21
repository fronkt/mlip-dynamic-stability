"""Lattice sensitivity of the perovskite soft-mode well (Referee 2, second round).

The referee asks why a sub-percent lattice change moves the soft-mode well depth so much in the
perovskites, and whether each MLIP's volume bias is systematic. Every number here is re-read
from deposited files (each hashed under "inputs"); nothing is measured.

  A. LATTICES. Each MLIP's relaxed conventional cubic lattice a per system, its deviation from
     our PBE (SSSP, Quantum ESPRESSO vc-relax) lattice where one exists, the inter-model spread
     (max-min)/mean, and spread / mean |model - PBE|. Three independent relaxation records are
     cross-checked (dft/geom/checks, the sscha_converged_grid unit JSONs, the dft/geom extxyz
     frames); primitive cells are converted to the conventional cube per prototype
     (perovskite 1:1, bcc x2 volume, fluorite x4 volume). systems.LATTICE_A is carried only as
     "start_value_unsourced": it is the unsourced starting value, not an experiment.

  B. WELL-DEPTH SENSITIVITY. d ln X / d ln a (and per ln V = /3) for the depth, |kappa2|,
     kappa4 and |omega_fit| from two pairs:
       (i)  PBE pair: PBE E(Q) at the PBE lattice vs at MACE-MP-0's lattice (pl_profile.csv).
            CONFOUNDED: the second curve runs along MACE's eigenvector, the first along PBE's.
       (ii) same model: each fine-tuned MACE-MP-0 replicate along its own eigenvector at its own
            and at the base lattice (finetune_mace30/lattice_diag.json); a pair whose deciding
            q-star changes between the lattices is marked not same-mode.
     The reported depth is the sampled-grid minimum (dft_reference._curve_metrics), not the
     fit's minimum; the curves are rebuilt from pw.out / c3a_curves.json and refitted with the
     production fit (finite_t._fit_double_well) to split depth - kappa2^2/(4 kappa4) exactly
     into fit residual, grid placement and sextic term.

  C. FINE-TUNING DRIFT. The 30-epoch fine-tuned lattices against base and PBE, the depth factor
     exp(s d ln a) each measured s predicts for the drift, against the measured wells at the
     base vs own lattice; and whether stress (or any volume variation) was in the training data.

numpy/pandas only (no scipy). Deterministic: no timestamps or git state; the JSON is written LF.
Writes results/revision/r2/lattice_sensitivity.json.

Usage: python scripts/lattice_sensitivity.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from mlip_dynstab.finite_t import _W_TO_OMEGA2, _fit_double_well
from mlip_dynstab.systems import LATTICE_A, load_specs

REPO = Path(__file__).resolve().parents[1]
REV = REPO / "results" / "revision"
OUT = REV / "r2" / "lattice_sensitivity.json"
CONFIG = REPO / "configs" / "curated_systems.yaml"

MODELS = ("mace_mp0", "chgnet", "orb_v2", "sevennet0", "mattersim")
PAIR_MODEL = "mace_mp0"                 # dft_checks.SELECT_MODEL: the MLIP lattice of the PBE pair
FT_SEEDS = ("seed0", "seed1", "seed2")
RY_EV = 13.605691932782346              # Ry of ase.io.espresso (create_units('2006')): parse_pw_out's reader
DLNA_PRACTICAL = 1e-3                   # "+0.1 % in a" as d ln a: pct = 100 (exp(s * 1e-3) - 1)
CUBIC_ANGLE_TOL_DEG = 0.05              # a relaxed cell counts as cubic within these tolerances
CUBIC_LEN_TOL = 1e-3                    # (max-min)/mean of the primitive lengths
# prototype -> (Z = V_conv / V_prim, atoms per primitive cell, a_conv / a_prim, primitive angle)
PROTO = {"cubic-perovskite": (1, 5, 1.0, 90.0),
         "bcc": (2, 1, 2.0 / math.sqrt(3.0), math.degrees(math.acos(-1.0 / 3.0))),
         "fluorite": (4, 3, math.sqrt(2.0), 60.0)}
STAR = {0: "Gamma", 1: "X", 2: "M", 3: "R"}   # simple-cubic stars by the number of 1/2 components


# --------------------------------------------------------------------------- inputs
def _glob_re(pat: str) -> re.Pattern:
    """A .gitattributes pattern as a regex on the repo-relative POSIX path."""
    anchored = "/" in pat.rstrip("/")
    pat, rx, i = pat.lstrip("/"), "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            rx, i = rx + "(?:.*/)?", i + 3
        elif pat.startswith("**", i):
            rx, i = rx + ".*", i + 2
        elif pat[i] == "*":
            rx, i = rx + "[^/]*", i + 1
        elif pat[i] == "?":
            rx, i = rx + "[^/]", i + 1
        else:
            rx, i = rx + re.escape(pat[i]), i + 1
    return re.compile(("" if anchored else "(?:.*/)?") + rx + r"\Z")


class Inputs:
    """Every file the analysis reads, with two hashes:
      sha256_lf      sha256 after CRLF -> LF (verify_claims.py's convention; the same on a Windows
                     core.autocrlf checkout and on Linux);
      git_blob_sha1  the git blob id, sha1("blob <n>\\0" + bytes) of the bytes git stores: CRLF -> LF
                     for text paths, the raw bytes for paths .gitattributes marks -text (there git
                     keeps a csv module's CRLF as committed), last matching rule wins."""

    def __init__(self):
        self.files: dict[str, dict] = {}
        self.rules = []
        ga = REPO / ".gitattributes"
        if ga.exists():
            for ln in ga.read_bytes().decode("utf-8").splitlines():
                f = ln.split()
                if f and not f[0].startswith("#"):
                    raw = True if ("-text" in f[1:] or "binary" in f[1:]) else (
                        False if any(a == "text" or a.startswith("text=") for a in f[1:]) else None)
                    if raw is not None:
                        self.rules.append((_glob_re(f[0]), raw))
            self.raw(ga)

    def _stored_raw(self, rel: str) -> bool:
        raw = False
        for rx, r in self.rules:
            if rx.match(rel):
                raw = r
        return raw

    def raw(self, path: Path) -> bytes:
        orig = Path(path).read_bytes()
        b = orig.replace(b"\r\n", b"\n")
        rel = Path(path).resolve().relative_to(REPO).as_posix()
        stored = orig if self._stored_raw(rel) else b
        self.files[rel] = {"git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(stored) + stored).hexdigest(),
                           "sha256_lf": hashlib.sha256(b).hexdigest(), "n_bytes_lf": len(b)}
        return b

    def text(self, path: Path) -> str:
        return self.raw(path).decode("utf-8", errors="replace")

    def json(self, path: Path):
        return json.loads(self.raw(path).decode("utf-8"))

    def csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.raw(path)))

    def manifest(self) -> dict:
        return {k: self.files[k] for k in sorted(self.files)}


INP = Inputs()


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


# --------------------------------------------------------------------------- geometry
def _cellpar_volume(a, b, c, al, be, ga) -> float:
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    return a * b * c * math.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)


def _cell_to_par(cell) -> list[float]:
    m = np.asarray(cell, float)
    L = np.linalg.norm(m, axis=1)

    def ang(i, j):
        return math.degrees(math.acos(float(m[i] @ m[j] / (L[i] * L[j]))))

    return [float(L[0]), float(L[1]), float(L[2]), ang(1, 2), ang(0, 2), ang(0, 1)]


def _conventional(par, proto: str, vol: float | None = None) -> dict:
    """Conventional cubic a from a primitive cell: volume-equivalent (Z V_prim)^(1/3), which is
    defined for any cell, and length-based mean(|a_i|) * a_conv/a_prim, which equals it only
    when the cell is still the cubic prototype."""
    Z, _, f, ang0 = PROTO[proto]
    L, A = np.array(par[:3], float), np.array(par[3:], float)
    V = _cellpar_volume(*par) if vol is None else float(vol)
    return {"a_conv_A": float((Z * V) ** (1.0 / 3.0)),
            "a_conv_from_lengths_A": float(L.mean() * f),
            "length_anisotropy": float((L.max() - L.min()) / L.mean()),
            "max_angle_dev_deg": float(np.abs(A - ang0).max()),
            "prim_volume_A3": V}


def _is_cubic(rec: dict) -> bool:
    return bool(rec["max_angle_dev_deg"] <= CUBIC_ANGLE_TOL_DEG
                and rec["length_anisotropy"] <= CUBIC_LEN_TOL)


def _star(q) -> str:
    if isinstance(q, str):
        q = [float(x) for x in q.split(",")]
    n = sum(1 for x in q if abs(abs(float(x)) - 0.5) < 1e-6)
    zero = sum(1 for x in q if abs(float(x)) < 1e-6)
    return STAR[n] if n + zero == 3 else "other"


# --------------------------------------------------------------------------- A
def grid_lattices(protos: dict) -> dict:
    """run.relaxed_cellpar of every sscha_converged_grid unit (the relax before SSCHA; present
    even when the SSCHA run later failed), averaged over the unit's temperatures."""
    per: dict = {}
    for p in sorted((REV / "sscha_converged_grid").glob("*.json")):
        d = INP.json(p)
        u, r = d["unit"], d.get("run") or {}
        cp = r.get("relaxed_cellpar")
        if cp is None:
            continue
        rec = _conventional(cp, protos[u["system"]])
        rec.update({"T_K": float(u["T"]), "start": d.get("start"), "status": r.get("status"),
                    "file": _rel(p)})
        per.setdefault((u["system"], u["model"]), []).append(rec)
    out = {}
    for key, runs in sorted(per.items()):
        a = np.array([x["a_conv_A"] for x in runs])
        rec = {"a_conv_A": float(a.mean()),
               "a_conv_runs_max_minus_min_A": float(a.max() - a.min()),
               "a_conv_from_lengths_A": float(np.mean([x["a_conv_from_lengths_A"] for x in runs])),
               "length_anisotropy": float(max(x["length_anisotropy"] for x in runs)),
               "max_angle_dev_deg": float(max(x["max_angle_dev_deg"] for x in runs)),
               "n_runs": len(runs), "T_K": [x["T_K"] for x in runs],
               "run_status": [x["status"] for x in runs]}
        rec["cubic_preserved"] = _is_cubic(rec)
        out[key] = rec
    return out


def checks_lattices(protos: dict) -> dict:
    """dft/geom/checks/<system>_<model>.json relaxed_prim_cell (the source of pl_lattice.csv)."""
    out = {}
    for p in sorted((REV / "dft" / "geom" / "checks").glob("*.json")):
        d = INP.json(p)
        cell = d.get("relaxed_prim_cell")
        if not cell:
            continue
        rec = _conventional(_cell_to_par(cell), protos[d["system"]],
                            vol=abs(float(np.linalg.det(np.asarray(cell, float)))))
        rec["cubic_preserved"] = _is_cubic(rec)
        out[(d["system"], d["model"])] = rec
    return out


_KV = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')


def _xyz_header(line: str) -> dict:
    return {m.group(1): (m.group(2) if m.group(2) is not None else m.group(3))
            for m in _KV.finditer(line)}


def extxyz_lattices(protos: dict) -> tuple[dict, list]:
    """First frame (Q = 0) of every dft/geom/*.extxyz: the supercell volume per primitive cell,
    converted. The lattice is path_model's own (role=ref files carry another model's pattern
    in this model's cell); a path_model != mlip_model header is reported, not used."""
    per, odd = {}, []
    for p in sorted((REV / "dft" / "geom").glob("*.extxyz")):
        lines = INP.text(p).split("\n", 2)
        n, h = int(lines[0].strip()), _xyz_header(lines[1])
        if h.get("path_model") != h.get("mlip_model"):
            odd.append(_rel(p))
            continue
        proto = protos[h["system"]]
        n_prim = PROTO[proto][1]
        if n % n_prim:
            odd.append(_rel(p))
            continue
        L = np.array([float(x) for x in h["Lattice"].split()]).reshape(3, 3)
        v_prim = abs(float(np.linalg.det(L))) * n_prim / n
        a = (PROTO[proto][0] * v_prim) ** (1.0 / 3.0)
        per.setdefault((h["system"], h["path_model"]), []).append(a)
    return ({k: {"a_conv_A": float(np.mean(v)), "max_minus_min_A": float(np.ptp(v)),
                 "n_files": len(v)} for k, v in sorted(per.items())}, odd)


def two_way(dev: pd.DataFrame) -> dict:
    """Sum-of-squares split of model - PBE (%) into system, model and residual effects."""
    D = dev.to_numpy(float)
    g = D.mean()
    se, me = D.mean(1) - g, D.mean(0) - g
    res = D - g - se[:, None] - me[None, :]
    ss_s, ss_m = D.shape[1] * float((se ** 2).sum()), D.shape[0] * float((me ** 2).sum())
    ss_r, ss_t = float((res ** 2).sum()), float(((D - g) ** 2).sum())
    i, j = np.unravel_index(int(np.abs(res).argmax()), res.shape)
    return {"grand_mean_pct": float(g),
            "system_effect_pct": dict(zip(dev.index, map(float, se))),
            "model_effect_pct": dict(zip(dev.columns, map(float, me))),
            "ss_system": ss_s, "ss_model": ss_m, "ss_residual": ss_r, "ss_total": ss_t,
            "share_system": ss_s / ss_t, "share_model": ss_m / ss_t, "share_residual": ss_r / ss_t,
            "largest_residual": {"system": dev.index[i], "model": dev.columns[j],
                                 "residual_pct": float(res[i, j])},
            "rank_by_system_smallest_first": {s: list(dev.loc[s].sort_values().index)
                                              for s in dev.index}}


def lattices(specs: dict, dft_summary: dict) -> dict:
    protos = {s: sp.prototype for s, sp in specs.items()}
    grid, checks = grid_lattices(protos), checks_lattices(protos)
    xyz, xyz_skipped = extxyz_lattices(protos)
    pl = INP.csv(REV / "dft_checks" / "pl_lattice.csv")
    pbe = {r.system: float(r.a_A) for r in pl.itertuples() if r.source.startswith("PBE vc-relax")}
    pl_m = {(r.system, r.source.split()[0]): (float(r.a_A), float(r.minus_pbe_pct))
            for r in pl.itertuples() if r.source.endswith(" relaxed")}
    pl_start = {r.system: float(r.a_A) for r in pl.itertuples() if r.source == "registry start"}

    systems = sorted({s for s, _ in grid} | {s for s, _ in checks})
    per_system, xchk = {}, {"checks_vs_grid": [], "checks_vs_extxyz": [], "pl_lattice_vs_checks": []}
    for s in systems:
        proto = protos[s]
        rows = {}
        for m in MODELS:
            g, c, x = grid.get((s, m)), checks.get((s, m)), xyz.get((s, m))
            if g is None and c is None:
                continue
            prim, src = (c, "dft/geom/checks") if c is not None else (g, "sscha_converged_grid")
            e = {"a_A": prim["a_conv_A"], "source": src,
                 "a_from_lengths_A": prim["a_conv_from_lengths_A"],
                 "length_anisotropy": prim["length_anisotropy"],
                 "max_angle_dev_deg": prim["max_angle_dev_deg"],
                 "cubic_preserved": prim["cubic_preserved"]}
            if g is not None:
                e["grid"] = g
                if c is not None:
                    d = g["a_conv_A"] - c["a_conv_A"]
                    e["grid_minus_checks_A"] = d
                    xchk["checks_vs_grid"].append({"system": s, "model": m, "proto": proto,
                                                   "delta_A": d})
            if x is not None:
                e["extxyz"] = x
                d = x["a_conv_A"] - prim["a_conv_A"]
                e["extxyz_minus_primary_A"] = d
                xchk["checks_vs_extxyz"].append({"system": s, "model": m, "delta_A": d})
            if s in pbe:
                e["minus_pbe_pct"] = 100.0 * (e["a_A"] / pbe[s] - 1.0)
                if (s, m) in pl_m:
                    e["pl_lattice_csv_a_A"], e["pl_lattice_csv_minus_pbe_pct"] = pl_m[(s, m)]
                    xchk["pl_lattice_vs_checks"].append(
                        {"system": s, "model": m, "delta_A": pl_m[(s, m)][0] - e["a_A"],
                         "delta_pct": pl_m[(s, m)][1] - e["minus_pbe_pct"]})
            rows[m] = e
        inc = [m for m, e in rows.items() if e["cubic_preserved"]]
        a = np.array([rows[m]["a_A"] for m in inc])
        ent = {"prototype": proto, "models": rows, "models_in_spread": inc,
               "excluded_not_cubic": {m: {"length_anisotropy": e["length_anisotropy"],
                                          "max_angle_dev_deg": e["max_angle_dev_deg"]}
                                      for m, e in rows.items() if not e["cubic_preserved"]},
               "start_value_unsourced_A": LATTICE_A.get(s)}
        if len(a):
            ent.update({"a_mean_A": float(a.mean()), "a_min_A": float(a.min()),
                        "a_max_A": float(a.max()), "spread_A": float(np.ptp(a)),
                        "spread_pct": float(100.0 * np.ptp(a) / a.mean())})
        if s in pbe:
            dv = np.array([rows[m]["minus_pbe_pct"] for m in inc])
            lat = dft_summary["pbe_lattice"]["systems"][s]["lattice"]
            ent["pbe"] = {"a_A": pbe[s], "volume_A3": lat.get("volume_A3"),
                          "final_scf_pressure_kbar": lat.get("final_scf_pressure_kbar"),
                          "registry_start_in_pl_lattice_A": pl_start.get(s),
                          "method": "QE vc-relax, PBE, SSSP 1.3.0 efficiency (dft_checks.py stage "
                                    "pbe-lattice)"}
            ent.update({"mean_minus_pbe_pct": float(dv.mean()),
                        "mean_abs_minus_pbe_pct": float(np.abs(dv).mean()),
                        "n_models_above_pbe": int((dv > 0).sum()), "n_models": int(len(dv)),
                        "spread_over_mean_abs_dev": float(ent["spread_pct"] / np.abs(dv).mean())})
        per_system[s] = ent

    pbe_sys = sorted(pbe)
    dev = pd.DataFrame({m: [per_system[s]["models"][m]["minus_pbe_pct"] for s in pbe_sys]
                        for m in MODELS}, index=pbe_sys)
    allv = dev.to_numpy().ravel()

    def mx(lst, k):
        return float(max(abs(r[k]) for r in lst)) if lst else None

    xsum = {"checks_vs_grid_max_abs_A": mx(xchk["checks_vs_grid"], "delta_A"),
            "checks_vs_grid_max_abs_A_by_prototype": {
                p: mx([r for r in xchk["checks_vs_grid"] if r["proto"] == p], "delta_A")
                for p in sorted({r["proto"] for r in xchk["checks_vs_grid"]})},
            "checks_vs_grid_systems": sorted({r["system"] for r in xchk["checks_vs_grid"]}),
            "checks_vs_grid_n_pairs": len(xchk["checks_vs_grid"]),
            "pl_lattice_systems_also_in_grid": sorted(set(pbe) & {s for s, _ in grid}),
            "primary_vs_extxyz_max_abs_A": mx(xchk["checks_vs_extxyz"], "delta_A"),
            "primary_vs_extxyz_n_pairs": len(xchk["checks_vs_extxyz"]),
            "extxyz_files_skipped": xyz_skipped,
            "pl_lattice_vs_checks_max_abs_A": mx(xchk["pl_lattice_vs_checks"], "delta_A"),
            "pl_lattice_vs_checks_max_abs_pct": mx(xchk["pl_lattice_vs_checks"], "delta_pct"),
            "pairs": xchk}
    return {"conventions": {
                "a_A": "volume-equivalent conventional a = (Z V_prim)^(1/3); Z = 1 perovskite, "
                       "2 bcc, 4 fluorite; equals the length-based value while the cell stays cubic",
                "primary_source": "dft/geom/checks relaxed_prim_cell where present (the cells the "
                                  "DFT checks and pl_lattice.csv used), else the mean of the "
                                  "sscha_converged_grid run.relaxed_cellpar over temperatures",
                "cubic_preserved": f"max |angle - ideal| <= {CUBIC_ANGLE_TOL_DEG} deg and "
                                   f"(max-min)/mean primitive length <= {CUBIC_LEN_TOL}; models "
                                   "failing it are excluded from the spread",
                "spread_pct": "100 (max - min) / mean over models_in_spread",
                "spread_over_mean_abs_dev": "spread_pct / mean |model - PBE| (%)",
                "start_value_unsourced": "mlip_dynstab/systems.py LATTICE_A: unsourced starting "
                                         "value of the relaxation, NOT an experimental lattice"},
            "per_system": per_system,
            "pbe_systems": pbe_sys,
            "minus_pbe_pct_table": {s: dict(dev.loc[s]) for s in pbe_sys},
            "bias_sign": {"n_positive": int((allv > 0).sum()), "n_total": int(allv.size),
                          "min_pct": float(allv.min()), "max_pct": float(allv.max())},
            "two_way_minus_pbe": two_way(dev),
            "cross_checks": xsum}


# --------------------------------------------------------------------------- B
def _omega_thz(a: float, m_eff: float) -> float:
    """dft_reference._omega_thz: harmonic frequency of V = a Q^2 for mass m_eff, signed."""
    om2 = 2.0 * a * _W_TO_OMEGA2 / m_eff
    return float(np.sign(om2) * np.sqrt(abs(om2)) / (2 * np.pi) / 1e12)


def _m_eff_implied(a: float, omega_thz: float) -> float:
    return float(2.0 * abs(a) * _W_TO_OMEGA2 / (2 * np.pi * omega_thz * 1e12) ** 2)


def _poly_min(a: float, b: float, c: float):
    """Exact minimum of the fit form V(Q) = aQ^2 + bQ^4 + cQ^6 over Q >= 0 (x = Q^2).
    Returns (depth_meV, Q*_A), or (None, None) if V is unbounded below."""
    if c < 0 or (c == 0 and b < 0) or (c == 0 and b == 0 and a < 0):
        return None, None
    xs = [0.0]
    coef = [3 * c, 2 * b, a] if c > 0 else ([2 * b, a] if b != 0 else [])
    for r in (np.roots(coef) if coef else []):
        if abs(r.imag) < 1e-12 and r.real > 0:
            xs.append(float(r.real))
    V = [a * x + b * x * x + c * x ** 3 for x in xs]
    i = int(np.argmin(V))
    return -min(V[i], 0.0) * 1000.0, math.sqrt(xs[i])


def _pw_energy_eV(txt: str):
    ok = ("JOB DONE" in txt and "convergence has been achieved" in txt
          and "convergence NOT achieved" not in txt)
    m = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", txt, re.M)
    return (float(m[-1]) * RY_EV) if (ok and m) else None


def pbe_lattice_curve(system: str, tag: str):
    """PBE E(Q) at the PBE lattice, rebuilt as dft_checks._curve does: frame energies from the
    pw.out '!' line, E(Q) - E(0) scaled by the reduced cell's image count."""
    meta_p = REV / "dft_checks" / "pbe_lattice" / system / f"{tag}_meta.json"
    if not meta_p.exists():
        return None, None
    meta = INP.json(meta_p)
    rows = []
    for job in meta["jobs"]:
        jd = REV / "dft_checks" / "qe" / job
        mem = INP.json(jd / "job.json")["members"][0]
        e = _pw_energy_eV(INP.text(jd / "pw.out"))
        if e is None:
            return None, meta
        rows.append((int(mem["frame"]), float(mem["Q"]), e, int(mem["n_images"])))
    rows.sort()
    if rows[0][0] != 0:
        return None, meta
    e0 = rows[0][2]
    return (np.array([r[1] for r in rows]), np.array([(r[2] - e0) * r[3] for r in rows])), meta


def fit_check(Q, dE, row: dict) -> dict:
    """Refit with the production fit; split depth - kappa2^2/(4 kappa4) into its parts."""
    a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
    out = {"kappa2_eV_A2": a, "kappa4_eV_A4": b, "kappa6_eV_A6": c,
           "depth_grid_meV": float(row["depth_meV"]), "Q_grid_min_A": float(row["Q_min_A"]),
           "min_at_edge": bool(row["min_at_edge"]), "omega_fit_thz": float(row["omega_harm_thz"])}
    dq = a * a / (4 * b) * 1000.0 if (a < 0 < b) else None
    dx, qx = _poly_min(a, b, c)
    out.update({"depth_quartic_kappa2sq_over_4kappa4_meV": dq,
                "Q_star_quartic_A": math.sqrt(-a / (2 * b)) if dq is not None else None,
                "depth_fit_exact_meV": dx, "Q_star_fit_exact_A": qx,
                "ratio_depth_grid_over_quartic": out["depth_grid_meV"] / dq if dq else None,
                "ratio_depth_grid_over_fit_exact": out["depth_grid_meV"] / dx if dx else None})
    qg = out["Q_grid_min_A"]
    e_fit_qg = -(a * qg ** 2 + b * qg ** 4 + c * qg ** 6) * 1000.0   # as a depth (meV)
    if Q is not None:
        ra, rb, rc = _fit_double_well(Q, dE)
        win = dE <= dE.min() + max(0.06, 5.0 * abs(min(dE.min(), 0.0)))
        out["curve"] = {"Q_A": [float(x) for x in Q], "dE_meV": [float(x) * 1000 for x in dE],
                        "fit_window_n_points": int(win.sum())}
        out["refit"] = {"a": ra, "b": rb, "c": rc,
                        "max_rel_diff_vs_csv": float(max(abs(ra - a) / abs(a), abs(rb - b) / abs(b),
                                                         abs(rc - c) / abs(c) if c else abs(rc))),
                        "depth_grid_recomputed_meV": float(-min(dE.min(), 0.0) * 1000.0)}
        out["refit"]["depth_grid_recomputed_minus_csv_meV"] = (
            out["refit"]["depth_grid_recomputed_meV"] - out["depth_grid_meV"])
        if qx is not None and Q is not None:
            lo = [float(x) for x in Q if x <= qx]
            hi = [float(x) for x in Q if x > qx]
            out["grid_points_bracketing_Q_star_A"] = [max(lo) if lo else None, min(hi) if hi else None]
    if dq is not None and dx is not None:
        # depth_grid - D_quartic = fit residual + grid placement + sextic term (exact identity)
        out["mismatch_split_meV"] = {
            "total_depth_grid_minus_quartic": out["depth_grid_meV"] - dq,
            "fit_residual_at_grid_min": out["depth_grid_meV"] - e_fit_qg,
            "grid_placement": e_fit_qg - dx,
            "sextic_term": dx - dq}
    out["fit_depth_at_grid_min_meV"] = e_fit_qg
    return out


def _sens(x1, x0, dlna):
    return math.log(x1 / x0) / dlna


def _pct(s):
    return 100.0 * (math.exp(s * DLNA_PRACTICAL) - 1.0)


def pbe_pair(dft_summary: dict, lat: dict) -> dict:
    prof = INP.csv(REV / "dft_checks" / "pl_profile.csv")
    rows = {(r["system"], r["curve"]): r for r in prof.to_dict("records")}
    c3a = INP.json(REV / "dft" / "c3a_curves.json")
    c3a_pbe = {c["stem"]: c for c in c3a["curves"] if c["curve"] == "pbe"}
    fits, pairs = {}, {}
    for (s, tag), row in sorted(rows.items()):
        curve, meta = None, None
        if tag.startswith("pbe_lattice"):
            curve, meta = pbe_lattice_curve(s, "prof" if tag == "pbe_lattice_deciding" else "profsoft")
        else:
            stem = dft_summary["pbe_lattice"]["systems"][s]["mlip_path_stem"]
            c = c3a_pbe[stem]
            curve = (np.array(c["Q_A"], float), np.array(c["dE_meV"], float) / 1000.0)
        fits[f"{s}|{tag}"] = fit_check(*(curve if curve else (None, None)), row)
    for s in sorted({k[0] for k in rows}):
        if (s, "pbe_lattice_deciding") not in rows or (s, "pbe_mlip_lattice") not in rows:
            continue
        ent = dft_summary["pbe_lattice"]["systems"][s]
        stem = ent["mlip_path_stem"]
        pat = INP.json(REV / "dft" / "geom" / "patterns" / f"{stem}.json")
        meta = INP.json(REV / "dft_checks" / "pbe_lattice" / s / "prof_meta.json")
        a0 = lat["per_system"][s]["pbe"]["a_A"]
        a1 = lat["per_system"][s]["models"][PAIR_MODEL]["a_A"]
        a1_pat = float(np.cbrt(abs(np.linalg.det(np.asarray(pat["prim_cell"], float)))))
        f0, f1 = fits[f"{s}|pbe_lattice_deciding"], fits[f"{s}|pbe_mlip_lattice"]
        m0, m1 = float(meta["m_eff"]), float(pat["m_eff"])
        dlna = math.log(a1 / a0)
        q0, q1 = ent["profiles"]["deciding"]["q"], ent["mlip_path_q"]
        X = {"depth_grid": (f0["depth_grid_meV"], f1["depth_grid_meV"]),
             "depth_fit_exact": (f0["depth_fit_exact_meV"], f1["depth_fit_exact_meV"]),
             "depth_quartic": (f0["depth_quartic_kappa2sq_over_4kappa4_meV"],
                               f1["depth_quartic_kappa2sq_over_4kappa4_meV"]),
             "abs_kappa2": (abs(f0["kappa2_eV_A2"]), abs(f1["kappa2_eV_A2"])),
             "kappa4": (f0["kappa4_eV_A4"], f1["kappa4_eV_A4"]),
             "abs_omega_fit": (abs(f0["omega_fit_thz"]), abs(f1["omega_fit_thz"])),
             "m_eff": (m0, m1)}
        sens = {k: {"at_pbe_lattice": v[0], "at_mlip_lattice": v[1], "ratio": v[1] / v[0],
                    "dln_dlna": _sens(v[1], v[0], dlna), "dln_dlnV": _sens(v[1], v[0], dlna) / 3.0}
                for k, v in X.items() if v[0] and v[1]}
        for k in ("depth_grid", "depth_fit_exact"):
            if k in sens:
                sens[k]["pct_per_plus_0p1pct_a"] = _pct(sens[k]["dln_dlna"])
        l2 = math.log(X["abs_kappa2"][1] / X["abs_kappa2"][0])
        l4 = math.log(X["kappa4"][1] / X["kappa4"][0])
        lg = math.log(X["depth_grid"][1] / X["depth_grid"][0])
        rest = lg - (2 * l2 - l4)
        dec = {"ln_depth_grid_ratio": lg, "two_ln_abs_kappa2_ratio": 2 * l2,
               "minus_ln_kappa4_ratio": -l4, "grid_and_sextic_remainder": rest,
               "share_two_ln_kappa2": 2 * l2 / lg, "share_minus_ln_kappa4": -l4 / lg,
               "share_remainder": rest / lg,
               "remainder_is": "ln[(D_grid/D_quartic) at MLIP lattice / (D_grid/D_quartic) at PBE "
                               "lattice]: grid placement + fit residual + sextic term"}
        if X["depth_fit_exact"][0] and X["depth_fit_exact"][1]:
            lx = math.log(X["depth_fit_exact"][1] / X["depth_fit_exact"][0])
            dec.update({"ln_depth_fit_exact_ratio": lx,
                        "fit_exact_share_two_ln_kappa2": 2 * l2 / lx,
                        "fit_exact_share_minus_ln_kappa4": -l4 / lx,
                        "fit_exact_sextic_remainder": lx - (2 * l2 - l4)})
        lm = math.log(m1 / m0)
        om = {"dln_abs_omega": math.log(X["abs_omega_fit"][1] / X["abs_omega_fit"][0]),
              "half_dln_abs_kappa2": 0.5 * l2, "minus_half_dln_m_eff": -0.5 * lm,
              "m_eff_implied_from_fit": [_m_eff_implied(f0["kappa2_eV_A2"], f0["omega_fit_thz"]),
                                         _m_eff_implied(f1["kappa2_eV_A2"], f1["omega_fit_thz"])]}
        edge = f0["min_at_edge"] or f1["min_at_edge"]
        sextic = bool(f0["kappa6_eV_A6"] or f1["kappa6_eV_A6"])
        pairs[s] = {
            "a_pbe_A": a0, "a_mlip_A": a1, "mlip_model": PAIR_MODEL,
            "a_mlip_from_pattern_prim_cell_A": a1_pat, "a_pbe_from_prof_meta_A": float(meta["a_pbe_A"]),
            "dln_a": dlna, "dln_V": 3 * dlna, "pct_a": 100 * (a1 / a0 - 1),
            "q_pbe_curve": q0, "q_mlip_curve": q1, "star_pbe_curve": _star(q0),
            "star_mlip_curve": _star(q1), "mlip_path_stem": stem,
            "m_eff_pbe_eigenvector_amu": m0, "m_eff_mlip_eigenvector_amu": m1,
            "dln_m_eff": lm,
            "sensitivity": sens, "decomposition": dec, "omega_decomposition": om,
            "min_at_edge_any": bool(edge), "sextic_in_fit": sextic,
            "interpretable": bool(not edge and _star(q0) == _star(q1)),
            "confound": "lattice AND eigenvector change together: the PBE-lattice curve runs along "
                        "PBE's own eigenvector (prof_meta m_eff), the MLIP-lattice curve along "
                        f"{PAIR_MODEL}'s eigenvector (pattern m_eff); same q and band, different "
                        "force constants. s is therefore an upper-level total, not a pure "
                        "lattice derivative"}
        if edge:
            pairs[s]["note"] = ("minimum at the Q-grid edge: depth_grid is a lower bound, so the "
                                "sensitivity is not interpretable")
    return {"fits": fits, "pairs": pairs}


def same_model(ld: dict) -> dict:
    out = {}
    for s, recs in ld["systems"].items():
        a_b = recs["base"]["a_relaxed_A"]
        rows = {}
        for t in FT_SEEDS:
            r = recs[t]
            a_o = r["a_relaxed_A"]
            d_o, d_b = r["own_lattice"]["well_depth_meV"], r["base_lattice"]["well_depth_meV"]
            q_o, q_b = r["own_lattice"]["decide_q"], r["base_lattice"]["decide_q"]
            dlna = math.log(a_b / a_o)
            sv = _sens(d_b, d_o, dlna)
            same = _star(q_o) == _star(q_b)
            rows[t] = {"a_own_A": a_o, "a_base_A": a_b, "dln_a_base_minus_own": dlna,
                       "depth_own_meV": d_o, "depth_base_meV": d_b, "ratio_base_over_own": d_b / d_o,
                       "q_own": q_o, "q_base": q_b, "star_own": _star(q_o), "star_base": _star(q_b),
                       "same_q_vector": q_o == q_b, "same_mode_star": same,
                       "dln_depth_dlna": sv, "dln_depth_dlnV": sv / 3.0,
                       "pct_per_plus_0p1pct_a": _pct(sv)}
        same_s = [v["dln_depth_dlna"] for v in rows.values() if v["same_mode_star"]]
        alls = [v["dln_depth_dlna"] for v in rows.values()]
        out[s] = {"replicates": rows,
                  "n_same_mode": len(same_s),
                  "same_mode_dln_depth_dlna": {"min": min(same_s), "max": max(same_s),
                                               "mean": float(np.mean(same_s))} if same_s else None,
                  "all_dln_depth_dlna_incl_mode_switch": {"min": min(alls), "max": max(alls)},
                  "base_model_own_lattice_depth_meV": recs["base"]["own_lattice"]["well_depth_meV"],
                  "base_model_decide_q": recs["base"]["own_lattice"]["decide_q"]}
        if same_s:
            out[s]["same_mode_pct_per_plus_0p1pct_a_mean_s"] = _pct(float(np.mean(same_s)))
        if len(same_s) < len(rows):
            out[s]["note"] = ("the deciding q-star differs between the own and base lattice in "
                              f"{len(rows) - len(same_s)} of {len(rows)} replicates: those ratios "
                              "compare different modes and are not a same-mode lattice derivative")
    return out


# --------------------------------------------------------------------------- C
def training_record() -> dict:
    """Was stress (or any volume variation) a fine-tuning target?"""
    qe = INP.text(REPO / "scripts" / "finetune" / "qe.py").split("\n")
    ts = [(i + 1, ln.strip()) for i, ln in enumerate(qe) if "tstress" in ln]
    data = INP.text(REPO / "scripts" / "finetune" / "data.py").split("\n")
    dl = [(i + 1, ln.strip()) for i, ln in enumerate(data) if "stress" in ln]
    cfg = INP.json(REPO / "scripts" / "finetune" / "mace_medium_finetune.json")
    seq = INP.text(REV / "finetune_mace30" / "train" / "box_sequence.sh").split("\n")
    runs = [ln for ln in seq if "mace.cli.run_train" in ln and not ln.lstrip().startswith("#")]

    def flag(ln, k):
        m = re.search(rf"--{k}=(\S+)", ln)
        return m.group(1) if m else None

    ds = REV / "finetune_mace30" / "dataset" / "mace_mp0"
    files = {}
    cells: dict = {}
    for name in ("train.extxyz", "valid.extxyz", "labelled_all.extxyz"):
        txt = INP.text(ds / name)
        props, keys = set(), set()
        lines = txt.split("\n")
        i = 0
        while i < len(lines) and lines[i].strip():
            n = int(lines[i].strip())
            h = _xyz_header(lines[i + 1])
            props.add(h.get("Properties"))
            keys |= set(h)
            if name == "labelled_all.extxyz":
                L = np.array([float(x) for x in h["Lattice"].split()]).reshape(3, 3)
                a = (abs(float(np.linalg.det(L))) * 5 / n) ** (1.0 / 3.0)   # perovskite prim
                cells.setdefault(h["system"], []).append(a)
            i += n + 2
        files[name] = {"properties": sorted(props), "header_keys": sorted(keys),
                       "n_case_insensitive_stress": txt.lower().count("stress"),
                       "n_case_insensitive_virial": txt.lower().count("virial")}
    tc = {s: {"n_configs": len(v), "a_conv_A_min": float(min(v)), "a_conv_A_max": float(max(v)),
              "n_distinct_cells_1e-8": len({round(x, 8) for x in v})} for s, v in sorted(cells.items())}
    stress_args = [k for k in cfg["args"] if "stress" in k.lower() or "virial" in k.lower()]
    return {"qe_tstress_lines": [{"file": "scripts/finetune/qe.py", "line": n, "text": t}
                                 for n, t in ts],
            "data_py_stress_lines": [{"file": "scripts/finetune/data.py", "line": n, "text": t}
                                     for n, t in dl],
            "config_args_with_stress_or_virial": stress_args,
            "config_energy_weight": cfg["args"].get("energy_weight"),
            "config_forces_weight": cfg["args"].get("forces_weight"),
            "run_train_lines": len(runs),
            "run_train_lines_mentioning_stress_or_loss": sum(
                1 for ln in runs if "stress" in ln.lower() or "--loss" in ln),
            "run_train_energy_weights": sorted({flag(ln, "energy_weight") for ln in runs}),
            "run_train_forces_weights": sorted({flag(ln, "forces_weight") for ln in runs}),
            "dataset_files": files,
            "training_cells": tc,
            "stress_in_loss": bool(stress_args or any("stress" in ln.lower() for ln in runs)
                                   or any(f["n_case_insensitive_stress"] for f in files.values())),
            "verdict": "energy and forces only: QE ran with tstress=.false., the extxyz files carry "
                       "no stress/virial property, and no run_train line sets a stress weight, "
                       "stress key or loss; every configuration of a system sits in one cell"}


def drift(ld: dict, lat: dict, pair: dict, same: dict, train: dict) -> dict:
    out = {}
    for s, recs in ld["systems"].items():
        a_b = recs["base"]["a_relaxed_A"]
        a_p = lat["per_system"][s]["pbe"]["a_A"]
        a_tr = train["training_cells"].get(s, {})
        S = {}
        pp = pair["pairs"].get(s)
        if pp:
            S["pbe_pair_depth_grid"] = pp["sensitivity"]["depth_grid"]["dln_dlna"]
            if "depth_fit_exact" in pp["sensitivity"]:
                S["pbe_pair_depth_fit_exact"] = pp["sensitivity"]["depth_fit_exact"]["dln_dlna"]
        bt = same.get("batio3_cubic", {}).get("same_mode_dln_depth_dlna")
        if bt:
            S["same_model_batio3_mean"] = bt["mean"]
        rows = {}
        for t in FT_SEEDS:
            r = recs[t]
            a_o = r["a_relaxed_A"]
            dlna = math.log(a_b / a_o)
            meas = r["base_lattice"]["well_depth_meV"] / r["own_lattice"]["well_depth_meV"]
            p_gpa = r["pressure_at_base_lattice_GPa"]
            pred = {}
            for k, sv in S.items():
                f = math.exp(sv * dlna)
                pred[k] = {"s": sv, "predicted_ratio": f, "predicted_over_measured": f / meas,
                           "fraction_of_ln_ratio_explained": sv * dlna / math.log(meas)}
            if s == "batio3_cubic" and "same_model_batio3_mean" in pred:
                pred["same_model_batio3_mean"]["in_sample"] = True
            rows[t] = {"a_ft_A": a_o, "drift_vs_base_pct": 100 * (a_o / a_b - 1),
                       "ft_minus_pbe_pct": 100 * (a_o / a_p - 1),
                       "fraction_of_base_to_pbe_gap_moved": (a_o - a_b) / (a_p - a_b),
                       "pressure_at_base_lattice_GPa": p_gpa,
                       "implied_bulk_modulus_GPa": -p_gpa / (3 * dlna),
                       "dln_a_base_over_ft": dlna,
                       "depth_own_lattice_meV": r["own_lattice"]["well_depth_meV"],
                       "depth_base_lattice_meV": r["base_lattice"]["well_depth_meV"],
                       "measured_ratio_base_over_own": meas,
                       "call_own_lattice_stable": r["own_lattice"]["pred_stable"],
                       "call_base_lattice_stable": r["base_lattice"]["pred_stable"],
                       "same_mode_star": _star(r["own_lattice"]["decide_q"]) == _star(
                           r["base_lattice"]["decide_q"]),
                       "predicted": pred}
        po = {k: [v["predicted"][k]["predicted_over_measured"] for v in rows.values()] for k in S}
        out[s] = {"a_base_A": a_b, "a_pbe_A": a_p, "base_minus_pbe_pct": 100 * (a_b / a_p - 1),
                  "base_pressure_at_own_lattice_GPa": recs["base"]["pressure_at_base_lattice_GPa"],
                  "training_cell_a_A": a_tr.get("a_conv_A_min"),
                  "training_cell_minus_base_A": (a_tr["a_conv_A_min"] - a_b) if a_tr else None,
                  "training_cells_distinct": a_tr.get("n_distinct_cells_1e-8"),
                  "replicates": rows,
                  "ranges": {
                      "a_ft_A": [min(v["a_ft_A"] for v in rows.values()),
                                 max(v["a_ft_A"] for v in rows.values())],
                      "drift_vs_base_pct": [min(v["drift_vs_base_pct"] for v in rows.values()),
                                            max(v["drift_vs_base_pct"] for v in rows.values())],
                      "ft_minus_pbe_pct": [min(v["ft_minus_pbe_pct"] for v in rows.values()),
                                           max(v["ft_minus_pbe_pct"] for v in rows.values())],
                      "pressure_at_base_lattice_GPa": [
                          min(v["pressure_at_base_lattice_GPa"] for v in rows.values()),
                          max(v["pressure_at_base_lattice_GPa"] for v in rows.values())],
                      "depth_own_lattice_meV": [min(v["depth_own_lattice_meV"] for v in rows.values()),
                                                max(v["depth_own_lattice_meV"] for v in rows.values())],
                      "depth_base_lattice_meV": [min(v["depth_base_lattice_meV"] for v in rows.values()),
                                                 max(v["depth_base_lattice_meV"] for v in rows.values())],
                      "measured_ratio": [min(v["measured_ratio_base_over_own"] for v in rows.values()),
                                         max(v["measured_ratio_base_over_own"] for v in rows.values())],
                      "predicted_over_measured": {k: [min(v), max(v)] for k, v in po.items()}}}
    return out


# --------------------------------------------------------------------------- recon
def recon_comparison(res: dict) -> list:
    """The values the reconnaissance read, against what this script computes."""
    A, B, C = res["A_lattices"], res["B_well_depth_sensitivity"], res["C_finetune_drift"]
    rows = []

    def add(item, recon, computed, agrees, note=None):
        rows.append({"item": item, "recon": recon, "computed": computed, "agrees": bool(agrees),
                     **({"note": note} if note else {})})

    for s, v in (("batio3_cubic", 4.0237), ("knbo3_cubic", 4.0269), ("cssnbr3_cubic", 5.8871)):
        x = A["per_system"][s]["pbe"]["a_A"]
        add(f"PBE a {s}", v, x, round(x, 4) == v)
    sysb = A["cross_checks"]["pl_lattice_systems_also_in_grid"]
    add("systems in both pl_lattice.csv and the sscha grid", 3, len(sysb), len(sysb) == 3,
        f"{sysb}: CsSnBr3 has no grid unit; checks-vs-grid overlap is "
        f"{A['cross_checks']['checks_vs_grid_systems']} instead")
    for s, d0, d1, k20, k21, k40, k41 in (("batio3_cubic", 48.74, 73.50, -17.04, -19.19, 1370, 1246),
                                          ("knbo3_cubic", 29.97, 87.80, -13.95, -21.50, 1286, 1305)):
        f0, f1 = B["pbe_pair"]["fits"][f"{s}|pbe_lattice_deciding"], B["pbe_pair"]["fits"][f"{s}|pbe_mlip_lattice"]
        got = [f0["depth_grid_meV"], f1["depth_grid_meV"], f0["kappa2_eV_A2"], f1["kappa2_eV_A2"],
               f0["kappa4_eV_A4"], f1["kappa4_eV_A4"]]
        ok = ([round(x, 2) for x in got[:4]] == [d0, d1, k20, k21]
              and [round(x) for x in got[4:]] == [k40, k41])
        add(f"PBE pair {s} depth, kappa2, kappa4", [d0, d1, k20, k21, k40, k41], got, ok)
    sm = B["same_model"]
    bt = [sm["batio3_cubic"]["replicates"][t]["dln_depth_dlna"] for t in FT_SEEDS]
    add("same-model BaTiO3 s (M point)", [135, 148, 136], bt, [round(x) for x in bt] == [135, 148, 136])
    kn = sm["knbo3_cubic"]["all_dln_depth_dlna_incl_mode_switch"]
    add("same-model KNbO3 s range (mode switches)", [73, 113], [kn["min"], kn["max"]],
        [round(kn["min"]), round(kn["max"])] == [73, 113] and sm["knbo3_cubic"]["n_same_mode"] == 0,
        "all 3 KNbO3 replicates switch q-star (M at own lattice; X or Gamma at base)")
    for s, lo, hi in (("batio3_cubic", 4.0129, 4.0152), ("knbo3_cubic", 4.0359, 4.0366)):
        r = C[s]["ranges"]["a_ft_A"]
        add(f"fine-tuned a {s}", [lo, hi], r, [round(r[0], 4), round(r[1], 4)] == [lo, hi])
    pr = [min(C[s]["ranges"]["pressure_at_base_lattice_GPa"][0] for s in C),
          max(C[s]["ranges"]["pressure_at_base_lattice_GPa"][1] for s in C)]
    add("pressure at base lattice (GPa)", [-2.79, -2.56], pr, [round(x, 2) for x in pr] == [-2.79, -2.56])
    wb = [min(C[s]["ranges"]["depth_base_lattice_meV"][0] for s in C),
          max(C[s]["ranges"]["depth_base_lattice_meV"][1] for s in C)]
    wo = [min(C[s]["ranges"]["depth_own_lattice_meV"][0] for s in C),
          max(C[s]["ranges"]["depth_own_lattice_meV"][1] for s in C)]
    add("wells at base lattice (meV)", [49, 64], wb, [round(x) for x in wb] == [49, 64])
    add("wells at own lattice (meV)", [28, 34], wo, [round(x) for x in wo] == [28, 34])
    ts = res["C_training"]["qe_tstress_lines"]
    add("tstress=.false. at qe.py:108", 108, [x["line"] for x in ts],
        any(x["line"] == 108 and "tstress = .false." in x["text"] for x in ts))
    return rows


# --------------------------------------------------------------------------- main
def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    INP.raw(CONFIG)                                         # hashed; load_specs re-reads it
    specs = {sp.id: sp for sp in load_specs(CONFIG)}
    for code in ("mlip_dynstab/systems.py", "mlip_dynstab/finite_t.py"):
        INP.raw(REPO / code)                                # LATTICE_A and the fit form
    dft_summary = INP.json(REV / "dft_checks" / "summary.json")
    ld = INP.json(REV / "finetune_mace30" / "lattice_diag.json")

    lat = lattices(specs, dft_summary)
    pair = pbe_pair(dft_summary, lat)
    same = same_model(ld)
    train = training_record()
    res = {
        "schema": "lattice_sensitivity/v1",
        "script": "scripts/lattice_sensitivity.py",
        "note": "zero-compute: re-reads deposited files only; every input is hashed under inputs",
        "A_lattices": lat,
        "B_well_depth_sensitivity": {
            "conventions": {
                "s": "d ln X / d ln a from a two-point pair: ln(X1/X0) / ln(a1/a0); per ln V = s/3",
                "pct_per_plus_0p1pct_a": "100 (exp(s * 0.001) - 1)",
                "depth_grid": "the reported depth: -min over the sampled Q grid "
                              "(dft_reference._curve_metrics, finite_t well_depth_meV)",
                "kappa2, kappa4, kappa6": "a, b, c of finite_t._fit_double_well: "
                                          "V(Q) = a Q^2 + b Q^4 + c Q^6 (eV, Q in Ang), least "
                                          "squares over the window dE <= min + max(0.06 eV, 5 depth); "
                                          "quartic refit if the sextic is not bounded",
                "depth_quartic": "kappa2^2 / (4 kappa4), ignores kappa6",
                "depth_fit_exact": "exact minimum of the fitted polynomial over Q >= 0",
                "omega_fit": "dft_reference._omega_thz(kappa2, m_eff); |omega| ~ sqrt(|kappa2| / m_eff)"},
            "pbe_pair": pair,
            "same_model": same},
        "C_training": train,
        "C_finetune_drift": drift(ld, lat, pair, same, train),
    }
    res["recon_comparison"] = recon_comparison(res)
    res["inputs"] = INP.manifest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")

    print("\n=== A. MLIP-relaxed conventional a (A) and model - PBE (%) ===\n")
    tab = []
    for s, e in lat["per_system"].items():
        cells = []
        for m in MODELS:
            v = e["models"].get(m)
            if v is None:
                cells.append("-")
                continue
            t = f"{v['a_A']:.4f}" + (f" ({v['minus_pbe_pct']:+.3f})" if "minus_pbe_pct" in v else "")
            cells.append(t + ("" if v["cubic_preserved"] else " NC"))
        sp = f"{e['spread_pct']:.3f}" if "spread_pct" in e else "-"
        rt = f"{e['spread_over_mean_abs_dev']:.3f}" if "spread_over_mean_abs_dev" in e else "-"
        pb = f"{e['pbe']['a_A']:.4f}" if "pbe" in e else "-"
        tab.append([s, pb] + cells + [sp, rt])
    print(_md_table(tab, ["system", "PBE"] + list(MODELS) + ["spread %", "spread/|dev|"]))
    tw = lat["two_way_minus_pbe"]
    print(f"\nall model - PBE positive: {lat['bias_sign']['n_positive']}/{lat['bias_sign']['n_total']}; "
          f"SS shares system {tw['share_system']:.3f}, model {tw['share_model']:.3f}, "
          f"residual {tw['share_residual']:.3f}")
    xc = lat["cross_checks"]
    print(f"cross-checks: checks vs grid max |da| = {xc['checks_vs_grid_max_abs_A']:.2e} A over "
          f"{xc['checks_vs_grid_n_pairs']} pairs; vs extxyz {xc['primary_vs_extxyz_max_abs_A']:.2e} A; "
          f"pl_lattice vs checks {xc['pl_lattice_vs_checks_max_abs_A']:.2e} A")

    print("\n=== B. PBE pair (PBE lattice -> MACE lattice; lattice + eigenvector confounded) ===\n")
    tab = []
    for s, p in pair["pairs"].items():
        sv = p["sensitivity"]
        tab.append([s, f"{p['pct_a']:.3f}",
                    f"{sv['depth_grid']['dln_dlna']:.1f}",
                    f"{sv['depth_fit_exact']['dln_dlna']:.1f}" if "depth_fit_exact" in sv else "-",
                    f"{sv['abs_kappa2']['dln_dlna']:.1f}", f"{sv['kappa4']['dln_dlna']:.1f}",
                    f"{sv['abs_omega_fit']['dln_dlna']:.1f}",
                    f"{sv['depth_grid']['pct_per_plus_0p1pct_a']:.1f}",
                    "yes" if p["interpretable"] else "NO"])
    print(_md_table(tab, ["system", "da %", "s depth(grid)", "s depth(fit)", "s |k2|", "s k4",
                          "s |w|", "% per +0.1% a", "interpretable"]))
    print("\ndepth_grid / (k2^2/4k4) per fit:")
    for k, f in pair["fits"].items():
        r = f["ratio_depth_grid_over_quartic"]
        print(f"  {k:40s} {r:.4f}" if r else f"  {k:40s} -",
              f"(refit max rel diff {f['refit']['max_rel_diff_vs_csv']:.1e})" if "refit" in f else "")
    print("\n=== B. same model, own eigenvector (fine-tuned MACE, base vs own lattice) ===\n")
    tab = []
    for s, e in same.items():
        for t, r in e["replicates"].items():
            tab.append([s, t, f"{r['depth_own_meV']:.2f}", f"{r['depth_base_meV']:.2f}",
                        f"{r['star_own']}->{r['star_base']}", f"{r['dln_depth_dlna']:.1f}",
                        f"{r['pct_per_plus_0p1pct_a']:.1f}"])
    print(_md_table(tab, ["system", "rep", "D own", "D base", "star", "s", "% per +0.1% a"]))

    print("\n=== C. fine-tuning drift ===\n")
    dr = res["C_finetune_drift"]
    tab = []
    for s, e in dr.items():
        for t, r in e["replicates"].items():
            pr = r["predicted"]
            tab.append([s, t, f"{r['a_ft_A']:.4f}", f"{r['drift_vs_base_pct']:+.3f}",
                        f"{r['ft_minus_pbe_pct']:+.3f}", f"{r['pressure_at_base_lattice_GPa']:.2f}",
                        f"{r['measured_ratio_base_over_own']:.3f}",
                        f"{pr['pbe_pair_depth_grid']['predicted_ratio']:.3f}",
                        f"{pr['pbe_pair_depth_fit_exact']['predicted_ratio']:.3f}"])
    print(_md_table(tab, ["system", "rep", "a_ft", "vs base %", "vs PBE %", "P GPa", "D ratio meas",
                          "pred (PBE grid s)", "pred (PBE fit s)"]))
    print(f"\nstress in fine-tuning loss: {train['stress_in_loss']}  ({train['verdict']})")
    bad = [r["item"] for r in res["recon_comparison"] if not r["agrees"]]
    print(f"\nrecon comparison: {len(res['recon_comparison']) - len(bad)}/{len(res['recon_comparison'])} "
          f"agree" + (f"; differ: {bad}" if bad else ""))
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()} ({len(res['inputs'])} inputs hashed)")


if __name__ == "__main__":
    main()
