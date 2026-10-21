"""How many of the soft-mode screen's instability calls are curvature-blind.

For each screened mode at each temperature the production screen (mlip_dynstab/finite_t.py,
softmode method generation 4) records two observables of the same 1-D SCHA free energy F(Q0):

  * the CALL (``_solve_scha``): F is scanned over 121 centroids Q0 in [0, 0.6] Ang and the
    mode CONDENSES if the global minimum sits at Q0 > 1.5 grid steps (0.0075 Ang). A unit
    (system, model, T) is called unstable if ANY screened mode condenses;
  * the CURVATURE (``_sym_curvature_freq``): the signed frequency of F''(0), the free-energy
    curvature at the symmetric point.

A condensed mode whose curvature is still positive is "curvature-blind": the global minimum
of F has moved off Q0 = 0 while Q0 = 0 remains a local minimum, so any criterion that reads
the curvature at the high-symmetry reference cannot see that condensation. The ledger already
counts these per unit (``ft_n_curv_blind``, finite_t.py ``n_curv_blind``). This script tallies
them over the canonical screen generation, per unit and per mode-temperature evaluation, with
breakdowns by temperature, registry family (``klass``), model and system, the three units
used as examples, and the attribution of the condensed modes whose curvature is NOT positive.

Nothing is measured. Inputs:
  results/ledger.parquet                 via analysis.load_canonical() (superseded softmode
                                         generations dropped; one row per unit is asserted)
  configs/curated_systems.yaml           via systems.load_specs() (family = ``klass``)
  results/curvature_identity_check.json  written by scripts/curvature_identity_check.py from the
                                         cached E(Q) maps; used only to attribute the
                                         non-positive curvatures, and only if its counts match
Output:
  results/revision/r2/curv_blind.json    (each input recorded with its sha256 and git blob hash,
                                         text hashed as git stores it, CRLF -> LF)

numpy/pandas only (scipy is not imported: finite_t imports it inside its solvers, never at
module level, and no solver is called here). Deterministic: no timestamps, no sampling.

Usage: python scripts/curv_blind_summary.py [--out results/revision/r2/curv_blind.json]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # match the convention in the sibling scripts

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

from mlip_dynstab import METHOD_VERSION, SOFTMODE_EQMAP_VERSION  # noqa: E402
from mlip_dynstab import analysis as A  # noqa: E402
from mlip_dynstab import finite_t as FT  # noqa: E402
from mlip_dynstab.systems import load_specs  # noqa: E402

LEDGER = REPO / "results" / "ledger.parquet"
IDENTITY = REPO / "results" / "curvature_identity_check.json"
SYSTEMS_YAML = REPO / "configs" / "curated_systems.yaml"
FINITE_T_SRC = REPO / "mlip_dynstab" / "finite_t.py"
MANIFEST = REPO / "results" / "MANIFEST.sha256"
OUT = REPO / "results" / "revision" / "r2" / "curv_blind.json"

KEYS = ["system", "model", "temperature_K"]
ILLUSTRATIVE = [("batio3_cubic", "mace_mp0", 100.0),
                ("knbo3_cubic", "mace_mp0", 300.0),
                ("srtio3_cubic", "mace_mp0", 100.0)]


# ------------------------------------------------------------------ provenance ----

def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def _git_index_blob(rel: str) -> str | None:
    """Blob id git has staged for ``rel`` (None if git is unavailable or the file untracked)."""
    try:
        r = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "--", rel],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return r.stdout.split()[1]


def _manifest_sha256(rel: str) -> str | None:
    if not MANIFEST.exists():
        return None
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and not line.startswith("#") and parts[2] == rel:
            return parts[0]
    return None


def _provenance(p: Path, text: bool) -> dict:
    """sha256 and git blob sha1 of a file. text=True hashes it as git stores it (CRLF -> LF),
    as build_esi_tables._sha256 / verify_claims / make_manifest do, so a Windows autocrlf
    checkout of the same commit gives the same digests; binary files are hashed raw."""
    data = Path(p).read_bytes()
    if text:
        data = data.replace(b"\r\n", b"\n")
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    rel = _rel(p)
    idx = _git_index_blob(rel)
    man = _manifest_sha256(rel)
    sha = hashlib.sha256(data).hexdigest()
    return {
        "path": rel,
        "bytes_hashed": len(data),
        "crlf_to_lf": bool(text),
        "sha256": sha,
        "git_blob_sha1": blob,
        "matches_git_index_blob": None if idx is None else bool(idx == blob),
        "matches_results_manifest_sha256": None if man is None else bool(man == sha),
    }


# ------------------------------------------------------------------- the screen ----

def _condense_threshold_ang() -> dict:
    """The call's Q0 tolerance, read from _solve_scha's own defaults (q_thresh = 1.5 steps)."""
    sig = inspect.signature(FT._solve_scha).parameters
    q_box, nq = float(sig["q_box"].default), int(sig["nq"].default)
    step = q_box / (nq - 1)
    return {"q_box_ang": q_box, "nq": nq, "grid_step_ang": step, "q_thresh_ang": 1.5 * step}


def load_screen() -> tuple[pd.DataFrame, dict]:
    """Canonical softmode rows (one per unit), with the generation filter checked."""
    raw = A.load_ledger(LEDGER)
    can = A.canonical(raw)
    sm = can[can["method"] == "softmode"].copy()
    versions = sorted(pd.to_numeric(sm["method_version"], errors="coerce").unique().tolist())
    assert versions == [float(METHOD_VERSION["softmode"])], versions
    assert not sm.duplicated(KEYS).any(), "a (system, model, T) unit appears twice"
    call_stable = sm["pred_stable"].astype(bool)
    assert (call_stable == sm["dynamically_stable"].astype(bool)).all()
    # "unstable" means ANY screened mode condenses
    assert (call_stable == (sm["ft_n_condensed"] == 0)).all()

    specs = {s.id: s for s in load_specs()}
    sm["family"] = sm["system"].map(lambda s: specs[s].klass)
    assert (sm["family"] == sm["klass"]).all(), "ledger klass disagrees with the registry"
    sm["borderline"] = sm["system"].map(lambda s: specs[s].borderline)

    gen = {
        "selector": "mlip_dynstab.analysis.load_canonical() / canonical(): per method, keep "
                    "the highest method_version present",
        "softmode_rows_in_raw_ledger": int((raw["method"] == "softmode").sum()),
        "softmode_rows_canonical": int(len(sm)),
        "superseded_softmode_rows_dropped": int((raw["method"] == "softmode").sum() - len(sm)),
        "one_row_per_unit_asserted": True,
    }
    return sm.sort_values(KEYS).reset_index(drop=True), gen


def explode_modes(sm: pd.DataFrame, q_thresh: float) -> pd.DataFrame:
    """One row per screened mode per unit (= per mode-temperature evaluation)."""
    rows = []
    for r in sm.itertuples(index=False):
        f = {k: str(getattr(r, k)).split(";") for k in
             ("ft_modes_q", "ft_modes_harm_thz", "ft_modes_curv_thz", "ft_modes_Q0_ang",
              "ft_modes_stable")}
        n = len(f["ft_modes_stable"])
        assert all(len(v) == n for v in f.values()) and n == int(r.ft_n_imag_screened)
        placeholder = int(r.ft_n_imag_total) == 0
        assert not placeholder or n == 1
        for i in range(n):
            rows.append({"system": r.system, "model": r.model, "temperature_K": r.temperature_K,
                         "family": r.family, "mode_index": i, "q": f["ft_modes_q"][i],
                         "harm_thz": float(f["ft_modes_harm_thz"][i]),
                         "curv_thz": float(f["ft_modes_curv_thz"][i]),
                         "Q0_ang": float(f["ft_modes_Q0_ang"][i]),
                         "condensed": f["ft_modes_stable"][i] == "0",
                         "placeholder": placeholder})
    m = pd.DataFrame(rows)
    # the call is argmin Q0 > q_thresh, mode by mode
    assert (m["condensed"] == (m["Q0_ang"] > q_thresh)).all()
    assert not m.loc[m["placeholder"], "condensed"].any()
    m["curv_blind"] = m["condensed"] & (m["curv_thz"] > 0)
    m["curv_nonpos"] = m["condensed"] & (m["curv_thz"] <= 0)
    # ft_modes_curv_thz is rounded to 3 dp; the ledger's n_curv_blind used the unrounded value.
    # They must agree unit by unit, or a |curvature| < 5e-4 THz sign is being misread.
    per = m.groupby(KEYS)[["curv_blind", "condensed"]].sum()
    chk = sm.set_index(KEYS)
    assert (per["curv_blind"].astype(int) == chk["ft_n_curv_blind"].astype(int)).all()
    assert (per["condensed"].astype(int) == chk["ft_n_condensed"].astype(int)).all()
    return m


def _frac(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def counts(units: pd.DataFrame, modes: pd.DataFrame) -> dict:
    uns = units[~units["pred_stable"].astype(bool)]
    nb = int((uns["ft_n_curv_blind"] > 0).sum())
    n_cond = int(modes["condensed"].sum())
    n_blind = int(modes["curv_blind"].sum())
    return {
        "n_units": int(len(units)),
        "n_units_unstable": int(len(uns)),
        "n_units_unstable_curv_blind": nb,
        "frac_unstable_units_curv_blind": _frac(nb, len(uns)),
        "n_screened_mode_evals": int((~modes["placeholder"]).sum()),
        "n_condensed_mode_evals": n_cond,
        "n_curv_blind_mode_evals": n_blind,
        "n_negative_curvature_mode_evals": int(modes["curv_nonpos"].sum()),
        "frac_condensed_mode_evals_curv_blind": _frac(n_blind, n_cond),
    }


def breakdown(units: pd.DataFrame, modes: pd.DataFrame, col: str) -> dict:
    out = {}
    for key, g in units.groupby(col, sort=True):
        k = str(int(key)) if col == "temperature_K" else str(key)
        out[k] = counts(g, modes[modes[col] == key])
    return out


def by_system(units: pd.DataFrame, modes: pd.DataFrame) -> dict:
    out = breakdown(units, modes, "system")
    for s, v in out.items():
        u = units[units["system"] == s]
        out[s] = {"family": str(u["family"].iloc[0]),
                  "borderline_label": bool(u["borderline"].iloc[0])} | v
    return out


def zero_blind_unstable(units: pd.DataFrame, modes: pd.DataFrame) -> list[dict]:
    z = units[~units["pred_stable"].astype(bool) & (units["ft_n_curv_blind"] == 0)]
    out = []
    for r in z.itertuples(index=False):
        mm = modes[(modes["system"] == r.system) & (modes["model"] == r.model)
                   & (modes["temperature_K"] == r.temperature_K) & modes["condensed"]]
        out.append({"system": r.system, "model": r.model, "T_K": float(r.temperature_K),
                    "family": r.family, "n_condensed": int(r.ft_n_condensed),
                    "condensed_modes": [{"q": x.q, "argmin_Q0_ang": x.Q0_ang,
                                         "curvature_thz": x.curv_thz}
                                        for x in mm.itertuples(index=False)]})
    return out


def illustrative(units: pd.DataFrame, modes: pd.DataFrame) -> list[dict]:
    out = []
    for sys_, mod, t in ILLUSTRATIVE:
        u = units[(units["system"] == sys_) & (units["model"] == mod)
                  & (units["temperature_K"] == t)]
        assert len(u) == 1, (sys_, mod, t)
        r = u.iloc[0]
        mm = modes[(modes["system"] == sys_) & (modes["model"] == mod)
                   & (modes["temperature_K"] == t)]
        cond = mm[mm["condensed"]]
        ladder = units[(units["system"] == sys_) & (units["model"] == mod)]
        entry = {
            "system": sys_, "model": mod, "T_K": t,
            "label_stable": bool(r["gt_stable"]),
            "transition_T_K": None if pd.isna(r["transition_T_K"]) else float(r["transition_T_K"]),
            "screen_call_stable": bool(r["pred_stable"]),
            "n_screened_modes": int(r["ft_n_imag_screened"]),
            "n_condensed": int(r["ft_n_condensed"]),
            "n_curv_blind": int(r["ft_n_curv_blind"]),
            "unit_argmin_Q0_ang": float(r["ft_order_param_Q_ang"]),
            "ledger_min_eff_freq_thz": round(float(r["min_eff_freq_thz"]), 3),
            "condensed_modes": [{"q": x.q, "harm_thz": x.harm_thz, "argmin_Q0_ang": x.Q0_ang,
                                 "curvature_thz": x.curv_thz, "curv_blind": bool(x.curv_blind)}
                                for x in cond.itertuples(index=False)],
            "all_screened_modes": [{"q": x.q, "harm_thz": x.harm_thz, "argmin_Q0_ang": x.Q0_ang,
                                    "curvature_thz": x.curv_thz, "condensed": bool(x.condensed)}
                                   for x in mm.itertuples(index=False)],
            "same_system_model_by_T": {
                str(int(x.temperature_K)): {"label_stable": bool(x.gt_stable),
                                            "screen_call_stable": bool(x.pred_stable),
                                            "n_condensed": int(x.ft_n_condensed),
                                            "n_curv_blind": int(x.ft_n_curv_blind)}
                for x in ladder.itertuples(index=False)},
        }
        if cond.empty:
            entry["note"] = ("no mode condenses at this T: the screen calls the unit stable, so "
                             "there is no condensed-mode curvature to report"
                             + ("; the label is unstable (a screen false-stable)"
                                if not entry["label_stable"] else ""))
        out.append(entry)
    # scripts/verify_claims.py [2.4] asserts this value; keep the two in step
    st = next(e for e in out if e["system"] == "srtio3_cubic" and e["T_K"] == 100.0)
    assert st["n_curv_blind"] == 1
    return out


def negative_attribution(modes: pd.DataFrame) -> dict:
    """Account for the condensed modes with curvature <= 0, from what the code records."""
    neg = modes[modes["curv_nonpos"]]
    real = modes[~modes["placeholder"]]
    ph = modes[modes["placeholder"]]
    out: dict = {
        "n_condensed_curvature_le_0": int(len(neg)),
        "n_exactly_zero": int((neg["curv_thz"] == 0).sum()),
        "n_negative_screened_mode_evals_any_call": int((real["curv_thz"] < 0).sum()),
        "n_negative_screened_mode_evals_not_condensed": int(
            ((real["curv_thz"] < 0) & ~real["condensed"]).sum()),
        "placeholder_rows_note": (
            f"{int((ph['curv_thz'] <= 0).sum())} further evaluations with curvature <= 0 are the "
            "placeholder rows of harmonically stable (system, model) pairs (ft_n_imag_total = 0), "
            "where finite_t records the softest commensurate harmonic frequency in place of a "
            "curvature; they are never condensed and are excluded from every mode count here"),
        "by_family": {str(k): int(v) for k, v in neg.groupby("family").size().items()},
        "by_system_model": {f"{s}/{mo}": int(v)
                            for (s, mo), v in neg.groupby(["system", "model"]).size().items()},
    }
    if not IDENTITY.exists():
        out["attribution"] = "unattributed (results/curvature_identity_check.json absent)"
        return out
    ident = json.loads(IDENTITY.read_text(encoding="utf-8"))
    roots = ident.get("negative_with_root_at_every_stencil_point", [])
    matched = []
    for rec in roots:
        sm, t = rec["unit"], float(rec["T"])
        hit = neg[(neg["system"] + "_" + neg["model"] == sm) & (neg["temperature_K"] == t)
                  & (np.abs(neg["harm_thz"] - rec["harm_thz"]) < 6e-4)
                  & (np.abs(neg["curv_thz"] - rec["omega_eff_thz"]) < 6e-4)]
        if len(hit) == 1:
            h = hit.iloc[0]
            matched.append({"system": h["system"], "model": h["model"], "T_K": t, "q": h["q"],
                            "harm_thz": h["harm_thz"], "curvature_thz": h["curv_thz"],
                            "fit_a_b_c": [rec["a"], rec["b"], rec["c"]]})
    n_id = int(ident.get("n_negative_omega_eff", -1))
    n_fb = int(ident.get("n_negative_with_width_fallback_at_a_stencil_point", -1))
    consistent = (n_id == len(neg) == out["n_negative_screened_mode_evals_any_call"]
                  and len(matched) == len(roots) and n_fb == n_id - len(roots)
                  and int(ident.get("n_mode_T_evaluations", -1)) == len(real))
    out["identity_check_counts"] = {
        "n_mode_T_evaluations": ident.get("n_mode_T_evaluations"),
        "n_negative_omega_eff": n_id,
        "n_negative_with_width_fallback_at_a_stencil_point": n_fb,
        "n_negative_with_root_at_every_stencil_point": len(roots),
    }
    out["counts_consistent_with_identity_check"] = bool(consistent)
    if consistent:
        out["attribution"] = {
            "width_solver_fallback_at_a_stencil_point": n_fb,
            "root_at_every_stencil_point": len(matched),
            "root_at_every_stencil_point_units": matched,
            "how_attributed": (
                "scripts/curvature_identity_check.py recomputes _sym_curvature_freq on the "
                "cached E(Q) maps of every non-placeholder mode at the four ladder temperatures "
                "(the same evaluations counted here) and classifies each negative value: either "
                "_scha_branch's width solver found no downward sign change of its residual at "
                "one of the three stencil points (0, dQ, 2 dQ; dQ = 0.005 Ang) and returned its "
                "grid-nearest fallback width, or a self-consistent root existed at all three. "
                "Its totals equal the counts here and its two listed root-at-every-point cases "
                "match two of these condensed modes by (unit, T, harmonic frequency, curvature), "
                "so the remainder is the fallback class by count; that file does not list the "
                "fallback cases individually. Every negative curvature on a screened mode "
                "belongs to a condensed mode."),
            "what_the_code_says": (
                "finite_t._sym_curvature_freq returns None when no bound Gaussian exists at "
                "Q0 = 0 ('not observed on the production set'); compute_finite_t_softmode then "
                "substitutes the signed effective frequency, which is negative for a condensed "
                "mode. The identity check skips None, so the equal totals mean no such "
                "substitution is among these values. scripts/curvature_identity_check.py's "
                "docstring states F''(0) = <V''>_{0,sigma} = M Omega^2 for a single even mode, "
                "i.e. the curvature is positive whenever a bound self-consistent Gaussian exists "
                "at Q0 = 0. For the root-at-every-point class it lists the fitted a, b, c 'so a "
                "fit artefact can be recognised: a > 0 means no small-Q instability'."),
            "root_cases_all_have_a_gt_0": bool(all(x["fit_a_b_c"][0] > 0 for x in matched)),
        }
    else:
        out["attribution"] = ("curvature <= 0; not attributed (counts differ from "
                              "results/curvature_identity_check.json)")
    return out


def blind_mode_stats(modes: pd.DataFrame, thr: dict) -> dict:
    c = modes[modes["condensed"]]
    b = modes[modes["curv_blind"]]
    return {
        "condensed_argmin_Q0_ang": {"min": float(c["Q0_ang"].min()),
                                    "median": float(c["Q0_ang"].median()),
                                    "max": float(c["Q0_ang"].max()),
                                    "min_in_grid_steps": round(float(c["Q0_ang"].min())
                                                               / thr["grid_step_ang"], 2),
                                    "n_at_scan_box_edge": int(
                                        (c["Q0_ang"] >= thr["q_box_ang"] - 1e-9).sum()),
                                    "n_curv_blind_at_scan_box_edge": int(
                                        (b["Q0_ang"] >= thr["q_box_ang"] - 1e-9).sum())},
        "curv_blind_curvature_thz": {"min": float(b["curv_thz"].min()),
                                     "median": float(b["curv_thz"].median()),
                                     "max": float(b["curv_thz"].max())},
        "note": ("ft_modes_* fields are stored rounded (Q0 to 4 dp, curvature and harmonic "
                 "frequency to 3 dp); the per-unit blind count parsed from them is asserted equal "
                 "to the ledger's ft_n_curv_blind, which used unrounded curvatures"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    thr = _condense_threshold_ang()
    units, gen = load_screen()
    modes = explode_modes(units, thr["q_thresh_ang"])
    modes_real = modes[~modes["placeholder"]]

    top = counts(units, modes_real)
    uns = units[~units["pred_stable"].astype(bool)]
    res = {
        "_generated_by": "scripts/curv_blind_summary.py",
        "_purpose": ("referee answer: share of the single-mode screen's instability calls that "
                     "are curvature-blind (global minimum of F(Q0) displaced, F''(0) > 0)"),
        "source": {
            "ledger": _provenance(LEDGER, text=False),
            "registry": _provenance(SYSTEMS_YAML, text=True),
            "curvature_identity_check": (_provenance(IDENTITY, text=True)
                                         if IDENTITY.exists() else None),
            "screen_code": _provenance(FINITE_T_SRC, text=True),
        },
        "screen": {
            "method": "softmode",
            "method_version": int(METHOD_VERSION["softmode"]),
            "eqmap_cache_version": int(SOFTMODE_EQMAP_VERSION),
            "generation_filter": gen,
            "condensation_call": thr | {
                "rule": "finite_t._solve_scha: mode condenses iff argmin_Q0 F(Q0) > q_thresh"},
            "unit_call": "unstable iff ANY screened mode condenses (ft_n_condensed > 0)",
            "curvature": "finite_t._sym_curvature_freq: sign(F''(0)) sqrt(|F''(0)|/M_eff)/2pi",
        },
        "definitions": {
            "unit": "one (system, model, T) row of the canonical softmode ledger",
            "mode_eval": ("one screened imaginary commensurate mode of one unit at that unit's T "
                          "(one entry of ft_modes_*); placeholder rows of units with no imaginary "
                          "mode are excluded"),
            "curv_blind_mode_eval": "condensed AND curvature > 0 (finite_t n_curv_blind)",
            "negative_curvature_mode_eval": ("condensed AND curvature <= 0; see "
                                             "negative_curvature_attribution"),
            "unit_curv_blind": "unstable unit with ft_n_curv_blind >= 1",
            "scope": ("all 400 units: borderline-labelled systems are included and labels are "
                      "not used; this is a property of the screen, not an accuracy"),
        },
        **{k: top[k] for k in ("n_units", "n_units_unstable", "n_units_unstable_curv_blind",
                               "frac_unstable_units_curv_blind", "n_screened_mode_evals",
                               "n_condensed_mode_evals", "n_curv_blind_mode_evals",
                               "n_negative_curvature_mode_evals",
                               "frac_condensed_mode_evals_curv_blind")},
        "n_units_unstable_with_a_negative_curvature_condensed_mode": int(
            (modes_real[modes_real["curv_nonpos"]].groupby(KEYS).size() > 0).sum()),
        "unstable_units_not_curv_blind": zero_blind_unstable(units, modes_real),
        "n_units_stable_with_curv_blind": int(
            (units.loc[units["pred_stable"].astype(bool), "ft_n_curv_blind"] > 0).sum()),
        "by_temperature": breakdown(units, modes_real, "temperature_K"),
        "by_family": breakdown(units, modes_real, "family"),
        "by_model": breakdown(units, modes_real, "model"),
        "by_system": by_system(units, modes_real),
        "illustrative_units": illustrative(units, modes_real),
        "condensed_mode_stats": blind_mode_stats(modes_real, thr),
        "negative_curvature_attribution": negative_attribution(modes),
        "family_note": ("family = registry klass (configs/curated_systems.yaml); 'perovskite' "
                        "pools the oxide perovskites, KTaO3 and the Cs halides -- by_system "
                        "separates them"),
    }
    assert len(uns) == res["n_units_unstable"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print(f"screen softmode v{res['screen']['method_version']}: {res['n_units']} units "
          f"({gen['superseded_softmode_rows_dropped']} superseded rows dropped)")
    print(f"unstable units: {res['n_units_unstable']}; curvature-blind: "
          f"{res['n_units_unstable_curv_blind']} ({res['frac_unstable_units_curv_blind']:.3f})")
    print(f"condensed mode-T evals: {res['n_condensed_mode_evals']}; curvature-blind "
          f"{res['n_curv_blind_mode_evals']}; curvature <= 0 "
          f"{res['n_negative_curvature_mode_evals']}")
    print("unstable units with no curvature-blind mode: " + ", ".join(
        f"{u['system']}/{u['model']}/{int(u['T_K'])}K"
        for u in res["unstable_units_not_curv_blind"]))
    for t, v in res["by_temperature"].items():
        print(f"  T={t:>3s} K  unstable {v['n_units_unstable']:3d}  blind units "
              f"{v['n_units_unstable_curv_blind']:3d}  condensed {v['n_condensed_mode_evals']:3d}"
              f"  blind {v['n_curv_blind_mode_evals']:3d}  <=0 {v['n_negative_curvature_mode_evals']:3d}")
    na = res["negative_curvature_attribution"]
    print(f"curvature <= 0 consistent with curvature_identity_check.json: "
          f"{na.get('counts_consistent_with_identity_check')}")
    print(f"ledger blob {res['source']['ledger']['git_blob_sha1']} "
          f"(matches index: {res['source']['ledger']['matches_git_index_blob']}, manifest: "
          f"{res['source']['ledger']['matches_results_manifest_sha256']})")
    print(f"wrote {_rel(args.out)}")


if __name__ == "__main__":
    main()
