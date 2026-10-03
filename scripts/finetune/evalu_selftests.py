"""Selftests of the evaluation code against existing results (BASE models only; no fine-tune).

  c3a_base    PBE curves rebuilt from pw.out equal dft_reference's own (c3a_curves.json), and the
              base model's recomputed well depths / depth ratios equal c3a_paths.csv.
  c3b_base    base-model force and energy errors on the C3b configurations equal c3b_units.csv.
  p2_base     the screen pipeline run here with the base calculator reproduces a production
              ledger call (BaTiO3 / MACE-MP-0 or CHGNet, 300 K) and its curvature frequency.
  s2_base     the harmonic control calls reproduce the ledger's (two controls).
  hook        MLIP_DYNSTAB_CKPT_<MODEL> loads a checkpoint (--ckpt), and an unset variable leaves
              the production loader unchanged (same energy as a direct mace_mp / CHGNet call).
  verdicts    the P1/P2/S2 verdict logic on synthetic replicate patterns.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from . import common as C
from . import evalu
from .pbe import PBEData


def _result(name, ok, **info):
    flag = "PASS" if ok else "FAIL"
    print(f"[selftest] {flag} {name}: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    return bool(ok)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(dict(workers=1, max_paths=None, systems=list(C.TEST_SYSTEMS)))
        self.__dict__.update(kw)


def _csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def t_c3a_base(args) -> bool:
    with tempfile.TemporaryDirectory(prefix="ft_selftest_") as tmp:
        return _t_c3a_base(args, Path(tmp))


def _t_c3a_base(args, out) -> bool:
    model = args.model
    pbe = PBEData(C.DFT_ROOT, C.TEST_SYSTEMS)
    paths = [p for p in pbe.c3a_paths() if p["status"] in ("complete", "partial")]
    rows = {(r["stem"], r["curve"]): r for r in _csv_rows(C.DFT_ROOT / "c3a_paths.csv")}
    curves = {(c["stem"], c["curve"]): c for c in C.jload(C.DFT_ROOT / "c3a_curves.json")["curves"]}
    in_csv = [p for p in paths if (p["stem"], "pbe") in rows]
    # 1. PBE curves
    d_pbe = []
    for p in in_csv:
        ref = curves[(p["stem"], "pbe")]
        mine = (p["E"] - p["E"][0]) * 1000.0
        if len(ref["dE_meV"]) == len(mine):
            d_pbe.append(float(np.abs(mine - np.array(ref["dE_meV"])).max()))
        else:
            d_pbe.append(float("inf"))
    ok1 = bool(d_pbe) and max(d_pbe) < 1e-6
    # 2. base model
    handle = evalu.load_handle(model, "base", out, args.device)
    res = evalu.part_c3a(_Args(device=args.device, out=out), handle, model, "base", out, pbe)
    dd, ratio_d = [], []
    for stem, r in res["paths"].items():
        ref = rows.get((stem, model))
        pb = rows.get((stem, "pbe"))
        if ref is None or pb is None:
            continue
        dd.append(abs(r["metrics"]["depth_meV"] - float(ref["depth_meV"])))
        if r["depth_ratio"] is not None and float(pb["depth_meV"]) >= evalu.DEPTH_FLOOR_MEV:
            ratio_d.append(abs(r["depth_ratio"] - float(ref["depth_meV"]) / float(pb["depth_meV"])))
    ok2 = bool(dd) and max(dd) < 0.05 and max(ratio_d) < 2e-3
    dec = [(s, round(r["depth_ratio"], 3)) for s, r in res["paths"].items()
           if r["role"] == "decide" and r["path_model"] == model and r["system"] in ("batio3_cubic", "knbo3_cubic")]
    return _result("c3a_base", ok1 and ok2, model=model, n_paths_rebuilt=len(paths), n_in_csv=len(in_csv),
                   max_pbe_curve_diff_meV=max(d_pbe) if d_pbe else None, n_depth_compared=len(dd),
                   max_base_depth_diff_meV=round(max(dd), 5) if dd else None,
                   max_ratio_diff=round(max(ratio_d), 6) if ratio_d else None,
                   own_deciding_ratios_batio3_knbo3=dec)


def t_c3b_base(args) -> bool:
    with tempfile.TemporaryDirectory(prefix="ft_selftest_") as tmp:
        return _t_c3b_base(args, Path(tmp))


def _t_c3b_base(args, out) -> bool:
    model = args.model
    pbe = PBEData(C.DFT_ROOT, C.TEST_SYSTEMS)
    handle = evalu.load_handle(model, "base", out, args.device)
    res = evalu.part_c3b(_Args(device=args.device, out=out), handle, model, "base", out, pbe)
    ref = [r for r in _csv_rows(C.DFT_ROOT / "c3b_units.csv")
           if r["eval_model"] == model and r["system"] == "batio3_cubic" and r["owner_model"] == "mace_mp0"]
    bad, n = [], 0
    for r in ref:
        sub = [c for c in res["configs"] if c["role"] == r["set"] and c["system"] == "batio3_cubic"]
        if not sub:
            continue
        n += 1
        rm = float(np.sqrt(sum(c["f_sumsq"] for c in sub) / sum(c["n_components"] for c in sub)))
        ee = [c["e_err_meV_atom"] for c in sub if "e_err_meV_atom" in c]
        e_rms = float(np.sqrt(np.mean(np.square(ee)))) if ee else None
        bad.append((r["set"], round(rm, 5), round(float(r["force_rmse"]), 5),
                    None if e_rms is None else round(e_rms, 4),
                    None if not r.get("e_err_rms_meV_per_atom") else round(float(r["e_err_rms_meV_per_atom"]), 4)))
    ok = n >= 1 and all(abs(b[1] - b[2]) < 2e-3 and (b[3] is None or b[4] is None or abs(b[3] - b[4]) < 0.1)
                        for b in bad)
    return _result("c3b_base", ok, model=model, n_sets=n, mine_vs_csv_f_and_e=bad)


def t_p2_base(args) -> bool:
    with tempfile.TemporaryDirectory(prefix="ft_selftest_") as tmp:
        return _t_p2_base(args, Path(tmp))


def _t_p2_base(args, out) -> bool:
    from mlip_dynstab.analysis import load_canonical
    model = args.model
    handle = evalu.load_handle(model, "base", out, args.device)
    a = _Args(device=args.device, out=out, systems=["batio3_cubic"])
    # one T and two systems is enough; p2 maps are cached per (system, model, tag) under out/p2cache
    old = C.LADDER_T
    C.LADDER_T = (300.0,)
    try:
        res = evalu.part_p2(a, handle, model, "base", out)
    finally:
        C.LADDER_T = old
    df = load_canonical()
    led = df[(df.method == "softmode") & (df.system == "batio3_cubic") & (df.model == model)
             & (df.temperature_K == 300.0)].iloc[0]
    row = res["rows"][0]
    ok = (row["pred_stable"] == bool(led.pred_stable)
          and abs(row["min_eff_freq_thz"] - float(led.min_eff_freq_thz)) < 0.05)
    return _result("p2_base", ok, model=model, mine_pred=row["pred_stable"], ledger_pred=bool(led.pred_stable),
                   mine_min_eff=round(row["min_eff_freq_thz"], 4), ledger_min_eff=round(float(led.min_eff_freq_thz), 4))


def t_s2_base(args) -> bool:
    from mlip_dynstab.analysis import load_canonical
    from mlip_dynstab.harmonic import compute_harmonic
    from mlip_dynstab.systems import build_atoms, get_spec
    model = args.model
    handle = evalu.load_handle(model, "base", Path(args.out), args.device)
    df = load_canonical()
    ok, info = True, {}
    for sid in ("si_diamond", "mgo_rocksalt"):
        res = compute_harmonic(build_atoms(get_spec(sid)), handle.calc, supercell=C.SUPERCELL, disp=C.FC_DISP_ANG)
        led = df[(df.method == "harmonic") & (df.system == sid) & (df.model == model)].iloc[0]
        info[sid] = (round(res.min_freq_thz, 4), round(float(led.min_freq_thz), 4))
        ok &= bool(res.dynamically_stable) == bool(led.pred_stable) and abs(res.min_freq_thz - float(led.min_freq_thz)) < 0.02
    return _result("s2_base", ok, model=model, mine_vs_ledger_min_freq=info)


def t_hook(args) -> bool:
    from mlip_dynstab import calculators as K
    model = args.model
    ck = getattr(args, "ckpt", None)
    # unset variable: the production loader is unchanged
    os.environ.pop(K._CKPT_ENV + model.upper(), None)
    from ase.build import bulk
    at = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    h0 = K._load_calculator(model, "cpu")
    a = at.copy()
    a.calc = h0.calc
    e0 = a.get_potential_energy()
    if model == "mace_mp0":
        from mace.calculators import mace_mp
        b = at.copy()
        b.calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    else:
        from chgnet.model.dynamics import CHGNetCalculator
        b = at.copy()
        b.calc = CHGNetCalculator(use_device="cpu")
    same = abs(e0 - b.get_potential_energy()) < 1e-10
    ok, info = same, {"unset_var_equals_direct_loader": same, "version": h0.version}
    if ck:
        p = Path(ck)
        os.environ[K._CKPT_ENV + model.upper()] = str(p)
        try:
            h1 = K._load_calculator(model, "cpu")
        finally:
            os.environ.pop(K._CKPT_ENV + model.upper(), None)
        c = at.copy()
        c.calc = h1.calc
        e1 = c.get_potential_energy()
        info.update({"ft_version": h1.version, "ft_energy_differs_from_base": abs(e1 - e0) > 1e-9})
        ok &= "finetuned" in h1.version
    return _result("hook", ok, **info)


def t_verdicts(args) -> bool:
    from . import summary as S
    ok = True
    # P2 logic: base wrong (called stable, label unstable)
    ok &= S._status(True, [False, False, False], "corrected") == "supported"
    ok &= S._status(True, [True, True, True], "corrected") == "refuted"
    ok &= S._status(True, [False, True, False], "corrected") == "unresolved"
    ok &= S._status(False, [False, False, False], "persist") == "supported"
    ok &= S._status(False, [True, True, True], "persist") == "refuted"
    ok &= S._status(False, [False, True, True], "persist") == "unresolved"
    ok &= S._status(True, [True, True, None], "persist") == "pending (replicates missing)"
    # P1 band logic
    ok &= S._verdict_band([0.9, 1.0, 1.1, 0.95, 1.0, 1.05], [[0.9, 1.0], [1.1, 0.95], [1.0, 1.05]], 0.8, 1.2) == "supported"
    ok &= S._verdict_band([0.5, 0.6, 0.55, 0.5, 0.6, 0.55], [[0.5, 0.6], [0.55, 0.5], [0.6, 0.55]], 0.8, 1.2) == "refuted"
    ok &= S._verdict_band([0.9, 0.95, 1.0, 1.0, 0.5, 0.55], [[0.9, 0.95], [1.0, 1.0], [0.5, 0.55]], 0.8, 1.2) == "unresolved"
    return _result("verdicts", ok)


TESTS = {"c3a_base": t_c3a_base, "c3b_base": t_c3b_base, "p2_base": t_p2_base,
         "s2_base": t_s2_base, "hook": t_hook, "verdicts": t_verdicts}
