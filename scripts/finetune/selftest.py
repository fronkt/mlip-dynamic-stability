"""Unit tests of the fine-tuning trial's parsing and scoring code against EXISTING data.

No training happens here.  What each test proves:

  qe_writer     pw_input() reproduces, byte for byte (except the one comment line naming the
                writing script), existing pw.in files of the C3b (BaTiO3, SrTiO3) and C3a (KNbO3,
                CsSnBr3) jobs from their geometry files: same cutoffs per element, k-mesh,
                smearing, conv_thr, pseudos.
  guard         the leakage guard rejects a configuration built from a held-out geometry plus a
                0.02 A rattle and accepts one 0.2 A away; the distance is mean-removed.
  c3a_base      PBE E(Q) curves rebuilt from the pw.out files equal dft_reference's own
                (results/revision/dft/c3a_curves.json), and the BASE model's well-depth ratio
                to PBE, recomputed here on the geometries, equals the one in c3a_paths.csv.
  c3b_base      PBE force/energy bookkeeping on the C3b jobs, base model, against c3b_units.csv.
  parse         parse_pw_out reads energies/forces of finished jobs consistently with ASE.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from . import common as C


def _result(name, ok, **info):
    flag = "PASS" if ok else "FAIL"
    print(f"[selftest] {flag} {name}: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    return bool(ok)


def t_qe_writer(args) -> bool:
    from . import qe
    sssp = qe.load_table(None)
    ok_all = True
    picks = []
    for jj in sorted(C.QE_ROOT.glob("b_*/job.json"))[:200]:
        m = C.jload(jj)
        if m["system"] in ("batio3_cubic", "srtio3_cubic") and m["members"][0]["frame"] == 0:
            picks.append(jj)
    picks = picks[:2] + [p for p in picks if "_ref" in p.parent.name][:1]
    for jj in sorted(C.QE_ROOT.glob("a_*knbo3*/job.json")) + sorted(C.QE_ROOT.glob("a_*cssnbr3*/job.json")):
        m = C.jload(jj)
        if not m["members"][0].get("reduction"):
            picks.append(jj)
            break
    # one of each system with no cell reduction
    seen = {C.jload(p)["system"] for p in picks}
    for s in ("knbo3_cubic", "cssnbr3_cubic"):
        if s in seen:
            continue
        for jj in sorted(C.QE_ROOT.glob(f"a_{s}_*/job.json")):
            if not C.jload(jj)["members"][0].get("reduction"):
                picks.append(jj)
                break
    n = 0
    for jj in picks:
        m = C.jload(jj)
        mem = m["members"][0]
        frames = C.read_frames(C.DFT_ROOT / mem["geom_file"])
        a = frames[int(mem["frame"])]
        got, settings = qe.pw_input(a, m["job"], sssp, 0.25, 0.005, script="scripts/dft_reference.py")
        want = (jj.parent / "pw.in").read_text(encoding="utf-8")
        same = got == want
        n += 1
        ok_all &= same
        if not same:
            gl, wl = got.splitlines(), want.splitlines()
            diff = [(i, g, w) for i, (g, w) in enumerate(zip(gl, wl)) if g != w][:3]
            print("   first differing lines:", diff, len(gl), len(wl))
        ok_all &= (settings["nk_irr_spglib"] == m["qe"]["nk_irr_spglib"]
                   and settings["kmesh"] == m["qe"]["kmesh"]
                   and settings["pseudos"] == m["qe"]["pseudos"])
    return _result("qe_writer", ok_all and n >= 4, n_jobs_compared=n,
                   systems=sorted({C.jload(p)["system"] for p in picks}))


def t_guard(args) -> bool:
    from . import gen
    ok = True
    info = {}
    for system in C.TEST_SYSTEMS:
        held = gen.HeldOut(system)
        i = int(np.argmax(np.sqrt((held.F ** 2).sum(-1).mean(-1))))       # the most displaced frame
        f0 = held.F[i]
        a = float(held.a[i])
        rng = np.random.default_rng(1)
        near = f0 + rng.normal(size=f0.shape) * (0.02 / a) / np.sqrt(3.0)        # ~0.02 A rattle
        far = f0 + rng.normal(size=f0.shape) * (0.2 / a) / np.sqrt(3.0)
        d_near, src_near, _ = held.nearest(near, a)
        d_far, _, _ = held.nearest(far, a)
        shifted = near + np.array([0.3, -0.2, 0.1])                       # rigid translation only
        d_shift, _, _ = held.nearest(shifted, a)
        info[system] = (round(d_near, 4), round(d_far, 3), round(d_shift, 4))
        ok &= d_near < C.LEAK_RMS_A and d_far > C.LEAK_RMS_A and abs(d_shift - d_near) < 1e-9
    return _result("guard", ok, near_far_shifted=info, radius=C.LEAK_RMS_A)


def t_configs_repro(args) -> bool:
    """The frozen configurations are exactly what the documented seeds regenerate from the stored
    force constants (no model needed), and the C2 sampler's own unit checks pass."""
    from . import gen
    out = Path(args.out)
    man = C.jload(out / "manifest.json")
    gen.force_spread().selftest_sampler()
    ok, info = True, {}
    held_cache = {}
    for key, ent in man["entries"].items():
        z = np.load(out / ent["fc_file"])
        fcd = {k: z[k] for k in ("fc", "masses", "numbers", "cell", "positions")}
        system = ent["system"]
        held = held_cache.setdefault(system, gen.HeldOut(system))
        frames, _, per_t, rej, _, _ = gen.draw_configs(system, ent["model"], fcd, held,
                                                       min_pair_ratio=ent.get("min_pair_ratio_rule", 0.0))
        disk = C.read_frames(out / ent["configs_file"])
        d = max(float(np.abs(a.get_positions() - b.get_positions()).max()) for a, b in zip(frames, disk))
        same = len(frames) == len(disk) == C.N_CONFIGS and d < 5e-8
        info[key] = round(d, 10)
        ok &= same
    return _result("configs_repro", ok, max_abs_position_diff_A=info)


def t_dataset_parse(args) -> bool:
    """parse_pw_out on finished C3b jobs: energies/forces equal an independent ASE read, positions
    equal the geometry, convergence flags set."""
    from ase.io import read
    from . import data
    ok, n = True, 0
    for jj in sorted(C.QE_ROOT.glob("b_batio3_cubic_mace_mp0_T100_s0_*/job.json"))[:2]:
        d = jj.parent
        r = data.parse_pw_out(d / "pw.out")
        a = read(d / "pw.out", format="espresso-out", index=-1)
        m = C.jload(jj)["members"][0]
        frame = C.read_frames(C.DFT_ROOT / m["geom_file"])[int(m["frame"])]
        ok &= (r["job_done"] and r["converged"] and abs(r["energy_eV"] - a.get_potential_energy()) < 1e-12
               and np.abs(r["forces"] - a.get_forces()).max() < 1e-12
               and data.pos_mismatch(r["atoms"], frame) < 1e-5)
        n += 1
    return _result("dataset_parse", ok and n == 2, n_jobs=n)


TESTS = {"qe_writer": t_qe_writer, "guard": t_guard, "configs_repro": t_configs_repro,
         "dataset_parse": t_dataset_parse}


def run(args) -> int:
    try:
        from . import evalu_selftests
        TESTS.update(evalu_selftests.TESTS)
    except ImportError:
        pass
    names = list(TESTS) if args.which == ["all"] else args.which
    bad = []
    for n in names:
        try:
            if not TESTS[n](args):
                bad.append(n)
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            _result(n, False, error=f"{type(exc).__name__}: {exc}")
            bad.append(n)
    print(f"[selftest] {len(names) - len(bad)}/{len(names)} passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0
