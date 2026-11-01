"""Does MACE-MP-0's own surface order at 100 K? Classical MD in the SSCHA's fixed 40-atom cell.

Exploratory, after the R2 centroid scan (tasks/md-order-check-2026-10-22.md, written before running;
not part of the 2026-10-21 pre-registration). Cubic BaTiO3 / MACE-MP-0 in the 2x2x2 cell of
results/revision/r2_centroid/gate/pattern.npz, cell fixed, Langevin NVT. Order is read from the Ti
off-centring relative to its O6 cage, Fourier-resolved over the eight q of the cell, through the
Binder cumulant of each 3-vector (0 disordered, 2/5 ordered).

    python scripts/md_order_check.py run --run A [--threads 4]     # resumable; A, B, C as in the note
    python scripts/md_order_check.py analyze                       # -> results/revision/r2_md/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parents[1]
PATTERN = REPO / "results" / "revision" / "r2_centroid" / "gate" / "pattern.npz"
OUT = REPO / "results" / "revision" / "r2_md"
DT_FS, FRICTION_PER_FS, EVERY = 2.0, 0.01, 5          # 2 fs step, frames every 10 fs
RUNS = {"A": {"T": 100.0, "start": "cubic", "eq_ps": 5.0, "prod_ps": 25.0, "seed": 101},
        "B": {"T": 100.0, "start": "gamma111", "eq_ps": 5.0, "prod_ps": 25.0, "seed": 202},
        "C": {"T": 20.0, "start": "gamma111", "eq_ps": 2.0, "prod_ps": 10.0, "seed": 303}}
U_ORDERED, U_DISORDERED = 0.30, 0.10
SSCHA_SIGMA = {"M": 0.0215, "G111": 0.0184}            # rms of the projections at the cubic centroid


def unthrottle() -> None:
    """Windows confines windowless background processes to the E-cores; opt this one out."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x0600, False, os.getpid())
    s = State(1, 1, 0)
    k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))
    k.CloseHandle(h)


def load_cell():
    p = np.load(PATTERN, allow_pickle=True)
    return {k: p[k] for k in ("symbols", "r0", "cell", "u_m", "u_g")} | {"q_gamma": float(p["q_gamma"])}


def cage_and_index(ref):
    """Six O neighbours of each Ti and each Ti's primitive-cell index in the 2x2x2 cell."""
    sym, r0, cell = ref["symbols"], ref["r0"], ref["cell"]
    inv = np.linalg.inv(cell)
    ti, o = np.where(sym == "Ti")[0], np.where(sym == "O")[0]
    cages, idx = [], []
    for i in ti:
        d = r0[o] - r0[i]
        f = d @ inv
        f -= np.round(f)
        dist = np.linalg.norm(f @ cell, axis=1)
        nb = o[np.argsort(dist)[:6]]
        if not np.all(np.sort(dist)[:6] < 2.3):
            raise RuntimeError("O6 cage not found")
        cages.append(nb)
        idx.append(np.floor((r0[i] @ inv) % 1.0 * 2 + 1e-6).astype(int))
    return ti, np.array(cages), np.array(idx)


def off_centring(pos, ref, ti, cages):
    """(frames, 8, 3) Ti displacement from the centroid of its O6 cage, minimum image."""
    cell = ref["cell"]
    inv = np.linalg.inv(cell)
    d = pos[:, cages, :] - pos[:, ti, None, :]
    f = d @ inv
    f -= np.round(f)
    return -(f @ cell).mean(axis=2)


QS = [(a, b, c) for a in (0, 0.5) for b in (0, 0.5) for c in (0, 0.5)]
QNAME = {(0, 0, 0): "G", (0.5, 0.5, 0.5): "R"}


def qname(q):
    if q in QNAME:
        return QNAME[q]
    n = sum(x > 0 for x in q)
    return ("X" if n == 1 else "M") + "".join("xyz"[k] for k in range(3) if (q[k] > 0) == (n == 1))


def binder(a):
    """Binder cumulant of a 3-vector series: 0 for a centred Gaussian, 2/5 for fixed magnitude."""
    m2 = np.mean(np.sum(a * a, axis=-1))
    m4 = np.mean(np.sum(a * a, axis=-1) ** 2)
    return float(1.0 - 0.6 * m4 / m2 ** 2) if m2 > 0 else None


def classify(u):
    if u is None:
        return "none"
    return "ordered" if u >= U_ORDERED else "disordered" if u <= U_DISORDERED else "inconclusive"


def atomic_save(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    for _ in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    raise RuntimeError(f"could not replace {path}")


def run(args) -> int:
    unthrottle()
    import torch
    torch.set_num_threads(args.threads)
    import warnings
    warnings.filterwarnings("ignore")
    from ase import Atoms, units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from mace.calculators import mace_mp
    import importlib.metadata as md

    spec = RUNS[args.run]
    ref = load_cell()
    OUT.mkdir(parents=True, exist_ok=True)
    traj = OUT / f"run_{args.run}{'_smoke' if args.smoke else ''}.npz"
    n_eq = int(round(spec["eq_ps"] * 1000 / DT_FS))
    n_tot = n_eq + int(round(spec["prod_ps"] * 1000 / DT_FS))
    if args.smoke:
        n_eq, n_tot = 20, 40
    r_start = ref["r0"] + (ref["q_gamma"] * ref["u_g"] if spec["start"] == "gamma111" else 0.0)
    at = Atoms(list(ref["symbols"]), positions=r_start, cell=ref["cell"], pbc=True)
    at.calc = mace_mp(model="medium", device="cpu", default_dtype="float32")
    frames, epot, temp, steps, step0 = [], [], [], [], 0
    if traj.exists():
        old = np.load(traj, allow_pickle=True)
        frames, epot, temp, steps = (list(old[k]) for k in ("pos", "epot", "temp", "step"))
        step0 = int(old["last_step"])
        at.positions = old["last_pos"]
        at.set_momenta(old["last_mom"])
        print(f"[{args.run}] resume at step {step0}", flush=True)
    else:
        MaxwellBoltzmannDistribution(at, temperature_K=spec["T"], rng=np.random.default_rng(spec["seed"]))
        Stationary(at)
    dyn = Langevin(at, timestep=DT_FS * units.fs, temperature_K=spec["T"],
                   friction=FRICTION_PER_FS / units.fs, fixcm=True,
                   rng=np.random.default_rng(spec["seed"] + step0))
    meta = {"run": args.run, **spec, "dt_fs": DT_FS, "friction_per_fs": FRICTION_PER_FS,
            "frame_every_steps": EVERY, "n_eq_steps": n_eq, "n_total_steps": n_tot,
            "mace_torch": md.version("mace-torch"), "torch": torch.__version__, "dtype": "float32",
            "threads": args.threads, "pattern": PATTERN.relative_to(REPO).as_posix()}
    t0, step = time.time(), step0

    def save():
        atomic_save(traj, pos=np.array(frames, dtype=np.float32), epot=np.array(epot),
                    temp=np.array(temp), step=np.array(steps), last_step=step,
                    last_pos=at.get_positions(), last_mom=at.get_momenta(),
                    meta=json.dumps(meta))

    while step < n_tot:
        dyn.run(EVERY)
        step += EVERY
        frames.append(at.get_positions().astype(np.float32))
        epot.append(at.get_potential_energy())
        temp.append(at.get_temperature())
        steps.append(step)
        if step % 500 == 0 or step >= n_tot or args.smoke:
            save()
            rate = (time.time() - t0) / (step - step0)
            print(f"[{args.run}] step {step}/{n_tot}  T {np.mean(temp[-100:]):6.1f} K  "
                  f"{rate:.3f} s/step  eta {(n_tot - step) * rate / 60:.0f} min", flush=True)
    save()
    print(f"[{args.run}] DONE {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return 0


def analyze(_args) -> int:
    ref = load_cell()
    ti, cages, idx = cage_and_index(ref)
    inv = np.linalg.inv(ref["cell"])
    res = {"script": "scripts/md_order_check.py", "note": "tasks/md-order-check-2026-10-22.md",
           "rule": {"U_ordered_min": U_ORDERED, "U_disordered_max": U_DISORDERED}, "runs": {}}
    static = off_centring((ref["r0"] + ref["q_gamma"] * ref["u_g"])[None], ref, ti, cages)[0]
    res["static_gamma111_minimum"] = {"Ti_off_centring_ang": float(np.linalg.norm(static, axis=1).mean()),
                                      "A_G_norm_ang": float(np.linalg.norm(static.mean(axis=0)))}
    for name, spec in RUNS.items():
        f = OUT / f"run_{name}.npz"
        if not f.exists():
            res["runs"][name] = {"status": "missing"}
            continue
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        prod = d["step"] > meta["n_eq_steps"]
        pos = d["pos"][prod].astype(float)
        u = off_centring(pos, ref, ti, cages)
        per_q = {}
        for q in QS:
            ph = np.cos(2 * np.pi * (idx @ np.array(q)))
            a = np.einsum("i,fij->fj", ph, u) / len(ti)
            h = len(a) // 2
            per_q[qname(q)] = {"mean_sq_ang2": float(np.mean(np.sum(a * a, axis=1))),
                               "rms_ang": float(np.sqrt(np.mean(np.sum(a * a, axis=1)))),
                               "time_avg_vec_ang": a.mean(axis=0).tolist(),
                               "binder": binder(a), "binder_halves": [binder(a[:h]), binder(a[h:])]}
        top = max(per_q, key=lambda k: per_q[k]["mean_sq_ang2"])
        bt = per_q[top]
        halves = [classify(x) for x in bt["binder_halves"]]
        comp = u.reshape(-1)
        disp = pos - ref["r0"][None]
        fr = disp @ inv
        fr -= np.round(fr)
        disp = fr @ ref["cell"]
        disp -= disp.mean(axis=1, keepdims=True)
        proj = {}
        for k, pat in (("M", ref["u_m"]), ("G111", ref["u_g"])):
            qv = disp.reshape(len(disp), -1) @ pat.ravel() / (pat.ravel() @ pat.ravel())
            proj[k] = {"mean_ang": float(qv.mean()), "rms_ang": float(qv.std()),
                       "sscha_rms_at_cubic_ang": SSCHA_SIGMA[k],
                       "hist": np.histogram(qv, bins=41, range=(-0.15, 0.15))[0].tolist()}
        res["runs"][name] = {
            "status": "complete" if int(d["last_step"]) >= meta["n_total_steps"] else "partial",
            "spec": spec, "n_prod_frames": int(prod.sum()),
            "T_mean_prod_K": float(np.mean(d["temp"][prod])),
            "dominant_q": top, "binder_dominant": bt["binder"], "class": classify(bt["binder"]),
            "class_halves": halves, "stationary": halves[0] == halves[1],
            "per_q": per_q,
            "Ti_off_centring_mean_abs_ang": float(np.linalg.norm(u, axis=2).mean()),
            "Ti_component_excess_kurtosis": float(np.mean(comp ** 4) / np.mean(comp ** 2) ** 2 - 3.0),
            "Ti_component_hist": np.histogram(comp, bins=41, range=(-0.2, 0.2))[0].tolist(),
            "projections": proj, "meta": meta}
    r = res["runs"]
    ok = all(r.get(k, {}).get("status") == "complete" for k in RUNS)
    if not ok:
        verdict = "INCOMPLETE"
    elif r["C"]["class"] != "ordered":
        verdict = "NOT_INTERPRETABLE (positive control did not order)"
    elif not (r["A"]["stationary"] and r["B"]["stationary"]):
        verdict = "NOT_STATIONARY"
    elif r["A"]["class"] == r["B"]["class"] == "disordered":
        verdict = "SURFACE_DOES_NOT_ORDER"
    elif r["A"]["class"] == r["B"]["class"] == "ordered":
        verdict = "SURFACE_ORDERS_CLASSICALLY"
    elif r["A"]["class"] == "disordered" and r["B"]["class"] == "ordered":
        verdict = "HYSTERESIS"
    else:
        verdict = "INCONCLUSIVE"
    res["verdict"] = verdict
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for k, v in r.items():
        if v.get("status") == "missing":
            print(f"{k}: missing")
            continue
        print(f"{k} ({v['spec']['T']:.0f} K from {v['spec']['start']}, {v['status']}): T {v['T_mean_prod_K']:.1f} K, "
              f"dominant q {v['dominant_q']} U {v['binder_dominant']:.3f} -> {v['class']} (halves {v['class_halves']}); "
              f"Ti |u| {v['Ti_off_centring_mean_abs_ang']:.4f} A, kurtosis {v['Ti_component_excess_kurtosis']:+.2f}; "
              f"Q_M rms {v['projections']['M']['rms_ang']:.4f}, Q_G111 rms {v['projections']['G111']['rms_ang']:.4f}")
    print("static Gamma[111] minimum:", res["static_gamma111_minimum"])
    print("VERDICT:", verdict)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--run", choices=sorted(RUNS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="40 steps to run_<X>_smoke.npz (plumbing only)")
    a = ap.parse_args(argv)
    if a.cmd == "run" and not a.run:
        ap.error("run needs --run")
    return run(a) if a.cmd == "run" else analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
