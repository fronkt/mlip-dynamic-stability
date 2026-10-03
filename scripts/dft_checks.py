"""Three numerical checks on the C3a PBE curves, so screen error can be told from DFT-setup,
functional and lattice error. Stages live here, are registered on ``scripts/dft_reference.py``
(``python scripts/dft_reference.py conv-inputs ...``) and also run stand-alone
(``python scripts/dft_checks.py conv-inputs ...``). Reads results/revision/dft (never writes
there); writes results/revision/dft_checks (``--out-root``). Nothing here touches the ledger.

Why. 16 screen errors persist on PBE (KNbO3 600 K all five models; CsSnBr3 300/600 K; ORB-v2
BaTiO3 300 K; bcc Zr 900 K, which is not covered here). C3a evaluated PBE at each MLIP's OWN
relaxed lattice along that MLIP's eigenvector, with production settings chosen for cost. Three
things could still be behind a persistent error besides the screen itself, one stage each:

  conv-inputs   (job prefix ``cv_``)  Is the PBE well depth converged in k-points and cutoff?
  xc-inputs     (job prefix ``xs_``)  Does a different semilocal functional change the call?
  pbe-lattice   (job prefix ``pl_``)  Is it the lattice or the eigenvector? PBE lattice, PBE
                                      force constants, PBE eigenvector, E(Q) along it.
  analyze-checks                      all of the above -> results/revision/dft_checks/

ACCEPTANCE CRITERION for conv (fixed here, before any variant has run). Production settings are
numerically converged for the screen's decision on a path iff, for EACH of the three variants
(k-spacing 0.15 instead of 0.25 / ecutwfc x1.3 with ecutrho 8x / both),
  (i)  |depth_variant - depth_production| <= 5 % of depth_production, where depth =
       -min(E(Q)-E(0)) over the 10 sampled amplitudes, meV per modulated cell; and
  (ii) the single-mode screen re-solved on the variant's E(Q) (production fit and solver,
       ``_curve_metrics`` + ``_screen_call``) returns the SAME stable/unstable call as the
       production E(Q) at EVERY temperature in T_SCREEN (50, 100, 300, 600, 900 K).
All three deciding paths (BaTiO3, KNbO3, CsSnBr3) must pass for the persistent-error analysis to
stand on the production numbers. Q_min is reported, not tested (on a flat well it jumps by a grid
step). A path whose minimum sits on the scan edge (every CsSnBr3 PBE curve, Q = 0.45 A) is
flagged: its 'depth' is then the edge value, a lower bound, and the call is edge-sensitive.
Deciding path = the MACE-MP-0 path with the most deciding temperatures (ties: highest T), i.e.
the path that decides the persistent 300/600 K calls (CsSnBr3: the R-point path).

APPROXIMATION in xc (state it wherever the numbers are quoted). input_dft = 'pbesol' is applied
to the SAME SSSP 1.3 PBE-efficiency pseudopotentials (PAW/US generated with PBE). The pseudo
core/valence partition and any non-linear core correction were built for PBE, so this is
"PBEsol with PBE-generated pseudopotentials", NOT the SSSP PBEsol library and not an all-electron
PBEsol result. The error this introduces is believed small against the PBE-PBEsol difference
itself but is not measured here. Geometry is unchanged (each MLIP's relaxed lattice and its own
eigenvector), so xc isolates the functional, not the PBEsol equilibrium lattice.

pbe-lattice runs in THREE dependent phases (two were asked for; the force constants need the PBE
lattice, which only exists after the vc-relax, and the profile needs the force constants):
  --phase A  vc-relax of the 5-atom cubic cell in PBE (production k-spacing and cutoffs).
  --phase B  needs A. Finite-displacement supercells at the PBE lattice (phonopy displacements,
             0.01 A as the MLIP screen) written as pw.x inputs with production settings. The
             supercell is 2x2x2 for any q whose denominators divide 2 (R, M, X: the production FC
             cell, exact at all 8 commensurate q) and 1x1x1 for Gamma only.
  --phase C  needs B. PBE force constants -> frequencies on the commensurate q mesh -> frozen-mode
             structures at the production 10 amplitudes (Q = 0.45*linspace(0,1,10)**2 A, max
             atomic component of u = 1) -> pw.x inputs, for the softest PBE band at the deciding q
             (the MACE deciding path's q; jobs ``pl_*_prof_*``) and for the softest PBE mode of
             the whole mesh (``pl_*_profsoft_*``; not written where it is the same mode).
             ``--select`` picks either. This is the screen's own procedure with PBE as the
             calculator.
Caveat: inside a degenerate band (Gamma T1u, R25 tilt triplet) the eigenvector phonopy returns
is arbitrary in the span, as in production; the PBE well depth can depend on that direction by
tens of percent (MACE BaTiO3 Gamma: 16.8 vs 20.9 meV). ``n_degenerate`` is recorded for the
selected band; the R-point CsSnBr3 profile and every Gamma profile are triplet cases.
The FD supercells are written through the same pw.x writer as production (same pseudos, cutoffs,
smearing, k-spacing rule) rather than phonopy's template-based QE writer, so that no setting can
drift between the three stages; phonopy supplies the displacement set and the force constants.

Box order (QE env and SSSP as for dft_reference; <qe> and <sssp> as in run_rsc_revision.sh):

    python scripts/dft_reference.py conv-inputs --sssp-dir <sssp>
    bash scripts/box/qe_queue.sh --qe-prefix <qe> --pseudo-dir <sssp> \\
         --root results/revision/dft_checks/qe --filter '^cv_' -r 8
    python scripts/dft_reference.py xc-inputs --sssp-dir <sssp>
    ... --filter '^xs_'
    python scripts/dft_reference.py pbe-lattice --phase A --sssp-dir <sssp>   ... --filter '^pl_.*_vcrelax$'
    python scripts/dft_reference.py pbe-lattice --phase B --sssp-dir <sssp>   ... --filter '^pl_.*_fd'
    python scripts/dft_reference.py pbe-lattice --phase C --sssp-dir <sssp>   ... --filter '^pl_.*_prof'
    python scripts/dft_reference.py analyze-checks                            # local, after pull

A phase whose predecessor has not finished says so per system and exits 3 (re-run it later);
every stage is resumable and never rewrites an input under a finished pw.out without --force.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# when launched as scripts/dft_reference.py the running module is __main__; alias it so that
# `import dft_reference` here is that module and not a second copy of it
if "dft_reference" not in sys.modules and getattr(sys.modules.get("__main__"), "__file__", "") \
        and Path(sys.modules["__main__"].__file__).name == "dft_reference.py":
    sys.modules["dft_reference"] = sys.modules["__main__"]

import numpy as np                                                    # noqa: E402

import dft_reference as dr                                            # noqa: E402

SCRIPT = "scripts/dft_checks.py"
CHECK_SYSTEMS = ("batio3_cubic", "knbo3_cubic", "cssnbr3_cubic")
CHECKS_OUT_DEFAULT = dr.REPO / "results" / "revision" / "dft_checks"
CV_VARIANTS = ("k", "e", "ke")             # k-spacing / cutoff / both
ACCEPT_DEPTH_REL = 0.05
SELECT_MODEL = "mace_mp0"
FALLBACK_KSPACING = 0.25                   # production insulator spacing, used only if no
FALLBACK_DEGAUSS = 0.005                   # production job.json can be read
C0_FALLBACK = 2.67                         # core-s per k-point (5-atom equiv.), measured median
DEGEN_TOL_THZ = 0.05
PBE_XC = (3, 4)                            # igcx, igcc printed by pw.x for PBE
PBESOL_XC = (10, 8)

NOT_READY = 3


class Deferred(Exception):
    """A phase whose predecessor has not finished."""


# ------------------------------------------------------------------ provenance ----

def _prov(args, stage):
    p = dr.provenance(args, stage)
    p["script"] = SCRIPT
    p["script_sha256"] = dr._sha256(Path(__file__))
    p["dft_reference_sha256"] = dr._sha256(Path(dr.__file__))
    return p


# ------------------------------------------------------------- pw.x input writer ----

def pw_text(atoms, job, system, sssp, *, kspacing, degauss, ecut_scale=1.0, input_dft=None,
            calculation="scf"):
    """pw.x input in the format of dft_reference.pw_input (byte-identical to it for
    ecut_scale = 1, input_dft = None, calculation = 'scf', apart from the job comment line,
    which conv-inputs checks against the production pw.in), plus the three things the checks
    vary: a cutoff scale (ecutrho = max(8*ecutwfc, scaled SSSP value)), input_dft and vc-relax."""
    from ase.data import atomic_masses, atomic_numbers
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    missing = [s for s in species if s not in sssp["table"]]
    if missing:
        raise SystemExit(f"SSSP json {sssp['json']} has no entry for {missing}")
    ecutwfc = max(sssp["table"][s]["ecutwfc"] for s in species)
    ecutrho = max(sssp["table"][s]["ecutrho"] for s in species)
    if ecut_scale != 1.0:
        ecutwfc = ecutwfc * ecut_scale
        ecutrho = max(8.0 * ecutwfc, ecutrho * ecut_scale)
    mesh = dr._kmesh(atoms.cell.array, kspacing)
    nat = len(atoms)
    conv = 1e-10 * nat
    vc = calculation == "vc-relax"
    L = ["&CONTROL",
         f"  ! job {job}, written by {SCRIPT}",
         "  ! pseudo_dir is taken from $ESPRESSO_PSEUDO (exported by scripts/box/qe_queue.sh)",
         f"  calculation = '{calculation}'", "  prefix = 'pwscf'", "  outdir = './scratch'",
         f"  disk_io = '{'low' if vc else 'none'}'", "  tprnfor = .true.",
         f"  tstress = {'.true.' if vc else '.false.'}", "  verbosity = 'low'"]
    if vc:
        L += ["  etot_conv_thr = 1.0d-6", "  forc_conv_thr = 1.0d-4", "  nstep = 150"]
    L += ["/", "&SYSTEM", "  ibrav = 0", f"  nat = {nat}", f"  ntyp = {len(species)}",
          f"  ecutwfc = {ecutwfc:.1f}", f"  ecutrho = {ecutrho:.1f}"]
    if input_dft:
        L += [f"  input_dft = '{input_dft}'"]
    L += ["  occupations = 'smearing'", "  smearing = 'mv'", f"  degauss = {degauss}", "/",
          "&ELECTRONS", f"  conv_thr = {dr._fortran(conv)}", "  mixing_beta = 0.4",
          "  electron_maxstep = 200", "/"]
    if vc:
        L += ["&IONS", "  ion_dynamics = 'bfgs'", "/",
              "&CELL", "  cell_dynamics = 'bfgs'", "  press = 0.0", "  press_conv_thr = 0.1", "/"]
    L += ["ATOMIC_SPECIES"]
    L += [f"{s} {atomic_masses[atomic_numbers[s]]:.4f} {sssp['table'][s]['filename']}"
          for s in species]
    L += ["CELL_PARAMETERS angstrom"]
    L += ["  " + " ".join(f"{x:.10f}" for x in v) for v in atoms.cell.array]
    L += ["ATOMIC_POSITIONS angstrom"]
    L += [f"{s} " + " ".join(f"{x:.10f}" for x in p)
          for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
    L += ["K_POINTS automatic", f"{mesh[0]} {mesh[1]} {mesh[2]} 0 0 0", ""]
    functional = ("PBE (from the SSSP PBE set)" if not input_dft else
                  f"{input_dft} via input_dft on SSSP 1.3 PBE-efficiency pseudopotentials "
                  "(PBE-generated; not the SSSP PBEsol set)")
    settings = {"is_metal": dr.is_metal(system), "smearing": "mv", "degauss_Ry": degauss,
                "kspacing_inv_A": kspacing, "kspacing_convention": dr.KSPACING_CONVENTION,
                "kmesh": mesh, "nk_full": int(np.prod(mesh)), "nk_tr_upper": dr._nk_tr_upper(mesh),
                "nk_irr_spglib": dr._nk_irr(atoms, mesh), "ecutwfc_Ry": ecutwfc,
                "ecutrho_Ry": ecutrho, "ecut_scale": ecut_scale, "conv_thr_Ry": conv,
                "mixing_beta": 0.4, "calculation": calculation,
                "disk_io": "low" if vc else "none", "input_dft": input_dft,
                "functional": functional,
                "pseudos": {s: sssp["table"][s] for s in species},
                "sssp_json": sssp["json"], "sssp_sha256": sssp["sha256"]}
    return "\n".join(L), settings


class CheckWriter:
    """One directory per pw.x job under <out>/qe/<job>/ (pw.in + job.json), the layout
    scripts/box/qe_queue.sh runs. Unlike dft_reference.JobWriter nothing is aliased by structure:
    the same structure is run under several settings here, so the structure alone is not a key.
    A pw.in is never changed under a finished pw.out unless --force (the old output is then
    renamed aside, so the queue and the analysis cannot read it for the new input).

    job.json carries the structure itself, so the analysis can check a pw.out against what was
    asked without the geometry files."""

    def __init__(self, out, prov, force=False):
        self.qe = Path(out) / "qe"
        self.prov, self.force = prov, force
        self.n_new = self.n_same = self.n_refused = 0
        self.meta = {}

    def would_refuse(self, job, text):
        d = self.qe / job
        return bool(not self.force and (d / "pw.in").exists() and (d / "pw.out").exists()
                    and (d / "pw.in").read_text(encoding="utf-8") != text)

    def add(self, job, atoms, system, text, settings, member, study):
        d = self.qe / job
        pw = d / "pw.in"
        if pw.exists():
            old = pw.read_text(encoding="utf-8")
            if old == text:
                self.n_same += 1
                if (d / "job.json").exists():
                    self.meta[job] = dr._load(d / "job.json")
                    return "same"
            elif (d / "pw.out").exists() and not self.force:
                print(f"[checks] REFUSED {job}: pw.in would change under an existing pw.out "
                      f"(delete the job dir or pass --force)")
                self.n_refused += 1
                return "refused"
            else:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                for f in ("pw.out", "pw.err", "pw.in.ran"):
                    if (d / f).exists():
                        os.replace(d / f, d / f"{f}.superseded-{stamp}")
        d.mkdir(parents=True, exist_ok=True)
        with open(pw, "w", encoding="utf-8", newline="\n") as fh:      # LF: pw.x runs on Linux
            fh.write(text)
        meta = {"job": job, "study": study, "system": system, "n_atoms": len(atoms),
                "structure_sha": dr._structure_sha(atoms),
                "structure": {"symbols": atoms.get_chemical_symbols(),
                              "cell": atoms.cell.array.tolist(),
                              "positions": atoms.get_positions().tolist()},
                "members": [member], "qe": settings,
                "pw_in_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "provenance": self.prov}
        dr._dump(d / "job.json", meta)
        self.meta[job] = meta
        self.n_new += 1
        return "new"

    def report(self, label):
        print(f"[{label}] jobs: {self.n_new} written, {self.n_same} unchanged, "
              f"{self.n_refused} refused")


# --------------------------------------------------- production artefacts (read only) ----

def _prod_index(dft):
    """(stem, frame) -> [(job.json, member)] over the production C3a jobs."""
    idx = {}
    for jj in sorted((Path(dft) / "qe").glob("*/job.json")):
        m = dr._load(jj)
        for mem in m.get("members", []):
            if mem.get("study") == "c3a":
                idx.setdefault((mem["stem"], int(mem["frame"])), []).append((m, mem))
    return idx


def _prod_costs(dft):
    """Measured production cost from results/revision/dft/qe_jobs.csv: core-h per job, and c0
    (core-s per irreducible k-point for a 5-atom equivalent, the unit of dft_reference.cost_core_h)
    as the median over finished jobs of the same system AND cell size. c0 differs 5-fold between
    the oxides (~2.6) and CsSnBr3 (~15-19: Cs/Sn/Br semicore electrons) and grows with the cell
    (BaTiO3 2.5 at 5 atoms, 3.6 at 40), so one global c0 would misprice every CsSnBr3 and every
    40-atom estimate."""
    p = Path(dft) / "qe_jobs.csv"
    if not p.exists():
        return {"core_h": {}, "c0": {}}
    core_h, per = {}, {}
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if not r.get("core_h"):
                continue
            core_h[r["job"]] = float(r["core_h"])
            if r.get("status") == "ok" and r.get("nk_irr") and r.get("n_atoms"):
                rest = r["job"].split("_", 1)[1] if "_" in r["job"] else r["job"]
                for s in CHECK_SYSTEMS:
                    if rest.startswith(s + "_"):
                        nat = int(r["n_atoms"])
                        per.setdefault(s, {}).setdefault(nat, []).append(
                            float(r["core_h"]) / dr.cost_core_h(nat, int(r["nk_irr"]), 1.0))
    return {"core_h": core_h,
            "c0": {s: {n: float(np.median(v)) for n, v in d.items()} for s, d in per.items()}}


def _c0(ctx, system, nat=10):
    """c0 of the production size nearest (in log n_atoms) to nat, for this system."""
    d = ctx["c0"].get(system)
    if not d:
        return C0_FALLBACK
    return d[min(d, key=lambda n: abs(math.log(n / nat)))]


def _job_done(dft, job):
    po = Path(dft) / "qe" / job / "pw.out"
    if not po.exists():
        return False
    t = po.read_text(encoding="utf-8", errors="replace")
    return "JOB DONE" in t and "convergence has been achieved" in t \
        and "convergence NOT achieved" not in t


def _patterns(dft, system=None):
    out = []
    for p in sorted((Path(dft) / "geom" / "patterns").glob("*.json")):
        d = dr._load(p)
        if system is None or d["system"] == system:
            out.append(d)
    return out


def _model_deciding_path(dft, system, model):
    """The model's deciding path that decides the most temperatures (ties: the highest T)."""
    cand = [d for d in _patterns(dft, system)
            if d["role"] == "decide" and d["path_model"] == model]
    if not cand:
        raise SystemExit(f"no deciding path for {system}/{model} under {dft}/geom/patterns")
    key = lambda d: (len(d.get("decide_T") or []), max(d.get("decide_T") or [0.0]),
                     d["depth_meV"])
    return max(cand, key=key), cand


def _load_path(dft, stem, ctx, fallback):
    """The path's frames in the cell production ran them in, plus the production jobs.

    The cell and atom order are taken from the production job.json (reduction lattice and
    representatives), so a variant differs from production ONLY in the setting being varied.
    Without production jobs the reduction is re-derived (flagged in the result)."""
    dft = Path(dft)
    pat = dr._load(dft / "geom" / "patterns" / f"{stem}.json")
    frames = dr._read_frames(dft / pat["geom_file"])
    system = pat["system"]
    out_frames, prod, src = [], [], "production job.json"
    for i, a in enumerate(frames):
        chosen = None
        for m, mem in ctx["idx"].get((stem, i), []):
            red = mem.get("reduction")
            if red:
                r = dr._reduce_frame(a, np.array(red["lattice"]), red["reps"])
                atoms = r[0] if r else None
            else:
                from ase import Atoms
                atoms = Atoms(symbols=a.get_chemical_symbols(), positions=a.get_positions(),
                              cell=a.get_cell(), pbc=True)
            if atoms is not None and dr._structure_sha(atoms) == m["structure_sha"]:
                chosen = (m, mem, atoms)
                break
        if chosen is None:
            out_frames, prod, src = None, [], "re-derived"
            break
        out_frames.append(chosen[2])
        prod.append({"job": chosen[0]["job"], "qe": chosen[0]["qe"], "reduction": chosen[1].get(
            "reduction")})
    n_img = 1
    if out_frames is None:
        ns = argparse.Namespace(kspacing_insulator=fallback["kspacing"],
                                kspacing_metal=fallback["kspacing"], no_reduce_cell=False)
        lat, reps, _ = dr._pick_cell(frames[-1], system, ns)
        out_frames = []
        for a in frames:
            if lat is None:
                from ase import Atoms
                out_frames.append(Atoms(symbols=a.get_chemical_symbols(), positions=a.get_positions(),
                                        cell=a.get_cell(), pbc=True))
            else:
                r = dr._reduce_frame(a, lat, reps)
                if r is None:
                    raise SystemExit(f"{stem}: frame not periodic in the re-derived cell")
                out_frames.append(r[0])
        n_img = len(frames[0]) // len(out_frames[0])
    elif prod and prod[0]["reduction"]:
        n_img = int(prod[0]["reduction"]["n_images"])
    return {"stem": stem, "system": system, "pat": pat, "frames": out_frames,
            "Q": [float(a.info["Q"]) for a in frames], "n_images": n_img,
            "prod": prod, "reduction_source": src,
            "prod_done": bool(prod) and all(_job_done(dft, p["job"]) for p in prod)}


def _prod_settings(P, fallback):
    if P["prod"]:
        q = P["prod"][0]["qe"]
        return float(q["kspacing_inv_A"]), float(q["degauss_Ry"])
    return fallback["kspacing"], fallback["degauss"]


def _estimate(core_h, nk_old, nk_new, ecut_scale, nat, c0):
    """Cost of a variant of a production job: its MEASURED core-h, times the ratio of irreducible
    k-points, times ecut_scale^2 (plane waves and FFT grid both grow ~ecut^1.5; the extra half
    power is a margin, so this errs high). Modelled with the system's c0 where the production
    job has no measurement."""
    base = core_h if core_h else dr.cost_core_h(nat, nk_old or 1, c0)
    return float(base * (nk_new / max(nk_old or 1, 1)) * ecut_scale ** 2)


def _fallback(args):
    return {"kspacing": args.kspacing_insulator or FALLBACK_KSPACING,
            "degauss": args.degauss_insulator or FALLBACK_DEGAUSS}


def _add_path(w, P, jobname, member_of, sssp, kspacing, degauss, ecut_scale, input_dft, study,
              ctx):
    """Write one path's frames under one setting set; refuses the whole path if any frame has a
    finished pw.out under different settings (one E(Q) must not mix settings)."""
    rows = []
    for i, a in enumerate(P["frames"]):
        job = jobname(i)
        text, st = pw_text(a, job, P["system"], sssp, kspacing=kspacing, degauss=degauss,
                           ecut_scale=ecut_scale, input_dft=input_dft)
        rows.append((i, job, a, text, st))
    blocked = [r[1] for r in rows if w.would_refuse(r[1], r[3])]
    if blocked:
        print(f"[checks] REFUSED path {P['stem']}: {len(blocked)} frame(s) have a finished "
              f"pw.out under different settings; --force redoes it")
        w.n_refused += len(rows)
        return None
    est, nk = 0.0, []
    for i, job, a, text, st in rows:
        w.add(job, a, P["system"], text, st, member_of(i), study)
        pj = P["prod"][i] if P["prod"] else None
        nk_old = pj["qe"].get("nk_irr_spglib") if pj else None
        est += _estimate(ctx["core_h"].get(pj["job"]) if pj else None, nk_old,
                         st["nk_irr_spglib"] or st["nk_tr_upper"], ecut_scale, len(a),
                         _c0(ctx, P["system"], len(a)))
        nk.append(st["nk_irr_spglib"])
    st0 = rows[0][4]
    return {"jobs": len(rows), "atoms": len(rows[0][2]), "kmesh": st0["kmesh"],
            "nk_irr": nk, "ecutwfc_Ry": st0["ecutwfc_Ry"], "ecutrho_Ry": st0["ecutrho_Ry"],
            "est_core_h": est}


# ------------------------------------------------------------------ conv-inputs ----

def _variants(args, kprod):
    return {"k": {"kspacing": args.kspacing_fine, "ecut_scale": 1.0,
                  "what": f"k-spacing {args.kspacing_fine:g} (production {kprod:g})"},
            "e": {"kspacing": kprod, "ecut_scale": args.ecut_scale,
                  "what": f"ecutwfc x{args.ecut_scale:g}, ecutrho 8x"},
            "ke": {"kspacing": args.kspacing_fine, "ecut_scale": args.ecut_scale,
                   "what": "both"}}


def _repro_check(P, sssp, kprod, degauss, dft):
    """Production pw.in regenerated by this writer vs the one on disk (job comment excluded):
    proves the cell, atom order, k-mesh and cutoffs the variants start from ARE production's."""
    n = same = 0
    for i, a in enumerate(P["frames"]):
        if not P["prod"]:
            break
        pj = P["prod"][i]["job"]
        p = Path(dft) / "qe" / pj / "pw.in"
        if not p.exists():
            continue
        text, _ = pw_text(a, pj, P["system"], sssp, kspacing=kprod, degauss=degauss)
        old = p.read_text(encoding="utf-8").split("\n")
        new = text.split("\n")
        n += 1
        same += int(len(old) == len(new) and all(x == y for k, (x, y) in enumerate(zip(old, new))
                                                  if k != 1))
    return {"checked": n, "identical": same}


def stage_conv_inputs(args):
    dft, out = Path(args.dft_root), Path(args.out_root)
    sssp = dr.load_sssp(args)
    prov = _prov(args, "conv-inputs")
    ctx = {"idx": _prod_index(dft), **_prod_costs(dft)}
    w = CheckWriter(out, prov, args.force)
    fb = _fallback(args)
    sel, tot, n_prob = [], {v: 0.0 for v in args.variants}, 0
    for system in args.systems:
        pat, cand = _model_deciding_path(dft, system, args.model)
        P = _load_path(dft, pat["stem"], ctx, fb)
        if P["reduction_source"] != "production job.json":
            print(f"[conv] NOTE {pat['stem']}: production jobs not found, cell re-derived")
        if not P["prod_done"] and not args.no_require_pbe:
            raise SystemExit(f"{pat['stem']}: production PBE jobs are not all finished under "
                             f"{dft}/qe (--no-require-pbe to write the inputs anyway)")
        kprod, degauss = _prod_settings(P, fb)
        repro = _repro_check(P, sssp, kprod, degauss, dft)
        if repro["checked"] != repro["identical"]:
            n_prob += 1
            print(f"[conv] !!! {pat['stem']}: regenerated PRODUCTION input differs from the "
                  f"production pw.in in {repro['checked'] - repro['identical']} of "
                  f"{repro['checked']} frames; variants would not differ from production by "
                  f"the setting alone", flush=True)
        entry = {"system": system, "stem": pat["stem"], "role": pat["role"], "q": pat["q"],
                 "decide_T": pat.get("decide_T"), "n_atoms_supercell": pat["n_atoms"],
                 "n_atoms_run": len(P["frames"][0]), "n_images": P["n_images"],
                 "production_job_example": P["prod"][0]["job"] if P["prod"] else None,
                 "reduction_source": P["reduction_source"], "repro_check": repro,
                 "production_kspacing": kprod, "candidates": [
                     {"stem": c["stem"], "decide_T": c.get("decide_T"), "depth_meV": c["depth_meV"]}
                     for c in cand], "variants": {}}
        for v in args.variants:
            spec = _variants(args, kprod)[v]
            r = _add_path(w, P, lambda i, v=v, s=pat["stem"]: f"cv_{v}_{s}_i{i:02d}",
                          lambda i, v=v, P=P: {"part": "conv", "variant": v, "stem": P["stem"],
                                               "system": P["system"], "frame": i,
                                               "Q": P["Q"][i], "n_images": P["n_images"],
                                               "role": P["pat"]["role"],
                                               "path_model": P["pat"]["path_model"]},
                          sssp, spec["kspacing"], degauss, spec["ecut_scale"], None, "conv", ctx)
            if r:
                entry["variants"][v] = dict(r, what=spec["what"])
                tot[v] += r["est_core_h"]
        sel.append(entry)
    w.report("conv-inputs")
    print(f"{'system':13s} {'var':3s} {'jobs':>4s} {'atoms':>5s} {'kmesh':>9s} {'ecut':>10s} "
          f"{'nk_irr(min..max)':>17s} {'est core-h':>10s}")
    for e in sel:
        for v, r in e["variants"].items():
            print(f"{e['system']:13s} {v:3s} {r['jobs']:4d} {r['atoms']:5d} "
                  f"{'x'.join(map(str, r['kmesh'])):>9s} "
                  f"{r['ecutwfc_Ry']:4.0f}/{r['ecutrho_Ry']:<5.0f} "
                  f"{min(r['nk_irr']):7d}..{max(r['nk_irr']):<7d} {r['est_core_h']:10.1f}")
    print(f"[conv-inputs] estimated core-h (production core-h of the same frames x irreducible-k "
          f"ratio x ecut_scale^2): {', '.join(f'{v} {t:.0f}' for v, t in tot.items())}; total "
          f"{sum(tot.values()):.0f}")
    if w.n_new + w.n_same:            # a run that refused everything must not overwrite the record
        dr._dump(out / "conv_selection.json", {
            "provenance": prov, "acceptance": {"depth_rel": ACCEPT_DEPTH_REL,
                                               "calls": "identical at every T in T_SCREEN",
                                               "T_SCREEN": list(dr.T_SCREEN)},
            "paths": sel, "est_core_h": tot})
    return 4 if n_prob else 0


# --------------------------------------------------------------------- xc-inputs ----

def stage_xc_inputs(args):
    dft, out = Path(args.dft_root), Path(args.out_root)
    sssp = dr.load_sssp(args)
    prov = _prov(args, "xc-inputs")
    ctx = {"idx": _prod_index(dft), **_prod_costs(dft)}
    w = CheckWriter(out, prov, args.force)
    fb = _fallback(args)
    paths, skipped, est = [], [], 0.0
    for system in args.systems:
        for pat in _patterns(dft, system):
            if pat["role"] not in ("decide", "ref") or not (dft / pat["geom_file"]).exists():
                continue
            P = _load_path(dft, pat["stem"], ctx, fb)
            if not P["prod_done"] and not args.no_require_pbe:
                skipped.append({"stem": pat["stem"], "why": "production PBE not finished"})
                continue
            kprod, degauss = _prod_settings(P, fb)
            r = _add_path(w, P, lambda i, s=pat["stem"]: f"xs_{s}_i{i:02d}",
                          lambda i, P=P: {"part": "xc", "stem": P["stem"], "system": P["system"],
                                          "frame": i, "Q": P["Q"][i], "n_images": P["n_images"],
                                          "role": P["pat"]["role"],
                                          "path_model": P["pat"]["path_model"],
                                          "decide_T": P["pat"].get("decide_T")},
                          sssp, kprod, degauss, 1.0, args.functional, "xc", ctx)
            if r:
                paths.append(dict(r, system=system, stem=pat["stem"], role=pat["role"],
                                  path_model=pat["path_model"]))
                est += r["est_core_h"]
    w.report("xc-inputs")
    by = {}
    for p in paths:
        by.setdefault(p["system"], []).append(p)
    for s, ps in by.items():
        print(f"  {s:13s} {len(ps):2d} paths x {dr.N_PTS} Q = {sum(p['jobs'] for p in ps):3d} jobs, "
              f"est {sum(p['est_core_h'] for p in ps):6.1f} core-h")
    for s in skipped:
        print(f"  skipped {s['stem']}: {s['why']}")
    print(f"[xc-inputs] {len(paths)} paths, {sum(p['jobs'] for p in paths)} jobs, est "
          f"{est:.0f} core-h (= the production core-h of the same frames; PBEsol costs the same "
          f"as PBE per SCF to first order)")
    if w.n_new + w.n_same:
        dr._dump(out / "xc_selection.json", {
            "provenance": prov, "functional": args.functional,
            "approximation": "input_dft on SSSP 1.3 PBE-efficiency pseudopotentials; not the "
                             "SSSP PBEsol library", "paths": paths, "skipped": skipped,
            "est_core_h": est})
    return 0


# ----------------------------------------------------------------- pbe-lattice ----

def _fd_supercell(q, override=None):
    if override:
        return tuple(int(x) for x in override)
    den = [Fraction(float(x)).limit_denominator(12).denominator for x in q]
    n = math.lcm(*den)
    return (1, 1, 1) if n == 1 else (n, n, n)


def _pl_dir(out, system):
    return Path(out) / "pbe_lattice" / system


def _prim(system):
    prim = dr.build_atoms(dr.get_spec(system))
    c = prim.cell.array
    if len(prim) != 5 or np.abs(c - np.eye(3) * c[0, 0]).max() > 1e-9:
        raise SystemExit(f"{system}: expected the 5-atom simple-cubic prototype")
    return prim


def parse_vcrelax(path):
    """What a pw.x vc-relax output says about the relaxed cell. Written against QE 7.x output
    ('bfgs converged in', 'Begin final coordinates', 'new unit-cell volume', CELL_PARAMETERS in the
    input's units, a final scf at the new cell whose P is the Pulay-contaminated residual)."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    r = {"job_done": "JOB DONE" in txt,
         "bfgs_converged": "bfgs converged" in txt and "End of BFGS Geometry Optimization" in txt,
         "bfgs_failed": ("bfgs failed" in txt) or ("maximum number of steps has been reached" in txt)}
    r.update(_pw_meta(txt))
    m = re.search(r"bfgs converged in\s+(\d+) scf cycles and\s+(\d+) bfgs steps", txt)
    r["n_scf_cycles"], r["n_bfgs_steps"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    r["converged"] = bool("convergence has been achieved" in txt
                          and txt.rfind("convergence NOT achieved") < txt.rfind(
                              "convergence has been achieved"))
    e = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", txt, flags=re.M)
    r["energy_Ry"] = float(e[-1]) if e else None
    p = re.findall(r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?[\d.]+)", txt)
    r["final_scf_pressure_kbar"] = float(p[-1]) if p else None
    i = txt.rfind("Begin final coordinates")
    if i >= 0:
        blk = txt[i: txt.find("End final coordinates", i) if "End final coordinates" in txt[i:]
                  else len(txt)]
        mv = re.search(r"new unit-cell volume\s*=\s*(-?[\d.]+)\s*a\.u\.\^3\s*\(\s*(-?[\d.]+)\s*Ang\^3",
                       blk)
        if mv:
            r["volume_A3"] = float(mv.group(2))
        mc = re.search(r"CELL_PARAMETERS\s*\(([^)]*)\)\s*\n"
                       r"\s*(\S+)\s+(\S+)\s+(\S+)\s*\n\s*(\S+)\s+(\S+)\s+(\S+)\s*\n"
                       r"\s*(\S+)\s+(\S+)\s+(\S+)", blk)
        if mc:
            unit = mc.group(1)
            cell = np.array([float(x) for x in mc.groups()[1:]]).reshape(3, 3)
            bohr = 0.529177210903
            ma = re.search(r"alat\s*=\s*([\d.]+)", unit)
            if ma:
                cell = cell * float(ma.group(1)) * bohr
            elif "bohr" in unit.lower():
                cell = cell * bohr
            r["cell_A"] = cell.tolist()
    return r


def _pw_meta(txt):
    r = {}
    m = re.search(r"Program PWSCF\s+v\.?\s*(\S+)", txt)
    r["qe_version"] = m.group(1) if m else None
    m = re.search(r"Number of MPI processes:\s+(\d+)", txt)
    r["nproc"] = int(m.group(1)) if m else None
    m = re.findall(r"number of k points=\s*(\d+)", txt)
    r["nk_irr"] = int(m[-1]) if m else None
    m = re.search(r"PWSCF\s*:\s*(.+?)CPU\s+(.+?)WALL", txt)
    r["wall_s"] = dr._qe_seconds(m.group(2)) if m else None
    m = re.search(r"Exchange-correlation\s*=.*\n\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", txt)
    r["xc"] = (int(m.group(3)), int(m.group(4))) if m else None
    return r


def _expected_xc(meta):
    return PBESOL_XC if (meta["qe"].get("input_dft") or "").lower() == "pbesol" else PBE_XC


def _pbe_prim_settings(dft, system, args):
    """Production k-spacing and smearing for this system, read from any production job."""
    for jj in sorted((Path(dft) / "qe").glob(f"a_{system}_*/job.json")):
        q = dr._load(jj)["qe"]
        return float(q["kspacing_inv_A"]), float(q["degauss_Ry"]), "production job.json"
    fb = _fallback(args)
    return fb["kspacing"], fb["degauss"], "fallback (no production job.json)"


def _read_vc(out, system):
    po = Path(out) / "qe" / f"pl_{system}_vcrelax" / "pw.out"
    if not po.exists():
        raise Deferred(f"{system}: vc-relax has not run (pw.out missing)")
    r = parse_vcrelax(po)
    if not r["job_done"] or not r["bfgs_converged"] or "cell_A" not in r:
        raise Deferred(f"{system}: vc-relax not finished or not converged "
                       f"(job_done {r['job_done']}, bfgs_converged {r['bfgs_converged']})")
    cell = np.array(r["cell_A"])
    a = float(np.mean(np.diag(cell)))
    r["a_A"] = a
    r["cubic_deviation_A"] = float(np.abs(cell - np.eye(3) * a).max())
    return r


def _unit_cell(system, a):
    from ase import Atoms
    prim = _prim(system)
    return Atoms(symbols=prim.get_chemical_symbols(), scaled_positions=prim.get_scaled_positions(),
                 cell=np.eye(3) * a, pbc=True)


def _phonopy(unit, sc):
    from phonopy import Phonopy
    from mlip_dynstab.harmonic import _ase_to_phonopy_atoms
    return Phonopy(_ase_to_phonopy_atoms(unit), supercell_matrix=np.diag(sc),
                   primitive_matrix="auto")


def _phase_a(args, sssp, prov, w, dft, ctx):
    rows = []
    for system in args.systems:
        kspacing, degauss, src = _pbe_prim_settings(dft, system, args)
        prim = _prim(system)
        job = f"pl_{system}_vcrelax"
        text, st = pw_text(prim, job, system, sssp, kspacing=kspacing, degauss=degauss,
                           calculation="vc-relax")
        if w.would_refuse(job, text):
            print(f"[pbe-lattice A] REFUSED {job}: finished output under another input "
                  f"(--force redoes it)")
            w.n_refused += 1
            continue
        w.add(job, prim, system, text, st, {"part": "vcrelax", "system": system}, "pbe-lattice")
        # ~10 SCF-equivalents: cubic symmetry keeps the relaxation to a few BFGS steps
        est = 10 * dr.cost_core_h(5, st["nk_irr_spglib"] or st["nk_tr_upper"],
                                  _c0(ctx, system, 5))
        rows.append((system, st["kmesh"], st["nk_irr_spglib"], est, src))
    for s, m, nk, est, src in rows:
        print(f"  {s:13s} vc-relax 5 atoms k={'x'.join(map(str, m))} nk_irr={nk} "
              f"est ~{est:.1f} core-h (settings: {src}; c0 {_c0(ctx, s, 5):.1f})")
    print(f"[pbe-lattice A] {len(rows)} jobs, est ~{sum(r[3] for r in rows):.0f} core-h "
          f"(~10 SCF-equivalents each at the system's measured c0)")
    return 0


def _phase_b(args, sssp, prov, w, dft, out, ctx):
    status, tot, n = 0, 0.0, 0
    for system in args.systems:
        try:
            vc = _read_vc(out, system)
        except Deferred as exc:
            print(f"[pbe-lattice B] DEFERRED {exc}", flush=True)
            status = NOT_READY
            continue
        kspacing, degauss, src = _pbe_prim_settings(dft, system, args)
        pat, _ = _model_deciding_path(dft, system, args.model)
        q = pat["q"]
        sc = _fd_supercell(q, args.fd_supercell)
        unit = _unit_cell(system, vc["a_A"])
        ph = _phonopy(unit, sc)
        ph.generate_displacements(distance=args.fd_disp,
                                  is_plusminus=True if args.fd_plusminus else "auto")
        from ase import Atoms
        names, est, built = [], 0.0, []
        for k, cell in enumerate(ph.supercells_with_displacements, start=1):
            a = Atoms(symbols=cell.symbols, scaled_positions=cell.scaled_positions,
                      cell=cell.cell, pbc=True)
            job = f"pl_{system}_fd{k:02d}"
            text, st = pw_text(a, job, system, sssp, kspacing=kspacing, degauss=degauss)
            built.append((k, job, a, text, st))
        refused = [b[1] for b in built if w.would_refuse(b[1], b[3])]
        if refused:
            # one set of force constants must come from one lattice and one setting set
            print(f"[pbe-lattice B] REFUSED {system}: {len(refused)} displaced cell(s) have a "
                  f"finished output under another input (the PBE lattice changed?); the whole "
                  f"set is left as it was and fd_meta.json is not rewritten. --force redoes it.")
            w.n_refused += len(built)
            status = max(status, 4)
            continue
        for k, job, a, text, st in built:
            w.add(job, a, system, text, st, {"part": "fd", "system": system, "index": k},
                  "pbe-lattice")
            names.append(job)
            est += dr.cost_core_h(len(a), st["nk_irr_spglib"] or st["nk_tr_upper"],
                                  _c0(ctx, system, len(a)))
        dr._dump(_pl_dir(out, system) / "fd_meta.json", {
            "provenance": prov, "system": system, "a_pbe_A": vc["a_A"],
            "vcrelax": dict(vc), "cell": unit.cell.array.tolist(), "symbols": unit.get_chemical_symbols(),
            "scaled_positions": unit.get_scaled_positions().tolist(),
            "supercell": list(sc), "disp_A": args.fd_disp, "plusminus": bool(args.fd_plusminus),
            "n_displacements": len(names), "jobs": names, "q_decide": q,
            "q_decide_model": args.model, "q_decide_stem": pat["stem"],
            "kspacing": kspacing, "degauss": degauss, "settings_source": src,
            "n_atoms_supercell": len(unit) * int(np.prod(sc)), "est_core_h": est})
        nat_sc = len(unit) * int(np.prod(sc))
        print(f"  {system:13s} a_PBE {vc['a_A']:.5f} A, supercell {'x'.join(map(str, sc))} "
              f"({nat_sc} atoms), {len(names)} displaced cells, est ~{est:.0f} core-h "
              f"(model: spglib-irreducible k x measured c0 {_c0(ctx, system, nat_sc):.1f} "
              f"x (nat/5)^2.2)")
        tot += est
        n += len(names)
    print(f"[pbe-lattice B] {n} jobs, est ~{tot:.0f} core-h")
    return status


def _pbe_phonons(out, system, jobs):
    """PBE force constants at the PBE lattice from the finished FD jobs. Used by phase C and by
    analyze-checks, so both read the same force constants. Raises Deferred if any is missing."""
    meta = dr._load(_pl_dir(out, system) / "fd_meta.json")
    unit = _unit_cell(system, meta["a_pbe_A"])
    sc = meta["supercell"]
    ph = _phonopy(unit, sc)
    ph.generate_displacements(distance=meta["disp_A"],
                              is_plusminus=True if meta["plusminus"] else "auto")
    cells = ph.supercells_with_displacements
    if len(cells) != len(meta["jobs"]):
        raise SystemExit(f"{system}: phonopy now gives {len(cells)} displacements, the inputs "
                         f"were written for {len(meta['jobs'])}")
    forces = []
    for k, cell in enumerate(cells):
        j = jobs.get(meta["jobs"][k])
        if j is None or j["row"]["status"] != "ok":
            raise Deferred(f"{system}: {meta['jobs'][k]} is "
                           f"{'missing' if j is None else j['row']['status']}")
        pos = np.array(j["meta"]["structure"]["positions"])
        want = cell.scaled_positions @ cell.cell
        if np.abs(pos - want).max() > 1e-5:
            raise SystemExit(f"{meta['jobs'][k]}: its input is not phonopy's displaced cell {k + 1}")
        forces.append(np.asarray(j["res"]["forces"], float))
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    n = [int(x) for x in sc]
    qs = [[i / n[0], j / n[1], k / n[2]] for i in range(n[0]) for j in range(n[1])
          for k in range(n[2])]
    ph.run_qpoints(qs, with_eigenvectors=False)
    freqs = np.array(ph.qpoints.frequencies)
    return {"ph": ph, "meta": meta, "qs": qs, "freqs": freqs, "supercell": n}


def _select_mode(qs, freqs, q_dec, mode):
    fr = freqs.copy()
    for qi, q in enumerate(qs):
        if max(abs(c) for c in q) < 1e-8:
            fr[qi, np.argsort(np.abs(fr[qi]))[:3]] = np.inf        # acoustic at Gamma
    if mode == "softest":
        iq, ib = np.unravel_index(int(np.argmin(fr)), fr.shape)
    else:
        qd = np.mod(np.round(np.asarray(q_dec, float), 8), 1.0)
        hit = [i for i, q in enumerate(qs) if np.allclose(np.mod(np.round(q, 8), 1.0), qd,
                                                          atol=1e-6)]
        if not hit:
            raise SystemExit(f"deciding q {q_dec} is not on the FD supercell's q mesh")
        iq = hit[0]
        ib = int(np.argmin(fr[iq]))
    nd = int(np.sum(np.abs(fr[iq] - fr[iq, ib]) < DEGEN_TOL_THZ))
    return int(iq), int(ib), nd


PROF_TAG = {"deciding": "prof", "softest": "profsoft"}


def _write_profile(w, sssp, system, tag, pp, iq, ib, nd, ctx, prov, sel, extra):
    """Freeze the chosen PBE mode at the production amplitudes and write its pw.x inputs."""
    from ase import Atoms
    from mlip_dynstab.finite_t import _imaginary_commensurate_modes, _mode_pattern
    meta, ph = pp["meta"], pp["ph"]
    q, f = pp["qs"][iq], float(pp["freqs"][iq, ib])
    dim, base, u, m_eff = _mode_pattern(ph, list(q), ib, max(pp["supercell"]))
    Qs = dr.Q_MAX * np.linspace(0.0, 1.0, dr.N_PTS) ** 2
    frames = []
    for Q in Qs:
        a = Atoms(symbols=base.get_chemical_symbols(), positions=base.get_positions() + Q * u,
                  cell=base.cell.array, pbc=True)
        a.info["Q"] = float(Q)
        frames.append(a)
    k_, d_ = meta["kspacing"], meta["degauss"]
    ns = argparse.Namespace(kspacing_insulator=k_, kspacing_metal=k_, no_reduce_cell=False)
    lat, reps, _ = dr._pick_cell(frames[-1], system, ns)
    run, reduction = frames, None
    if lat is not None:
        red = [dr._reduce_frame(a, lat, reps) for a in frames]
        if any(r is None for r in red):
            print(f"[pbe-lattice C] {system}/{tag}: a frame is not periodic in the reduced cell; "
                  f"running the full {len(frames[0])}-atom cell")
        else:
            run = [r[0] for r in red]
            reduction = {"n_images": len(frames[0]) // len(run[0]), "lattice": lat.tolist(),
                         "reps": reps}
    n_img = reduction["n_images"] if reduction else 1
    texts = []
    for i, a in enumerate(run):
        job = f"pl_{system}_{tag}_i{i:02d}"
        text, st = pw_text(a, job, system, sssp, kspacing=k_, degauss=d_)
        texts.append((i, job, a, text, st))
    if any(w.would_refuse(t[1], t[3]) for t in texts):
        print(f"[pbe-lattice C] REFUSED {system}/{tag}: finished outputs under other inputs; "
              f"--force redoes it")
        w.n_refused += len(texts)
        return None
    names, est = [], 0.0
    for i, job, a, text, st in texts:
        w.add(job, a, system, text, st,
              {"part": tag, "system": system, "frame": i, "Q": float(Qs[i]), "n_images": n_img},
              "pbe-lattice")
        names.append(job)
        est += dr.cost_core_h(len(a), st["nk_irr_spglib"] or st["nk_tr_upper"],
                              _c0(ctx, system, len(a)))
    imag, n_imag = _imaginary_commensurate_modes(ph, pp["supercell"], dr.DEFAULT_IMAG_TOL_THZ, 24)
    dr._dump(_pl_dir(w.qe.parent, system) / f"{tag}_meta.json", {
        "provenance": prov, "system": system, "tag": tag, "select": sel,
        "q": [float(x) for x in q], "band": ib, "freq_thz": f, "n_degenerate": nd,
        "dim": [int(x) for x in dim], "m_eff": float(m_eff), "n_atoms_supercell": len(base),
        "n_atoms_run": len(run[0]), "Qs": [float(x) for x in Qs], "reduction": reduction,
        "u": np.asarray(u).tolist(), "a_pbe_A": meta["a_pbe_A"], "jobs": names,
        "q_decide": meta["q_decide"], "q_decide_stem": meta["q_decide_stem"],
        "n_imaginary_modes_total": int(n_imag),
        "imaginary_modes": [{"q": m["q"], "band": m["band"], "freq_thz": m["harm_thz"]}
                            for m in imag],
        "mode_pattern": "phonopy eigenvector via mlip_dynstab.finite_t._mode_pattern (the "
                        "screen's own construction; arbitrary inside a degenerate band)",
        "est_core_h": est, **extra})
    warn = "  DEGENERATE band: direction inside the span is arbitrary" if nd > 1 else ""
    print(f"  {system:13s} {sel:8s} q={tuple(round(float(x), 3) for x in q)} band {ib} "
          f"{f:+.3f} THz (n_degenerate {nd}); {len(frames[0])} -> {len(run[0])} atoms; "
          f"{len(names)} jobs est ~{est:.1f} core-h{warn}")
    return est


def _phase_c(args, sssp, prov, w, dft, out, ctx):
    status, tot, n = 0, 0.0, 0
    jobs = _collect(out, args.pos_tol, [], only="pl_")
    for system in args.systems:
        try:
            pp = _pbe_phonons(out, system, jobs)
        except Deferred as exc:
            print(f"[pbe-lattice C] DEFERRED {exc}", flush=True)
            status = NOT_READY
            continue
        except FileNotFoundError:
            print(f"[pbe-lattice C] DEFERRED {system}: run phase B first", flush=True)
            status = NOT_READY
            continue
        q_dec = pp["meta"]["q_decide"]
        picked = {}
        for sel in [s for s in ("deciding", "softest") if s in args.select]:
            iq, ib, nd = _select_mode(pp["qs"], pp["freqs"], q_dec, sel)
            if sel == "softest" and (iq, ib) == picked.get("deciding", (None, None, None))[:2]:
                print(f"  {system:13s} softest    is the deciding-q mode; one profile written")
                continue
            picked[sel] = (iq, ib, nd)
            est = _write_profile(w, sssp, system, PROF_TAG[sel], pp, iq, ib, nd, ctx, prov, sel,
                                 {})
            if est is not None:
                tot += est
                n += dr.N_PTS
    print(f"[pbe-lattice C] {n} jobs, est ~{tot:.0f} core-h")
    return status


def stage_pbe_lattice(args):
    dft, out = Path(args.dft_root), Path(args.out_root)
    sssp = dr.load_sssp(args)
    prov = _prov(args, f"pbe-lattice-{args.phase}")
    w = CheckWriter(out, prov, args.force)
    ctx = _prod_costs(dft)
    if args.phase == "A":
        rc = _phase_a(args, sssp, prov, w, dft, ctx)
    elif args.phase == "B":
        rc = _phase_b(args, sssp, prov, w, dft, out, ctx)
    else:
        rc = _phase_c(args, sssp, prov, w, dft, out, ctx)
    w.report(f"pbe-lattice {args.phase}")
    return rc


# ---------------------------------------------------------------- analysis plumbing ----

def _collect(out, pos_tol, flags, only=None):
    """Every job under <out>/qe with its parsed result, status decided by the rules of
    dft_reference.stage_analyze (finished, SCF converged, pw.x ran THIS input, structure and
    functional as asked)."""
    from ase import Atoms
    jobs = {}
    for jj in sorted((Path(out) / "qe").glob("*/job.json")):
        m = dr._load(jj)
        name = m["job"]
        if only and not name.startswith(only):
            continue
        d = jj.parent
        part = m["members"][0].get("part") if m["members"] else None
        row = {"job": name, "study": m["study"], "part": part, "n_atoms": m["n_atoms"],
               "kmesh": "x".join(map(str, m["qe"]["kmesh"])),
               "nk_irr_spglib": m["qe"].get("nk_irr_spglib"), "status": "pending"}
        res = None
        po = d / "pw.out"
        if po.exists():
            txt = po.read_text(encoding="utf-8", errors="replace")
            r = parse_vcrelax(po) if part == "vcrelax" else dr.parse_pw_out(po)
            meta = _pw_meta(txt)
            row.update({k: meta.get(k) for k in ("qe_version", "nproc", "nk_irr", "wall_s")})
            ran, pin = d / "pw.in.ran", d / "pw.in"
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
            elif meta["xc"] is not None and meta["xc"] != _expected_xc(m):
                row["status"] = "wrong_functional"
                flags.append(f"{name}: pw.x ran XC {meta['xc']}, input asked for "
                             f"{_expected_xc(m)}; excluded")
            elif part == "vcrelax":
                if not r["bfgs_converged"] or "cell_A" not in r:
                    row["status"] = "vc_not_converged"
                    flags.append(f"{name}: vc-relax did not reach 'bfgs converged', excluded")
                else:
                    row["status"] = "ok"
                    res = r
            else:
                exp = Atoms(symbols=m["structure"]["symbols"], positions=m["structure"]["positions"],
                            cell=m["structure"]["cell"], pbc=True)
                mis = dr._pos_mismatch(r["atoms"], exp)
                row["pos_mismatch_A"] = mis
                if mis > pos_tol:
                    row["status"] = "structure_mismatch"
                    flags.append(f"{name}: pw.out structure differs from its input by "
                                 f"{mis:.2e} A, excluded")
                else:
                    row["status"] = "ok"
                    res = r
            if row.get("wall_s") and row.get("nproc"):
                row["core_h"] = row["wall_s"] * row["nproc"] / 3600.0
        jobs[name] = {"meta": m, "row": row, "res": res}
    return jobs


def _settings_sig(meta):
    q = meta["qe"]
    return json.dumps([q["kmesh"], q["ecutwfc_Ry"], q["ecutrho_Ry"], q["smearing"],
                       q["degauss_Ry"], q.get("input_dft"),
                       sorted(v["filename"] for v in q["pseudos"].values())])


def _curve(jobs, group):
    """E(Q)-E(0) (eV, per modulated cell) from {frame: job}. Mirrors stage_analyze: Q=0 and at
    least 4 points are needed, one settings set only; energies of a reduced cell are scaled by
    the image count."""
    ok = [i for i in sorted(group) if jobs[group[i]]["row"]["status"] == "ok"]
    c = {"n": len(group), "n_ok": len(ok)}
    if 0 not in ok or len(ok) < 4:
        c["status"] = "pending" if not ok else "insufficient"
        return c
    if len({_settings_sig(jobs[group[i]]["meta"]) for i in ok}) > 1:
        c["status"] = "mixed_settings"
        return c
    E0 = jobs[group[0]]["res"]["energy_eV"]
    Qs, dE = [], []
    for i in ok:
        mem = jobs[group[i]]["meta"]["members"][0]
        Qs.append(float(mem["Q"]))
        dE.append((jobs[group[i]]["res"]["energy_eV"] - E0) * int(mem["n_images"]))
    c.update({"status": "complete" if len(ok) == len(group) else "partial",
              "Qs": np.array(Qs), "dE": np.array(dE)})
    return c


class CallBatch:
    """Fit + single-mode screen solve for many curves in one batch (each solve is ~1 s)."""

    def __init__(self):
        self.items = {}

    def add(self, key, Qs, dE, m_eff):
        self.items[key] = (np.asarray(Qs, float), np.asarray(dE, float), float(m_eff))

    def run(self, workers):
        self.metrics = {}
        for k, (Qs, dE, m) in self.items.items():
            mt = dr._curve_metrics(Qs, dE, m)
            mt["min_at_edge"] = bool(int(np.argmin(dE)) == len(dE) - 1)
            self.metrics[k] = mt
        tasks, keys = [], []
        for k, mt in self.metrics.items():
            for T in dr.T_SCREEN:
                tasks.append((mt["a"], mt["b"], mt["c"], self.items[k][2], T))
                keys.append((k, T))
        self.calls = dict(zip(keys, dr._solve_all(tasks, workers)))


def _production_refs(dft, flags):
    dft = Path(dft)
    refs = {"curves": {}, "calls": {}, "unit": {}, "files": {}}
    p = dft / "c3a_curves.json"
    if p.exists():
        for c in dr._load(p)["curves"]:
            if c["curve"] == "pbe":
                refs["curves"][c["stem"]] = {"Q": np.array(c["Q_A"], float),
                                             "dE": np.array(c["dE_meV"], float) / 1000.0}
        refs["files"]["c3a_curves.json"] = dr._sha256(p)
    else:
        flags.append(f"{p} missing: no production curves to compare with")
    for name, key in (("c3a_calls.csv", "calls"), ("c3a_unit_calls.csv", "unit")):
        p = dft / name
        if not p.exists():
            flags.append(f"{p} missing")
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if key == "calls":
                    refs["calls"][(r["stem"], float(r["T"]))] = r["pbe_stable"] == "True"
                else:
                    refs["unit"][(r["system"], r["model"], float(r["T"]))] = r
        refs["files"][name] = dr._sha256(p)
    return refs


def _gt(system, T):
    from mlip_dynstab.cli import _finite_t_gt
    return bool(_finite_t_gt(dr.get_spec(system), T))


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], (dict, list, tuple)):
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


_PAT_CACHE = {}          # stem -> production pattern facts, filled by _load_patterns


def _subgrid(Qv, Qp):
    """Indices of the Q values Qv within the production grid Qp, or None if any is not on it
    (a partial curve is a subset of the production grid; a different grid is an error)."""
    Qv, Qp = np.asarray(Qv, float), np.asarray(Qp, float)
    idx = np.array([int(np.argmin(np.abs(Qp - q))) for q in Qv])
    return idx if np.abs(Qp[idx] - Qv).max() <= 1e-9 else None


# ------------------------------------------------------------------- analysis: conv ----

def _analyze_conv(jobs, refs, workers, flags):
    groups = {}
    for name, j in jobs.items():
        mem = j["meta"]["members"][0]
        if j["meta"]["study"] == "conv":
            groups.setdefault((mem["system"], mem["stem"]), {}).setdefault(
                mem["variant"], {})[int(mem["frame"])] = name
    if not groups:
        return None, {}
    batch, ents = CallBatch(), []
    for (system, stem), variants in sorted(groups.items()):
        pat = _PAT_CACHE.get(stem)
        if pat is None:
            flags.append(f"{stem}: no production pattern under geom/patterns (m_eff unknown); "
                         f"skipped")
            continue
        m_eff = pat["m_eff"]
        mem0 = jobs[next(iter(next(iter(variants.values())).values()))]["meta"]["members"][0]
        ent = {"system": system, "stem": stem, "role": mem0["role"], "path_model": mem0["path_model"],
               "m_eff": m_eff, "variants": {}, "curves": {}}
        pc = refs["curves"].get(stem)
        if pc is None:
            flags.append(f"{stem}: no production PBE curve in c3a_curves.json")
        else:
            batch.add((stem, "prod"), pc["Q"], pc["dE"], m_eff)
            ent["curves"]["prod"] = pc
        for v in CV_VARIANTS:
            if v not in variants:
                continue
            c = _curve(jobs, variants[v])
            ent["variants"][v] = {"status": c["status"], "n_ok": c["n_ok"], "n": c["n"]}
            if "dE" in c:
                idx = None if pc is None else _subgrid(c["Qs"], pc["Q"])
                if pc is not None and idx is None:
                    flags.append(f"{stem}/{v}: Q grid differs from the production curve")
                    ent["variants"][v]["status"] = "grid_mismatch"
                    continue
                batch.add((stem, v), c["Qs"], c["dE"], m_eff)
                ent["curves"][v] = {"Q": c["Qs"], "dE": c["dE"], "idx": idx}
        ents.append(ent)
    batch.run(workers)

    paths, rows, calls, any_incomplete = [], [], [], False
    for ent in ents:
        stem, system = ent["stem"], ent["system"]
        prod = batch.metrics.get((stem, "prod"))
        entry = {"system": system, "stem": stem, "role": ent["role"], "variants": {}}
        for T in dr.T_SCREEN:                      # self-check of the pipeline against C3a
            if prod is not None and (stem, float(T)) in refs["calls"]:
                if refs["calls"][(stem, float(T))] != batch.calls[((stem, "prod"), T)]["stable"]:
                    flags.append(f"{stem} T={T:g}: production curve re-solved here disagrees "
                                 f"with c3a_calls.csv pbe_stable")
        names = ["prod"] + [v for v in CV_VARIANTS if v in ent["variants"]]
        for v in names:
            key = (stem, v)
            if key not in batch.metrics:
                st = ent["variants"].get(v, {}).get("status", "no_production_curve")
                entry["variants"][v] = {"status": st}
                continue
            mt = batch.metrics[key]
            row = {"system": system, "stem": stem, "variant": v,
                   "status": "production" if v == "prod" else ent["variants"][v]["status"],
                   "n_Q": len(batch.items[key][0]), **{k: mt[k] for k in (
                       "depth_meV", "Q_min_A", "min_at_edge", "a", "b", "c", "bounded",
                       "omega_harm_thz")}}
            if v != "prod" and prod is not None:
                dpr, dv = prod["depth_meV"], mt["depth_meV"]
                row["depth_rel_change"] = (dv - dpr) / dpr if dpr > 1e-9 else None
                row["depth_ok"] = (abs(dv - dpr) <= ACCEPT_DEPTH_REL * dpr if dpr > 1e-9
                                   else abs(dv - dpr) <= 0.1)
                row["dQ_min_A"] = mt["Q_min_A"] - prod["Q_min_A"]
                Qp = batch.items[(stem, "prod")]
                idx = ent["curves"][v]["idx"]               # the variant's Q values in the grid
                diff = (batch.items[key][1] - Qp[1][idx]) * 1000
                win = dr._fit_window(Qp[1])[idx]
                row["rmse_vs_prod_window_meV"] = (float(np.sqrt(np.mean(diff[win] ** 2)))
                                                  if win.any() else None)
                row["max_abs_vs_prod_meV"] = float(np.abs(diff).max())
                same = [batch.calls[(key, T)]["stable"] == batch.calls[((stem, "prod"), T)]["stable"]
                        for T in dr.T_SCREEN]
                row["n_T_call_changed"] = int(len(same) - sum(same))
                row["calls_identical"] = bool(all(same))
                row["pass"] = bool(row["depth_ok"] and row["calls_identical"]
                                   and ent["variants"][v]["status"] == "complete")
            for T in dr.T_SCREEN:
                cl = batch.calls[(key, T)]
                crow = {"system": system, "stem": stem, "variant": v, "T": T,
                        "gt_stable": _gt(system, T), "stable": cl["stable"],
                        "Q0_A": cl["Q0_A"], "curv_thz": cl["curv_thz"]}
                if v != "prod" and prod is not None:
                    crow["prod_stable"] = batch.calls[((stem, "prod"), T)]["stable"]
                    crow["same_as_prod"] = crow["stable"] == crow["prod_stable"]
                calls.append(crow)
            rows.append(row)
            entry["variants"][v] = {k: row[k] for k in row if k not in ("system", "stem")}
        vs = [entry["variants"].get(v, {}) for v in CV_VARIANTS]
        have = all(v.get("status") == "complete" for v in vs)
        entry["all_variants_complete"] = have
        entry["converged"] = bool(have and all(v.get("pass") for v in vs))
        entry["failed_variants"] = [v for v in CV_VARIANTS
                                    if entry["variants"].get(v, {}).get("pass") is False]
        entry["min_at_edge"] = bool(prod and prod["min_at_edge"])
        any_incomplete |= not have
        paths.append(entry)
    verdict = ("incomplete" if any_incomplete else
               "converged" if all(p["converged"] for p in paths) else "NOT converged")
    summ = {"acceptance": f"depth within {ACCEPT_DEPTH_REL:.0%} and screen call identical at "
                          f"every T in {list(dr.T_SCREEN)}, for each variant, on every path",
            "verdict": verdict, "paths": paths}
    return summ, {"conv_variants": rows, "conv_calls": calls}


def _load_patterns(dft):
    _PAT_CACHE.clear()
    for d in _patterns(dft):
        _PAT_CACHE[d["stem"]] = {"m_eff": d["m_eff"], "q": d["q"], "band": d["band"],
                                 "role": d["role"], "path_model": d["path_model"],
                                 "decide_T": d.get("decide_T") or [], "harm_thz": d["harm_thz"],
                                 "system": d["system"], "depth_meV": d["depth_meV"],
                                 "n_atoms": d["n_atoms"]}


# --------------------------------------------------------------------- analysis: xc ----

def _analyze_xc(jobs, refs, workers, flags):
    groups = {}
    for name, j in jobs.items():
        mem = j["meta"]["members"][0]
        if j["meta"]["study"] == "xc":
            groups.setdefault((mem["system"], mem["path_model"], mem["stem"]), {})[
                int(mem["frame"])] = name
    if not groups:
        return None, {}
    batch, ents = CallBatch(), {}
    for (system, model, stem), grp in sorted(groups.items()):
        pat = _PAT_CACHE.get(stem)
        c = _curve(jobs, grp)
        ent = {"system": system, "model": model, "stem": stem, "status": c["status"],
               "role": pat["role"] if pat else None, "decide_T": pat["decide_T"] if pat else [],
               "n_ok": c["n_ok"], "n": c["n"]}
        pc = refs["curves"].get(stem)
        if pat is None or pc is None:
            flags.append(f"{stem}: production pattern or PBE curve missing")
            ents[stem] = ent
            continue
        if "dE" in c and _subgrid(c["Qs"], pc["Q"]) is None:
            flags.append(f"{stem}: PBEsol Q grid differs from the production curve")
            ent["status"] = "grid_mismatch"
            ents[stem] = ent
            continue
        batch.add((stem, "pbe"), pc["Q"], pc["dE"], pat["m_eff"])
        if "dE" in c:
            batch.add((stem, "pbesol"), c["Qs"], c["dE"], pat["m_eff"])
        ents[stem] = ent
    batch.run(workers)

    path_rows, call_rows = [], []
    for stem, ent in ents.items():
        kp, ks = (stem, "pbe"), (stem, "pbesol")
        if kp not in batch.metrics:
            continue
        row = {"system": ent["system"], "model": ent["model"], "stem": stem, "role": ent["role"],
               "status": ent["status"], "n_Q_pbesol": ent["n_ok"],
               "decide_T": ",".join(f"{t:g}" for t in ent["decide_T"])}
        for tag, k in (("pbe", kp), ("pbesol", ks)):
            if k in batch.metrics:
                mt = batch.metrics[k]
                row.update({f"{tag}_depth_meV": mt["depth_meV"], f"{tag}_Q_min_A": mt["Q_min_A"],
                            f"{tag}_min_at_edge": mt["min_at_edge"],
                            f"{tag}_omega_harm_thz": mt["omega_harm_thz"]})
        if ks in batch.metrics and batch.metrics[kp]["depth_meV"] > 1e-9:
            row["depth_ratio_pbesol_over_pbe"] = (batch.metrics[ks]["depth_meV"]
                                                  / batch.metrics[kp]["depth_meV"])
        path_rows.append(row)
        for T in dr.T_SCREEN:
            c = {"system": ent["system"], "model": ent["model"], "stem": stem, "T": T,
                 "gt_stable": _gt(ent["system"], T),
                 "pbe_stable": batch.calls[(kp, T)]["stable"]}
            if ks in batch.metrics:
                c["pbesol_stable"] = batch.calls[(ks, T)]["stable"]
                c["pbesol_Q0_A"] = batch.calls[(ks, T)]["Q0_A"]
                c["pbe_Q0_A"] = batch.calls[(kp, T)]["Q0_A"]
            call_rows.append(c)

    # unit level: the screen's rule (unstable if ANY mapped mode condenses) over the selected
    # deciding/reference paths, PBE and PBEsol on the SAME paths
    unit_rows, units = [], {}
    for e in ents.values():
        units.setdefault((e["system"], e["model"]), []).append(e["stem"])
    for (system, model), stems in sorted(units.items()):
        have = [s for s in stems if (s, "pbesol") in batch.metrics
                and ents[s]["status"] in ("complete", "partial")]
        complete = len(have) == len(stems) and all(ents[s]["status"] == "complete" for s in stems)
        if not have:
            continue
        for T in dr.T_SCREEN:
            pbe = all(batch.calls[((s, "pbe"), T)]["stable"] for s in have)
            sol = all(batch.calls[((s, "pbesol"), T)]["stable"] for s in have)
            led = dr._ledger_call(system, model, T)
            prod_u = refs["unit"].get((system, model, float(T)))
            gt = _gt(system, T)
            row = {"system": system, "model": model, "T": T, "in_ladder": led is not None,
                   "gt_stable": gt, "mlip_pred_stable_ledger": led, "pbe_backed_stable": pbe,
                   "pbesol_backed_stable": sol, "n_paths": len(have), "n_paths_selected": len(stems),
                   "unit_complete": complete}
            if prod_u is not None:
                row["c3a_unit_pbe_backed_stable"] = prod_u["pbe_backed_stable"] == "True"
                if row["c3a_unit_pbe_backed_stable"] != pbe:
                    flags.append(f"{system}/{model} T={T:g}: PBE-backed call on the {len(have)} "
                                 f"selected paths ({pbe}) differs from c3a_unit_calls.csv "
                                 f"({row['c3a_unit_pbe_backed_stable']}): coverage differs")
            if led is not None:
                row["pbe_error"] = bool(pbe != gt)
                row["pbesol_error"] = bool(sol != gt)
                row["mlip_error"] = bool(led != gt)
                row["persistent_pbe_error"] = row["pbe_error"]
                row["pbesol_changes_call"] = bool(sol != pbe)
            unit_rows.append(row)
    lad = [r for r in unit_rows if r["in_ladder"] and r["unit_complete"]]
    pers = [r for r in lad if r["pbe_error"]]
    summ = {
        "functional": "PBEsol (input_dft) on SSSP 1.3 PBE pseudopotentials; NOT the SSSP PBEsol set",
        "n_unit_rows_ladder_complete": len(lad),
        "pbe_errors": len(pers),
        "pbe_errors_also_wrong_in_pbesol": sum(r["pbesol_error"] for r in pers),
        "pbe_errors_fixed_by_pbesol": sum(not r["pbesol_error"] for r in pers),
        "new_errors_from_pbesol": sum(r["pbesol_error"] and not r["pbe_error"] for r in lad),
        "calls_changed_by_pbesol": sum(r["pbesol_changes_call"] for r in lad),
        "persistent_errors": [{"system": r["system"], "model": r["model"], "T": r["T"],
                               "gt_stable": r["gt_stable"], "pbe": r["pbe_backed_stable"],
                               "pbesol": r["pbesol_backed_stable"]} for r in pers],
        "by_system": {s: {"pbe_errors": sum(r["pbe_error"] for r in lad if r["system"] == s),
                          "pbesol_errors": sum(r["pbesol_error"] for r in lad if r["system"] == s)}
                      for s in sorted({r["system"] for r in lad})},
        "n_paths": len(path_rows),
        "depth_ratio_pbesol_over_pbe": _range([r.get("depth_ratio_pbesol_over_pbe")
                                               for r in path_rows])}
    return summ, {"xc_paths": path_rows, "xc_calls": call_rows, "xc_unit_calls": unit_rows}


def _range(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else {"min": float(min(xs)), "median": float(np.median(xs)),
                                "max": float(max(xs)), "n": len(xs)}


# ------------------------------------------------------------ analysis: pbe-lattice ----

def _mlip_lattice(dft, system):
    out = {}
    for model in dr.MODELS:
        p = Path(dft) / "geom" / "checks" / f"{system}_{model}.json"
        if not p.exists():
            continue
        rec = dr._load(p)
        cell = rec.get("relaxed_prim_cell")
        if cell:
            c = np.array(cell)
            out[model] = {"a_A": float(np.cbrt(abs(np.linalg.det(c)))),
                          "cubic_deviation_A": float(np.abs(c - np.eye(3) * np.mean(np.diag(c))).max())}
    return out


def _analyze_pl(out, dft, jobs, refs, workers, flags):
    systems = sorted({j["meta"]["system"] for j in jobs.values()
                      if j["meta"]["study"] == "pbe-lattice"})
    if not systems:
        return None, {}
    batch = CallBatch()
    per, lat_rows, ph_rows = {}, [], []
    for system in systems:
        ent = {"phase_A": "pending", "phase_B": "pending", "phase_C": "pending"}
        mlip = _mlip_lattice(dft, system)
        reg_a = float(_prim(system).cell.array[0, 0])
        vcj = jobs.get(f"pl_{system}_vcrelax")
        a_pbe = None
        if vcj is not None and vcj["row"]["status"] == "ok":
            r = vcj["res"]
            cell = np.array(r["cell_A"])
            a_pbe = float(np.mean(np.diag(cell)))
            ent["phase_A"] = "done"
            ent["lattice"] = {"a_pbe_A": a_pbe, "registry_start_A": reg_a,
                              "cubic_deviation_A": float(np.abs(cell - np.eye(3) * a_pbe).max()),
                              "volume_A3": r.get("volume_A3"), "bfgs_steps": r.get("n_bfgs_steps"),
                              "final_scf_pressure_kbar": r.get("final_scf_pressure_kbar"),
                              "mlip_a_A": {m: v["a_A"] for m, v in mlip.items()},
                              "mlip_minus_pbe_pct": {m: 100 * (v["a_A"] - a_pbe) / a_pbe
                                                     for m, v in mlip.items()}}
            if abs(r.get("final_scf_pressure_kbar") or 0.0) > 5.0:
                flags.append(f"{system}: vc-relax final-scf pressure "
                             f"{r['final_scf_pressure_kbar']:.1f} kbar (> 5): Pulay stress at "
                             f"the fixed production cutoff; a_PBE is the constant-PW minimum")
            lat_rows.append({"system": system, "source": "PBE vc-relax (this check)",
                             "a_A": a_pbe, "minus_pbe_pct": 0.0})
            lat_rows.append({"system": system, "source": "registry start", "a_A": reg_a,
                             "minus_pbe_pct": 100 * (reg_a - a_pbe) / a_pbe})
            for m, v in mlip.items():
                lat_rows.append({"system": system, "source": f"{m} relaxed", "a_A": v["a_A"],
                                 "minus_pbe_pct": 100 * (v["a_A"] - a_pbe) / a_pbe})
        elif vcj is not None:
            ent["phase_A"] = vcj["row"]["status"]
        fdp = _pl_dir(out, system) / "fd_meta.json"
        pp = None
        if fdp.exists():
            try:
                pp = _pbe_phonons(out, system, jobs)
                ent["phase_B"] = "done"
            except Deferred as exc:
                ent["phase_B"] = f"incomplete ({exc})"
        if pp is not None:
            fr = pp["freqs"]
            for qi, q in enumerate(pp["qs"]):
                for b in range(min(6, fr.shape[1])):
                    ph_rows.append({"system": system, "source": "PBE FD at PBE lattice",
                                    "q": ",".join(dr._frac(x) for x in q), "band": b,
                                    "freq_thz": float(fr[qi, b])})
            g = [i for i, q in enumerate(pp["qs"]) if max(abs(c) for c in q) < 1e-8]
            ent["gamma_acoustic_max_abs_thz"] = (float(np.sort(np.abs(fr[g[0]]))[:3].max())
                                                 if g else None)
            ent["n_displacements"] = pp["meta"]["n_displacements"]
            ent["supercell"] = pp["meta"]["supercell"]
        for d in _patterns(dft, system):
            if d["role"] == "decide":
                ph_rows.append({"system": system, "source": f"{d['path_model']} (MLIP lattice)",
                                "stem": d["stem"], "q": ",".join(dr._frac(x) for x in d["q"]),
                                "band": d["band"], "freq_thz": d["harm_thz"]})
        mp, _ = _model_deciding_path(dft, system, SELECT_MODEL)
        ent["profiles"] = {}
        for sel, tag in PROF_TAG.items():
            pm_p = _pl_dir(out, system) / f"{tag}_meta.json"
            grp = {}
            for name, j in jobs.items():
                mem = j["meta"]["members"][0]
                if j["meta"]["study"] == "pbe-lattice" and mem.get("part") == tag \
                        and j["meta"]["system"] == system:
                    grp[int(mem["frame"])] = name
            if not (pm_p.exists() and grp):
                continue
            pm = dr._load(pm_p)
            c = _curve(jobs, grp)
            ent["profiles"][sel] = {
                "status": c["status"], "tag": tag,
                **{k: pm[k] for k in ("q", "band", "freq_thz", "n_degenerate", "dim", "m_eff",
                                      "q_decide", "n_atoms_run", "n_imaginary_modes_total")},
                "imaginary_modes": pm["imaginary_modes"][:6]}
            if "dE" in c:
                batch.add((system, f"pbe_lattice_{sel}"), c["Qs"], c["dE"], pm["m_eff"])
        pc = refs["curves"].get(mp["stem"])
        if pc is not None and ent["profiles"]:
            batch.add((system, "pbe_mlip_lattice"), pc["Q"], pc["dE"], mp["m_eff"])
            ent["mlip_path_stem"] = mp["stem"]
            ent["mlip_path_decide_T"] = mp.get("decide_T")
            ent["mlip_path_q"] = mp["q"]
        ent["phase_C"] = ({s: p["status"] for s, p in ent["profiles"].items()}
                          or "pending")
        per[system] = ent
    batch.run(workers)

    curve_tags = ("pbe_lattice_deciding", "pbe_lattice_softest", "pbe_mlip_lattice")
    prof_rows, call_rows = [], []
    for system, ent in per.items():
        for tag in curve_tags:
            k = (system, tag)
            if k not in batch.metrics:
                continue
            mt = batch.metrics[k]
            ent.setdefault("profile", {})[tag] = {x: mt[x] for x in (
                "depth_meV", "Q_min_A", "min_at_edge", "omega_harm_thz", "a", "b", "c")}
            prof_rows.append({"system": system, "curve": tag, **ent["profile"][tag],
                              "n_Q": len(batch.items[k][0])})
        have = [t for t in curve_tags if (system, t) in batch.metrics]
        for T in (dr.T_SCREEN if have else ()):
            gt = _gt(system, T)
            led = dr._ledger_call(system, SELECT_MODEL, T)
            row = {"system": system, "T": T, "gt_stable": gt, "mlip_ledger_stable": led}
            for t in have:
                row[f"{t}_stable"] = batch.calls[((system, t), T)]["stable"]
                row[f"{t}_Q0_A"] = batch.calls[((system, t), T)]["Q0_A"]
                if led is not None:
                    row[f"{t}_error"] = bool(row[f"{t}_stable"] != gt)
            call_rows.append(row)
    summ = {"note": "single-mode screen: a 'condenses' result is a call of unstable; a 'stable' "
                    "result is the call only if no other mode condenses. pbe_lattice_deciding = "
                    "softest PBE band at the MACE deciding q; pbe_lattice_softest = softest PBE "
                    "mode of the whole q mesh; pbe_mlip_lattice = production PBE at MACE's "
                    "lattice along MACE's eigenvector",
            "systems": per}
    return summ, {"pl_lattice": lat_rows, "pl_phonons": ph_rows, "pl_profile": prof_rows,
                  "pl_calls": call_rows}


# ---------------------------------------------------------------------- analyze ----

def stage_analyze_checks(args):
    out, dft = Path(args.out_root), Path(args.dft_root)
    prov = _prov(args, "analyze-checks")
    flags = []
    prov["production_inputs_sha256"] = {}
    jobs = _collect(out, args.pos_tol, flags)
    refs = _production_refs(dft, flags)
    prov["production_inputs_sha256"] = refs["files"]
    _load_patterns(dft)
    summary = {"provenance": prov, "flags": flags}
    tables = {}
    for name, fn, args_ in (("conv", lambda: _analyze_conv(jobs, refs, args.workers, flags), "conv"),
                            ("xc", lambda: _analyze_xc(jobs, refs, args.workers, flags), "xc"),
                            ("pbe_lattice", lambda: _analyze_pl(out, dft, jobs, refs, args.workers,
                                                                flags), "pbe-lattice")):
        if args_ not in args.stages:
            continue
        s, t = fn()
        summary[name] = s if s is not None else {"status": "no jobs written for this check"}
        tables.update(t)
    rows = [dict(j["row"]) for j in jobs.values()]
    tables["qe_jobs"] = rows
    meas = [r for r in rows if r.get("core_h")]
    pend = [r for r in rows if r["status"] in ("pending", "incomplete", "stale_output")]
    c0 = [r["core_h"] / dr.cost_core_h(r["n_atoms"], r["nk_irr"], 1.0) for r in meas
          if r.get("nk_irr")]
    summary["cost"] = {"n_jobs": len(rows),
                       "by_status": {s: sum(1 for r in rows if r["status"] == s)
                                     for s in sorted({r["status"] for r in rows})},
                       "core_h_spent": float(sum(r.get("core_h") or 0.0 for r in rows)),
                       "core_h_spent_by_prefix": {p: float(sum(r.get("core_h") or 0.0 for r in rows
                                                               if r["job"].startswith(p)))
                                                  for p in ("cv_", "xs_", "pl_")},
                       "c0_core_s_measured_median": float(np.median(c0)) if c0 else None,
                       "n_measured": len(meas),
                       "core_h_remaining_model": (float(sum(dr.cost_core_h(
                           r["n_atoms"], r["nk_irr_spglib"] or 1, float(np.median(c0)))
                           for r in pend)) if c0 else None)}
    summary["conventions"] = {
        "energies": "E(Q)-E(0) within one settings set, scaled by the image count of a reduced "
                    "cell, per modulated cell, as in dft_reference.stage_analyze",
        "production": "PBE curve of the same path from results/revision/dft/c3a_curves.json "
                      "(MLIP relaxed lattice, production settings), re-solved here",
        "calls": "mlip_dynstab.finite_t._solve_scha via dft_reference._screen_call, T in T_SCREEN",
        "xc_approximation": "PBEsol via input_dft on SSSP PBE pseudopotentials"}
    out.mkdir(parents=True, exist_ok=True)
    dr._dump(out / "summary.json", summary)
    for name, rows_ in tables.items():
        _write_csv(out / f"{name}.csv", rows_)
    cv = summary.get("conv", {})
    print(f"[analyze-checks] jobs {summary['cost']['by_status']}; flags {len(flags)}")
    if cv.get("verdict"):
        print(f"[analyze-checks] conv: {cv['verdict']}")
    xc = summary.get("xc", {})
    if xc.get("pbe_errors") is not None:
        print(f"[analyze-checks] xc: {xc['pbe_errors']} PBE errors, "
              f"{xc['pbe_errors_also_wrong_in_pbesol']} also wrong in PBEsol, "
              f"{xc['new_errors_from_pbesol']} new")
    for f in flags[:20]:
        print(f"  FLAG {f}")
    return 0


# -------------------------------------------------------------------------- CLI ----

def _common(p):
    p.add_argument("--out-root", type=Path, default=CHECKS_OUT_DEFAULT,
                   help="where the checks' jobs and results go")
    p.add_argument("--dft-root", type=Path, default=dr.OUT_DEFAULT,
                   help="production C3a results (read only)")
    p.add_argument("--systems", nargs="+", default=list(CHECK_SYSTEMS), choices=list(CHECK_SYSTEMS))
    p.add_argument("--force", action="store_true", help="redo work whose output exists")


def _qe(p):
    p.add_argument("--sssp-dir", default=None,
                   help="directory holding the SSSP 1.3 PBE efficiency json (and UPFs)")
    p.add_argument("--sssp-json", default=None)
    p.add_argument("--kspacing-insulator", type=float, default=None,
                   help="fallback only (1/A); the production value is read from the production "
                        "job.json")
    p.add_argument("--degauss-insulator", type=float, default=None, help="fallback only (Ry)")
    p.add_argument("--no-require-pbe", action="store_true",
                   help="write inputs even where the production PBE jobs are not finished")


def register(sub):
    p = sub.add_parser("conv-inputs", help="k-point/cutoff convergence of the deciding paths")
    _common(p)
    _qe(p)
    p.add_argument("--model", default=SELECT_MODEL, choices=list(dr.MODELS),
                   help="whose deciding path is converged")
    p.add_argument("--variants", nargs="+", default=list(CV_VARIANTS), choices=list(CV_VARIANTS),
                   help="k = finer k-spacing, e = larger cutoff, ke = both")
    p.add_argument("--kspacing-fine", type=float, default=0.15)
    p.add_argument("--ecut-scale", type=float, default=1.3)
    p.set_defaults(func=stage_conv_inputs)

    p = sub.add_parser("xc-inputs", help="PBEsol on every deciding path of the three systems")
    _common(p)
    _qe(p)
    p.add_argument("--functional", default="pbesol")
    p.set_defaults(func=stage_xc_inputs)

    p = sub.add_parser("pbe-lattice", help="PBE lattice, PBE force constants, PBE eigenvector")
    _common(p)
    _qe(p)
    p.add_argument("--phase", required=True, choices=["A", "B", "C"],
                   help="A vc-relax, B finite-displacement supercells (needs A), "
                        "C E(Q) profile (needs B)")
    p.add_argument("--model", default=SELECT_MODEL, choices=list(dr.MODELS),
                   help="whose deciding path supplies the deciding q")
    p.add_argument("--select", nargs="+", default=["deciding", "softest"],
                   choices=["deciding", "softest"],
                   help="C: profile along the softest band at the deciding q (prof_*) and/or "
                        "along the softest mode of the whole q mesh (profsoft_*; skipped where "
                        "it is the deciding mode)")
    p.add_argument("--fd-disp", type=float, default=0.01,
                   help="B: finite displacement in A, as the MLIP screen")
    p.add_argument("--fd-plusminus", action="store_true",
                   help="B: +/- for every displacement (doubles the FD jobs)")
    p.add_argument("--fd-supercell", nargs=3, type=int, default=None,
                   help="B: override the FD supercell (default: 2x2x2, or 1x1x1 for Gamma only)")
    p.add_argument("--pos-tol", type=float, default=1e-3)
    p.set_defaults(func=stage_pbe_lattice)

    p = sub.add_parser("analyze-checks", help="conv / xc / pbe-lattice results (local)")
    _common(p)
    p.add_argument("--stages", nargs="+", default=["conv", "xc", "pbe-lattice"],
                   choices=["conv", "xc", "pbe-lattice"])
    p.add_argument("--pos-tol", type=float, default=1e-3,
                   help="max A between the pw.out structure and its input")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="processes for the screen solves")
    p.set_defaults(func=stage_analyze_checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    register(sub)
    args = ap.parse_args(argv)
    args.out_root, args.dft_root = Path(args.out_root), Path(args.dft_root)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
