"""PBE reference data of the held-out sets, rebuilt from the pw.out files.

The logic is a copy of the job -> geometry bookkeeping in scripts/dft_reference.py:stage_analyze
(copied rather than imported: that file is being extended), so that the held-out evaluation does
not depend on when someone last re-ran `analyze`.  selftest.c3a_base checks the rebuilt PBE
curves against dft_reference's own results/revision/dft/c3a_curves.json.

A job serves a geometry frame only while that frame (in the job's reduced cell, if any) still
hashes to the structure the job was written for.  Finished = 'JOB DONE', converged, pw.in.ran
identical to pw.in, structure equal to the frame to --pos-tol.  A path needs its Q = 0 frame and
at least four points; all its jobs must share k-mesh, cutoffs, smearing, pseudos and cell.
Energies of reduced-cell jobs are multiplied by the image count and every supercell atom takes
its representative's force, so a path is always compared on the identical supercell structures
the MLIP was evaluated on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import common as C
from .data import parse_pw_out, pos_mismatch
from .qe import structure_sha

POS_TOL = 1e-3


def reduce_frame(frame, lat, reps=None, tol=1e-5):
    """(reduced Atoms, map supercell atom -> reduced atom, representatives) or None.  Copy of
    dft_reference._reduce_frame."""
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


class PBEData:
    def __init__(self, dft_root: Path = C.DFT_ROOT, systems=C.TEST_SYSTEMS, pos_tol=POS_TOL,
                 prefixes=("a_", "ax_", "b_")):
        self.root = Path(dft_root)
        self.systems = set(systems)
        self.flags = []
        self.jobs, self.member_of, self._geom = {}, {}, {}
        for jj in sorted((self.root / "qe").glob("*/job.json")):
            m = C.jload(jj)
            if not m["job"].startswith(prefixes) or m["system"] not in self.systems:
                continue
            self.jobs[m["job"]] = m
            for mem in m["members"]:
                gf, k = self.root / mem["geom_file"], int(mem["frame"])
                frames = self._frames(gf)
                if k >= len(frames):
                    self.flags.append(f"{m['job']}: member {mem['geom_file']}#{k} no longer exists")
                    continue
                a, red = frames[k], mem.get("reduction")
                if red:
                    r_ = reduce_frame(a, np.array(red["lattice"]), red["reps"])
                    a = r_[0] if r_ is not None else None
                if a is None or structure_sha(a) != m["structure_sha"]:
                    self.flags.append(f"{m['job']}: {mem['geom_file']}#{k} now holds a different structure")
                    continue
                self.member_of[(mem["geom_file"], k)] = (m["job"], mem)
        served = {}
        for key, (jname, mem) in self.member_of.items():
            served.setdefault(jname, mem)
        self.results = {}
        self.status = {}
        for name, m in self.jobs.items():
            po = self.root / "qe" / name / "pw.out"
            if not po.exists():
                self.status[name] = "pending"
                continue
            r = parse_pw_out(po)
            ran, pin = self.root / "qe" / name / "pw.in.ran", self.root / "qe" / name / "pw.in"
            if not r["job_done"]:
                self.status[name] = "incomplete"
            elif ran.exists() and ran.read_bytes() != pin.read_bytes():
                self.status[name] = "stale_output"
            elif not r["converged"]:
                self.status[name] = "scf_not_converged"
            elif "parse_error" in r:
                self.status[name] = "parse_error"
            elif name not in served:
                self.status[name] = "orphaned"
            else:
                mem = served[name]
                frame = self._frames(self.root / mem["geom_file"])[int(mem["frame"])]
                red = mem.get("reduction")
                if red:
                    frame = reduce_frame(frame, np.array(red["lattice"]), red["reps"])[0]
                mis = pos_mismatch(r["atoms"], frame)
                if mis > pos_tol:
                    self.status[name] = "structure_mismatch"
                    self.flags.append(f"{name}: pw.out structure differs from its geometry by {mis:.2e} A")
                else:
                    self.status[name] = "ok"
                    self.results[name] = r
        self.n_ok = sum(1 for s in self.status.values() if s == "ok")

    def _frames(self, gf):
        if gf not in self._geom:
            self._geom[gf] = C.read_frames(gf) if gf.exists() else []
        return self._geom[gf]

    def frames(self, geom_rel):
        return self._frames(self.root / geom_rel)

    def dft(self, geom_rel, frame):
        hit = self.member_of.get((geom_rel, frame))
        if not hit or hit[0] not in self.results:
            return None
        r, red = self.results[hit[0]], hit[1].get("reduction")
        if not red:
            return {"energy_eV": r["energy_eV"], "forces": r["forces"]}
        return {"energy_eV": r["energy_eV"] * red["n_images"],
                "forces": r["forces"][np.asarray(red["map"])]}

    def settings(self, geom_rel, frame):
        hit = self.member_of.get((geom_rel, frame))
        if not hit:
            return None
        q, red = self.jobs[hit[0]]["qe"], hit[1].get("reduction")
        return json.dumps([q["kmesh"], q["ecutwfc_Ry"], q["ecutrho_Ry"], q["smearing"],
                           q["degauss_Ry"], sorted(v["filename"] for v in q["pseudos"].values()),
                           None if not red else np.round(red["lattice"], 6).tolist()])

    # ---------------------------------------------------------------- C3a paths ----

    def c3a_paths(self):
        """One dict per path file with PBE: Q, E, F over the frames that have it."""
        out = []
        for p in sorted((self.root / "geom" / "patterns").glob("*.json")):
            pat = C.jload(p)
            if pat["system"] not in self.systems:
                continue
            gf = self.root / pat["geom_file"]
            if not gf.exists():
                continue
            frames = self._frames(gf)
            have = [i for i in range(len(frames)) if self.dft(pat["geom_file"], i) is not None]
            rec = {"stem": pat["stem"], "system": pat["system"], "path_model": pat["path_model"],
                   "pattern_model": pat["pattern_model"], "role": pat["role"], "q": pat["q"],
                   "band": pat["band"], "dim": pat["dim"], "n_atoms": pat["n_atoms"],
                   "m_eff": pat["m_eff"], "decide_T": pat.get("decide_T", []),
                   "geom_file": pat["geom_file"], "n_Q": len(frames), "n_Q_dft": len(have),
                   "cache_check_pass": pat["check"].get("pass")}
            if 0 not in have or len(have) < 4:
                rec["status"] = "pending" if not have else "insufficient_dft"
                out.append(rec)
                continue
            if len({self.settings(pat["geom_file"], i) for i in have}) > 1:
                rec["status"] = "mixed_settings"
                self.flags.append(f"{pat['stem']}: its jobs differ in k-mesh/cutoffs/cell; excluded")
                out.append(rec)
                continue
            rec["status"] = "complete" if len(have) == len(frames) else "partial"
            rec["idx"] = np.array(have)
            rec["Q"] = np.array([float(frames[i].info["Q"]) for i in have])
            rec["E"] = np.array([self.dft(pat["geom_file"], i)["energy_eV"] for i in have])
            rec["F"] = [self.dft(pat["geom_file"], i)["forces"] for i in have]
            out.append(rec)
        return out

    # ---------------------------------------------------------------- C3b files ----

    def c3b_sets(self):
        """{geom file: [(frame, role, info, dft or None)]} for geom_c3b files of the systems."""
        out = {}
        for gf in sorted((self.root / "geom_c3b").glob("*.extxyz")):
            if gf.name.endswith(".tmp.extxyz"):
                continue
            rel = gf.relative_to(self.root).as_posix()
            frames = self._frames(gf)
            if not frames or frames[0].info.get("system") not in self.systems:
                continue
            out[rel] = [(a, a.info.get("role"), dict(a.info), self.dft(rel, k))
                        for k, a in enumerate(frames)]
        return out
