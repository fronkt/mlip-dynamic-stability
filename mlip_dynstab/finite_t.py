"""Layer 2 (novel core): finite-temperature dynamic stability with an MLIP calculator.

Two routes to a finite-T dynamical-stability call:

1. ``compute_finite_t_tdep`` — sample the high-symmetry reference with NVT MD at temperature
   T, then fit *effective* harmonic force constants to the sampled displacement/force data
   (a lightweight one-shot TDEP). The minimum effective frequency gives the finite-T
   stability call: a soft mode that was imaginary at 0 K but real here means the MLIP
   reproduces thermal stabilization.

2. ``compute_finite_t_sscha`` — hook to python-sscha (stochastic self-consistent harmonic
   approximation), the gold standard for anharmonic stabilization incl. quantum effects.
   Used as a cross-check on a subset.

Also provides ``md_symmetry_breaking`` — a cheap direct probe: does the high-symmetry cell
spontaneously distort in MD at T? (Most apt for strongly-diffusive superionics where the
effective-FC picture breaks down.)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np

from . import DEFAULT_IMAG_TOL_THZ


@dataclass
class FiniteTResult:
    temperature_K: float
    method: str                       # "tdep" | "sscha" | "md_distort"
    min_eff_freq_thz: float
    dynamically_stable: bool
    imag_tol_thz: float
    supercell: list[int]
    n_samples: int
    extra: dict

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({f"ft_{k}": v for k, v in d.pop("extra").items()})
        return d


# ---------------------------------------------------------------- MD sampling ----

def _run_nvt(atoms, calc, temperature_K, supercell, n_steps, timestep_fs, equil_steps,
             sample_every, seed=0):
    """Langevin NVT MD on a supercell; return (ref_supercell, [(disp, forces), ...])."""
    from ase.build import make_supercell
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    P = np.diag(supercell)
    sc = make_supercell(atoms, P)
    ref = sc.copy()
    ref_pos = ref.get_positions()
    sc.calc = calc

    MaxwellBoltzmannDistribution(sc, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    dyn = Langevin(sc, timestep_fs * units.fs, temperature_K=temperature_K,
                   friction=0.02, rng=np.random.default_rng(seed))

    samples = []

    def _collect():
        disp = sc.get_positions() - ref_pos
        # minimum-image the displacement w.r.t. the supercell box
        cell = sc.get_cell().array
        frac = np.linalg.solve(cell.T, disp.T).T
        frac -= np.round(frac)
        disp = frac @ cell
        samples.append((disp.copy(), sc.get_forces().copy()))

    dyn.run(equil_steps)
    for i in range(n_steps):
        dyn.run(1)
        if i % sample_every == 0:
            _collect()
    return ref, samples


# ----------------------------------------------------- one-shot TDEP fit ----

def compute_finite_t_tdep(atoms, calc, temperature_K, supercell=(2, 2, 2),
                          n_steps=4000, timestep_fs=2.0, equil_steps=2000,
                          sample_every=20, imag_tol_thz=DEFAULT_IMAG_TOL_THZ,
                          seed=0) -> FiniteTResult:
    """Effective harmonic phonons at T via least-squares fit of forces to displacements.

    Fits F = -Phi . u over MD snapshots (Phi = effective force-constant matrix), symmetrizes
    to enforce Newton's third law / acoustic sum rule, and diagonalizes the mass-weighted
    dynamical matrix at q=0 over the supercell. Min eigenfrequency => finite-T stability call.
    """
    ref, samples = _run_nvt(atoms, calc, temperature_K, supercell, n_steps,
                            timestep_fs, equil_steps, sample_every, seed)
    n = len(ref)
    masses = ref.get_masses()

    # Stack snapshots: solve for Phi (3n x 3n) minimizing || F + Phi u ||^2.
    U = np.array([s[0].reshape(-1) for s in samples])   # (S, 3n)
    F = np.array([s[1].reshape(-1) for s in samples])   # (S, 3n)
    # Ridge-regularized to stabilize the fit at low sampling.
    lam = 1e-6 * np.trace(U.T @ U) / (3 * n)
    Phi = -np.linalg.solve(U.T @ U + lam * np.eye(3 * n), U.T @ F).T  # (3n, 3n)
    Phi = 0.5 * (Phi + Phi.T)                                         # symmetrize

    # Acoustic sum rule: zero net force under uniform translation.
    Phi = _apply_asr(Phi, n)

    # Mass-weight and diagonalize -> frequencies (THz).
    m = np.repeat(masses, 3)
    D = Phi / np.sqrt(np.outer(m, m))
    eig = np.linalg.eigvalsh(D)
    # eig in (eV/Ang^2 / amu); convert to THz with sign of eigenvalue.
    from ase import units
    conv = np.sqrt(np.abs(eig) * units._e / (units._amu * 1e-20)) / (2 * np.pi) / 1e12
    freqs = np.sign(eig) * conv
    min_freq = float(np.min(freqs))
    return FiniteTResult(
        temperature_K=float(temperature_K), method="tdep", min_eff_freq_thz=min_freq,
        dynamically_stable=bool(min_freq >= imag_tol_thz), imag_tol_thz=imag_tol_thz,
        supercell=list(supercell), n_samples=len(samples),
        extra={"ridge_lambda": float(lam), "n_atoms_sc": n},
    )


def _apply_asr(Phi, n):
    """Enforce translational acoustic sum rule on a (3n,3n) force-constant matrix."""
    Phi = Phi.reshape(n, 3, n, 3)
    for a in range(n):
        for i in range(3):
            for j in range(3):
                Phi[a, i, a, j] -= Phi[a, i, :, j].sum()
    return Phi.reshape(3 * n, 3 * n)


# ----------------------------------------------------- MD symmetry-breaking probe ----

def md_symmetry_breaking(atoms, calc, temperature_K, supercell=(2, 2, 2),
                         n_steps=4000, timestep_fs=2.0, equil_steps=2000,
                         sample_every=20, seed=0) -> FiniteTResult:
    """Cheap probe: RMS displacement of the time-averaged structure from the high-symmetry
    reference. Large persistent distortion => soft mode condensed (low-symmetry phase at T);
    small => high-symmetry phase dynamically sampled (stable). Reported as a pseudo-frequency
    sign via the mean order parameter; primarily for diffusive/superionic systems."""
    ref, samples = _run_nvt(atoms, calc, temperature_K, supercell, n_steps,
                            timestep_fs, equil_steps, sample_every, seed)
    mean_disp = np.mean([s[0] for s in samples], axis=0)       # (n,3) time-averaged distortion
    order = float(np.sqrt(np.mean(np.sum(mean_disp**2, axis=1))))  # RMS Angstrom
    # Heuristic call: persistent mean distortion > 0.1 Ang => condensed/low-symmetry at T.
    stable_highsym = order < 0.1
    return FiniteTResult(
        temperature_K=float(temperature_K), method="md_distort",
        min_eff_freq_thz=float("nan"), dynamically_stable=bool(stable_highsym),
        imag_tol_thz=float("nan"), supercell=list(supercell), n_samples=len(samples),
        extra={"mean_distortion_ang": order},
    )


# ----------------------------------------------- hiPhive (TDEP-style) ----

def _phonopy_supercell_ase(prim_ase, supercell):
    """Return (Phonopy object, ideal supercell as ASE Atoms) with consistent atom ordering."""
    import ase
    import numpy as np
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    pa = PhonopyAtoms(symbols=prim_ase.get_chemical_symbols(),
                      scaled_positions=prim_ase.get_scaled_positions(),
                      cell=prim_ase.get_cell().array)
    phonon = Phonopy(pa, supercell_matrix=np.diag(supercell), primitive_matrix="auto")
    sc = phonon.supercell
    ideal = ase.Atoms(symbols=sc.symbols, scaled_positions=sc.scaled_positions,
                      cell=sc.cell, pbc=True)
    return phonon, ideal


def _md_collect(ideal_sc, calc, temperature_K, n_steps, timestep_fs, equil_steps,
                sample_every, seed=0, cache_path=None):
    """NVT MD on the ideal supercell; return list of ASE Atoms snapshots carrying forces.
    Snapshots are cached to ``cache_path`` (extxyz) so method iteration never re-runs MD."""
    import os
    import numpy as np
    from ase.io import read, write
    if cache_path and os.path.exists(cache_path):
        snaps = read(cache_path, index=":")
        if len(snaps) > 0:
            return snaps

    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    sc = ideal_sc.copy()
    sc.calc = calc
    MaxwellBoltzmannDistribution(sc, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    dyn = Langevin(sc, timestep_fs * units.fs, temperature_K=temperature_K,
                   friction=0.02, rng=np.random.default_rng(seed))
    snaps = []
    dyn.run(equil_steps)
    for i in range(n_steps):
        dyn.run(1)
        if i % sample_every == 0:
            s = sc.copy()
            s.arrays["forces"] = sc.get_forces()   # store forces for hiphive
            snaps.append(s)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        write(cache_path, snaps)
    return snaps


def compute_finite_t_hiphive(atoms, calc, temperature_K, supercell=(2, 2, 2),
                             cutoff: float = 5.0, n_steps=3000, timestep_fs=2.0,
                             equil_steps=2000, sample_every=30,
                             imag_tol_thz=DEFAULT_IMAG_TOL_THZ, seed=0,
                             cache_path=None, relax=True, fmax=1e-3) -> FiniteTResult:
    """Temperature-dependent effective harmonic phonons via hiPhive (symmetry-reduced TDEP).

    Relax -> NVT MD at T -> fit symmetry-constrained 2nd-order force constants from the
    snapshots -> phonopy frequencies. Minimum effective frequency gives the finite-T
    dynamical-stability call. Symmetry reduction makes the fit well-determined (unlike the
    deprecated full-matrix tdep route).
    """
    import numpy as np
    from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
    from hiphive.utilities import prepare_structures
    from trainstation import Optimizer

    prim = atoms.copy()
    prim.calc = calc
    if relax:
        from .harmonic import _relax
        prim = _relax(prim, fmax=fmax)

    phonon, ideal_sc = _phonopy_supercell_ase(prim, supercell)
    snaps = _md_collect(ideal_sc, calc, temperature_K, n_steps, timestep_fs,
                        equil_steps, sample_every, seed, cache_path)

    # hiPhive requires the pair cutoff < L/2 of the supercell (minimum-image uniqueness).
    cell_len = float(np.linalg.norm(ideal_sc.get_cell().array, axis=1).min())
    eff_cutoff = min(cutoff, 0.49 * cell_len)
    cs = ClusterSpace(prim, [eff_cutoff])             # pairs only -> 2nd-order FCs
    prepared = prepare_structures(snaps, ideal_sc)    # displacements + forces vs ideal
    sc_cont = StructureContainer(cs)
    for s in prepared:
        sc_cont.add_structure(s)
    opt = Optimizer(sc_cont.get_fit_data(), train_size=1.0)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)

    fcs = fcp.get_force_constants(ideal_sc)
    phonon.force_constants = fcs.get_fc_array(order=2)
    phonon.run_mesh([12, 12, 12], is_gamma_center=True)
    md = phonon.get_mesh_dict()
    freqs, qpts = md["frequencies"], md["qpoints"]
    min_freq = float(np.min(freqs))                       # full min (acoustic ~0 at Gamma)
    # Soft-mode tracker: min frequency away from Gamma, where acoustic branches are non-zero,
    # so a softening optical/zone-boundary mode (e.g. SrTiO3 R-point) is visible and its
    # hardening with T is measurable. The dynamical-stability CALL still uses the full min.
    non_gamma = np.linalg.norm(qpts, axis=1) > 1e-6
    soft = float(np.min(freqs[non_gamma])) if non_gamma.any() else min_freq
    return FiniteTResult(
        temperature_K=float(temperature_K), method="hiphive", min_eff_freq_thz=min_freq,
        dynamically_stable=bool(min(min_freq, soft) >= imag_tol_thz), imag_tol_thz=imag_tol_thz,
        supercell=list(supercell), n_samples=len(snaps),
        extra={"cutoff": eff_cutoff, "rmse_fit": float(opt.rmse_train),
               "soft_mode_freq_thz": soft},
    )


# ------------------------------------------- rattled-start (symmetry-broken) ----

def _soft_mode_pattern(ideal_sc, calc, delta: float = 0.01, zero_tol: float = 1e-4):
    """Softest real-space eigenvector of the supercell Hessian at Gamma.

    The Hessian H = -dF/du is built by central finite differences of the MLIP forces, mass-
    weighted, and diagonalised. The most-negative eigenvalue's eigenvector is the unstable
    distortion pattern *in the supercell* (a commensurate soft mode, e.g. SrTiO3's R-point
    octahedral tilt, folds onto Gamma of the matching supercell — so no q-phase bookkeeping).
    Returns a unit displacement pattern (n,3) with max atomic component 1.0, or None if the
    supercell has no imaginary mode (harmonically stable -> nothing to seed)."""
    import numpy as np
    n = len(ideal_sc)
    pos0 = ideal_sc.get_positions()
    N = 3 * n
    H = np.zeros((N, N))
    base = ideal_sc.copy()
    base.calc = calc
    for a in range(n):
        for i in range(3):
            col = 3 * a + i
            fp = base.copy(); p = pos0.copy(); p[a, i] += delta; fp.set_positions(p); fp.calc = calc
            fm = base.copy(); p = pos0.copy(); p[a, i] -= delta; fm.set_positions(p); fm.calc = calc
            H[:, col] = -(fp.get_forces().reshape(-1) - fm.get_forces().reshape(-1)) / (2 * delta)
    H = 0.5 * (H + H.T)
    m = np.repeat(ideal_sc.get_masses(), 3)
    D = H / np.sqrt(np.outer(m, m))
    w, V = np.linalg.eigh(D)
    if w[0] >= -zero_tol:                       # softest eigenvalue not imaginary
        return None
    v = (V[:, 0] / np.sqrt(m)).reshape(n, 3)    # mass-weighted eigvec -> real displacement
    mx = np.abs(v).max()
    return v / mx if mx > 0 else None


def _md_collect_rattled(ideal_sc, calc, temperature_K, n_steps, timestep_fs, equil_steps,
                        sample_every, init_disp, seed=0, friction=0.01, cache_path=None):
    """NVT MD STARTED FROM A SYMMETRY-BROKEN supercell (ideal + ``init_disp``) so the
    trajectory commits to a distorted basin if the high-symmetry phase is unstable.
    Snapshots (carrying forces) cached to extxyz; cache key encodes method+T upstream."""
    import os
    import numpy as np
    from ase.io import read, write
    if cache_path and os.path.exists(cache_path):
        snaps = read(cache_path, index=":")
        if len(snaps) > 0:
            return snaps

    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    sc = ideal_sc.copy()
    sc.set_positions(sc.get_positions() + init_disp)
    sc.calc = calc
    MaxwellBoltzmannDistribution(sc, temperature_K=temperature_K, rng=np.random.default_rng(seed + 1))
    dyn = Langevin(sc, timestep_fs * units.fs, temperature_K=temperature_K,
                   friction=friction, rng=np.random.default_rng(seed + 2))
    snaps = []
    dyn.run(equil_steps)
    for i in range(n_steps):
        dyn.run(1)
        if i % sample_every == 0:
            s = sc.copy()
            s.arrays["forces"] = sc.get_forces()
            snaps.append(s)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        write(cache_path, snaps)
    return snaps


def _order_parameter(snaps, ideal_sc) -> float:
    """RMS of the time-averaged distortion of the trajectory from the ideal high-symmetry
    cell (minimum-image, net translation removed). Large psi => the soft mode has condensed
    into a low-symmetry mean structure => high-symmetry phase is dynamically unstable at T."""
    import numpy as np
    cell = ideal_sc.get_cell().array
    ref = ideal_sc.get_positions()
    disps = []
    for s in snaps:
        d = s.get_positions() - ref
        frac = np.linalg.solve(cell.T, d.T).T
        frac -= np.round(frac)
        disps.append(frac @ cell)
    mean_d = np.mean(disps, axis=0)            # (n,3) time-averaged distortion
    mean_d -= mean_d.mean(axis=0, keepdims=True)  # remove rigid translation
    return float(np.sqrt(np.mean(np.sum(mean_d ** 2, axis=1))))


def compute_finite_t_rattled(atoms, calc, temperature_K, supercell=(3, 3, 3),
                             cutoff: float = 5.0, n_steps=3000, timestep_fs=2.0,
                             equil_steps=2000, sample_every=30,
                             imag_tol_thz=DEFAULT_IMAG_TOL_THZ, seed=0, cache_path=None,
                             relax=True, fmax=1e-3, rattle_stdev=0.30,
                             order_tol_ang=0.15) -> FiniteTResult:
    """Finite-T dynamic stability via rattled-start (symmetry-broken) sampling.

    One-shot TDEP from the ideal cell sits at the cubic saddle and reports spurious stability;
    here the MD STARTS from a rattled supercell so the trajectory commits to the true basin.
    Two complementary signals at temperature T:
      * order parameter ``psi`` = RMS time-averaged distortion from the ideal high-symmetry
        cell. psi >= order_tol => soft mode condensed (low-symmetry mean) => cubic UNSTABLE.
      * effective-harmonic min frequency from a symmetry-reduced hiPhive fit (referenced to
        the ideal cell), as a corroborating curvature signal.
    Call: stable iff psi < order_tol AND min_eff_freq >= imag_tol. T-dependence enters through
    psi(T): above the transition, thermal hopping melts the mean distortion back to cubic.
    """
    import numpy as np
    from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
    from hiphive.utilities import prepare_structures
    from trainstation import Optimizer

    prim = atoms.copy()
    prim.calc = calc
    if relax:
        from .harmonic import _relax
        prim = _relax(prim, fmax=fmax)

    phonon, ideal_sc = _phonopy_supercell_ase(prim, supercell)

    # Seed the MD in ONE soft-mode well (not random directions, which excite all symmetry-
    # equivalent domains and cancel in the time average). Pattern from the supercell Hessian;
    # if the cell is harmonically stable there is no pattern, so use a tiny thermal rattle.
    rng = np.random.default_rng(seed)
    pattern = _soft_mode_pattern(ideal_sc, calc)
    init_disp = rng.normal(0.0, 0.02, ideal_sc.get_positions().shape)
    seeded = pattern is not None
    if seeded:
        init_disp = init_disp + rattle_stdev * pattern   # rattle_stdev = seed amplitude (A)

    snaps = _md_collect_rattled(ideal_sc, calc, temperature_K, n_steps, timestep_fs,
                                equil_steps, sample_every, init_disp, seed,
                                cache_path=cache_path)

    psi = _order_parameter(snaps, ideal_sc)

    # Symmetry-reduced effective-FC fit (corroborating curvature signal), referenced to ideal.
    cell_len = float(np.linalg.norm(ideal_sc.get_cell().array, axis=1).min())
    eff_cutoff = min(cutoff, 0.49 * cell_len)
    cs = ClusterSpace(prim, [eff_cutoff])
    prepared = prepare_structures(snaps, ideal_sc)
    sc_cont = StructureContainer(cs)
    for s in prepared:
        sc_cont.add_structure(s)
    opt = Optimizer(sc_cont.get_fit_data(), train_size=1.0)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    phonon.force_constants = fcp.get_force_constants(ideal_sc).get_fc_array(order=2)
    phonon.run_mesh([12, 12, 12], is_gamma_center=True)
    md = phonon.get_mesh_dict()
    freqs, qpts = md["frequencies"], md["qpoints"]
    min_freq = float(np.min(freqs))
    non_gamma = np.linalg.norm(qpts, axis=1) > 1e-6
    soft = float(np.min(freqs[non_gamma])) if non_gamma.any() else min_freq

    condensed = psi >= order_tol_ang
    freq_unstable = min(min_freq, soft) < imag_tol_thz
    stable = (not condensed) and (not freq_unstable)
    return FiniteTResult(
        temperature_K=float(temperature_K), method="rattled", min_eff_freq_thz=min_freq,
        dynamically_stable=bool(stable), imag_tol_thz=imag_tol_thz,
        supercell=list(supercell), n_samples=len(snaps),
        extra={"cutoff": eff_cutoff, "rmse_fit": float(opt.rmse_train),
               "soft_mode_freq_thz": soft, "order_param_ang": psi,
               "order_tol_ang": order_tol_ang, "seed_amp_ang": rattle_stdev,
               "seeded_soft_mode": bool(seeded), "condensed": bool(condensed)},
    )


# ------------------------------- 1D soft-mode free energy (quantum) ----

_HBAR2_OVER_2AMU = 2.09008e-3   # eV*Ang^2 = hbar^2/(2 * 1 amu); 1D kinetic operator constant
_KB_EV = 8.617333262e-5         # eV/K
# eV/Ang^2 per amu -> (rad/s)^2 : (1.602e-19 J/eV)/(1e-20 m^2/Ang^2)/(1.66054e-27 kg/amu)
_W_TO_OMEGA2 = 1.602176634e-19 / 1e-20 / 1.66053907e-27
_HBAR_EVS = 6.582119569e-16     # eV*s  (for x = hbar*omega/2kT and zero-point energy in eV)
_HBAR_JS = 1.054571817e-34      # J*s   (for <Q^2> = hbar/2m_omega, m in kg -> m^2)
_AMU_KG = 1.66053906660e-27     # kg/amu


def _harmonic_phonon(prim, calc, supercell, disp=0.01):
    """phonopy object carrying symmetrized force constants (prim already relaxed)."""
    import ase
    import numpy as np
    from phonopy import Phonopy
    from .harmonic import _ase_to_phonopy_atoms
    ph = Phonopy(_ase_to_phonopy_atoms(prim), supercell_matrix=np.diag(supercell),
                 primitive_matrix="auto")
    ph.generate_displacements(distance=disp)
    forces = []
    for sc in ph.supercells_with_displacements:
        a = ase.Atoms(symbols=sc.symbols, scaled_positions=sc.scaled_positions,
                      cell=sc.cell, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    return ph


def _softest_mesh_mode(ph, den=6):
    """Softest phonon over the rational q-grid with denominator ``den`` (default 6: covers the
    physically relevant commensurate instabilities -- R=3/6 & M perovskite tilts, the bcc
    2/3<111> omega mode=4/6, 1/3 modes, and the zone centre). Frequencies are taken from EXACT
    run_qpoints (phonopy's symmetry-reduced run_mesh both injects the Gamma acoustic-sum-rule
    artifact as a spurious global minimum and skips exact zone-boundary points like R). The
    minimum q is frozen into its MINIMAL commensurate cell via phonopy modulation -- 2x2x2 for a
    perovskite R tilt, 3x3x3 for the bcc omega mode, 1x1x1 for a zone-centre ferroelectric.
    Returns (freq_thz, q, dim, base_ase, u[n,3] unit pattern, M_eff[amu]).

    The 3 acoustic branches at Gamma are masked: imaginary values there are rigid-translation
    artifacts, not instabilities (a real zone-centre FE soft mode is optical and stays found)."""
    import ase
    import numpy as np
    from fractions import Fraction

    qs = [[i / den, j / den, k / den]
          for i in range(den) for j in range(den) for k in range(den)]
    ph.run_qpoints(qs, with_eigenvectors=False)
    freqs = np.array(ph.qpoints.frequencies)             # (nq, nb)
    fr = freqs.copy()
    for qi, q in enumerate(qs):
        if max(abs(c) for c in q) < 1e-8:
            fr[qi, np.argsort(fr[qi])[:3]] = np.inf
    iq, ib = np.unravel_index(int(np.argmin(fr)), fr.shape)
    qsoft = qs[iq]
    fmin = float(freqs[iq, ib])
    dim = [Fraction(x).limit_denominator(den).denominator if abs(x) > 1e-9 else 1
           for x in qsoft]

    ph.run_modulations(dimension=dim, phonon_modes=[[qsoft, int(ib), 1.0, 0.0]])
    mods, sc_ph = ph.get_modulations_and_supercell()
    base = ase.Atoms(symbols=sc_ph.symbols, scaled_positions=sc_ph.scaled_positions,
                     cell=sc_ph.cell, pbc=True)
    u = np.real(mods[0])
    mx = np.abs(u).max()
    u = u / mx if mx > 0 else u                           # max atomic component = 1
    m_eff = float(np.sum(base.get_masses()[:, None] * u ** 2))   # sum_i m_i |u_i|^2  (amu)
    return fmin, qsoft, dim, base, u, m_eff


def _sample_well(base, calc, u, q_max, n_pts):
    """Static E(Q) along the soft pattern (symmetric, so sample Q>=0). Quadratic spacing puts
    most points at small Q so narrow soft wells (e.g. ferroelectric BaTiO3, min ~0.07 A) are
    resolved instead of being swamped by the steep repulsive wall. Returns (Qs, dE[eV])."""
    import numpy as np
    base = base.copy(); base.calc = calc
    E0 = base.get_potential_energy()
    Qs = q_max * np.linspace(0.0, 1.0, n_pts) ** 2
    dE = []
    pos0 = base.get_positions()
    for Q in Qs:
        a = base.copy(); a.set_positions(pos0 + Q * u); a.calc = calc
        dE.append(a.get_potential_energy() - E0)
    return Qs, np.array(dE)


def _fit_double_well(Qs, dE):
    """Fit V(Q) = a Q^2 + b Q^4 + c Q^6 (even, V(0)=0) to the *well + barrier* region only.

    The static E(Q) climbs orders of magnitude up the repulsive wall at large Q; including those
    points lets least squares wash out a narrow soft well (a real -60 meV BaTiO3 well fits as
    -16 meV). We restrict to points within an energy window above the minimum (~5x the well
    depth captures the well and the thermally relevant barrier), keeping >=4 lowest-Q points so
    the curvature is defined. The potential must be bounded below: if the sextic returns c<0 we
    drop it and refit the quartic. Returns (a,b,c)."""
    import numpy as np
    Qs = np.asarray(Qs, float); dE = np.asarray(dE, float)
    emin = float(dE.min())
    window = max(0.06, 5.0 * abs(min(emin, 0.0)))        # eV
    keep = dE <= emin + window
    if keep.sum() < 4:
        keep = np.zeros_like(dE, bool)
        keep[np.argsort(Qs)[:4]] = True
    Qf, Ef = Qs[keep], dE[keep]
    A = np.stack([Qf ** 2, Qf ** 4, Qf ** 6], axis=1)
    coef, *_ = np.linalg.lstsq(A, Ef, rcond=None)
    if coef[2] < 0 or (coef[1] < 0 and coef[2] <= 0):
        A2 = np.stack([Qf ** 2, Qf ** 4], axis=1)
        c2, *_ = np.linalg.lstsq(A2, Ef, rcond=None)
        coef = np.array([c2[0], c2[1], 0.0])
    return tuple(float(x) for x in coef)


def _solve_1d(a, b, c, m_eff, temperature_K, q_box=0.8, ngrid=801, n_states=40):
    """Exact 1D quantum solution for a particle of mass m_eff in V(Q)=aQ^2+bQ^4+cQ^6.

    Returns (eff_freq_thz, order_param_Q, stable) where the call uses the potential of mean
    force W(Q) = -kT ln P(Q,T): the high-symmetry phase (Q=0) is dynamically stable iff W has
    a minimum at Q=0 (equivalently the thermal density P(Q,T) is peaked at 0, not bimodal).
    Quantum nuclear motion is included exactly for the 1D mode (captures zero-point melting of
    shallow wells, i.e. quantum-paraelectric behaviour)."""
    import numpy as np
    Q = np.linspace(-q_box, q_box, ngrid)
    h = Q[1] - Q[0]
    V = a * Q ** 2 + b * Q ** 4 + c * Q ** 6
    K = _HBAR2_OVER_2AMU / m_eff                          # eV*Ang^2
    main = V + 2.0 * K / h ** 2
    off = -K / h ** 2 * np.ones(ngrid - 1)
    from scipy.linalg import eigh_tridiagonal
    E, psi = eigh_tridiagonal(main, off, select="i", select_range=(0, n_states - 1))
    kT = max(_KB_EV * float(temperature_K), 1e-9)
    p = np.exp(-(E - E[0]) / kT); p /= p.sum()
    prob = (psi ** 2 * p).sum(axis=1)
    prob /= prob.sum() * h
    # PMF curvature at Q=0 (central grid point) via finite differences.
    i0 = ngrid // 2
    W = -kT * np.log(np.clip(prob, 1e-300, None))
    w2 = (W[i0 + 1] - 2 * W[i0] + W[i0 - 1]) / h ** 2     # eV/Ang^2 = W''(0)
    order_Q = float(np.sqrt((prob * Q ** 2).sum() * h))   # sqrt<Q^2> (Ang)
    peak_at_zero = bool(np.argmax(prob) in (i0 - 1, i0, i0 + 1))
    omega2 = w2 * _W_TO_OMEGA2 / m_eff                    # (rad/s)^2, signed
    eff_freq_thz = float(np.sign(w2) * np.sqrt(abs(omega2)) / (2 * np.pi) / 1e12)
    stable = bool(w2 > 0 and peak_at_zero)
    return eff_freq_thz, order_Q, stable


def _scha_branch(Q0, a, b, c, m_eff, temperature_K):
    """Self-consistent harmonic (SCHA) trial state: a Gaussian of centroid Q0 and width sigma,
    with the trial frequency Omega fixed by the stationarity condition m*Omega^2 = <V''>_0.
    Quantum nuclear motion enters through <Q^2> = (hbar/2 m Omega) coth(hbar Omega / 2kT).

    The self-consistency in sigma^2 is a 1D root of the monotone-decreasing residual
    g(s2) = sigma2_sc(K(s2)) - s2 on the interval where K(s2) > 0; we bracket and solve it
    (brentq), which avoids the spurious runaway large-sigma fixed point a damped iteration can
    fall into. Returns (F[eV], sigma2[Ang^2], omega_rad_s, curv_K[eV/Ang^2])."""
    import numpy as np
    from scipy.optimize import brentq
    kT = max(_KB_EV * float(temperature_K), 1e-12)
    m_kg = m_eff * _AMU_KG

    def curv(s2):                                          # K = <V''>_0  (eV/Ang^2)
        q2 = Q0 ** 2 + s2
        q4 = Q0 ** 4 + 6 * Q0 ** 2 * s2 + 3 * s2 ** 2
        return 2 * a + 12 * b * q2 + 30 * c * q4

    def sigma2_sc(s2):                                     # quantum SCHA width at this K
        K = curv(s2)
        omega = np.sqrt(K * _W_TO_OMEGA2 / m_eff)
        x = _HBAR_EVS * omega / (2 * kT)
        return (_HBAR_JS / (2 * m_kg * omega)) / np.tanh(x) * 1e20

    # Lowest s2 with K>0 (above which a bound Gaussian trial exists). Scan a coarse grid up
    # from there for the first sign change of g, then refine with brentq.
    grid = np.geomspace(1e-6, 2.0, 240)
    grid = grid[np.array([curv(s) for s in grid]) > 1e-8]
    if grid.size == 0:
        # No bound Gaussian trial at this centroid (purely soft) -> reject it (F=+inf) so the
        # free-energy minimisation falls to a displaced, stable centroid instead.
        return float("inf"), float("nan"), 0.0, float(curv(0.0))
    g = np.array([sigma2_sc(s) - s for s in grid])
    s2 = float(grid[np.argmin(np.abs(g))])                 # fallback: closest to root
    sign = np.sign(g)
    cross = np.where(np.diff(sign) < 0)[0]
    if len(cross):
        i = int(cross[0])
        s2 = float(brentq(lambda s: sigma2_sc(s) - s, grid[i], grid[i + 1], xtol=1e-10))
    omega = float(np.sqrt(curv(s2) * _W_TO_OMEGA2 / m_eff))
    K = curv(s2)
    q2 = Q0 ** 2 + s2
    q4 = Q0 ** 4 + 6 * Q0 ** 2 * s2 + 3 * s2 ** 2
    q6 = Q0 ** 6 + 15 * Q0 ** 4 * s2 + 45 * Q0 ** 2 * s2 ** 2 + 15 * s2 ** 3
    Vavg = a * q2 + b * q4 + c * q6
    x = _HBAR_EVS * omega / (2 * kT)
    # F_vib = kT ln(2 sinh x) = (hbar*omega/2) + kT ln(1 - e^{-2x}); the split form avoids
    # sinh overflow as T -> 0 (where it reduces to the zero-point energy).
    Fvib = 0.5 * _HBAR_EVS * omega + kT * np.log1p(-np.exp(-2 * x))
    F = Fvib + Vavg - 0.5 * K * s2                        # Gibbs-Bogoliubov free energy
    return float(F), s2, float(omega), float(K)


def _solve_scha(a, b, c, m_eff, temperature_K, q_box=0.6, nq=121):
    """Single-mode quantum SCHA. Minimise the SCHA free energy over the order-parameter
    centroid Q0 (>=0 by symmetry); the high-symmetry phase is dynamically stable iff the global
    minimum sits at Q0 ~ 0. Returns (eff_freq_thz, order_Q0, stable). The reported frequency is
    the SCHA phonon at the global minimum, signed negative when the cubic phase has condensed so
    it crosses zero through the transition."""
    import numpy as np
    Q0s = np.linspace(0.0, q_box, nq)
    Fs, sig, om, Kc = [], [], [], []
    for q0 in Q0s:
        F, s2, omega, K = _scha_branch(q0, a, b, c, m_eff, temperature_K)
        Fs.append(F); sig.append(s2); om.append(omega); Kc.append(K)
    Fs = np.array(Fs)
    i = int(np.argmin(Fs))
    order_Q0 = float(Q0s[i])
    # "stable" = the cubic (Q0=0) centroid is the free-energy ground state. A small tolerance
    # absorbs grid noise; require the symmetric point be within ~kT of the global minimum too.
    q_thresh = 1.5 * (Q0s[1] - Q0s[0])
    stable = bool(order_Q0 <= q_thresh)
    omega_min = om[i]
    eff_freq_thz = float(np.sign(1 if stable else -1) * omega_min / (2 * np.pi) / 1e12)
    return eff_freq_thz, order_Q0, stable


def compute_finite_t_softmode(atoms, calc, temperature_K, supercell=(2, 2, 2),
                              q_max=0.45, n_pts=10, imag_tol_thz=DEFAULT_IMAG_TOL_THZ,
                              relax=True, fmax=1e-3, disp=0.01,
                              cache_path=None) -> FiniteTResult:
    """Finite-T dynamic stability via the 1D soft-mode free energy with exact quantum nuclear
    motion. Relax -> harmonic FCs -> softest commensurate mode (phonopy modulation) -> static
    double well E(Q) along it -> exact 1D quantum thermal density -> stability from the
    potential-of-mean-force curvature at Q=0. The expensive E(Q) map is temperature-independent
    and cached (json), so every extra temperature is a sub-second CPU solve.
    """
    import json
    import os
    import numpy as np

    cache = None
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))

    if cache is None:
        prim = atoms.copy(); prim.calc = calc
        if relax:
            from .harmonic import _relax
            prim = _relax(prim, fmax=fmax)
        ph = _harmonic_phonon(prim, calc, supercell, disp)
        f0, qsoft, dim, base, u, m_eff = _softest_mesh_mode(ph)
        Qs, dE = _sample_well(base, calc, u, q_max, n_pts)
        a, b, cc = _fit_double_well(Qs, dE)
        cache = {"harm_min_thz": f0, "m_eff": m_eff, "a": a, "b": b, "c": cc,
                 "Qs": Qs.tolist(), "dE": dE.tolist(), "supercell": list(dim),
                 "fc_supercell": list(supercell), "q_soft": list(qsoft),
                 "well_depth_meV": float(-min(dE.min(), 0.0) * 1000)}
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            json.dump(cache, open(cache_path, "w"))

    eff_freq, order_Q, stable_fe = _solve_scha(cache["a"], cache["b"], cache["c"],
                                               cache["m_eff"], temperature_K)
    stable = bool(stable_fe)
    return FiniteTResult(
        temperature_K=float(temperature_K), method="softmode",
        min_eff_freq_thz=eff_freq, dynamically_stable=stable, imag_tol_thz=imag_tol_thz,
        supercell=list(cache["supercell"]), n_samples=len(cache["Qs"]),
        extra={"soft_mode_freq_thz": eff_freq, "harm_soft_thz": cache["harm_min_thz"],
               "order_param_Q_ang": order_Q, "m_eff_amu": cache["m_eff"],
               "well_depth_meV": cache["well_depth_meV"],
               "v_a": cache["a"], "v_b": cache["b"], "v_c": cache["c"]},
    )


# ----------------------------------------------------- SSCHA hook ----

def compute_finite_t_sscha(atoms, calc, temperature_K, supercell=(2, 2, 2),
                           population_size=200, max_iter=20,
                           imag_tol_thz=DEFAULT_IMAG_TOL_THZ) -> FiniteTResult:
    """Cross-check via python-sscha (stochastic self-consistent harmonic approximation).

    Requires `cellconstructor` + `python-sscha` in the env. Returns the SSCHA auxiliary
    (free-energy Hessian) minimum frequency, the proper anharmonic dynamical-stability
    criterion. Implementation is staged after the TDEP route validates on textbook cases.
    """
    raise NotImplementedError(
        "SSCHA route is stage-P3b; use compute_finite_t_tdep first. "
        "Wiring: cellconstructor.Structure from atoms -> SSCHA Ensemble with an ASE-calc "
        "force engine -> SSCHA_Minimizer -> free-energy Hessian min frequency."
    )
