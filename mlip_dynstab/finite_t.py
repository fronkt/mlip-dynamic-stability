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
