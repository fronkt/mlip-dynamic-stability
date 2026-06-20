"""Layer 1: harmonic (0 K) phonons via phonopy finite displacements + an MLIP calculator.

Returns the minimum phonon frequency over a q-mesh and a dynamical-stability call. This is
the *credibility-anchor* layer: it should reproduce published harmonic stability rates
(npj Comput Mater 2025) on shared materials, validating the harness before the novel
finite-T layer is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np

from . import DEFAULT_IMAG_TOL_THZ


@dataclass
class HarmonicResult:
    min_freq_thz: float          # most-negative frequency over the mesh (THz; <0 => imaginary)
    n_imag_modes: int            # count of imaginary branches below tol on the mesh
    dynamically_stable: bool     # min_freq_thz >= imag_tol_thz
    imag_tol_thz: float
    supercell: list[int]
    mesh: list[int]
    n_atoms_prim: int

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _ase_to_phonopy_atoms(atoms):
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        scaled_positions=atoms.get_scaled_positions(),
                        cell=atoms.get_cell().array)


def compute_harmonic(atoms, calc, supercell=(2, 2, 2), mesh=(12, 12, 12),
                     disp: float = 0.01, imag_tol_thz: float = DEFAULT_IMAG_TOL_THZ,
                     relax: bool = True, fmax: float = 1e-3) -> HarmonicResult:
    """Finite-displacement harmonic phonons of ``atoms`` using ASE ``calc`` for forces.

    By default the cell is first relaxed to the MLIP's own minimum (cell + positions, with
    symmetry preserved via the high-symmetry starting point). Phonons are then evaluated at
    that minimum — the physically correct place to ask about dynamical stability for *that*
    potential.
    """
    from phonopy import Phonopy
    import ase

    work = atoms.copy()
    work.calc = calc
    if relax:
        work = _relax(work, fmax=fmax)

    ph = Phonopy(_ase_to_phonopy_atoms(work),
                 supercell_matrix=np.diag(supercell),
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

    ph.run_mesh(mesh, with_eigenvectors=False)
    freqs = ph.get_mesh_dict()["frequencies"]  # (nq, nbands) THz
    min_freq = float(np.min(freqs))
    n_imag = int(np.sum(freqs < imag_tol_thz))
    return HarmonicResult(
        min_freq_thz=min_freq, n_imag_modes=n_imag,
        dynamically_stable=bool(min_freq >= imag_tol_thz),
        imag_tol_thz=imag_tol_thz, supercell=list(supercell), mesh=list(mesh),
        n_atoms_prim=len(ph.primitive),
    )


def _relax(atoms, fmax: float = 1e-3, steps: int = 300):
    from ase.constraints import ExpCellFilter
    from ase.optimize import FIRE
    ecf = ExpCellFilter(atoms)
    FIRE(ecf, logfile=None).run(fmax=fmax, steps=steps)
    return atoms
