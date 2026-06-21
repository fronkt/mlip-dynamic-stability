"""Diagnostic: get SrTiO3's soft-mode eigenvector from PHONOPY (proven to give -1.83 THz at
R), build the real-space 2x2x2 modulation, and measure the static double well E(Q)."""
import numpy as np
import ase
from phonopy import Phonopy
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.harmonic import _relax, _ase_to_phonopy_atoms

handle = get_calculator('mace_mp0', device='cuda')
calc = handle.calc
prim = build_atoms(get_spec('srtio3_cubic')); prim.calc = calc
prim = _relax(prim)

ph = Phonopy(_ase_to_phonopy_atoms(prim), supercell_matrix=np.diag((2, 2, 2)),
             primitive_matrix="auto")
ph.generate_displacements(distance=0.01)
forces = []
for sc in ph.supercells_with_displacements:
    a = ase.Atoms(symbols=sc.symbols, scaled_positions=sc.scaled_positions, cell=sc.cell, pbc=True)
    a.calc = calc
    forces.append(a.get_forces())
ph.forces = np.array(forces)
ph.produce_force_constants(); ph.symmetrize_force_constants()

# frequencies + eigenvectors at R = (1/2,1/2,1/2)
ph.run_qpoints([[0.5, 0.5, 0.5]], with_eigenvectors=True)
qd = ph.get_qpoints_dict()
freqs = qd["frequencies"][0]              # THz (negative = imaginary)
print('R-point freqs (THz):', np.round(freqs, 3), flush=True)
imag = [i for i, f in enumerate(freqs) if f < -0.1]
print('imaginary band indices at R:', imag, flush=True)

# Build the real-space 2x2x2 modulation along the softest imaginary band.
band = int(np.argmin(freqs))
ph.run_modulations(dimension=[2, 2, 2], phonon_modes=[[[0.5, 0.5, 0.5], band, 1.0, 0.0]])
mods, sc_ph = ph.get_modulations_and_supercell()
u = np.real(mods[0])                       # (natom_sc, 3) displacement, max-normalised below
u = u / np.abs(u).max()
base = ase.Atoms(symbols=sc_ph.symbols, scaled_positions=sc_ph.scaled_positions,
                 cell=sc_ph.cell, pbc=True)
base.calc = calc
E0 = base.get_potential_energy()
nfu = len(base) // 5
print(f'supercell {len(base)} atoms ({nfu} f.u.); static E(Q) along phonopy R-mode '
      f'(meV per f.u., rel cubic):', flush=True)
for Q in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    a = base.copy(); a.set_positions(base.get_positions() + Q * u); a.calc = calc
    print(f'  Q={Q:.2f}A  dE={(a.get_potential_energy() - E0) / nfu * 1000:+8.2f} meV/fu', flush=True)
print('DIAG_DONE', flush=True)
