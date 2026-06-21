import numpy as np
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.finite_t import _phonopy_supercell_ase, _soft_mode_pattern
from mlip_dynstab.harmonic import _relax

handle = get_calculator('mace_mp0', device='cuda')
prim = build_atoms(get_spec('srtio3_cubic')); prim.calc = handle.calc
prim = _relax(prim)
phonon, ideal = _phonopy_supercell_ase(prim, (2, 2, 2))
pat = _soft_mode_pattern(ideal, handle.calc)
print('pattern found:', pat is not None, '| n_atoms', len(ideal), flush=True)
if pat is not None:
    ideal.calc = handle.calc
    E0 = ideal.get_potential_energy()
    print('static E(Q) double-well along soft mode (meV/supercell, rel to cubic):', flush=True)
    for Q in [-0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6]:
        a = ideal.copy(); a.set_positions(ideal.get_positions() + Q * pat); a.calc = handle.calc
        print(f'  Q={Q:+.2f}A  dE={(a.get_potential_energy() - E0) * 1000:+8.2f} meV', flush=True)
print('DIAG_DONE', flush=True)
