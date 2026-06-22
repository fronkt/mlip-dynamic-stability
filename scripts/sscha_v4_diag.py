"""Bounded diagnostic: is the SSCHA false-stable on FE perovskites a cheap include_v4 fix, or a
deep problem (the auxiliary dyn stuck at the ForcePositiveDefinite start)?

Runs the full SSCHA pipeline once on cubic BaTiO3 (mace, T=100), then reports three minimum
frequencies on the SAME converged state:
  - harmonic (start, before ForcePositiveDefinite)  -> should be deeply imaginary (FE soft mode)
  - converged auxiliary dyn (the SCHA trial matrix)  -> did the relaxation recover softness?
  - free-energy Hessian, include_v4=False vs True     -> does v4 recover the instability?

If include_v4=True turns the Hessian imaginary, it is a one-line fix. If both stay positive while
the harmonic was deeply imaginary, the SCHA trial collapsed to the cubic min and SSCHA cannot
adjudicate this system as configured.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ase.phonons import Phonons as ASEPhonons
import cellconstructor as CC
import cellconstructor.Phonons
import sscha, sscha.Ensemble, sscha.SchaMinimizer, sscha.Relax

from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator

RY_TO_THZ = CC.Units.RY_TO_CM * 2.99792458e-2
T = 100.0
SC = (2, 2, 2)


def minfreq(dyn):
    w, _ = dyn.DiagonalizeSupercell()
    w = np.sort(np.asarray(w))
    nonac = w[3:] if w.size > 3 else w
    return float(nonac[0]) * RY_TO_THZ


spec = get_spec("batio3_cubic")
atoms = build_atoms(spec)
handle = get_calculator("mace_mp0", device="cuda")
calc = handle.calc

from mlip_dynstab.harmonic import _relax
prim = atoms.copy(); prim.calc = calc
prim = _relax(prim, fmax=1e-3); prim.calc = calc

aph = ASEPhonons(prim, calc, supercell=SC, delta=0.03, name="/tmp/_v4diag")
aph.clean(); aph.run(); aph.read(acoustic=True)
dyn0 = CC.Phonons.get_dyn_from_ase_phonons(aph)
print(f"harmonic (pre-FPD) min freq: {minfreq(dyn0):+.3f} THz")

dyn = dyn0.Copy(); dyn.ForcePositiveDefinite(); dyn.Symmetrize()
print(f"after ForcePositiveDefinite: {minfreq(dyn):+.3f} THz")

np.random.seed(0)
ens = sscha.Ensemble.Ensemble(dyn.Copy(), T, supercell=dyn.GetSupercell())
minim = sscha.SchaMinimizer.SSCHA_Minimizer(ens, root_representation="root2")
minim.min_step_dyn = 0.5; minim.meaningful_factor = 1e-4; minim.max_ka = 20
relaxer = sscha.Relax.SSCHA(minim, ase_calculator=calc, N_configs=256, max_pop=8, save_ensemble=False)
relaxer.relax(get_stress=False)
final_dyn = relaxer.minim.dyn
print(f"converged auxiliary dyn min freq: {minfreq(final_dyn):+.3f} THz")

he = sscha.Ensemble.Ensemble(final_dyn, T, supercell=final_dyn.GetSupercell())
he.generate(512); he.get_energy_forces(calc, compute_stress=False)
for v4 in (False, True):
    hess = he.get_free_energy_hessian(include_v4=v4)
    print(f"free-energy Hessian (include_v4={v4}): {minfreq(hess):+.3f} THz")
print("DIAG_DONE")
