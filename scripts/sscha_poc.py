"""SSCHA proof-of-concept: harmonic FCs (phonopy) -> cellconstructor dyn -> SSCHA relaxation
with an MLIP ASE calculator -> free-energy Hessian -> dynamic-stability frequency vs T.

Usage: python scripts/sscha_poc.py <model> <system> [supercell] [Tlist comma]
Validates the harness: SrTiO3 should harden (imaginary -> real) across its transition.
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.phonons import Phonons as ASEPhonons
import cellconstructor as CC
import cellconstructor.Phonons
import sscha
import sscha.Ensemble
import sscha.SchaMinimizer
import sscha.Relax
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.harmonic import _relax

model = sys.argv[1] if len(sys.argv) > 1 else "mace_mp0"
sysid = sys.argv[2] if len(sys.argv) > 2 else "srtio3_cubic"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
Ts = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["100", "300", "600"])]
sc = (n, n, n)
N_CONF, MAX_POP = 200, 8

calc = get_calculator(model, device="cuda").calc
prim = build_atoms(get_spec(sysid)); prim.calc = calc
prim = _relax(prim, fmax=1e-3)

# Harmonic dynamical matrix via ASE finite-displacement phonons (displaces only unit-cell
# atoms -> cheap), converted natively to a cellconstructor dyn.
aph = ASEPhonons(prim, calc, supercell=sc, delta=0.03, name="/root/_ase_ph")
aph.clean()
aph.run()
aph.read(acoustic=True)
dyn = CC.Phonons.get_dyn_from_ase_phonons(aph)
dyn.ForcePositiveDefinite()
dyn.Symmetrize()
print("CC dyn ready; supercell", dyn.GetSupercell(), flush=True)

RY_TO_THZ = CC.Units.RY_TO_CM * 2.99792458e-2  # Ry-freq -> cm^-1 -> THz (1 cm^-1 = 0.02998 THz)

print(f"\n=== SSCHA {model} {sysid} sc={sc} ===", flush=True)
for T in Ts:
    ens = sscha.Ensemble.Ensemble(dyn.Copy(), T, supercell=dyn.GetSupercell())
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ens)
    relax = sscha.Relax.SSCHA(minim, ase_calculator=calc, N_configs=N_CONF, max_pop=MAX_POP,
                              save_ensemble=False)
    relax.relax(get_stress=False)
    hess = relax.minim.ensemble.get_free_energy_hessian(include_v4=False)
    w, _ = hess.DiagonalizeSupercell()
    wmin = float(np.min(w))
    thz = np.sign(wmin) * np.sqrt(abs(wmin)) * RY_TO_THZ if False else wmin * RY_TO_THZ
    print(f"T={T:6.0f}  min_hessian_freq={thz:+8.3f} THz  stable={wmin > -1e-9}", flush=True)
print("POC_DONE", flush=True)
