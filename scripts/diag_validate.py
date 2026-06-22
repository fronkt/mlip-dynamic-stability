"""Validate the softmode method across well shapes: shallow (SrTiO3), deep narrow ferroelectric
(BaTiO3), a true control (MgO), and a deep bcc soft mode (Zr). Prints the per-T stability call
vs ground truth so the fit/SCHA fixes can be checked before the full grid."""
import sys
import numpy as np
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.finite_t import compute_finite_t_softmode

model = sys.argv[1] if len(sys.argv) > 1 else "mace_mp0"
systems = sys.argv[2:] if len(sys.argv) > 2 else ["srtio3_cubic", "batio3_cubic", "mgo_rocksalt", "zr_bcc"]
calc = get_calculator(model, device="cuda").calc
for sysid in systems:
    spec = get_spec(sysid); atoms = build_atoms(spec)
    cache = f"results/cache/softmode_{sysid}_{model}_sc222.json"
    print(f"\n=== {sysid}  Tc={spec.transition_T_K} ft_stable={spec.finite_T_stable} ===", flush=True)
    line = []
    for T in [100, 300, 600, 900]:
        r = compute_finite_t_softmode(atoms, calc, T, supercell=(2, 2, 2), cache_path=cache)
        gt = spec.finite_T_stable if spec.transition_T_K is None else (T >= spec.transition_T_K)
        ok = "ok" if r.dynamically_stable == gt else "XX"
        e = r.extra
        line.append(f"T={T}:{('S' if r.dynamically_stable else 'U')}/{('S' if gt else 'U')}({ok})")
    import json
    cc = json.load(open(cache))
    print(f"  q={cc.get('q_soft')} dim={cc.get('supercell')} harm={cc['harm_min_thz']:.2f} "
          f"well={cc['well_depth_meV']:.1f}meV  " + "  ".join(line), flush=True)
print("\nDIAG_DONE", flush=True)
