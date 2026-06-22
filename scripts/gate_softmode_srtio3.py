"""Verification gate: 1D quantum soft-mode free energy on SrTiO3 (MACE-MP-0).

The E(Q) double well is computed once (cached) then the 1D quantum thermal solve is run over a
temperature ladder. PASS criteria: (1) harmonic mode is imaginary (~-2 THz, R-point); (2) a
real double well is found (well_depth ~ few meV/f.u.); (3) the effective (PMF) frequency
hardens monotonically with T toward zero/positive — i.e. the cubic phase becomes dynamically
stable as T rises through the transition, not at 0 K. Numbers are printed for inspection.
"""
import numpy as np
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.finite_t import compute_finite_t_softmode

handle = get_calculator("mace_mp0", device="cuda")
atoms = build_atoms(get_spec("srtio3_cubic"))
cache = "results/cache/softmode_srtio3_cubic_mace_mp0_sc222.json"

print("T(K)   harm_thz  well_meV/fu  m_eff_amu   Q0_ang   eff_thz   stable", flush=True)
for T in [0.0, 25.0, 50.0, 100.0, 150.0, 300.0, 600.0]:
    r = compute_finite_t_softmode(atoms, handle.calc, T, supercell=(2, 2, 2),
                                  q_max=0.45, n_pts=10, cache_path=cache)
    e = r.extra
    nfu = len(atoms) // 5 if len(atoms) >= 5 else 1
    print(f"{T:6.0f} {e['harm_soft_thz']:8.3f} {e['well_depth_meV']:10.3f} "
          f"{e['m_eff_amu']:10.3f} {e['order_param_Q_ang']:8.4f} "
          f"{r.min_eff_freq_thz:8.3f}   {r.dynamically_stable}", flush=True)
print("GATE_DONE", flush=True)
