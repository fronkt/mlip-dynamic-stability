"""Why does _softest_commensurate_mode pick a convex mode? Print the relaxed structure's
commensurate-q frequency table, the (q,band) argmin lands on, and the E(Q) along it vs along
the explicit R-point band-0 mode."""
import numpy as np
import ase
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.finite_t import _harmonic_phonon, _softest_commensurate_mode, _sample_well
from mlip_dynstab.harmonic import _relax

calc = get_calculator("mace_mp0", device="cuda").calc
prim = build_atoms(get_spec("srtio3_cubic")); prim.calc = calc
prim = _relax(prim, fmax=1e-3)
print("relaxed cell diag (A):", np.round(np.diag(prim.get_cell()), 4), flush=True)

ph = _harmonic_phonon(prim, calc, (2, 2, 2), 0.01)
n = [2, 2, 2]
qs = [[i / n[0], j / n[1], k / n[2]] for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]
ph.run_qpoints(qs, with_eigenvectors=True)
freqs = ph.qpoints.frequencies
iq, ib = np.unravel_index(int(np.argmin(freqs)), freqs.shape)
print("min over all commensurate q -> q=", qs[iq], "band=", ib, "f=", round(float(freqs[iq, ib]), 3), flush=True)
for qi, q in enumerate(qs):
    print(f"  q={q}  min_band_freq={float(freqs[qi].min()):+7.3f}", flush=True)

fmin, base, u, m_eff = _softest_commensurate_mode(ph, (2, 2, 2))
print(f"\n_softest pick: fmin={fmin:.3f} m_eff={m_eff:.1f} natoms={len(base)} max|u|={np.abs(u).max():.3f}", flush=True)
Qs, dE = _sample_well(base, calc, u, 0.30, 7)
nfu = len(base) // 5
print("E(Q) along _softest pick (meV/fu):", flush=True)
for q, e in zip(Qs, dE):
    print(f"  Q={q:.3f}  dE={e/nfu*1000:+9.3f}", flush=True)
print("DIAG_DONE", flush=True)
