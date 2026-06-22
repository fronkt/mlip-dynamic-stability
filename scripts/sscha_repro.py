"""m6 (referee report): SSCHA stochastic reproducibility. Run the same unit with several random
seeds and report the spread of the minimum free-energy-Hessian frequency, so the bcc cross-model
margins (which discriminate models at the ~0.4 THz level) can be shown to exceed the stochastic
noise. Default: bcc-Zr / MACE-MP-0 / 100 K, 4 seeds. Prints a one-line summary; does not touch
the ledger."""
import sys, numpy as np
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.finite_t import compute_finite_t_sscha

system = sys.argv[1] if len(sys.argv) > 1 else "zr_bcc"
model = sys.argv[2] if len(sys.argv) > 2 else "mace_mp0"
T = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3]

atoms = build_atoms(get_spec(system))
calc = get_calculator(model, device="cuda").calc
freqs = []
for s in seeds:
    r = compute_finite_t_sscha(atoms, calc, T, supercell=(2, 2, 2), seed=s)
    freqs.append(r.min_eff_freq_thz)
    print(f"[seed {s}] min_eff_freq = {r.min_eff_freq_thz:+.4f} THz  stable={r.dynamically_stable}")
f = np.array(freqs)
print(f"REPRO {system}/{model}/T={T}: mean={f.mean():+.4f} std={f.std():.4f} "
      f"range=[{f.min():+.4f},{f.max():+.4f}] THz  n={len(f)}")
