import time, sys, warnings
warnings.filterwarnings("ignore")
from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator
from mlip_dynstab.finite_t import compute_finite_t_sscha

model = sys.argv[1] if len(sys.argv) > 1 else "mace_mp0"
sysid = sys.argv[2] if len(sys.argv) > 2 else "ti_bcc"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
T = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0

calc = get_calculator(model, device="cuda").calc
at = build_atoms(get_spec(sysid))
t0 = time.time()
r = compute_finite_t_sscha(at, calc, T, supercell=(n, n, n),
                           n_configs=200, max_pop=6, n_hessian=400)
print(f"RESULT {sysid} {n}x{n}x{n} T={T:.0f} min_freq={r.min_eff_freq_thz:+.3f} "
      f"stable={r.dynamically_stable} time={time.time()-t0:.0f}s "
      f"lowest6={[round(x,2) for x in r.extra['lowest6_thz']]}", flush=True)
print("TUNE_DONE", flush=True)
