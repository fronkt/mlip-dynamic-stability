"""Per-env smoke test: build si_diamond, get forces with each checkpoint this env serves.
Also pulls the checkpoints so later stages never hit the network. Never touches the ledger."""
import sys
import time

from mlip_dynstab.systems import get_spec, build_atoms
from mlip_dynstab.calculators import get_calculator

model = sys.argv[1]
ckpts = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else [None]
atoms = build_atoms(get_spec("si_diamond")).repeat((2, 2, 2))
atoms.rattle(0.02, seed=0)
for ck in ckpts:
    t = time.time()
    kw = {} if ck is None else {"model": ck}
    h = get_calculator(model, device="cuda", **kw)
    a = atoms.copy()
    a.calc = h.calc
    f = a.get_forces()
    e = a.get_potential_energy()
    print(f"SMOKE {model} ckpt={ck} version={h.version} E={e:.5f} |F|max={abs(f).max():.4f} "
          f"dtype={getattr(f, 'dtype', None)} t={time.time() - t:.1f}s", flush=True)
