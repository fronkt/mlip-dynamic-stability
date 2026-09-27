#!/bin/bash
# Build /root/env-<model> from envs/lock-<model>-2026-08-17.txt (the pinned August environments).
# Conda-built '@ file://' lines are rewritten to '==<version>' when the wheel name carries one,
# otherwise dropped (they are base-image helpers: certifi, archspec, conda itself ...).
# '-e git+...mlip_dynamic_stability' lines are dropped; the harness is installed editable from
# the checked-out repo instead.
m=$1; py=$2
export PATH=/root/.local/bin:$PATH
REPO=/root/mlip-dynamic-stability
L=$REPO/envs/lock-$m-2026-08-17.txt
F=/root/lock-$m.filtered.txt
python3 - "$L" "$F" <<'EOF'
import re, sys
out = []
for line in open(sys.argv[1]):
    s = line.strip()
    if not s or s.startswith('#') or s.startswith('-e '):
        continue
    if ' @ file://' in s:
        name = s.split(' @ ')[0]
        m = re.search(r'/([A-Za-z0-9_.]+?)-(\d[^-/]*)-(?:cp|py)\d', s)
        if m and m.group(1).lower().replace('_', '-') == name.lower().replace('_', '-'):
            out.append(f"{name}=={m.group(2)}")
        continue
    if s.startswith(('conda', 'mamba', 'libmambapy', 'menuinst')):
        continue
    out.append(s)
open(sys.argv[2], 'w').write('\n'.join(out) + '\n')
print(sys.argv[2], len(out), 'requirements')
EOF
rm -rf /root/env-$m
uv venv --python "$py" /root/env-$m
export VIRTUAL_ENV=/root/env-$m
uv pip install --python /root/env-$m/bin/python -r "$F" \
   --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
rc=$?
uv pip install --python /root/env-$m/bin/python --no-deps -e $REPO
/root/env-$m/bin/python -c "import torch, numpy, ase, phonopy; print('ENVCHECK', torch.__version__, torch.cuda.is_available(), numpy.__version__, ase.__version__, phonopy.__version__)"
/root/env-$m/bin/python -c "import cellconstructor, sscha; print('SSCHA_OK')" 2>&1 | tail -1
echo "ENV_DONE $m rc=$rc $(date -u)"
