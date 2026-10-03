#!/bin/bash
# QE + SSSP on the grid box (vast 54043018, 2026-10-03), the same versions and checksums as
# scripts/box/setup_rsc_box.sh used on 2026-09-27: conda-forge qe=7.5 at /root/qe, SSSP 1.3.0 PBE
# efficiency from the Materials Cloud Archive API (doi 10.24435/materialscloud:f3-ym) at /root/sssp.
set -x
export MAMBA_ROOT_PREFIX=/root/mamba
if [ ! -x /root/bin/micromamba ]; then
  mkdir -p /root/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /root bin/micromamba
fi
if [ ! -x /root/qe/bin/pw.x ]; then
  /root/bin/micromamba create -y -p /root/qe -c conda-forge qe=7.5 > /root/qe_install.log 2>&1
fi
/root/qe/bin/pw.x -h 2>&1 | head -2
ls -la /root/qe/bin/pw.x && echo QE_DONE

API=https://archive.materialscloud.org/api/records/rcyfm-68h65/files
TAR=SSSP_1.3.0_PBE_efficiency.tar.gz;  TAR_MD5=a58f1b3373f330179fd0832c48bb9a52
JSON=SSSP_1.3.0_PBE_efficiency.json;   JSON_MD5=3153c4b20fc90a44fba0236627525644
mkdir -p /root/sssp /root/sssp_download && cd /root/sssp_download || exit 1
[ -f $TAR ] || curl -sSL -o $TAR "$API/$TAR/content"
[ -f $JSON ] || curl -sSL -o $JSON "$API/$JSON/content"
echo "$TAR_MD5  $TAR" | md5sum -c - || { echo SSSP_TAR_MD5_FAIL; exit 1; }
echo "$JSON_MD5  $JSON" | md5sum -c - || { echo SSSP_JSON_MD5_FAIL; exit 1; }
tar xzf $TAR -C /root/sssp && cp $JSON /root/sssp/
# every UPF the json names must be present with the json's md5
python3 - <<'PY'
import hashlib, json, os, sys
d = json.load(open("/root/sssp/SSSP_1.3.0_PBE_efficiency.json"))
bad = []
for el, rec in d.items():
    p = os.path.join("/root/sssp", rec["filename"])
    if not os.path.exists(p) or hashlib.md5(open(p, "rb").read()).hexdigest() != rec["md5"]:
        bad.append(el)
print("SSSP_CHECK", "ok" if not bad else f"FAIL {bad}")
sys.exit(1 if bad else 0)
PY
echo "QE_SETUP_DONE $(date -u)"
