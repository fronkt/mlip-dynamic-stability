#!/bin/bash
# Update the box checkout without losing the C1 work/ ensembles, then smoke-test C1c (~10 min).
cd /root/mlip-dynamic-stability || exit 1
mv results/revision/sscha_seeds /root/aside/sscha_seeds_box
git pull -q origin rsc-figure-fixes && git log --oneline -1
cp -r /root/aside/sscha_seeds_box/work results/revision/sscha_seeds/work
diff <(cd /root/aside/sscha_seeds_box && ls *.json) <(cd results/revision/sscha_seeds && ls *.json) && echo "json set identical"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
timeout 1h /root/env-mattersim/bin/python -u scripts/sscha_seed_study.py --converge --system zr_bcc --model mattersim --T 50 --device cuda \
  --conv-n-configs 100 --conv-max-pop 2 --conv-n-hessian 100 --conv-steps-per-pop 30 --n-boot 2 --unit-timeout 900 --out-dir /root/c1c_smoke > /root/logs/c1c_smoke.log 2>&1
echo "smoke rc=$?"
/root/env-mattersim/bin/python scripts/sscha_seed_study.py --converge --summarize --out-dir /root/c1c_smoke
/root/env-mattersim/bin/python - <<'EOF'
import json
for s in "AB":
    r = json.load(open(f"/root/c1c_smoke/zr_bcc_mattersim_50K_sc222_start{s}.json"))["run"]
    rx = r.get("relax") or {}; gc = rx.get("gradient_checks") or {}; st = rx.get("steps") or []
    fr = (r.get("fresh_gradient_check") or {}).get("split_null") or {}
    print(s, r["status"], r.get("error"), "| par/ser", gc.get("max_rel_diff_parallel_vs_serial"), "errs", gc.get("n_errors"),
          "| gc_err_real", sorted({round(x["gc_err_real"], 8) for x in st if x.get("gc_err_real")})[:4],
          "| fresh lin", fr.get("lin_rel_err"), "R", fr.get("R"), fr.get("error"),
          "| start", (r.get("start") or {}).get("min_nonac_thz"), "replaced", (r.get("start_construction") or {}).get("n_replaced"))
EOF
echo SMOKE_DONE
