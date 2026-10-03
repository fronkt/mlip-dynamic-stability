"""Stage `train`: the exact box commands for the 3 x 2 fine-tunes, and a local runner for smoke tests.

Settings are fixed in scripts/finetune/mace_medium_finetune.json and chgnet_finetune.json (frozen
before any training data exists; nothing is tuned).  Layout under results/revision/finetune/:

  dataset/<model>/{train,valid}.extxyz        from the `dataset` stage
  models/mace_mp0/seed<k>/ft_mace_mp0_seed<k>.model     (+ checkpoints/, logs/, results/)
  models/chgnet/seed<k>/best.pth.tar          (+ result.json)

A fine-tune that diverges (NaN) is re-run once with the same settings and the next unused seed
(3, then 4, ...), and the failure is recorded: that is the pre-registered rule, and it needs a
human decision, so `--emit` writes the rerun command as a comment, not as a default.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from . import common as C

MACE_JSON = C.FT_DIR / "mace_medium_finetune.json"
CHGNET_JSON = C.FT_DIR / "chgnet_finetune.json"
ENV_PY = {"mace_mp0": "/root/env-mace/bin/python", "chgnet": "/root/env-chgnet/bin/python"}


def mace_args(out: Path, seed: int, work_dir: Path, device: str, epochs=None, train=None, valid=None):
    st = C.jload(MACE_JSON)["args"]
    ddir = out / "dataset" / "mace_mp0"
    a = [f"--name=ft_mace_mp0_seed{seed}", f"--seed={seed}", f"--work_dir={work_dir}",
         f"--train_file={train or ddir / 'train.extxyz'}", f"--valid_file={valid or ddir / 'valid.extxyz'}",
         f"--device={device}"]
    for k, v in st.items():
        if k == "max_num_epochs" and epochs is not None:
            v = epochs
        if isinstance(v, bool):
            if k in ("ema", "amsgrad", "save_cpu"):          # store_true flags
                a += [f"--{k}"] if v else []
            else:                                              # str2bool options
                a += [f"--{k}={'True' if v else 'False'}"]
        else:
            a += [f"--{k}={v}"]
    return a


def mace_cmd(py: str, *a, **kw):
    return [py, "-m", "mace.cli.run_train", *mace_args(*a, **kw)]


def chgnet_cmd(py: str, out: Path, seed: int, device: str, ddir=None, odir=None, extra=()):
    return [py, "scripts/finetune/chgnet_finetune_run.py",
            f"--dataset-dir={ddir or out / 'dataset' / 'chgnet'}", f"--seed={seed}",
            f"--out-dir={odir or out / 'models' / 'chgnet' / f'seed{seed}'}", f"--device={device}",
            *extra]


def _sh(cmd) -> str:
    """Shell-quote a command; a bare $VAR (the PY_* interpreter variables) stays expandable."""
    return " ".join(str(c) if str(c).startswith("$") else shlex.quote(str(c).replace("\\", "/"))
                    for c in cmd)


def emit(out: Path) -> str:
    """The whole box sequence, in order.  ``out`` is repo-relative (the box clones the repo at
    /root/mlip-dynamic-stability and every step cd's there)."""
    q = "results/revision/dft/qe"
    L = ["#!/bin/bash",
         "# Pre-registered fine-tuning trial: tasks/preregistration-finetune-2026-10-03.md",
         "# Settings: scripts/finetune/mace_medium_finetune.json, scripts/finetune/chgnet_finetune.json",
         "# Run the STEPS in order (each is resumable).  This file is a runbook, not one unattended job:",
         "# step 1 is ~1.2k core-hours of pw.x, the rest ~4 GPU-hours.",
         "cd /root/mlip-dynamic-stability",
         "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
         "run() { echo \"===== $* $(date -u)\"; \"$@\"; rc=$?; [ $rc -ne 0 ] && echo \"[rc=$rc] $*\"; return 0; }",
         "PY_MACE=/root/env-mace/bin/python; PY_CHG=/root/env-chgnet/bin/python",
         "",
         "# STEP 0. inputs (generated locally, NOT committed): results/revision/finetune/{manifest.json,configs/,fc/}",
         f"#         and {q}/ft_*/ (246 dirs). Copy them to the box (rsync) or commit them first, then:",
         f"ls {q} | grep -c '^ft_'          # expect 246",
         "",
         "# STEP 1. PBE single points (qe_queue.sh unchanged; 246 jobs, ~1.2k core-h: ~21 h at 56 cores, ~29 h at 40)",
         "export ESPRESSO_PSEUDO=/root/sssp   # the directory that holds the UPF files (it was /root/sssp/upf for C3a/C3b)",
         f"bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir $ESPRESSO_PSEUDO -r 8 -n 5 --filter '^ft_' \\",
         "    --log /root/logs/qe_ft.log < /dev/null",
         "",
         "# STEP 2. training / validation files (needs the base model: one run per env)",
         "run $PY_MACE -u scripts/finetune_trial.py dataset --model mace_mp0 --device cuda",
         "run $PY_CHG  -u scripts/finetune_trial.py dataset --model chgnet   --device cuda",
         "",
         "# STEP 3. fine-tunes: MACE-MP-0 medium x seeds 0,1,2 (~0.3 h each on the 3090, float64), CHGNet x 3 (~0.1 h each)"]
    for s in C.REPLICATE_SEEDS:
        wd = out / "models" / "mace_mp0" / f"seed{s}"
        L.append("run " + _sh(mace_cmd("$PY_MACE", out, s, wd, "cuda")))
    for s in C.REPLICATE_SEEDS:
        L.append("run " + _sh(chgnet_cmd("$PY_CHG", out, s, "cuda")))
    L += ["# A fine-tune that ends in NaN is re-run once with the same settings and the next unused seed",
          "# (pre-registration, 'Analysis') and the failure recorded; e.g. CHGNet seed 3:",
          "# run " + _sh(chgnet_cmd("$PY_CHG", out, 3, "cuda")),
          "",
          "# STEP 4. evaluation: base (the foundation model, same code path) and each replicate;",
          "#         parts c3a (P1, S1), c3b (S1), p2 (P2 screen pipeline), s2 (controls); ~0.3 h per model/tag",
          ]
    for m, py in (("mace_mp0", "$PY_MACE"), ("chgnet", "$PY_CHG")):
        for t in ("base", "seed0", "seed1", "seed2"):
            L.append(f"run {py} -u scripts/finetune_trial.py evaluate --model {m} --tag {t} --part all --device cuda --workers 8")
    L += ["",
          "# STEP 5. P3: converged SSCHA, BaTiO3 100 K, seed-0 fine-tuned MACE-MP-0 (start A, grid recipe; ~0.5 h)",
          "MLIP_DYNSTAB_CKPT_MACE_MP0=$(pwd)/" + C.rel(out / "models" / "mace_mp0" / "seed0" / "ft_mace_mp0_seed0.model")
          + " \\",
          "  $PY_MACE -u scripts/sscha_seed_study.py --preset grid --model mace_mp0 --system batio3_cubic --T 100 \\",
          "  --start A --device cuda --out-dir " + C.rel(out / "sscha_converged_grid") + " 2>&1 | tee /root/logs/ft_p3_sscha.log",
          "",
          "# STEP 6. summary (any env, no GPU) -> results/revision/finetune/summary.json",
          "run /root/env-mace/bin/python scripts/finetune_trial.py evaluate --part summary",
          "# then pull results/revision/finetune/{dataset,models,eval,p2cache,sscha_converged_grid,summary.json} and",
          f"# {q}/ft_*/pw.out back; models/ are not committed unless you decide to (MACE ~45 MB each).",
          ""]
    return "\n".join(L)


def stage_train(args) -> int:
    out = Path(args.out)
    if args.emit:
        text = emit(Path(C.rel(out)))
        p = out / "train" / "box_sequence.sh"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        print(text)
        print(f"[train] wrote {C.rel(p)}")
        return 0
    if args.smoke:
        return _smoke(args)
    if args.run:
        if args.model is None or args.seed is None:
            raise SystemExit("--run needs --model and --seed")
        if args.model == "mace_mp0":
            cmd = mace_cmd(sys.executable, out, args.seed, out / "models" / "mace_mp0" / f"seed{args.seed}",
                           args.device)
        else:
            cmd = chgnet_cmd(sys.executable, out, args.seed, args.device)
        print("[train] " + " ".join(shlex.quote(str(c)) for c in cmd))
        return subprocess.run(cmd, cwd=str(C.REPO)).returncode
    raise SystemExit("give --emit, --run (with --model/--seed) or --smoke")


def _smoke(args) -> int:
    """Tiny CPU fine-tune on 2 training + 1 validation configurations for 1 epoch, into
    <out>/_smoke: proves the command lines, file formats and checkpoint loading run, nothing more."""
    out = Path(args.out)
    model = args.model or "mace_mp0"
    src = out / "dataset" / model
    if not (src / "train.extxyz").exists():
        raise SystemExit(f"{src}/train.extxyz missing: run `dataset` (with --mock for a plumbing test)")
    sm = out / "_smoke" / model
    sm.mkdir(parents=True, exist_ok=True)
    tr = C.read_frames(src / "train.extxyz")
    va = C.read_frames(src / "valid.extxyz")
    # two training configurations from different systems when possible
    pick = []
    seen = set()
    for a in tr:
        if a.info["system"] not in seen:
            pick.append(a)
            seen.add(a.info["system"])
        if len(pick) == 2:
            break
    pick = (pick + tr)[:2]
    C.write_frames(sm / "train.extxyz", pick)
    C.write_frames(sm / "valid.extxyz", va[:1])
    seed = args.seed if args.seed is not None else 0
    if model == "mace_mp0":
        wd = sm / f"seed{seed}"
        cmd = mace_cmd(sys.executable, out, seed, wd, "cpu", epochs=1,
                       train=sm / "train.extxyz", valid=sm / "valid.extxyz")
    else:
        cmd = chgnet_cmd(sys.executable, out, seed, "cpu", ddir=sm, odir=sm / f"seed{seed}",
                         extra=["--epochs=1"])
    print("[train --smoke] " + " ".join(shlex.quote(str(c)) for c in cmd))
    return subprocess.run(cmd, cwd=str(C.REPO)).returncode
