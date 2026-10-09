"""Fine-tuning trial (RSC Advances R2.1): does fine-tuning toward PBE remove the finite-T mis-calls?

Design: tasks/preregistration-finetune-2026-10-03.md (committed 033b3d8; a pre-registration, not
edited here).  This script only implements it.  Everything it writes lives under
results/revision/finetune/ (outputs) and results/revision/dft/qe/ft_* (the pw.x jobs the box queue
runs unchanged).  It never writes results/ledger.*.

Stages, each resumable (it skips outputs that exist; --force redoes):

  configs     LOCAL, any env with the base model (several --models run as one child process per
              model: MACE sets torch's default dtype to float64, which breaks CHGNet in-process).
              40 phonon-rattled 2x2x2 cells per (test system, base model) from that model's own
              relaxed-cell harmonic force constants (the C2 sampler, imported from
              scripts/force_spread.py), 13/13/14 draws at 100/300/600 K, fixed seeds, leakage guard
              (redraw within 0.05 A RMS of any held-out C3a/C3b geometry of the system).  Writes
              configs/, fc/ and manifest.json.  --min-pair-ratio R adds an OPTIONAL contact redraw
              rule (default off: not in the pre-registration).
                  python scripts/finetune_trial.py configs --device cpu
  qe-inputs   LOCAL.  pw.x inputs ft_<system>_<model>_c<NN> (+ _ref, the undisplaced cell, which is
              not a training configuration) in the exact format and settings of the C3b jobs;
              prints the job count and the modelled core-hours (from the measured C3a/C3b costs).
                  python scripts/finetune_trial.py qe-inputs
  (box)       bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir <UPF dir>                   -r 8 -n 5 --filter '^ft_' --log /root/logs/qe_ft.log
  dataset     BOX/LOCAL, in the base model's env.  Parse pw.out, apply the per-system energy
              offset (data.py explains it), random 90/10 split with a fixed seed, write
              dataset/<model>/{train,valid,labelled_all}.extxyz and dataset.json.
                  python scripts/finetune_trial.py dataset --model mace_mp0 --device cuda
  train       --emit writes the whole box runbook (results/revision/finetune/train/box_sequence.sh:
              QE queue, dataset, 3 seeds x 2 models, evaluation, P3, summary); --run runs one
              fine-tune here; --smoke is a 2-configuration 1-epoch CPU plumbing test.
                  python scripts/finetune_trial.py train --emit
  evaluate    BOX, per model env, per checkpoint tag (base, seed0, seed1, seed2) and part
              (c3a, c3b, p2, s2); `--part p3-cmd` prints the SSCHA command (P3); `--part summary`
              (any env) merges everything into results/revision/finetune/summary.json
              (P1/P2/P3/S1/S2, all replicates).
                  python scripts/finetune_trial.py evaluate --model mace_mp0 --tag seed0 --part all
                  python scripts/finetune_trial.py evaluate --part summary
  selftest    Unit tests of the parsing/scoring code against existing data (no training): pw.in
              writer byte-equality with the C3b jobs, leakage guard, config reproducibility, base
              models reproduce c3a_paths.csv / c3b_units.csv / the ledger, checkpoint hook.
                  python scripts/finetune_trial.py selftest --which all --model mace_mp0 --device cpu

A fine-tuned checkpoint is loaded by mlip_dynstab.calculators when MLIP_DYNSTAB_CKPT_<MODEL> names
it (unset in production, so nothing changes); that is how P3 runs scripts/sscha_seed_study.py
unedited.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from finetune import common as C  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--out", type=Path, default=C.OUT_DEFAULT,
                       help="output root (default results/revision/finetune)")
        p.add_argument("--systems", nargs="+", default=list(C.TEST_SYSTEMS),
                       choices=list(C.TEST_SYSTEMS))
        p.add_argument("--force", action="store_true", help="redo work whose output exists")

    p = sub.add_parser("configs", help="phonon-rattled training configurations + leakage guard")
    common(p)
    p.add_argument("--models", nargs="+", default=list(C.BASE_MODELS), choices=list(C.BASE_MODELS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-pair-ratio", type=float, default=0.0,
                   help="OPTIONAL extra redraw rule, off (0) by default because the "
                        "pre-registration has none: redraw a draw whose closest atom pair is "
                        "nearer than this fraction of the reference distance of the same "
                        "species pair (force_spread.py's variant (g) uses 0.75)")

    p = sub.add_parser("qe-inputs", help="pw.x inputs for the training configurations")
    common(p)
    p.add_argument("--models", nargs="+", default=list(C.BASE_MODELS), choices=list(C.BASE_MODELS))
    p.add_argument("--qe-root", type=Path, default=C.QE_ROOT)
    p.add_argument("--sssp-json", default=None,
                   help="the SSSP json on the box; if given, its cutoffs/filenames must equal "
                        "scripts/finetune/sssp_cutoffs.json (harvested from the C3a/C3b jobs)")
    p.add_argument("--kspacing-insulator", type=float, default=0.25)
    p.add_argument("--degauss-insulator", type=float, default=0.005)
    p.add_argument("--no-write", action="store_true", help="count and cost only")

    p = sub.add_parser("dataset", help="parse pw.out into train/valid extxyz per base model")
    common(p)
    p.add_argument("--model", required=True, choices=list(C.BASE_MODELS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--qe-root", type=Path, default=C.QE_ROOT)
    p.add_argument("--allow-partial", action="store_true",
                   help="build from the jobs that finished (smoke tests only)")
    p.add_argument("--mock", action="store_true",
                   help="PLUMBING TEST ONLY: labels = base-model prediction + synthetic noise; "
                        "refuses to write into results/revision/finetune (needs --out <scratch>)")

    p = sub.add_parser("train", help="emit / run the fine-tunes")
    common(p)
    p.add_argument("--emit", action="store_true", help="write results/.../train/box_train.sh")
    p.add_argument("--run", action="store_true", help="run one fine-tune here")
    p.add_argument("--model", choices=list(C.BASE_MODELS))
    p.add_argument("--seed", type=int, choices=list(C.REPLICATE_SEEDS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run (2 configurations, 1 epoch) into <out>/_smoke; plumbing only")

    p = sub.add_parser("evaluate", help="held-out C3a/C3b, P2 screen, S2 controls, summary")
    common(p)
    p.add_argument("--model", choices=list(C.BASE_MODELS))
    p.add_argument("--tag", default=None, help="base | seed0 | seed1 | seed2")
    p.add_argument("--part", default="all",
                   choices=["all", "c3a", "c3b", "p2", "s2", "summary", "p3-cmd"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dft-root", type=Path, default=C.DFT_ROOT)
    p.add_argument("--base-eval-from", type=Path, default=None,
                   help="summary only: read base-model evaluations missing under <out>/eval from "
                        "<this>/eval (the 30-epoch re-run reuses results/revision/finetune)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="explicit checkpoint file (default: the trained one under <out>/models)")
    p.add_argument("--max-paths", type=int, default=None, help="smoke: limit C3a paths")

    p = sub.add_parser("selftest", help="unit tests against existing data")
    common(p)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", choices=list(C.BASE_MODELS), default="mace_mp0")
    p.add_argument("--which", nargs="+", default=["all"])
    p.add_argument("--ckpt", type=Path, default=None, help="a fine-tuned checkpoint for the hook test")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "configs":
        from finetune import gen
        return gen.stage_configs(args)
    if args.stage == "qe-inputs":
        from finetune import qe
        return qe.stage_qe_inputs(args)
    if args.stage == "dataset":
        from finetune import data
        return data.stage_dataset(args)
    if args.stage == "train":
        from finetune import train
        return train.stage_train(args)
    if args.stage == "evaluate":
        from finetune import evalu
        return evalu.stage_evaluate(args)
    if args.stage == "selftest":
        from finetune import selftest
        return selftest.run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
