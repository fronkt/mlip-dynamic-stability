"""One CHGNet fine-tune (box or local), settings from scripts/finetune/chgnet_finetune.json.

    /root/env-chgnet/bin/python scripts/finetune/chgnet_finetune_run.py \
        --dataset-dir results/revision/finetune/dataset/chgnet --seed 0 \
        --out-dir results/revision/finetune/models/chgnet/seed0 --device cuda

Writes <out-dir>/best.pth.tar (the trainer's bestE checkpoint: lowest VALIDATION energy MAE),
<out-dir>/result.json (history, settings, hashes).  Never reads a held-out set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def load_split(path):
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor
    frames = read(path, index=":", format="extxyz")
    structs = [AseAtomsAdaptor.get_structure(a) for a in frames]
    e_per_atom = [float(a.info["REF_energy"]) / len(a) for a in frames]
    forces = [np.asarray(a.arrays["REF_forces"], float).tolist() for a in frames]
    return frames, structs, e_per_atom, forces


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--settings", type=Path, default=HERE / "chgnet_finetune.json")
    ap.add_argument("--epochs", type=int, default=None, help="SMOKE TESTS ONLY (default: the fixed 5)")
    ap.add_argument("--max-train", type=int, default=None, help="SMOKE TESTS ONLY")
    ap.add_argument("--max-valid", type=int, default=None, help="SMOKE TESTS ONLY")
    args = ap.parse_args(argv)

    import torch
    from chgnet.data.dataset import StructureData, get_loader
    from chgnet.model import CHGNet
    from chgnet.trainer import Trainer

    st = json.loads(args.settings.read_text(encoding="utf-8"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = CHGNet.load(use_device=args.device)
    for layer in [model.atom_embedding, model.bond_embedding, model.angle_embedding,
                  model.bond_basis_expansion, model.angle_basis_expansion,
                  model.atom_conv_layers[:-1], model.bond_conv_layers, model.angle_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    _, s_tr, e_tr, f_tr = load_split(args.dataset_dir / "train.extxyz")
    _, s_va, e_va, f_va = load_split(args.dataset_dir / "valid.extxyz")
    if args.max_train:
        s_tr, e_tr, f_tr = s_tr[:args.max_train], e_tr[:args.max_train], f_tr[:args.max_train]
    if args.max_valid:
        s_va, e_va, f_va = s_va[:args.max_valid], e_va[:args.max_valid], f_va[:args.max_valid]
    gc = model.graph_converter
    train_ds = StructureData(s_tr, e_tr, f_tr, graph_converter=gc)
    valid_ds = StructureData(s_va, e_va, f_va, graph_converter=gc)
    bs = int(st["loader"]["batch_size"])
    pin = args.device.startswith("cuda")
    train_loader = get_loader(train_ds, batch_size=bs, pin_memory=pin)
    valid_loader = get_loader(valid_ds, batch_size=bs, pin_memory=pin)

    tr = dict(st["trainer"])
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    trainer = Trainer(model=model, targets=tr["targets"], optimizer=tr["optimizer"],
                      scheduler=tr["scheduler"], criterion=tr["criterion"], epochs=tr["epochs"],
                      learning_rate=tr["learning_rate"], energy_loss_ratio=tr["energy_loss_ratio"],
                      force_loss_ratio=tr["force_loss_ratio"], use_device=args.device,
                      print_freq=6, torch_seed=args.seed, data_seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.train(train_loader, valid_loader, test_loader=None, save_dir=str(args.out_dir),
                  train_composition_model=bool(st["train_composition_model"]))
    best = sorted(args.out_dir.glob("bestE_*.pth.tar"))
    if not best:
        raise SystemExit("no bestE checkpoint written (training diverged or produced NaN?)")
    shutil.copyfile(best[-1], args.out_dir / "best.pth.tar")
    hist = trainer.training_history
    res = {"seed": args.seed, "device": args.device, "n_train": len(s_tr), "n_valid": len(s_va),
           "n_trainable_params": n_train_params, "settings": st, "epochs_run": tr["epochs"],
           "best_checkpoint": best[-1].name, "best_sha256": sha(args.out_dir / "best.pth.tar"),
           "history": {k: {kk: [float(x) for x in vv] if isinstance(vv, list) else vv
                           for kk, vv in v.items()} for k, v in hist.items()},
           "train_sha256": sha(args.dataset_dir / "train.extxyz"),
           "valid_sha256": sha(args.dataset_dir / "valid.extxyz"),
           "nan": any(not np.isfinite(x) for v in hist.values() for x in v.get("val", [])
                      if isinstance(x, float))}
    (args.out_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[chgnet-ft] seed {args.seed}: best {best[-1].name}; wrote {args.out_dir / 'best.pth.tar'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
