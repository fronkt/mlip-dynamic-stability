"""Model-agnostic ASE-calculator factory for foundation MLIPs.

Each backend is imported *inside* its branch so that a given environment only needs the
one MLIP package it was set up for (see ``envs/``). This avoids the torch-version conflicts
that arise from trying to co-install MACE + CHGNet + ORB + SevenNet + MatterSim.

Usage (inside a model's env):
    from mlip_dynstab.calculators import get_calculator
    calc = get_calculator("mace_mp0", device="cuda")
    atoms.calc = calc

Every calculator is wrapped so we can record an exact, reproducible ``model_version`` string
in the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CalcHandle:
    """A loaded calculator plus provenance for the ledger."""
    name: str          # canonical model key, e.g. "mace_mp0"
    version: str       # resolved package/checkpoint version string
    calc: Any          # ASE Calculator instance
    device: str


def get_calculator(name: str, device: str = "cuda", **kw) -> CalcHandle:
    """Return a CalcHandle for ``name``. Raises ImportError if the env lacks the backend."""
    name = name.lower()
    if name == "mace_mp0":
        return _mace_mp0(device, **kw)
    if name == "chgnet":
        return _chgnet(device, **kw)
    if name in ("orb_v2", "orb_v3"):
        return _orb(name, device, **kw)
    if name == "sevennet0":
        return _sevennet(device, **kw)
    if name == "mattersim":
        return _mattersim(device, **kw)
    raise ValueError(f"Unknown model '{name}'. Supported: see mlip_dynstab.SUPPORTED_MODELS")


def _pkg_version(mod_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(mod_name)
    except Exception:
        return "unknown"


def _mace_mp0(device: str, model: str = "medium", **kw) -> CalcHandle:
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, device=device, default_dtype="float64", **kw)
    return CalcHandle("mace_mp0", f"mace-torch={_pkg_version('mace-torch')};ckpt={model}", calc, device)


def _chgnet(device: str, **kw) -> CalcHandle:
    from chgnet.model.dynamics import CHGNetCalculator
    calc = CHGNetCalculator(use_device=device, **kw)
    return CalcHandle("chgnet", f"chgnet={_pkg_version('chgnet')}", calc, device)


def _orb(name: str, device: str, **kw) -> CalcHandle:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    loader = pretrained.orb_v2 if name == "orb_v2" else pretrained.orb_v3_conservative_inf_omat
    orbff = loader(device=device)
    calc = ORBCalculator(orbff, device=device, **kw)
    return CalcHandle(name, f"orb-models={_pkg_version('orb-models')};{name}", calc, device)


def _sevennet(device: str, model: str = "7net-0", **kw) -> CalcHandle:
    from sevenn.calculator import SevenNetCalculator
    calc = SevenNetCalculator(model=model, device=device, **kw)
    return CalcHandle("sevennet0", f"sevenn={_pkg_version('sevenn')};ckpt={model}", calc, device)


def _mattersim(device: str, model: str = "mattersim-v1.0.0-5m", **kw) -> CalcHandle:
    from mattersim.forcefield import MatterSimCalculator
    calc = MatterSimCalculator(load_path=model, device=device, **kw)
    return CalcHandle("mattersim", f"mattersim={_pkg_version('mattersim')};ckpt={model}", calc, device)
