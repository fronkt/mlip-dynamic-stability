"""Curated-system registry: load configs/curated_systems.yaml and produce ASE Atoms.

Structures come from Materials Project (when ``mp_id`` is given) or are built as canonical
prototypes (bcc metals, alpha-AgI) when no single MP entry matches the high-symmetry phase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "curated_systems.yaml"


@dataclass
class SystemSpec:
    id: str
    formula: str
    prototype: str
    space_group: str
    mp_id: Optional[str]
    klass: str
    harmonic_stable: bool
    finite_T_stable: bool
    transition_T_K: Optional[float]
    soft_mode_q: Optional[str]
    ref: str = ""
    notes: str = ""
    raw: dict = field(default_factory=dict)


def load_specs(config: os.PathLike | None = None) -> list[SystemSpec]:
    path = Path(config) if config else _CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for d in data["systems"]:
        out.append(SystemSpec(
            id=d["id"], formula=d["formula"], prototype=d["prototype"],
            space_group=d["space_group"], mp_id=d.get("mp_id"), klass=d["klass"],
            harmonic_stable=bool(d["harmonic_stable"]), finite_T_stable=bool(d["finite_T_stable"]),
            transition_T_K=d.get("transition_T_K"), soft_mode_q=d.get("soft_mode_q"),
            ref=d.get("ref", ""), notes=d.get("notes", ""), raw=d,
        ))
    return out


def get_spec(system_id: str, config: os.PathLike | None = None) -> SystemSpec:
    for s in load_specs(config):
        if s.id == system_id:
            return s
    raise KeyError(f"system id '{system_id}' not in registry")


def build_atoms(spec: SystemSpec, mp_api_key: str | None = None):
    """Return an ASE Atoms for the high-symmetry reference cell of ``spec``.

    Prefers the MP relaxed conventional cell when ``mp_id`` is set; otherwise builds a
    canonical prototype. The cell is the *reference* (often high-symmetry, possibly
    dynamically unstable) phase — that is the point of the benchmark.
    """
    if spec.mp_id:
        return _atoms_from_mp(spec.mp_id, mp_api_key)
    return _atoms_from_prototype(spec)


def _atoms_from_mp(mp_id: str, mp_api_key: str | None):
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    key = mp_api_key or os.environ.get("MP_API_KEY")
    with MPRester(key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    return AseAtomsAdaptor.get_atoms(structure)


def _atoms_from_prototype(spec: SystemSpec):
    from ase.build import bulk
    proto = spec.prototype
    if proto == "bcc":
        # lattice constants (Angstrom) for the high-T bcc reference phase
        a = {"Ti": 3.27, "Zr": 3.57, "Hf": 3.53}.get(spec.formula, 3.3)
        return bulk(spec.formula, "bcc", a=a, cubic=True)
    if proto == "alpha-AgI":
        # alpha-AgI: bcc iodine framework (Ag disordered). Use bcc-I conventional cell with
        # Ag on the cube center as a starting reference; MD-distortion probe handles disorder.
        from ase import Atoms
        a = 5.06
        return Atoms("AgI", scaled_positions=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                     cell=[a, a, a], pbc=True)
    raise ValueError(f"No prototype builder for '{proto}' (system {spec.id}); add mp_id or a builder.")
