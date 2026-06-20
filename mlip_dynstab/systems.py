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


# Approximate cubic lattice constants (Angstrom) for the high-symmetry reference phase.
# These are starting points only: compute_harmonic relaxes each cell to the MLIP's own
# minimum before phonons, so the benchmark does not depend on these exact values — it
# depends only on building the correct high-symmetry prototype + chemistry.
LATTICE_A = {
    "srtio3_cubic": 3.905, "batio3_cubic": 4.00, "knbo3_cubic": 4.02,
    "pbtio3_cubic": 3.97, "ktao3_cubic": 3.99,
    "cspbi3_cubic": 6.29, "cssni3_cubic": 6.22, "cssnbr3_cubic": 5.80,
    "ti_bcc": 3.27, "zr_bcc": 3.57, "hf_bcc": 3.53,
    "zro2_cubic": 5.09, "hfo2_cubic": 5.08, "ceo2_cubic": 5.41,
    "agi_bcc": 5.06,
    "si_diamond": 5.43, "mgo_rocksalt": 4.21, "nacl_rocksalt": 5.64,
    "cu_fcc": 3.61, "c_diamond": 3.567,
}

# prototype label -> ASE crystalstructure name (perovskite handled separately via pymatgen)
_ASE_PROTO = {
    "bcc": "bcc", "fcc": "fcc", "diamond": "diamond",
    "rocksalt": "rocksalt", "fluorite": "fluorite", "alpha-AgI": "cesiumchloride",
}


def build_atoms(spec: SystemSpec, mp_api_key: str | None = None):
    """Return an ASE Atoms for the high-symmetry reference cell of ``spec``.

    Builds a canonical prototype offline (no API key) — fully reproducible. The cell is the
    *reference* (often high-symmetry, possibly dynamically unstable) phase; that is the point
    of the benchmark. ``_atoms_from_mp`` remains available if an MP-sourced cell is preferred.
    """
    return _atoms_from_prototype(spec)


def _atoms_from_mp(mp_id: str, mp_api_key: str | None):
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    key = mp_api_key or os.environ.get("MP_API_KEY")
    with MPRester(key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    return AseAtomsAdaptor.get_atoms(structure)


def _split_perovskite_species(formula: str) -> list[str]:
    """ABX3 -> [A, B, X] in formula order (e.g. SrTiO3 -> [Sr, Ti, O])."""
    import re
    elems = re.findall(r"[A-Z][a-z]?", formula)
    if len(elems) != 3:
        raise ValueError(f"perovskite formula '{formula}' did not parse to 3 species: {elems}")
    return elems


def _atoms_from_prototype(spec: SystemSpec):
    from ase.build import bulk
    a = LATTICE_A.get(spec.id)
    if a is None:
        raise ValueError(f"No lattice constant registered for '{spec.id}'")
    proto = spec.prototype

    if proto == "cubic-perovskite":
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        A, B, X = _split_perovskite_species(spec.formula)
        struct = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(a),
            species=[A, B, X],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],  # A 1a, B 1b, X 3c
        )
        return AseAtomsAdaptor.get_atoms(struct)

    cs = _ASE_PROTO.get(proto)
    if cs is None:
        raise ValueError(f"No prototype builder for '{proto}' (system {spec.id})")
    # ASE builds binary prototypes (rocksalt/fluorite/cesiumchloride) and elemental
    # (bcc/fcc/diamond) directly from the formula.
    return bulk(spec.formula, cs, a=a)
