"""Public DFT-phonon reference loaders for the harmonic baseline (Layer 1).

Ground truth comes from published DFT phonon datasets — no new DFT is run. Supported:
  - phonondb / Petretto et al. 2018 MP phonons (~1.5k materials, DFPT)
  - JARVIS-DFT phonons (jarvis-tools)
The loader returns a normalized record: (material_id, formula, dft_min_freq_thz,
dft_dynamically_stable). For Layer 2, ground truth is the literature-curated label in
configs/curated_systems.yaml (see systems.SystemSpec.harmonic_stable / finite_T_stable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from . import DEFAULT_IMAG_TOL_THZ


@dataclass
class PhononRef:
    material_id: str
    formula: str
    dft_min_freq_thz: float
    dft_dynamically_stable: bool
    source: str


def from_jarvis(max_n: Optional[int] = None,
                imag_tol_thz: float = DEFAULT_IMAG_TOL_THZ) -> Iterator[PhononRef]:
    """Yield phonon references from the JARVIS-DFT database (requires jarvis-tools).

    JARVIS stores phonon band data for a subset; we read the minimum frequency where
    available and classify stability with the shared tolerance.
    """
    from jarvis.db.figshare import data as jdata
    rows = jdata("dft_3d")
    n = 0
    for r in rows:
        ph = r.get("phonon") or r.get("ph_bandstructure")
        if not ph:
            continue
        min_freq = _jarvis_min_freq(ph)
        if min_freq is None:
            continue
        yield PhononRef(
            material_id=str(r.get("jid")), formula=str(r.get("formula")),
            dft_min_freq_thz=float(min_freq),
            dft_dynamically_stable=bool(min_freq >= imag_tol_thz), source="jarvis_dft_3d",
        )
        n += 1
        if max_n and n >= max_n:
            return


def _jarvis_min_freq(ph) -> Optional[float]:
    # JARVIS phonon payloads vary; try common shapes. Frequencies stored in cm^-1 -> THz.
    import numpy as np
    try:
        freqs = None
        if isinstance(ph, dict):
            freqs = ph.get("frequencies") or ph.get("freq")
        if freqs is None:
            return None
        arr = np.array(freqs, dtype=float)
        return float(arr.min()) * 0.0299793  # cm^-1 -> THz
    except Exception:
        return None


def from_phonondb_yaml(path: str, imag_tol_thz: float = DEFAULT_IMAG_TOL_THZ) -> PhononRef:
    """Load a single phonopy ``phonopy.yaml``/band result and classify stability."""
    import numpy as np
    import phonopy
    ph = phonopy.load(path)
    ph.run_mesh([12, 12, 12])
    freqs = ph.get_mesh_dict()["frequencies"]
    min_freq = float(np.min(freqs))
    return PhononRef(material_id=path, formula="", dft_min_freq_thz=min_freq,
                     dft_dynamically_stable=bool(min_freq >= imag_tol_thz), source="phonondb")
