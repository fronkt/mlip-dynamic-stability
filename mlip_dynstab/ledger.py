"""Hashed, append-only results ledger.

The unit of work is one ``(system, model, temperature, method)`` tuple. Each becomes one
row keyed by a deterministic hash of the *inputs* (system id, model name+version, T, method,
and the run settings dict). Re-running an already-recorded unit is a no-op, which is what
makes the whole pipeline resumable after an interruption.

Storage is parquet via pandas; a JSONL mirror is kept for human-readable diffing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_DEFAULT = Path(__file__).resolve().parent.parent / "results" / "ledger.parquet"


def unit_hash(system_id: str, model: str, model_version: str, temperature_K: float,
              method: str, settings: dict[str, Any] | None = None) -> str:
    payload = {
        "system": system_id, "model": model, "model_version": model_version,
        "T": float(temperature_K), "method": method, "settings": settings or {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _load(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def has_unit(uhash: str, path: Path | str = _DEFAULT) -> bool:
    df = _load(Path(path))
    return (not df.empty) and (uhash in set(df.get("uhash", [])))


def record(row: dict[str, Any], path: Path | str = _DEFAULT, overwrite: bool = False) -> str:
    """Append ``row`` (must contain a 'uhash'); idempotent on uhash. Returns the uhash.

    If ``overwrite`` is True and the uhash already exists, the existing row is replaced
    (used by the CLI ``--force`` path to recompute a unit).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "uhash" not in row:
        raise KeyError("row must contain 'uhash' (use unit_hash(...))")
    df = _load(path)
    if not df.empty and row["uhash"] in set(df["uhash"]):
        if not overwrite:
            return row["uhash"]  # already present; do not duplicate
        df = df[df["uhash"] != row["uhash"]]  # drop stale row, fall through to re-append
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(path, index=False)
    with open(path.with_suffix(".jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row["uhash"]


def load(path: Path | str = _DEFAULT) -> pd.DataFrame:
    return _load(Path(path))
