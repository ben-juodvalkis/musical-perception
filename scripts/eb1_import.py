"""Thin shim: expose EB-1's measured all-pairs estimator under a stable name.

Imports the estimator verbatim from scripts/eb1-estimator-bakeoff.py (whose
hyphenated filename is not importable) so every consumer scores the SAME code
that was measured, never a re-implementation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent / "eb1-estimator-bakeoff.py"
_spec = importlib.util.spec_from_file_location("_eb1", _SRC)
_eb1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eb1)

est_all_pairs = _eb1.est_all_pairs


def all_pairs_bpm(ev: np.ndarray) -> float | None:
    return est_all_pairs(np.asarray(ev, dtype=float))
