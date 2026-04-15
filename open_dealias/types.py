from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(slots=True)
class DealiasResult:
    """Container returned by all dealiased outputs.

    Arrays may be 1D, 2D, or 3D depending on the method.
    """

    velocity: np.ndarray
    folds: np.ndarray
    confidence: np.ndarray
    reference: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VadFit:
    """Uniform-wind VAD fit used as a reference anchor."""

    u: float
    v: float
    offset: float
    rms: float
    iterations: int
    reference: np.ndarray


@dataclass(slots=True)
class RadarSweep:
    """A single real radar sweep converted to array-first form."""

    radar_id: str
    sweep_index: int
    azimuth_deg: np.ndarray
    elevation_deg: float
    range_m: np.ndarray
    nyquist: float
    velocity: np.ndarray
    reflectivity: np.ndarray | None = None
    scan_time: datetime | None = None
    key: str | None = None
    local_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
