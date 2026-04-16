from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .result_state import ResultProvenance, ResultState


def _default_result_state(
    velocity: np.ndarray,
    metadata: dict[str, Any],
    existing: ResultState | None,
) -> ResultState:
    if existing is not None:
        return existing
    method = str(metadata.get("method", "unknown"))
    family = str(metadata.get("paper_family", metadata.get("paper", method)))
    fill_policy = metadata.get("fill_policy")
    notes = ()
    if metadata.get("bootstrap_method"):
        notes = (f"bootstrap_method={metadata['bootstrap_method']}",)
    valid = np.isfinite(velocity)
    return ResultState.from_masks(
        valid,
        resolved_mask=valid,
        provenance=(ResultProvenance(source=method, parent=family, details={"paper_family": family}),),
        fill_policy=None if fill_policy is None else str(fill_policy),
        notes=notes,
    )


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
    result_state: ResultState | None = None

    def __post_init__(self) -> None:
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.folds = np.asarray(self.folds, dtype=np.int16)
        self.confidence = np.asarray(self.confidence, dtype=float)
        if self.folds.shape != self.velocity.shape:
            raise ValueError("folds must match velocity shape")
        if self.confidence.shape != self.velocity.shape:
            raise ValueError("confidence must match velocity shape")
        if self.reference is not None:
            self.reference = np.asarray(self.reference, dtype=float)
            if self.reference.shape != self.velocity.shape:
                raise ValueError("reference must match velocity shape")
        self.metadata = dict(self.metadata)
        self.result_state = _default_result_state(self.velocity, self.metadata, self.result_state)
        self.metadata.setdefault("valid_gates", int(self.result_state.valid_gates))
        self.metadata.setdefault("resolved_gates", int(self.result_state.resolved_gates))
        self.metadata.setdefault("unresolved_gates", int(self.result_state.unresolved_gates))
        self.metadata.setdefault("resolved_fraction", float(self.result_state.resolved_fraction))


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
