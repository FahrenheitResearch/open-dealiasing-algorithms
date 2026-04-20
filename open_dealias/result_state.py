from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .types import DealiasResult


class ResultStatus(str, Enum):
    UNKNOWN = "unknown"
    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


@dataclass(slots=True)
class ResultProvenance:
    """Lightweight provenance record for a dealiased result."""

    source: str
    stage: str | None = None
    parent: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResultState:
    """Resolved/unresolved summary plus optional provenance and masks."""

    status: ResultStatus = ResultStatus.UNKNOWN
    valid_gates: int = 0
    resolved_gates: int = 0
    unresolved_gates: int = 0
    resolved_fraction: float = 0.0
    fill_policy: str | None = None
    provenance: tuple[ResultProvenance, ...] = ()
    resolved_mask: np.ndarray | None = None
    unresolved_mask: np.ndarray | None = None
    notes: tuple[str, ...] = ()

    @property
    def assigned_gates(self) -> int:
        return self.resolved_gates

    @classmethod
    def from_masks(
        cls,
        valid_mask: np.ndarray,
        *,
        resolved_mask: np.ndarray | None = None,
        unresolved_mask: np.ndarray | None = None,
        provenance: tuple[ResultProvenance, ...] = (),
        fill_policy: str | None = None,
        notes: tuple[str, ...] = (),
    ) -> "ResultState":
        valid = np.asarray(valid_mask, dtype=bool)
        resolved = None if resolved_mask is None else np.asarray(resolved_mask, dtype=bool)
        unresolved = None if unresolved_mask is None else np.asarray(unresolved_mask, dtype=bool)

        arrays = [valid]
        if resolved is not None:
            arrays.append(resolved)
        if unresolved is not None:
            arrays.append(unresolved)
        valid_b, *rest = np.broadcast_arrays(*arrays)
        valid_b = np.asarray(valid_b, dtype=bool)
        idx = 0
        if resolved is not None:
            resolved_b = np.asarray(rest[idx], dtype=bool)
            idx += 1
        else:
            resolved_b = None
        if unresolved is not None:
            unresolved_b = np.asarray(rest[idx], dtype=bool)
        else:
            unresolved_b = None

        if resolved_b is None and unresolved_b is None:
            raise ValueError("at least one of resolved_mask or unresolved_mask is required")

        if resolved_b is None:
            resolved_b = valid_b & ~unresolved_b  # type: ignore[operator]
        if unresolved_b is None:
            unresolved_b = valid_b & ~resolved_b

        overlap = valid_b & resolved_b & unresolved_b
        if np.any(overlap):
            raise ValueError("resolved_mask and unresolved_mask overlap")

        valid_gates = int(np.count_nonzero(valid_b))
        resolved_gates = int(np.count_nonzero(valid_b & resolved_b))
        unresolved_gates = int(np.count_nonzero(valid_b & unresolved_b))

        if valid_gates == 0:
            status = ResultStatus.UNKNOWN
        elif unresolved_gates == 0:
            status = ResultStatus.RESOLVED
        elif resolved_gates == 0:
            status = ResultStatus.UNRESOLVED
        else:
            status = ResultStatus.PARTIAL

        resolved_fraction = float(resolved_gates / valid_gates) if valid_gates else 0.0
        return cls(
            status=status,
            valid_gates=valid_gates,
            resolved_gates=resolved_gates,
            unresolved_gates=unresolved_gates,
            resolved_fraction=resolved_fraction,
            fill_policy=fill_policy,
            provenance=provenance,
            resolved_mask=np.array(resolved_b, dtype=bool, copy=True),
            unresolved_mask=np.array(unresolved_b, dtype=bool, copy=True),
            notes=notes,
        )

    @property
    def coverage(self) -> float:
        return self.resolved_fraction


def attach_result_state(result: "DealiasResult", state: ResultState) -> "DealiasResult":
    """Return a copy of a DealiasResult with state attached."""
    metadata = result.metadata
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.update(
            {
                "valid_gates": state.valid_gates,
                "resolved_gates": state.resolved_gates,
                "unresolved_gates": state.unresolved_gates,
                "resolved_fraction": state.resolved_fraction,
            }
        )
    return replace(result, result_state=state, metadata=metadata)


def build_result_state(
    observed: np.ndarray,
    velocity: np.ndarray,
    *,
    source: str,
    parent: str | None = None,
    stage: str | None = None,
    fill_policy: str | None = None,
    details: dict[str, Any] | None = None,
    notes: tuple[str, ...] = (),
    valid_mask: np.ndarray | None = None,
    resolved_mask: np.ndarray | None = None,
) -> ResultState:
    """Construct a ResultState from observed and corrected fields."""

    obs = np.asarray(observed, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    if valid_mask is None:
        valid_mask = np.isfinite(obs)
    if resolved_mask is None:
        resolved_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(vel)
    provenance = (
        ResultProvenance(
            source=str(source),
            stage=None if stage is None else str(stage),
            parent=None if parent is None else str(parent),
            details={} if details is None else dict(details),
        ),
    )
    return ResultState.from_masks(
        valid_mask,
        resolved_mask=resolved_mask,
        provenance=provenance,
        fill_policy=fill_policy,
        notes=notes,
    )


def attach_result_state_from_fields(
    result: "DealiasResult",
    observed: np.ndarray,
    *,
    source: str,
    parent: str | None = None,
    stage: str | None = None,
    fill_policy: str | None = None,
    details: dict[str, Any] | None = None,
    notes: tuple[str, ...] = (),
    valid_mask: np.ndarray | None = None,
    resolved_mask: np.ndarray | None = None,
) -> "DealiasResult":
    return attach_result_state(
        result,
        build_result_state(
            observed,
            result.velocity,
            source=source,
            parent=parent,
            stage=stage,
            fill_policy=fill_policy,
            details=details,
            notes=notes,
            valid_mask=valid_mask,
            resolved_mask=resolved_mask,
        ),
    )
