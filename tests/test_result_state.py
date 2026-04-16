from __future__ import annotations

import numpy as np

from open_dealias.result_state import (
    ResultProvenance,
    ResultState,
    ResultStatus,
    attach_result_state,
)
from open_dealias.types import DealiasResult


def test_result_state_from_masks_counts_and_status():
    valid = np.array([[True, True], [True, False]])
    resolved = np.array([[True, False], [True, False]])
    provenance = (ResultProvenance(source="region_graph", stage="bootstrap", parent="zw06", details={"block": 8}),)

    state = ResultState.from_masks(
        valid,
        resolved_mask=resolved,
        provenance=provenance,
        fill_policy="cleanup_then_fill",
        notes=("seeded from neighbor consensus",),
    )

    assert state.status is ResultStatus.PARTIAL
    assert state.valid_gates == 3
    assert state.resolved_gates == 2
    assert state.assigned_gates == 2
    assert state.unresolved_gates == 1
    assert state.resolved_fraction == 2 / 3
    assert state.fill_policy == "cleanup_then_fill"
    assert state.provenance[0].source == "region_graph"
    assert np.array_equal(state.resolved_mask, resolved)
    assert np.array_equal(state.unresolved_mask, np.array([[False, True], [False, False]]))


def test_result_state_attach_to_dealias_result_is_non_destructive():
    velocity = np.array([1.0, 2.0])
    folds = np.array([0, 1], dtype=np.int16)
    confidence = np.array([0.9, 0.4])
    result = DealiasResult(velocity=velocity, folds=folds, confidence=confidence, metadata={"method": "demo"})

    state = ResultState.from_masks(np.array([True, True]), resolved_mask=np.array([True, False]))
    attached = attach_result_state(result, state)

    assert result.result_state is not None
    assert result.result_state is not state
    assert attached.result_state is state
    assert np.array_equal(attached.velocity, velocity)
    assert np.array_equal(attached.folds, folds)
    assert np.array_equal(attached.confidence, confidence)
    assert attached.metadata["method"] == "demo"


def test_dealias_result_auto_populates_default_result_state():
    result = DealiasResult(
        velocity=np.array([[1.0, np.nan], [2.0, 3.0]]),
        folds=np.array([[0, 0], [1, 1]], dtype=np.int16),
        confidence=np.array([[0.9, 0.0], [0.8, 0.7]]),
        metadata={"method": "region_graph", "paper_family": "PyARTRegionGraphLite", "fill_policy": "native_cleanup"},
    )

    assert result.result_state is not None
    assert result.result_state.status is ResultStatus.RESOLVED
    assert result.result_state.valid_gates == 3
    assert result.result_state.resolved_gates == 3
    assert result.result_state.unresolved_gates == 0
    assert result.result_state.fill_policy == "native_cleanup"
    assert result.result_state.provenance[0].source == "region_graph"
    assert result.metadata["resolved_gates"] == 3
    assert result.metadata["unresolved_gates"] == 0


def test_result_state_from_masks_with_no_valid_gates_is_unknown():
    state = ResultState.from_masks(np.array([[False, False]]), resolved_mask=np.array([[False, False]]))
    assert state.status is ResultStatus.UNKNOWN
    assert state.valid_gates == 0
    assert state.resolved_fraction == 0.0
