from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from examples.benchmark_support import (
    changed_gates,
    compare_dealias_results,
    discover_archive_files,
    mae,
    open_zw06_sweep_anchor,
    open_zw06_volume_anchor,
    record_scored_result,
    record_skipped_result,
    run_solver_pair,
    unresolved_fraction,
)


def test_record_skipped_result_marks_unavailable_methods():
    entry = record_skipped_result("dual_prf", "no verified paired low/high-PRF input")
    assert entry["status"] == "skipped"
    assert entry["name"] == "dual_prf"
    assert "paired low/high-PRF" in entry["skip_reason"]


def test_record_scored_result_uses_open_metrics():
    observed = np.array([[1.0, 2.0], [3.0, 4.0]])
    candidate = np.array([[1.0, 5.0], [np.nan, 4.0]])
    reference = np.array([[2.0, 2.0], [3.0, 6.0]])

    entry = record_scored_result("zw06", candidate, observed, reference, 0.25, metadata={"method": "demo"})

    assert entry["status"] == "computed"
    assert entry["changed_gates"] == changed_gates(candidate, observed)
    assert entry["unresolved_fraction"] == unresolved_fraction(candidate, observed)
    assert entry["mae_vs_pyart"] == mae(candidate, reference)
    assert entry["metadata"]["method"] == "demo"


def test_open_zw06_sweep_anchor_returns_independent_previous_context():
    previous_observed = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    anchor, metadata = open_zw06_sweep_anchor(previous_observed, 10.0)

    assert anchor.shape == previous_observed.shape
    assert metadata["source"] == "open_zw06"
    assert metadata["nyquist"] == 10.0


def test_open_zw06_volume_anchor_returns_one_anchor_per_sweep():
    previous_volume = np.stack(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ],
        axis=0,
    )

    anchor, metadata = open_zw06_volume_anchor(previous_volume, np.array([10.0, 12.0]))

    assert anchor.shape == previous_volume.shape
    assert len(metadata) == 2
    assert metadata[0]["source"] == "open_zw06"
    assert metadata[1]["nyquist"] == 12.0


def test_discover_archive_files_filters_zip_files(tmp_path):
    (tmp_path / "a.zip").write_text("zip", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "c.ZIP").write_text("zip", encoding="utf-8")

    files = discover_archive_files(tmp_path)

    assert [path.name for path in files] == ["a.zip", "c.ZIP"]


def test_run_solver_pair_switches_backend_and_preserves_python_fallback():
    module = SimpleNamespace(_NATIVE_BACKEND="native")

    def solver(*args, **kwargs):
        return "native" if module._NATIVE_BACKEND is not None else "python"

    module.solver = solver
    payload = run_solver_pair(module, "solver")

    assert payload["backend_available"] is True
    assert payload["public_result"] == "native"
    assert payload["python_result"] == "python"
    assert payload["public_runtime_s"] >= 0.0
    assert payload["python_runtime_s"] >= 0.0
    assert module._NATIVE_BACKEND == "native"


def test_compare_dealias_results_reports_differences():
    native = SimpleNamespace(
        velocity=np.array([[1.0, 2.0], [3.0, np.nan]]),
        folds=np.array([[0, 1], [2, 3]], dtype=np.int16),
        confidence=np.array([[0.9, 0.8], [0.7, 0.6]]),
        reference=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    python = SimpleNamespace(
        velocity=np.array([[1.0, 4.0], [3.0, np.nan]]),
        folds=np.array([[0, 0], [2, 4]], dtype=np.int16),
        confidence=np.array([[0.9, 0.7], [0.7, 0.5]]),
        reference=np.array([[1.0, 3.0], [3.0, 4.0]]),
    )

    payload = compare_dealias_results(native, python)

    assert payload["folds_equal"] is False
    assert payload["folds_mismatch_count"] == 2
    assert payload["velocity_mae"] == 2.0 / 3.0
    assert payload["confidence_mae"] == pytest.approx(0.05)
    assert payload["reference_mae"] == 0.25
