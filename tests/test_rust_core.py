from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import open_dealias.continuity as continuity_module
import open_dealias._core as core_module
import open_dealias.multipass as multipass_module
import open_dealias.region_graph as region_graph_module
import open_dealias.recursive as recursive_module
import open_dealias.qc as qc_module
import open_dealias.variational as variational_module
import open_dealias.volume3d as volume3d_module
from open_dealias._core import (
    _python_fold_counts,
    _python_shift2d,
    _python_shift3d,
    _python_unfold_to_reference,
    _python_wrap_to_nyquist,
    fold_counts,
    neighbor_stack,
    shift2d,
    shift3d,
    texture_3x3,
    unfold_to_reference,
    wrap_to_nyquist,
)
from open_dealias._rust_bridge import get_rust_backend
from open_dealias.continuity import dealias_radial_es90, dealias_sweep_es90
from open_dealias.dual_prf import _python_dealias_dual_prf, dealias_dual_prf
from open_dealias.multipass import dealias_sweep_zw06
from open_dealias.qc import build_velocity_qc_mask
from open_dealias.recursive import dealias_sweep_recursive
from open_dealias.variational import dealias_sweep_variational
from open_dealias.volume3d import dealias_volume_3d
from examples.region_variational_migration_benchmark import _default_archive_path, _time_case, _time_region_graph_case, load_real_like_case


backend = get_rust_backend()
pytestmark = pytest.mark.skipif(backend is None, reason="Rust backend not built")


def test_wrap_to_nyquist_matches_python_semantics() -> None:
    data = np.array([-31.0, -10.0, -0.0, 0.0, 9.999, 10.0, 29.5, np.nan, np.inf, -np.inf])
    expected = _python_wrap_to_nyquist(data, 10.0)
    rust = backend.wrap_to_nyquist(np.asarray(data, dtype=float), 10.0)
    public = wrap_to_nyquist(data, 10.0)

    np.testing.assert_allclose(rust, expected, equal_nan=True)
    np.testing.assert_allclose(public, expected, equal_nan=True)


def test_fold_counts_matches_python_semantics() -> None:
    unfolded = np.array([-15.0, -5.0, 5.0, 15.0, np.nan])
    observed = np.array([-5.0, -5.0, -5.0, -5.0, -5.0])
    expected = _python_fold_counts(unfolded, observed, 10.0)
    rust = backend.fold_counts(np.asarray(unfolded, dtype=float), np.asarray(observed, dtype=float), 10.0)
    public = fold_counts(unfolded, observed, 10.0)

    np.testing.assert_array_equal(rust, expected)
    np.testing.assert_array_equal(public, expected)


def test_unfold_to_reference_matches_python_semantics() -> None:
    observed = np.array([2.0, 2.0, 2.0, np.nan])
    reference = np.array([102.0, -98.0, 22.0, 0.0])
    expected = _python_unfold_to_reference(observed, reference, 10.0, max_abs_fold=2)
    rust = backend.unfold_to_reference(
        np.asarray(observed, dtype=float),
        np.asarray(reference, dtype=float),
        10.0,
        2,
    )
    public = unfold_to_reference(observed, reference, 10.0, max_abs_fold=2)

    np.testing.assert_allclose(rust, expected, equal_nan=True)
    np.testing.assert_allclose(public, expected, equal_nan=True)


def test_shift2d_matches_python_semantics() -> None:
    field = np.arange(12, dtype=float).reshape(3, 4)
    expected = _python_shift2d(field, shift_az=4, shift_range=-2, wrap_azimuth=False)
    rust = backend.shift2d(np.asarray(field, dtype=float), 4, -2, False)
    public = shift2d(field, 4, -2, wrap_azimuth=False)

    np.testing.assert_allclose(rust, expected, equal_nan=True)
    np.testing.assert_allclose(public, expected, equal_nan=True)


def test_shift3d_matches_python_semantics() -> None:
    volume = np.arange(24, dtype=float).reshape(2, 3, 4)
    expected = _python_shift3d(volume, shift_az=-1, shift_range=1, wrap_azimuth=True)
    rust = backend.shift3d(np.asarray(volume, dtype=float), -1, 1, True)
    public = shift3d(volume, -1, 1, wrap_azimuth=True)

    np.testing.assert_allclose(rust, expected, equal_nan=True)
    np.testing.assert_allclose(public, expected, equal_nan=True)


def test_neighbor_stack_matches_python_semantics() -> None:
    field = np.arange(12, dtype=float).reshape(3, 4)
    expected = np.stack(
        [
            _python_shift2d(field, shift_az=-1, shift_range=0, wrap_azimuth=True),
            _python_shift2d(field, shift_az=1, shift_range=0, wrap_azimuth=True),
            _python_shift2d(field, shift_az=0, shift_range=-1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=0, shift_range=1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=-1, shift_range=-1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=-1, shift_range=1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=1, shift_range=-1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=1, shift_range=1, wrap_azimuth=True),
        ],
        axis=0,
    )
    rust = backend.neighbor_stack(np.asarray(field, dtype=float), True, True)
    public = neighbor_stack(field, include_diagonals=True, wrap_azimuth=True)

    np.testing.assert_allclose(rust, expected, equal_nan=True)
    np.testing.assert_allclose(public, expected, equal_nan=True)


def test_texture_matches_python_semantics() -> None:
    field = np.array(
        [
            [0.0, 1.0, 2.0, np.nan],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        dtype=float,
    )
    stack = np.stack(
        [
            _python_shift2d(field, shift_az=0, shift_range=0, wrap_azimuth=True),
            _python_shift2d(field, shift_az=-1, shift_range=0, wrap_azimuth=True),
            _python_shift2d(field, shift_az=1, shift_range=0, wrap_azimuth=True),
            _python_shift2d(field, shift_az=0, shift_range=-1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=0, shift_range=1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=-1, shift_range=-1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=-1, shift_range=1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=1, shift_range=-1, wrap_azimuth=True),
            _python_shift2d(field, shift_az=1, shift_range=1, wrap_azimuth=True),
        ],
        axis=0,
    )
    expected = np.nanstd(stack, axis=0)
    rust = backend.texture_3x3(np.asarray(field, dtype=float), True)
    public = texture_3x3(field, wrap_azimuth=True)

    np.testing.assert_allclose(rust, expected, equal_nan=True)
    np.testing.assert_allclose(public, expected, equal_nan=True)


def test_qc_mask_matches_python_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    velocity = np.array([[1.0, 2.0, np.nan], [4.0, 30.0, 6.0]], dtype=float)
    reflectivity = np.array([[5.0, -10.0, 0.0], [10.0, 10.0, 10.0]], dtype=float)
    texture = np.array([[1.0, 1.0, 1.0], [1.0, 20.0, 1.0]], dtype=float)

    monkeypatch.setattr(qc_module, "_NATIVE_BACKEND", None)
    expected = qc_module.build_velocity_qc_mask(
        velocity,
        reflectivity=reflectivity,
        texture=texture,
        min_reflectivity=-5.0,
        max_texture=12.0,
        min_gate_fraction_in_ray=0.20,
    )
    monkeypatch.setattr(qc_module, "_NATIVE_BACKEND", backend)
    public = build_velocity_qc_mask(
        velocity,
        reflectivity=reflectivity,
        texture=texture,
        min_reflectivity=-5.0,
        max_texture=12.0,
        min_gate_fraction_in_ray=0.20,
    )
    np.testing.assert_array_equal(public, expected)


def test_dual_prf_matches_python_semantics() -> None:
    low = np.array([-6.0, 5.0, np.nan, 3.0, -2.0], dtype=float)
    high = np.array([14.0, 5.0, 7.0, np.nan, -2.0], dtype=float)
    reference = np.array([14.0, 5.0, 7.0, 3.0, -2.0], dtype=float)

    expected = _python_dealias_dual_prf(low, high, 10.0, 16.0, reference=reference)
    public = dealias_dual_prf(low, high, 10.0, 16.0, reference=reference)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(public.reference, expected.reference, equal_nan=True)


def test_radial_es90_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    obs = np.array([-6.0, -5.0, np.nan, 4.0, 5.0, 6.0], dtype=float)
    reference = np.array([14.0, 15.0, np.nan, 24.0, 25.0, 26.0], dtype=float)

    monkeypatch.setattr(continuity_module, "_NATIVE_BACKEND", None)
    expected = continuity_module.dealias_radial_es90(obs, 10.0, reference=reference, max_gap=2, max_abs_step=12.0)
    monkeypatch.setattr(continuity_module, "_NATIVE_BACKEND", backend)
    public = dealias_radial_es90(obs, 10.0, reference=reference, max_gap=2, max_abs_step=12.0)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    assert public.metadata["seed_index"] == expected.metadata["seed_index"]


def test_sweep_es90_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0, 5.0],
            [-7.0, -6.0, 5.0, 6.0],
            [np.nan, -7.0, 6.0, 7.0],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [14.0, 15.0, 24.0, 25.0],
            [13.0, 14.0, 25.0, 26.0],
            [np.nan, 13.0, 26.0, 27.0],
        ],
        dtype=float,
    )

    monkeypatch.setattr(continuity_module, "_NATIVE_BACKEND", None)
    expected = continuity_module.dealias_sweep_es90(observed, 10.0, reference=reference, max_gap=2, max_abs_step=12.0)
    monkeypatch.setattr(continuity_module, "_NATIVE_BACKEND", backend)
    public = dealias_sweep_es90(observed, 10.0, reference=reference, max_gap=2, max_abs_step=12.0)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)


def test_zw06_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0, 5.0],
            [-7.0, -6.0, 5.0, 6.0],
            [np.nan, -7.0, 6.0, 7.0],
            [-8.0, -7.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [14.0, 15.0, 24.0, 25.0],
            [13.0, 14.0, 25.0, 26.0],
            [np.nan, 13.0, 26.0, 27.0],
            [12.0, 13.0, 27.0, 28.0],
        ],
        dtype=float,
    )

    monkeypatch.setattr(multipass_module, "_NATIVE_BACKEND", None)
    expected = multipass_module.dealias_sweep_zw06(observed, 10.0, reference=reference)
    monkeypatch.setattr(multipass_module, "_NATIVE_BACKEND", backend)
    public = dealias_sweep_zw06(observed, 10.0, reference=reference)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    assert public.metadata["iterations_used"] == expected.metadata["iterations_used"]


def test_variational_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0, 5.0],
            [-7.0, -6.0, 5.0, 6.0],
            [np.nan, -7.0, 6.0, 7.0],
            [-8.0, -7.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [14.0, 15.0, 24.0, 25.0],
            [13.0, 14.0, 25.0, 26.0],
            [np.nan, 13.0, 26.0, 27.0],
            [12.0, 13.0, 27.0, 28.0],
        ],
        dtype=float,
    )

    monkeypatch.setattr(variational_module, "_NATIVE_BACKEND", None)
    expected = variational_module.dealias_sweep_variational(observed, 10.0, reference=reference, max_iterations=4)
    monkeypatch.setattr(variational_module, "_NATIVE_BACKEND", backend)
    public = dealias_sweep_variational(observed, 10.0, reference=reference, max_iterations=4)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    assert public.metadata["iterations_used"] == expected.metadata["iterations_used"]


def test_volume3d_dispatches_to_native_backend_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [
                [-6.0, -5.0, 4.0, 5.0],
                [-7.0, -6.0, 5.0, 6.0],
                [np.nan, -7.0, 6.0, 7.0],
            ],
            [
                [-5.0, -4.0, 5.0, 6.0],
                [-6.0, -5.0, 6.0, 7.0],
                [-7.0, -6.0, 7.0, 8.0],
            ],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [
                [14.0, 15.0, 24.0, 25.0],
                [13.0, 14.0, 25.0, 26.0],
                [np.nan, 13.0, 26.0, 27.0],
            ],
            [
                [15.0, 16.0, 25.0, 26.0],
                [14.0, 15.0, 26.0, 27.0],
                [13.0, 14.0, 27.0, 28.0],
            ],
        ],
        dtype=float,
    )

    monkeypatch.setattr(core_module, "_NATIVE_BACKEND", None)
    monkeypatch.setattr(multipass_module, "_NATIVE_BACKEND", None)
    monkeypatch.setattr(volume3d_module, "_NATIVE_BACKEND", None)
    expected = volume3d_module.dealias_volume_3d(observed, 10.0, reference_volume=reference, max_iterations=3)

    class FakeVolume3DBackend:
        def dealias_volume_3d(self, obs, nyquist, ref, wrap_azimuth, max_iterations):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            np.testing.assert_allclose(ref, reference, equal_nan=True)
            assert nyquist.shape == (observed.shape[0],)
            assert wrap_azimuth is True
            assert max_iterations == 3
            return (
                expected.velocity,
                expected.folds,
                expected.confidence,
                expected.reference,
                {"backend": "fake_volume3d", "method": "volume_3d_continuity"},
            )

    monkeypatch.setattr(volume3d_module, "_NATIVE_BACKEND", FakeVolume3DBackend())
    public = dealias_volume_3d(observed, 10.0, reference_volume=reference, max_iterations=3)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(public.reference, expected.reference, equal_nan=True)
    assert public.metadata["backend"] == "fake_volume3d"
    assert public.metadata["method"] == "volume_3d_continuity"


def test_region_graph_dispatches_to_native_backend_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0, 5.0],
            [-7.0, -6.0, 5.0, 6.0],
            [np.nan, -7.0, 6.0, 7.0],
            [-8.0, -7.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [14.0, 15.0, 24.0, 25.0],
            [13.0, 14.0, 25.0, 26.0],
            [np.nan, 13.0, 26.0, 27.0],
            [12.0, 13.0, 27.0, 28.0],
        ],
        dtype=float,
    )

    expected = region_graph_module._python_dealias_sweep_region_graph(
        observed,
        10.0,
        reference=reference,
        block_shape=(2, 2),
        reference_weight=0.70,
        max_iterations=4,
        max_abs_fold=6,
        wrap_azimuth=True,
    )

    class FakeRegionGraphBackend:
        def dealias_sweep_region_graph(
            self,
            obs,
            nyquist,
            ref,
            block_shape,
            reference_weight,
            max_iterations,
            max_abs_fold,
            wrap_azimuth,
        ):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            assert nyquist == 10.0
            assert block_shape == (2, 2)
            assert reference_weight == pytest.approx(0.70)
            assert max_iterations == 4
            assert max_abs_fold == 6
            assert wrap_azimuth is True
            np.testing.assert_allclose(ref, reference, equal_nan=True)
            return (
                expected.velocity,
                expected.folds,
                expected.confidence,
                expected.reference,
                {"backend": "fake_region_graph", "method": "region_graph_sweep"},
            )

    monkeypatch.setattr(region_graph_module, "_NATIVE_BACKEND", FakeRegionGraphBackend())
    public = region_graph_module.dealias_sweep_region_graph(
        observed,
        10.0,
        reference=reference,
        block_shape=(2, 2),
        reference_weight=0.70,
        max_iterations=4,
        max_abs_fold=6,
        wrap_azimuth=True,
    )

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(public.reference, expected.reference, equal_nan=True)
    assert public.metadata["backend"] == "fake_region_graph"
    assert public.metadata["method"] == "region_graph_sweep"
    assert public.metadata["paper_family"] == "PyARTRegionGraphLite"
    assert public.metadata["block_shape"] == [2, 2]


def test_region_graph_benchmark_hook_records_real_like_case_if_archive_present():
    archive_path = _default_archive_path(Path("archive_data"))
    if archive_path is None:
        pytest.skip("no local archive_data cases available")

    pytest.importorskip("pyart")

    case = load_real_like_case(archive_path, sweep=1, crop_half_size=18)
    result = _time_region_graph_case(case)
    metrics = result["metrics"]

    assert isinstance(metrics["backend_available"], bool)
    assert metrics["public_runtime_s"] >= 0.0
    assert metrics["python_runtime_s"] >= 0.0
    assert metrics["parity"]["velocity_mae"] >= 0.0
    assert metrics["parity"]["folds_mismatch_count"] >= 0
    assert np.isfinite(metrics["mae_vs_reference"])
    assert metrics["metadata"]["method"] == "region_graph_sweep"
    assert metrics["metadata"]["region_count"] > 0


def test_recursive_dispatches_to_native_backend_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0, 5.0],
            [-7.0, -6.0, 5.0, 6.0],
            [np.nan, -7.0, 6.0, 7.0],
            [-8.0, -7.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [14.0, 15.0, 24.0, 25.0],
            [13.0, 14.0, 25.0, 26.0],
            [np.nan, 13.0, 26.0, 27.0],
            [12.0, 13.0, 27.0, 28.0],
        ],
        dtype=float,
    )

    expected = recursive_module._python_dealias_sweep_recursive(
        observed,
        10.0,
        reference=reference,
        max_depth=4,
        min_leaf_cells=12,
        split_texture_fraction=0.55,
        reference_weight=0.65,
        max_abs_fold=6,
        wrap_azimuth=True,
    )

    class FakeRecursiveBackend:
        def dealias_sweep_recursive(
            self,
            obs,
            nyquist,
            ref,
            max_depth,
            min_leaf_cells,
            split_texture_fraction,
            reference_weight,
            max_abs_fold,
            wrap_azimuth,
        ):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            np.testing.assert_allclose(ref, reference, equal_nan=True)
            assert nyquist == 10.0
            assert max_depth == 4
            assert min_leaf_cells == 12
            assert split_texture_fraction == pytest.approx(0.55)
            assert reference_weight == pytest.approx(0.65)
            assert max_abs_fold == 6
            assert wrap_azimuth is True
            return (
                expected.velocity,
                expected.folds,
                expected.confidence,
                expected.reference,
                {"backend": "fake_recursive", "method": "recursive_region_refinement"},
            )

    monkeypatch.setattr(recursive_module, "_NATIVE_BACKEND", FakeRecursiveBackend())
    public = dealias_sweep_recursive(
        observed,
        10.0,
        reference=reference,
        max_depth=4,
        min_leaf_cells=12,
        split_texture_fraction=0.55,
        reference_weight=0.65,
        max_abs_fold=6,
        wrap_azimuth=True,
    )

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(public.reference, expected.reference, equal_nan=True)
    assert public.metadata["backend"] == "fake_recursive"
    assert public.metadata["method"] == "recursive_region_refinement"
    assert public.metadata["paper_family"] == "R2D2StyleRecursiveLite"
    assert public.metadata["bootstrap_method"] == "region_graph_sweep"


def test_recursive_benchmark_hook_records_real_like_case_if_archive_present():
    archive_path = _default_archive_path(Path("archive_data"))
    if archive_path is None:
        pytest.skip("no local archive_data cases available")

    pytest.importorskip("pyart")

    case = load_real_like_case(archive_path, sweep=1, crop_half_size=18)
    result = _time_case(case)
    metrics = result["recursive"]

    assert isinstance(metrics["backend_available"], bool)
    assert metrics["public_runtime_s"] >= 0.0
    assert metrics["python_runtime_s"] >= 0.0
    assert metrics["parity"]["velocity_mae"] >= 0.0
    assert metrics["parity"]["folds_mismatch_count"] >= 0
    assert np.isfinite(metrics["mae_vs_reference"])
    assert metrics["metadata"]["method"] in {"recursive_region_refinement", "recursive_region_refinement_fallback_region_graph"}
    assert metrics["metadata"]["leaf_count"] > 0


def test_volume3d_dispatches_to_native_backend_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = np.array(
        [
            [
                [-6.0, -5.0, 4.0, 5.0],
                [-7.0, -6.0, 5.0, 6.0],
            ],
            [
                [-5.0, -4.0, 5.0, 6.0],
                [-6.0, -5.0, 6.0, 7.0],
            ],
        ],
        dtype=float,
    )
    nyquist = np.array([10.0, 10.0], dtype=float)
    reference = np.array(
        [
            [
                [14.0, 15.0, 24.0, 25.0],
                [13.0, 14.0, 25.0, 26.0],
            ],
            [
                [15.0, 16.0, 25.0, 26.0],
                [14.0, 15.0, 26.0, 27.0],
            ],
        ],
        dtype=float,
    )

    expected = volume3d_module.dealias_volume_3d(observed, nyquist, reference_volume=reference, max_iterations=3)

    class FakeVolume3DBackend:
        def dealias_volume_3d(self, obs, nyq, ref, wrap_azimuth, max_iterations):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            np.testing.assert_allclose(nyq, nyquist, equal_nan=True)
            np.testing.assert_allclose(ref, reference, equal_nan=True)
            assert wrap_azimuth is True
            assert max_iterations == 3
            return (
                expected.velocity,
                expected.folds,
                expected.confidence,
                expected.reference,
                {"backend": "fake_volume3d", "method": "volume_3d_continuity"},
            )

    monkeypatch.setattr(volume3d_module, "_NATIVE_BACKEND", FakeVolume3DBackend())
    public = dealias_volume_3d(observed, nyquist, reference_volume=reference, max_iterations=3)

    np.testing.assert_allclose(public.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(public.folds, expected.folds)
    np.testing.assert_allclose(public.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(public.reference, expected.reference, equal_nan=True)
    assert public.metadata["backend"] == "fake_volume3d"
    assert public.metadata["method"] == "volume_3d_continuity"
    assert public.metadata["paper_family"] == "UNRAVEL-style-3D"


def test_volume3d_benchmark_hook_records_real_like_case_if_archive_present():
    archive_path = _default_archive_path(Path("archive_data"))
    if archive_path is None:
        pytest.skip("no local archive_data cases available")

    pytest.importorskip("pyart")

    case = load_real_like_case(archive_path, sweep=1, crop_half_size=18)
    result = _time_case(case)
    metrics = result.get("volume_3d")
    if metrics is None:
        pytest.skip("volume3d inputs unavailable for this archive case")

    assert isinstance(metrics["backend_available"], bool)
    assert metrics["public_runtime_s"] >= 0.0
    assert metrics["python_runtime_s"] >= 0.0
    assert metrics["parity"]["velocity_mae"] >= 0.0
    assert metrics["parity"]["folds_mismatch_count"] >= 0
    assert np.isfinite(metrics["mae_vs_reference"])
    assert metrics["metadata"]["method"] == "volume_3d_continuity"
    assert metrics["metadata"]["seed_sweep"] >= 0
