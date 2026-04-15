from __future__ import annotations

import numpy as np

import open_dealias.fourdd as fourdd_module


class _SweepBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def dealias_sweep_jh01(
        self,
        observed: np.ndarray,
        nyquist: float,
        previous_corrected: np.ndarray | None,
        background_reference: np.ndarray | None,
        shift_az: int,
        shift_range: int,
        wrap_azimuth: bool,
        refine_with_multipass: bool,
    ):
        self.calls.append(
            (
                observed.copy(),
                float(nyquist),
                None if previous_corrected is None else previous_corrected.copy(),
                None if background_reference is None else background_reference.copy(),
                int(shift_az),
                int(shift_range),
                bool(wrap_azimuth),
                bool(refine_with_multipass),
            )
        )
        velocity = np.full(observed.shape, 11.0, dtype=float)
        folds = np.full(observed.shape, 3, dtype=np.int16)
        confidence = np.full(observed.shape, 0.75, dtype=float)
        reference = np.full(observed.shape, 17.0, dtype=float)
        metadata = {"native": "sweep"}
        return velocity, folds, confidence, reference, metadata


class _VolumeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def dealias_volume_jh01(
        self,
        observed_volume: np.ndarray,
        nyquist: np.ndarray,
        azimuth_deg: np.ndarray,
        elevation_deg: np.ndarray,
        previous_volume: np.ndarray | None,
        background_uv,
        shift_az: int,
        shift_range: int,
        wrap_azimuth: bool,
    ):
        self.calls.append(
            (
                observed_volume.copy(),
                np.asarray(nyquist, dtype=float).copy(),
                azimuth_deg.copy(),
                elevation_deg.copy(),
                None if previous_volume is None else previous_volume.copy(),
                background_uv,
                int(shift_az),
                int(shift_range),
                bool(wrap_azimuth),
            )
        )
        velocity = np.full(observed_volume.shape, 23.0, dtype=float)
        folds = np.full(observed_volume.shape, -2, dtype=np.int16)
        confidence = np.full(observed_volume.shape, 0.5, dtype=float)
        reference = np.full(observed_volume.shape, 31.0, dtype=float)
        metadata = {"native": "volume"}
        return velocity, folds, confidence, reference, metadata


def _sweep_inputs():
    observed = np.array(
        [
            [-6.0, -5.0, 4.0, 5.0],
            [-7.0, -6.0, 5.0, 6.0],
            [-8.0, -7.0, 6.0, 7.0],
        ],
        dtype=float,
    )
    previous_corrected = np.array(
        [
            [14.0, 15.0, 24.0, 25.0],
            [13.0, 14.0, 25.0, 26.0],
            [12.0, 13.0, 26.0, 27.0],
        ],
        dtype=float,
    )
    background_reference = np.array(
        [
            [16.0, 17.0, 26.0, 27.0],
            [15.0, 16.0, 27.0, 28.0],
            [14.0, 15.0, 28.0, 29.0],
        ],
        dtype=float,
    )
    return observed, previous_corrected, background_reference


def _volume_inputs():
    observed = np.array(
        [
            [
                [-6.0, -5.0, 4.0, 5.0],
                [-7.0, -6.0, 5.0, 6.0],
                [-8.0, -7.0, 6.0, 7.0],
            ],
            [
                [-4.0, -3.0, 6.0, 7.0],
                [-5.0, -4.0, 7.0, 8.0],
                [-6.0, -5.0, 8.0, 9.0],
            ],
        ],
        dtype=float,
    )
    previous_volume = np.array(
        [
            [
                [14.0, 15.0, 24.0, 25.0],
                [13.0, 14.0, 25.0, 26.0],
                [12.0, 13.0, 26.0, 27.0],
            ],
            [
                [16.0, 17.0, 26.0, 27.0],
                [15.0, 16.0, 27.0, 28.0],
                [14.0, 15.0, 28.0, 29.0],
            ],
        ],
        dtype=float,
    )
    azimuth_deg = np.array([0.0, 120.0, 240.0], dtype=float)
    elevation_deg = np.array([1.0, 4.0], dtype=float)
    return observed, previous_volume, azimuth_deg, elevation_deg


def test_jh01_sweep_dispatches_when_native_method_exists(monkeypatch):
    observed, previous_corrected, background_reference = _sweep_inputs()
    backend = _SweepBackend()
    monkeypatch.setattr(fourdd_module, "_NATIVE_BACKEND", backend, raising=False)

    result = fourdd_module.dealias_sweep_jh01(
        observed,
        10.0,
        previous_corrected=previous_corrected,
        background_reference=background_reference,
        shift_az=1,
        shift_range=-1,
        wrap_azimuth=False,
        refine_with_multipass=True,
    )

    assert len(backend.calls) == 1
    call = backend.calls[0]
    np.testing.assert_allclose(call[0], observed)
    assert call[1] == 10.0
    np.testing.assert_allclose(call[2], previous_corrected)
    np.testing.assert_allclose(call[3], background_reference)
    assert call[4:] == (1, -1, False, True)
    np.testing.assert_allclose(result.velocity, 11.0)
    np.testing.assert_array_equal(result.folds, np.full(observed.shape, 3, dtype=np.int16))
    np.testing.assert_allclose(result.confidence, 0.75)
    np.testing.assert_allclose(result.reference, 17.0)
    assert result.metadata["native"] == "sweep"


def test_jh01_sweep_falls_back_without_native_method(monkeypatch):
    observed, previous_corrected, background_reference = _sweep_inputs()

    monkeypatch.setattr(fourdd_module, "_NATIVE_BACKEND", object(), raising=False)
    fallback_with_missing_method = fourdd_module.dealias_sweep_jh01(
        observed,
        10.0,
        previous_corrected=previous_corrected,
        background_reference=background_reference,
        shift_az=1,
        shift_range=-1,
        wrap_azimuth=False,
        refine_with_multipass=True,
    )

    monkeypatch.setattr(fourdd_module, "_NATIVE_BACKEND", None, raising=False)
    fallback_without_backend = fourdd_module.dealias_sweep_jh01(
        observed,
        10.0,
        previous_corrected=previous_corrected,
        background_reference=background_reference,
        shift_az=1,
        shift_range=-1,
        wrap_azimuth=False,
        refine_with_multipass=True,
    )

    np.testing.assert_allclose(fallback_with_missing_method.velocity, fallback_without_backend.velocity, equal_nan=True)
    np.testing.assert_array_equal(fallback_with_missing_method.folds, fallback_without_backend.folds)
    np.testing.assert_allclose(fallback_with_missing_method.confidence, fallback_without_backend.confidence, equal_nan=True)
    np.testing.assert_allclose(fallback_with_missing_method.reference, fallback_without_backend.reference, equal_nan=True)
    assert fallback_with_missing_method.metadata == fallback_without_backend.metadata


def test_jh01_volume_dispatches_when_native_method_exists(monkeypatch):
    observed, previous_volume, azimuth_deg, elevation_deg = _volume_inputs()
    backend = _VolumeBackend()
    monkeypatch.setattr(fourdd_module, "_NATIVE_BACKEND", backend, raising=False)

    result = fourdd_module.dealias_volume_jh01(
        observed,
        np.array([10.0, 12.0], dtype=float),
        azimuth_deg,
        elevation_deg,
        previous_volume=previous_volume,
        background_uv=(4.5, -1.5),
        shift_az=2,
        shift_range=1,
        wrap_azimuth=False,
    )

    assert len(backend.calls) == 1
    call = backend.calls[0]
    np.testing.assert_allclose(call[0], observed)
    np.testing.assert_allclose(call[1], np.array([10.0, 12.0], dtype=float))
    np.testing.assert_allclose(call[2], azimuth_deg)
    np.testing.assert_allclose(call[3], elevation_deg)
    np.testing.assert_allclose(call[4], previous_volume)
    assert call[5] == (4.5, -1.5)
    assert call[6:] == (2, 1, False)
    np.testing.assert_allclose(result.velocity, 23.0)
    np.testing.assert_array_equal(result.folds, np.full(observed.shape, -2, dtype=np.int16))
    np.testing.assert_allclose(result.confidence, 0.5)
    np.testing.assert_allclose(result.reference, 31.0)
    assert result.metadata["native"] == "volume"


def test_jh01_volume_falls_back_without_native_method(monkeypatch):
    observed, previous_volume, azimuth_deg, elevation_deg = _volume_inputs()

    monkeypatch.setattr(fourdd_module, "_NATIVE_BACKEND", object(), raising=False)
    fallback_with_missing_method = fourdd_module.dealias_volume_jh01(
        observed,
        np.array([10.0, 12.0], dtype=float),
        azimuth_deg,
        elevation_deg,
        previous_volume=previous_volume,
        background_uv=(4.5, -1.5),
        shift_az=2,
        shift_range=1,
        wrap_azimuth=False,
    )

    monkeypatch.setattr(fourdd_module, "_NATIVE_BACKEND", None, raising=False)
    fallback_without_backend = fourdd_module.dealias_volume_jh01(
        observed,
        np.array([10.0, 12.0], dtype=float),
        azimuth_deg,
        elevation_deg,
        previous_volume=previous_volume,
        background_uv=(4.5, -1.5),
        shift_az=2,
        shift_range=1,
        wrap_azimuth=False,
    )

    np.testing.assert_allclose(fallback_with_missing_method.velocity, fallback_without_backend.velocity, equal_nan=True)
    np.testing.assert_array_equal(fallback_with_missing_method.folds, fallback_without_backend.folds)
    np.testing.assert_allclose(fallback_with_missing_method.confidence, fallback_without_backend.confidence, equal_nan=True)
    np.testing.assert_allclose(fallback_with_missing_method.reference, fallback_without_backend.reference, equal_nan=True)
    assert fallback_with_missing_method.metadata == fallback_without_backend.metadata
