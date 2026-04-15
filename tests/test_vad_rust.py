from __future__ import annotations

import numpy as np

from open_dealias import dealias_sweep_xu11, estimate_uniform_wind_vad
from open_dealias import vad as vad_module
from open_dealias.types import DealiasResult, VadFit


def test_estimate_uniform_wind_vad_dispatches_to_native_backend(monkeypatch) -> None:
    observed = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    azimuth_deg = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
    reference = np.full_like(observed, 42.0, dtype=float)

    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def estimate_uniform_wind_vad(self, obs, nyquist, az, **kwargs):
            self.calls.append(((obs, nyquist, az), kwargs))
            return VadFit(u=12.5, v=-3.25, offset=0.5, rms=1.75, iterations=4, reference=reference)

    backend = FakeBackend()
    monkeypatch.setattr(vad_module, "_NATIVE_BACKEND", backend)

    fit = estimate_uniform_wind_vad(
        observed,
        10.0,
        azimuth_deg,
        elevation_deg=0.5,
        sign=-1.0,
        max_iterations=3,
        trim_quantile=0.9,
        search_radius=24.0,
    )

    assert len(backend.calls) == 1
    (obs_arg, nyq_arg, az_arg), kwargs = backend.calls[0]
    np.testing.assert_allclose(obs_arg, observed)
    assert nyq_arg == 10.0
    np.testing.assert_allclose(az_arg, azimuth_deg)
    assert kwargs == {
        "elevation_deg": 0.5,
        "sign": -1.0,
        "max_iterations": 3,
        "trim_quantile": 0.9,
        "search_radius": 24.0,
    }
    assert fit.u == 12.5
    assert fit.v == -3.25
    assert fit.offset == 0.5
    assert fit.rms == 1.75
    assert fit.iterations == 4
    np.testing.assert_allclose(fit.reference, reference)


def test_dealias_sweep_xu11_dispatches_to_native_backend(monkeypatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0],
            [-7.0, -6.0, 5.0],
            [-8.0, -7.0, 6.0],
            [-9.0, -8.0, 7.0],
        ],
        dtype=float,
    )
    azimuth_deg = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
    external_reference = np.full_like(observed, 24.0, dtype=float)
    velocity = observed + 20.0
    folds = np.ones_like(observed, dtype=np.int16)
    confidence = np.full_like(observed, 0.75, dtype=float)
    reference = np.full_like(observed, 33.0, dtype=float)
    metadata = {"backend": "rust_xu11", "method": "vad_seeded_multipass"}

    class FakeBackend:
        def dealias_sweep_xu11(self, obs, nyquist, az, **kwargs):
            np.testing.assert_allclose(obs, observed)
            np.testing.assert_allclose(az, azimuth_deg)
            assert nyquist == 10.0
            assert kwargs["elevation_deg"] == 0.5
            np.testing.assert_allclose(kwargs["external_reference"], external_reference)
            assert kwargs["sign"] == -1.0
            assert kwargs["refine_with_multipass"] is False
            return DealiasResult(velocity=velocity, folds=folds, confidence=confidence, reference=reference, metadata=metadata)

        def estimate_uniform_wind_vad(self, *args, **kwargs):  # pragma: no cover - must not be used in this test
            raise AssertionError("dispatch should short-circuit to dealias_sweep_xu11")

    monkeypatch.setattr(vad_module, "_NATIVE_BACKEND", FakeBackend())

    result = dealias_sweep_xu11(
        observed,
        10.0,
        azimuth_deg,
        elevation_deg=0.5,
        external_reference=external_reference,
        sign=-1.0,
        refine_with_multipass=False,
    )

    np.testing.assert_allclose(result.velocity, velocity)
    np.testing.assert_array_equal(result.folds, folds)
    np.testing.assert_allclose(result.confidence, confidence)
    np.testing.assert_allclose(result.reference, reference)
    assert result.metadata == metadata


def test_python_fallbacks_remain_available_without_native_backend(monkeypatch) -> None:
    observed = np.array(
        [
            [-6.0, -5.0, 4.0],
            [-7.0, -6.0, 5.0],
            [-8.0, -7.0, 6.0],
            [-9.0, -8.0, 7.0],
        ],
        dtype=float,
    )
    azimuth_deg = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
    reference = np.full_like(observed, 24.0, dtype=float)

    monkeypatch.setattr(vad_module, "_NATIVE_BACKEND", None)

    expected_fit = vad_module._python_estimate_uniform_wind_vad(
        observed,
        10.0,
        azimuth_deg,
        elevation_deg=0.5,
        sign=1.0,
        max_iterations=2,
        trim_quantile=0.9,
        search_radius=24.0,
    )
    public_fit = estimate_uniform_wind_vad(
        observed,
        10.0,
        azimuth_deg,
        elevation_deg=0.5,
        sign=1.0,
        max_iterations=2,
        trim_quantile=0.9,
        search_radius=24.0,
    )

    expected_result = vad_module._python_dealias_sweep_xu11(
        observed,
        10.0,
        azimuth_deg,
        elevation_deg=0.5,
        external_reference=reference,
        sign=1.0,
        refine_with_multipass=False,
    )
    public_result = dealias_sweep_xu11(
        observed,
        10.0,
        azimuth_deg,
        elevation_deg=0.5,
        external_reference=reference,
        sign=1.0,
        refine_with_multipass=False,
    )

    assert public_fit.u == expected_fit.u
    assert public_fit.v == expected_fit.v
    assert public_fit.offset == expected_fit.offset
    assert public_fit.rms == expected_fit.rms
    assert public_fit.iterations == expected_fit.iterations
    np.testing.assert_allclose(public_fit.reference, expected_fit.reference, equal_nan=True)
    np.testing.assert_allclose(public_result.velocity, expected_result.velocity, equal_nan=True)
    np.testing.assert_array_equal(public_result.folds, expected_result.folds)
    np.testing.assert_allclose(public_result.confidence, expected_result.confidence, equal_nan=True)
    np.testing.assert_allclose(public_result.reference, expected_result.reference, equal_nan=True)
    assert public_result.metadata == expected_result.metadata
