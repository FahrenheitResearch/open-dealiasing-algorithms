from __future__ import annotations

import numpy as np
import pytest

import open_dealias.ml as ml_module
from open_dealias import build_reference_from_uv, fit_ml_reference_model, dealias_sweep_ml, wrap_to_nyquist


def _make_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    az = np.linspace(0.0, 360.0, 8, endpoint=False)
    rg = np.linspace(0.0, 1.0, 6)[None, :]
    azr = np.deg2rad(az)[:, None]
    reference = build_reference_from_uv(az, 6, u=9.0, v=2.0) + 4.0 * rg
    target = build_reference_from_uv(az, 6, u=11.0, v=1.0) + 18.0 * rg + 2.5 * np.sin(2.0 * azr) * np.exp(-((rg - 0.45) ** 2) / 0.04)
    observed = wrap_to_nyquist(target, 10.0)
    return az, observed, target, reference


def test_fit_ml_reference_model_dispatches_to_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    az, observed, target, reference = _make_case()

    monkeypatch.setattr(ml_module, "_NATIVE_BACKEND", None)
    expected = fit_ml_reference_model(
        observed,
        target,
        nyquist=10.0,
        reference=reference,
        azimuth_deg=az,
        ridge=0.75,
    )

    class FakeBackend:
        def fit_ml_reference_model(self, obs, tgt, *, nyquist, reference, azimuth_deg, ridge):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            np.testing.assert_allclose(tgt, target, equal_nan=True)
            np.testing.assert_allclose(reference, reference_field, equal_nan=True)
            np.testing.assert_allclose(azimuth_deg, az, equal_nan=True)
            assert nyquist == pytest.approx(10.0)
            assert ridge == pytest.approx(0.75)
            return (
                expected.weights,
                expected.feature_names,
                expected.ridge,
                expected.train_rmse,
                expected.mode,
                expected.nyquist,
            )

    reference_field = reference.copy()
    monkeypatch.setattr(ml_module, "_NATIVE_BACKEND", FakeBackend())

    model = fit_ml_reference_model(
        observed,
        target,
        nyquist=10.0,
        reference=reference_field,
        azimuth_deg=az,
        ridge=0.75,
    )

    assert np.allclose(model.weights, expected.weights, equal_nan=True)
    assert model.feature_names == expected.feature_names
    assert model.ridge == pytest.approx(expected.ridge)
    assert model.train_rmse == pytest.approx(expected.train_rmse)
    assert model.mode == expected.mode
    assert model.nyquist == pytest.approx(expected.nyquist)


def test_dealias_sweep_ml_dispatches_to_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    az, observed, target, reference = _make_case()

    monkeypatch.setattr(ml_module, "_NATIVE_BACKEND", None)
    expected_model = fit_ml_reference_model(
        observed,
        target,
        nyquist=10.0,
        reference=reference,
        azimuth_deg=az,
        ridge=0.9,
    )
    expected = dealias_sweep_ml(
        observed,
        10.0,
        model=expected_model,
        reference=reference,
        azimuth_deg=az,
        refine_with_variational=False,
    )

    class FakeBackend:
        def dealias_sweep_ml(self, obs, nyquist, *, model, training_target, reference, azimuth_deg, ridge, refine_with_variational):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            assert nyquist == pytest.approx(10.0)
            assert isinstance(model, type(expected_model))
            np.testing.assert_allclose(model.weights, expected_model.weights, equal_nan=True)
            assert model.feature_names == expected_model.feature_names
            assert model.ridge == pytest.approx(expected_model.ridge)
            assert model.train_rmse == pytest.approx(expected_model.train_rmse)
            assert model.mode == expected_model.mode
            assert model.nyquist == pytest.approx(expected_model.nyquist)
            assert training_target is None
            np.testing.assert_allclose(reference, reference_field, equal_nan=True)
            np.testing.assert_allclose(azimuth_deg, az, equal_nan=True)
            assert ridge == pytest.approx(1.0)
            assert refine_with_variational is False
            return (
                expected.velocity,
                expected.folds,
                expected.confidence,
                expected.reference,
                dict(expected.metadata),
            )

    reference_field = reference.copy()
    monkeypatch.setattr(ml_module, "_NATIVE_BACKEND", FakeBackend())

    result = dealias_sweep_ml(
        observed,
        10.0,
        model=expected_model,
        reference=reference_field,
        azimuth_deg=az,
        refine_with_variational=False,
    )

    np.testing.assert_allclose(result.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(result.folds, expected.folds)
    np.testing.assert_allclose(result.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(result.reference, expected.reference, equal_nan=True)
    assert result.metadata == expected.metadata


def test_dealias_sweep_ml_dispatches_with_training_target(monkeypatch: pytest.MonkeyPatch) -> None:
    az, observed, target, reference = _make_case()

    monkeypatch.setattr(ml_module, "_NATIVE_BACKEND", None)
    expected = dealias_sweep_ml(
        observed,
        10.0,
        training_target=target,
        reference=reference,
        azimuth_deg=az,
        refine_with_variational=False,
    )

    class FakeBackend:
        def dealias_sweep_ml(self, obs, nyquist, *, model, training_target, reference, azimuth_deg, ridge, refine_with_variational):
            np.testing.assert_allclose(obs, observed, equal_nan=True)
            assert nyquist == pytest.approx(10.0)
            assert model is None
            np.testing.assert_allclose(training_target, target, equal_nan=True)
            np.testing.assert_allclose(reference, reference_field, equal_nan=True)
            np.testing.assert_allclose(azimuth_deg, az, equal_nan=True)
            assert ridge == pytest.approx(1.0)
            assert refine_with_variational is False
            return (
                expected.velocity,
                expected.folds,
                expected.confidence,
                expected.reference,
                dict(expected.metadata),
            )

    reference_field = reference.copy()
    monkeypatch.setattr(ml_module, "_NATIVE_BACKEND", FakeBackend())

    result = dealias_sweep_ml(
        observed,
        10.0,
        training_target=target,
        reference=reference_field,
        azimuth_deg=az,
        refine_with_variational=False,
    )

    np.testing.assert_allclose(result.velocity, expected.velocity, equal_nan=True)
    np.testing.assert_array_equal(result.folds, expected.folds)
    np.testing.assert_allclose(result.confidence, expected.confidence, equal_nan=True)
    np.testing.assert_allclose(result.reference, expected.reference, equal_nan=True)
    assert result.metadata == expected.metadata
