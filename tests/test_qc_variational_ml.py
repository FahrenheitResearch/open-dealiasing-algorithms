from __future__ import annotations

import numpy as np

from open_dealias import build_reference_from_uv, unfold_to_reference, wrap_to_nyquist
from open_dealias.ml import dealias_sweep_ml, fit_ml_reference_model
from open_dealias.qc import apply_velocity_qc, build_velocity_qc_mask, estimate_velocity_texture
from open_dealias.variational import dealias_sweep_variational


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def test_qc_mask_blanks_weak_noisy_gates():
    velocity = np.array(
        [
            [2.0, 2.1, 2.2, np.nan],
            [2.0, 11.0, -11.0, np.nan],
            [1.9, 2.0, 2.1, np.nan],
        ],
        dtype=float,
    )
    reflectivity = np.array(
        [
            [20.0, 18.0, 17.0, -20.0],
            [21.0, -15.0, -15.0, -20.0],
            [19.0, 18.0, 18.0, -20.0],
        ],
        dtype=float,
    )
    texture = estimate_velocity_texture(velocity)
    mask = build_velocity_qc_mask(
        velocity,
        reflectivity=reflectivity,
        texture=texture,
        min_reflectivity=-5.0,
        max_texture=6.0,
        min_gate_fraction_in_ray=0.0,
    )
    filtered = apply_velocity_qc(velocity, mask=mask)

    assert mask[1, 1] == 0
    assert mask[1, 2] == 0
    assert np.isnan(filtered[1, 1])
    assert np.isfinite(filtered[0, 1])


def test_variational_improves_over_coarse_reference_branch():
    az = np.linspace(0.0, 360.0, 120, endpoint=False)
    rg = np.linspace(0.0, 1.0, 110)[None, :]
    azr = np.deg2rad(az)[:, None]
    background = build_reference_from_uv(az, 110, u=7.0, v=2.0)
    coarse_reference = background + 9.5 * rg
    truth = background + 23.0 * rg + 5.5 * np.sin(3.0 * azr) * np.exp(-((rg - 0.58) ** 2) / 0.02)
    obs = wrap_to_nyquist(truth, 10.0)

    baseline = unfold_to_reference(obs, coarse_reference, 10.0)
    result = dealias_sweep_variational(obs, 10.0, reference=coarse_reference)

    assert np.any(result.folds != 0)
    assert result.metadata["iterations_used"] >= 1
    assert mae(result.velocity, truth) + 0.8 < mae(baseline, truth)


def test_ml_reference_model_fits_and_predicts_training_case():
    az = np.linspace(0.0, 360.0, 100, endpoint=False)
    rg = np.linspace(0.0, 1.0, 80)[None, :]
    azr = np.deg2rad(az)[:, None]
    truth = build_reference_from_uv(az, 80, u=13.0, v=-4.0) + 2.0 * np.sin(2.0 * azr) * np.exp(-((rg - 0.3) ** 2) / 0.04)
    obs = wrap_to_nyquist(truth, 10.0)

    model = fit_ml_reference_model(obs, truth, nyquist=10.0, azimuth_deg=az)
    result = dealias_sweep_ml(obs, 10.0, model=model, azimuth_deg=az, refine_with_variational=False)

    assert model.train_rmse < 0.3
    assert mae(result.velocity, truth) < 0.3


def test_ml_assisted_solver_generalizes_to_related_case():
    az = np.linspace(0.0, 360.0, 120, endpoint=False)
    rg = np.linspace(0.0, 1.0, 90)[None, :]
    azr = np.deg2rad(az)[:, None]

    train_truth = build_reference_from_uv(az, 90, u=12.0, v=-3.0) + 2.5 * np.sin(2.0 * azr) * np.exp(-((rg - 0.35) ** 2) / 0.05)
    train_obs = wrap_to_nyquist(train_truth, 10.0)
    model = fit_ml_reference_model(train_obs, train_truth, nyquist=10.0, azimuth_deg=az)

    test_truth = build_reference_from_uv(az, 90, u=13.0, v=-2.0) + 3.0 * np.sin(2.0 * azr) * np.exp(-((rg - 0.42) ** 2) / 0.05) + 20.0 * rg
    test_obs = wrap_to_nyquist(test_truth, 10.0)
    coarse_reference = build_reference_from_uv(az, 90, u=10.0, v=0.0)
    baseline = unfold_to_reference(test_obs, coarse_reference, 10.0)
    result = dealias_sweep_ml(
        test_obs,
        10.0,
        model=model,
        reference=coarse_reference,
        azimuth_deg=az,
        refine_with_variational=True,
    )

    assert np.any(result.folds != 0)
    assert mae(result.velocity, test_truth) + 1.0 < mae(baseline, test_truth)
