from __future__ import annotations

import numpy as np

from open_dealias import build_reference_from_uv, dealias_sweep_es90, dealias_sweep_jh01, wrap_to_nyquist


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def test_sweep_es90_uses_reference_and_prior_ray_context():
    az = np.linspace(0.0, 360.0, 90, endpoint=False)
    rg = np.linspace(-1.0, 1.0, 70)[None, :]
    azr = np.deg2rad(az)[:, None]
    background = build_reference_from_uv(az, 70, u=16.0, v=5.0)
    truth = background + 3.0 * np.sin(3.0 * azr) * np.exp(-((rg - 0.25) ** 2) / 0.07)
    obs = wrap_to_nyquist(truth, 10.0)
    obs[6:10, 18:24] = np.nan

    res = dealias_sweep_es90(obs, 10.0, reference=background, max_gap=4, max_abs_step=8.0)
    assert mae(res.velocity, truth) < 0.9


def test_jh01_can_cold_start_from_background_reference():
    az = np.linspace(0.0, 360.0, 120, endpoint=False)
    azr = np.deg2rad(az)[:, None]
    rg = np.linspace(-1.0, 1.0, 90)[None, :]
    background = build_reference_from_uv(az, 90, u=14.0, v=-4.0)
    truth = background + 2.5 * np.cos(2.0 * azr) * np.exp(-((rg + 0.15) ** 2) / 0.05)
    obs = wrap_to_nyquist(truth, 10.0)

    res = dealias_sweep_jh01(obs, 10.0, background_reference=background)
    assert mae(res.velocity, truth) < 1.2
