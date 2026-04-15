from __future__ import annotations

import numpy as np

from open_dealias import build_reference_from_uv, wrap_to_nyquist
from open_dealias.dual_prf import dealias_dual_prf
from open_dealias.volume3d import dealias_volume_3d


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def test_dual_prf_recovers_high_branch_from_paired_observations():
    az = np.linspace(0.0, 360.0, 120, endpoint=False)
    azr = np.deg2rad(az)[:, None]
    rg = np.linspace(-1.0, 1.0, 96)[None, :]
    background = build_reference_from_uv(az, 96, u=6.0, v=-3.0, elevation_deg=0.5)
    storm = 4.0 * np.exp(-((rg - 0.18) ** 2) / 0.03) * np.sin(2.0 * azr)
    truth = background + storm + 22.0

    low = wrap_to_nyquist(truth, 10.0)
    high = wrap_to_nyquist(truth, 16.0)
    res = dealias_dual_prf(low, high, 10.0, 16.0)

    assert res.metadata['method'] == 'dual_prf_pair_search'
    assert res.metadata['paired_gates'] == truth.size
    assert mae(res.velocity, truth) < 1e-6
    assert np.all(res.confidence > 0.95)


def test_dual_prf_falls_back_to_single_sweep_reference_when_partner_missing():
    truth = np.array([-26.0, -14.0, -2.0, 11.0, 23.0, 35.0], dtype=float)
    low = wrap_to_nyquist(truth, 10.0)
    high = np.full_like(low, np.nan)
    res = dealias_dual_prf(low, high, 10.0, 16.0, reference=truth)

    assert res.metadata['paired_gates'] == 0
    assert mae(res.velocity, truth) < 1e-6
    assert np.all(res.confidence > 0.75)


def test_volume3d_propagates_branch_through_adjacent_sweeps():
    az = np.linspace(0.0, 360.0, 96, endpoint=False)
    azr = np.deg2rad(az)[:, None]
    rg = np.linspace(0.0, 1.0, 72)[None, :]
    background = build_reference_from_uv(az, 72, u=5.0, v=4.0, elevation_deg=1.0)
    blob = 3.5 * np.exp(-((rg - 0.55) ** 2) / 0.02) * np.cos(3.0 * azr)

    truth = np.stack(
        [
            background + blob + 1.0,
            background + blob + 9.0,
            background + blob + 17.0,
        ],
        axis=0,
    )
    observed = wrap_to_nyquist(truth, 10.0)
    observed[1, 18:24, 28:34] = np.nan
    observed[2, 30:36, 40:46] = np.nan

    reference = np.full_like(truth, np.nan)
    reference[0] = truth[0]

    res = dealias_volume_3d(observed, 10.0, reference_volume=reference, max_iterations=4)

    assert res.metadata['seed_sweep'] == 0
    assert res.velocity.shape == truth.shape
    assert mae(res.velocity, truth) < 0.9
    assert mae(res.velocity[2], truth[2]) < 0.9


def test_volume3d_accepts_per_sweep_nyquist_and_keeps_shape():
    az = np.linspace(0.0, 360.0, 60, endpoint=False)
    azr = np.deg2rad(az)[:, None]
    rg = np.linspace(0.0, 1.0, 40)[None, :]
    base = build_reference_from_uv(az, 40, u=3.0, v=-2.0, elevation_deg=0.7)
    truth = np.stack(
        [
            base + 2.0 * np.sin(2.0 * azr) * np.exp(-((rg - 0.2) ** 2) / 0.04) + 4.0,
            base + 2.0 * np.sin(2.0 * azr) * np.exp(-((rg - 0.2) ** 2) / 0.04) + 13.0,
        ],
        axis=0,
    )
    nyquist = np.array([10.0, 12.0], dtype=float)
    observed = np.stack([wrap_to_nyquist(truth[0], nyquist[0]), wrap_to_nyquist(truth[1], nyquist[1])], axis=0)
    reference = np.full_like(truth, np.nan)
    reference[0] = truth[0]

    res = dealias_volume_3d(observed, nyquist, reference_volume=reference)

    assert res.velocity.shape == truth.shape
    assert mae(res.velocity, truth) < 0.9
    assert res.folds.shape == truth.shape
