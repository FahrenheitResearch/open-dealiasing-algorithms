from __future__ import annotations

import numpy as np

from open_dealias import build_reference_from_uv, unfold_to_reference, wrap_to_nyquist
from open_dealias.region_graph import dealias_sweep_region_graph
from open_dealias.recursive import dealias_sweep_recursive


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def test_region_graph_recovers_fold_progression_better_than_coarse_reference():
    az = np.linspace(0.0, 360.0, 180, endpoint=False)
    rg = np.linspace(0.0, 1.0, 150)[None, :]
    azr = np.deg2rad(az)[:, None]

    background = build_reference_from_uv(az, 150, u=9.0, v=3.0)
    coarse_reference = background + 8.0 * rg
    truth = background + 26.0 * rg + 6.5 * np.exp(-((rg - 0.64) ** 2) / 0.015) * np.sin(3.0 * azr)
    obs = wrap_to_nyquist(truth, 10.0)

    baseline = unfold_to_reference(obs, coarse_reference, 10.0)
    result = dealias_sweep_region_graph(obs, 10.0, reference=coarse_reference)

    assert np.any(result.folds != 0)
    assert result.metadata["region_count"] > 4
    assert mae(result.velocity, truth) + 0.75 < mae(baseline, truth)


def test_recursive_refines_high_shear_split_case():
    az = np.linspace(0.0, 360.0, 180, endpoint=False)
    rg = np.linspace(0.0, 1.0, 160)[None, :]
    azr = np.deg2rad(az)[:, None]

    background = build_reference_from_uv(az, 160, u=8.0, v=-2.0)
    coarse_reference = background + 11.0 * rg
    lobes = (
        8.5 * np.exp(-((rg - 0.27) ** 2) / 0.010) * np.cos(4.0 * azr)
        + 10.5 * np.exp(-((rg - 0.76) ** 2) / 0.016) * np.sin(3.0 * azr)
    )
    truth = background + 25.0 * rg + lobes
    truth = truth.copy()
    truth[52:78, 68:98] = np.nan
    obs = wrap_to_nyquist(truth, 10.0)

    baseline = unfold_to_reference(obs, coarse_reference, 10.0)
    result = dealias_sweep_recursive(obs, 10.0, reference=coarse_reference)

    assert np.any(result.folds != 0)
    assert result.metadata["leaf_count"] >= 4
    assert mae(result.velocity, truth) + 0.60 < mae(baseline, truth)


def test_region_graph_skips_sparse_blocks_and_leaves_them_unresolved():
    obs = np.full((8, 8), np.nan, dtype=float)
    obs[0, 0] = 5.0
    obs[0, 1] = 6.0

    result = dealias_sweep_region_graph(obs, 10.0, block_shape=(4, 4))

    assert result.metadata["region_count"] == 0
    assert result.metadata["skipped_sparse_blocks"] >= 1
    assert result.metadata["min_region_area"] == 4
    assert result.metadata["min_valid_fraction"] == 0.15
    assert not np.any(np.isfinite(result.velocity))
