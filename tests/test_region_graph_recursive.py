from __future__ import annotations

import numpy as np

from open_dealias import build_reference_from_uv, unfold_to_reference, wrap_to_nyquist
from open_dealias._rust_bridge import backend_policy
from open_dealias.region_graph import dealias_sweep_region_graph
from open_dealias.recursive import dealias_sweep_recursive
from open_dealias.variational import dealias_sweep_variational


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _build_missing_wedge_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, slice]]:
    az = np.linspace(0.0, 360.0, 144, endpoint=False)
    rg = np.linspace(0.0, 1.0, 128)[None, :]
    azr = np.deg2rad(az)[:, None]

    background = build_reference_from_uv(az, 128, u=8.5, v=-1.5)
    truth = background + 24.0 * rg + 4.0 * np.sin(3.0 * azr)
    truth += 8.0 * np.exp(-((rg - 0.72) ** 2) / 0.006) * np.sin(azr - 1.1)
    truth = truth.copy()
    truth[24:62, 64:110] += 20.0
    truth[34:78, 82:118] = np.nan

    observed = wrap_to_nyquist(truth, 10.0)
    sector = (slice(20, 66), slice(60, 112))
    return truth, observed, truth.copy(), sector


def _sector_sign_mismatch_count(
    truth: np.ndarray,
    candidate: np.ndarray,
    sector: tuple[slice, slice],
) -> int:
    truth_sector = truth[sector]
    candidate_sector = candidate[sector]
    mask = np.isfinite(truth_sector) & np.isfinite(candidate_sector)
    if not np.any(mask):
        return 0
    return int(np.count_nonzero(np.signbit(truth_sector[mask]) != np.signbit(candidate_sector[mask])))


def test_region_graph_recovers_fold_progression_better_than_coarse_reference():
    az = np.linspace(0.0, 360.0, 180, endpoint=False)
    rg = np.linspace(0.0, 1.0, 150)[None, :]
    azr = np.deg2rad(az)[:, None]

    background = build_reference_from_uv(az, 150, u=9.0, v=3.0)
    coarse_reference = background + 8.0 * rg
    truth = background + 26.0 * rg + 6.5 * np.exp(-((rg - 0.64) ** 2) / 0.015) * np.sin(3.0 * azr)
    obs = wrap_to_nyquist(truth, 10.0)

    baseline = unfold_to_reference(obs, coarse_reference, 10.0)
    with backend_policy("python"):
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

    with backend_policy("python"):
        result = dealias_sweep_region_graph(obs, 10.0, block_shape=(4, 4))

    assert result.metadata["region_count"] == 1
    assert result.metadata["seedable_region_count"] == 0
    assert result.metadata["assigned_regions"] == 0
    assert result.metadata["unresolved_region_count"] == 1
    assert result.metadata["skipped_sparse_blocks"] >= 1
    assert result.metadata["min_region_area"] == 4
    assert result.metadata["min_valid_fraction"] == 0.15
    assert not np.any(np.isfinite(result.velocity))


def test_region_graph_missing_wedge_stays_conservative_without_opposite_sign_sector():
    truth, obs, reference, sector = _build_missing_wedge_case()

    with backend_policy("python"):
        result = dealias_sweep_region_graph(
            obs,
            10.0,
            reference=reference,
            block_shape=(8, 8),
            min_region_area=8,
            min_valid_fraction=0.30,
        )

    overlap = np.isfinite(truth) & np.isfinite(result.velocity)
    assert np.any(overlap)
    assert _sector_sign_mismatch_count(truth, result.velocity, sector) == 0
    assert result.metadata["skipped_sparse_blocks"] >= 1
    assert result.metadata["assigned_regions"] < result.metadata["region_count"]
    assert result.result_state.unresolved_gates > 0
    assert mae(result.velocity[overlap], truth[overlap]) < 1.5


def test_variational_missing_wedge_bootstrap_does_not_create_opposite_sign_sector():
    truth, obs, reference, sector = _build_missing_wedge_case()

    with backend_policy("python"):
        region_graph = dealias_sweep_region_graph(
            obs,
            10.0,
            reference=reference,
            block_shape=(8, 8),
            min_region_area=8,
            min_valid_fraction=0.30,
        )
        variational = dealias_sweep_variational(obs, 10.0, reference=reference, max_iterations=6)

    assert variational.metadata["bootstrap_method"] == "2d_multipass"
    assert _sector_sign_mismatch_count(truth, variational.velocity, sector) == 0
    assert variational.result_state.unresolved_gates <= region_graph.result_state.unresolved_gates



def test_variational_defaults_to_zw06_bootstrap():
    az = np.linspace(0.0, 360.0, 120, endpoint=False)
    truth = build_reference_from_uv(az, 64, u=14.0, v=-3.0)
    obs = wrap_to_nyquist(truth, 10.0)

    with backend_policy("python"):
        result = dealias_sweep_variational(obs, 10.0)

    assert result.metadata["bootstrap_method"] in {"2d_multipass", "zw06"}
