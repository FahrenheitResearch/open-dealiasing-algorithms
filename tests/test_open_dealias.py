from __future__ import annotations

import numpy as np

from open_dealias import (
    build_reference_from_uv,
    dealias_radial_es90,
    dealias_sweep_jh01,
    dealias_sweep_xu11,
    dealias_sweep_zw06,
    dealias_volume_jh01,
    estimate_uniform_wind_vad,
    wrap_to_nyquist,
)
from open_dealias.synthetic import make_folded_sweep, make_smooth_radial, make_temporal_pair


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))


def test_wrap_roundtrip_unfold_to_reference_simple():
    truth = np.array([-23.0, -5.0, 3.0, 28.0])
    obs = wrap_to_nyquist(truth, 10.0)
    ref = truth.copy()
    from open_dealias import unfold_to_reference

    unfolded = unfold_to_reference(obs, ref, 10.0)
    assert np.allclose(unfolded, truth)


def test_radial_es90_restores_smooth_radial():
    truth, obs = make_smooth_radial(n_gates=200, nyquist=12.0, seed=1)
    res = dealias_radial_es90(obs, 12.0)
    assert mae(res.velocity, truth) < 1.6


def test_zw06_uses_reference_to_recover_global_fold():
    az, truth, obs = make_folded_sweep(n_az=180, n_range=160, nyquist=10.0, seed=3)
    ref = build_reference_from_uv(az, obs.shape[1], u=18.0, v=0.0)
    res = dealias_sweep_zw06(obs, 10.0, reference=ref)
    assert mae(res.velocity, truth) < 2.0


def test_vad_fit_is_reasonable_on_uniform_background():
    az = np.linspace(0.0, 360.0, 180, endpoint=False)
    ref = build_reference_from_uv(az, 40, u=14.0, v=-4.0, elevation_deg=0.5)
    obs = wrap_to_nyquist(ref, 10.0)
    fit = estimate_uniform_wind_vad(obs, 10.0, az, elevation_deg=0.5)
    assert abs(fit.u - 14.0) < 1.0
    assert abs(fit.v + 4.0) < 1.0


def test_xu11_improves_over_unanchored_multipass_on_uniform_background():
    az = np.linspace(0.0, 360.0, 180, endpoint=False)
    ref = build_reference_from_uv(az, 50, u=18.0, v=0.0)
    obs = wrap_to_nyquist(ref, 10.0)
    plain = dealias_sweep_zw06(obs, 10.0)
    vad = dealias_sweep_xu11(obs, 10.0, az)
    assert mae(vad.velocity, ref) + 0.5 < mae(plain.velocity, ref)


def test_jh01_uses_previous_sweep_anchor():
    az, truth_prev, obs_prev, truth_curr, obs_curr = make_temporal_pair(n_az=180, n_range=120, nyquist=10.0, seed=4)
    res = dealias_sweep_jh01(obs_curr, 10.0, previous_corrected=truth_prev, shift_az=2)
    assert mae(res.velocity, truth_curr) < 1.0


def test_volume_jh01_runs_descending_and_returns_3d():
    az, truth_prev, obs_prev, truth_curr, obs_curr = make_temporal_pair(n_az=90, n_range=60, nyquist=10.0, seed=5)
    volume_obs = np.stack([obs_curr + 0.5, obs_curr], axis=0)
    prev_volume = np.stack([truth_prev + 0.5, truth_prev], axis=0)
    res = dealias_volume_jh01(
        volume_obs,
        10.0,
        azimuth_deg=az,
        elevation_deg=np.array([1.5, 0.5]),
        previous_volume=prev_volume,
        shift_az=2,
    )
    assert res.velocity.shape == volume_obs.shape
    assert len(res.metadata['elevation_order_desc']) == 2
