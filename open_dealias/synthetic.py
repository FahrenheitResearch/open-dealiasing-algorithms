from __future__ import annotations

import numpy as np

from ._core import wrap_to_nyquist


def make_smooth_radial(n_gates: int = 200, nyquist: float = 12.0, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_gates)
    truth = 18.0 * np.sin(1.7 * np.pi * x) + 4.0 * x + 0.2 * rng.normal(size=n_gates)
    observed = wrap_to_nyquist(truth, nyquist)
    return truth, observed


def make_folded_sweep(
    n_az: int = 180,
    n_range: int = 220,
    nyquist: float = 10.0,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    azr = np.deg2rad(az)
    rg = np.linspace(0.0, 1.0, n_range)
    background = 18.0 * np.sin(azr)[:, None]
    shear = (12.0 * rg[None, :]) * np.cos(2.0 * azr)[:, None]
    meso = 9.0 * np.exp(-((rg[None, :] - 0.55) ** 2) / 0.02) * np.sin(3.0 * azr)[:, None]
    truth = background + shear + meso + 0.4 * rng.normal(size=(n_az, n_range))
    observed = wrap_to_nyquist(truth, nyquist)
    return az, truth, observed


def make_temporal_pair(
    n_az: int = 180,
    n_range: int = 220,
    nyquist: float = 10.0,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    az, truth_prev, obs_prev = make_folded_sweep(n_az=n_az, n_range=n_range, nyquist=nyquist, seed=seed)
    truth_curr = np.roll(truth_prev, shift=2, axis=0) + 0.8 * np.linspace(-1, 1, n_range)[None, :]
    obs_curr = wrap_to_nyquist(truth_curr, nyquist)
    return az, truth_prev, obs_prev, truth_curr, obs_curr
