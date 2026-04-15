from __future__ import annotations

import numpy as np

from ._core import combine_references, unfold_to_reference, wrap_to_nyquist
from .multipass import dealias_sweep_zw06
from .types import DealiasResult, VadFit


def build_reference_from_uv(
    azimuth_deg: np.ndarray,
    n_range: int,
    *,
    u: float,
    v: float,
    elevation_deg: float = 0.0,
    offset: float = 0.0,
    sign: float = 1.0,
) -> np.ndarray:
    """Build a sweep-wide radial-velocity reference from a uniform horizontal wind.

    The default sign convention assumes positive radial velocity is away from the radar.
    """
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    el = np.deg2rad(float(elevation_deg))
    radial = sign * (np.cos(el) * (u * np.sin(az) + v * np.cos(az)) + offset)
    return np.repeat(radial[:, None], int(n_range), axis=1)


def estimate_uniform_wind_vad(
    observed: np.ndarray,
    nyquist: float,
    azimuth_deg: np.ndarray,
    *,
    elevation_deg: float = 0.0,
    sign: float = 1.0,
    max_iterations: int = 6,
    trim_quantile: float = 0.85,
    search_radius: float | None = None,
) -> VadFit:
    """Estimate a uniform-wind VAD anchor with torus-aware search plus refit.

    This uses a coarse wrapped least-squares search in (u, v) space to find the
    correct Nyquist branch, then unwraps toward that candidate and refines with
    ordinary least squares. It is intentionally compact but much more robust than
    starting the fit from zero.
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2:
        raise ValueError('observed must be 2D [azimuth, range]')
    if nyquist <= 0:
        raise ValueError('nyquist must be positive')

    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    if az.ndim != 1 or az.size != obs.shape[0]:
        raise ValueError('azimuth_deg must be 1D and match observed.shape[0]')

    el = np.deg2rad(float(elevation_deg))
    x1 = sign * np.cos(el) * np.sin(az)
    x2 = sign * np.cos(el) * np.cos(az)
    X = np.column_stack([x1, x2, np.ones_like(x1)])

    # Collapse along range first; this mirrors the VAD use of azimuthal structure.
    y_wrapped = np.nanmedian(obs, axis=1)
    valid = np.isfinite(y_wrapped)
    if np.count_nonzero(valid) < 8:
        reference = np.zeros_like(obs)
        return VadFit(u=0.0, v=0.0, offset=0.0, rms=float('inf'), iterations=0, reference=reference)

    if search_radius is None:
        finite_obs = np.abs(obs[np.isfinite(obs)])
        amp = float(np.nanmax(finite_obs)) if finite_obs.size else nyquist
        search_radius = max(3.0 * nyquist, amp + 2.0 * nyquist, 25.0)

    # Coarse torus search.
    coarse = np.arange(-search_radius, search_radius + 1e-9, 1.0, dtype=float)
    pred = x1[valid, None, None] * coarse[None, :, None] + x2[valid, None, None] * coarse[None, None, :]
    residual = wrap_to_nyquist(y_wrapped[valid, None, None] - pred, nyquist)
    score = np.nanmean(residual**2, axis=0)
    iu, iv = np.unravel_index(np.nanargmin(score), score.shape)
    u = float(coarse[iu])
    v = float(coarse[iv])

    # Fine torus search around the coarse optimum.
    fine = np.arange(-2.0, 2.0001, 0.25, dtype=float)
    uf = u + fine
    vf = v + fine
    pred = x1[valid, None, None] * uf[None, :, None] + x2[valid, None, None] * vf[None, None, :]
    residual = wrap_to_nyquist(y_wrapped[valid, None, None] - pred, nyquist)
    score = np.nanmean(residual**2, axis=0)
    iu, iv = np.unravel_index(np.nanargmin(score), score.shape)
    u = float(uf[iu])
    v = float(vf[iv])
    offset = 0.0
    rms = float('inf')

    for iteration in range(1, max_iterations + 1):
        reference = build_reference_from_uv(azimuth_deg, obs.shape[1], u=u, v=v, elevation_deg=elevation_deg, offset=offset, sign=sign)
        unfolded = unfold_to_reference(obs, reference, nyquist)
        y = np.nanmedian(unfolded, axis=1)
        valid = np.isfinite(y)
        if np.count_nonzero(valid) < 8:
            break

        beta, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
        residual = y[valid] - X[valid] @ beta
        cut = np.quantile(np.abs(residual), trim_quantile)
        keep = valid.copy()
        keep[valid] = np.abs(residual) <= cut
        if np.count_nonzero(keep) >= 8:
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            residual = y[keep] - X[keep] @ beta
        rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
        new_u, new_v, new_offset = map(float, beta)
        if max(abs(new_u - u), abs(new_v - v), abs(new_offset - offset)) < 1e-3:
            u, v, offset = new_u, new_v, new_offset
            break
        u, v, offset = new_u, new_v, new_offset

    reference = build_reference_from_uv(azimuth_deg, obs.shape[1], u=u, v=v, elevation_deg=elevation_deg, offset=offset, sign=sign)
    return VadFit(u=u, v=v, offset=offset, rms=float(rms), iterations=int(iteration), reference=reference)


def dealias_sweep_xu11(
    observed: np.ndarray,
    nyquist: float,
    azimuth_deg: np.ndarray,
    *,
    elevation_deg: float = 0.0,
    external_reference: np.ndarray | None = None,
    sign: float = 1.0,
    refine_with_multipass: bool = True,
) -> DealiasResult:
    """Xu-style VAD-seeded dealiasing.

    1. estimate a uniform-wind VAD anchor,
    2. combine it with any external reference,
    3. unfold toward that reference,
    4. optionally refine with the 2D multipass solver.
    """
    obs = np.asarray(observed, dtype=float)
    fit = estimate_uniform_wind_vad(obs, nyquist, azimuth_deg, elevation_deg=elevation_deg, sign=sign)
    ref = combine_references(fit.reference, external_reference)
    if ref is None:
        ref = fit.reference

    if not refine_with_multipass:
        first_guess = unfold_to_reference(obs, ref, nyquist)
        confidence = np.where(np.isfinite(first_guess), 0.80, 0.0)
        from ._core import fold_counts
        return DealiasResult(
            velocity=first_guess,
            folds=fold_counts(first_guess, obs, nyquist),
            confidence=confidence,
            reference=ref,
            metadata={
                'paper_family': 'Xu2011',
                'method': 'vad_reference_only',
                'u': fit.u,
                'v': fit.v,
                'offset': fit.offset,
                'vad_rms': fit.rms,
                'vad_iterations': fit.iterations,
            },
        )

    result = dealias_sweep_zw06(obs, nyquist, reference=ref)
    result.reference = ref
    result.metadata.update({
        'paper_family': 'Xu2011+ZhangWang2006',
        'method': 'vad_seeded_multipass',
        'u': fit.u,
        'v': fit.v,
        'offset': fit.offset,
        'vad_rms': fit.rms,
        'vad_iterations': fit.iterations,
    })
    return result
