from __future__ import annotations

import numpy as np

from ._core import combine_references, fold_counts, shift2d, shift3d, unfold_to_reference
from .multipass import dealias_sweep_zw06
from .types import DealiasResult
from .vad import build_reference_from_uv


def dealias_sweep_jh01(
    observed: np.ndarray,
    nyquist: float,
    previous_corrected: np.ndarray | None = None,
    *,
    background_reference: np.ndarray | None = None,
    shift_az: int = 0,
    shift_range: int = 0,
    wrap_azimuth: bool = True,
    refine_with_multipass: bool = True,
) -> DealiasResult:
    """James–Houze-style temporal / 4DD-assisted sweep dealiasing."""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2:
        raise ValueError('observed must be 2D')

    prev = None
    if previous_corrected is not None:
        prev = shift2d(previous_corrected, shift_az=shift_az, shift_range=shift_range, wrap_azimuth=wrap_azimuth)
    ref = combine_references(prev, background_reference)
    if ref is None:
        raise ValueError('previous_corrected or background_reference is required')

    first_guess = unfold_to_reference(obs, ref, nyquist)
    if not refine_with_multipass:
        confidence = np.where(np.isfinite(first_guess), 0.85, 0.0)
        return DealiasResult(
            velocity=first_guess,
            folds=fold_counts(first_guess, obs, nyquist),
            confidence=confidence,
            reference=ref,
            metadata={
                'paper_family': 'JamesHouze2001',
                'method': 'temporal_reference_only',
                'shift_az': int(shift_az),
                'shift_range': int(shift_range),
            },
        )

    result = dealias_sweep_zw06(obs, nyquist, reference=ref, wrap_azimuth=wrap_azimuth)
    result.reference = ref
    result.metadata.update({
        'paper_family': 'JamesHouze2001+ZhangWang2006',
        'method': 'temporal_multipass',
        'shift_az': int(shift_az),
        'shift_range': int(shift_range),
    })
    return result


def dealias_volume_jh01(
    observed_volume: np.ndarray,
    nyquist: float | np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    *,
    previous_volume: np.ndarray | None = None,
    background_uv: tuple[float, float] | tuple[np.ndarray, np.ndarray] | None = None,
    shift_az: int = 0,
    shift_range: int = 0,
    wrap_azimuth: bool = True,
) -> DealiasResult:
    """A compact descending-elevation 4DD-style volume solver.

    The sweep order follows the James–Houze idea of starting aloft, where clutter
    is lower and gate-to-gate shear is often easier to manage.
    """
    obs = np.asarray(observed_volume, dtype=float)
    if obs.ndim != 3:
        raise ValueError('observed_volume must be 3D [sweep, azimuth, range]')
    az = np.asarray(azimuth_deg, dtype=float)
    el = np.asarray(elevation_deg, dtype=float)
    if el.ndim != 1 or el.size != obs.shape[0]:
        raise ValueError('elevation_deg must be length n_sweeps')
    if az.ndim != 1 or az.size != obs.shape[1]:
        raise ValueError('azimuth_deg must be length n_azimuth')

    if np.isscalar(nyquist):
        nyq = np.full(obs.shape[0], float(nyquist), dtype=float)
    else:
        nyq = np.asarray(nyquist, dtype=float)
        if nyq.shape != (obs.shape[0],):
            raise ValueError('nyquist must be scalar or length n_sweeps')

    corrected = np.full(obs.shape, np.nan, dtype=float)
    confidence = np.zeros(obs.shape, dtype=float)
    reference = np.full(obs.shape, np.nan, dtype=float)
    per_sweep_meta: list[dict[str, object]] = [dict() for _ in range(obs.shape[0])]

    shifted_prev = None if previous_volume is None else shift3d(previous_volume, shift_az=shift_az, shift_range=shift_range, wrap_azimuth=wrap_azimuth)
    order = np.argsort(el)[::-1]

    for rank, sweep in enumerate(order):
        bg_ref = None
        if background_uv is not None:
            if isinstance(background_uv[0], np.ndarray):
                u = float(np.asarray(background_uv[0], dtype=float)[sweep])
                v = float(np.asarray(background_uv[1], dtype=float)[sweep])
            else:
                u = float(background_uv[0])
                v = float(background_uv[1])
            bg_ref = build_reference_from_uv(az, obs.shape[2], u=u, v=v, elevation_deg=float(el[sweep]))

        prev_ref = None if shifted_prev is None else shifted_prev[sweep]
        higher_ref = None
        if rank > 0:
            higher_ref = corrected[order[rank - 1]]

        res = dealias_sweep_jh01(
            obs[sweep],
            float(nyq[sweep]),
            previous_corrected=combine_references(prev_ref, higher_ref),
            background_reference=bg_ref,
            shift_az=0,
            shift_range=0,
            wrap_azimuth=wrap_azimuth,
            refine_with_multipass=True,
        )
        corrected[sweep] = res.velocity
        confidence[sweep] = res.confidence
        if res.reference is not None:
            reference[sweep] = res.reference
        per_sweep_meta[sweep] = res.metadata

    return DealiasResult(
        velocity=corrected,
        folds=fold_counts(corrected, obs, float(np.max(nyq))),
        confidence=confidence,
        reference=reference,
        metadata={
            'paper_family': 'JamesHouze2001',
            'method': 'descending_volume_4dd_lite',
            'elevation_order_desc': order.tolist(),
            'shift_az': int(shift_az),
            'shift_range': int(shift_range),
            'per_sweep': per_sweep_meta,
        },
    )
