from __future__ import annotations

import numpy as np

from ._core import combine_references, fold_counts, shift2d, shift3d, unfold_to_reference
from ._rust_bridge import get_rust_backend
from .multipass import dealias_sweep_zw06
from .types import DealiasResult
from .vad import build_reference_from_uv


_NATIVE_BACKEND = get_rust_backend()


def _gate_stats(observed: np.ndarray, corrected: np.ndarray) -> dict[str, int | float]:
    valid = np.isfinite(observed)
    assigned = np.isfinite(corrected)
    valid_gates = int(np.sum(valid))
    assigned_gates = int(np.sum(assigned))
    unresolved_gates = int(np.sum(valid & ~assigned))
    resolved_fraction = float(assigned_gates / valid_gates) if valid_gates else 0.0
    return {
        'valid_gates': valid_gates,
        'assigned_gates': assigned_gates,
        'unresolved_gates': unresolved_gates,
        'resolved_fraction': resolved_fraction,
    }


def _fold_counts_by_sweep(corrected: np.ndarray, observed: np.ndarray, nyquist: float | np.ndarray) -> np.ndarray:
    if np.isscalar(nyquist):
        return fold_counts(corrected, observed, float(nyquist))
    nyq = np.asarray(nyquist, dtype=float)
    if nyq.ndim != 1 or nyq.shape[0] != corrected.shape[0]:
        raise ValueError('nyquist must be scalar or length n_sweeps')
    folds = np.zeros(corrected.shape, dtype=np.int16)
    for sweep in range(corrected.shape[0]):
        folds[sweep] = fold_counts(corrected[sweep], observed[sweep], float(nyq[sweep]))
    return folds


def _native_method(name: str):
    if _NATIVE_BACKEND is None:
        return None
    method = getattr(_NATIVE_BACKEND, name, None)
    return method if callable(method) else None


def _coerce_native_result(result, *, reference: np.ndarray | None) -> DealiasResult:
    if isinstance(result, DealiasResult):
        return result
    if len(result) == 5:
        velocity, folds, confidence, ref_out, metadata = result
    elif len(result) == 4:
        velocity, folds, confidence, metadata = result
        ref_out = reference
    else:
        raise ValueError("native dealiaser must return a 4- or 5-tuple")
    return DealiasResult(
        velocity=np.asarray(velocity, dtype=float),
        folds=np.asarray(folds, dtype=np.int16),
        confidence=np.asarray(confidence, dtype=float),
        reference=None if ref_out is None else np.asarray(ref_out, dtype=float),
        metadata=dict(metadata),
    )


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

    native = _native_method('dealias_sweep_jh01')
    if native is not None:
        native_result = native(
            obs,
            float(nyquist),
            None if previous_corrected is None else np.asarray(previous_corrected, dtype=float),
            None if background_reference is None else np.asarray(background_reference, dtype=float),
            int(shift_az),
            int(shift_range),
            bool(wrap_azimuth),
            bool(refine_with_multipass),
        )
        result = _coerce_native_result(native_result, reference=ref)
        result.reference = ref if result.reference is None else result.reference
        return result

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
                'fill_policy': 'temporal_reference_only',
                **_gate_stats(obs, first_guess),
            },
        )

    result = dealias_sweep_zw06(obs, nyquist, reference=ref, wrap_azimuth=wrap_azimuth)
    result.reference = ref
    result.metadata.update({
        'paper_family': 'JamesHouze2001+ZhangWang2006',
        'method': 'temporal_multipass',
        'shift_az': int(shift_az),
        'shift_range': int(shift_range),
        'fill_policy': 'temporal_reference_then_multipass_cleanup',
        **_gate_stats(obs, result.velocity),
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

    native = _native_method('dealias_volume_jh01')
    if native is not None:
        native_result = native(
            obs,
            nyq,
            az,
            el,
            None if previous_volume is None else np.asarray(previous_volume, dtype=float),
            background_uv,
            int(shift_az),
            int(shift_range),
            bool(wrap_azimuth),
        )
        return _coerce_native_result(native_result, reference=None)

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
        folds=_fold_counts_by_sweep(corrected, obs, nyq),
        confidence=confidence,
        reference=reference,
        metadata={
            'paper_family': 'JamesHouze2001',
            'method': 'descending_volume_4dd_lite',
            'elevation_order_desc': order.tolist(),
            'shift_az': int(shift_az),
            'shift_range': int(shift_range),
            'per_sweep': per_sweep_meta,
            'fill_policy': 'descending_volume_reference_then_cleanup',
            **_gate_stats(obs, corrected),
        },
    )
