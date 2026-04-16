from __future__ import annotations

from typing import Iterable

import numpy as np

from ._core import combine_references, gaussian_confidence, unfold_to_reference
from .result_state import attach_result_state_from_fields
from ._rust_bridge import get_rust_backend, resolve_rust_backend
from .multipass import dealias_sweep_zw06
from .types import DealiasResult

__all__ = ['dealias_volume_3d']


_NATIVE_BACKEND = get_rust_backend()


def _resolve_nyquist(nyquist: float | np.ndarray, n_sweeps: int) -> np.ndarray:
    if np.isscalar(nyquist):
        value = float(nyquist)
        if value <= 0:
            raise ValueError('nyquist must be positive')
        return np.full(n_sweeps, value, dtype=float)
    arr = np.asarray(nyquist, dtype=float)
    if arr.shape != (n_sweeps,):
        raise ValueError('nyquist must be scalar or length n_sweeps')
    if np.any(arr <= 0):
        raise ValueError('nyquist values must be positive')
    return arr


def _fold_counts_volume(corrected: np.ndarray, observed: np.ndarray, nyquist: np.ndarray) -> np.ndarray:
    nyq = np.asarray(nyquist, dtype=float)
    if nyq.ndim == 0:
        folds = np.rint((np.asarray(corrected, dtype=float) - np.asarray(observed, dtype=float)) / (2.0 * float(nyq)))
    else:
        folds = np.rint((np.asarray(corrected, dtype=float) - np.asarray(observed, dtype=float)) / (2.0 * nyq[:, None, None]))
    folds = np.where(np.isfinite(folds), folds, 0.0)
    return folds.astype(np.int16)


def _choose_seed_sweep(observed: np.ndarray, reference_volume: np.ndarray | None) -> int:
    valid_counts = np.sum(np.isfinite(observed), axis=(1, 2))
    if reference_volume is not None:
        ref_counts = np.sum(np.isfinite(reference_volume), axis=(1, 2))
        if np.any(ref_counts > 0):
            ranked = np.lexsort((np.arange(observed.shape[0]), -ref_counts))
            return int(ranked[0])
    if np.any(valid_counts > 0):
        ranked = np.lexsort((np.abs(np.arange(observed.shape[0]) - observed.shape[0] // 2), -valid_counts))
        return int(ranked[0])
    return observed.shape[0] // 2


def _sweep_reference(
    corrected: np.ndarray,
    sweep_index: int,
    *,
    reference_volume: np.ndarray | None = None,
) -> np.ndarray | None:
    refs: list[np.ndarray | None] = []
    if reference_volume is not None:
        refs.append(reference_volume[sweep_index])
    if sweep_index > 0:
        refs.append(corrected[sweep_index - 1])
    if sweep_index + 1 < corrected.shape[0]:
        refs.append(corrected[sweep_index + 1])
    refs.append(corrected[sweep_index])
    return combine_references(*refs)


def _sweep_order_from_seed(seed: int, n_sweeps: int) -> list[int]:
    order = [seed]
    step = 1
    while seed - step >= 0 or seed + step < n_sweeps:
        if seed + step < n_sweeps:
            order.append(seed + step)
        if seed - step >= 0:
            order.append(seed - step)
        step += 1
    return order


def _refine_volume_pass(
    observed: np.ndarray,
    nyquist: np.ndarray,
    corrected: np.ndarray,
    confidence: np.ndarray,
    reference_volume: np.ndarray | None,
    order: list[int],
    *,
    wrap_azimuth: bool,
) -> tuple[int, list[dict[str, object]]]:
    changed = 0
    per_sweep_meta: list[dict[str, object]] = [dict() for _ in range(observed.shape[0])]
    for sweep in order:
        sweep_ref = _sweep_reference(corrected, sweep, reference_volume=reference_volume)
        if sweep_ref is None and np.any(np.isfinite(corrected[sweep])):
            sweep_ref = corrected[sweep]
        if sweep_ref is None and not np.any(np.isfinite(observed[sweep])):
            continue

        if sweep_ref is None:
            res = dealias_sweep_zw06(observed[sweep], float(nyquist[sweep]), wrap_azimuth=wrap_azimuth)
        else:
            res = dealias_sweep_zw06(observed[sweep], float(nyquist[sweep]), reference=sweep_ref, wrap_azimuth=wrap_azimuth)

        if np.any(~np.isclose(res.velocity, corrected[sweep], equal_nan=True)):
            changed += int(np.sum(np.isfinite(res.velocity) & ~np.isclose(res.velocity, corrected[sweep], equal_nan=True)))

        corrected[sweep] = res.velocity
        confidence[sweep] = np.maximum(confidence[sweep], res.confidence)
        per_sweep_meta[sweep] = dict(res.metadata)

    return changed, per_sweep_meta


def dealias_volume_3d(
    observed_volume: Iterable[float] | np.ndarray,
    nyquist: float | np.ndarray,
    *,
    reference_volume: np.ndarray | None = None,
    wrap_azimuth: bool = True,
    max_iterations: int = 4,
) -> DealiasResult:
    """3D continuity solver that links sweeps through adjacent-volume context.

    The implementation is intentionally modular and public-facing:
    - seed one sweep from a trusted reference when available,
    - propagate outward sweep by sweep using adjacent-sweep references,
    - let the existing 2D multipass solver handle in-sweep continuity.
    """
    obs = np.asarray(observed_volume, dtype=float)
    if obs.ndim != 3:
        raise ValueError('observed_volume must be 3D [sweep, azimuth, range]')

    ref = None if reference_volume is None else np.asarray(reference_volume, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError('reference_volume must match observed_volume shape')

    nyq = _resolve_nyquist(nyquist, obs.shape[0])

    backend = resolve_rust_backend(_NATIVE_BACKEND)
    if backend is not None and hasattr(backend, 'dealias_volume_3d'):
        native_result = backend.dealias_volume_3d(
            obs,
            nyq,
            ref,
            bool(wrap_azimuth),
            int(max_iterations),
        )
        if isinstance(native_result, DealiasResult):
            return native_result
        values = tuple(native_result)
        if len(values) == 5:
            velocity, folds, confidence, ref_out, metadata = values
        elif len(values) == 4:
            velocity, folds, confidence, metadata = values
            ref_out = ref
        else:  # pragma: no cover - defensive against future API drift.
            raise ValueError('native volume3d backend returned an unexpected result shape')
        meta = dict(metadata)
        meta.setdefault('paper_family', 'UNRAVEL-style-3D')
        meta.setdefault('method', 'volume_3d_continuity')
        meta.setdefault('wrap_azimuth', bool(wrap_azimuth))
        meta.setdefault('max_iterations', int(max_iterations))
        return attach_result_state_from_fields(
            DealiasResult(
            velocity=np.asarray(velocity, dtype=float),
            folds=np.asarray(folds, dtype=np.int16),
            confidence=np.asarray(confidence, dtype=float),
            reference=None if ref_out is None else np.asarray(ref_out, dtype=float),
            metadata=meta,
            ),
            obs,
            source="volume_3d_continuity",
            parent="UNRAVEL-style-3D",
            fill_policy=str(meta.get("fill_policy", "volume_reference_then_cleanup")),
        )

    corrected = np.full(obs.shape, np.nan, dtype=float)
    confidence = np.zeros(obs.shape, dtype=float)
    reference = np.full(obs.shape, np.nan, dtype=float)

    seed = _choose_seed_sweep(obs, ref)
    seed_ref = None if ref is None else ref[seed]
    seed_result = dealias_sweep_zw06(obs[seed], float(nyq[seed]), reference=seed_ref, wrap_azimuth=wrap_azimuth)
    corrected[seed] = seed_result.velocity
    confidence[seed] = seed_result.confidence
    if seed_result.reference is not None:
        reference[seed] = seed_result.reference
    elif seed_ref is not None:
        reference[seed] = seed_ref
    else:
        reference[seed] = seed_result.velocity

    order = _sweep_order_from_seed(seed, obs.shape[0])
    reverse_order = list(reversed(order))
    iterations_used = 1
    per_sweep_meta: list[dict[str, object]] = [dict() for _ in range(obs.shape[0])]
    per_sweep_meta[seed] = dict(seed_result.metadata)

    for iteration in range(max_iterations):
        current_order = order if iteration % 2 == 0 else reverse_order
        changed, pass_meta = _refine_volume_pass(
            obs,
            nyq,
            corrected,
            confidence,
            ref,
            current_order,
            wrap_azimuth=wrap_azimuth,
        )
        iterations_used += 1
        for idx, meta in enumerate(pass_meta):
            if meta:
                per_sweep_meta[idx] = meta
        if changed == 0:
            break

    # Final cleanup pass: if any sweep remains unresolved, use the adjacent-sweep
    # consensus as a plain reference target so every gate gets a value.
    for sweep in range(obs.shape[0]):
        sweep_ref = _sweep_reference(corrected, sweep, reference_volume=ref)
        if sweep_ref is None:
            continue
        if np.any(np.isfinite(obs[sweep])):
            unfolded = np.where(
                np.isfinite(obs[sweep]),
                unfold_to_reference(obs[sweep], sweep_ref, float(nyq[sweep])),
                sweep_ref,
            )
            filled_sweep = np.where(np.isfinite(corrected[sweep]), corrected[sweep], unfolded)
        else:
            filled_sweep = sweep_ref
        corrected[sweep] = filled_sweep
        confidence[sweep] = np.maximum(
            confidence[sweep],
            gaussian_confidence(np.abs(filled_sweep - sweep_ref), 0.45 * float(nyq[sweep])),
        )

    folds = _fold_counts_volume(corrected, obs, nyq)
    return attach_result_state_from_fields(
        DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=reference,
        metadata={
            'paper_family': 'UNRAVEL-style-3D',
            'method': 'volume_3d_continuity',
            'seed_sweep': int(seed),
            'iterations_used': int(iterations_used),
            'wrap_azimuth': bool(wrap_azimuth),
            'sweep_order': order,
            'per_sweep': per_sweep_meta,
        },
        ),
        obs,
        source="volume_3d_continuity",
        parent="UNRAVEL-style-3D",
        fill_policy="volume_reference_then_cleanup",
    )
