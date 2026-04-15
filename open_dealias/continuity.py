from __future__ import annotations

from typing import Iterable

import numpy as np

from ._core import as_float_array, combine_references, fold_counts, gaussian_confidence, unfold_to_reference
from .types import DealiasResult


def _pick_seed(observed: np.ndarray, reference: np.ndarray | None = None) -> int | None:
    finite = np.flatnonzero(np.isfinite(observed))
    if finite.size == 0:
        return None
    if reference is not None:
        overlap = finite[np.isfinite(reference[finite])]
        if overlap.size:
            center = np.median(overlap)
            return int(overlap[np.argmin(np.abs(overlap - center))])
    return int(finite[finite.size // 2])


def _walk_radial(
    observed: np.ndarray,
    nyquist: float,
    corrected: np.ndarray,
    confidence: np.ndarray,
    *,
    reference: np.ndarray | None,
    seed_index: int,
    direction: int,
    max_gap: int,
    max_abs_step: float | None,
) -> None:
    idx = seed_index + direction
    last_valid = seed_index
    last_valid_two = None

    while 0 <= idx < observed.size:
        if not np.isfinite(observed[idx]):
            idx += direction
            continue

        local_refs: list[float] = []
        if last_valid is not None and abs(idx - last_valid) <= max_gap and np.isfinite(corrected[last_valid]):
            local_refs.append(float(corrected[last_valid]))
            if last_valid_two is not None and np.isfinite(corrected[last_valid_two]):
                slope = corrected[last_valid] - corrected[last_valid_two]
                local_refs.append(float(corrected[last_valid] + slope))
        if reference is not None and np.isfinite(reference[idx]):
            local_refs.append(float(reference[idx]))

        if not local_refs:
            corrected[idx] = observed[idx]
            confidence[idx] = 0.15
            last_valid_two = last_valid
            last_valid = idx
            idx += direction
            continue

        ref_value = float(np.median(np.asarray(local_refs, dtype=float)))
        candidate = float(unfold_to_reference(observed[idx], ref_value, nyquist))

        if max_abs_step is not None and last_valid is not None and np.isfinite(corrected[last_valid]):
            step = abs(candidate - corrected[last_valid])
            if step > max_abs_step and reference is not None and np.isfinite(reference[idx]):
                candidate = float(unfold_to_reference(observed[idx], reference[idx], nyquist))
                ref_value = float(reference[idx])

        mismatch = abs(candidate - ref_value)
        corrected[idx] = candidate
        confidence[idx] = float(gaussian_confidence(np.asarray(mismatch), 0.45 * nyquist))
        last_valid_two = last_valid
        last_valid = idx
        idx += direction


def dealias_radial_es90(
    observed: Iterable[float] | np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    seed_index: int | None = None,
    max_gap: int = 3,
    max_abs_step: float | None = None,
) -> DealiasResult:
    """Eilts–Smith-style radial continuity dealiasing.

    This is a compact, public, implementation-oriented variant of the classic
    local-environment continuity family described by Eilts and Smith (1990).
    """
    obs = as_float_array(observed)
    if obs.ndim != 1:
        raise ValueError('observed must be 1D')
    if nyquist <= 0:
        raise ValueError('nyquist must be positive')

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError('reference must match observed shape')

    corrected = np.full(obs.shape, np.nan, dtype=float)
    confidence = np.zeros(obs.shape, dtype=float)

    if seed_index is None:
        seed_index = _pick_seed(obs, ref)
    if seed_index is None:
        return DealiasResult(corrected, np.zeros(obs.shape, dtype=np.int16), confidence, reference=ref, metadata={'paper': 'EiltsSmith1990', 'seed_index': None})
    if not np.isfinite(obs[seed_index]):
        raise ValueError('seed_index points to a non-finite gate')

    if ref is not None and np.isfinite(ref[seed_index]):
        corrected[seed_index] = float(unfold_to_reference(obs[seed_index], ref[seed_index], nyquist))
        confidence[seed_index] = 0.98
    else:
        corrected[seed_index] = float(obs[seed_index])
        confidence[seed_index] = 0.80

    _walk_radial(obs, nyquist, corrected, confidence, reference=ref, seed_index=seed_index, direction=1, max_gap=max_gap, max_abs_step=max_abs_step)
    _walk_radial(obs, nyquist, corrected, confidence, reference=ref, seed_index=seed_index, direction=-1, max_gap=max_gap, max_abs_step=max_abs_step)

    folds = fold_counts(corrected, obs, nyquist)
    return DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=ref,
        metadata={
            'paper': 'EiltsSmith1990',
            'method': 'radial_continuity',
            'seed_index': int(seed_index),
            'max_gap': int(max_gap),
            'max_abs_step': None if max_abs_step is None else float(max_abs_step),
        },
    )


def dealias_sweep_es90(
    observed: np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_gap: int = 3,
    max_abs_step: float | None = None,
) -> DealiasResult:
    """Process a sweep one radial at a time using prior-radial context."""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2:
        raise ValueError('observed must be 2D [azimuth, range]')
    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError('reference must match observed shape')

    corrected = np.full(obs.shape, np.nan, dtype=float)
    confidence = np.zeros(obs.shape, dtype=float)

    for ray in range(obs.shape[0]):
        ray_ref = None
        if ray > 0:
            ray_ref = corrected[ray - 1]
        combined = combine_references(ray_ref, None if ref is None else ref[ray])
        radial = dealias_radial_es90(
            obs[ray],
            nyquist,
            reference=combined,
            max_gap=max_gap,
            max_abs_step=max_abs_step,
        )
        corrected[ray] = radial.velocity
        confidence[ray] = radial.confidence

    folds = fold_counts(corrected, obs, nyquist)
    return DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=ref,
        metadata={
            'paper': 'EiltsSmith1990',
            'method': 'sweep_radial_continuity',
            'max_gap': int(max_gap),
            'max_abs_step': None if max_abs_step is None else float(max_abs_step),
        },
    )
