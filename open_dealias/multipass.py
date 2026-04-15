from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ._core import (
    as_float_array,
    combine_references,
    fold_counts,
    gaussian_confidence,
    neighbor_stack,
    texture_3x3,
    unfold_to_reference,
)
from .types import DealiasResult


DEFAULT_PASSES: tuple[dict[str, float | int | bool], ...] = (
    {'min_neighbors': 3, 'max_mismatch_fraction': 0.35, 'reference_only': False},
    {'min_neighbors': 2, 'max_mismatch_fraction': 0.60, 'reference_only': True},
    {'min_neighbors': 1, 'max_mismatch_fraction': 1.10, 'reference_only': True},
)


def _safe_nanmedian(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        with np.errstate(all='ignore'):
            return np.nanmedian(stack, axis=axis)


def dealias_sweep_zw06(
    observed: np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    passes: Sequence[dict[str, float | int | bool]] = DEFAULT_PASSES,
    weak_threshold_fraction: float = 0.35,
    wrap_azimuth: bool = True,
    max_iterations_per_pass: int = 12,
    include_diagonals: bool = True,
    recenter_without_reference: bool = True,
) -> DealiasResult:
    """Zhang–Wang-style automated 2D multipass dealiasing.

    The implementation is intentionally compact and public-facing. It follows the
    paper family idea directly: find a weak or low-texture anchor region, then use
    increasingly relaxed horizontal continuity passes across the sweep.
    """
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError('observed must be 2D [azimuth, range]')
    if nyquist <= 0:
        raise ValueError('nyquist must be positive')

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError('reference must match observed shape')

    valid = np.isfinite(obs)
    corrected = np.full(obs.shape, np.nan, dtype=float)
    confidence = np.zeros(obs.shape, dtype=float)
    texture = texture_3x3(obs, wrap_azimuth=wrap_azimuth)

    # Seed from a trusted reference when available.
    if ref is not None:
        candidate = unfold_to_reference(obs, ref, nyquist)
        mismatch = np.abs(candidate - ref)
        seed_mask = valid & np.isfinite(candidate) & (mismatch <= 0.80 * nyquist)
        corrected[seed_mask] = candidate[seed_mask]
        confidence[seed_mask] = gaussian_confidence(mismatch[seed_mask], 0.45 * nyquist)

    # Weak-wind / low-texture fallback seeds, inspired by the weakest-wind anchor idea.
    unresolved = valid & ~np.isfinite(corrected)
    if np.any(unresolved):
        weak_threshold = weak_threshold_fraction * nyquist
        finite_texture = texture[np.isfinite(texture)]
        if finite_texture.size:
            texture_cut = float(np.quantile(finite_texture, 0.25))
        else:
            texture_cut = weak_threshold
        weak_seed = unresolved & ((np.abs(obs) <= weak_threshold) | (texture <= texture_cut))
        corrected[weak_seed] = obs[weak_seed]
        confidence[weak_seed] = 0.72

    # Make sure at least one gate is seeded.
    if not np.any(np.isfinite(corrected)) and np.any(valid):
        flat_idx = np.nanargmin(np.where(valid, np.abs(obs), np.nan))
        i, j = np.unravel_index(flat_idx, obs.shape)
        corrected[i, j] = obs[i, j]
        confidence[i, j] = 0.55

    total_assigned = int(np.isfinite(corrected).sum())
    iterations_used = 0

    for spec in passes:
        min_neighbors = int(spec['min_neighbors'])
        max_mismatch = float(spec['max_mismatch_fraction']) * nyquist
        reference_only = bool(spec['reference_only'])

        for _ in range(max_iterations_per_pass):
            iterations_used += 1
            unresolved = valid & ~np.isfinite(corrected)
            if not np.any(unresolved):
                break

            neigh = neighbor_stack(corrected, include_diagonals=include_diagonals, wrap_azimuth=wrap_azimuth)
            neigh_count = np.sum(np.isfinite(neigh), axis=0)
            neigh_ref = _safe_nanmedian(neigh, axis=0)
            combined_ref = neigh_ref
            if ref is not None:
                combined_ref = combined_ref.copy()
                sparse = (~np.isfinite(combined_ref)) | (neigh_count < max(2, min_neighbors))
                combined_ref[sparse] = np.where(np.isfinite(ref[sparse]), ref[sparse], combined_ref[sparse])
                blend = np.isfinite(neigh_ref) & np.isfinite(ref) & (neigh_count < (min_neighbors + 1))
                combined_ref[blend] = 0.5 * (neigh_ref[blend] + ref[blend])
            if combined_ref is None:
                break

            candidate = unfold_to_reference(obs, combined_ref, nyquist)
            mismatch = np.abs(candidate - combined_ref)
            enough_neighbors = neigh_count >= min_neighbors
            if reference_only and ref is not None:
                enough_neighbors = enough_neighbors | np.isfinite(ref)

            assign = unresolved & enough_neighbors & np.isfinite(candidate) & (mismatch <= max_mismatch)
            if not np.any(assign):
                break

            corrected[assign] = candidate[assign]
            confidence[assign] = np.maximum(confidence[assign], gaussian_confidence(mismatch[assign], 0.40 * nyquist))
            total_assigned += int(assign.sum())

    unresolved = valid & ~np.isfinite(corrected)
    if np.any(unresolved) and ref is not None:
        candidate = unfold_to_reference(obs, ref, nyquist)
        mismatch = np.abs(candidate - ref)
        assign = unresolved & np.isfinite(candidate) & (mismatch <= 1.10 * nyquist)
        corrected[assign] = candidate[assign]
        confidence[assign] = np.maximum(confidence[assign], 0.45)

    unresolved = valid & ~np.isfinite(corrected)
    if np.any(unresolved):
        neigh = neighbor_stack(corrected, include_diagonals=include_diagonals, wrap_azimuth=wrap_azimuth)
        neigh_ref = _safe_nanmedian(neigh, axis=0)
        candidate = unfold_to_reference(obs, neigh_ref, nyquist)
        assign = unresolved & np.isfinite(candidate)
        corrected[assign] = candidate[assign]
        confidence[assign] = np.maximum(confidence[assign], 0.25)

    if ref is not None and np.any(np.isfinite(corrected)):
        # A final radial cleanup respects the paper family's use of reference
        # radials/gates while letting along-ray continuity recover strong shear.
        from .continuity import dealias_sweep_es90
        cleanup_ref = combine_references(corrected, ref)
        radial = dealias_sweep_es90(obs, nyquist, reference=cleanup_ref)
        corrected = radial.velocity
        confidence = np.maximum(confidence, radial.confidence)

    if recenter_without_reference and ref is None and np.any(np.isfinite(corrected)):
        folds = fold_counts(corrected, obs, nyquist)
        finite_folds = folds[np.isfinite(corrected)]
        if finite_folds.size:
            shift = int(np.rint(np.median(finite_folds)))
            corrected = corrected - 2.0 * nyquist * shift
            confidence = confidence.copy()
    folds = fold_counts(corrected, obs, nyquist)

    return DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=ref,
        metadata={
            'paper_family': 'JingWiener1993+ZhangWang2006',
            'method': '2d_multipass',
            'seeded_gates': int(np.sum(confidence >= 0.70)),
            'assigned_gates': int(np.sum(np.isfinite(corrected))),
            'iterations_used': int(iterations_used),
            'wrap_azimuth': bool(wrap_azimuth),
            'passes': [dict(p) for p in passes],
        },
    )
