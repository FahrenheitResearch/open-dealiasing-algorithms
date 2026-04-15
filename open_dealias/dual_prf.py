from __future__ import annotations

from typing import Iterable

import numpy as np

from ._core import as_float_array, combine_references, fold_counts, gaussian_confidence, unfold_to_reference, wrap_to_nyquist
from ._rust_bridge import get_rust_backend
from .types import DealiasResult

__all__ = ['dealias_dual_prf']

_NATIVE_BACKEND = get_rust_backend()


def _best_branch_against_partner(
    base: np.ndarray,
    partner: np.ndarray,
    base_nyquist: float,
    partner_nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_abs_fold: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the most plausible unfolded branch from one PRF against the other."""
    if max_abs_fold < 0:
        raise ValueError('max_abs_fold must be non-negative')

    base = np.asarray(base, dtype=float)
    partner = np.asarray(partner, dtype=float)
    if base.shape != partner.shape:
        raise ValueError('base and partner must have the same shape')

    flat = np.isfinite(base) & np.isfinite(partner)
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != base.shape:
            raise ValueError('reference must match the observed shape')
    else:
        ref = None

    best_value = np.full(base.shape, np.nan, dtype=float)
    best_fold = np.zeros(base.shape, dtype=np.int16)
    best_gap = np.full(base.shape, np.nan, dtype=float)

    if not np.any(flat):
        return best_value, best_fold, best_gap

    base_flat = base[flat]
    partner_flat = partner[flat]
    if ref is not None:
        ref_flat = ref[flat]
    else:
        ref_flat = None

    best_score = np.full(base_flat.shape, np.inf, dtype=float)
    best_candidate = base_flat.copy()
    best_k = np.zeros(base_flat.shape, dtype=np.int16)
    best_partner_gap = np.full(base_flat.shape, np.inf, dtype=float)

    scale = max(float(base_nyquist), float(partner_nyquist), 1.0)
    for k in range(-max_abs_fold, max_abs_fold + 1):
        candidate = base_flat + 2.0 * float(base_nyquist) * k
        partner_gap = np.abs(wrap_to_nyquist(candidate - partner_flat, partner_nyquist))
        score = partner_gap + 0.02 * abs(k)
        if ref_flat is not None:
            score = score + 0.10 * np.abs(candidate - ref_flat) / scale

        better = score < best_score - 1e-12
        tied = np.isclose(score, best_score, atol=1e-12, rtol=0.0)
        if ref_flat is not None:
            tie_break = np.abs(candidate - ref_flat) < np.abs(best_candidate - ref_flat)
        else:
            tie_break = np.abs(candidate) < np.abs(best_candidate)
        update = better | (tied & tie_break)
        if np.any(update):
            best_score[update] = score[update]
            best_candidate[update] = candidate[update]
            best_k[update] = k
            best_partner_gap[update] = partner_gap[update]

    best_value[flat] = best_candidate
    best_fold[flat] = best_k
    best_gap[flat] = best_partner_gap
    return best_value, best_fold, best_gap


def _python_dealias_dual_prf(
    low_observed: Iterable[float] | np.ndarray,
    high_observed: Iterable[float] | np.ndarray,
    low_nyquist: float,
    high_nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_abs_fold: int = 32,
) -> DealiasResult:
    """Unfold paired low/high PRF observations by selecting the branch that best agrees.

    The solver is intentionally simple and explicit:
    1. search low-PRF fold candidates against the high-PRF observation,
    2. search high-PRF fold candidates against the low-PRF observation,
    3. combine the two branch estimates with an optional external reference.
    """
    low = as_float_array(low_observed)
    high = as_float_array(high_observed)
    if low.shape != high.shape:
        raise ValueError('low_observed and high_observed must have the same shape')
    if low.ndim == 0:
        raise ValueError('observations must have at least one dimension')
    if low_nyquist <= 0 or high_nyquist <= 0:
        raise ValueError('nyquist values must be positive')

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != low.shape:
        raise ValueError('reference must match the observed shape')

    low_best, low_fold, low_gap = _best_branch_against_partner(
        low,
        high,
        float(low_nyquist),
        float(high_nyquist),
        reference=ref,
        max_abs_fold=max_abs_fold,
    )
    high_best, high_fold, high_gap = _best_branch_against_partner(
        high,
        low,
        float(high_nyquist),
        float(low_nyquist),
        reference=ref,
        max_abs_fold=max_abs_fold,
    )

    combined = combine_references(low_best, high_best, ref)
    if combined is None:
        combined = np.full(low.shape, np.nan, dtype=float)

    # Fill missing gates from whichever PRF branch was available.
    combined = np.where(np.isfinite(combined), combined, low_best)
    combined = np.where(np.isfinite(combined), combined, high_best)

    valid_low = np.isfinite(low)
    valid_high = np.isfinite(high)
    both_valid = valid_low & valid_high
    only_low = valid_low & ~valid_high
    only_high = valid_high & ~valid_low

    if np.any(only_low):
        if ref is not None:
            fallback = unfold_to_reference(low[only_low], ref[only_low], float(low_nyquist))
        else:
            fallback = low[only_low]
        combined[only_low] = fallback

    if np.any(only_high):
        if ref is not None:
            fallback = unfold_to_reference(high[only_high], ref[only_high], float(high_nyquist))
        else:
            fallback = high[only_high]
        combined[only_high] = fallback

    if np.any(both_valid):
        pair_gap = np.abs(low_best[both_valid] - high_best[both_valid])
        branch_confidence = gaussian_confidence(pair_gap, 0.40 * min(float(low_nyquist), float(high_nyquist)))
        if ref is not None:
            ref_gap = np.abs(combined[both_valid] - ref[both_valid])
            branch_confidence = np.maximum(branch_confidence, gaussian_confidence(ref_gap, 0.55 * max(float(low_nyquist), float(high_nyquist))))
    else:
        pair_gap = np.array([], dtype=float)
        branch_confidence = np.array([], dtype=float)

    confidence = np.zeros(low.shape, dtype=float)
    if np.any(both_valid):
        confidence[both_valid] = branch_confidence
    if np.any(only_low):
        confidence[only_low] = 0.55 if ref is None else 0.80
    if np.any(only_high):
        confidence[only_high] = 0.55 if ref is None else 0.80

    if np.any(both_valid):
        pair_gap_full = np.full(low.shape, np.nan, dtype=float)
        pair_gap_full[both_valid] = pair_gap
    else:
        pair_gap_full = np.full(low.shape, np.nan, dtype=float)

    folds = fold_counts(combined, low, float(low_nyquist))
    metadata = {
        'paper_family': 'DualPRF',
        'method': 'dual_prf_pair_search',
        'low_nyquist': float(low_nyquist),
        'high_nyquist': float(high_nyquist),
        'max_abs_fold': int(max_abs_fold),
        'low_valid_gates': int(np.sum(valid_low)),
        'high_valid_gates': int(np.sum(valid_high)),
        'paired_gates': int(np.sum(both_valid)),
        'low_branch_mean_fold': float(np.nanmean(low_fold[both_valid])) if np.any(both_valid) else 0.0,
        'high_branch_mean_fold': float(np.nanmean(high_fold[both_valid])) if np.any(both_valid) else 0.0,
        'mean_pair_gap': float(np.nanmean(pair_gap_full)) if np.any(np.isfinite(pair_gap_full)) else float('nan'),
        'max_pair_gap': float(np.nanmax(pair_gap_full)) if np.any(np.isfinite(pair_gap_full)) else float('nan'),
    }

    return DealiasResult(
        velocity=combined,
        folds=folds,
        confidence=confidence,
        reference=ref if ref is not None else combine_references(low_best, high_best),
        metadata=metadata,
    )


def dealias_dual_prf(
    low_observed: Iterable[float] | np.ndarray,
    high_observed: Iterable[float] | np.ndarray,
    low_nyquist: float,
    high_nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_abs_fold: int = 32,
) -> DealiasResult:
    low = as_float_array(low_observed)
    high = as_float_array(high_observed)
    ref = None if reference is None else np.asarray(reference, dtype=float)

    if _NATIVE_BACKEND is not None:
        velocity, folds, confidence, ref_out, metadata = _NATIVE_BACKEND.dealias_dual_prf(
            low,
            high,
            float(low_nyquist),
            float(high_nyquist),
            ref,
            int(max_abs_fold),
        )
        return DealiasResult(
            velocity=np.asarray(velocity, dtype=float),
            folds=np.asarray(folds, dtype=np.int16),
            confidence=np.asarray(confidence, dtype=float),
            reference=np.asarray(ref_out, dtype=float),
            metadata=dict(metadata),
        )

    return _python_dealias_dual_prf(
        low,
        high,
        low_nyquist,
        high_nyquist,
        reference=ref,
        max_abs_fold=max_abs_fold,
    )
