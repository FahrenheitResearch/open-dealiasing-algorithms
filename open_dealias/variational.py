from __future__ import annotations

import warnings

import numpy as np

from ._core import as_float_array, fold_counts, gaussian_confidence, neighbor_stack, unfold_to_reference
from .result_state import attach_result_state_from_fields
from ._rust_bridge import get_rust_backend, resolve_rust_backend
from .region_graph import dealias_sweep_region_graph
from .types import DealiasResult

__all__ = ["dealias_sweep_variational"]

_NATIVE_BACKEND = get_rust_backend()


def _safe_nanmean(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            return np.nanmean(arr, axis=axis)


def dealias_sweep_variational(
    observed: np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_abs_fold: int = 8,
    neighbor_weight: float = 1.0,
    reference_weight: float = 0.50,
    smoothness_weight: float = 0.20,
    max_iterations: int = 8,
    wrap_azimuth: bool = True,
) -> DealiasResult:
    """Global-ish coordinate-descent solver over integer fold offsets.

    This is an educational variational / MRF-style approximation:
    each gate chooses the fold offset that best matches its neighbors and an
    optional reference field under a smoothness objective, then the field is
    iteratively updated until stable.
    """
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError("observed must be 2D [azimuth, range]")
    if nyquist <= 0:
        raise ValueError("nyquist must be positive")

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError("reference must match observed shape")

    valid = np.isfinite(obs)
    if not np.any(valid):
        return attach_result_state_from_fields(
            DealiasResult(
            velocity=np.full(obs.shape, np.nan, dtype=float),
            folds=np.zeros(obs.shape, dtype=np.int16),
            confidence=np.zeros(obs.shape, dtype=float),
            reference=ref,
            metadata={"paper_family": "VariationalLite", "method": "coordinate_descent", "iterations_used": 0},
            ),
            obs,
            source="coordinate_descent",
            parent="VariationalLite",
        )

    bootstrap = dealias_sweep_region_graph(obs, nyquist, reference=ref, wrap_azimuth=wrap_azimuth)
    if ref is not None:
        corrected = np.where(np.isfinite(bootstrap.velocity), bootstrap.velocity, unfold_to_reference(obs, ref, nyquist))
    else:
        corrected = bootstrap.velocity.copy()
    corrected = np.where(valid, corrected, np.nan)

    backend = resolve_rust_backend(_NATIVE_BACKEND)
    if backend is not None:
        velocity, folds, confidence, metadata = backend.dealias_sweep_variational_refine(
            obs,
            corrected,
            float(nyquist),
            ref,
            int(max_abs_fold),
            float(neighbor_weight),
            float(reference_weight),
            float(smoothness_weight),
            int(max_iterations),
            bool(wrap_azimuth),
        )
        meta = dict(metadata)
        meta["bootstrap_method"] = bootstrap.metadata.get("method")
        return attach_result_state_from_fields(
            DealiasResult(
            velocity=np.asarray(velocity, dtype=float),
            folds=np.asarray(folds, dtype=np.int16),
            confidence=np.asarray(confidence, dtype=float),
            reference=ref,
            metadata=meta,
            ),
            obs,
            source=str(meta.get("method", "coordinate_descent")),
            parent=str(meta.get("paper_family", "VariationalLite")),
            notes=(f"bootstrap_method={bootstrap.metadata.get('method')}",),
        )

    folds = fold_counts(corrected, obs, nyquist).astype(int)

    offsets = list(range(-max_abs_fold, max_abs_fold + 1))
    iterations_used = 0
    changed_total = 0

    for _ in range(max_iterations):
        iterations_used += 1
        changed = 0
        neigh = neighbor_stack(corrected, include_diagonals=True, wrap_azimuth=wrap_azimuth)
        neigh_mean = _safe_nanmean(neigh, axis=0)

        for row, col in np.argwhere(valid):
            center_fold = int(folds[row, col])
            candidate_folds = range(max(-max_abs_fold, center_fold - 2), min(max_abs_fold, center_fold + 2) + 1)
            best_fold = center_fold
            best_value = corrected[row, col]
            best_score = float("inf")

            local_neighbors = neigh[:, row, col]
            local_neighbors = local_neighbors[np.isfinite(local_neighbors)]
            local_ref = float(ref[row, col]) if ref is not None and np.isfinite(ref[row, col]) else None
            local_mean = float(neigh_mean[row, col]) if np.isfinite(neigh_mean[row, col]) else None

            for fold in candidate_folds:
                candidate = float(obs[row, col] + 2.0 * nyquist * fold)
                score = smoothness_weight * abs(fold)
                if local_neighbors.size:
                    score += neighbor_weight * float(np.mean((candidate - local_neighbors) ** 2))
                elif local_mean is not None:
                    score += neighbor_weight * (candidate - local_mean) ** 2
                if local_ref is not None:
                    score += reference_weight * (candidate - local_ref) ** 2
                if score < best_score:
                    best_score = score
                    best_fold = int(fold)
                    best_value = candidate

            if best_fold != center_fold:
                folds[row, col] = best_fold
                corrected[row, col] = best_value
                changed += 1

        changed_total += changed
        if changed == 0:
            break

    target = neigh_mean if ref is None else np.where(np.isfinite(ref), ref, neigh_mean)
    mismatch = np.abs(corrected - target)
    confidence = np.where(valid, gaussian_confidence(mismatch, 0.45 * nyquist), 0.0)

    return attach_result_state_from_fields(
        DealiasResult(
        velocity=corrected,
        folds=folds.astype(np.int16),
        confidence=confidence,
        reference=ref,
        metadata={
            "paper_family": "VariationalLite",
            "method": "coordinate_descent",
            "iterations_used": int(iterations_used),
            "changed_gates": int(changed_total),
            "max_abs_fold": int(max_abs_fold),
            "wrap_azimuth": bool(wrap_azimuth),
            "bootstrap_method": bootstrap.metadata.get("method"),
        },
        ),
        obs,
        source="coordinate_descent",
        parent="VariationalLite",
        notes=(f"bootstrap_method={bootstrap.metadata.get('method')}",),
    )
