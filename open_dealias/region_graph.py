from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import inspect
from typing import Iterable
import warnings

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
from .result_state import attach_result_state_from_fields
from ._rust_bridge import get_rust_backend, resolve_rust_backend
from .types import DealiasResult


@dataclass(slots=True)
class _Region:
    region_id: int
    row0: int
    row1: int
    col0: int
    col1: int
    mean_obs: float
    texture: float
    area: int
    neighbors: set[int] = field(default_factory=set)
    boundary_weight: dict[int, int] = field(default_factory=dict)


_NATIVE_BACKEND = get_rust_backend()
_DEFAULT_MIN_REGION_AREA = 4
_DEFAULT_MIN_VALID_FRACTION = 0.15


def _safe_nanmedian(data: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            value = np.nanmedian(data)
    return float(value) if np.isfinite(value) else float("nan")


def _wrap_delta(a: np.ndarray | float, b: np.ndarray | float, nyquist: float) -> np.ndarray:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    delta = ((arr_a - arr_b + nyquist) % (2.0 * nyquist)) - nyquist
    return np.where(np.isfinite(arr_a) & np.isfinite(arr_b), delta, np.nan)


def _choose_block_shape(shape: tuple[int, int], block_shape: tuple[int, int] | None) -> tuple[int, int]:
    rows, cols = shape
    if block_shape is not None:
        br, bc = block_shape
        br = int(max(1, min(rows, br)))
        bc = int(max(1, min(cols, bc)))
        return br, bc
    br = rows // 18 if rows >= 18 else rows
    bc = cols // 18 if cols >= 18 else cols
    br = int(max(4, min(16, br if br > 0 else 4)))
    bc = int(max(4, min(16, bc if bc > 0 else 4)))
    br = min(br, rows)
    bc = min(bc, cols)
    return br, bc


def _region_mean(reference: np.ndarray | None, row0: int, row1: int, col0: int, col1: int) -> float:
    if reference is None:
        return float("nan")
    return _safe_nanmedian(reference[row0:row1, col0:col1])


def _build_regions(
    observed: np.ndarray,
    *,
    reference: np.ndarray | None,
    block_shape: tuple[int, int] | None,
    wrap_azimuth: bool,
    min_region_area: int,
    min_valid_fraction: float,
) -> tuple[list[_Region], np.ndarray, int]:
    rows, cols = observed.shape
    block_rows, block_cols = _choose_block_shape(observed.shape, block_shape)
    texture_map = texture_3x3(observed, wrap_azimuth=wrap_azimuth)

    n_row_blocks = max(1, int(np.ceil(rows / block_rows)))
    n_col_blocks = max(1, int(np.ceil(cols / block_cols)))
    block_ids = np.full((n_row_blocks, n_col_blocks), -1, dtype=int)
    regions: list[_Region] = []
    skipped_sparse_blocks = 0

    for bi in range(n_row_blocks):
        r0 = bi * block_rows
        r1 = min(rows, r0 + block_rows)
        for bj in range(n_col_blocks):
            c0 = bj * block_cols
            c1 = min(cols, c0 + block_cols)
            block = observed[r0:r1, c0:c1]
            finite = np.isfinite(block)
            finite_count = int(np.count_nonzero(finite))
            if finite_count == 0:
                continue
            total_cells = int(block.size)
            valid_fraction = finite_count / total_cells
            if finite_count < min_region_area or valid_fraction < min_valid_fraction:
                skipped_sparse_blocks += 1
                continue
            region_id = len(regions)
            block_ids[bi, bj] = region_id
            mean_obs = _safe_nanmedian(block)
            texture = _safe_nanmedian(texture_map[r0:r1, c0:c1])
            if not np.isfinite(texture):
                texture = 0.0
            regions.append(
                _Region(
                    region_id=region_id,
                    row0=r0,
                    row1=r1,
                    col0=c0,
                    col1=c1,
                    mean_obs=mean_obs,
                    texture=texture,
                    area=finite_count,
                )
            )

    for bi in range(n_row_blocks):
        for bj in range(n_col_blocks):
            region_id = block_ids[bi, bj]
            if region_id < 0:
                continue
            region = regions[region_id]
            if bj + 1 < n_col_blocks:
                right_id = block_ids[bi, bj + 1]
                if right_id >= 0 and right_id != region_id:
                    region.neighbors.add(right_id)
                    regions[right_id].neighbors.add(region_id)
                    edge = min(region.row1 - region.row0, regions[right_id].row1 - regions[right_id].row0)
                    region.boundary_weight[right_id] = max(region.boundary_weight.get(right_id, 0), edge)
                    regions[right_id].boundary_weight[region_id] = max(regions[right_id].boundary_weight.get(region_id, 0), edge)
            if bi + 1 < n_row_blocks:
                down_id = block_ids[bi + 1, bj]
                if down_id >= 0 and down_id != region_id:
                    region.neighbors.add(down_id)
                    regions[down_id].neighbors.add(region_id)
                    edge = min(region.col1 - region.col0, regions[down_id].col1 - regions[down_id].col0)
                    region.boundary_weight[down_id] = max(region.boundary_weight.get(down_id, 0), edge)
                    regions[down_id].boundary_weight[region_id] = max(regions[down_id].boundary_weight.get(region_id, 0), edge)
            if wrap_azimuth and n_row_blocks > 1 and bi == 0:
                wrap_id = block_ids[n_row_blocks - 1, bj]
                if wrap_id >= 0 and wrap_id != region_id:
                    region.neighbors.add(wrap_id)
                    regions[wrap_id].neighbors.add(region_id)
                    edge = min(region.col1 - region.col0, regions[wrap_id].col1 - regions[wrap_id].col0)
                    region.boundary_weight[wrap_id] = max(region.boundary_weight.get(wrap_id, 0), edge)
                    regions[wrap_id].boundary_weight[region_id] = max(regions[wrap_id].boundary_weight.get(region_id, 0), edge)

    for region in regions:
        if reference is not None:
            region_ref = _region_mean(reference, region.row0, region.row1, region.col0, region.col1)
            if np.isfinite(region_ref):
                region.texture = max(region.texture, 0.5 * abs(region.mean_obs - region_ref))

    return regions, block_ids, skipped_sparse_blocks


def _pick_seed_region(regions: list[_Region], reference: np.ndarray | None) -> int:
    if not regions:
        return -1
    if reference is None:
        scores = []
        for region in regions:
            score = abs(region.mean_obs) + 0.2 * region.texture - 0.01 * region.area
            scores.append(score)
        return int(np.argmin(scores))

    scores = []
    for region in regions:
        region_ref = _region_mean(reference, region.row0, region.row1, region.col0, region.col1)
        if np.isfinite(region_ref):
            score = abs(region.mean_obs - region_ref) + 0.1 * region.texture - 0.01 * region.area
        else:
            score = abs(region.mean_obs) + 0.1 * region.texture - 0.01 * region.area
        scores.append(score)
    return int(np.argmin(scores))


def _best_fold_for_region(
    region: _Region,
    fold_map: dict[int, int],
    regions: list[_Region],
    *,
    nyquist: float,
    reference: np.ndarray | None,
    reference_weight: float,
    max_abs_fold: int,
) -> tuple[int, float, float]:
    neighbor_means: list[float] = []
    neighbor_weights: list[float] = []
    for nb in region.neighbors:
        if nb not in fold_map:
            continue
        neighbor = regions[nb]
        corrected_mean = neighbor.mean_obs + 2.0 * nyquist * fold_map[nb]
        neighbor_means.append(float(corrected_mean))
        neighbor_weights.append(float(max(1, region.boundary_weight.get(nb, 1))))

    region_ref = None
    if reference is not None:
        region_ref = _region_mean(reference, region.row0, region.row1, region.col0, region.col1)

    if neighbor_means:
        target = float(np.average(neighbor_means, weights=neighbor_weights))
        center = int(np.rint((target - region.mean_obs) / (2.0 * nyquist)))
    elif np.isfinite(region_ref):
        center = int(np.rint((float(region_ref) - region.mean_obs) / (2.0 * nyquist)))
    else:
        center = 0
    center = int(np.clip(center, -max_abs_fold, max_abs_fold))

    candidate_folds = range(max(-max_abs_fold, center - 3), min(max_abs_fold, center + 3) + 1)
    best_fold = center
    best_score = float("inf")
    best_mean = region.mean_obs + 2.0 * nyquist * center

    for fold in candidate_folds:
        candidate_mean = region.mean_obs + 2.0 * nyquist * fold
        score = 0.35 * abs(fold)
        for nm, weight in zip(neighbor_means, neighbor_weights):
            score += weight * abs(candidate_mean - nm)
        if np.isfinite(region_ref):
            score += reference_weight * abs(candidate_mean - float(region_ref))
        if score < best_score:
            best_score = score
            best_fold = int(fold)
            best_mean = float(candidate_mean)
    return best_fold, best_mean, float(best_score)


def _propagate_region_folds(
    regions: list[_Region],
    *,
    nyquist: float,
    reference: np.ndarray | None,
    reference_weight: float,
    max_abs_fold: int,
    max_iterations: int,
) -> tuple[dict[int, int], dict[int, float], dict[int, float]]:
    fold_map: dict[int, int] = {}
    mean_map: dict[int, float] = {}
    score_map: dict[int, float] = {}
    if not regions:
        return fold_map, mean_map, score_map

    seed = _pick_seed_region(regions, reference)
    seed_region = regions[seed]
    seed_fold, seed_mean, seed_score = _best_fold_for_region(
        seed_region,
        fold_map,
        regions,
        nyquist=nyquist,
        reference=reference,
        reference_weight=reference_weight,
        max_abs_fold=max_abs_fold,
    )
    fold_map[seed] = seed_fold
    mean_map[seed] = seed_mean
    score_map[seed] = seed_score

    queue: deque[int] = deque([seed])
    while queue:
        rid = queue.popleft()
        for nb in regions[rid].neighbors:
            if nb in fold_map:
                continue
            if not any(parent in fold_map for parent in regions[nb].neighbors):
                continue
            fold, mean, score = _best_fold_for_region(
                regions[nb],
                fold_map,
                regions,
                nyquist=nyquist,
                reference=reference,
                reference_weight=reference_weight,
                max_abs_fold=max_abs_fold,
            )
            fold_map[nb] = fold
            mean_map[nb] = mean
            score_map[nb] = score
            queue.append(nb)

    unresolved = [region.region_id for region in regions if region.region_id not in fold_map]
    for rid in unresolved:
        fold, mean, score = _best_fold_for_region(
            regions[rid],
            fold_map,
            regions,
            nyquist=nyquist,
            reference=reference,
            reference_weight=reference_weight,
            max_abs_fold=max_abs_fold,
        )
        fold_map[rid] = fold
        mean_map[rid] = mean
        score_map[rid] = score

    for _ in range(max_iterations):
        changes = 0
        for region in regions:
            current_fold = fold_map[region.region_id]
            current_mean = mean_map[region.region_id]
            current_score = score_map[region.region_id]
            best_fold, best_mean, best_score = _best_fold_for_region(
                region,
                fold_map,
                regions,
                nyquist=nyquist,
                reference=reference,
                reference_weight=reference_weight,
                max_abs_fold=max_abs_fold,
            )
            if best_fold != current_fold and best_score + 1e-8 < current_score:
                fold_map[region.region_id] = best_fold
                mean_map[region.region_id] = best_mean
                score_map[region.region_id] = best_score
                changes += 1
            else:
                mean_map[region.region_id] = current_mean
                score_map[region.region_id] = current_score
        if changes == 0:
            break
    return fold_map, mean_map, score_map


def _expand_region_solution(
    observed: np.ndarray,
    regions: list[_Region],
    fold_map: dict[int, int],
    mean_map: dict[int, float],
    score_map: dict[int, float],
    *,
    nyquist: float,
    reference: np.ndarray | None,
    wrap_azimuth: bool,
) -> tuple[np.ndarray, np.ndarray]:
    coarse = np.full(observed.shape, np.nan, dtype=float)
    confidence = np.zeros(observed.shape, dtype=float)
    covered = np.zeros(observed.shape, dtype=bool)
    for region in regions:
        corrected_mean = mean_map[region.region_id]
        coarse[region.row0:region.row1, region.col0:region.col1] = corrected_mean
        covered[region.row0:region.row1, region.col0:region.col1] = True
        scale = max(0.30 * nyquist, 1.0 + 0.12 * region.texture)
        conf = float(np.clip(np.exp(-0.5 * (score_map[region.region_id] / max(scale, 1e-6)) ** 2), 0.05, 0.99))
        confidence[region.row0:region.row1, region.col0:region.col1] = conf

    coarse_neighbors = neighbor_stack(coarse, include_diagonals=True, wrap_azimuth=wrap_azimuth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            smooth = np.nanmedian(coarse_neighbors, axis=0)
    reference_field = combine_references(coarse, smooth, reference)
    if reference_field is None:
        reference_field = coarse

    corrected = unfold_to_reference(observed, reference_field, nyquist)
    corrected = np.where(np.isfinite(observed) & covered, corrected, np.nan)
    confidence = np.where(covered, confidence, 0.0)
    if np.any(np.isfinite(reference_field)):
        mismatch = np.abs(corrected - reference_field)
        confidence = np.where(np.isfinite(mismatch), np.maximum(confidence, gaussian_confidence(mismatch, 0.40 * nyquist)), confidence)
    return corrected, confidence


def _python_dealias_sweep_region_graph(
    observed: Iterable[float] | np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    block_shape: tuple[int, int] | None = None,
    reference_weight: float = 0.75,
    max_iterations: int = 6,
    max_abs_fold: int = 8,
    wrap_azimuth: bool = True,
    min_region_area: int = _DEFAULT_MIN_REGION_AREA,
    min_valid_fraction: float = _DEFAULT_MIN_VALID_FRACTION,
) -> DealiasResult:
    """Sweep-wide region graph solver inspired by Py-ART's dynamic network family.

    The sweep is partitioned into connected rectangular regions, each region gets a
    fold offset by consensus with neighboring regions, and the offsets are merged
    iteratively until the graph stabilizes.
    """
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError("observed must be 2D [azimuth, range]")
    if nyquist <= 0:
        raise ValueError("nyquist must be positive")

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError("reference must match observed shape")
    if min_region_area < 0:
        raise ValueError("min_region_area must be non-negative")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    valid = np.isfinite(obs)
    if not np.any(valid):
        return attach_result_state_from_fields(
            DealiasResult(
                velocity=np.full(obs.shape, np.nan, dtype=float),
                folds=np.zeros(obs.shape, dtype=np.int16),
                confidence=np.zeros(obs.shape, dtype=float),
                reference=ref,
                metadata={
                    "paper_family": "PyARTRegionGraphLite",
                    "method": "region_graph_sweep",
                    "region_count": 0,
                    "merge_iterations": 0,
                    "min_region_area": int(min_region_area),
                    "min_valid_fraction": float(min_valid_fraction),
                    "skipped_sparse_blocks": 0,
                },
            ),
            obs,
            source="region_graph_sweep",
            parent="PyARTRegionGraphLite",
            fill_policy="region_graph_consensus_conservative",
        )

    regions, block_ids, skipped_sparse_blocks = _build_regions(
        obs,
        reference=ref,
        block_shape=block_shape,
        wrap_azimuth=wrap_azimuth,
        min_region_area=int(min_region_area),
        min_valid_fraction=float(min_valid_fraction),
    )
    if not regions:
        return attach_result_state_from_fields(
            DealiasResult(
                velocity=np.full(obs.shape, np.nan, dtype=float),
                folds=np.zeros(obs.shape, dtype=np.int16),
                confidence=np.zeros(obs.shape, dtype=float),
                reference=ref,
                metadata={
                    "paper_family": "PyARTRegionGraphLite",
                    "method": "region_graph_sweep",
                    "region_count": 0,
                    "merge_iterations": 0,
                    "block_shape": [int(v) for v in _choose_block_shape(obs.shape, block_shape)],
                    "wrap_azimuth": bool(wrap_azimuth),
                    "block_grid_shape": [int(v) for v in block_ids.shape],
                    "min_region_area": int(min_region_area),
                    "min_valid_fraction": float(min_valid_fraction),
                    "skipped_sparse_blocks": int(skipped_sparse_blocks),
                },
            ),
            obs,
            source="region_graph_sweep",
            parent="PyARTRegionGraphLite",
            fill_policy="region_graph_consensus_conservative",
        )
    fold_map, mean_map, score_map = _propagate_region_folds(
        regions,
        nyquist=nyquist,
        reference=ref,
        reference_weight=reference_weight,
        max_abs_fold=max_abs_fold,
        max_iterations=max_iterations,
    )
    corrected, confidence = _expand_region_solution(
        obs,
        regions,
        fold_map,
        mean_map,
        score_map,
        nyquist=nyquist,
        reference=ref,
        wrap_azimuth=wrap_azimuth,
    )
    folds = fold_counts(corrected, obs, nyquist)

    metadata = {
        "paper_family": "PyARTRegionGraphLite",
        "method": "region_graph_sweep",
        "region_count": int(len(regions)),
        "assigned_regions": int(len(fold_map)),
        "seed_region": int(_pick_seed_region(regions, ref)) if regions else None,
        "block_shape": [int(v) for v in _choose_block_shape(obs.shape, block_shape)],
        "merge_iterations": int(max_iterations),
        "wrap_azimuth": bool(wrap_azimuth),
        "average_fold": float(np.nanmean(folds[np.isfinite(corrected)])) if np.any(np.isfinite(corrected)) else 0.0,
        "regions_with_reference": int(sum(np.isfinite(_region_mean(ref, r.row0, r.row1, r.col0, r.col1)) for r in regions)) if ref is not None else 0,
        "block_grid_shape": [int(v) for v in block_ids.shape],
        "min_region_area": int(min_region_area),
        "min_valid_fraction": float(min_valid_fraction),
        "skipped_sparse_blocks": int(skipped_sparse_blocks),
    }
    return attach_result_state_from_fields(
        DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=ref,
        metadata=metadata,
        ),
        obs,
        source="region_graph_sweep",
        parent="PyARTRegionGraphLite",
        fill_policy="region_graph_consensus_conservative",
    )


def dealias_sweep_region_graph(
    observed: Iterable[float] | np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    block_shape: tuple[int, int] | None = None,
    reference_weight: float = 0.75,
    max_iterations: int = 6,
    max_abs_fold: int = 8,
    wrap_azimuth: bool = True,
    min_region_area: int = _DEFAULT_MIN_REGION_AREA,
    min_valid_fraction: float = _DEFAULT_MIN_VALID_FRACTION,
) -> DealiasResult:
    """Sweep-wide region graph solver inspired by Py-ART's dynamic network family.

    The sweep is partitioned into connected rectangular regions, each region gets a
    fold offset by consensus with neighboring regions, and the offsets are merged
    iteratively until the graph stabilizes.
    """
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError("observed must be 2D [azimuth, range]")
    if nyquist <= 0:
        raise ValueError("nyquist must be positive")

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError("reference must match observed shape")
    if min_region_area < 0:
        raise ValueError("min_region_area must be non-negative")
    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    backend = resolve_rust_backend(_NATIVE_BACKEND)
    if backend is not None and hasattr(backend, "dealias_sweep_region_graph"):
        backend_method = backend.dealias_sweep_region_graph
        supports_conservative_args = True
        try:
            signature = inspect.signature(backend_method)
            accepts_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values())
            positional_params = [
                param
                for param in signature.parameters.values()
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if not accepts_varargs and len(positional_params) < 10:
                supports_conservative_args = False
        except (TypeError, ValueError):
            supports_conservative_args = True

        if supports_conservative_args:
            native_result = backend_method(
                obs,
                float(nyquist),
                ref,
                None if block_shape is None else tuple(int(v) for v in block_shape),
                float(reference_weight),
                int(max_iterations),
                int(max_abs_fold),
                bool(wrap_azimuth),
                int(min_region_area),
                float(min_valid_fraction),
            )
        else:
            native_result = backend_method(
                obs,
                float(nyquist),
                ref,
                None if block_shape is None else tuple(int(v) for v in block_shape),
                float(reference_weight),
                int(max_iterations),
                int(max_abs_fold),
                bool(wrap_azimuth),
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
            raise ValueError("native region_graph backend returned an unexpected result shape")
        meta = dict(metadata)
        meta.setdefault("paper_family", "PyARTRegionGraphLite")
        meta.setdefault("method", "region_graph_sweep")
        meta.setdefault("merge_iterations", int(max_iterations))
        meta.setdefault("wrap_azimuth", bool(wrap_azimuth))
        meta.setdefault("block_shape", [int(v) for v in _choose_block_shape(obs.shape, block_shape)])
        meta.setdefault("min_region_area", int(min_region_area))
        meta.setdefault("min_valid_fraction", float(min_valid_fraction))
        meta.setdefault("skipped_sparse_blocks", 0)
        return attach_result_state_from_fields(
            DealiasResult(
            velocity=np.asarray(velocity, dtype=float),
            folds=np.asarray(folds, dtype=np.int16),
            confidence=np.asarray(confidence, dtype=float),
            reference=None if ref_out is None else np.asarray(ref_out, dtype=float),
            metadata=meta,
            ),
            obs,
            source="region_graph_sweep",
            parent="PyARTRegionGraphLite",
            fill_policy=str(meta.get("fill_policy", "region_graph_consensus_conservative")),
        )

    return _python_dealias_sweep_region_graph(
        obs,
        nyquist,
        reference=ref,
        block_shape=block_shape,
        reference_weight=reference_weight,
        max_iterations=max_iterations,
        max_abs_fold=max_abs_fold,
        wrap_azimuth=wrap_azimuth,
        min_region_area=min_region_area,
        min_valid_fraction=min_valid_fraction,
    )
