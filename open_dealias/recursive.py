from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import warnings

import numpy as np

from ._core import as_float_array, combine_references, fold_counts, gaussian_confidence, neighbor_stack, texture_3x3, unfold_to_reference
from ._rust_bridge import get_rust_backend
from .region_graph import (
    _Region,
    _expand_region_solution,
    _propagate_region_folds,
    _safe_nanmedian,
    _wrap_delta,
    dealias_sweep_region_graph,
)
from .types import DealiasResult

_NATIVE_BACKEND = get_rust_backend()


@dataclass(slots=True)
class _Node:
    row0: int
    row1: int
    col0: int
    col1: int
    depth: int
    mean_obs: float
    texture: float
    area: int
    children: list["_Node"] = field(default_factory=list)
    leaf_ids: list[int] = field(default_factory=list)


def _safe_nanmedian_axis(data: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            return np.nanmedian(data, axis=axis)


def _node_stats(observed: np.ndarray, row0: int, row1: int, col0: int, col1: int, *, wrap_azimuth: bool) -> tuple[float, float, int]:
    block = observed[row0:row1, col0:col1]
    texture_map = texture_3x3(block, wrap_azimuth=wrap_azimuth)
    mean_obs = _safe_nanmedian(block)
    texture = _safe_nanmedian(texture_map)
    if not np.isfinite(texture):
        texture = 0.0
    area = int(np.count_nonzero(np.isfinite(block)))
    return mean_obs, texture, area


def _profile_energy(block: np.ndarray, axis: int, nyquist: float) -> tuple[float, int]:
    if axis == 0:
        profile = _safe_nanmedian_axis(block, axis=1)
    else:
        profile = _safe_nanmedian_axis(block, axis=0)
    if profile.size < 2:
        return 0.0, -1
    diffs = np.abs(_wrap_delta(profile[1:], profile[:-1], nyquist))
    if not np.any(np.isfinite(diffs)):
        return 0.0, -1
    cut = int(np.nanargmax(diffs)) + 1
    energy = float(np.nanmedian(diffs[np.isfinite(diffs)])) if np.any(np.isfinite(diffs)) else 0.0
    return energy, cut


def _split_node(
    node: _Node,
    observed: np.ndarray,
    nyquist: float,
    *,
    depth_limit: int,
    min_leaf_cells: int,
    split_texture_fraction: float,
    wrap_azimuth: bool,
) -> None:
    block = observed[node.row0:node.row1, node.col0:node.col1]
    node.mean_obs, node.texture, node.area = _node_stats(observed, node.row0, node.row1, node.col0, node.col1, wrap_azimuth=wrap_azimuth)
    rows = node.row1 - node.row0
    cols = node.col1 - node.col0
    if node.depth >= depth_limit or node.area <= min_leaf_cells or rows <= 2 or cols <= 2:
        return
    if node.texture <= split_texture_fraction * nyquist and min(rows, cols) <= max(6, min(rows, cols) // 2):
        return

    row_energy, row_cut = _profile_energy(block, 0, nyquist)
    col_energy, col_cut = _profile_energy(block, 1, nyquist)
    axis = 0 if row_energy >= col_energy else 1
    cut = row_cut if axis == 0 else col_cut

    if cut <= 0 or cut >= (rows if axis == 0 else cols):
        axis = 1 - axis
        cut = col_cut if axis == 1 else row_cut
        if cut <= 0 or cut >= (rows if axis == 0 else cols):
            cut = (rows if axis == 0 else cols) // 2

    if axis == 0:
        split_row = node.row0 + cut
        top = observed[node.row0:split_row, node.col0:node.col1]
        bottom = observed[split_row:node.row1, node.col0:node.col1]
        if np.count_nonzero(np.isfinite(top)) < min_leaf_cells or np.count_nonzero(np.isfinite(bottom)) < min_leaf_cells:
            split_row = node.row0 + rows // 2
            top = observed[node.row0:split_row, node.col0:node.col1]
            bottom = observed[split_row:node.row1, node.col0:node.col1]
            if np.count_nonzero(np.isfinite(top)) < min_leaf_cells or np.count_nonzero(np.isfinite(bottom)) < min_leaf_cells:
                return
        node.children = [
            _Node(node.row0, split_row, node.col0, node.col1, node.depth + 1, *_node_stats(observed, node.row0, split_row, node.col0, node.col1, wrap_azimuth=wrap_azimuth)),
            _Node(split_row, node.row1, node.col0, node.col1, node.depth + 1, *_node_stats(observed, split_row, node.row1, node.col0, node.col1, wrap_azimuth=wrap_azimuth)),
        ]
    else:
        split_col = node.col0 + cut
        left = observed[node.row0:node.row1, node.col0:split_col]
        right = observed[node.row0:node.row1, split_col:node.col1]
        if np.count_nonzero(np.isfinite(left)) < min_leaf_cells or np.count_nonzero(np.isfinite(right)) < min_leaf_cells:
            split_col = node.col0 + cols // 2
            left = observed[node.row0:node.row1, node.col0:split_col]
            right = observed[node.row0:node.row1, split_col:node.col1]
            if np.count_nonzero(np.isfinite(left)) < min_leaf_cells or np.count_nonzero(np.isfinite(right)) < min_leaf_cells:
                return
        node.children = [
            _Node(node.row0, node.row1, node.col0, split_col, node.depth + 1, *_node_stats(observed, node.row0, node.row1, node.col0, split_col, wrap_azimuth=wrap_azimuth)),
            _Node(node.row0, node.row1, split_col, node.col1, node.depth + 1, *_node_stats(observed, node.row0, node.row1, split_col, node.col1, wrap_azimuth=wrap_azimuth)),
        ]

    for child in node.children:
        _split_node(
            child,
            observed,
            nyquist,
            depth_limit=depth_limit,
            min_leaf_cells=min_leaf_cells,
            split_texture_fraction=split_texture_fraction,
            wrap_azimuth=wrap_azimuth,
        )


def _collect_leaves(node: _Node, leaves: list[_Node]) -> None:
    if not node.children:
        node.leaf_ids = [len(leaves)]
        leaves.append(node)
        return
    node.leaf_ids = []
    for child in node.children:
        _collect_leaves(child, leaves)
        node.leaf_ids.extend(child.leaf_ids)


def _touches(a: _Region, b: _Region, rows: int, *, wrap_azimuth: bool) -> bool:
    row_overlap = min(a.row1, b.row1) - max(a.row0, b.row0)
    col_overlap = min(a.col1, b.col1) - max(a.col0, b.col0)
    if a.col1 == b.col0 or b.col1 == a.col0:
        return row_overlap > 0
    if a.row1 == b.row0 or b.row1 == a.row0:
        return col_overlap > 0
    if wrap_azimuth and rows > 1 and ((a.row0 == 0 and b.row1 == rows) or (b.row0 == 0 and a.row1 == rows)):
        return col_overlap > 0
    return False


def _build_leaf_regions(leaves: list[_Node], rows: int, *, wrap_azimuth: bool) -> list[_Region]:
    regions: list[_Region] = []
    for rid, leaf in enumerate(leaves):
        regions.append(
            _Region(
                region_id=rid,
                row0=leaf.row0,
                row1=leaf.row1,
                col0=leaf.col0,
                col1=leaf.col1,
                mean_obs=leaf.mean_obs,
                texture=leaf.texture,
                area=leaf.area,
            )
        )
    for i, left in enumerate(regions):
        for j in range(i + 1, len(regions)):
            right = regions[j]
            if not _touches(left, right, rows, wrap_azimuth=wrap_azimuth):
                continue
            left.neighbors.add(j)
            right.neighbors.add(i)
            if left.col1 == right.col0 or right.col1 == left.col0:
                weight = max(1, min(left.row1, right.row1) - max(left.row0, right.row0))
            else:
                weight = max(1, min(left.col1, right.col1) - max(left.col0, right.col0))
            left.boundary_weight[j] = weight
            right.boundary_weight[i] = weight
    return regions


def _leaf_mean(leaves: list[_Node], fold_map: dict[int, int], nyquist: float, ids: list[int]) -> float:
    values = []
    weights = []
    for lid in ids:
        leaf = leaves[lid]
        if lid not in fold_map:
            continue
        values.append(leaf.mean_obs + 2.0 * nyquist * fold_map[lid])
        weights.append(max(1, leaf.area))
    if not values:
        return float("nan")
    return float(np.average(values, weights=weights))


def _node_anchor(node: _Node, leaves: list[_Node], fold_map: dict[int, int], nyquist: float, reference: np.ndarray | None) -> float:
    child_mean = _leaf_mean(leaves, fold_map, nyquist, node.leaf_ids)
    if reference is None:
        return child_mean
    ref_slice = reference[node.row0:node.row1, node.col0:node.col1]
    ref_mean = _safe_nanmedian(ref_slice)
    return float(np.nanmedian([v for v in (child_mean, ref_mean) if np.isfinite(v)])) if np.isfinite(child_mean) or np.isfinite(ref_mean) else float("nan")


def _shift_subtree(node: _Node, leaves: list[_Node], fold_map: dict[int, int], delta: int) -> None:
    for lid in node.leaf_ids:
        fold_map[lid] = fold_map.get(lid, 0) + int(delta)


def _best_fold_from_targets(
    mean_obs: float,
    target_means: list[float],
    target_weights: list[float],
    *,
    ref_mean: float | None,
    nyquist: float,
    reference_weight: float,
    max_abs_fold: int,
) -> tuple[int, float, float]:
    if target_means:
        target = float(np.average(target_means, weights=target_weights))
        center = int(np.rint((target - mean_obs) / (2.0 * nyquist)))
    elif ref_mean is not None and np.isfinite(ref_mean):
        center = int(np.rint((float(ref_mean) - mean_obs) / (2.0 * nyquist)))
    else:
        center = 0
    center = int(np.clip(center, -max_abs_fold, max_abs_fold))

    candidate_folds = range(max(-max_abs_fold, center - 3), min(max_abs_fold, center + 3) + 1)
    best_fold = center
    best_mean = mean_obs + 2.0 * nyquist * center
    best_score = float("inf")
    for fold in candidate_folds:
        candidate_mean = mean_obs + 2.0 * nyquist * fold
        score = 0.35 * abs(fold)
        for target, weight in zip(target_means, target_weights):
            score += weight * abs(candidate_mean - target)
        if ref_mean is not None and np.isfinite(ref_mean):
            score += reference_weight * abs(candidate_mean - float(ref_mean))
        if score < best_score:
            best_fold = int(fold)
            best_mean = float(candidate_mean)
            best_score = float(score)
    return best_fold, best_mean, best_score


def _directional_refine(
    leaves: list[_Node],
    regions: list[_Region],
    fold_map: dict[int, int],
    mean_map: dict[int, float],
    *,
    nyquist: float,
    reference: np.ndarray | None,
    axis: int,
    reference_weight: float,
    max_abs_fold: int,
    reverse: bool = False,
) -> int:
    order = sorted(
        range(len(leaves)),
        key=lambda lid: (
            (leaves[lid].col0 + leaves[lid].col1) / 2.0 if axis == 1 else (leaves[lid].row0 + leaves[lid].row1) / 2.0,
            (leaves[lid].row0 + leaves[lid].row1) / 2.0 if axis == 1 else (leaves[lid].col0 + leaves[lid].col1) / 2.0,
        ),
        reverse=reverse,
    )
    changes = 0
    for lid in order:
        leaf = leaves[lid]
        target_means: list[float] = []
        target_weights: list[float] = []
        for other_id in order:
            if other_id == lid:
                continue
            other = leaves[other_id]
            if fold_map.get(other_id) is None:
                continue
            if axis == 1:
                if reverse:
                    if other.col0 < leaf.col1:
                        continue
                else:
                    if other.col1 > leaf.col0:
                        continue
                overlap = min(leaf.row1, other.row1) - max(leaf.row0, other.row0)
                if overlap <= 0:
                    continue
                distance = abs((leaf.col0 + leaf.col1) / 2.0 - (other.col0 + other.col1) / 2.0)
            else:
                if reverse:
                    if other.row0 < leaf.row1:
                        continue
                else:
                    if other.row1 > leaf.row0:
                        continue
                overlap = min(leaf.col1, other.col1) - max(leaf.col0, other.col0)
                if overlap <= 0:
                    continue
                distance = abs((leaf.row0 + leaf.row1) / 2.0 - (other.row0 + other.row1) / 2.0)
            corrected = mean_map[other_id]
            target_means.append(float(corrected))
            target_weights.append(float(max(1.0, overlap) / (1.0 + distance)))

        ref_mean = None
        if reference is not None:
            ref_mean = _safe_nanmedian(reference[leaf.row0:leaf.row1, leaf.col0:leaf.col1])

        best_fold, best_mean, best_score = _best_fold_from_targets(
            leaf.mean_obs,
            target_means,
            target_weights,
            ref_mean=ref_mean,
            nyquist=nyquist,
            reference_weight=reference_weight,
            max_abs_fold=max_abs_fold,
        )
        current_fold = fold_map.get(lid, 0)
        current_mean = mean_map.get(lid, leaf.mean_obs + 2.0 * nyquist * current_fold)
        current_score = _best_fold_from_targets(
            leaf.mean_obs,
            [current_mean],
            [1.0],
            ref_mean=ref_mean,
            nyquist=nyquist,
            reference_weight=reference_weight,
            max_abs_fold=max_abs_fold,
        )[2]
        if best_fold != current_fold and best_score + 1e-8 < current_score:
            fold_map[lid] = best_fold
            mean_map[lid] = best_mean
            changes += 1
    return changes


def _refine_tree(node: _Node, leaves: list[_Node], fold_map: dict[int, int], nyquist: float, reference: np.ndarray | None) -> int:
    if not node.children:
        return 0
    changes = 0
    anchor = _node_anchor(node, leaves, fold_map, nyquist, reference)
    if np.isfinite(anchor):
        for child in node.children:
            child_mean = _leaf_mean(leaves, fold_map, nyquist, child.leaf_ids)
            if not np.isfinite(child_mean):
                continue
            delta = int(np.rint((anchor - child_mean) / (2.0 * nyquist)))
            if delta != 0 and abs((child_mean + 2.0 * nyquist * delta) - anchor) < abs(child_mean - anchor):
                _shift_subtree(child, leaves, fold_map, delta)
                changes += 1
    for child in node.children:
        changes += _refine_tree(child, leaves, fold_map, nyquist, reference)
    return changes


def _make_reference_field(
    observed: np.ndarray,
    leaves: list[_Node],
    fold_map: dict[int, int],
    nyquist: float,
    *,
    reference: np.ndarray | None,
    wrap_azimuth: bool,
) -> np.ndarray:
    coarse = np.full(observed.shape, np.nan, dtype=float)
    for lid, leaf in enumerate(leaves):
        corrected_mean = leaf.mean_obs + 2.0 * nyquist * fold_map[lid]
        coarse[leaf.row0:leaf.row1, leaf.col0:leaf.col1] = corrected_mean
    smooth = np.nanmedian(neighbor_stack(coarse, include_diagonals=True, wrap_azimuth=wrap_azimuth), axis=0)
    combined = combine_references(coarse, smooth, reference)
    return coarse if combined is None else combined


def _solution_cost(candidate: np.ndarray, observed: np.ndarray, reference: np.ndarray | None, nyquist: float) -> float:
    valid = np.isfinite(candidate) & np.isfinite(observed)
    if not np.any(valid):
        return float("inf")
    # Penalize large wrapped discontinuities plus disagreement with a trusted bootstrap reference.
    dy = np.abs(_wrap_delta(candidate[:, 1:], candidate[:, :-1], nyquist))
    dx = np.abs(_wrap_delta(candidate[1:, :], candidate[:-1, :], nyquist))
    continuity = float(np.nanmean(dy)) + float(np.nanmean(dx))
    if reference is not None:
        ref_valid = valid & np.isfinite(reference)
        ref_cost = float(np.nanmean(np.abs(candidate[ref_valid] - reference[ref_valid]))) if np.any(ref_valid) else 0.0
    else:
        ref_cost = 0.0
    return continuity + 0.35 * ref_cost


def _python_dealias_sweep_recursive(
    observed: Iterable[float] | np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_depth: int = 5,
    min_leaf_cells: int = 24,
    split_texture_fraction: float = 0.60,
    reference_weight: float = 0.70,
    max_abs_fold: int = 8,
    wrap_azimuth: bool = True,
) -> DealiasResult:
    """R2D2-style recursive sweep solver with hierarchical local refinement."""
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError("observed must be 2D [azimuth, range]")
    if nyquist <= 0:
        raise ValueError("nyquist must be positive")

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError("reference must match observed shape")

    bootstrap = dealias_sweep_region_graph(
        obs,
        nyquist,
        reference=ref,
        wrap_azimuth=wrap_azimuth,
    )
    bootstrap_ref = combine_references(ref, bootstrap.velocity)

    root_mean, root_texture, root_area = _node_stats(obs, 0, obs.shape[0], 0, obs.shape[1], wrap_azimuth=wrap_azimuth)
    root = _Node(0, obs.shape[0], 0, obs.shape[1], 0, root_mean, root_texture, root_area)
    _split_node(
        root,
        obs,
        nyquist,
        depth_limit=max_depth,
        min_leaf_cells=min_leaf_cells,
        split_texture_fraction=split_texture_fraction,
        wrap_azimuth=wrap_azimuth,
    )

    leaves: list[_Node] = []
    _collect_leaves(root, leaves)
    regions = _build_leaf_regions(leaves, obs.shape[0], wrap_azimuth=wrap_azimuth)
    fold_map, mean_map, score_map = _propagate_region_folds(
        regions,
        nyquist=nyquist,
        reference=bootstrap_ref,
        reference_weight=reference_weight,
        max_abs_fold=max_abs_fold,
        max_iterations=max_depth + 1,
    )

    # Directional sweeps help the recursive solver accumulate fold changes across
    # the range dimension when a coarse reference is intentionally underfit.
    for _ in range(2):
        changes = 0
        changes += _directional_refine(
            leaves,
            regions,
            fold_map,
            mean_map,
            nyquist=nyquist,
            reference=bootstrap_ref,
            axis=1,
            reference_weight=reference_weight,
            max_abs_fold=max_abs_fold,
            reverse=False,
        )
        changes += _directional_refine(
            leaves,
            regions,
            fold_map,
            mean_map,
            nyquist=nyquist,
            reference=bootstrap_ref,
            axis=1,
            reference_weight=reference_weight,
            max_abs_fold=max_abs_fold,
            reverse=True,
        )
        if changes == 0:
            break

    # Hierarchical local refinement: pull child subtrees toward the node anchor.
    for _ in range(2):
        changes = _refine_tree(root, leaves, fold_map, nyquist, bootstrap_ref)
        if changes == 0:
            break

    for lid, leaf in enumerate(leaves):
        regions[lid].mean_obs = leaf.mean_obs
        regions[lid].texture = leaf.texture
        regions[lid].area = leaf.area
        mean_map[lid] = leaf.mean_obs + 2.0 * nyquist * fold_map[lid]

    corrected, confidence = _expand_region_solution(
        obs,
        regions,
        fold_map,
        mean_map,
        score_map,
        nyquist=nyquist,
        reference=bootstrap_ref,
        wrap_azimuth=wrap_azimuth,
    )
    folds = fold_counts(corrected, obs, nyquist)

    if bootstrap_ref is not None and np.any(np.isfinite(corrected)):
        mismatch = np.abs(corrected - bootstrap_ref)
        confidence = np.where(np.isfinite(mismatch), np.maximum(confidence, gaussian_confidence(mismatch, 0.38 * nyquist)), confidence)

    recursive_cost = _solution_cost(corrected, obs, bootstrap_ref, nyquist)
    bootstrap_cost = _solution_cost(bootstrap.velocity, obs, bootstrap_ref, nyquist)
    if recursive_cost > bootstrap_cost + 1e-8:
        corrected = bootstrap.velocity
        folds = bootstrap.folds
        confidence = np.maximum(confidence, bootstrap.confidence)
        method = "recursive_region_refinement_fallback_region_graph"
    else:
        method = "recursive_region_refinement"

    return DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=bootstrap_ref,
        metadata={
            "paper_family": "R2D2StyleRecursiveLite",
            "method": method,
            "leaf_count": int(len(leaves)),
            "max_depth": int(max_depth),
            "split_texture_fraction": float(split_texture_fraction),
            "reference_weight": float(reference_weight),
            "wrap_azimuth": bool(wrap_azimuth),
            "root_texture": float(root.texture),
            "bootstrap_method": bootstrap.metadata.get("method"),
            "bootstrap_region_count": bootstrap.metadata.get("region_count"),
        },
    )


def dealias_sweep_recursive(
    observed: Iterable[float] | np.ndarray,
    nyquist: float,
    *,
    reference: np.ndarray | None = None,
    max_depth: int = 5,
    min_leaf_cells: int = 24,
    split_texture_fraction: float = 0.60,
    reference_weight: float = 0.70,
    max_abs_fold: int = 8,
    wrap_azimuth: bool = True,
) -> DealiasResult:
    """R2D2-style recursive sweep solver with hierarchical local refinement."""
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError("observed must be 2D [azimuth, range]")
    if nyquist <= 0:
        raise ValueError("nyquist must be positive")

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError("reference must match observed shape")

    if _NATIVE_BACKEND is not None and hasattr(_NATIVE_BACKEND, "dealias_sweep_recursive"):
        native_result = _NATIVE_BACKEND.dealias_sweep_recursive(
            obs,
            float(nyquist),
            ref,
            int(max_depth),
            int(min_leaf_cells),
            float(split_texture_fraction),
            float(reference_weight),
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
            raise ValueError("native recursive backend returned an unexpected result shape")
        meta = dict(metadata)
        meta.setdefault("paper_family", "R2D2StyleRecursiveLite")
        meta.setdefault("method", "recursive_region_refinement")
        meta.setdefault("bootstrap_method", "region_graph_sweep")
        meta.setdefault("max_depth", int(max_depth))
        meta.setdefault("wrap_azimuth", bool(wrap_azimuth))
        return DealiasResult(
            velocity=np.asarray(velocity, dtype=float),
            folds=np.asarray(folds, dtype=np.int16),
            confidence=np.asarray(confidence, dtype=float),
            reference=None if ref_out is None else np.asarray(ref_out, dtype=float),
            metadata=meta,
        )

    return _python_dealias_sweep_recursive(
        obs,
        nyquist,
        reference=ref,
        max_depth=max_depth,
        min_leaf_cells=min_leaf_cells,
        split_texture_fraction=split_texture_fraction,
        reference_weight=reference_weight,
        max_abs_fold=max_abs_fold,
        wrap_azimuth=wrap_azimuth,
    )
