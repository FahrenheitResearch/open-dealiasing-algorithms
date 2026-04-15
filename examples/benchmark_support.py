from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np

from open_dealias import dealias_sweep_zw06


def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if np.any(mask) else float("nan")


def changed_gates(candidate: np.ndarray, observed: np.ndarray) -> int:
    mask = np.isfinite(candidate) & np.isfinite(observed)
    return int(np.count_nonzero(np.abs(candidate[mask] - observed[mask]) > 1e-6))


def unresolved_fraction(candidate: np.ndarray, observed: np.ndarray) -> float:
    obs_mask = np.isfinite(observed)
    if not np.any(obs_mask):
        return float("nan")
    unresolved = obs_mask & ~np.isfinite(candidate)
    return float(np.mean(unresolved))


def record_scored_result(
    name: str,
    candidate: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray,
    runtime_s: float,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "status": "computed",
        "runtime_s": float(runtime_s),
        "changed_gates": changed_gates(candidate, observed),
        "unresolved_fraction": unresolved_fraction(candidate, observed),
        "mae_vs_pyart": mae(candidate, reference),
        "metadata": {} if metadata is None else dict(metadata),
    }


def record_skipped_result(name: str, reason: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "skipped",
        "skip_reason": reason,
    }


def open_zw06_sweep_anchor(previous_observed: np.ndarray, nyquist: float) -> tuple[np.ndarray, dict[str, object]]:
    result = dealias_sweep_zw06(previous_observed, nyquist)
    metadata = {
        "source": "open_zw06",
        "paper_family": result.metadata.get("paper_family"),
        "method": result.metadata.get("method"),
        "nyquist": float(nyquist),
    }
    return result.velocity, metadata


def open_zw06_volume_anchor(previous_volume_observed: np.ndarray, nyquist_by_sweep: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    obs = np.asarray(previous_volume_observed, dtype=float)
    nyq = np.asarray(nyquist_by_sweep, dtype=float)
    if obs.ndim != 3:
        raise ValueError("previous_volume_observed must be 3D [sweep, azimuth, range]")
    if nyq.shape != (obs.shape[0],):
        raise ValueError("nyquist_by_sweep must have one value per sweep")

    anchors: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for sweep_index, sweep_obs in enumerate(obs):
        result = dealias_sweep_zw06(sweep_obs, float(nyq[sweep_index]))
        anchors.append(result.velocity)
        metadata.append(
            {
                "sweep_index": int(sweep_index),
                "source": "open_zw06",
                "paper_family": result.metadata.get("paper_family"),
                "method": result.metadata.get("method"),
                "nyquist": float(nyq[sweep_index]),
            }
        )
    return np.stack(anchors, axis=0), metadata


def safe_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmax(np.abs(a[mask] - b[mask])))


def compare_dealias_results(native: Any, python: Any) -> dict[str, object]:
    native_velocity = np.asarray(native.velocity, dtype=float)
    python_velocity = np.asarray(python.velocity, dtype=float)
    native_folds = np.asarray(native.folds)
    python_folds = np.asarray(python.folds)
    native_confidence = np.asarray(native.confidence, dtype=float)
    python_confidence = np.asarray(python.confidence, dtype=float)

    payload: dict[str, object] = {
        "velocity_mae": mae(native_velocity, python_velocity),
        "velocity_max_abs_diff": safe_max_abs_diff(native_velocity, python_velocity),
        "folds_equal": bool(np.array_equal(native_folds, python_folds)),
        "folds_mismatch_count": int(np.count_nonzero(native_folds != python_folds)),
        "confidence_mae": mae(native_confidence, python_confidence),
        "confidence_max_abs_diff": safe_max_abs_diff(native_confidence, python_confidence),
    }

    native_reference = getattr(native, "reference", None)
    python_reference = getattr(python, "reference", None)
    if native_reference is not None and python_reference is not None:
        native_reference = np.asarray(native_reference, dtype=float)
        python_reference = np.asarray(python_reference, dtype=float)
        payload["reference_mae"] = mae(native_reference, python_reference)
        payload["reference_max_abs_diff"] = safe_max_abs_diff(native_reference, python_reference)

    return payload


def discover_archive_files(archive_root: Path) -> list[Path]:
    if not archive_root.exists():
        return []
    return sorted(path for path in archive_root.iterdir() if path.is_file() and path.suffix.lower() == ".zip")


def run_solver_pair(
    module: Any,
    solver_name: str,
    *args: object,
    backend_attr: str = "_NATIVE_BACKEND",
    **kwargs: object,
) -> dict[str, object]:
    solver = getattr(module, solver_name)

    public_start = time.perf_counter()
    public_result = solver(*args, **kwargs)
    public_runtime_s = time.perf_counter() - public_start

    python_result = public_result
    python_runtime_s = public_runtime_s
    backend_available = False

    if hasattr(module, backend_attr):
        backend = getattr(module, backend_attr)
        backend_available = backend is not None
        setattr(module, backend_attr, None)
        try:
            python_start = time.perf_counter()
            python_result = solver(*args, **kwargs)
            python_runtime_s = time.perf_counter() - python_start
        finally:
            setattr(module, backend_attr, backend)

    return {
        "public_result": public_result,
        "public_runtime_s": float(public_runtime_s),
        "python_result": python_result,
        "python_runtime_s": float(python_runtime_s),
        "backend_available": backend_available,
    }
