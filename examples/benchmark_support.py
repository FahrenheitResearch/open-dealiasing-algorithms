from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np

from open_dealias import dealias_sweep_zw06
from open_dealias._rust_bridge import backend_policy, get_backend_policy, resolve_rust_backend


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
    reference_name: str = "reference",
) -> dict[str, object]:
    return {
        "name": name,
        "status": "computed",
        "runtime_s": float(runtime_s),
        "changed_gates": changed_gates(candidate, observed),
        "unresolved_fraction": unresolved_fraction(candidate, observed),
        "resolved_fraction": 1.0 - unresolved_fraction(candidate, observed),
        "reference_name": str(reference_name),
        "metric_scope": "reference_consistency",
        "mae_vs_reference": mae(candidate, reference),
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

    native_state = getattr(native, "result_state", None)
    python_state = getattr(python, "result_state", None)
    native_resolved = None if native_state is None else getattr(native_state, "resolved_mask", None)
    python_resolved = None if python_state is None else getattr(python_state, "resolved_mask", None)
    if native_resolved is not None and python_resolved is not None:
        native_resolved = np.asarray(native_resolved, dtype=bool)
        python_resolved = np.asarray(python_resolved, dtype=bool)
        payload["resolved_mask_equal"] = bool(np.array_equal(native_resolved, python_resolved))
        payload["resolved_mask_mismatch_count"] = int(np.count_nonzero(native_resolved != python_resolved))

    native_provenance = None if native_state is None else getattr(native_state, "provenance", None)
    python_provenance = None if python_state is None else getattr(python_state, "provenance", None)
    if native_provenance is not None and python_provenance is not None:
        payload["provenance_equal"] = bool(
            np.array_equal(np.asarray(native_provenance), np.asarray(python_provenance))
        )

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
    backend_method: str | None = None,
    repeats: int = 3,
    warmup: int = 1,
    **kwargs: object,
) -> dict[str, object]:
    solver = getattr(module, solver_name)
    baseline_policy = get_backend_policy()
    original_backend = getattr(module, backend_attr) if hasattr(module, backend_attr) else None

    def _timed_call(policy_name: str) -> tuple[object, float]:
        result = None
        with backend_policy(policy_name):
            if hasattr(module, backend_attr):
                if policy_name == "python":
                    setattr(module, backend_attr, None)
                else:
                    setattr(module, backend_attr, original_backend)
            for _ in range(max(0, int(warmup))):
                result = solver(*args, **kwargs)
            samples: list[float] = []
            for _ in range(max(1, int(repeats))):
                start = time.perf_counter()
                result = solver(*args, **kwargs)
                samples.append(time.perf_counter() - start)
        if hasattr(module, backend_attr):
            setattr(module, backend_attr, original_backend)
        return result, float(np.median(np.asarray(samples, dtype=float)))

    public_result, public_runtime_s = _timed_call("auto")
    python_result, python_runtime_s = _timed_call("python")

    backend_available = False

    if hasattr(module, backend_attr):
        backend = original_backend
        with backend_policy("auto"):
            resolved_backend = resolve_rust_backend(backend)
        if resolved_backend is not None:
            method_name = solver_name if backend_method is None else backend_method
            method = getattr(resolved_backend, method_name, None)
            backend_available = callable(method)

    return {
        "public_result": public_result,
        "public_runtime_s": float(public_runtime_s),
        "python_result": python_result,
        "python_runtime_s": float(python_runtime_s),
        "python_policy_result": python_result,
        "python_policy_runtime_s": float(python_runtime_s),
        "backend_available": backend_available,
        "native_acceleration_available": backend_available,
        "initial_policy": baseline_policy,
        "public_policy": "auto",
        "python_policy": "python",
        "comparison_mode": "prepared_solver_call_auto_vs_transitive_python_policy",
        "comparison_scope": "prepared_solver_call",
        "repeats": int(max(1, repeats)),
        "warmup": int(max(0, warmup)),
    }
