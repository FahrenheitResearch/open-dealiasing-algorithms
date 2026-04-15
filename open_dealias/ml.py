from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import numpy as np

from ._core import as_float_array, fold_counts, gaussian_confidence, neighbor_stack, texture_3x3, unfold_to_reference
from ._rust_bridge import get_rust_backend
from .types import DealiasResult
from .variational import dealias_sweep_variational

__all__ = ["LinearBranchModel", "fit_ml_reference_model", "dealias_sweep_ml"]


_NATIVE_BACKEND = get_rust_backend()


@dataclass(slots=True)
class LinearBranchModel:
    """Simple ridge-regression reference predictor for ML-assisted dealiasing."""

    weights: np.ndarray
    feature_names: list[str]
    ridge: float
    train_rmse: float
    mode: str
    nyquist: float | None


def _native_method(name: str):
    if _NATIVE_BACKEND is None:
        return None
    method = getattr(_NATIVE_BACKEND, name, None)
    return method if callable(method) else None


def _safe_nanmedian(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            return np.nanmedian(arr, axis=axis)


def _build_feature_stack(
    observed: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    azimuth_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    rows, cols = observed.shape
    row_frac = np.linspace(0.0, 1.0, rows, dtype=float)[:, None]
    col_frac = np.linspace(0.0, 1.0, cols, dtype=float)[None, :]
    texture = texture_3x3(observed, wrap_azimuth=True)
    neigh = neighbor_stack(observed, include_diagonals=True, wrap_azimuth=True)
    neigh_med = _safe_nanmedian(neigh, axis=0)

    features = [
        np.ones_like(observed),
        np.where(np.isfinite(observed), observed, 0.0),
        np.where(np.isfinite(neigh_med), neigh_med, 0.0),
        np.where(np.isfinite(texture), texture, 0.0),
        np.broadcast_to(row_frac, observed.shape),
        np.broadcast_to(col_frac, observed.shape),
    ]
    names = ["bias", "observed", "neighbor_median", "texture", "row_frac", "col_frac"]

    if azimuth_deg is not None:
        az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
        if az.shape != (rows,):
            raise ValueError("azimuth_deg must have length n_azimuth")
        sin_az = np.sin(az)[:, None]
        cos_az = np.cos(az)[:, None]
        features.extend([np.broadcast_to(sin_az, observed.shape), np.broadcast_to(cos_az, observed.shape)])
        names.extend(["sin_az", "cos_az"])

    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != observed.shape:
            raise ValueError("reference must match observed shape")
        ref_feature = np.where(np.isfinite(ref), ref, 0.0)
    else:
        ref_feature = np.zeros_like(observed)
    features.append(ref_feature)
    names.append("reference")

    stack = np.stack(features, axis=-1)
    return stack, names


def _coerce_linear_branch_model(native_result: Any) -> LinearBranchModel:
    if isinstance(native_result, LinearBranchModel):
        return native_result
    if isinstance(native_result, dict):
        weights = native_result.get("weights")
        feature_names = native_result.get("feature_names")
        ridge = native_result.get("ridge")
        train_rmse = native_result.get("train_rmse")
        mode = native_result.get("mode")
        nyquist = native_result.get("nyquist")
    else:
        values = tuple(native_result)
        if len(values) == 6:
            weights, feature_names, ridge, train_rmse, mode, nyquist = values
        elif len(values) == 5:
            weights, feature_names, ridge, train_rmse, mode = values
            nyquist = None
        else:  # pragma: no cover - defensive against future API drift.
            raise ValueError("native ML backend returned an unexpected result shape")
    return LinearBranchModel(
        weights=np.asarray(weights, dtype=float),
        feature_names=[str(name) for name in feature_names],
        ridge=float(ridge),
        train_rmse=float(train_rmse),
        mode=str(mode),
        nyquist=None if nyquist is None else float(nyquist),
    )


def _coerce_dealias_result(native_result: Any, *, reference: np.ndarray | None) -> DealiasResult:
    if isinstance(native_result, DealiasResult):
        return native_result
    if isinstance(native_result, dict):
        velocity = native_result.get("velocity")
        folds = native_result.get("folds")
        confidence = native_result.get("confidence")
        ref_out = native_result.get("reference", reference)
        metadata = native_result.get("metadata", {})
    else:
        values = tuple(native_result)
        if len(values) == 5:
            velocity, folds, confidence, ref_out, metadata = values
        elif len(values) == 4:
            velocity, folds, confidence, metadata = values
            ref_out = reference
        else:  # pragma: no cover - defensive against future API drift.
            raise ValueError("native ML backend returned an unexpected result shape")
    return DealiasResult(
        velocity=np.asarray(velocity, dtype=float),
        folds=np.asarray(folds, dtype=np.int16),
        confidence=np.asarray(confidence, dtype=float),
        reference=None if ref_out is None else np.asarray(ref_out, dtype=float),
        metadata=dict(metadata),
    )


def fit_ml_reference_model(
    observed: np.ndarray,
    target_velocity: np.ndarray,
    *,
    nyquist: float | None = None,
    reference: np.ndarray | None = None,
    azimuth_deg: np.ndarray | None = None,
    ridge: float = 1.0,
) -> LinearBranchModel:
    """Fit a lightweight ridge-regression model that predicts unfolded velocity."""
    obs = as_float_array(observed)
    target = np.asarray(target_velocity, dtype=float)
    if obs.ndim != 2 or target.shape != obs.shape:
        raise ValueError("observed and target_velocity must be 2D with the same shape")

    native_method = _native_method("fit_ml_reference_model")
    if native_method is not None:
        native_result = native_method(
            obs,
            target,
            nyquist=None if nyquist is None else float(nyquist),
            reference=None if reference is None else np.asarray(reference, dtype=float),
            azimuth_deg=None if azimuth_deg is None else np.asarray(azimuth_deg, dtype=float),
            ridge=float(ridge),
        )
        return _coerce_linear_branch_model(native_result)

    X, names = _build_feature_stack(obs, reference=reference, azimuth_deg=azimuth_deg)
    mask = np.isfinite(target) & np.isfinite(obs)
    X_fit = X[mask]
    target_fit = target[mask]
    if X_fit.shape[0] < X_fit.shape[1]:
        raise ValueError("not enough finite training gates to fit ML reference model")

    reg = float(max(ridge, 1e-8))
    xtx = X_fit.T @ X_fit
    xtx.flat[:: xtx.shape[0] + 1] += reg
    if nyquist is not None:
        fold_target = np.rint((target_fit - obs[mask]) / (2.0 * float(nyquist)))
        weights = np.linalg.solve(xtx, X_fit.T @ fold_target)
        pred_fold = np.rint(X_fit @ weights)
        pred_velocity = obs[mask] + 2.0 * float(nyquist) * pred_fold
        rmse = float(np.sqrt(np.mean((target_fit - pred_velocity) ** 2))) if target_fit.size else 0.0
        mode = "fold"
    else:
        weights = np.linalg.solve(xtx, X_fit.T @ target_fit)
        pred_velocity = X_fit @ weights
        rmse = float(np.sqrt(np.mean((target_fit - pred_velocity) ** 2))) if target_fit.size else 0.0
        mode = "velocity"
    return LinearBranchModel(weights=weights, feature_names=names, ridge=reg, train_rmse=rmse, mode=mode, nyquist=nyquist)


def _predict_reference(
    observed: np.ndarray,
    model: LinearBranchModel,
    *,
    reference: np.ndarray | None = None,
    azimuth_deg: np.ndarray | None = None,
) -> np.ndarray:
    X, names = _build_feature_stack(observed, reference=reference, azimuth_deg=azimuth_deg)
    if names != model.feature_names:
        raise ValueError("feature set does not match the supplied model")
    pred = X @ model.weights
    if model.mode == "fold":
        if model.nyquist is None:
            raise ValueError("fold model is missing nyquist")
        pred = np.asarray(observed, dtype=float) + 2.0 * float(model.nyquist) * np.rint(pred)
    return np.asarray(pred, dtype=float)


def dealias_sweep_ml(
    observed: np.ndarray,
    nyquist: float,
    *,
    model: LinearBranchModel | None = None,
    training_target: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    azimuth_deg: np.ndarray | None = None,
    ridge: float = 1.0,
    refine_with_variational: bool = True,
) -> DealiasResult:
    """ML-assisted branch selection using a fitted linear reference predictor.

    If no model is supplied:
    - use `training_target` when available,
    - otherwise fit against a pseudo-target generated from the external reference
      or a variational bootstrap.
    """
    obs = as_float_array(observed)
    if obs.ndim != 2:
        raise ValueError("observed must be 2D [azimuth, range]")
    if nyquist <= 0:
        raise ValueError("nyquist must be positive")

    ref = None if reference is None else np.asarray(reference, dtype=float)
    if ref is not None and ref.shape != obs.shape:
        raise ValueError("reference must match observed shape")

    target = None if training_target is None else np.asarray(training_target, dtype=float)
    if target is not None and target.shape != obs.shape:
        raise ValueError("training_target must match observed shape")

    native_method = _native_method("dealias_sweep_ml")
    if native_method is not None:
        native_result = native_method(
            obs,
            float(nyquist),
            model=model,
            training_target=target,
            reference=ref,
            azimuth_deg=None if azimuth_deg is None else np.asarray(azimuth_deg, dtype=float),
            ridge=float(ridge),
            refine_with_variational=bool(refine_with_variational),
        )
        return _coerce_dealias_result(native_result, reference=ref)

    trained_from = "supplied_model"
    if model is None:
        if target is None:
            if ref is not None:
                target = unfold_to_reference(obs, ref, nyquist)
                trained_from = "reference_pseudo_target"
            else:
                bootstrap = dealias_sweep_variational(obs, nyquist, reference=None)
                target = bootstrap.velocity
                trained_from = "variational_pseudo_target"
        else:
            trained_from = "training_target"
        model = fit_ml_reference_model(obs, target, nyquist=nyquist, reference=ref, azimuth_deg=azimuth_deg, ridge=ridge)

    predicted_ref = _predict_reference(obs, model, reference=ref, azimuth_deg=azimuth_deg)
    corrected = unfold_to_reference(obs, predicted_ref, nyquist)
    corrected = np.where(np.isfinite(obs), corrected, np.nan)
    confidence = np.where(
        np.isfinite(corrected),
        gaussian_confidence(np.abs(corrected - predicted_ref), 0.50 * nyquist),
        0.0,
    )

    metadata = {
        "paper_family": "MLAssistLite",
        "method": "ridge_reference_predictor",
        "trained_from": trained_from,
        "train_rmse": float(model.train_rmse),
        "ridge": float(model.ridge),
        "feature_names": list(model.feature_names),
    }

    if refine_with_variational:
        refined = dealias_sweep_variational(obs, nyquist, reference=predicted_ref)
        corrected = refined.velocity
        confidence = np.maximum(confidence, refined.confidence)
        metadata["refine_method"] = refined.metadata.get("method")
        metadata["refine_iterations"] = refined.metadata.get("iterations_used")

    folds = fold_counts(corrected, obs, nyquist)
    return DealiasResult(
        velocity=corrected,
        folds=folds,
        confidence=confidence,
        reference=predicted_ref,
        metadata=metadata,
    )
