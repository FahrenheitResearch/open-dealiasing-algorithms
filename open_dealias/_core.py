from __future__ import annotations

from typing import Iterable
import warnings

import numpy as np


ArrayLike = np.ndarray | Iterable[float]


def as_float_array(data: ArrayLike | np.ndarray, *, copy: bool = True) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr.copy() if copy else arr


def wrap_to_nyquist(velocity: np.ndarray | float, nyquist: float) -> np.ndarray:
    """Wrap velocity into [-nyquist, nyquist)."""
    if nyquist <= 0:
        raise ValueError('nyquist must be positive')
    vel = np.asarray(velocity, dtype=float)
    wrapped = ((vel + nyquist) % (2.0 * nyquist)) - nyquist
    wrapped = np.where(np.isfinite(vel), wrapped, np.nan)
    return wrapped


def fold_counts(unfolded: np.ndarray, observed: np.ndarray, nyquist: float) -> np.ndarray:
    if nyquist <= 0:
        raise ValueError('nyquist must be positive')
    counts = np.rint((np.asarray(unfolded, dtype=float) - np.asarray(observed, dtype=float)) / (2.0 * nyquist))
    counts = np.where(np.isfinite(counts), counts, 0)
    return counts.astype(np.int16)


def unfold_to_reference(
    observed: np.ndarray | float,
    reference: np.ndarray | float,
    nyquist: float,
    *,
    max_abs_fold: int = 32,
) -> np.ndarray:
    """Choose the unfolded candidate closest to a reference field."""
    if nyquist <= 0:
        raise ValueError('nyquist must be positive')
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(reference, dtype=float)
    folds = np.rint((ref - obs) / (2.0 * nyquist))
    folds = np.clip(folds, -max_abs_fold, max_abs_fold)
    out = obs + 2.0 * nyquist * folds
    out = np.where(np.isfinite(obs) & np.isfinite(ref), out, np.nan)
    return out


def combine_references(*refs: np.ndarray | None) -> np.ndarray | None:
    valid = [np.asarray(r, dtype=float) for r in refs if r is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0].copy()
    stack = np.stack(valid, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        with np.errstate(all='ignore'):
            return np.nanmedian(stack, axis=0)


def gaussian_confidence(mismatch: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), 1e-6)
    mismatch = np.asarray(mismatch, dtype=float)
    out = np.exp(-0.5 * (mismatch / scale) ** 2)
    out = np.where(np.isfinite(mismatch), out, 0.0)
    return out


def shift2d(field: np.ndarray, shift_az: int = 0, shift_range: int = 0, *, wrap_azimuth: bool = True) -> np.ndarray:
    """Integer shift for sweep-aligned references."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError('field must be 2D')

    out = arr.copy()
    if shift_az:
        out = np.roll(out, shift_az, axis=0)
        if not wrap_azimuth:
            if shift_az > 0:
                out[:shift_az, :] = np.nan
            else:
                out[shift_az:, :] = np.nan
    if shift_range:
        out = np.roll(out, shift_range, axis=1)
        if shift_range > 0:
            out[:, :shift_range] = np.nan
        else:
            out[:, shift_range:] = np.nan
    return out


def shift3d(volume: np.ndarray, shift_az: int = 0, shift_range: int = 0, *, wrap_azimuth: bool = True) -> np.ndarray:
    arr = np.asarray(volume, dtype=float)
    if arr.ndim != 3:
        raise ValueError('volume must be 3D')
    return np.stack([shift2d(s, shift_az=shift_az, shift_range=shift_range, wrap_azimuth=wrap_azimuth) for s in arr], axis=0)


def _shift_for_stack(arr: np.ndarray, da: int, dr: int, *, wrap_azimuth: bool = True) -> np.ndarray:
    shifted = shift2d(arr, shift_az=da, shift_range=dr, wrap_azimuth=wrap_azimuth)
    return shifted


def neighbor_stack(field: np.ndarray, *, include_diagonals: bool = True, wrap_azimuth: bool = True) -> np.ndarray:
    """Return stacked neighbors for a 2D sweep."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError('field must be 2D')
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if include_diagonals:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    return np.stack([_shift_for_stack(arr, da, dr, wrap_azimuth=wrap_azimuth) for da, dr in offsets], axis=0)


def texture_3x3(field: np.ndarray, *, wrap_azimuth: bool = True) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError('field must be 2D')
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    stack = np.stack([_shift_for_stack(arr, da, dr, wrap_azimuth=wrap_azimuth) for da, dr in offsets], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        with np.errstate(all='ignore'):
            return np.nanstd(stack, axis=0)
