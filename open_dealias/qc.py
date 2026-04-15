from __future__ import annotations

import numpy as np

from ._core import texture_3x3

__all__ = ["estimate_velocity_texture", "build_velocity_qc_mask", "apply_velocity_qc"]


def estimate_velocity_texture(
    velocity: np.ndarray,
    *,
    wrap_azimuth: bool = True,
) -> np.ndarray:
    """Estimate local 3x3 velocity texture for QC decisions."""
    arr = np.asarray(velocity, dtype=float)
    if arr.ndim != 2:
        raise ValueError("velocity must be 2D [azimuth, range]")
    return texture_3x3(arr, wrap_azimuth=wrap_azimuth)


def build_velocity_qc_mask(
    velocity: np.ndarray,
    *,
    reflectivity: np.ndarray | None = None,
    texture: np.ndarray | None = None,
    min_reflectivity: float | None = -5.0,
    max_texture: float | None = 12.0,
    min_gate_fraction_in_ray: float = 0.02,
    wrap_azimuth: bool = True,
) -> np.ndarray:
    """Build a conservative valid-gate mask for dealiasing.

    The mask removes non-finite gates, optionally filters weak reflectivity and
    high-texture gates, and blanks nearly empty rays that are unlikely to support
    stable continuity.
    """
    vel = np.asarray(velocity, dtype=float)
    if vel.ndim != 2:
        raise ValueError("velocity must be 2D [azimuth, range]")

    mask = np.isfinite(vel)
    if reflectivity is not None:
        refl = np.asarray(reflectivity, dtype=float)
        if refl.shape != vel.shape:
            raise ValueError("reflectivity must match velocity shape")
        if min_reflectivity is not None:
            mask &= np.isfinite(refl) & (refl >= float(min_reflectivity))

    if texture is None and max_texture is not None:
        texture = estimate_velocity_texture(vel, wrap_azimuth=wrap_azimuth)
    if texture is not None:
        tex = np.asarray(texture, dtype=float)
        if tex.shape != vel.shape:
            raise ValueError("texture must match velocity shape")
        if max_texture is not None:
            mask &= np.isfinite(tex) & (tex <= float(max_texture))

    ray_fraction = np.mean(mask, axis=1)
    sparse_rays = ray_fraction < float(min_gate_fraction_in_ray)
    if np.any(sparse_rays):
        mask[sparse_rays, :] = False

    return mask


def apply_velocity_qc(
    velocity: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    reflectivity: np.ndarray | None = None,
    texture: np.ndarray | None = None,
    min_reflectivity: float | None = -5.0,
    max_texture: float | None = 12.0,
    min_gate_fraction_in_ray: float = 0.02,
    wrap_azimuth: bool = True,
) -> np.ndarray:
    """Apply a conservative QC mask and return a NaN-masked velocity field."""
    vel = np.asarray(velocity, dtype=float)
    if mask is None:
        mask = build_velocity_qc_mask(
            vel,
            reflectivity=reflectivity,
            texture=texture,
            min_reflectivity=min_reflectivity,
            max_texture=max_texture,
            min_gate_fraction_in_ray=min_gate_fraction_in_ray,
            wrap_azimuth=wrap_azimuth,
        )
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != vel.shape:
            raise ValueError("mask must match velocity shape")

    return np.where(mask, vel, np.nan)
