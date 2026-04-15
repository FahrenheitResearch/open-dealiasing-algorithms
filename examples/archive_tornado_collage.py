from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyart

from open_dealias import (
    apply_velocity_qc,
    dealias_sweep_recursive,
    dealias_sweep_region_graph,
    dealias_sweep_variational,
    estimate_uniform_wind_vad,
    load_pal_table,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a dense tornado archive collage with open dealiasing panels.")
    parser.add_argument("archive_path", help="Path to a local Level II or MSG31 archive file.")
    parser.add_argument("--sweep", type=int, default=0, help="Sweep index to render.")
    parser.add_argument(
        "--velocity-palette",
        default="ALPHA-Velo.pal",
        help="Path to a .pal file used for velocity-like panels.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to examples/output/<archive>_s<sweep>_collage.png",
    )
    parser.add_argument("--zoom-span-km", type=float, default=12.0, help="Half-width of the zoom box in km.")
    return parser.parse_args()


def _filled(field) -> np.ndarray:
    if hasattr(field, "filled"):
        field = field.filled(np.nan)
    return np.asarray(field, dtype=float)


def _resolve_nyquist(radar, sweep: int, velocity: np.ndarray) -> float:
    try:
        nyquist = float(np.asarray(radar.get_nyquist_vel(sweep)).reshape(-1)[0])
    except Exception:
        nyquist = float("nan")
    if np.isfinite(nyquist) and nyquist > 0.0:
        return nyquist

    finite = np.abs(velocity[np.isfinite(velocity)])
    if not finite.size:
        raise ValueError("unable to infer nyquist velocity from an empty sweep")
    inferred = float(np.ceil(np.nanmax(finite) * 2.0) / 2.0)
    inferred = max(inferred, 1.0)
    if radar.instrument_parameters and "nyquist_velocity" in radar.instrument_parameters:
        try:
            data = radar.instrument_parameters["nyquist_velocity"]["data"]
            sl = radar.get_slice(sweep)
            data[sl] = inferred
        except Exception:
            pass
    return inferred


def _patch_zero_nyquist_volume(radar) -> None:
    if not radar.instrument_parameters or "nyquist_velocity" not in radar.instrument_parameters:
        return
    data = radar.instrument_parameters["nyquist_velocity"]["data"]
    for sweep in range(radar.nsweeps):
        try:
            current = float(np.asarray(radar.get_nyquist_vel(sweep)).reshape(-1)[0])
        except Exception:
            current = float("nan")
        if np.isfinite(current) and current > 0.0:
            continue
        try:
            velocity = _filled(radar.get_field(sweep, "velocity"))
        except Exception:
            continue
        finite = np.abs(velocity[np.isfinite(velocity)])
        if not finite.size:
            continue
        inferred = float(np.ceil(np.nanmax(finite) * 2.0) / 2.0)
        inferred = max(inferred, 1.0)
        sl = radar.get_slice(sweep)
        data[sl] = inferred


def _auto_zoom_center(x_km: np.ndarray, y_km: np.ndarray, velocity: np.ndarray, reflectivity: np.ndarray) -> tuple[float, float]:
    rng = np.hypot(x_km, y_km)
    mask = np.isfinite(velocity) & np.isfinite(reflectivity) & (reflectivity > 25.0) & (rng > 5.0) & (rng < 80.0)
    pos_idx = np.argwhere(mask & (velocity > 10.0))
    neg_idx = np.argwhere(mask & (velocity < -10.0))
    if not len(pos_idx) or not len(neg_idx):
        iy, ix = np.unravel_index(np.nanargmax(np.abs(np.where(np.isfinite(velocity), velocity, np.nan))), velocity.shape)
        return float(x_km[iy, ix]), float(y_km[iy, ix])

    pos_vals = velocity[pos_idx[:, 0], pos_idx[:, 1]]
    neg_vals = velocity[neg_idx[:, 0], neg_idx[:, 1]]
    pos_keep = pos_idx[np.argsort(pos_vals)[-120:]]
    neg_keep = neg_idx[np.argsort(neg_vals)[:120]]

    best_score = -1.0e9
    best_center = (float(x_km[0, 0]), float(y_km[0, 0]))
    for pi in pos_keep:
        px = x_km[pi[0], pi[1]]
        py = y_km[pi[0], pi[1]]
        pv = velocity[pi[0], pi[1]]
        dx = x_km[neg_keep[:, 0], neg_keep[:, 1]] - px
        dy = y_km[neg_keep[:, 0], neg_keep[:, 1]] - py
        dist = np.hypot(dx, dy)
        nv = velocity[neg_keep[:, 0], neg_keep[:, 1]]
        score = (pv - nv) - 2.0 * dist
        j = int(np.argmax(score))
        if float(score[j]) > best_score:
            ni = neg_keep[j]
            best_score = float(score[j])
            best_center = (
                float((px + x_km[ni[0], ni[1]]) / 2.0),
                float((py + y_km[ni[0], ni[1]]) / 2.0),
            )
    return best_center


def _safe_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = np.isfinite(a) & np.isfinite(b)
    out = np.full_like(a, np.nan, dtype=float)
    out[mask] = np.abs(a[mask] - b[mask])
    return out


def _storm_window(x_km: np.ndarray, y_km: np.ndarray, reflectivity: np.ndarray) -> tuple[float, float, float, float] | None:
    mask = np.isfinite(reflectivity) & (reflectivity > 15.0)
    if not np.any(mask):
        return None
    xs = x_km[mask]
    ys = y_km[mask]
    pad = 12.0
    return (
        float(np.nanmin(xs) - pad),
        float(np.nanmax(xs) + pad),
        float(np.nanmin(ys) - pad),
        float(np.nanmax(ys) + pad),
    )


def main() -> None:
    args = _parse_args()
    archive_path = Path(args.archive_path)
    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "examples" / "output" / f"{archive_path.stem}_s{args.sweep}_collage.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    radar = pyart.io.read_nexrad_archive(str(archive_path))
    _patch_zero_nyquist_volume(radar)
    sweep = int(args.sweep)
    sl = radar.get_slice(sweep)

    reflectivity = _filled(radar.get_field(sweep, "reflectivity"))
    velocity = _filled(radar.get_field(sweep, "velocity"))
    x_km, y_km, _ = radar.get_gate_x_y_z(sweep)
    x_km = x_km / 1000.0
    y_km = y_km / 1000.0
    azimuth_deg = np.asarray(radar.get_azimuth(sweep), dtype=float)
    elevation_deg = float(radar.fixed_angle["data"][sweep])
    nyquist = _resolve_nyquist(radar, sweep, velocity)

    qc_velocity = apply_velocity_qc(
        velocity,
        reflectivity=reflectivity,
        min_reflectivity=-5.0,
        max_texture=18.0,
        min_gate_fraction_in_ray=0.01,
    )
    vad_fit = estimate_uniform_wind_vad(qc_velocity, nyquist, azimuth_deg, elevation_deg=elevation_deg)
    region = dealias_sweep_region_graph(qc_velocity, nyquist, reference=vad_fit.reference)
    recursive = dealias_sweep_recursive(qc_velocity, nyquist, reference=vad_fit.reference)
    variational = dealias_sweep_variational(qc_velocity, nyquist, reference=vad_fit.reference, max_iterations=4)
    pyart_region = _filled(pyart.correct.dealias_region_based(radar, vel_field="velocity", keep_original=False)["data"][sl])

    center_x, center_y = _auto_zoom_center(x_km, y_km, velocity, reflectivity)
    storm_window = _storm_window(x_km, y_km, reflectivity)
    zoom_span = float(args.zoom_span_km)

    palette = load_pal_table(args.velocity_palette)
    velocity_cmap, vel_vmin, vel_vmax = palette.matplotlib_colormap(name=f"{palette.name}_internal")
    diff_max = max(
        5.0,
        float(np.nanpercentile(np.concatenate([
            _safe_absdiff(region.velocity, pyart_region)[np.isfinite(_safe_absdiff(region.velocity, pyart_region))],
            _safe_absdiff(variational.velocity, pyart_region)[np.isfinite(_safe_absdiff(variational.velocity, pyart_region))],
        ]), 99.0))
        if (
            np.any(np.isfinite(_safe_absdiff(region.velocity, pyart_region)))
            or np.any(np.isfinite(_safe_absdiff(variational.velocity, pyart_region)))
        )
        else 5.0
    )

    changed_mask = np.where(
        np.isfinite(region.velocity) & np.isfinite(qc_velocity) & (np.abs(region.velocity - qc_velocity) > 1e-6),
        1.0,
        0.0,
    )
    changed_mask[~np.isfinite(qc_velocity)] = np.nan

    panels = [
        ("Reflectivity", reflectivity, "NWSRef", -10.0, 75.0),
        ("Raw Velocity", velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("QC Velocity", qc_velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("Py-ART Region", pyart_region, velocity_cmap, vel_vmin, vel_vmax),
        ("Open Region Graph", region.velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("Open Recursive", recursive.velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("Open Variational", variational.velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("|Region - Py-ART|", _safe_absdiff(region.velocity, pyart_region), "inferno", 0.0, diff_max),
        ("|Variational - Py-ART|", _safe_absdiff(variational.velocity, pyart_region), "inferno", 0.0, diff_max),
        ("Region Changed Gates", changed_mask, ListedColormap(["#222222", "#ffea00"]), 0.0, 1.0),
        ("Reflectivity Zoom", reflectivity, "NWSRef", -10.0, 75.0),
        ("Raw Velocity Zoom", velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("Py-ART Region Zoom", pyart_region, velocity_cmap, vel_vmin, vel_vmax),
        ("Open Region Graph Zoom", region.velocity, velocity_cmap, vel_vmin, vel_vmax),
        ("Open Variational Zoom", variational.velocity, velocity_cmap, vel_vmin, vel_vmax),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(24, 15), constrained_layout=True)
    for idx, (title, field, cmap, vmin, vmax) in enumerate(panels):
        ax = axes.flat[idx]
        mesh = ax.pcolormesh(x_km, y_km, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        if idx < 10 and storm_window is not None:
            ax.set_xlim(storm_window[0], storm_window[1])
            ax.set_ylim(storm_window[2], storm_window[3])
            ax.plot(center_x, center_y, marker="x", color="white", markersize=8, markeredgewidth=1.5)
        if idx >= 10:
            ax.set_xlim(center_x - zoom_span, center_x + zoom_span)
            ax.set_ylim(center_y - zoom_span, center_y + zoom_span)
            ax.plot(center_x, center_y, marker="x", color="white", markersize=8, markeredgewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("km east/west")
        ax.set_ylabel("km north/south")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.8)

    fig.suptitle(
        f"{archive_path.name} sweep {sweep}  elev={elevation_deg:.2f}  nyquist={nyquist:.2f} m/s  "
        f"velocity palette={palette.name} ({palette.units}, scale={palette.scale:g})"
    )
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
