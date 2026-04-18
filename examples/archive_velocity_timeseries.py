from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyart

from open_dealias import (
    apply_velocity_qc,
    dealias_sweep_region_graph,
    estimate_uniform_wind_vad,
    load_pal_table,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a fast dealiased velocity time-series sheet from an archived event zip.")
    parser.add_argument(
        "--zip-path",
        default="archive_data/May 6th, 2024 Barnsdall-Bartlesville EF4 (TTUL).zip",
        help="Archive zip containing Level II or similar radar files.",
    )
    parser.add_argument("--sweep", type=int, default=2, help="Sweep index used for every frame in the time series.")
    parser.add_argument("--frames", type=int, default=12, help="Number of consecutive frames to render around the peak frame.")
    parser.add_argument(
        "--velocity-palette",
        default="color_tables/awips_bv.pal",
        help="Path to a .pal file for velocity display.",
    )
    parser.add_argument(
        "--output-prefix",
        default="ttul_timeseries_region_graph",
        help="Prefix for PNG and JSON outputs under examples/output.",
    )
    parser.add_argument("--zoom-pad-km", type=float, default=10.0, help="Padding around the union storm box.")
    parser.add_argument(
        "--selection-mode",
        choices=["sequence", "top"],
        default="sequence",
        help="Pick consecutive frames around the peak or the top-scoring frames across the event.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=["storm", "vrot"],
        default="vrot",
        help="Crop around the union storm envelope or center each frame on the vrot marker.",
    )
    parser.add_argument(
        "--vrot-span-km",
        type=float,
        default=18.0,
        help="Half-width of the vrot-centered crop when --crop-mode vrot is used.",
    )
    return parser.parse_args()


def _filled(field) -> np.ndarray:
    if hasattr(field, "filled"):
        field = field.filled(np.nan)
    return np.asarray(field, dtype=float)


def _extract_zip_members(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        for info in infos:
            out_path = out_dir / info.filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists() or out_path.stat().st_size != info.file_size:
                zf.extract(info, out_dir)
            extracted.append(out_path)
    return sorted(extracted)


def _resolve_nyquist(radar, sweep: int, velocity: np.ndarray) -> float:
    try:
        nyquist = float(np.asarray(radar.get_nyquist_vel(sweep)).reshape(-1)[0])
    except Exception:
        nyquist = float("nan")
    if np.isfinite(nyquist) and nyquist > 0.0:
        return nyquist

    finite = np.abs(velocity[np.isfinite(velocity)])
    if not finite.size:
        raise ValueError("unable to infer nyquist from an empty velocity field")
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


def _frame_score(x_km: np.ndarray, y_km: np.ndarray, velocity: np.ndarray, reflectivity: np.ndarray) -> dict[str, float] | None:
    rng = np.hypot(x_km, y_km)
    mask = np.isfinite(velocity) & np.isfinite(reflectivity) & (reflectivity > 25.0) & (rng > 5.0) & (rng < 90.0)
    pos_idx = np.argwhere(mask & (velocity > 8.0))
    neg_idx = np.argwhere(mask & (velocity < -8.0))
    if len(pos_idx) < 4 or len(neg_idx) < 4:
        return None

    pos_vals = velocity[pos_idx[:, 0], pos_idx[:, 1]]
    neg_vals = velocity[neg_idx[:, 0], neg_idx[:, 1]]
    pos_keep = pos_idx[np.argsort(pos_vals)[-80:]]
    neg_keep = neg_idx[np.argsort(neg_vals)[:80]]

    best_score = -1.0e9
    best: dict[str, float] | None = None
    for pi in pos_keep:
        px = x_km[pi[0], pi[1]]
        py = y_km[pi[0], pi[1]]
        pv = velocity[pi[0], pi[1]]
        dx = x_km[neg_keep[:, 0], neg_keep[:, 1]] - px
        dy = y_km[neg_keep[:, 0], neg_keep[:, 1]] - py
        dist = np.hypot(dx, dy)
        nv = velocity[neg_keep[:, 0], neg_keep[:, 1]]
        score = (pv - nv) - 2.5 * dist
        j = int(np.argmax(score))
        if float(score[j]) > best_score:
            ni = neg_keep[j]
            best_score = float(score[j])
            best = {
                "score": best_score,
                "delta_v": float(pv - nv[j]),
                "dist_km": float(dist[j]),
                "vrot_proxy": float(0.5 * (pv - nv[j])),
                "center_x_km": float((px + x_km[ni[0], ni[1]]) / 2.0),
                "center_y_km": float((py + y_km[ni[0], ni[1]]) / 2.0),
            }
    return best


def _storm_window(x_km: np.ndarray, y_km: np.ndarray, reflectivity: np.ndarray) -> tuple[float, float, float, float] | None:
    mask = np.isfinite(reflectivity) & (reflectivity > 15.0)
    if not np.any(mask):
        return None
    return (
        float(np.nanmin(x_km[mask])),
        float(np.nanmax(x_km[mask])),
        float(np.nanmin(y_km[mask])),
        float(np.nanmax(y_km[mask])),
    )


def main() -> None:
    args = _parse_args()
    zip_path = Path(args.zip_path)
    extract_dir = REPO_ROOT / ".cache" / "archive_timeseries" / zip_path.stem
    files = _extract_zip_members(zip_path, extract_dir)

    summaries: list[dict[str, object]] = []
    for path in files:
        radar = pyart.io.read_nexrad_archive(str(path))
        _patch_zero_nyquist_volume(radar)
        if args.sweep >= radar.nsweeps:
            continue
        velocity = _filled(radar.get_field(args.sweep, "velocity"))
        reflectivity = _filled(radar.get_field(args.sweep, "reflectivity"))
        x_km, y_km, _ = radar.get_gate_x_y_z(args.sweep)
        x_km = x_km / 1000.0
        y_km = y_km / 1000.0
        nyquist = _resolve_nyquist(radar, args.sweep, velocity)
        summary = _frame_score(x_km, y_km, velocity, reflectivity)
        if summary is None:
            continue
        summary.update(
            {
                "path": str(path),
                "name": path.name,
                "elevation_deg": float(radar.fixed_angle["data"][args.sweep]),
                "nyquist_mps": nyquist,
            }
        )
        summaries.append(summary)

    if not summaries:
        raise RuntimeError("no usable frames were found in the archive")

    summaries.sort(key=lambda item: item["name"])
    scores = np.asarray([float(item["score"]) for item in summaries], dtype=float)
    peak_index = int(np.nanargmax(scores))
    if args.selection_mode == "top":
        selected = sorted(summaries, key=lambda item: float(item["score"]), reverse=True)[: int(args.frames)]
        selected = sorted(selected, key=lambda item: item["name"])
    else:
        half = max(1, int(args.frames) // 2)
        start = max(0, peak_index - half)
        stop = min(len(summaries), start + int(args.frames))
        start = max(0, stop - int(args.frames))
        selected = summaries[start:stop]

    palette = load_pal_table(args.velocity_palette)
    velocity_cmap, vel_vmin, vel_vmax = palette.matplotlib_colormap(name=f"{palette.name}_internal")

    rendered: list[dict[str, object]] = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []

    for item in selected:
        radar = pyart.io.read_nexrad_archive(item["path"])
        _patch_zero_nyquist_volume(radar)
        sweep = int(args.sweep)
        velocity = _filled(radar.get_field(sweep, "velocity"))
        reflectivity = _filled(radar.get_field(sweep, "reflectivity"))
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
        window = _storm_window(x_km, y_km, reflectivity)
        if window is not None:
            xmins.append(window[0])
            xmaxs.append(window[1])
            ymins.append(window[2])
            ymaxs.append(window[3])

        rendered.append(
            {
                **item,
                "x_km": x_km,
                "y_km": y_km,
                "velocity": region.velocity,
                "reflectivity": reflectivity,
            }
        )

    if xmins:
        xmin = min(xmins) - float(args.zoom_pad_km)
        xmax = max(xmaxs) + float(args.zoom_pad_km)
        ymin = min(ymins) - float(args.zoom_pad_km)
        ymax = max(ymaxs) + float(args.zoom_pad_km)
    else:
        xmin, xmax, ymin, ymax = -80.0, 80.0, -80.0, 80.0

    frame_cols = 2
    nrows = int(ceil(len(rendered) / frame_cols))
    ncols = frame_cols * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.8, nrows * 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, item in enumerate(rendered):
        refl_ax = axes[idx * 2]
        vel_ax = axes[idx * 2 + 1]

        if args.crop_mode == "storm":
            x0, x1, y0, y1 = xmin, xmax, ymin, ymax
        else:
            span = float(args.vrot_span_km)
            x0 = float(item["center_x_km"]) - span
            x1 = float(item["center_x_km"]) + span
            y0 = float(item["center_y_km"]) - span
            y1 = float(item["center_y_km"]) + span

        refl_mesh = refl_ax.pcolormesh(
            item["x_km"],
            item["y_km"],
            item["reflectivity"],
            shading="auto",
            cmap="NWSRef",
            vmin=-10.0,
            vmax=75.0,
        )
        vel_mesh = vel_ax.pcolormesh(
            item["x_km"],
            item["y_km"],
            item["velocity"],
            shading="auto",
            cmap=velocity_cmap,
            vmin=vel_vmin,
            vmax=vel_vmax,
        )
        for ax in (refl_ax, vel_ax):
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.plot(float(item["center_x_km"]), float(item["center_y_km"]), marker="x", color="white", markersize=7, markeredgewidth=1.4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("km east/west")
            ax.set_ylabel("km north/south")

        refl_ax.set_title(
            f"{item['name']} refl\nscore={item['score']:.1f}  vrot~{item['vrot_proxy']:.1f}  d={item['dist_km']:.2f} km",
            fontsize=9,
        )
        vel_ax.set_title(f"{item['name']} vel", fontsize=9)
        fig.colorbar(refl_mesh, ax=refl_ax, shrink=0.8)
        fig.colorbar(vel_mesh, ax=vel_ax, shrink=0.8)

    for ax in axes[len(rendered) * 2:]:
        ax.axis("off")

    fig.suptitle(
        f"{zip_path.stem}  sweep {args.sweep}  fast dealias: region_graph  palette={palette.name}  "
        f"mode={args.selection_mode}/{args.crop_mode}",
        fontsize=13,
    )

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{args.output_prefix}.png"
    json_path = out_dir / f"{args.output_prefix}.json"
    fig.savefig(png_path, dpi=170)

    payload = {
        "zip_path": str(zip_path),
        "sweep": int(args.sweep),
        "frames_requested": int(args.frames),
        "palette": palette.name,
        "selected_frames": [
            {
                key: value
                for key, value in item.items()
                if key not in {"x_km", "y_km", "velocity", "reflectivity", "path"}
            }
            for item in rendered
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(png_path)
    print(json_path)


if __name__ == "__main__":
    main()
