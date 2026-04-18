from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyart

from open_dealias import (
    apply_velocity_qc,
    dealias_sweep_es90,
    dealias_sweep_jh01,
    dealias_sweep_ml,
    dealias_sweep_recursive,
    dealias_sweep_region_graph,
    dealias_sweep_variational,
    dealias_sweep_xu11,
    dealias_sweep_zw06,
    estimate_uniform_wind_vad,
    fit_ml_reference_model,
    load_pal_table,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a one-frame collage comparing open dealiasing algorithms.")
    parser.add_argument("archive_path", help="Path to the selected archive file.")
    parser.add_argument("--previous-path", default=None, help="Optional previous archive file. Defaults to the previous file in the same folder.")
    parser.add_argument("--sweep", type=int, default=1, help="Sweep index to render.")
    parser.add_argument("--velocity-palette", default="color_tables/awips_bv.pal", help="Path to the velocity palette.")
    parser.add_argument("--zoom-span-km", type=float, default=18.0, help="Half-width of the vrot-centered crop.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to examples/output/<archive>_all_algorithms.png",
    )
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
        raise ValueError("unable to infer nyquist from empty velocity field")
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


def _infer_previous_path(current_path: Path) -> Path:
    siblings = sorted(path for path in current_path.parent.iterdir() if path.is_file())
    try:
        idx = siblings.index(current_path)
    except ValueError as exc:
        raise FileNotFoundError(f"{current_path} not found among siblings") from exc
    if idx == 0:
        raise FileNotFoundError(f"no previous file found before {current_path.name}")
    return siblings[idx - 1]


def _auto_zoom_center(x_km: np.ndarray, y_km: np.ndarray, velocity: np.ndarray, reflectivity: np.ndarray) -> tuple[float, float]:
    rng = np.hypot(x_km, y_km)
    mask = np.isfinite(velocity) & np.isfinite(reflectivity) & (reflectivity > 25.0) & (rng > 5.0) & (rng < 90.0)
    pos_idx = np.argwhere(mask & (velocity > 8.0))
    neg_idx = np.argwhere(mask & (velocity < -8.0))
    if len(pos_idx) < 4 or len(neg_idx) < 4:
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
        score = (pv - nv) - 2.5 * dist
        j = int(np.argmax(score))
        if float(score[j]) > best_score:
            ni = neg_keep[j]
            best_score = float(score[j])
            best_center = (
                float((px + x_km[ni[0], ni[1]]) / 2.0),
                float((py + y_km[ni[0], ni[1]]) / 2.0),
            )
    return best_center


def _metric_mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _metric_changed(candidate: np.ndarray, observed: np.ndarray) -> int:
    mask = np.isfinite(candidate) & np.isfinite(observed)
    if not np.any(mask):
        return 0
    return int(np.count_nonzero(np.abs(candidate[mask] - observed[mask]) > 1e-6))


def main() -> None:
    args = _parse_args()
    archive_path = Path(args.archive_path)
    previous_path = Path(args.previous_path) if args.previous_path else _infer_previous_path(archive_path)
    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "examples" / "output" / f"{archive_path.stem}_all_algorithms.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    radar = pyart.io.read_nexrad_archive(str(archive_path))
    previous_radar = pyart.io.read_nexrad_archive(str(previous_path))
    _patch_zero_nyquist_volume(radar)
    _patch_zero_nyquist_volume(previous_radar)

    sweep = int(args.sweep)
    sl = radar.get_slice(sweep)
    prev_sl = previous_radar.get_slice(sweep)

    reflectivity = _filled(radar.get_field(sweep, "reflectivity"))
    velocity = _filled(radar.get_field(sweep, "velocity"))
    previous_velocity = _filled(previous_radar.get_field(sweep, "velocity"))
    x_km, y_km, _ = radar.get_gate_x_y_z(sweep)
    x_km = x_km / 1000.0
    y_km = y_km / 1000.0
    azimuth_deg = np.asarray(radar.get_azimuth(sweep), dtype=float)
    previous_azimuth_deg = np.asarray(previous_radar.get_azimuth(sweep), dtype=float)
    elevation_deg = float(radar.fixed_angle["data"][sweep])
    nyquist = _resolve_nyquist(radar, sweep, velocity)
    previous_nyquist = _resolve_nyquist(previous_radar, sweep, previous_velocity)

    qc_velocity = apply_velocity_qc(
        velocity,
        reflectivity=reflectivity,
        min_reflectivity=-5.0,
        max_texture=18.0,
        min_gate_fraction_in_ray=0.01,
    )
    previous_qc = apply_velocity_qc(
        previous_velocity,
        reflectivity=_filled(previous_radar.get_field(sweep, "reflectivity")),
        min_reflectivity=-5.0,
        max_texture=18.0,
        min_gate_fraction_in_ray=0.01,
    )

    current_pyart = _filled(pyart.correct.dealias_region_based(radar, vel_field="velocity", keep_original=False)["data"][sl])
    previous_pyart = _filled(pyart.correct.dealias_region_based(previous_radar, vel_field="velocity", keep_original=False)["data"][prev_sl])

    vad_fit = estimate_uniform_wind_vad(qc_velocity, nyquist, azimuth_deg, elevation_deg=elevation_deg)
    ml_model = fit_ml_reference_model(
        previous_qc,
        previous_pyart,
        nyquist=previous_nyquist,
        azimuth_deg=previous_azimuth_deg,
    )

    results: list[tuple[str, np.ndarray]] = [
        ("Reflectivity", reflectivity),
        ("Raw Velocity", velocity),
        ("QC Velocity", qc_velocity),
        ("Py-ART Region", current_pyart),
        ("ES90", dealias_sweep_es90(qc_velocity, nyquist, reference=vad_fit.reference, max_gap=4, max_abs_step=12.0).velocity),
        ("ZW06", dealias_sweep_zw06(qc_velocity, nyquist, reference=vad_fit.reference).velocity),
        ("XU11", dealias_sweep_xu11(qc_velocity, nyquist, azimuth_deg, elevation_deg=elevation_deg).velocity),
        ("JH01", dealias_sweep_jh01(qc_velocity, nyquist, previous_corrected=previous_pyart, background_reference=vad_fit.reference).velocity),
        ("Region Graph", dealias_sweep_region_graph(qc_velocity, nyquist, reference=vad_fit.reference).velocity),
        ("Recursive", dealias_sweep_recursive(qc_velocity, nyquist, reference=vad_fit.reference).velocity),
        ("Variational", dealias_sweep_variational(qc_velocity, nyquist, reference=vad_fit.reference, max_iterations=4).velocity),
        ("ML", dealias_sweep_ml(qc_velocity, nyquist, model=ml_model, reference=vad_fit.reference, azimuth_deg=azimuth_deg).velocity),
    ]

    center_x, center_y = _auto_zoom_center(x_km, y_km, velocity, reflectivity)
    span = float(args.zoom_span_km)
    palette = load_pal_table(args.velocity_palette)
    velocity_cmap, vel_vmin, vel_vmax = palette.matplotlib_colormap(name=f"{palette.name}_internal")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
    axes = axes.ravel()
    for ax, (name, field) in zip(axes, results):
        if name == "Reflectivity":
            mesh = ax.pcolormesh(x_km, y_km, field, shading="auto", cmap="NWSRef", vmin=-10.0, vmax=75.0)
            title = name
        else:
            mesh = ax.pcolormesh(x_km, y_km, field, shading="auto", cmap=velocity_cmap, vmin=vel_vmin, vmax=vel_vmax)
            changed = _metric_changed(field, qc_velocity)
            mae = _metric_mae(field, current_pyart) if name != "Py-ART Region" else 0.0
            title = f"{name}\nchg={changed}  mae={mae:.2f}"
        ax.set_xlim(center_x - span, center_x + span)
        ax.set_ylim(center_y - span, center_y + span)
        ax.plot(center_x, center_y, marker="x", color="white", markersize=7, markeredgewidth=1.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("km east/west")
        ax.set_ylabel("km north/south")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, shrink=0.8)

    fig.suptitle(
        f"{archive_path.name} sweep {sweep}  previous={previous_path.name}  palette={palette.name}",
        fontsize=13,
    )
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
