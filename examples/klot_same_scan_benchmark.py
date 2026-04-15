from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyart

from examples.benchmark_support import (
    open_zw06_sweep_anchor,
    open_zw06_volume_anchor,
    compare_dealias_results,
    record_scored_result,
    record_skipped_result,
    run_solver_pair,
)
from open_dealias import (
    apply_velocity_qc,
    build_velocity_qc_mask,
    dealias_sweep_es90,
    dealias_sweep_jh01,
    dealias_sweep_recursive,
    dealias_sweep_region_graph,
    dealias_sweep_variational,
    dealias_sweep_xu11,
    dealias_sweep_zw06,
    dealias_volume_jh01,
    download_nexrad_key,
    estimate_uniform_wind_vad,
    list_nexrad_keys,
    load_nexrad_sweep,
)
import open_dealias.volume3d as volume3d_module

DEFAULT_TARGET_KEY = "2026/04/15/KLOT/KLOT20260415_152916_V06"
DEFAULT_PREVIOUS_KEY = "2026/04/15/KLOT/KLOT20260415_152449_V06"
DEFAULT_TARGET_SWEEP = 5
DEFAULT_VOLUME_SWEEPS = [0, 1, 2, 3, 4, 5]
DEFAULT_RANGE_GATE_START = 450
DEFAULT_RANGE_GATE_STOP = 850
DEFAULT_OUTPUT_PREFIX = "klot_same_scan_benchmark"


def _masked_to_nan(data) -> np.ndarray:
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=float)


def _pyart_dealias_volume(path: Path) -> tuple[object, np.ndarray]:
    radar = pyart.io.read_nexrad_archive(str(path))
    field = pyart.correct.dealias_region_based(radar, vel_field="velocity", keep_original=False)
    return radar, _masked_to_nan(field["data"])


def _qc_sweep(sweep, *, range_start: int, range_stop: int) -> tuple[np.ndarray, np.ndarray]:
    velocity = sweep.velocity[:, range_start:range_stop]
    reflectivity = None if sweep.reflectivity is None else sweep.reflectivity[:, range_start:range_stop]
    mask = build_velocity_qc_mask(
        velocity,
        reflectivity=reflectivity,
        min_reflectivity=-5.0,
        max_texture=18.0,
        min_gate_fraction_in_ray=0.01,
    )
    return apply_velocity_qc(velocity, mask=mask), mask


def _qc_volume(radar, sweeps: list[int], *, range_start: int, range_stop: int) -> np.ndarray:
    volume = []
    for sweep in sweeps:
        vel = _masked_to_nan(radar.get_field(sweep, "velocity"))
        refl = _masked_to_nan(radar.get_field(sweep, "reflectivity")) if "reflectivity" in radar.fields else None
        vel = vel[:, range_start:range_stop]
        refl = None if refl is None else refl[:, range_start:range_stop]
        volume.append(
            apply_velocity_qc(
                vel,
                reflectivity=refl,
                min_reflectivity=-5.0,
                max_texture=18.0,
                min_gate_fraction_in_ray=0.01,
            )
        )
    return np.stack(volume, axis=0)


def _latest_pair_for_radar(radar_id: str) -> tuple[str, str]:
    keys = list_nexrad_keys(radar_id, datetime.now(timezone.utc))
    if len(keys) < 2:
        raise ValueError(f"need at least two archives for radar {radar_id.upper()} to run temporal benchmarks")
    return keys[-1], keys[-2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark open dealiasing algorithms against Py-ART on one real scan without Py-ART leakage into temporal or supervised inputs.")
    parser.add_argument("--target-key", default=DEFAULT_TARGET_KEY, help="Full Level II S3 key for the target volume.")
    parser.add_argument("--previous-key", default=DEFAULT_PREVIOUS_KEY, help="Full Level II S3 key for the previous volume.")
    parser.add_argument("--target-sweep", type=int, default=DEFAULT_TARGET_SWEEP, help="Sweep index to benchmark.")
    parser.add_argument(
        "--volume-sweeps",
        default=",".join(str(v) for v in DEFAULT_VOLUME_SWEEPS),
        help="Comma-separated sweep indices used by volume algorithms.",
    )
    parser.add_argument("--range-start", type=int, default=DEFAULT_RANGE_GATE_START, help="First range gate to include.")
    parser.add_argument("--range-stop", type=int, default=DEFAULT_RANGE_GATE_STOP, help="Exclusive last range gate to include.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Output filename prefix under examples/output.")
    parser.add_argument(
        "--latest-radar",
        default=None,
        help="Radar ID to auto-select the latest and previous archive from the current UTC day.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.latest_radar:
        target_key, previous_key = _latest_pair_for_radar(args.latest_radar)
    else:
        target_key, previous_key = args.target_key, args.previous_key

    target_sweep_index = int(args.target_sweep)
    volume_sweeps = [int(value) for value in str(args.volume_sweeps).split(",") if str(value).strip()]
    range_start = int(args.range_start)
    range_stop = int(args.range_stop)
    output_prefix = str(args.output_prefix)

    cache_dir = REPO_ROOT / ".cache" / "nexrad"
    target_path = download_nexrad_key(target_key, out_dir=cache_dir)
    previous_path = download_nexrad_key(previous_key, out_dir=cache_dir)

    target_sweep = load_nexrad_sweep(target_path, sweep=target_sweep_index)
    previous_sweep = load_nexrad_sweep(previous_path, sweep=target_sweep_index)

    target_radar, target_pyart_volume = _pyart_dealias_volume(target_path)
    previous_radar, _previous_pyart_volume = _pyart_dealias_volume(previous_path)
    pyart_target = target_pyart_volume[target_radar.get_slice(target_sweep_index)][:, range_start:range_stop]

    target_obs, qc_mask = _qc_sweep(target_sweep, range_start=range_start, range_stop=range_stop)
    previous_obs, _ = _qc_sweep(previous_sweep, range_start=range_start, range_stop=range_stop)
    target_volume_obs = _qc_volume(target_radar, volume_sweeps, range_start=range_start, range_stop=range_stop)
    previous_volume_obs = _qc_volume(previous_radar, volume_sweeps, range_start=range_start, range_stop=range_stop)
    if target_sweep_index not in volume_sweeps:
        raise ValueError("target_sweep must be included in volume_sweeps for the volume methods")
    target_volume_local_index = volume_sweeps.index(target_sweep_index)

    vad_fit = estimate_uniform_wind_vad(
        target_obs,
        target_sweep.nyquist,
        target_sweep.azimuth_deg,
        elevation_deg=target_sweep.elevation_deg,
    )
    previous_open_sweep, previous_open_sweep_meta = open_zw06_sweep_anchor(previous_obs, previous_sweep.nyquist)
    previous_open_volume, previous_open_volume_meta = open_zw06_volume_anchor(
        previous_volume_obs,
        np.array([previous_radar.get_nyquist_vel(i) for i in volume_sweeps], dtype=float),
    )

    results: dict[str, dict[str, object]] = {}
    skipped: list[dict[str, object]] = []
    panel_fields: list[tuple[str, np.ndarray] | tuple[str, str]] = [
        (
            "reflectivity",
            target_sweep.reflectivity[:, range_start:range_stop] if target_sweep.reflectivity is not None else np.where(qc_mask, 0.0, np.nan),
        ),
        ("observed", target_obs),
        ("pyart_region_based", pyart_target),
    ]

    algorithms: list[tuple[str, callable]] = [
        ("es90", lambda: dealias_sweep_es90(target_obs, target_sweep.nyquist, reference=vad_fit.reference, max_gap=4, max_abs_step=12.0)),
        ("zw06", lambda: dealias_sweep_zw06(target_obs, target_sweep.nyquist)),
        ("xu11", lambda: dealias_sweep_xu11(target_obs, target_sweep.nyquist, target_sweep.azimuth_deg, elevation_deg=target_sweep.elevation_deg)),
        ("jh01", lambda: dealias_sweep_jh01(target_obs, target_sweep.nyquist, previous_corrected=previous_open_sweep)),
        (
            "jh01_volume",
            lambda: dealias_volume_jh01(
                target_volume_obs,
                np.array([target_radar.get_nyquist_vel(i) for i in volume_sweeps], dtype=float),
                azimuth_deg=np.asarray(target_radar.get_azimuth(target_sweep_index), dtype=float),
                elevation_deg=np.asarray(target_radar.fixed_angle["data"][volume_sweeps], dtype=float),
                previous_volume=previous_open_volume,
            ),
        ),
        ("region_graph", lambda: dealias_sweep_region_graph(target_obs, target_sweep.nyquist, reference=vad_fit.reference)),
        ("recursive", lambda: dealias_sweep_recursive(target_obs, target_sweep.nyquist, reference=vad_fit.reference)),
        ("variational", lambda: dealias_sweep_variational(target_obs, target_sweep.nyquist, reference=vad_fit.reference, max_iterations=4)),
    ]

    for name, runner in algorithms:
        start = time.perf_counter()
        result = runner()
        runtime = time.perf_counter() - start
        field = result.velocity[target_volume_local_index] if result.velocity.ndim == 3 else result.velocity
        results[name] = record_scored_result(name, field, target_obs, pyart_target, runtime, metadata=result.metadata)
        panel_fields.append((name, field))

    volume3d_pair = run_solver_pair(
        volume3d_module,
        "dealias_volume_3d",
        target_volume_obs,
        np.array([target_radar.get_nyquist_vel(i) for i in volume_sweeps], dtype=float),
        reference_volume=previous_open_volume,
    )
    volume3d_public = volume3d_pair["public_result"]
    volume3d_python = volume3d_pair["python_result"]
    volume3d_field = volume3d_public.velocity[target_volume_local_index] if volume3d_public.velocity.ndim == 3 else volume3d_public.velocity
    results["volume_3d"] = record_scored_result(
        "volume_3d",
        volume3d_field,
        target_obs,
        pyart_target,
        volume3d_pair["public_runtime_s"],
        metadata=volume3d_public.metadata,
    )
    results["volume_3d"].update(
        {
            "backend_available": volume3d_pair["backend_available"],
            "python_runtime_s": volume3d_pair["python_runtime_s"],
            "parity": compare_dealias_results(volume3d_public, volume3d_python),
        }
    )
    panel_fields.append(("volume_3d", volume3d_field))

    skipped.append(
        record_skipped_result(
            "dual_prf",
            "no verified paired low/high-PRF partner is available in this benchmark case, so no synthetic or Py-ART-derived partner is used",
        )
    )
    skipped.append(
        record_skipped_result(
            "ml",
            "no fair supervised training target is available without leakage; this benchmark does not train ML on Py-ART or pseudo-truth targets",
        )
    )

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{output_prefix}_metrics.json"
    png_path = out_dir / f"{output_prefix}.png"

    payload = {
        "target_key": target_key,
        "previous_key": previous_key,
        "target_sweep": target_sweep_index,
        "volume_sweeps": volume_sweeps,
        "range_gate_start": range_start,
        "range_gate_stop": range_stop,
        "nyquist_target": target_sweep.nyquist,
        "valid_qc_velocity_gates": int(np.isfinite(target_obs).sum()),
        "previous_sweep_anchor": previous_open_sweep_meta,
        "previous_volume_anchor": {
            "source": "open_zw06",
            "sweeps": previous_open_volume_meta,
        },
        "algorithms": results,
        "skipped": skipped,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 18), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(np.concatenate([f[np.isfinite(f)] for _, f in panel_fields[1:] if np.any(np.isfinite(f))])), 99.5))
    vmax = max(vmax, target_sweep.nyquist)

    summary_panels = [
        ("skipped", "\n".join(f"{item['name']}: {item['skip_reason']}" for item in skipped)),
    ]
    display_panels: list[tuple[str, np.ndarray] | tuple[str, str]] = panel_fields + summary_panels

    for ax, panel in zip(axes.flat, display_panels):
        title, field = panel
        if isinstance(field, np.ndarray) and title == "reflectivity":
            image = ax.imshow(field, aspect="auto", origin="lower", cmap="NWSRef", vmin=-20, vmax=70)
            ax.set_title(title)
            ax.set_xlabel("Range gate")
            ax.set_ylabel("Azimuth ray")
            fig.colorbar(image, ax=ax, shrink=0.75)
            continue
        elif isinstance(field, np.ndarray):
            image = ax.imshow(field, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Range gate")
            ax.set_ylabel("Azimuth ray")
            fig.colorbar(image, ax=ax, shrink=0.75)
            continue
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.02, 0.95, str(field), ha="left", va="top", fontsize=10, family="monospace", transform=ax.transAxes, wrap=True)

    for ax in axes.flat[len(display_panels):]:
        ax.axis("off")

    plt.suptitle(f"{target_sweep.radar_id} open-anchor benchmark {target_key.rsplit('/', 1)[-1]} sweep {target_sweep_index}")
    plt.savefig(png_path, dpi=150)

    print(f"target archive: {target_path}")
    print(f"previous archive: {previous_path}")
    print(f"target sweep: {target_sweep_index} elev={target_sweep.elevation_deg:.2f} nyquist={target_sweep.nyquist:.2f}")
    print(f"metrics: {json_path}")
    print(f"figure: {png_path}")
    for name, metrics in results.items():
        print(
            f"{name:12s} changed={metrics['changed_gates']:6d} unresolved={metrics['unresolved_fraction']:.4f} "
            f"mae_vs_pyart={metrics['mae_vs_pyart']:.3f} runtime={metrics['runtime_s']:.2f}s"
        )
    for item in skipped:
        print(f"{item['name']:12s} skipped reason={item['skip_reason']}")


if __name__ == "__main__":
    main()
