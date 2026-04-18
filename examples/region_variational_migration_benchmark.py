from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.benchmark_support import (
    changed_gates,
    compare_dealias_results,
    discover_archive_files,
    mae,
    open_zw06_volume_anchor,
    run_solver_pair,
    unresolved_fraction,
)
from open_dealias import (
    apply_velocity_qc,
    build_reference_from_uv,
    dealias_sweep_region_graph,
    dealias_sweep_recursive,
    dealias_sweep_variational,
    estimate_uniform_wind_vad,
    load_pal_table,
    wrap_to_nyquist,
)
import open_dealias.volume3d as volume3d_module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark region_graph and variational on synthetic and real-like cases with public-vs-Python fallback parity."
    )
    parser.add_argument("--archive-root", default="archive_data", help="Folder containing local archive zip files.")
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Optional explicit archive zip path. Defaults to a KTLX archive if present, otherwise the first discovered zip.",
    )
    parser.add_argument("--sweep", type=int, default=1, help="Sweep index to use for the archive case.")
    parser.add_argument("--crop-half-size", type=int, default=28, help="Half-size of the square crop around the vrot center.")
    parser.add_argument("--output-prefix", default="region_variational_migration_benchmark", help="Prefix for JSON and PNG outputs.")
    parser.add_argument("--synthetic-only", action="store_true", help="Skip the archive case and run synthetic cases only.")
    parser.add_argument("--velocity-palette", default="color_tables/awips_bv.pal", help="Palette used for velocity panels.")
    return parser.parse_args()


def _filled(field) -> np.ndarray:
    if hasattr(field, "filled"):
        field = field.filled(np.nan)
    return np.asarray(field, dtype=float)


def _center_indices_from_couplet(velocity: np.ndarray, reflectivity: np.ndarray) -> tuple[int, int]:
    rng = np.arange(velocity.shape[1])[None, :]
    mask = np.isfinite(velocity) & np.isfinite(reflectivity) & (reflectivity > 25.0) & (rng > 4)
    pos_idx = np.argwhere(mask & (velocity > 8.0))
    neg_idx = np.argwhere(mask & (velocity < -8.0))
    if len(pos_idx) < 4 or len(neg_idx) < 4:
        iy, ix = np.unravel_index(np.nanargmax(np.abs(np.where(np.isfinite(velocity), velocity, np.nan))), velocity.shape)
        return int(iy), int(ix)

    pos_vals = velocity[pos_idx[:, 0], pos_idx[:, 1]]
    neg_vals = velocity[neg_idx[:, 0], neg_idx[:, 1]]
    pos_keep = pos_idx[np.argsort(pos_vals)[-80:]]
    neg_keep = neg_idx[np.argsort(neg_vals)[:80]]

    best_score = -1.0e9
    best_center = (int(pos_keep[0, 0]), int(pos_keep[0, 1]))
    for pi in pos_keep:
        pv = velocity[pi[0], pi[1]]
        dr = neg_keep[:, 0] - pi[0]
        dc = neg_keep[:, 1] - pi[1]
        dist = np.hypot(dr, dc)
        nv = velocity[neg_keep[:, 0], neg_keep[:, 1]]
        score = (pv - nv) - 2.5 * dist
        j = int(np.argmax(score))
        if float(score[j]) > best_score:
            ni = neg_keep[j]
            best_score = float(score[j])
            best_center = (int(round((pi[0] + ni[0]) / 2.0)), int(round((pi[1] + ni[1]) / 2.0)))
    return best_center


def _crop_square(*arrays: np.ndarray, center: tuple[int, int], half_size: int) -> tuple[np.ndarray, ...]:
    row, col = center
    r0 = max(0, row - half_size)
    r1 = min(arrays[0].shape[0], row + half_size)
    c0 = max(0, col - half_size)
    c1 = min(arrays[0].shape[1], col + half_size)
    return tuple(arr[r0:r1, c0:c1] for arr in arrays)


def build_synthetic_cases(crop_half_size: int = 28) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    nyquist = 10.0

    az1 = np.linspace(0.0, 360.0, 150, endpoint=False)
    rg1 = np.linspace(0.0, 1.0, 120)[None, :]
    azr1 = np.deg2rad(az1)[:, None]
    base_ref1 = build_reference_from_uv(az1, 120, u=9.0, v=2.5)
    coarse_ref1 = base_ref1 + 10.0 * rg1
    truth1 = base_ref1 + 18.0 * rg1 + 4.5 * np.sin(3.0 * azr1) * np.exp(-((rg1 - 0.58) ** 2) / 0.025)
    obs1 = wrap_to_nyquist(truth1, nyquist)
    obs1[38:54, 26:40] = np.nan
    refl1 = 24.0 + 22.0 * np.exp(-((rg1 - 0.50) ** 2) / 0.06) + 6.0 * np.cos(2.0 * azr1)
    center1 = _center_indices_from_couplet(obs1, refl1)
    obs1, refl1, truth1, coarse_ref1 = _crop_square(obs1, refl1, truth1, coarse_ref1, center=center1, half_size=crop_half_size)
    cases.append(
        {
            "kind": "synthetic",
            "name": "synthetic_shear_bridge",
            "sweep": 0,
            "nyquist": nyquist,
            "observed": obs1,
            "reflectivity": refl1,
            "reference": coarse_ref1,
            "truth": truth1,
            "crop_center": center1,
        }
    )

    az2 = np.linspace(0.0, 360.0, 180, endpoint=False)
    rg2 = np.linspace(0.0, 1.0, 150)[None, :]
    azr2 = np.deg2rad(az2)[:, None]
    base_ref2 = build_reference_from_uv(az2, 150, u=7.5, v=-3.0)
    coarse_ref2 = base_ref2 + 12.0 * rg2
    truth2 = base_ref2 + 22.0 * rg2 + 6.0 * np.exp(-((rg2 - 0.72) ** 2) / 0.020) * np.cos(4.0 * azr2)
    truth2[52:72, 65:92] = np.nan
    obs2 = wrap_to_nyquist(truth2, nyquist)
    refl2 = 20.0 + 32.0 * np.exp(-((rg2 - 0.63) ** 2) / 0.030) + 4.0 * np.sin(3.0 * azr2)
    center2 = _center_indices_from_couplet(obs2, refl2)
    obs2, refl2, truth2, coarse_ref2 = _crop_square(obs2, refl2, truth2, coarse_ref2, center=center2, half_size=crop_half_size)
    cases.append(
        {
            "kind": "synthetic",
            "name": "synthetic_split_couplet",
            "sweep": 0,
            "nyquist": nyquist,
            "observed": obs2,
            "reflectivity": refl2,
            "reference": coarse_ref2,
            "truth": truth2,
            "crop_center": center2,
        }
    )
    return cases


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
    return max(inferred, 1.0)


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


def _default_archive_path(archive_root: Path) -> Path | None:
    preferred = archive_root / "KTLX Newcastle - Moore, OK EF5 May 20, 2013.zip"
    if preferred.exists():
        return preferred
    archives = discover_archive_files(archive_root)
    return archives[0] if archives else None


def _extract_archive_member(archive_path: Path, member: str) -> Path:
    cache_dir = REPO_ROOT / ".cache" / "region_variational_migration" / archive_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = {Path(info.filename).name: info for info in zf.infolist() if not info.is_dir()}
        if member not in members:
            raise FileNotFoundError(f"{member} not found inside {archive_path.name}")
        info = members[member]
        extracted = cache_dir / info.filename
        if not extracted.exists() or extracted.stat().st_size != info.file_size:
            zf.extract(info, cache_dir)
        return extracted


def _select_archive_members(archive_path: Path, preferred_member: str | None = None) -> tuple[Path, Path | None]:
    if archive_path.suffix.lower() != ".zip":
        return archive_path, None

    with zipfile.ZipFile(archive_path) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if not members:
            raise ValueError(f"{archive_path} does not contain any files")
        member_names = [Path(info.filename).name for info in members]

    if preferred_member is None:
        if "Moore" in archive_path.name:
            preferred_member = "KTLX20130520_200356_V06.gz"
        else:
            preferred_member = member_names[0]

    if preferred_member not in member_names:
        preferred_member = member_names[0]

    current_index = member_names.index(preferred_member)
    previous_index = current_index - 1 if current_index > 0 else None
    current_path = _extract_archive_member(archive_path, member_names[current_index])
    previous_path = None if previous_index is None else _extract_archive_member(archive_path, member_names[previous_index])
    return current_path, previous_path


def _qc_volume(radar, sweeps: list[int]) -> np.ndarray:
    volume = []
    for sweep in sweeps:
        velocity = _filled(radar.get_field(sweep, "velocity"))
        reflectivity = _filled(radar.get_field(sweep, "reflectivity"))
        volume.append(
            apply_velocity_qc(
                velocity,
                reflectivity=reflectivity,
                min_reflectivity=-5.0,
                max_texture=18.0,
                min_gate_fraction_in_ray=0.01,
            )
        )
    if not volume:
        raise ValueError("no sweeps available for volume QC")
    min_rows = min(field.shape[0] for field in volume)
    min_cols = min(field.shape[1] for field in volume)
    cropped = [field[:min_rows, :min_cols] for field in volume]
    return np.stack(cropped, axis=0)


def load_real_like_case(archive_path: Path, *, sweep: int, crop_half_size: int, member_name: str | None = None) -> dict[str, Any]:
    import pyart

    materialized, previous_materialized = _select_archive_members(archive_path, member_name)
    radar = pyart.io.read_nexrad_archive(str(materialized))
    previous_radar = None if previous_materialized is None else pyart.io.read_nexrad_archive(str(previous_materialized))
    _patch_zero_nyquist_volume(radar)
    if previous_radar is not None:
        _patch_zero_nyquist_volume(previous_radar)
    if sweep >= radar.nsweeps:
        raise ValueError(f"sweep {sweep} is not available in {archive_path.name}")

    velocity = _filled(radar.get_field(sweep, "velocity"))
    reflectivity = _filled(radar.get_field(sweep, "reflectivity"))
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
    pyart_region = _filled(pyart.correct.dealias_region_based(radar, vel_field="velocity", keep_original=False)["data"][radar.get_slice(sweep)])
    center = _center_indices_from_couplet(qc_velocity, reflectivity)
    qc_velocity, reflectivity, pyart_region, reference = _crop_square(
        qc_velocity,
        reflectivity,
        pyart_region,
        vad_fit.reference,
        center=center,
        half_size=crop_half_size,
    )
    volume_sweeps = list(range(radar.nsweeps))
    target_volume_obs = _qc_volume(radar, volume_sweeps)
    target_volume_nyquist = np.array([float(radar.get_nyquist_vel(i)) for i in volume_sweeps], dtype=float)
    previous_volume_obs = None
    previous_volume_nyquist = None
    if previous_radar is not None and previous_radar.nsweeps == radar.nsweeps:
        previous_volume_obs = _qc_volume(previous_radar, volume_sweeps)
        previous_volume_nyquist = np.array([float(previous_radar.get_nyquist_vel(i)) for i in volume_sweeps], dtype=float)
        common_rows = min(target_volume_obs.shape[1], previous_volume_obs.shape[1])
        common_cols = min(target_volume_obs.shape[2], previous_volume_obs.shape[2])
        target_volume_obs = target_volume_obs[:, :common_rows, :common_cols]
        previous_volume_obs = previous_volume_obs[:, :common_rows, :common_cols]
    return {
        "kind": "archive",
        "name": archive_path.stem,
        "sweep": int(sweep),
        "archive_path": str(archive_path),
        "materialized_path": str(materialized),
        "previous_materialized_path": None if previous_materialized is None else str(previous_materialized),
        "nyquist": nyquist,
        "observed": qc_velocity,
        "reflectivity": reflectivity,
        "reference": reference,
        "truth": None,
        "external_reference": pyart_region,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "crop_center": center,
        "volume_sweeps": volume_sweeps,
        "target_volume_obs": target_volume_obs,
        "target_volume_nyquist": target_volume_nyquist,
        "previous_volume_obs": previous_volume_obs,
        "previous_volume_nyquist": previous_volume_nyquist,
    }


def _time_region_graph_case(case: dict[str, Any]) -> dict[str, Any]:
    observed = np.asarray(case["observed"], dtype=float)
    reference = np.asarray(case["reference"], dtype=float)
    nyquist = float(case["nyquist"])

    import open_dealias.region_graph as region_graph_module

    region_pair = run_solver_pair(region_graph_module, "dealias_sweep_region_graph", observed, nyquist, reference=reference)
    region_public = region_pair["public_result"]
    region_python = region_pair["python_result"]

    return {
        "public_result": region_public,
        "python_result": region_python,
        "metrics": {
            "public_runtime_s": region_pair["public_runtime_s"],
            "python_runtime_s": region_pair["python_runtime_s"],
            "backend_available": region_pair["backend_available"],
            "parity": compare_dealias_results(region_public, region_python),
            "changed_gates": changed_gates(region_public.velocity, observed),
            "unresolved_fraction": unresolved_fraction(region_public.velocity, observed),
            "mae_vs_reference": mae(region_public.velocity, reference),
            "mae_vs_truth": None if case.get("truth") is None else mae(region_public.velocity, np.asarray(case["truth"], dtype=float)),
            "metadata": dict(region_public.metadata),
            "array": region_public.velocity,
        },
    }


def _time_case(case: dict[str, Any], *, include_volume3d: bool = False) -> dict[str, Any]:
    observed = np.asarray(case["observed"], dtype=float)
    reference = np.asarray(case["reference"], dtype=float)
    nyquist = float(case["nyquist"])

    import open_dealias.variational as variational_module
    import open_dealias.recursive as recursive_module

    region = _time_region_graph_case(case)
    recursive_pair = run_solver_pair(
        recursive_module,
        "dealias_sweep_recursive",
        observed,
        nyquist,
        reference=reference,
        max_depth=4,
    )
    variational_pair = run_solver_pair(
        variational_module,
        "dealias_sweep_variational",
        observed,
        nyquist,
        reference=reference,
        max_iterations=4,
    )

    region_public = region["public_result"]
    recursive_public = recursive_pair["public_result"]
    recursive_python = recursive_pair["python_result"]
    variational_public = variational_pair["public_result"]
    variational_python = variational_pair["python_result"]
    volume3d_metrics: dict[str, Any] | None = None
    if include_volume3d and case.get("previous_volume_obs") is not None and case.get("previous_volume_nyquist") is not None:
        previous_volume_obs = np.asarray(case["previous_volume_obs"], dtype=float)
        previous_volume_nyquist = np.asarray(case["previous_volume_nyquist"], dtype=float)
        target_volume_obs = np.asarray(case["target_volume_obs"], dtype=float)
        target_volume_nyquist = np.asarray(case["target_volume_nyquist"], dtype=float)
        previous_open_volume, previous_open_volume_meta = open_zw06_volume_anchor(previous_volume_obs, previous_volume_nyquist)
        volume3d_pair = run_solver_pair(
            volume3d_module,
            "dealias_volume_3d",
            target_volume_obs,
            target_volume_nyquist,
            reference_volume=previous_open_volume,
        )
        volume3d_public = volume3d_pair["public_result"]
        volume3d_python = volume3d_pair["python_result"]
        volume_slice = volume3d_public.velocity[case["volume_sweeps"].index(case["sweep"])]
        volume3d_metrics = {
            "public_runtime_s": volume3d_pair["public_runtime_s"],
            "python_runtime_s": volume3d_pair["python_runtime_s"],
            "backend_available": volume3d_pair["backend_available"],
            "parity": compare_dealias_results(volume3d_public, volume3d_python),
            "changed_gates": changed_gates(volume_slice, observed),
            "unresolved_fraction": unresolved_fraction(volume_slice, observed),
            "mae_vs_reference": mae(volume_slice, reference),
            "metadata": dict(volume3d_public.metadata),
            "previous_volume_anchor": previous_open_volume_meta,
            "array": volume3d_public.velocity,
        }

    result: dict[str, Any] = {
        "kind": case["kind"],
        "name": case["name"],
        "sweep": int(case["sweep"]),
        "nyquist": nyquist,
        "shape": list(observed.shape),
        "observed_finite_fraction": float(np.mean(np.isfinite(observed))),
        "reference_finite_fraction": float(np.mean(np.isfinite(reference))),
        "region_graph": region["metrics"],
        "recursive": {
            "public_runtime_s": recursive_pair["public_runtime_s"],
            "python_runtime_s": recursive_pair["python_runtime_s"],
            "backend_available": recursive_pair["backend_available"],
            "parity": compare_dealias_results(recursive_public, recursive_python),
            "changed_gates": changed_gates(recursive_public.velocity, observed),
            "unresolved_fraction": unresolved_fraction(recursive_public.velocity, observed),
            "mae_vs_reference": mae(recursive_public.velocity, reference),
            "mae_vs_truth": None if case.get("truth") is None else mae(recursive_public.velocity, np.asarray(case["truth"], dtype=float)),
            "metadata": dict(recursive_public.metadata),
            "array": recursive_public.velocity,
        },
        "variational": {
            "public_runtime_s": variational_pair["public_runtime_s"],
            "python_runtime_s": variational_pair["python_runtime_s"],
            "backend_available": variational_pair["backend_available"],
            "parity": compare_dealias_results(variational_public, variational_python),
            "changed_gates": changed_gates(variational_public.velocity, observed),
            "unresolved_fraction": unresolved_fraction(variational_public.velocity, observed),
            "mae_vs_reference": mae(variational_public.velocity, reference),
            "mae_vs_truth": None if case.get("truth") is None else mae(variational_public.velocity, np.asarray(case["truth"], dtype=float)),
            "metadata": dict(variational_public.metadata),
            "array": variational_public.velocity,
        },
    }

    if volume3d_metrics is not None:
        result["volume_3d"] = volume3d_metrics

    if case.get("external_reference") is not None:
        external_reference = np.asarray(case["external_reference"], dtype=float)
        result["region_graph"]["mae_vs_external_reference"] = mae(region_public.velocity, external_reference)
        result["recursive"]["mae_vs_external_reference"] = mae(recursive_public.velocity, external_reference)
        result["variational"]["mae_vs_external_reference"] = mae(variational_public.velocity, external_reference)
        if volume3d_metrics is not None:
            result["volume_3d"]["mae_vs_external_reference"] = mae(volume3d_public.velocity[case["volume_sweeps"].index(case["sweep"])], external_reference)

    result["cross_solver"] = {
        "variational_vs_region": compare_dealias_results(variational_public, region_public),
        "recursive_vs_region": compare_dealias_results(recursive_public, region_public),
        "recursive_vs_variational": compare_dealias_results(recursive_public, variational_public),
    }
    return result


def _render_summary(cases: list[dict[str, Any]], results: list[dict[str, Any]], palette_path: str, output_path: Path) -> None:
    palette = load_pal_table(palette_path)
    velocity_cmap, vel_vmin, vel_vmax = palette.matplotlib_colormap(name=f"{palette.name}_internal")

    rows = len(cases)
    fig, axes = plt.subplots(rows, 4, figsize=(20, max(6.0, 4.6 * rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, (case, result) in enumerate(zip(cases, results)):
        observed = np.asarray(case["observed"], dtype=float)
        region = np.asarray(result["region_graph"]["array"], dtype=float)
        variational = np.asarray(result["variational"]["array"], dtype=float)
        diff = np.abs(variational - region)

        panels = [
            ("Observed", observed, velocity_cmap, vel_vmin, vel_vmax),
            ("Region Graph", region, velocity_cmap, vel_vmin, vel_vmax),
            ("Variational", variational, velocity_cmap, vel_vmin, vel_vmax),
            ("|Variational - Region|", diff, "inferno", 0.0, max(float(np.nanpercentile(diff[np.isfinite(diff)], 99.0)) if np.any(np.isfinite(diff)) else 1.0, 1e-6)),
        ]

        for col, (title, field, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Range gate")
            ax.set_ylabel("Azimuth ray")
            fig.colorbar(image, ax=ax, shrink=0.8)

        region_metrics = result["region_graph"]
        variational_metrics = result["variational"]
        axes[row, 0].text(
            0.02,
            0.98,
            (
                f"{case['name']}\n"
                f"region_rt={region_metrics['public_runtime_s']:.3f}s  par={region_metrics['parity']['velocity_mae']:.3g}\n"
                f"var_rt={variational_metrics['public_runtime_s']:.3f}s  par={variational_metrics['parity']['velocity_mae']:.3g}"
            ),
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
            color="white",
        )

    fig.suptitle(f"region_graph vs variational migration benchmark  palette={palette.name}", fontsize=14)
    fig.savefig(output_path, dpi=160)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if key != "array"}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    args = _parse_args()
    cases = build_synthetic_cases(crop_half_size=int(args.crop_half_size))

    archive_case: dict[str, Any] | None = None
    if not args.synthetic_only:
        archive_root = Path(args.archive_root)
        archive_path = Path(args.archive_path) if args.archive_path else _default_archive_path(archive_root)
        if archive_path is not None:
            archive_case = load_real_like_case(archive_path, sweep=int(args.sweep), crop_half_size=int(args.crop_half_size))
            cases.append(archive_case)

    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(_time_case(case, include_volume3d=True))

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix)
    json_path = out_dir / f"{output_prefix}.json"
    png_path = out_dir / f"{output_prefix}.png"

    payload = {
        "cases": [_json_safe(result) for result in results],
        "archive_case_included": archive_case is not None,
        "archive_case_name": None if archive_case is None else archive_case["name"],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _render_summary(cases, results, args.velocity_palette, png_path)
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
