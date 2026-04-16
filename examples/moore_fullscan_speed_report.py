from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.benchmark_support import compare_dealias_results, open_zw06_volume_anchor, run_solver_pair
import open_dealias.continuity as continuity_module
import open_dealias.fourdd as fourdd_module
import open_dealias.ml as ml_module
import open_dealias.multipass as multipass_module
import open_dealias.qc as qc_module
import open_dealias.region_graph as region_graph_module
import open_dealias.recursive as recursive_module
import open_dealias.vad as vad_module
import open_dealias.variational as variational_module
import open_dealias.volume3d as volume3d_module
from open_dealias import apply_velocity_qc, estimate_uniform_wind_vad


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report full-scan Python vs native-backed runtime on the Moore tornado case.")
    parser.add_argument(
        "--archive-path",
        default="archive_data/KTLX Newcastle - Moore, OK EF5 May 20, 2013.zip",
        help="Zip archive or direct Level II file path for the event.",
    )
    parser.add_argument(
        "--target-fragment",
        default="200356",
        help="Substring identifying the target scan filename inside the archive zip.",
    )
    parser.add_argument("--sweep", type=int, default=1, help="Sweep index for the 2D full-scan comparisons.")
    parser.add_argument(
        "--output-prefix",
        default="moore_fullscan_speed_report",
        help="Prefix for the JSON report written into examples/output.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed samples per solver/policy.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of untimed warmup calls per solver/policy.")
    parser.add_argument(
        "--skip-volume",
        action="store_true",
        help="Skip previous-volume-dependent solvers and anchors for a quicker prepared-input timing run.",
    )
    parser.add_argument(
        "--solvers",
        default="",
        help="Comma-separated subset of solver keys to run, for example 'build_velocity_qc_mask,es90,zw06,region_graph'. Empty means all applicable solvers.",
    )
    return parser.parse_args()


def _filled(field: Any) -> np.ndarray:
    if hasattr(field, "filled"):
        field = field.filled(np.nan)
    return np.asarray(field, dtype=float)


def _materialize_archive_member(archive_path: Path, fragment: str) -> tuple[Path, Path | None]:
    if archive_path.suffix.lower() != ".zip":
        return archive_path, None

    cache_dir = REPO_ROOT / ".cache" / "moore_fullscan_speed" / archive_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = sorted((info for info in zf.infolist() if not info.is_dir()), key=lambda info: Path(info.filename).name)
        matches = [info for info in members if fragment in Path(info.filename).name]
        if not matches:
            raise ValueError(f"no archive member contains fragment {fragment!r}")
        target_info = matches[0]
        previous_info = None
        index = members.index(target_info)
        if index > 0:
            previous_info = members[index - 1]

        target_path = cache_dir / target_info.filename
        if not target_path.exists() or target_path.stat().st_size != target_info.file_size:
            zf.extract(target_info, cache_dir)
        previous_path = None
        if previous_info is not None:
            previous_path = cache_dir / previous_info.filename
            if not previous_path.exists() or previous_path.stat().st_size != previous_info.file_size:
                zf.extract(previous_info, cache_dir)
        return target_path, previous_path


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
        velocity = _filled(radar.get_field(sweep, "velocity"))
        finite = np.abs(velocity[np.isfinite(velocity)])
        if not finite.size:
            continue
        inferred = float(np.ceil(np.nanmax(finite) * 2.0) / 2.0)
        inferred = max(inferred, 1.0)
        sl = radar.get_slice(sweep)
        data[sl] = inferred


def _load_case(target_path: Path, previous_path: Path | None, sweep: int) -> dict[str, Any]:
    import pyart

    target_radar = pyart.io.read_nexrad_archive(str(target_path))
    _patch_zero_nyquist_volume(target_radar)
    if sweep >= target_radar.nsweeps:
        raise ValueError(f"sweep {sweep} is not available in {target_path.name}")

    velocity = _filled(target_radar.get_field(sweep, "velocity"))
    reflectivity = _filled(target_radar.get_field(sweep, "reflectivity"))
    azimuth_deg = np.asarray(target_radar.get_azimuth(sweep), dtype=float)
    elevation_deg = float(target_radar.fixed_angle["data"][sweep])
    nyquist = _resolve_nyquist(target_radar, sweep, velocity)
    qc_velocity = apply_velocity_qc(
        velocity,
        reflectivity=reflectivity,
        min_reflectivity=-5.0,
        max_texture=18.0,
        min_gate_fraction_in_ray=0.01,
    )
    vad_fit = estimate_uniform_wind_vad(qc_velocity, nyquist, azimuth_deg, elevation_deg=elevation_deg)

    target_shape = velocity.shape
    target_volume_obs: list[np.ndarray] | None = None
    target_volume_nyquist: list[float] | None = None
    target_volume_elevation: list[float] | None = None
    selected_sweeps: list[int] | None = None
    previous_volume_obs = None
    previous_volume_nyquist = None
    if previous_path is not None:
        previous_radar = pyart.io.read_nexrad_archive(str(previous_path))
        _patch_zero_nyquist_volume(previous_radar)
        common_sweeps = min(previous_radar.nsweeps, target_radar.nsweeps)
        selected_sweeps = []
        target_volume_obs = []
        target_volume_nyquist = []
        target_volume_elevation = []
        previous_volume_obs = []
        previous_volume_nyquist = []
        for idx in range(common_sweeps):
            target_sweep_velocity = _filled(target_radar.get_field(idx, "velocity"))
            previous_sweep_velocity = _filled(previous_radar.get_field(idx, "velocity"))
            if target_sweep_velocity.shape != target_shape or previous_sweep_velocity.shape != target_shape:
                continue
            target_sweep_reflectivity = _filled(target_radar.get_field(idx, "reflectivity"))
            previous_sweep_reflectivity = _filled(previous_radar.get_field(idx, "reflectivity"))
            target_sweep_nyquist = _resolve_nyquist(target_radar, idx, target_sweep_velocity)
            previous_sweep_nyquist = _resolve_nyquist(previous_radar, idx, previous_sweep_velocity)
            target_volume_obs.append(
                apply_velocity_qc(
                    target_sweep_velocity,
                    reflectivity=target_sweep_reflectivity,
                    min_reflectivity=-5.0,
                    max_texture=18.0,
                    min_gate_fraction_in_ray=0.01,
                )
            )
            previous_volume_obs.append(
                apply_velocity_qc(
                    previous_sweep_velocity,
                    reflectivity=previous_sweep_reflectivity,
                    min_reflectivity=-5.0,
                    max_texture=18.0,
                    min_gate_fraction_in_ray=0.01,
                )
            )
            target_volume_nyquist.append(target_sweep_nyquist)
            target_volume_elevation.append(float(target_radar.fixed_angle["data"][idx]))
            previous_volume_nyquist.append(previous_sweep_nyquist)
            selected_sweeps.append(idx)
        if not selected_sweeps:
            target_volume_obs = None
            target_volume_nyquist = None
            target_volume_elevation = None
            previous_volume_obs = None
            previous_volume_nyquist = None

    return {
        "scan": target_path.name,
        "previous_scan": None if previous_path is None else previous_path.name,
        "sweep": int(sweep),
        "shape": list(qc_velocity.shape),
        "nyquist": float(nyquist),
        "qc_velocity": qc_velocity,
        "reflectivity": reflectivity,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "reference": np.asarray(vad_fit.reference, dtype=float),
        "volume_sweeps": selected_sweeps,
        "target_volume_obs": None if target_volume_obs is None else np.asarray(target_volume_obs, dtype=float),
        "target_volume_nyquist": None if target_volume_nyquist is None else np.asarray(target_volume_nyquist, dtype=float),
        "target_volume_elevation": None if target_volume_elevation is None else np.asarray(target_volume_elevation, dtype=float),
        "previous_volume_obs": None if previous_volume_obs is None else np.asarray(previous_volume_obs, dtype=float),
        "previous_volume_nyquist": None if previous_volume_nyquist is None else np.asarray(previous_volume_nyquist, dtype=float),
    }


def _solver_summary(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "auto_runtime_s": float(pair["public_runtime_s"]),
        "python_runtime_s": float(pair["python_runtime_s"]),
        "python_policy_runtime_s": float(pair["python_policy_runtime_s"]),
        "speedup": float(pair["python_runtime_s"] / pair["public_runtime_s"]) if pair["public_runtime_s"] > 0 else float("nan"),
        "backend_available": bool(pair["backend_available"]),
        "native_acceleration_available": bool(pair["native_acceleration_available"]),
        "comparison_mode": str(pair["comparison_mode"]),
        "comparison_scope": str(pair["comparison_scope"]),
        "auto_policy": str(pair["public_policy"]),
        "python_policy": str(pair["python_policy"]),
        "repeats": int(pair["repeats"]),
        "warmup": int(pair["warmup"]),
        "parity": compare_dealias_results(pair["public_result"], pair["python_result"]),
        "metadata": dict(pair["public_result"].metadata),
    }


def main() -> None:
    args = _parse_args()
    selected_solvers = {name.strip() for name in str(args.solvers).split(",") if name.strip()}

    def want(name: str) -> bool:
        return not selected_solvers or name in selected_solvers

    prep_start = time.perf_counter()
    archive_path = (REPO_ROOT / args.archive_path).resolve() if not Path(args.archive_path).is_absolute() else Path(args.archive_path)
    target_path, previous_path = _materialize_archive_member(archive_path, args.target_fragment)
    case = _load_case(target_path, previous_path, int(args.sweep))

    observed = np.asarray(case["qc_velocity"], dtype=float)
    reflectivity = np.asarray(case["reflectivity"], dtype=float)
    reference = np.asarray(case["reference"], dtype=float)
    nyquist = float(case["nyquist"])
    prep_runtime_s = time.perf_counter() - prep_start

    results: dict[str, dict[str, Any]] = {}

    if want("build_velocity_qc_mask"):
        qc_pair = run_solver_pair(
            qc_module,
            "build_velocity_qc_mask",
            observed,
            reflectivity=reflectivity,
            texture=None,
            min_reflectivity=-5.0,
            max_texture=18.0,
            min_gate_fraction_in_ray=0.01,
            wrap_azimuth=True,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        results["build_velocity_qc_mask"] = {
            "auto_runtime_s": float(qc_pair["public_runtime_s"]),
            "python_runtime_s": float(qc_pair["python_runtime_s"]),
            "python_policy_runtime_s": float(qc_pair["python_policy_runtime_s"]),
            "speedup": float(qc_pair["python_runtime_s"] / qc_pair["public_runtime_s"]) if qc_pair["public_runtime_s"] > 0 else float("nan"),
            "backend_available": bool(qc_pair["backend_available"]),
            "native_acceleration_available": bool(qc_pair["native_acceleration_available"]),
            "comparison_mode": str(qc_pair["comparison_mode"]),
            "comparison_scope": str(qc_pair["comparison_scope"]),
            "auto_policy": str(qc_pair["public_policy"]),
            "python_policy": str(qc_pair["python_policy"]),
            "repeats": int(qc_pair["repeats"]),
            "warmup": int(qc_pair["warmup"]),
        }

    if want("es90"):
        results["es90"] = _solver_summary(
            run_solver_pair(
                continuity_module,
                "dealias_sweep_es90",
                observed,
                nyquist,
                reference=reference,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    if want("zw06"):
        results["zw06"] = _solver_summary(
            run_solver_pair(
                multipass_module,
                "dealias_sweep_zw06",
                observed,
                nyquist,
                reference=reference,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    shared_anchor_runtime_s = 0.0
    if want("xu11"):
        results["xu11"] = _solver_summary(
            run_solver_pair(
                vad_module,
                "dealias_sweep_xu11",
                observed,
                nyquist,
                case["azimuth_deg"],
                elevation_deg=case["elevation_deg"],
                external_reference=reference,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    if want("region_graph"):
        results["region_graph"] = _solver_summary(
            run_solver_pair(
                region_graph_module,
                "dealias_sweep_region_graph",
                observed,
                nyquist,
                reference=reference,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    if (
        not args.skip_volume
        and want("jh01")
        and case.get("previous_volume_obs") is not None
        and case.get("volume_sweeps") is not None
        and case["sweep"] in case["volume_sweeps"]
    ):
        current_index = case["volume_sweeps"].index(case["sweep"])
        previous_current_sweep = np.asarray(case["previous_volume_obs"], dtype=float)[current_index]
        anchor_start = time.perf_counter()
        previous_open_current_sweep = open_zw06_volume_anchor(
            np.asarray([previous_current_sweep], dtype=float),
            np.asarray([np.asarray(case["previous_volume_nyquist"], dtype=float)[current_index]], dtype=float),
        )[0][0]
        shared_anchor_runtime_s += time.perf_counter() - anchor_start
        results["jh01"] = _solver_summary(
            run_solver_pair(
                fourdd_module,
                "dealias_sweep_jh01",
                observed,
                nyquist,
                previous_corrected=previous_open_current_sweep,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    if want("recursive"):
        results["recursive"] = _solver_summary(
            run_solver_pair(
                recursive_module,
                "dealias_sweep_recursive",
                observed,
                nyquist,
                reference=reference,
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    if want("variational"):
        results["variational"] = _solver_summary(
            run_solver_pair(
                variational_module,
                "dealias_sweep_variational",
                observed,
                nyquist,
                reference=reference,
                max_iterations=4,
                backend_method="dealias_sweep_variational_refine",
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    if want("ml"):
        ml_pair = run_solver_pair(
            ml_module,
            "dealias_sweep_ml",
            observed,
            nyquist,
            training_target=reference,
            reference=reference,
            azimuth_deg=case["azimuth_deg"],
            refine_with_variational=False,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        results["ml"] = _solver_summary(ml_pair)

    if (
        not args.skip_volume
        and (want("volume_3d") or want("jh01_volume"))
        and case["previous_volume_obs"] is not None
        and case["previous_volume_nyquist"] is not None
    ):
        anchor_start = time.perf_counter()
        previous_open_volume, previous_open_volume_meta = open_zw06_volume_anchor(
            np.asarray(case["previous_volume_obs"], dtype=float),
            np.asarray(case["previous_volume_nyquist"], dtype=float),
        )
        shared_anchor_runtime_s += time.perf_counter() - anchor_start
        if want("volume_3d"):
            volume_pair = run_solver_pair(
                volume3d_module,
                "dealias_volume_3d",
                np.asarray(case["target_volume_obs"], dtype=float),
                np.asarray(case["target_volume_nyquist"], dtype=float),
                reference_volume=previous_open_volume,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            results["volume_3d"] = _solver_summary(volume_pair)
            results["volume_3d"]["previous_volume_anchor"] = previous_open_volume_meta
        if want("jh01_volume"):
            results["jh01_volume"] = _solver_summary(
                run_solver_pair(
                    fourdd_module,
                    "dealias_volume_jh01",
                    np.asarray(case["target_volume_obs"], dtype=float),
                    np.asarray(case["target_volume_nyquist"], dtype=float),
                    azimuth_deg=case["azimuth_deg"],
                    elevation_deg=np.asarray(case["target_volume_elevation"], dtype=float),
                    previous_volume=previous_open_volume,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
            )
            results["jh01_volume"]["previous_volume_anchor"] = previous_open_volume_meta

    aggregate_keys = list(results)
    total_auto = float(sum(results[name]["auto_runtime_s"] for name in aggregate_keys))
    total_python = float(sum(results[name]["python_runtime_s"] for name in aggregate_keys))
    parity_issues = [
        {
            "name": name,
            "folds_mismatch_count": int(metrics["parity"]["folds_mismatch_count"]),
            "velocity_mae": float(metrics["parity"]["velocity_mae"]),
            "velocity_max_abs_diff": float(metrics["parity"]["velocity_max_abs_diff"]),
        }
        for name, metrics in results.items()
        if isinstance(metrics, dict)
        and isinstance(metrics.get("parity"), dict)
        and (
            int(metrics["parity"].get("folds_mismatch_count", 0)) > 0
            or float(metrics["parity"].get("velocity_mae", 0.0)) > 0.0
        )
    ]

    payload = {
        "scan": case["scan"],
        "previous_scan": case["previous_scan"],
        "sweep": case["sweep"],
        "shape": case["shape"],
        "nyquist": case["nyquist"],
        "benchmark_scope": {
            "comparison_mode": "prepared_solver_call_auto_vs_transitive_python_policy",
            "timed_stage": "solver_entrypoints_on_prepared_case",
            "comparison_meaning": "Each timed pair runs once under package-wide auto policy and once under package-wide python policy so transitive native dispatch is disabled in the comparison run.",
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "skip_volume": bool(args.skip_volume),
            "selected_solvers": sorted(selected_solvers) if selected_solvers else "all_applicable",
            "excludes": [
                "archive download and extraction",
                "Py-ART archive loading",
                "shared QC and VAD preparation performed in _load_case",
                "shared previous-volume anchor generation outside the individual solver timers",
            ],
            "shared_case_prep_runtime_s": float(prep_runtime_s),
            "shared_anchor_runtime_s": float(shared_anchor_runtime_s),
        },
        "measured": results,
        "aggregate": {
            "measured_calls": aggregate_keys,
            "total_auto_solver_runtime_s": total_auto,
            "total_python_solver_runtime_s": total_python,
            "solver_stage_speedup": float(total_python / total_auto) if total_auto > 0 else float("nan"),
        },
        "parity_issues": parity_issues,
    }

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.output_prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json_path)
    print(f"shared case prep runtime: {prep_runtime_s:.3f}s")
    print(f"shared anchor runtime:    {shared_anchor_runtime_s:.3f}s")
    print(f"aggregate package-python solver runtime: {total_python:.3f}s")
    print(f"aggregate package-auto solver runtime:   {total_auto:.3f}s")
    print(f"aggregate solver-stage speedup: {payload['aggregate']['solver_stage_speedup']:.2f}x")
    print("note: these are prepared solver-call timings, not full archive-to-product wall time.")
    for name, metrics in results.items():
        print(
            f"{name:22s} auto={metrics['auto_runtime_s']:.3f}s "
            f"python_policy={metrics['python_policy_runtime_s']:.3f}s "
            f"speedup={metrics['speedup']:.2f}x"
        )
    if parity_issues:
        print("parity issues detected on this real case:")
        for item in parity_issues:
            print(
                f"  {item['name']}: folds_mismatch_count={item['folds_mismatch_count']} "
                f"velocity_mae={item['velocity_mae']:.6f} "
                f"velocity_max_abs_diff={item['velocity_max_abs_diff']:.3f}"
            )


if __name__ == "__main__":
    main()
