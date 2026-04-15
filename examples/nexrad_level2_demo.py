from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_dealias import dealias_sweep_xu11, dealias_sweep_zw06
from open_dealias.nexrad import download_nexrad_key, find_nexrad_key, load_nexrad_sweep


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _masked_to_nan(data):
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=float)


def _pyart_region_reference(path: Path, sweep_index: int) -> np.ndarray:
    import pyart

    radar = pyart.io.read_nexrad_archive(str(path))
    field = pyart.correct.dealias_region_based(radar, vel_field="velocity", keep_original=False)
    sl = radar.get_slice(sweep_index)
    return _masked_to_nan(field["data"][sl])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _changed(candidate: np.ndarray, observed: np.ndarray) -> int:
    mask = np.isfinite(candidate) & np.isfinite(observed)
    return int(np.count_nonzero(np.abs(candidate[mask] - observed[mask]) > 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run open dealiasing on a real NEXRAD Level II archive")
    parser.add_argument("--radar", default="KLOT", help="4-letter radar ID, default: KLOT")
    parser.add_argument(
        "--time",
        default=None,
        help="Target UTC time like 2026-04-15T15:33:00Z. If omitted, latest available scan is used.",
    )
    parser.add_argument("--sweep", default="auto", help="Sweep index or 'auto'")
    parser.add_argument("--cache-dir", default=".cache/nexrad", help="Local archive cache directory")
    parser.add_argument("--skip-pyart-reference", action="store_true", help="Skip the slower Py-ART reference pass")
    args = parser.parse_args()

    target_time = _parse_time(args.time)
    key = find_nexrad_key(args.radar, target_time=target_time, latest=target_time is None)
    archive_path = download_nexrad_key(key, out_dir=args.cache_dir)
    sweep = load_nexrad_sweep(archive_path, sweep=args.sweep)

    zw = dealias_sweep_zw06(sweep.velocity, sweep.nyquist)
    xu = dealias_sweep_xu11(sweep.velocity, sweep.nyquist, sweep.azimuth_deg, elevation_deg=sweep.elevation_deg)
    pyart_ref = None if args.skip_pyart_reference else _pyart_region_reference(archive_path, sweep.sweep_index)

    fields = [arr for arr in [sweep.velocity, zw.velocity, xu.velocity, pyart_ref] if arr is not None]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([arr[np.isfinite(arr)] for arr in fields])), 99.5))
    vmax = max(vmax, sweep.nyquist)

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_stamp = key.rsplit("/", 1)[-1].replace("/", "_")
    out_path = out_dir / f"nexrad_level2_{scan_stamp}_s{sweep.sweep_index}.png"

    fig, axes = plt.subplots(1, 5 if pyart_ref is not None else 4, figsize=(24 if pyart_ref is not None else 20, 4.5), constrained_layout=True)
    panels = []
    if sweep.reflectivity is not None:
        panels.append((sweep.reflectivity, "Reflectivity", "NWSRef"))
    else:
        panels.append((np.where(np.isfinite(sweep.velocity), 0.0, np.nan), "Gate mask", "gray"))
    panels.extend(
        [
            (sweep.velocity, "Observed velocity", "RdBu_r"),
            (zw.velocity, "open_dealias zw06", "RdBu_r"),
            (xu.velocity, "open_dealias xu11", "RdBu_r"),
        ]
    )
    if pyart_ref is not None:
        panels.append((pyart_ref, "Py-ART region-based", "RdBu_r"))

    for ax, (field, title, cmap) in zip(np.atleast_1d(axes), panels):
        if title == "Reflectivity":
            image = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap, vmin=-20, vmax=70)
        else:
            image = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Range gate")
        ax.set_ylabel("Azimuth ray")
        fig.colorbar(image, ax=ax, shrink=0.8)

    plt.suptitle(
        f"{sweep.radar_id} {key.rsplit('/', 1)[-1]} sweep {sweep.sweep_index} "
        f"elev={sweep.elevation_deg:.2f} nyq={sweep.nyquist:.2f} m/s"
    )
    plt.savefig(out_path, dpi=150)

    print(f"archive: {archive_path}")
    print(f"key: {key}")
    print(f"sweep: {sweep.sweep_index} elevation={sweep.elevation_deg:.2f} nyquist={sweep.nyquist:.2f}")
    print(f"valid velocity gates: {int(np.isfinite(sweep.velocity).sum())}")
    print(f"zw06 changed gates: {_changed(zw.velocity, sweep.velocity)}")
    print(f"xu11 changed gates: {_changed(xu.velocity, sweep.velocity)}")
    if pyart_ref is not None:
        print(f"pyart changed gates: {_changed(pyart_ref, sweep.velocity)}")
        print(f"zw06 vs pyart MAE: {_mae(zw.velocity, pyart_ref):.3f} m/s")
        print(f"xu11 vs pyart MAE: {_mae(xu.velocity, pyart_ref):.3f} m/s")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
