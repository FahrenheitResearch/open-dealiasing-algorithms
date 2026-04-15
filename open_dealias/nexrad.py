from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .types import RadarSweep


NEXRAD_LEVEL2_BUCKET = "unidata-nexrad-level2"
_KEY_RE = re.compile(
    r"(?P<radar>[A-Z0-9]{4})(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<suffix>[^./]+))?(?:\..+)?$"
)


def _utc_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as exc:
        raise ImportError("boto3 and botocore are required for NEXRAD archive access") from exc
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def build_nexrad_prefix(radar_id: str, when: date | datetime) -> str:
    radar = radar_id.upper()
    day = when.date() if isinstance(when, datetime) else when
    return f"{day:%Y/%m/%d}/{radar}/"


def parse_nexrad_key(key: str) -> dict[str, object]:
    name = key.rsplit("/", 1)[-1]
    match = _KEY_RE.match(name)
    if match is None:
        raise ValueError(f"unrecognized NEXRAD key format: {key}")
    scan_time = datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    return {
        "key": key,
        "name": name,
        "radar_id": match.group("radar"),
        "scan_time": scan_time,
        "suffix": match.group("suffix"),
    }


def select_nexrad_key(
    keys: Iterable[str],
    target_time: datetime | None = None,
    *,
    latest: bool = False,
    prefer_before: bool = True,
) -> str:
    parsed = [(key, parse_nexrad_key(key)["scan_time"]) for key in keys]
    if not parsed:
        raise ValueError("no NEXRAD keys were supplied")
    if latest or target_time is None:
        return max(parsed, key=lambda item: item[1])[0]

    target = _utc_datetime(target_time)
    if prefer_before:
        before = [item for item in parsed if item[1] <= target]
        if before:
            return max(before, key=lambda item: item[1])[0]
    return min(parsed, key=lambda item: abs((item[1] - target).total_seconds()))[0]


def list_nexrad_keys(
    radar_id: str,
    when: date | datetime,
    *,
    bucket: str = NEXRAD_LEVEL2_BUCKET,
) -> list[str]:
    client = _s3_client()
    prefix = build_nexrad_prefix(radar_id, when)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            if not name or "MDM" in name:
                continue
            keys.append(key)
    return sorted(keys)


def find_nexrad_key(
    radar_id: str,
    target_time: datetime | None = None,
    *,
    latest: bool = False,
    bucket: str = NEXRAD_LEVEL2_BUCKET,
    lookback_days: int = 2,
) -> str:
    target = _utc_datetime(target_time)
    days: list[date] = [target.date() - timedelta(days=offset) for offset in range(max(1, lookback_days))]
    keys: list[str] = []
    for day in days:
        keys.extend(list_nexrad_keys(radar_id, day, bucket=bucket))
        if latest and keys:
            return select_nexrad_key(keys, latest=True)
    if not keys:
        raise FileNotFoundError(f"no Level II archives found for radar {radar_id.upper()}")
    return select_nexrad_key(keys, target_time=target, latest=latest, prefer_before=True)


def download_nexrad_key(
    key: str,
    out_dir: str | Path = ".cache/nexrad",
    *,
    bucket: str = NEXRAD_LEVEL2_BUCKET,
    overwrite: bool = False,
) -> Path:
    client = _s3_client()
    out_path = Path(out_dir) / key.rsplit("/", 1)[-1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not out_path.exists():
        client.download_file(bucket, key, str(out_path))
    return out_path


def _masked_to_nan(data: np.ndarray) -> np.ndarray:
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=float)


def _auto_sweep_index(radar, velocity_field: str) -> int:
    counts: list[tuple[int, int]] = []
    for sweep in range(radar.nsweeps):
        arr = _masked_to_nan(radar.get_field(sweep, velocity_field))
        count = int(np.isfinite(arr).sum())
        if count:
            counts.append((count, sweep))
    if not counts:
        raise ValueError(f"field {velocity_field!r} has no valid sweeps")
    counts.sort()
    return counts[-1][1]


def load_nexrad_sweep(
    archive_path: str | Path,
    *,
    sweep: int | str = "auto",
    velocity_field: str = "velocity",
    reflectivity_field: str = "reflectivity",
) -> RadarSweep:
    try:
        import pyart
    except ImportError as exc:
        raise ImportError("Py-ART is required to read NEXRAD Level II archives") from exc

    path = Path(archive_path)
    radar = pyart.io.read_nexrad_archive(str(path))
    sweep_index = _auto_sweep_index(radar, velocity_field) if sweep == "auto" else int(sweep)

    velocity = _masked_to_nan(radar.get_field(sweep_index, velocity_field))
    reflectivity = None
    if reflectivity_field in radar.fields:
        reflectivity = _masked_to_nan(radar.get_field(sweep_index, reflectivity_field))

    azimuth_deg = np.asarray(radar.get_azimuth(sweep_index), dtype=float)
    range_m = np.asarray(radar.range["data"], dtype=float)
    nyquist_raw = np.asarray(radar.get_nyquist_vel(sweep_index), dtype=float).reshape(-1)
    nyquist = float(nyquist_raw[0])
    scan_time = None
    key = None
    try:
        parsed = parse_nexrad_key(path.name)
        scan_time = parsed["scan_time"]
        key = parsed["key"]
    except ValueError:
        pass

    radar_id = str(radar.metadata.get("instrument_name") or path.name[:4]).upper()
    return RadarSweep(
        radar_id=radar_id,
        sweep_index=sweep_index,
        azimuth_deg=azimuth_deg,
        elevation_deg=float(radar.fixed_angle["data"][sweep_index]),
        range_m=range_m,
        nyquist=nyquist,
        velocity=velocity,
        reflectivity=reflectivity,
        scan_time=scan_time,
        key=key,
        local_path=str(path),
        metadata={
            "nsweeps": int(radar.nsweeps),
            "nrays": int(radar.nrays),
            "ngates": int(radar.ngates),
            "available_fields": list(radar.fields.keys()),
            "velocity_field": velocity_field,
            "reflectivity_field": reflectivity_field if reflectivity is not None else None,
        },
    )
