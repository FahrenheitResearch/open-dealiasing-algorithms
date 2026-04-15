from __future__ import annotations

from datetime import date, datetime, timezone

from open_dealias import build_nexrad_prefix, parse_nexrad_key, select_nexrad_key


def test_build_nexrad_prefix_uses_archive_layout():
    assert build_nexrad_prefix("klot", date(2026, 4, 15)) == "2026/04/15/KLOT/"


def test_parse_nexrad_key_extracts_radar_and_time():
    parsed = parse_nexrad_key("2026/04/15/KLOT/KLOT20260415_153343_V06")
    assert parsed["radar_id"] == "KLOT"
    assert parsed["scan_time"] == datetime(2026, 4, 15, 15, 33, 43, tzinfo=timezone.utc)
    assert parsed["suffix"] == "V06"


def test_select_nexrad_key_prefers_latest_before_target():
    keys = [
        "2026/04/15/KLOT/KLOT20260415_150000_V06",
        "2026/04/15/KLOT/KLOT20260415_151500_V06",
        "2026/04/15/KLOT/KLOT20260415_153000_V06",
    ]
    target = datetime(2026, 4, 15, 15, 20, 0, tzinfo=timezone.utc)
    selected = select_nexrad_key(keys, target_time=target)
    assert selected.endswith("151500_V06")


def test_select_nexrad_key_latest_returns_last_scan():
    keys = [
        "2026/04/15/KLOT/KLOT20260415_150000_V06",
        "2026/04/15/KLOT/KLOT20260415_151500_V06",
        "2026/04/15/KLOT/KLOT20260415_153000_V06",
    ]
    selected = select_nexrad_key(keys, latest=True)
    assert selected.endswith("153000_V06")
