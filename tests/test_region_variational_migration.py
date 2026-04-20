from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.benchmark_support import discover_archive_files
from examples.region_variational_migration_benchmark import (
    _time_case,
    build_synthetic_cases,
)


def test_synthetic_cases_are_well_formed_and_nontrivial():
    cases = build_synthetic_cases(crop_half_size=18)

    assert len(cases) == 2
    for case in cases:
        observed = np.asarray(case["observed"], dtype=float)
        reference = np.asarray(case["reference"], dtype=float)
        truth = np.asarray(case["truth"], dtype=float)

        assert observed.shape == reference.shape == truth.shape
        assert np.isfinite(case["nyquist"])
        assert np.any(np.isfinite(observed))
        assert np.any(np.isfinite(reference))
        assert np.any(np.isfinite(truth))


def test_region_variational_pair_keeps_python_parity_on_synthetic_case():
    case = build_synthetic_cases(crop_half_size=18)[0]
    result = _time_case(case)

    assert result["region_graph"]["parity"]["velocity_mae"] == 0.0
    assert result["region_graph"]["parity"]["folds_equal"] is True
    assert result["variational"]["parity"]["velocity_mae"] == 0.0
    assert result["variational"]["parity"]["folds_equal"] is True
    assert result["region_graph"]["metadata"]["region_count"] > 0
    assert result["variational"]["metadata"]["iterations_used"] >= 1
    assert result["variational"]["metadata"]["bootstrap_method"] == "2d_multipass"


def test_local_archive_data_is_discoverable_when_present():
    archives = discover_archive_files(Path("archive_data"))
    if not archives:
        pytest.skip("no local archive_data cases available")

    assert all(path.suffix.lower() == ".zip" for path in archives)
    assert len(archives) >= 1
