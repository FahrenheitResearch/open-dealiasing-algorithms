from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_dealias import build_reference_from_uv, dealias_sweep_xu11, dealias_sweep_zw06, wrap_to_nyquist


if __name__ == "__main__":
    azimuth_deg = np.linspace(0.0, 360.0, 180, endpoint=False)
    azimuth_rad = np.deg2rad(azimuth_deg)[:, None]
    range_axis = np.linspace(-1.0, 1.0, 80)[None, :]
    reference = build_reference_from_uv(azimuth_deg, 80, u=18.0, v=-3.0)
    truth = reference + 3.0 * np.sin(2.0 * azimuth_rad) * np.exp(-((range_axis - 0.10) ** 2) / 0.08)
    observed = wrap_to_nyquist(truth, 10.0)

    res_2d = dealias_sweep_zw06(observed, 10.0)
    res_vad = dealias_sweep_xu11(observed, 10.0, azimuth_deg)
    res_ref = dealias_sweep_zw06(observed, 10.0, reference=reference)

    mae_2d = np.mean(np.abs(res_2d.velocity - truth))
    mae_vad = np.mean(np.abs(res_vad.velocity - truth))
    mae_ref = np.mean(np.abs(res_ref.velocity - truth))

    print(f"2D multipass MAE:         {mae_2d:.3f} m/s")
    print(f"Reference-anchored MAE:   {mae_ref:.3f} m/s")
    print(f"VAD-seeded + refine MAE:  {mae_vad:.3f} m/s")
    print("Metadata:", res_vad.metadata)
