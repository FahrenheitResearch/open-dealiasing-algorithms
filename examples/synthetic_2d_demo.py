from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_dealias import build_reference_from_uv, dealias_sweep_xu11, dealias_sweep_zw06, wrap_to_nyquist


def main() -> None:
    nyquist = 11.0
    azimuths = np.linspace(0.0, 360.0, 90, endpoint=False)
    gate_ranges = np.arange(140)

    reference = build_reference_from_uv(azimuths, gate_ranges.size, u=15.0, v=7.0, elevation_deg=0.5)

    az = np.deg2rad(azimuths)[:, None]
    rg = np.linspace(-1.0, 1.0, gate_ranges.size)[None, :]

    embedded = 12.0 * np.exp(-((rg - 0.10) ** 2) / 0.03) * np.sin(3.0 * az)
    shear_band = 8.0 * np.tanh((rg + 0.25) / 0.09)
    true = reference + embedded + shear_band

    observed = wrap_to_nyquist(true, nyquist)
    observed[18:24, 70:85] = np.nan
    observed[52:56, 30:34] = np.nan

    region = dealias_sweep_zw06(observed, nyquist)
    env = dealias_sweep_xu11(observed, nyquist, azimuths, external_reference=reference)

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_2d_demo.png"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        (true, "True velocity"),
        (observed, "Aliased observed"),
        (region.velocity, "2D multipass"),
        (env.velocity, "VAD + reference seeded"),
    ]

    for ax, (field, title) in zip(axes.flat, panels):
        image = ax.imshow(field, aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("Range gate")
        ax.set_ylabel("Azimuth ray")
        fig.colorbar(image, ax=ax, shrink=0.8)

    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
