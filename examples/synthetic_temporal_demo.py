from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_dealias import dealias_sweep_jh01, wrap_to_nyquist


def main() -> None:
    nyquist = 10.0
    n_az = 80
    n_rng = 120

    az = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)[:, None]
    rg = np.linspace(-1.0, 1.0, n_rng)[None, :]

    previous = 8.0 * np.sin(az) + 7.0 * rg + 5.0 * np.exp(-((rg - 0.2) ** 2) / 0.02) * np.cos(2.0 * az)
    current_true = np.roll(previous, shift=2, axis=0)
    current_true = np.roll(current_true, shift=3, axis=1)
    current_true[:, :3] = np.nan

    observed = wrap_to_nyquist(current_true, nyquist)
    observed[15:20, 44:50] = np.nan
    observed[46:50, 70:78] = np.nan

    result = dealias_sweep_jh01(
        observed,
        nyquist,
        previous_corrected=previous,
        shift_az=2,
        shift_range=3,
    )

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_temporal_demo.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        (observed, "Aliased current"),
        (result.velocity, "Temporal continuity"),
        (current_true, "True current"),
    ]

    for ax, (field, title) in zip(axes, panels):
        image = ax.imshow(field, aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xlabel("Range gate")
        ax.set_ylabel("Azimuth ray")
        fig.colorbar(image, ax=ax, shrink=0.8)

    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
