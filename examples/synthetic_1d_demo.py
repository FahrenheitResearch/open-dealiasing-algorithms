from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_dealias import dealias_radial_es90, wrap_to_nyquist
from open_dealias.synthetic import make_smooth_radial


def main() -> None:
    nyquist = 12.0
    gates = np.arange(320)
    true, observed = make_smooth_radial(n_gates=gates.size, nyquist=nyquist, seed=7)
    true = true + np.linspace(-18.0, 18.0, gates.size)
    observed = wrap_to_nyquist(true, nyquist)

    observed[105:109] = np.nan
    observed[220:223] = np.nan

    result = dealias_radial_es90(observed, nyquist, max_gap=4)

    out_dir = REPO_ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_1d_demo.png"

    plt.figure(figsize=(10, 5))
    plt.plot(gates, true, label="true velocity")
    plt.plot(gates, observed, label="aliased observed")
    plt.plot(gates, result.velocity, label="dealiased reference")
    plt.xlabel("Gate index")
    plt.ylabel("Velocity (m s$^{-1}$)")
    plt.title("Synthetic 1D radial continuity example")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
