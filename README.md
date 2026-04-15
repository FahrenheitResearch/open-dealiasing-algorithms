# open-dealiasing-algorithms

An open technical reference and runnable starter kit for weather radar Doppler
velocity dealiasing.

The point of this repo is simple: dealiasing should not be treated like
proprietary magic. Public papers, ROC documentation, Py-ART, and newer open
projects already expose the main algorithm families. This repo turns that public
trail into a coherent open-source base with code, docs, examples, tests, and a
TypeScript port.

## Why this repo exists

Radar users notice that different apps often show different dealiased velocity
fields. That does not automatically mean one app has secret physics and another
does not. Differences can come from:

- different quality control and masking before any unwrap step,
- different seed or background-wind choices,
- sweep-local versus temporal or volume-aware continuity,
- different unresolved-gate policies,
- different display interpolation or smoothing after dealiasing.

This repo is built to explain those differences without inventing internals for
closed-source products.

## What is implemented

The current Python package, `open_dealias`, exposes paper-mapped reference
implementations for the main public families that kept recurring across the
uploaded summaries and public references:

- `dealias_radial_es90`
  Eilts and Smith style radial continuity / local-environment logic
- `dealias_sweep_zw06`
  Jing and Wiener plus Zhang and Wang style sweep-wide 2D continuity
- `dealias_sweep_xu11`
  Xu style VAD-seeded dealiasing
- `dealias_sweep_jh01`
  James and Houze style temporal / 4DD-assisted sweep dealiasing
- `dealias_volume_jh01`
  a compact descending-elevation volume extension
- `dealias_sweep_region_graph`
  Py-ART-style dynamic-network / region-graph sweep solver
- `dealias_sweep_recursive`
  R2D2-style recursive region refinement
- `dealias_dual_prf`
  dual-PRF / staggered-PRT paired branch search
- `dealias_volume_3d`
  UNRAVEL-style modular 3D continuity
- `dealias_sweep_variational`
  variational / global coordinate-descent branch solver
- `dealias_sweep_ml`
  lightweight ML-assisted branch predictor with variational refinement

These are not claims of bit-for-bit parity with any vendor or government
production chain. They are compact, public, implementation-oriented versions of
the best-supported public method families.

## Public grounding

The strongest public trail behind this repo is:

- Eilts and Smith for local continuity
- Jing and Wiener plus Zhang and Wang for sweep-wide 2D methods
- Xu et al. for VAD/background-wind anchoring
- James and Houze for prior-volume / 4DD logic
- ROC material on VDA and 2DVDA
- Py-ART `dealias_region_based` and `dealias_fourdd`
- newer open descendants such as UNRAVEL and R2D2

See [docs/likely_papers_from_uploads.md](docs/likely_papers_from_uploads.md) and
[docs/paper_map.md](docs/paper_map.md).

## Repo layout

```text
open-dealiasing-algorithms/
|-- README.md
|-- EVALUATION.md
|-- REPO_SELECTION.md
|-- MERGE_MAP.md
|-- pyproject.toml
|-- requirements.txt
|-- docs/
|   |-- algorithm-taxonomy.md
|   |-- app-comparison-framework.md
|   |-- bibliography.md
|   |-- implementation-notes.md
|   |-- likely_papers_from_uploads.md
|   |-- paper_map.md
|   `-- test-cases.md
|-- open_dealias/
|   |-- __init__.py
|   |-- _core.py
|   |-- continuity.py
|   |-- dual_prf.py
|   |-- fourdd.py
|   |-- ml.py
|   |-- multipass.py
|   |-- nexrad.py
|   |-- qc.py
|   |-- recursive.py
|   |-- region_graph.py
|   |-- synthetic.py
|   |-- types.py
|   |-- vad.py
|   |-- variational.py
|   `-- volume3d.py
|-- examples/
|   |-- README.md
|   |-- js_demo.mjs
|   |-- klot_same_scan_benchmark.py
|   |-- nexrad_level2_demo.py
|   |-- python_demo.py
|   |-- synthetic_1d_demo.py
|   |-- synthetic_2d_demo.py
|   `-- synthetic_temporal_demo.py
|-- js/
|   |-- open_dealias.test.mjs
|   |-- open_dealias.ts
|   |-- package.json
|   `-- tsconfig.json
`-- tests/
    |-- test_dualprf_volume3d.py
    |-- test_nexrad_helpers.py
    |-- test_open_dealias.py
    |-- test_qc_variational_ml.py
    |-- test_reference_scenarios.py
    `-- test_region_graph_recursive.py
```

## Install

Python:

```bash
pip install -e .[dev]
```

For real Level II archive access:

```bash
pip install -e .[realdata]
```

TypeScript:

```bash
cd js
npm install
npm test
```

## Quick start

```python
import numpy as np
from open_dealias import dealias_sweep_xu11, dealias_sweep_zw06

observed = ...
nyquist = 10.0
azimuth_deg = np.linspace(0.0, 360.0, observed.shape[0], endpoint=False)

plain = dealias_sweep_zw06(observed, nyquist)
vad_seeded = dealias_sweep_xu11(observed, nyquist, azimuth_deg)
```

Common entry points:

```python
from open_dealias import dealias_radial_es90, dealias_sweep_es90
from open_dealias import dealias_sweep_zw06
from open_dealias import estimate_uniform_wind_vad, dealias_sweep_xu11
from open_dealias import dealias_sweep_jh01, dealias_volume_jh01
```

## Examples and tests

Quick smoke runs:

```bash
pytest -q
python examples/python_demo.py
python examples/nexrad_level2_demo.py --radar KLOT
python examples/klot_same_scan_benchmark.py
python examples/synthetic_1d_demo.py
python examples/synthetic_2d_demo.py
python examples/synthetic_temporal_demo.py
```

The plotting demos write PNGs into `examples/output/`.

## Real radar data

The repo now includes a real NEXRAD Level II bridge:

- `open_dealias.nexrad.list_nexrad_keys`
- `open_dealias.nexrad.find_nexrad_key`
- `open_dealias.nexrad.download_nexrad_key`
- `open_dealias.nexrad.load_nexrad_sweep`

That path uses the currently accessible public `unidata-nexrad-level2` archive
bucket and converts a real archive sweep into plain NumPy arrays that match the
rest of the package API.

The example below defaults to the latest available scan for the requested radar:

```bash
python examples/nexrad_level2_demo.py --radar KLOT
```

For a specific archive time:

```bash
python examples/nexrad_level2_demo.py --radar KLOT --time 2026-04-15T15:33:00Z
```

For the fixed proof benchmark used in this repo:

```bash
python examples/klot_same_scan_benchmark.py
```

## Evidence policy

When this repo talks about a closed-source app, every claim should be read as
one of:

- `public documentation`
- `observable behavior / inference`
- `unknown / cannot verify`

That is deliberate. For products like GR2, RadarScope, RadarOmega, or the user
shorthand "Yalldar", the public algorithm details are often incomplete or absent.
This repo treats undocumented internals as unknown unless stronger evidence
exists.

## What is still missing

This is a strong open baseline, not a finished operational stack. It still lacks:

- real archived Level II benchmark manifests,
- a real ingest/output layer for radar formats,
- explicit unresolved-mask and QC pipelines,
- CI-backed comparison notebooks and figures,
- implementations of multi-PRF, global-optimization, and ML-assisted families,
- a Rust core with PyO3 and WASM bindings.

## Provenance

This final repo was built by auditing multiple candidate repos and a standalone
`open-dealias-lib` package. The docs spine comes from the stricter
`central-best` candidate, while the executable library core and JS port come
from `open-dealias-lib`.

See [REPO_SELECTION.md](REPO_SELECTION.md) for the candidate ranking and
[MERGE_MAP.md](MERGE_MAP.md) for the final merge map.

## License

MIT. See [LICENSE](LICENSE).
