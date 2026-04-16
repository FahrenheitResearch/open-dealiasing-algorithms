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
uploaded summaries and public references. The names below are family labels,
not claims that each function is a standalone, paper-faithful solver.

Core solvers and core solver families:

- `dealias_radial_es90`
  Eilts and Smith style radial continuity / local-environment logic
- `dealias_sweep_zw06`
  Jing and Wiener plus Zhang and Wang style sweep-wide 2D continuity
- `dealias_sweep_region_graph`
  Py-ART-style dynamic-network / region-graph sweep solver
- `dealias_sweep_recursive`
  R2D2-style recursive region refinement
- `dealias_dual_prf`
  dual-PRF / staggered-PRT paired branch search
- `dealias_sweep_variational`
  variational / global coordinate-descent branch solver

Anchor and refinement compositions:

- `dealias_sweep_xu11`
  Xu-style VAD-seeded composition: fit a background wind, unwrap toward it,
  then optionally refine with multipass continuity
- `dealias_sweep_jh01`
  James and Houze style temporal composition for a sweep
- `dealias_volume_jh01`
  descending-elevation volume composition built around temporal anchors
- `dealias_volume_3d`
  UNRAVEL-style multi-sweep composition built from sweep-level continuity
- `dealias_sweep_ml`
  lightweight ML-assisted branch predictor with optional refinement

These are compact, public, implementation-oriented versions of the best
supported public method families. They are not claims of bit-for-bit parity
with any vendor or government production chain, and several of them deliberately
compose a seed or cleanup step around a smaller core solver.

Every `DealiasResult` now also carries a `result_state` summary with resolved /
unresolved counts, coverage, fill policy, and lightweight provenance metadata.
That is still a conservative reference-layer reporting model, not a calibrated
operational uncertainty product.

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
|-- Cargo.toml
|-- pyproject.toml
|-- setup.py
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
|   |-- _rust_bridge.py
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
|   |-- benchmark_support.py
|   |-- js_demo.mjs
|   |-- klot_same_scan_benchmark.py
|   |-- moore_fullscan_speed_report.py
|   |-- nexrad_level2_demo.py
|   |-- python_demo.py
|   |-- region_variational_migration_benchmark.py
|   |-- synthetic_1d_demo.py
|   |-- synthetic_2d_demo.py
|   `-- synthetic_temporal_demo.py
|-- js/
|   |-- open_dealias.test.mjs
|   |-- open_dealias.ts
|   |-- package.json
|   `-- tsconfig.json
|-- rust/
|   |-- open_dealias_core/
|   `-- open_dealias_py/
`-- tests/
    |-- test_benchmark_support.py
    |-- test_dualprf_volume3d.py
    |-- test_fourdd_rust.py
    |-- test_ml_rust.py
    |-- test_nexrad_helpers.py
    |-- test_open_dealias.py
    |-- test_qc_variational_ml.py
    |-- test_reference_scenarios.py
    |-- test_rust_core.py
    |-- test_region_graph_recursive.py
    `-- test_vad_rust.py
```

## Install

Python:

```bash
pip install -e .[dev]
```

That editable install now builds the Rust extension when a Rust toolchain is
available. "Rust-backed" currently means the native extension can replace the
lowest-level hot-path helpers in `open_dealias._core` and any public solver
entry point that explicitly dispatches to it. Higher-level composition,
benchmark orchestration, I/O, plotting, and fallback glue remain Python. In
practice, that means some solver families are partly native-backed and some are
still fully Python-composed today.

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
python examples/moore_fullscan_speed_report.py
python examples/synthetic_1d_demo.py
python examples/synthetic_2d_demo.py
python examples/synthetic_temporal_demo.py
```

The plotting demos write PNGs into `examples/output/`.

## Rust backend

This repo now includes a Rust migration under `rust/`:

- `rust/open_dealias_core`
  pure Rust implementations of the low-level array helpers and the solver
  entry points that have been ported so far
- `rust/open_dealias_py`
  a PyO3 extension that exposes native-backed helpers and solver entry points
  to Python
- `open_dealias/_rust_bridge.py`
  runtime detection of the native extension

The exact native coverage is intentionally described at the function level in
the Python modules and tests. Some solver families are native-backed end to
end, others still call native helpers from Python composition, and others are
still pure Python.

For a current prepared solver-stack speed check on a real tornado case:

```bash
python examples/moore_fullscan_speed_report.py
```

For a faster iteration path while keeping the measurement honest:

```bash
python examples/moore_fullscan_speed_report.py --skip-volume --repeats 1 --warmup 0
```

The checked Moore-case report in `examples/output/` is a single-case regression
signal on `KTLX20130520_200356_V06.gz`. It times prepared solver entrypoints
under package-wide `auto` and package-wide `python` policy, so it is useful for
native-vs-fallback regression and not a full archive-to-product wall-clock
benchmark for the full space of storm modes, QC settings, or scan strategies.

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

For the real-data reference-consistency benchmark used in this repo:

```bash
python examples/klot_same_scan_benchmark.py
```

That benchmark compares scored methods against Py-ART as an open reference,
but it does not train on Py-ART targets or synthesize missing support data. If a
method cannot be run fairly on the chosen case, the script marks it skipped and
records the reason.

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
- stronger unresolved-mask policies and richer QC pipelines beyond the current first-class `result_state` reporting,
- CI-backed comparison notebooks and figures,
- WASM / JS bindings on top of the Rust core.

Several high-level solvers also still favor "fill the field" behavior over a
fully conservative unresolved-state model. That is a design choice in this repo
today, not a claim that the unresolved problem is solved.

## Provenance

This final repo was built by auditing multiple candidate repos and a standalone
`open-dealias-lib` package. The docs spine comes from the stricter
`central-best` candidate, while the executable library core and JS port come
from `open-dealias-lib`.

See [REPO_SELECTION.md](REPO_SELECTION.md) for the candidate ranking and
[MERGE_MAP.md](MERGE_MAP.md) for the final merge map.

## License

MIT. See [LICENSE](LICENSE).
