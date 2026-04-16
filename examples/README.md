# Examples

This repo ships two kinds of examples:

- quick numeric smoke tests:
  - `python_demo.py`
  - `js_demo.mjs`
- real-data archive demo:
  - `nexrad_level2_demo.py`
  - `klot_same_scan_benchmark.py`
  - `moore_fullscan_speed_report.py`
- plotting demos that write PNGs into `examples/output/`:
  - `synthetic_1d_demo.py`
  - `synthetic_2d_demo.py`
  - `synthetic_temporal_demo.py`

The plotting demos are synthetic by design. They are meant to show branch
selection, reference anchoring, and temporal stabilization clearly, not to claim
operational skill on real archived radar cases.

Suggested run order:

```bash
python examples/python_demo.py
python examples/nexrad_level2_demo.py --radar KLOT
python examples/klot_same_scan_benchmark.py
python examples/moore_fullscan_speed_report.py
python examples/synthetic_1d_demo.py
python examples/synthetic_2d_demo.py
python examples/synthetic_temporal_demo.py
```

For the JS demo:

```bash
cd js
npm install
npm test
cd ..
node examples/js_demo.mjs
```

The real-data demo pulls a public Level II archive from the current
`unidata-nexrad-level2` bucket, converts one sweep into plain NumPy arrays, runs
the open dealiasers, and optionally compares them to Py-ART's open region-based
solver. That is a case study / regression path, not a claim of full benchmark
coverage.

The fixed KLOT benchmark uses one target archive and one previous archive, then
scores the methods that can be run fairly on that case against Py-ART on the
same target sweep. Temporal methods get prior-state anchors from the previous
archive rather than from Py-ART, and methods that would need synthetic or
leaky support data are marked skipped with a recorded reason. Treat that output
as a scoped regression test on one archive pair, not a full evaluation of the
method families.

The Moore full-scan speed report measures prepared solver entry points under
package-wide `auto` and package-wide `python` backend policy on the
`KTLX20130520_200356_V06.gz` tornado case and writes a JSON summary into
`examples/output/`. It is a useful migration/regression speed check, but not a
universal latency claim across all scan types or a full archive-to-product wall
clock benchmark.

For a faster day-to-day loop, use the quick path:

```bash
python examples/moore_fullscan_speed_report.py --skip-volume --repeats 1 --warmup 0
```

That skips the previous-volume methods and keeps the report scoped to prepared
solver entry points. You can also limit the run further with
`--solvers build_velocity_qc_mask,es90,zw06,region_graph,recursive,variational`.
