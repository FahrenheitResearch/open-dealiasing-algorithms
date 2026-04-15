# Merge map

This repo was assembled from the audited candidate packages in this folder. This
file records what was kept and why.

## Final base

Primary docs base:

- `open-dealiasing-algorithms-central-best.zip`

Primary code base:

- `open-dealias-lib.zip`

Reason:

- `central-best` had the strongest documentation spine, the best conservative
  language around proprietary apps, and a clean starter-repo layout.
- `open-dealias-lib` had the stronger executable surface: a real `open_dealias`
  package, typed result containers, clearer paper-based API names, a passing
  Python test suite, and a TypeScript port.

## Files and sections kept from each source

### From `open-dealias-lib`

Kept as the core executable library:

- `open_dealias/_core.py`
- `open_dealias/types.py`
- `open_dealias/continuity.py`
- `open_dealias/multipass.py`
- `open_dealias/vad.py`
- `open_dealias/fourdd.py`
- `open_dealias/synthetic.py`
- `open_dealias/__init__.py`
- `tests/test_open_dealias.py`
- `js/open_dealias.ts`
- `js/open_dealias.test.mjs`
- `js/open_dealias.test.ts`
- `js/package.json`
- `js/tsconfig.json`
- `examples/python_demo.py`
- `examples/js_demo.mjs`
- `docs/paper_map.md`

Why:

- better API discipline
- direct paper-family naming
- stronger array-first portability
- better starting point for Rust, PyO3, or WASM later

### From `open-dealiasing-algorithms-central-best`

Kept as the documentation and evaluation spine:

- `docs/algorithm-taxonomy.md`
- `docs/app-comparison-framework.md`
- `docs/bibliography.md`
- `docs/implementation-notes.md`
- `docs/test-cases.md`
- `EVALUATION.md`
- `REPO_SELECTION.md`

Adapted into this final repo:

- root `README.md`
- plotting demos in `examples/`

Why:

- strongest conservative treatment of proprietary-app claims
- best repo-level narrative
- useful evaluation framing and black-box stress-test guidance

## Selective ideas pulled from the numbered candidates

The numbered ZIPs were treated as donors, not primary bases.

- `(3)` informed the emphasis on test-backed code and explicit stress-test
  language.
- `(2)` reinforced packaging discipline and the split between live-app and
  research-grade expectations.
- `(4)` reinforced the narrow wording around GR2 and other proprietary apps.
- `(1)` contributed older example framing but no primary code.
- `(5)` added no material value over `(4)`.

## Final edits added on top of the donors

This merged repo also adds:

- `docs/likely_papers_from_uploads.md`
- this `MERGE_MAP.md`
- `requirements.txt`
- `examples/README.md`
- `.github/workflows/ci.yml`

It also updates:

- `README.md` to describe the merged repo accurately
- `pyproject.toml` to expose a coherent project surface
- example scripts so they target `open_dealias`
- the synthetic helper typing in `open_dealias/synthetic.py`
- the Python test suite with extra reference scenarios

## What was intentionally not kept

Not carried into the final repo:

- donor cache directories such as `__pycache__` and `.pytest_cache`
- duplicate educational Python packages that overlapped the `open_dealias`
  modules
- any wording that implied certainty about closed-source app internals
- any synthetic-only claims presented as operational proof
