# Algorithm taxonomy for Doppler velocity dealiasing

This document classifies the public dealiasing families used in this repo and
separates **core solvers** from **anchor/refinement compositions**. That
distinction matters because many practical algorithms are not a single unwrap
routine. They are a stack of QC, seeding, reference construction, main solve,
and cleanup.

## 1. The problem

Weather radar reports radial velocity only within an unambiguous interval
bounded by the Nyquist velocity \(V_N\). If the true value exceeds that range,
the observed velocity is wrapped by an integer multiple of \(2V_N\):

```text
v_true = v_obs + 2 k V_N
```

for some integer fold count `k`.

Dealiasing is the problem of choosing the most plausible `k` at each gate.

## 2. What makes dealiasing hard

The arithmetic is simple. The ambiguity is not.

Real cases contain:

- missing gates,
- low-SNR or noisy gates,
- clutter and biological contamination,
- strong shear,
- disconnected echo islands,
- changing geometry across sweeps or volumes,
- multiple plausible reference states.

So dealiasing is better viewed as a constrained inference problem than as a
single arithmetic correction.

## 3. Taxonomy rule: core solver vs composition

Use these labels when reading the repo:

- **Core solver**: decides fold counts directly from the current sweep or
  volume.
- **Anchor provider**: estimates a reference field but does not by itself solve
  every ambiguity.
- **Refinement pass**: adjusts an already-bootstrapped field.
- **Hybrid composition**: combines several of the above into one public API.

In this repo, several functions are hybrid compositions even though their names
look like single methods.

## 4. Core solver families

### A. Gate-to-gate continuity / 1D radial unwrap

Core idea:

Choose a seed on one radial, then walk outward or inward, keeping each next
gate as close as possible to the previous corrected gate.

Strengths:

- fast,
- easy to reason about,
- good as a cleanup pass.

Weaknesses:

- brittle after a bad seed,
- weak across gaps,
- poor for disconnected regions.

Public examples:

- Ray and Ziegler (1977),
- Eilts and Smith (1990).

Repo mapping:

- `dealias_radial_es90`

### B. Sweep-wide 2D continuity

Core idea:

Treat the sweep as a 2D azimuth-range field and propagate fold choices through
local neighborhoods instead of one radial at a time.

Strengths:

- uses azimuthal context,
- usually more stable than 1D continuity,
- handles organized convection better.

Weaknesses:

- still sensitive to seed quality,
- can still fail on disconnected or cluttered regions,
- can smooth away real extremes if QC is too permissive.

Public examples:

- Jing and Wiener (1993),
- Zhang and Wang (2006),
- Py-ART `dealias_region_based`.

Repo mapping:

- `dealias_sweep_zw06`
- `dealias_sweep_region_graph`

### C. Region graph / dynamic network / recursive region solvers

Core idea:

Partition the sweep into regions, assign each region a fold, then propagate
consensus through the region graph or recurse on ambiguous subregions.

Strengths:

- cleaner handling of coherent areas,
- more global than a gate-by-gate walk,
- often better failure containment than a naive sweep walk.

Weaknesses:

- region boundaries matter,
- coarse partitioning can hide detail,
- still depends on seeding and QC.

Public examples:

- Py-ART `dealias_region_based`,
- R2D2,
- UNRAVEL-style region or volume continuity.

Repo mapping:

- `dealias_sweep_region_graph`
- `dealias_sweep_recursive`

### D. Variational / global objective solvers

Core idea:

Choose fold counts by minimizing a score over neighbors, references, and
smoothness terms. This is closer to a global optimization than to a local walk.

Strengths:

- flexible objective function,
- can encode both local and reference penalties,
- useful when the field is noisy but still structured.

Weaknesses:

- score design matters a lot,
- can converge to a locally consistent but globally wrong branch,
- more expensive than simple continuity.

Repo mapping:

- `dealias_sweep_variational`

### E. Dual-PRF / staggered-PRT support

Important caveat:

This is partly an acquisition strategy, not just a postprocessing solver.
Without the paired PRF data, a downstream app cannot invent the missing
information.

Repo mapping:

- `dealias_dual_prf`

## 5. Anchor providers and compositions

These are not pure solvers. They produce a reference or a first guess, then
hand that off to a core solver or cleanup pass.

### A. Environmental wind / VAD / sounding anchors

Core idea:

Estimate a background wind profile, then choose folds that place the sweep
closest to that reference.

Strengths:

- useful for sparse or disconnected echoes,
- can break global ambiguity,
- often good as a first guess.

Weaknesses:

- dangerous when the background wind is unrepresentative,
- can over-smooth true local extremes,
- depends on support data that may be unavailable.

Repo mapping:

- `estimate_uniform_wind_vad` (anchor provider),
- `dealias_sweep_xu11` (anchor + refinement composition).

### B. Temporal / previous-volume anchoring

Core idea:

Use a prior dealiased sweep or volume as a reference for the current scan.

Strengths:

- stabilizes slow evolution,
- helps maintain continuity between scans,
- useful when geometry stays comparable.

Weaknesses:

- can propagate previous mistakes,
- weaker after geometry changes,
- not safe as an unquestioned truth source.

Repo mapping:

- `dealias_sweep_jh01`
- `dealias_volume_jh01`

### C. Multi-stage sweep or volume compositions

Core idea:

Run a bootstrap solver, then a cleanup solver, then maybe a temporal or volume
pass. These methods are useful operationally but should be described as
compositions.

Repo mapping:

- `dealias_sweep_xu11`
- `dealias_sweep_jh01`
- `dealias_volume_jh01`
- `dealias_volume_3d`
- `dealias_sweep_ml`

## 6. Where QC fits

QC is not an afterthought. It is part of the solver contract.

Important QC layers:

- reflectivity or echo support,
- velocity texture,
- clutter and non-meteorological masks,
- gate-level or ray-level validity,
- unresolved-gate policy.

Practical rule:

It is usually better to keep a gate unresolved than to let it seed a large bad
region.

## 7. What "Rust-backed" means in this repo

The docs use "Rust-backed" in a narrow sense:

- some low-level array helpers have native implementations,
- some public solver entry points can dispatch to a native backend when it is
  available,
- Python still owns a lot of composition, orchestration, I/O, plotting, and
  fallback behavior.

So "Rust-backed" does **not** mean "every solver is fully native and
independent." It means the hot path has native support where the code currently
dispatches to it.

## 8. Practical comparison guidance

When comparing methods, report which layer you are testing:

- core solver only,
- anchor-assisted composition,
- refinement/cleanup pass,
- full hybrid pipeline.

Do not compare a hybrid pipeline against a bare core solver as if they were the
same class of algorithm.

## 9. Summary table

| Family | What it is | Repo examples | Typical role |
| --- | --- | --- | --- |
| 1D continuity | Core solver | `dealias_radial_es90` | Fast local unwrap |
| Sweep-wide 2D continuity | Core solver | `dealias_sweep_zw06` | Main sweep solver |
| Region graph / recursive | Core solver | `dealias_sweep_region_graph`, `dealias_sweep_recursive` | Better regional context |
| Variational objective | Core solver | `dealias_sweep_variational` | Global-ish score minimization |
| Dual-PRF / staggered-PRT | Support-data driven solver | `dealias_dual_prf` | Acquisition-side ambiguity reduction |
| VAD / wind anchor | Anchor provider | `estimate_uniform_wind_vad` | Background reference |
| VAD-seeded composition | Hybrid composition | `dealias_sweep_xu11` | Anchor + sweep solve |
| Temporal / 4D | Hybrid composition | `dealias_sweep_jh01`, `dealias_volume_jh01` | Prior-state stabilization |
| Multi-sweep continuity | Hybrid composition | `dealias_volume_3d` | Volume-level refinement |
| ML-assisted | Hybrid composition | `dealias_sweep_ml` | Assistive branch prediction |

## 10. Bottom line

The strongest open stack is usually a hybrid:

1. conservative QC,
2. a reasonable anchor if one exists,
3. a core sweep solver,
4. targeted cleanup,
5. explicit unresolved handling.

That is more honest than pretending there is one universal dealiasing trick.
