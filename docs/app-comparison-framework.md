# Framework for comparing radar apps without inventing internals

This document is intentionally cautious. The point is to make **black-box app comparisons more rigorous** without pretending that undocumented internals are known.

## 1. Why comparison is tricky

A radar app can display a dealiased velocity field without telling you:

- whether the algorithm is purely spatial or also temporal,
- what QC gates were removed before dealiasing,
- whether environmental winds or prior volumes are used,
- what happens to unresolved gates,
- whether additional interpolation or smoothing is applied after dealiasing.

As a result, output differences do **not** map one-to-one to algorithm names.

## 2. Evidence labels

Every statement about an app or processing chain should be tagged as one of:

### public documentation

Backed by public manuals, papers, API docs, official product descriptions, or open-source code.

### observable behavior / hypothesis

A careful inference from repeated output behavior. Useful, but still an inference.

### unknown / cannot verify

The correct label when internals are not publicly documented and output behavior is not enough to justify a stronger claim.

## 3. What can be inferred from outputs

You can often infer broad behavior patterns, such as:

- whether the system is conservative and leaves ambiguous gates blank,
- whether temporal continuity seems to be used,
- whether isolated far-range patches get anchored by a background field,
- whether dealiasing appears sweep-local or more volume-aware,
- whether QC is aggressively masking noisy gates before unwrap,
- whether a derived product like storm-relative velocity is more stable than base velocity alone.

These are **method-family clues**, not proofs of implementation.

## 4. What cannot be inferred from outputs alone

You generally cannot infer:

- exact cost functions,
- exact thresholds,
- code lineage,
- whether the method is a clone of a specific paper,
- whether an app uses 2DVDA, 4DD, or another named algorithm internally,
- whether a model, sounding, or previous volume was used behind the scenes,
- whether a learned model is involved,
- how many internal passes or cleanup stages were applied.

A visually similar result is not enough to claim algorithm identity.

## 5. Why two apps can look different even if both “dealias”

Differences can come from any combination of:

- different prefilters and gate masks,
- different seed selection,
- different trust in background winds,
- different tolerance for high-shear boundaries,
- different unresolved-gate policies,
- different smoothing or display interpolation,
- different handling of scan restarts or playback state.

This means an app comparison must measure the **whole pipeline**, not just the final unwrap.

## 6. Recommended black-box test design

### A. Use the same archived source data

Use archived Level II or equivalent radial data for the same radar, same time, same elevation, and same scan strategy.

### B. Lock down display settings

Keep color tables, smoothing, interpolation, storm-motion assumptions, and product types as consistent as possible.

### C. Test base velocity and derived velocity separately

A system may show a reasonable storm-relative velocity even if the base dealiased velocity differs because later steps compensate.

### D. Include cold-start and warm-state tests

Start playback from the exact target volume and also from several volumes earlier. Differences can expose temporal continuity or initialization dependence.

### E. Record unresolved-gate behavior

Count how often an app chooses to blank a gate rather than force a fold. Conservative no-data handling can be a feature, not a bug.

### F. Separate acquisition differences from post-processing differences

If one source uses multi-PRF or staggered-PRT-derived inputs and another does not, note that clearly.

### G. Evaluate by case family

Do not judge an algorithm on one pretty example. Use at least:

- supercell / mesocyclone,
- tropical cyclone,
- QLCS shear zone,
- weak sparse echoes,
- noisy low-SNR environment,
- near-range strong couplet,
- disconnected far-range echoes.

### H. Run named black-box stress tests

A small repeatable stress-test set is often more informative than a large pile of screenshots. Useful named tests include:

- **cold-start versus warm-start**: start playback at the target volume, then repeat with several prior volumes loaded first;
- **disconnected-island anchoring**: inspect isolated far-range echo patches with little local continuity support;
- **high-shear couplet**: check whether a tight near-range velocity couplet is preserved or smeared;
- **noisy-bridge test**: test whether weak noisy gates incorrectly connect two otherwise separate regions;
- **weak-flow / patchy-echo test**: inspect whether quiet sparse echoes are aggressively forced onto a branch or left conservatively unresolved.

## 7. Suggested measurements

### Quantitative

- fraction of gates changed,
- fraction of gates left unresolved,
- fold-count agreement to a trusted reference when available,
- absolute velocity error on synthetic cases,
- temporal consistency across volumes,
- runtime and latency.

### Qualitative

- preservation of velocity couplets,
- containment of failure regions,
- stability across scan restart,
- sensitivity to noisy or patchy echoes,
- plausibility of isolated far-range regions.

## 8. Comparison table template

| System or app | Public algorithm disclosure | Likely method family | Confidence | Notes |
| --- | --- | --- | --- | --- |
| Example app | None | Unknown | Low | Black-box comparison only |
| Open reference implementation | Fully documented | Region-based + reference velocity | High | Behavior traceable to source code |

## 9. A defensible writeup style

When writing app comparisons, prefer language like:

- “This output is consistent with a conservative unresolved-gate policy.”
- “This behavior suggests some form of temporal anchoring, but that cannot be verified from outputs alone.”
- “No public documentation was located for the exact algorithm.”

Avoid language like:

- “App X definitely uses algorithm Y.”
- “This proves the app uses AI.”
- “This is obviously a clone of 2DVDA.”

## 10. Recommended black-box experiment checklist

1. Save the raw source volume.
2. Note radar, VCP, Nyquist velocity, sweep, and time.
3. Capture the app output from cold start.
4. Capture the app output after warm playback.
5. Mark gates that differ from the raw aliased field.
6. Mark gates that remain unresolved.
7. Compare against an open baseline such as Py-ART or a curated reference implementation.
8. Repeat across at least one strong-shear and one sparse-echo case.
9. Keep a log of all settings.
10. Distinguish documented fact from inference in the writeup.

## 11. Bottom line

A fair app comparison does not need secret knowledge. It needs:

- controlled cases,
- explicit evidence labels,
- repeatable capture methods,
- careful separation of fact, inference, and unknowns.

That discipline is the main defense against accidental misinformation in discussions of proprietary radar tools.
