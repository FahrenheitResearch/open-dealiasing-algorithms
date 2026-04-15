# Canonical hard test cases for radar velocity dealiasing

This document defines a set of difficult case families that should be used when comparing dealiasing algorithms.

The goal is not to find only “pretty wins.” The goal is to expose failure modes.

## 1. What makes a good test suite

A useful dealiasing test suite should vary:

- Nyquist velocity,
- echo coverage and connectivity,
- environmental wind strength,
- storm mode,
- availability of previous volume state,
- availability of dual-PRF or staggered-PRT information,
- quality-control severity.

## 2. Case family summary

| Case family | Why it is hard | What to inspect |
| --- | --- | --- |
| Supercell with strong mesocyclone | Compact high shear and possible multi-fold ambiguity | couplet preservation, boundary errors, false wedges |
| Tropical cyclone | Large areas of broad strong flow with embedded mesoscale structure | absolute interval placement over wide area |
| High-shear QLCS | Sharp boundaries and linear organization | continuity breaks and merge errors |
| Noisy low-SNR environment | Weak fragmented returns and poor seed quality | unresolved fraction versus false corrections |
| Near-range velocity couplet | Very short spatial scales near radar | oversmoothing and wrong absolute folds |
| Weak flow with patchy echoes | Sparse connectivity and low trust in local gradients | dependence on background or temporal references |
| Fast storm motion / strong environmental wind | Rapid changes and large broad flow | temporal stability and absolute anchoring |
| Dual-PRF available vs unavailable | Different ambiguity structure | fair separation of acquisition versus post-processing |

## 3. Supercells with strong mesocyclones

### Why this case matters

This is where users most notice dealiased velocity differences. Small-scale rotational structure can produce gate-to-gate gradients that challenge local continuity assumptions.

### Common failures

- incorrect sign reversal in part of the couplet,
- wedge-shaped radial streaks,
- over-smoothing that weakens the couplet,
- region merge errors across the hook / inflow side.

### What to evaluate

- does the algorithm preserve a compact inbound/outbound pair?
- are errors contained locally or do they contaminate a large region?
- does the algorithm behave consistently across adjacent volumes?

## 4. Tropical cyclones

### Why this case matters

Broad areas can exceed the Nyquist interval, especially when scan strategy yields relatively low unambiguous velocity. Embedded mesovortices and eyewall gradients complicate the background.

### Common failures

- entire bands placed in the wrong Nyquist interval,
- poor continuity across wide azimuth spans,
- over-reliance on a background field that misses internal asymmetries.

### What to evaluate

- large-scale absolute interval placement,
- continuity across outer rainbands,
- behavior near eyewall asymmetries,
- cold-start versus warm-state differences.

## 5. High-shear QLCS segments

### Why this case matters

Linear convective systems can have sharp convergence boundaries and embedded circulations that are narrower than many smoothness assumptions expect.

### Common failures

- broken shear lines,
- fold-count flips along the line,
- contamination by noisy leading-edge or trailing-stratiform regions.

### What to evaluate

- line continuity,
- embedded circulation retention,
- false convergence / divergence artifacts.

## 6. Noisy low-SNR environments

### Why this case matters

Sparse or weak echoes reduce local continuity and make noisy gates disproportionately influential.

### Common failures

- random isolated fold mistakes,
- aggressive filling of noise,
- over-masking that leaves almost no usable field.

### What to evaluate

- unresolved-gate fraction,
- false-correction rate,
- dependence on seed thresholds,
- whether the method fails conservatively.

## 7. Near-range velocity couplets

### Why this case matters

Near the radar, the spatial sampling can resolve very sharp gradients over only a few gates. Simple continuity logic is often least trustworthy exactly where warning-relevant detail is highest.

### Common failures

- couplet collapse into a smoother but wrong field,
- alternating fold mistakes in adjacent gates,
- ring-like artifacts from poor azimuth wrap handling.

### What to evaluate

- local peak magnitude retention,
- spatial localization,
- sensitivity to gate spacing and azimuth count.

## 8. Weak flow with patchy echoes

### Why this case matters

When the true winds are modest but coverage is sparse, the problem is less about massive aliasing and more about choosing the correct interval in disconnected pieces.

### Common failures

- arbitrary absolute offsets in isolated patches,
- overconfidence in poor seeds,
- inconsistent behavior between similar disconnected echoes.

### What to evaluate

- benefit of environmental references,
- benefit of previous-volume anchoring,
- consistency across isolated islands.

## 9. Fast storm motion and strong environmental wind

### Why this case matters

Temporal continuity assumptions become harder to trust when storm motion is fast or environmental winds are strong relative to the Nyquist interval.

### Common failures

- prior-volume error carryover,
- misregistration between volumes,
- temporal smoothing that lags rapid evolution.

### What to evaluate

- cold-start versus running-state output,
- sensitivity to temporal remapping,
- stability without loss of evolving structure.

## 10. Dual-PRF available versus unavailable

### Why this case matters

An algorithm may look “better” simply because the source data had more ambiguity information available.

### Required comparison discipline

- compare like with like,
- document scan strategy and PRF context,
- separate acquisition-side ambiguity reduction from software-side unfolding skill.

## 11. Additional stressors to combine with every case

For each canonical case, it is worth creating variants with:

- random missing rays,
- reflectivity-threshold masking,
- injected noisy gates,
- delayed or absent previous-volume state,
- intentionally biased background wind references.

These variants reveal whether a method fails gracefully.

## 12. Suggested scoring sheet

For each case, record:

- source and timestamp,
- sweep and Nyquist velocity,
- algorithm version,
- QC configuration,
- temporal state available? yes/no,
- background reference available? yes/no,
- unresolved fraction,
- obvious false-correction regions,
- couplet or shear-line preservation,
- runtime.

## 13. “Impossible” or near-impossible cases

A good test set should include a few situations where no method can guarantee a correct answer from the available information alone, for example:

- tiny disconnected echo islands with no reliable background field,
- very strong shear spanning multiple Nyquist intervals over one or two gates,
- corrupted returns mixed with non-meteorological targets,
- a bad previous volume anchoring a sparse current field.

These cases are valuable because they reward **honest uncertainty handling**.
