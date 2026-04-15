# Implementation notes for practical Doppler velocity dealiasing

This document is for developers building dealiasing into an open radar stack that consumes Level II or similar radial data.

## 1. Think in terms of a pipeline, not a single function

A practical dealiasing pipeline usually has these stages:

1. ingest and normalize geometry,
2. standardize sign conventions and units,
3. apply conservative quality control,
4. choose seeds or a background reference,
5. run the main dealiasing pass,
6. apply targeted cleanup,
7. evaluate and flag unresolved regions.

The most common mistake is to focus on step 5 while neglecting steps 2–4.

## 2. Data model: rays, sweeps, volumes

### Per-ray

A ray is a single azimuth sample at one elevation cut. Per-ray logic is natural for 1D continuity methods.

### Per-sweep

A sweep is a 2D azimuth–range field at fixed elevation. Most practical open methods are easiest to reason about here.

### Volume

A volume stacks multiple sweeps in time order. Temporal continuity methods and some 3D methods need this broader context.

### Recommendation

Treat the sweep as the default working object, but preserve access to per-ray metadata and prior-volume state.

## 3. Scan geometry matters

### Azimuth wrap-around

Azimuth is periodic. Neighbor logic should usually connect the first and last ray of a sweep.

### Range spacing

Gate spacing controls how much physical shear can occur between gates. A continuity threshold that is acceptable at coarse spacing can be too aggressive at fine spacing.

### Elevation and height

A background wind profile is only useful if you map each gate to an appropriate height. Even a simple flat-earth approximation should be stated as an approximation.

### Changing geometry across volumes

Temporal continuity is only easy if scan geometry matches. If azimuth counts, gate spacing, or cut order change, you need remapping before you can trust prior volumes.

## 4. Quality-control prerequisites

A dealiased field is only as good as the gates allowed to seed it.

Recommended QC inputs when available:

- reflectivity support,
- spectrum width,
- signal quality / SNR,
- velocity texture,
- clutter maps,
- dual-polarization indicators,
- known non-meteorological target masks.

### Velocity texture

Texture is one of the most useful generic indicators. High local texture often marks noise, clutter edges, or aliased discontinuities that should not be trusted as seeds.

### Conservative rule

It is usually better to leave questionable gates unresolved than to let them anchor a large region incorrectly.

## 5. Seed selection

### Why seed choice matters

A wrong seed does not only harm one gate. It can set the absolute Nyquist interval for an entire connected region.

### Good seed candidates

- low-texture gates,
- gates with strong meteorological support,
- gates that agree with a trusted reference wind,
- gates near large, coherent precipitation regions,
- temporally stable gates from a previous dealiased volume.

### Poor seed candidates

- isolated weak returns,
- gates next to clutter edges,
- gates inside the strongest part of a tornado-scale couplet,
- noisy low-SNR patches,
- disconnected tiny regions with no reference.

## 6. Continuity thresholds

A continuity threshold should reflect expected physical gradients and data resolution, not just a fixed fraction of the Nyquist interval.

Questions to ask:

- how far apart are gates in range?
- what azimuth spacing does the sweep use?
- how strong is the environmental shear?
- are you near the radar where beam geometry magnifies local gradients?
- is the field convective, tropical, or weak-flow stratiform?

### Practical guidance

- Use tighter thresholds for seed trust than for later cleanup.
- Allow different thresholds for radial and azimuthal directions.
- Avoid one global threshold for every weather regime.

## 7. Handling missing gates

Missing data is not just an inconvenience. It breaks continuity.

### Recommended behavior

- keep missing gates as missing unless a strong regional or reference-based reason exists to fill them,
- avoid letting large gaps bridge separate regions automatically,
- treat gap-spanning corrections as lower confidence.

### Common mistake

Connecting across a large gap with the same logic used for adjacent valid gates.

## 8. Handling noisy gates and low-SNR environments

Weak echoes are doubly hard:

- the observed velocity is less reliable,
- the continuity graph becomes fragmented.

Practical responses:

- raise seed standards,
- trust temporal or environmental references more,
- lower expectations for full coverage,
- report unresolved fractions explicitly.

## 9. Biological contamination and non-meteorological echoes

Birds, insects, ground clutter, wind farms, sidelobes, and anomalous propagation can all create velocity patterns that are internally smooth enough to fool a continuity-based method.

Recommendations:

- use non-velocity variables when available to mask non-meteorological targets,
- do not interpret local smoothness as proof of meteorological validity,
- keep clutter suppression ahead of dealiasing in the pipeline.

## 10. Range-folded and mixed-origin gates

Velocity dealiasing is not a cure for every bad gate. If the return itself is contaminated by range ambiguity or mixed targets, choosing a fold count may still produce a physically meaningless result.

Recommendation:

- tag such gates as low-confidence or unresolved,
- avoid scoring them as “corrected” just because the output looks smooth.

## 11. Velocity texture and small-scale shear

Texture is useful, but beware a trap: real mesocyclones and tornado signatures are also high-gradient regions.

A good pipeline does not simply delete all high-texture gates. It distinguishes:

- high texture from noise or mixed targets,
- high gradient from genuine compact meteorological structure.

That usually means combining texture with reflectivity support, spatial coherence, and perhaps prior-volume context.

## 12. Environmental-wind references

### When they help

- disconnected echo islands,
- weak echo cases,
- broad stratiform flow,
- tropical outer bands,
- initial seeding of a larger 2D method.

### When they hurt

- tornadic supercells,
- convergence lines,
- rear-flank jets,
- internal eyewall asymmetries,
- any place where local winds depart sharply from the background.

### Practical rule

Use the background to choose a plausible Nyquist interval, then let local continuity override it where the data support a coherent local structure.

## 13. Previous-volume logic

Temporal continuity can be extremely effective, but only if you manage state carefully.

Recommendations:

- store not just the corrected velocity but also confidence or provenance,
- remap the previous field to the current geometry before using it,
- gate temporal trust by elapsed time, scan similarity, and echo evolution,
- do not let a low-confidence prior field dominate a strong current local signal.

## 14. Per-ray versus per-sweep versus volume logic

### Per-ray is best for

- simple educational baselines,
- cleanup passes,
- very fast conservative operations.

### Per-sweep is best for

- mainline continuity-based dealiasing,
- region-growing,
- graph-based connected-region methods.

### Volume-wide logic is best for

- temporal stabilization,
- vertically consistent ambiguity resolution,
- data assimilation-oriented workflows.

### Practical design

Use different scopes for different stages instead of forcing one scope to do everything.

## 15. Correction of isolated folded gates versus volume-wide ambiguity

These are not the same problem.

### Isolated folded-gate correction

A few bad gates inside an otherwise correct field. Local methods usually handle this well.

### Volume-wide ambiguity resolution

Large parts of the sweep sit in the wrong Nyquist interval. Local fixes may not know the absolute offset without a good seed or reference.

This distinction matters because a method that is excellent at isolated cleanup may still fail at global ambiguity placement.

## 16. Evaluation metrics to compute during development

### When synthetic truth is available

- fold-count accuracy,
- absolute velocity error,
- fraction of gates with correct sign,
- error localized near strong gradients.

### When real truth is not available

- self-consistency with neighboring gates,
- temporal consistency across volumes,
- agreement with trusted open baselines,
- unresolved-gate fraction,
- failure containment,
- expert review of known signatures.

### Always measure

- runtime,
- memory use,
- sensitivity to parameter changes,
- cold-start versus warm-state behavior.

## 17. Logging and reproducibility

For serious comparison work, log:

- source file and radar,
- sweep index and elevation,
- Nyquist velocity,
- QC thresholds,
- seed strategy,
- reference field source,
- whether previous-volume state was used,
- unresolved-gate count,
- version or commit hash.

Without that metadata, it is very hard to compare algorithms fairly.

## 18. A practical open pipeline recommendation

A sensible open baseline for Level II-like data is:

1. normalize units and sign convention,
2. compute a conservative gate filter,
3. estimate velocity texture,
4. build a modest background reference if available,
5. run a 2D region-based main pass,
6. run a 1D radial cleanup pass,
7. optionally stabilize with previous-volume information,
8. leave low-confidence regions unresolved instead of forcing a pretty answer.

That design is not the only option, but it is hard to regret.


## 19. Implementation-oriented pseudocode

### A. 1D radial continuity unwrap

```text
choose a trusted seed gate
set corrected[seed] = observed[seed] or nearest value to background reference

for each direction away from seed:
    remember the last trusted corrected gate
    for each next valid gate:
        local_ref = last trusted corrected value
        candidate = observed gate shifted by 2*k*Nyquist closest to local_ref

        if candidate jump is implausibly large and a background reference exists:
            candidate = observed gate shifted closest to background reference

        write corrected gate
        update confidence from mismatch
```

### B. 2D region-growing unwrap

```text
estimate local texture on the sweep
pick a low-texture seed, preferably one close to a background reference
push seed into a frontier queue

while frontier is not empty:
    pop the lowest-mismatch frontier gate
    for each uncorrected neighbor:
        gather corrected neighboring values
        add background reference at that gate if available
        if no local references exist:
            skip for now

        local_ref = median(local references)
        candidate = observed gate shifted by 2*k*Nyquist closest to local_ref

        if mismatch is acceptable:
            accept candidate
            push neighbor into frontier
        else:
            leave gate for later or another region

repeat from a new seed until all connected valid regions are handled
run a light reconciliation pass near region boundaries
```

### C. Environmental-wind-seeded unwrap

```text
build a radial velocity reference field from:
    uniform wind, VAD profile, sounding, or model background

for each gate:
    first_guess = observed gate shifted into the Nyquist interval nearest reference gate

optionally refine:
    run 2D region-growing using first_guess as the regional anchor
```

### D. Temporal continuity aided unwrap

```text
remap previous dealiased sweep/volume to the current sweep geometry
for each gate:
    first_guess = observed gate shifted into the Nyquist interval nearest previous corrected value

optionally refine:
    run 2D region-growing using first_guess as a temporal anchor

guardrails:
    reduce trust when time gap is large
    reduce trust when scan geometry changes
    do not let a low-confidence previous field override strong current local evidence
```

## 20. Practical caveat

These pseudocode blocks are deliberately clean and educational. Real systems add:

- richer QC,
- confidence propagation,
- region splitting and merging heuristics,
- geometry remapping,
- scan-mode awareness,
- product-specific edge handling,
- explicit failure containment logic.
