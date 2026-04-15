# Algorithm taxonomy for Doppler velocity dealiasing

This document classifies the major public families of Doppler velocity dealiasing algorithms and explains where each class tends to work, where it fails, and how “academic” or “operational” it tends to be.

## 1. The problem: velocity aliasing

A weather radar reports radial velocity only within an **unambiguous interval** bounded by the **Nyquist velocity** \(V_N\). If the true radial velocity \(v_t\) exceeds that interval, the observed velocity \(v_o\) is wrapped by an integer multiple of \(2V_N\):

```text
v_t = v_o + 2 k V_N
```

for some integer fold count \(k\).

The dealiasing task is to recover the most plausible \(k\) at each gate.

Important distinctions:

- **Velocity aliasing** is not the same as **range folding**, though they are often related operationally.
- A dealiased field is still only as good as the gates that survive clutter suppression, biological filtering, low-SNR masking, and geometric consistency checks.
- The problem is partly local and partly global. A single gate may be easy to correct, while the sweep-wide or volume-wide ambiguity pattern is much harder.

## 2. Why dealiasing is difficult in practice

The basic modulo arithmetic is easy. The hard part is choosing the correct Nyquist interval when the field contains:

- missing gates,
- strong azimuthal or radial shear,
- small disconnected echo regions,
- mixed meteorological and non-meteorological targets,
- scan-to-scan timing changes,
- weak or noisy returns,
- multiple candidate reference states that all look locally plausible.

In other words, real dealiasing is a **constrained inference problem**, not just a wrap correction.

## 3. Family A: gate-to-gate continuity / 1D radial unwrap

### Core idea

Assume that along a single radial, neighboring valid gates should not jump by more than a physically reasonable amount. Starting from a seed gate, choose the fold count at the next gate so that the corrected velocity is closest to the previous corrected value.

### Typical workflow

1. choose a seed gate or seed value,
2. move outward and/or inward along the radial,
3. at each step, choose the Nyquist interval nearest the local reference,
4. break or reseed when continuity is not credible.

### Strengths

- Extremely fast.
- Easy to implement and explain.
- Often good enough when the field is well connected and shear is moderate.
- Useful as a cleanup pass even when the main algorithm is more sophisticated.

### Weaknesses and failure modes

- Can create long radial streaks or wedge-shaped errors after a bad seed.
- Vulnerable when gate-to-gate shear approaches or exceeds the Nyquist interval.
- Brittle across missing segments, weak echo gaps, or clutter contamination.
- Handles isolated patches poorly without an external reference.

### Operational practicality

Very high. A 1D continuity pass is one of the most operationally practical building blocks, but it is rarely sufficient by itself for the hardest severe-weather cases.

### Public examples

- Ray and Ziegler (1977)
- Eilts and Smith (1990)
- legacy continuity-oriented operational schemes

## 4. Family B: region-growing / path-following / 2D sweep methods

### Core idea

Treat a sweep as a 2D field in azimuth–range space. Rather than dealiasing one radial independently, identify locally coherent regions or paths and propagate fold choices through the connected structure.

### Common variants

- flood-fill from seed gates,
- path-following with local smoothness checks,
- graph-based region merging,
- dynamic network reduction on velocity regions,
- recursive handling of high-shear subregions.

### Why it helps

Many meteorological structures are more coherent in 2D than in a single radial. A purely radial method may break a curved shear zone that a 2D method can follow.

### Strengths

- Better at using azimuthal continuity and neighborhood context.
- Usually superior to simple 1D radial unwrap in meso-scale flow and organized convection.
- Can separate and merge connected regions in a more controlled way.

### Weaknesses and failure modes

- Still depends heavily on seed selection and QC.
- Disconnected echo islands remain difficult.
- If noisy or cluttered gates are allowed into the graph, entire regions can be mis-unfolded.
- Region boundaries can be wrong where shear is strongest.

### Operational practicality

High. This is one of the most important practical families for open implementations.

### Public examples

- Bergen and Albers (1988)
- Jing and Wiener (1993)
- Zhang and Wang (2006)
- Py-ART `dealias_region_based`
- R2D2
- UNRAVEL

## 5. Family C: multi-PRF / dual-PRF / staggered-PRT methods

### Core idea

Use observations acquired with more than one pulse repetition frequency or pulse repetition time to extend the effective unambiguous velocity or recover the correct fold.

### Important caveat

This is not purely an after-the-fact software trick. It depends on the acquisition strategy. If the radar did not collect the data with the needed timing diversity, a downstream app cannot invent that information.

### Strengths

- Can dramatically reduce or postpone aliasing in favorable scan modes.
- Very effective in systems designed around the method.
- Operationally valuable for larger ranges or lower-Nyquist situations.

### Weaknesses and failure modes

- Added sensitivity to noise and phase differences.
- Shear-induced problems can remain severe.
- Scan-strategy constraints matter.
- Corrections may behave differently at different trips or ranges.

### Operational practicality

High when the radar and volume coverage pattern support it. Otherwise unavailable.

### Practical note

Multi-PRF capability changes the problem itself. Comparing an app using single-PRF data to one using dual-PRF-derived inputs is not an apples-to-apples dealiasing comparison.

## 6. Family D: environmental wind / VAD / model-assisted methods

### Core idea

Estimate a background radial velocity field from a wind profile, VAD analysis, sounding, model analysis, or a simple environmental wind table. Then choose the fold count that places each observation closest to that reference.

### Why it matters

Pure continuity methods struggle with disconnected precipitation patches or weak echoes. A reference field can supply the missing absolute anchor.

### Strengths

- Helps sparse echoes and isolated valid regions.
- Gives a principled way to pick the correct Nyquist co-interval when local continuity is ambiguous.
- Often useful as a first guess before regional refinement.

### Weaknesses and failure modes

- Catastrophic when the background wind is unrepresentative.
- Very risky near mesocyclones, convergence boundaries, rear-flank flow, tropical eyewalls, or tornado-scale signatures.
- Can over-smooth genuine local extremes if trusted too strongly.

### Operational practicality

High. Reference-field assistance is a practical operational ingredient, but it should usually be combined with local continuity and QC rather than used blindly gate-by-gate.

### Public examples

- legacy environmental wind tables,
- modified VAD-based references,
- model-assisted 3D background wind fields,
- Py-ART reference-velocity options.

## 7. Family E: spatial smoothness and QC-driven methods

### Core idea

Use velocity texture, reflectivity support, clutter flags, dual-polarization variables, or other QC signals to decide which gates are trustworthy enough to anchor dealiasing and which should be masked or handled conservatively.

### Why this matters

Many apparent “dealiasing failures” are really **QC failures that poison dealiasing**. A good unwrap method can still fail if clutter, sidelobe contamination, biological targets, or noisy weak echoes are allowed to seed or connect regions.

### Strengths

- Often produces the biggest real-world improvement for the least algorithmic complexity.
- Helps prevent error propagation.
- Supports stable, conservative operational behavior.

### Weaknesses and failure modes

- Over-masking destroys continuity and leaves too much missing data.
- Under-masking lets bad gates contaminate entire regions.
- Texture thresholds that work in one regime may fail in another.

### Operational practicality

Very high. Good QC is operationally mandatory.

### Practical conclusion

A “better” unwrap algorithm can look worse than a simpler one if its QC front end is weaker.

## 8. Family F: temporal continuity / previous-volume / 4D methods

### Core idea

Use a prior dealiased scan or prior volume as a reference for the current one, optionally together with sounding or model data. The assumption is that the atmosphere does not change arbitrarily between adjacent volumes.

### Strengths

- Excellent for stabilizing large-scale flow.
- Often improves continuity of isolated regions that are ambiguous in a single sweep.
- Helpful in continuous playback or operational pipelines with persistent state.

### Weaknesses and failure modes

- Previous errors can propagate forward.
- Cold start is hard: the first volume still needs a trustworthy solution.
- Geometry changes, missing rays, or timing differences complicate remapping.
- Rapid storm motion or rapidly evolving small-scale circulations reduce usefulness.

### Operational practicality

High, especially in persistent operational processing chains. Moderate in ad hoc desktop viewing where state may reset or playback may start mid-event.

### Public examples

- James and Houze (2001) 4DD
- Py-ART `dealias_fourdd`
- model-analysis-assisted time-continuity methods

## 9. Family G: 2D vs 3D vs volume-wide optimization

### 2D methods

Operate within one sweep. They are simpler, faster, and easier to reason about. They are often the best default for live use.

### 3D methods

Use neighboring elevations or a full sweep stack. They can improve vertical consistency and help when a small region is ambiguous in one tilt but obvious in adjacent tilts.

### Volume-wide optimization methods

Pose dealiasing as a larger optimization over a graph, gradient field, or smoothness constraint. These may use least squares, variational formulations, torus mappings, or recursive region splits.

### Trade-off

As scope widens from 1D to 2D to 3D to full volume, you often gain global consistency but also gain:

- more assumptions,
- more complexity,
- more ways for a wrong anchor to contaminate a larger region,
- more tuning and computational cost.

## 10. Family H: variational and global methods

### Core idea

Define an objective that penalizes implausible gradients or discontinuities and solve for fold counts or corrected velocities that best satisfy the objective.

### Strengths

- Can produce elegant, coherent solutions.
- Useful when a stronger global constraint is justified.
- Attractive for data assimilation contexts.

### Weaknesses and failure modes

- Often depends on assumptions about smoothness or background consistency that fail in convective extremes.
- More complex to implement correctly.
- Can hide local mistakes behind globally plausible fields.

### Operational practicality

Medium. These methods are important and worthy of study, but they are less often the simplest answer for a live viewer.

## 11. Family I: learned or machine-learning-assisted approaches

### Core idea

Train a model to imitate an existing dealiasing output, predict fold counts, or assist with QC / seed selection.

### Realistic benefits

- Fast inference after training.
- Useful when the goal is to emulate a known public or operational target.
- Potentially attractive for portable deployments.

### Hard limits

- The model learns the biases of its training target.
- Out-of-distribution behavior is a major concern.
- Hard guarantees about continuity, topology, or physical consistency are difficult.
- A learned model is not automatically “more advanced” than a transparent algorithm.

### Practical position

Useful as an assistive or emulation approach. Not yet a substitute for well-understood QC + continuity + reference-field logic in most open operational settings.

## 12. Where methods usually fail

### Mesocyclones and tornado-scale couplets

The field may change by more than one Nyquist interval over very short distances. Local continuity assumptions become fragile.

### High-shear QLCS segments

Strong convergence or shear lines can cause region-merge mistakes and sharp boundary failures.

### Tropical cyclones

Broad strong winds can exceed low Nyquist limits over large areas, while embedded mesovortices add local complexity.

### Noisy weak-echo environments

Sparse valid gates make connectivity weak and make external references disproportionately important.

### Near-range couplets

Small geometric spacing near the radar makes gate-to-gate gradients especially challenging.

### Biological contamination and clutter

These targets can create locally smooth but meteorologically incorrect structures that contaminate region growing.

## 13. Operationally practical versus primarily academic

| Family | Real-time practicality | External data needed | Best use |
| --- | --- | --- | --- |
| 1D continuity | High | No | Fast cleanup and simple baseline |
| 2D region-based | High | No | Mainline open reference method |
| Environmental / VAD-assisted | High | Often yes | First guess and disconnected regions |
| Temporal / 4D | High in persistent pipelines | Previous volume or reference profile | Stabilization and continuity |
| Multi-PRF / staggered-PRT | High when acquisition supports it | Special scan mode | Acquisition-side ambiguity reduction |
| Variational / global | Medium | Sometimes | Research and assimilation contexts |
| Learned assist | Medium to low | Training data | Emulation, acceleration, assistive tools |

## 14. Practical synthesis

The most defensible open operational stack is usually **hybrid**, not doctrinaire.

A robust real-time design often looks like:

1. conservative QC,
2. reference-field first guess when available,
3. 2D regional continuity as the main unwrap,
4. targeted 1D cleanup,
5. optional temporal stabilization,
6. explicit unresolved/no-data handling.

That combination usually outperforms any single elegant idea applied in isolation.
