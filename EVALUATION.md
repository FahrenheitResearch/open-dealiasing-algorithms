# Evaluation of radar velocity dealiasing algorithms

This document explains how to compare dealiasing algorithms fairly.

## 1. Evaluation principles

A fair comparison should separate:

- the quality of the source data,
- the quality-control front end,
- the main dealiasing logic,
- the unresolved-gate policy,
- any temporal state or background-field help,
- the display layer.

Do not compare two algorithms as if they are pure unwrap functions if one also uses heavier QC, previous volumes, or multi-PRF-derived inputs.

Also distinguish the solver role:

- core solver: directly chooses fold counts from the current sweep or volume,
- anchor provider: supplies a reference field but does not itself resolve the full sweep,
- refinement/cleanup pass: adjusts an already-bootstrapped field,
- hybrid composition: combines several of the above.

That distinction matters because a hybrid can be very useful operationally while
still being a poor basis for claiming that one "solver" is universally better
than another.

## 2. What to treat as "truth"

There is no single truth source for all cases. Use the strongest available option and say which one you used.

### Synthetic truth

Best for controlled testing. You know the true velocity and the true fold counts exactly.

### High-confidence operational reference

Useful when a curated public operational product or a trusted open method is available, but still not perfect truth.

### Manual expert analysis

Useful for case studies of couplet preservation and obvious failure regions. Valuable, but subjective.

### Consistency-only evaluation

Acceptable when hard truth is unavailable, but weaker. Measure temporal stability, continuity, and containment instead of absolute correctness.

## 3. Accuracy metrics

### Fold-count accuracy

If the true fold count \(k\) is known, this is the cleanest metric. Wrong fold count matters more than small residual velocity error.

### Absolute velocity error

Compare dealiased velocity to truth in m/s. Report mean, median, RMSE, and high-percentile errors.

### Sign error rate

Important for warning-relevant interpretation. Even one sign flip in the wrong place can matter more than a small smooth error.

### Local-gradient preservation

Measure whether strong but real gradients are preserved or falsely smoothed away.

## 4. Robustness metrics

### Unresolved-gate fraction

A conservative algorithm may leave ambiguous gates blank. This should be reported, not hidden.

If a solver always fills every gate, say so. If it only fills gates after a
cleanup or fallback pass, report that separately from the main solver output.

### Failure containment

When the algorithm is wrong, how large is the bad region? Small localized failures are operationally better than sweep-wide catastrophes.

### Sensitivity to missing data

Test with removed rays, missing gate blocks, and fragmented echoes.

### Sensitivity to noise and QC thresholds

A good algorithm should not collapse when the texture threshold moves slightly.

### Cold-start versus warm-state stability

Temporal methods must be evaluated both with and without prior state.

## 5. Computational metrics

### Latency

How long does the algorithm take per sweep or per volume?

### Throughput

How many sweeps or volumes per second can it handle?

### Memory footprint

Important for browser, mobile, or low-resource deployments.

### Scalability

Does runtime grow linearly with gate count, region count, or both?

## 6. Operational reliability metrics

### Graceful failure behavior

Does the algorithm blank questionable gates, or does it invent a dramatic but wrong structure?

### Parameter stability

How much retuning is needed across tropical, convective, and weak-echo cases?

### Temporal consistency

Does the output flicker or jump between volumes for the same slowly evolving structure?

### Dependency risk

How strongly does performance depend on previous-volume state, soundings, or model data that may be unavailable in some settings?

## 7. Suggested benchmark protocol

1. Separate cases by storm mode.
2. Note Nyquist velocity and scan strategy.
3. Run each algorithm with a documented QC configuration.
4. Evaluate synthetic and real cases separately.
5. Record unresolved fractions and failure regions, not just successful gates.
6. Include cold-start tests.
7. Include parameter sensitivity sweeps for at least one threshold.
8. Report runtime on the same hardware.
9. If a method needs unavailable support data or would require reference leakage to run, mark it skipped rather than synthesizing inputs.
10. Label each run as `core`, `anchor-assisted`, `cleanup-assisted`, or `hybrid` so the results stay comparable.

## 8. Useful qualitative review questions

- Does the couplet remain compact and physically plausible?
- Are broad tropical winds placed in the right Nyquist interval over large areas?
- Does the method respect disconnected sparse echoes, or force them into arbitrary continuity?
- When the algorithm is uncertain, does it behave honestly?

## 9. Known impossible or near-impossible cases

There are cases where the available data do not uniquely determine the correct answer.

Examples:

- tiny disconnected valid patches with no reliable background field,
- extreme shear spanning multiple Nyquist intervals over a single sample spacing,
- corrupted or mixed-target returns,
- sparse weak-echo scenes with a bad previous volume,
- strong local departures from the environmental wind in the only valid region.

A good evaluation acknowledges these cases instead of pretending every gate has a provable answer.

## 10. Suggested reporting table

| Metric | Why it matters |
| --- | --- |
| Fold-count accuracy | Direct measure of ambiguity resolution |
| Absolute velocity error | Magnitude performance |
| Sign error rate | Warning relevance |
| Unresolved-gate fraction | Conservative honesty |
| Failure containment area | Catastrophe control |
| Runtime per sweep | Live usability |
| Cold-start degradation | Temporal dependence |
| Parameter sensitivity | Robustness |

## 11. What not to do

- Do not use a vendor app output as unquestioned truth unless you clearly state that it is only a reference.
- Do not compare algorithms with different QC masks and then attribute all differences to the unwrap logic.
- Do not hide unresolved gates by interpolating them away before scoring.
- Do not evaluate only one photogenic case.
- Do not describe a composition as a pure solver if it bootstraps from another method.
- Do not treat a single-case speed or accuracy check as a general benchmark unless you explicitly say it is one.

## 12. Bottom line

The best dealiasing algorithm is not always the one that fills the most gates or produces the smoothest image. In operational practice, the best algorithm is often the one that:

- gets the hard parts right often enough,
- contains its failures,
- avoids spectacular false structure,
- and behaves predictably when it does not know.
