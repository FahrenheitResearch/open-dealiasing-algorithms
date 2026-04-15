# Repository selection notes

This file records how the curated **central best** version was chosen from five candidate GPT-generated repositories.

## Overall recommendation

Use this repository as the starting point.

The base chosen here was the unnumbered `open-dealiasing-algorithms.zip` candidate because it had the best balance of:

- complete deliverables,
- the cleanest package structure,
- a working test suite,
- a `pyproject.toml`,
- examples that run successfully,
- and the strongest out-of-box behavior on simple synthetic reference cases.

## Ranking of the five candidates

1. **`open-dealiasing-algorithms.zip`** — best overall starting point
2. **`open-dealiasing-algorithms (3).zip`** — strong alternate, especially for concise writing and app-claim caution
3. **`open-dealiasing-algorithms (4).zip`** — strongest single documentation pass on some sections, but weaker code/package baseline
4. **`open-dealiasing-algorithms (1).zip`** — good educational framing and examples, but weaker out-of-box algorithm anchoring and no tests
5. **`open-dealiasing-algorithms (2).zip`** — broad bibliography and solid implementation-notes coverage, but weaker default algorithm behavior and one temporal example that timed out during audit

## What was audited

The audit focused on four things:

1. **Repository completeness**
   - required docs present,
   - readable structure,
   - packaging metadata,
   - tests and examples.

2. **Technical writing quality**
   - whether the docs explain algorithm families clearly,
   - whether proprietary-app claims stay conservative,
   - whether practical implementation constraints are called out.

3. **Reference-code quality**
   - whether the modules are readable and reasonably modular,
   - whether default behavior works on basic synthetic cases,
   - whether tests exist and pass.

4. **Operational honesty**
   - whether the repo distinguishes public documentation from hypothesis,
   - whether “unknown / cannot verify” is used appropriately,
   - whether preprocessing/QC is treated as first-class.

## Synthetic audit summary

A small common synthetic benchmark was run against the candidate code where the APIs allowed it. The benchmark was not meant to prove operational superiority; it was only a sanity check for default educational behavior.

| Candidate | Tests present and passing | Example scripts | 1D continuity default | 2D region default | Environmental reference default | Temporal reference default |
| --- | --- | --- | --- | --- | --- | --- |
| `open-dealiasing-algorithms.zip` | Yes | All audited examples ran | Recovered simple ramp | Recovered simple field | Recovered reference-anchored field | Recovered previous-volume case |
| `open-dealiasing-algorithms (3).zip` | Yes | Audited example ran | Recovered simple ramp | Recovered simple field | Recovered reference-anchored field | Recovered previous-volume case |
| `open-dealiasing-algorithms (4).zip` | No tests | Audited examples ran | Left global fold ambiguity unresolved by default | Recovered simple field | Recovered reference-anchored field | Recovered previous-volume case |
| `open-dealiasing-algorithms (1).zip` | No tests | Audited examples ran | Left global fold ambiguity unresolved by default | Left global fold ambiguity unresolved by default | Recovered reference-anchored field | Recovered previous-volume case |
| `open-dealiasing-algorithms (2).zip` | No tests | Two examples ran; temporal example timed out in audit | Left global fold ambiguity unresolved by default | Left global fold ambiguity unresolved by default | Recovered reference-anchored field | Recovered previous-volume case |

### Important nuance on the weaker continuity scores

The weaker 1D / 2D default results above do **not** necessarily mean those candidate algorithms are useless. In several cases, the implementations are written to accept local continuity or the first observed gate as a default seed, which leaves the **global branch ambiguity** unresolved unless a stronger seed or reference is supplied.

For an educational starter kit, though, the better default behavior in the chosen base is valuable because it demonstrates the “absolute branch anchoring” issue more clearly without extra setup.

## What this curated repo kept from the other candidates

This curated version starts from the strongest base repo and folds in a few targeted ideas from the other variants:

- from **candidate (3)**:
  - stronger wording around the ambiguity of the user shorthand **“Yalldar”**,
  - good discipline around treating informal product-name mappings as interpretations rather than facts.

- from **candidate (4)**:
  - stronger wording on the narrow scope of what can be said publicly about **GR2Analyst**,
  - helpful README language around selected public references.

- from **candidate (2)**:
  - useful named black-box stress tests such as **cold-start vs warm-start**, **disconnected-island anchoring**, and **noisy-bridge** tests.

## What this curated repo intentionally did not import

These parts were intentionally left out:

- the weaker default continuity/region behavior from candidates **(1)**, **(2)**, and **(4)**;
- the lack of tests in candidates **(1)**, **(2)**, and **(4)**;
- the timed-out temporal example from candidate **(2)**;
- any wording that drifted too close to implied certainty about closed-source app internals.

## Practical takeaway

If you want one repo to keep building on, use this one.

If you want one alternate repo worth reading for wording and app-comparison nuance, read **candidate (3)** alongside it.

If you want extra phrasing ideas for conservative proprietary-app discussion, mine **candidate (4)** selectively, but do not use it as the primary code baseline.
