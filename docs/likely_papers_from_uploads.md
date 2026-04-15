# Likely papers behind the uploaded summaries

This note turns the uploaded summary material into a conservative public-paper
map. The point is not to claim that any closed app definitely uses one specific
paper. The point is to identify the public method families that kept recurring
and that plausibly sit underneath many dealiasing stacks.

## Most likely public families

### Eilts and Smith 1990

Paper family:

- local environment constraints
- radial continuity
- nearby-gate logic with local branch selection

Why it matters:

- it is one of the classic public foundations for fast radial dealiasing
- ROC descriptions of legacy VDA behavior line up with this family
- it is the clearest open explanation for "continuity along the ray" behavior

Repo surface:

- `open_dealias.dealias_radial_es90`
- `open_dealias.dealias_sweep_es90`

### Jing and Wiener 1993

Paper family:

- sweep-wide 2D dealiasing
- connected-region or neighborhood-based horizontal continuity

Why it matters:

- it moves beyond isolated 1D ray logic
- it is a strong public ancestor for sweep-level connected-field approaches

Repo surface:

- part of `open_dealias.dealias_sweep_zw06`

### Zhang and Wang 2006

Paper family:

- automated 2D multipass dealiasing
- strict-to-relaxed sweep propagation
- more operationally practical sweep-wide continuity

Why it matters:

- ROC 2DVDA material makes this family especially relevant for WSR-88D product
  discussions
- it is a natural public explanation for many "works better than simple radial
  continuity" behaviors

Repo surface:

- `open_dealias.dealias_sweep_zw06`

### Xu et al. 2011

Paper family:

- VAD-based anchoring
- environmental or background-wind assisted branch choice

Why it matters:

- it directly addresses disconnected or weak-echo scenes where local continuity
  alone does not determine the absolute branch
- public ROC and Py-ART style reference-wind thinking fits the same broad class

Repo surface:

- `open_dealias.estimate_uniform_wind_vad`
- `open_dealias.dealias_sweep_xu11`

### James and Houze 2001

Paper family:

- previous-volume assisted dealiasing
- 4DD style time continuity

Why it matters:

- it is the clearest public anchor for warm-state or playback-history behavior
- Py-ART's `dealias_fourdd` sits in the same public family

Repo surface:

- `open_dealias.dealias_sweep_jh01`
- `open_dealias.dealias_volume_jh01`

## Public operational and open-source trail

The uploaded summaries also pointed to a consistent broader trail:

- ROC says 2DVDA is the default WSR-88D algorithm for Level 3 elevation-based
  velocity products
- ROC legacy VDA descriptions mention radial continuity, nearby-neighbor logic,
  and environmental wind table fallback
- Py-ART documents a region-based solver plus a FourDD interface that uses a
  previous volume and/or sounding reference
- newer open projects such as UNRAVEL and R2D2 show that the space is still
  active and not locked behind vendor secrecy

## What this does and does not imply

This mapping supports statements like:

- "These are the strongest public paper families behind modern dealiasing."
- "A closed app could plausibly use one or more of these families."
- "Observed behavior may be consistent with continuity, reference-wind, or
  temporal assistance."

It does not support statements like:

- "App X definitely uses Zhang and Wang 2006."
- "App Y is definitely a clone of 2DVDA."
- "A different visual output proves a proprietary algorithm."

That distinction matters. Public family mapping is useful; invented certainty is
not.
