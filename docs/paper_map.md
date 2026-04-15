# Paper map

This library intentionally names methods after the paper families they came from.

## `dealias_radial_es90`

Closest public source:

- Eilts, M. D., and S. D. Smith (1990), *Efficient Dealiasing of Doppler Velocities Using Local Environment Constraints*

What is implemented:

- radial-by-radial continuity,
- optional background reference anchoring,
- local extrapolation and step limiting.

## `dealias_sweep_zw06`

Closest public sources:

- Jing, Z., and G. Wiener (1993), *Two-Dimensional Dealiasing of Doppler Velocities*
- Zhang, J., and S. Wang (2006), *An Automated 2D Multipass Doppler Radar Velocity Dealiasing Scheme*

What is implemented:

- low-texture / weak-wind seeding,
- sweep-wide horizontal continuity,
- strict-to-relaxed multipass propagation,
- optional external reference anchoring.

## `dealias_sweep_xu11`

Closest public source:

- Xu, Q., K. Nai, L. Wei, P. Zhang, S. Liu, and D. Parrish (2011), *A VAD-Based Dealiasing Method for Radar Velocity Data Quality Control*

What is implemented:

- iterative uniform-wind VAD fitting,
- unwrap-and-refit loops,
- VAD reference construction,
- optional 2D multipass refinement.

## `dealias_sweep_jh01` / `dealias_volume_jh01`

Closest public source:

- James, C. N., and R. A. Houze Jr. (2001), *A Real-Time Four-Dimensional Doppler Dealiasing Scheme*

What is implemented:

- previous-volume reference anchoring,
- descending-elevation processing,
- optional background wind fusion,
- 2D multipass cleanup on top of temporal guidance.

## Related open implementations worth comparing

- Py-ART `dealias_region_based`
- Py-ART `dealias_fourdd`
- UNRAVEL
- R2D2

Those projects are highly relevant, but this repo keeps the implementation surface small and portable.
