# Annotated bibliography

This is a curated bibliography for public radar velocity dealiasing work and closely related operational references. The goal is not to be exhaustive; it is to cover the papers and resources most useful for understanding the algorithmic landscape.

## Foundational and classic methods

### Ray, P. S., and C. Ziegler, 1977: *De-Aliasing First-Moment Doppler Estimates.*

A compact early paper that frames dealiasing as a continuity problem and remains useful as a conceptual baseline. It is still one of the cleanest entry points for understanding why local continuity can work at all.

### Bargen, D. W., and R. C. Brown, 1980: *Interactive Radar Velocity Unfolding.*

An influential early operational-style contribution. Useful for understanding the history of practical human-in-the-loop and continuity-driven unfolding before later automation matured.

### Bergen, W. R., and S. C. Albers, 1988: *Two- and Three-Dimensional De-aliasing of Doppler Radar Velocities.*

A key classic for moving beyond 1D ray logic into 2D and 3D structure. Important historically because it makes clear that broader spatial context can improve continuity while also introducing new failure paths.

### Eilts, M. D., and S. D. Smith, 1990: *Efficient Dealiasing of Doppler Velocities Using Local Environment Constraints.*

One of the most important practical continuity papers. It anchors the idea that local environment constraints can make dealiasing efficient enough for operational settings.

### Jing, Z., and G. Wiener, 1993: *Two-Dimensional Dealiasing of Doppler Velocities.*

A foundational 2D dealiasing paper. Still important for understanding how sweep-level organization can outperform independent radial handling.

## Operational and public-weather-radar references

### NOAA Radar Operations Center (ROC): *2DVDA (Two-Dimensional Velocity Dealiasing Algorithm).*

Public ROC material that identifies 2DVDA as the default WSR-88D algorithm contributing to Level 3 elevation-based velocity products. Essential for any public discussion of U.S. operational velocity dealiasing.

### WSR-88D legacy velocity dealiasing algorithm documentation.

Useful for understanding older continuity-oriented operational logic, including use of neighboring gates/rays and environmental wind information. Important as a public contrast to newer 2D operational methods.

### ROC material on MPDA / VCP 112 / multi-PRF processing.

Important for understanding that some ambiguity problems are changed upstream by acquisition strategy. Crucial when comparing post-processing methods fairly.

### NSSL Doppler radar velocity guide and associated educational references.

Excellent for teaching aliasing, Nyquist limits, and why strong winds or tropical systems cause repeated folding. A good public reference for examples and terminology.

## Temporal, reference-assisted, and data-assimilation-oriented methods

### James, C. N., and R. A. Houze Jr., 2001: *A Real-Time Four-Dimensional Doppler Dealiasing Scheme.*

A cornerstone temporal/4D reference. Very useful for understanding how prior-volume information can stabilize dealiasing in difficult low-Nyquist situations.

### Gong, J., L. Wang, and Q. Xu, 2003: *A Three-Step Dealiasing Method for Doppler Velocity Data Quality Control.*

Important because it explicitly combines multiple ideas, including reference-field construction and staged processing. Good for readers building hybrid pipelines.

### Gao, J., and K. K. Droegemeier, 2004: *A Variational Technique for Dealiasing Doppler Radial Velocity Data.*

A clear entry in the variational family. Valuable for understanding how dealiasing can be posed as a broader optimization problem rather than only a local continuity walk.

### Haase, G., and T. Landelius, 2004: *Dealiasing of Doppler Radar Velocities Using a Torus Mapping.*

One of the best-known global/geometric formulations. Worth reading because it shows both the elegance and the assumption sensitivity of global approaches.

### Lim, E., and J. Sun, 2010: *A Velocity Dealiasing Technique Using Rapidly Updated Analysis from a Four-Dimensional Variational Doppler Radar Data Assimilation System.*

Important for assimilation-oriented workflows. Shows how an updated analysis can act as a dynamic reference field, with the associated benefits and risks of dependence on model-quality background flow.

### Xu, Q., K. Nai, L. Wei, and coauthors, 2011: *A VAD-Based Dealiasing Method for Radar Velocity Data Quality Control.*

A valuable public reference for VAD-based background estimation in automated QC and dealiasing. Useful when thinking about lightweight environmental references.

## Modern operationally practical region methods

### Zhang, J., and S. Wang, 2006: *An Automated 2D Multipass Doppler Radar Velocity Dealiasing Scheme.*

A major operationally relevant paper that combines 2D structure with multipass logic. Very useful bridge between classic 2D theory and more modern practical pipelines.

### Feldmann, M., C. N. James, M. Boscacci, D. Leuenberger, M. Gabella, and A. M. Robertson, 2020: *R2D2: A Region-Based Recursive Doppler Dealiasing Algorithm for Operational Weather Radar.*

A modern region-based method built specifically for operational weather radar and high-shear situations. Particularly relevant for mesocyclones and convergence lines.

### Louf, V., and coauthors, 2020: *UNRAVEL: A Robust Modular Velocity Dealiasing Technique for Doppler Radar.*

A key open-source-oriented modern reference. Important because it is modular, practical, and explicitly designed to work without external reference velocity data.

## Open software references

### Py-ART: `dealias_region_based`

One of the most important open reference implementations available to developers. The public docs and source make it an ideal baseline for experimentation and comparison.

### Py-ART: `dealias_fourdd`

A practical open implementation of a temporal / reference-assisted family. Especially useful for understanding the interface and data requirements of 4D-style methods in open software.

### PyDDA user guidance on gate filtering and velocity texture

Useful not because it is a dedicated dealiasing package, but because it clearly demonstrates how quality control, texture thresholds, and gate filters strongly affect downstream velocity processing.

## Later operational and evaluation work

### He, G., J. Sun, Z. Ying, and L. Zhang, 2019: *A Radar Radial Velocity Dealiasing Algorithm for Radar Data Assimilation and its Evaluation with Observations from Multiple Radar Networks.*

Important because it evaluates a dealiasing algorithm across multiple radar networks and severe-weather regimes. Good for developers who care about robustness across hardware and scan differences.

### Alford, A. A., and coauthors, 2022: work on staggered pulse repetition time and related operational correction issues.

Useful for understanding the modern relationship between dual-PRF/staggered-PRT acquisition and downstream correction quality. Important for fair benchmarking.

## Learned approaches

### Veillette, M. S., and coauthors, 2023: *A Deep Learning–Based Velocity Dealiasing Algorithm Derived from the WSR-88D Open Radar Product Generator.*

A serious public example of learned dealiasing intended to emulate an existing operational target. Valuable both for what it shows is possible and for highlighting the dependence of ML on the chosen training reference.

## How to use this bibliography

A good reading order is:

1. Ray and Ziegler (1977)
2. Eilts and Smith (1990)
3. Jing and Wiener (1993)
4. James and Houze (2001)
5. Zhang and Wang (2006)
6. ROC 2DVDA material
7. Py-ART region-based and fourdd docs
8. R2D2 and UNRAVEL
9. Gao and Droegemeier / Haase and Landelius / Lim and Sun
10. Veillette et al. (2023)

## A note on proprietary app discussions

If you are reading this bibliography in order to infer what a commercial app “must” be doing, resist that impulse. Public literature can tell you what method families are available and plausible. It does not prove that a closed-source app uses any one of them unless there is public documentation saying so.
