use ndarray::{
    Array2, Array3, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD, Axis, IxDyn, Zip,
};
use std::collections::{HashMap, HashSet, VecDeque};

mod types;
pub use types::*;

#[inline]
fn validate_nyquist(nyquist: f64) -> Result<()> {
    if nyquist <= 0.0 {
        Err(DealiasError::InvalidNyquist(nyquist))
    } else {
        Ok(())
    }
}

#[inline]
fn is_finite(value: f64) -> bool {
    value.is_finite()
}

#[inline]
fn wrap_scalar(value: f64, nyquist: f64) -> f64 {
    (value + nyquist).rem_euclid(2.0 * nyquist) - nyquist
}

#[inline]
fn gaussian_confidence_scalar(mismatch: f64, scale: f64) -> f64 {
    let safe_scale = scale.max(1e-6);
    if mismatch.is_finite() {
        (-0.5 * (mismatch / safe_scale).powi(2)).exp()
    } else {
        0.0
    }
}

fn nanmedian_small(values: &[f64]) -> Option<f64> {
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = finite.len() / 2;
    if finite.len() % 2 == 1 {
        Some(finite[mid])
    } else {
        Some(0.5 * (finite[mid - 1] + finite[mid]))
    }
}

#[inline]
fn unfold_scalar(obs: f64, reference: f64, nyquist: f64, max_abs_fold: i16) -> f64 {
    let limit = max_abs_fold as f64;
    let folds = ((reference - obs) / (2.0 * nyquist))
        .round_ties_even()
        .clamp(-limit, limit);
    obs + 2.0 * nyquist * folds
}

fn quantile_linear(mut values: Vec<f64>, q: f64) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if values.len() == 1 {
        return Some(values[0]);
    }
    let q = q.clamp(0.0, 1.0);
    let h = q * (values.len() - 1) as f64;
    let lower = h.floor() as usize;
    let upper = h.ceil() as usize;
    if lower == upper {
        Some(values[lower])
    } else {
        let weight = h - lower as f64;
        Some(values[lower] * (1.0 - weight) + values[upper] * weight)
    }
}

pub fn wrap_to_nyquist(velocity: ArrayViewD<'_, f64>, nyquist: f64) -> Result<ArrayD<f64>> {
    validate_nyquist(nyquist)?;
    let mut out = ArrayD::from_elem(velocity.raw_dim(), f64::NAN);
    Zip::from(out.view_mut())
        .and(velocity)
        .for_each(|out_value, &vel| {
            if is_finite(vel) {
                *out_value = (vel + nyquist).rem_euclid(2.0 * nyquist) - nyquist;
            }
        });
    Ok(out)
}

pub fn fold_counts(
    unfolded: ArrayViewD<'_, f64>,
    observed: ArrayViewD<'_, f64>,
    nyquist: f64,
) -> Result<ArrayD<i16>> {
    validate_nyquist(nyquist)?;
    let mut out = ArrayD::from_elem(unfolded.raw_dim(), 0i16);
    Zip::from(out.view_mut())
        .and(unfolded)
        .and(observed)
        .for_each(|out_value, &unfolded_value, &observed_value| {
            let ratio = (unfolded_value - observed_value) / (2.0 * nyquist);
            let rounded = ratio.round_ties_even();
            *out_value = if rounded.is_finite() {
                rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16
            } else {
                0
            };
        });
    Ok(out)
}

pub fn unfold_to_reference(
    observed: ArrayViewD<'_, f64>,
    reference: ArrayViewD<'_, f64>,
    nyquist: f64,
    max_abs_fold: i16,
) -> Result<ArrayD<f64>> {
    validate_nyquist(nyquist)?;
    if max_abs_fold < 0 {
        return Err(DealiasError::InvalidMaxAbsFold(max_abs_fold));
    }
    let limit = max_abs_fold as f64;
    let mut out = ArrayD::from_elem(observed.raw_dim(), f64::NAN);
    Zip::from(out.view_mut())
        .and(observed)
        .and(reference)
        .for_each(|out_value, &obs_value, &ref_value| {
            if is_finite(obs_value) && is_finite(ref_value) {
                let folds = ((ref_value - obs_value) / (2.0 * nyquist))
                    .round_ties_even()
                    .clamp(-limit, limit);
                *out_value = obs_value + 2.0 * nyquist * folds;
            }
        });
    Ok(out)
}

pub fn shift2d(
    field: ArrayView2<'_, f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> Result<Array2<f64>> {
    let (rows, cols) = field.dim();
    let mut out = Array2::from_elem((rows, cols), f64::NAN);
    for out_az in 0..rows {
        let src_az = if wrap_azimuth {
            Some((out_az as isize - shift_az).rem_euclid(rows as isize) as usize)
        } else {
            let idx = out_az as isize - shift_az;
            (0..rows as isize).contains(&idx).then_some(idx as usize)
        };
        if let Some(src_az) = src_az {
            for out_range in 0..cols {
                let src_range = out_range as isize - shift_range;
                if (0..cols as isize).contains(&src_range) {
                    out[(out_az, out_range)] = field[(src_az, src_range as usize)];
                }
            }
        }
    }
    Ok(out)
}

pub fn shift3d(
    volume: ArrayView3<'_, f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> Result<Array3<f64>> {
    let mut out = Array3::from_elem(volume.dim(), f64::NAN);
    for (index, in_slice) in volume.axis_iter(Axis(0)).enumerate() {
        let shifted = shift2d(in_slice, shift_az, shift_range, wrap_azimuth)?;
        out.index_axis_mut(Axis(0), index).assign(&shifted);
    }
    Ok(out)
}

pub fn neighbor_stack(
    field: ArrayView2<'_, f64>,
    include_diagonals: bool,
    wrap_azimuth: bool,
) -> Result<Array3<f64>> {
    let mut offsets = vec![(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];
    if include_diagonals {
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)]);
    }
    let (rows, cols) = field.dim();
    let mut out = Array3::from_elem((offsets.len(), rows, cols), f64::NAN);
    for (index, (da, dr)) in offsets.into_iter().enumerate() {
        let shifted = shift2d(field, da, dr, wrap_azimuth)?;
        out.index_axis_mut(Axis(0), index).assign(&shifted);
    }
    Ok(out)
}

pub fn texture_3x3(field: ArrayView2<'_, f64>, wrap_azimuth: bool) -> Result<Array2<f64>> {
    let (rows, cols) = field.dim();
    let mut out = Array2::from_elem((rows, cols), f64::NAN);
    let offsets = [
        (0isize, 0isize),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];

    for row in 0..rows {
        for col in 0..cols {
            let mut count = 0usize;
            let mut sum = 0.0f64;
            let mut sumsq = 0.0f64;
            for (da, dr) in offsets {
                let src_row = if wrap_azimuth {
                    (row as isize - da).rem_euclid(rows as isize) as usize
                } else {
                    let idx = row as isize - da;
                    if !(0..rows as isize).contains(&idx) {
                        continue;
                    }
                    idx as usize
                };
                let src_col = col as isize - dr;
                if !(0..cols as isize).contains(&src_col) {
                    continue;
                }
                let value = field[(src_row, src_col as usize)];
                if value.is_finite() {
                    count += 1;
                    sum += value;
                    sumsq += value * value;
                }
            }
            if count > 0 {
                let mean = sum / count as f64;
                let variance = (sumsq / count as f64) - mean * mean;
                out[(row, col)] = variance.max(0.0).sqrt();
            }
        }
    }
    Ok(out)
}

pub fn build_velocity_qc_mask(
    velocity: ArrayView2<'_, f64>,
    reflectivity: Option<ArrayView2<'_, f64>>,
    texture: Option<ArrayView2<'_, f64>>,
    min_reflectivity: Option<f64>,
    max_texture: Option<f64>,
    min_gate_fraction_in_ray: f64,
    wrap_azimuth: bool,
) -> Result<Array2<bool>> {
    if let Some(refl) = reflectivity {
        if refl.dim() != velocity.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reflectivity must match velocity shape",
            ));
        }
    }
    if let Some(tex) = texture {
        if tex.dim() != velocity.dim() {
            return Err(DealiasError::ShapeMismatch(
                "texture must match velocity shape",
            ));
        }
    }

    let computed_texture = if texture.is_none() && max_texture.is_some() {
        Some(texture_3x3(velocity, wrap_azimuth)?)
    } else {
        None
    };

    let (rows, cols) = velocity.dim();
    let mut mask = Array2::from_elem((rows, cols), false);
    for row in 0..rows {
        let mut row_valid = 0usize;
        for col in 0..cols {
            let vel = velocity[(row, col)];
            let mut keep = vel.is_finite();
            if keep {
                if let (Some(refl), Some(min_refl)) = (reflectivity, min_reflectivity) {
                    let refl_value = refl[(row, col)];
                    keep = refl_value.is_finite() && refl_value >= min_refl;
                }
            }
            if keep {
                if let Some(max_tex) = max_texture {
                    let tex_value = if let Some(tex) = texture {
                        tex[(row, col)]
                    } else if let Some(ref tex) = computed_texture {
                        tex[(row, col)]
                    } else {
                        f64::NAN
                    };
                    keep = tex_value.is_finite() && tex_value <= max_tex;
                }
            }
            mask[(row, col)] = keep;
            if keep {
                row_valid += 1;
            }
        }
        if (row_valid as f64 / cols as f64) < min_gate_fraction_in_ray {
            for col in 0..cols {
                mask[(row, col)] = false;
            }
        }
    }
    Ok(mask)
}

fn pick_seed(
    observed: ArrayView1<'_, f64>,
    reference: Option<ArrayView1<'_, f64>>,
) -> Option<usize> {
    let finite: Vec<usize> = observed
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect();
    if finite.is_empty() {
        return None;
    }
    if let Some(ref_field) = reference {
        let overlap: Vec<usize> = finite
            .iter()
            .copied()
            .filter(|index| ref_field[*index].is_finite())
            .collect();
        if !overlap.is_empty() {
            let center = overlap.iter().map(|idx| *idx as f64).sum::<f64>() / overlap.len() as f64;
            return overlap.into_iter().min_by(|a, b| {
                ((*a as f64 - center).abs())
                    .partial_cmp(&((*b as f64 - center).abs()))
                    .unwrap()
            });
        }
    }
    Some(finite[finite.len() / 2])
}

fn walk_radial(
    observed: ArrayView1<'_, f64>,
    nyquist: f64,
    corrected: &mut [f64],
    confidence: &mut [f64],
    reference: Option<ArrayView1<'_, f64>>,
    seed_index: usize,
    direction: isize,
    max_gap: usize,
    max_abs_step: Option<f64>,
) {
    let mut idx = seed_index as isize + direction;
    let mut last_valid = Some(seed_index);
    let mut last_valid_two: Option<usize> = None;

    while (0..observed.len() as isize).contains(&idx) {
        let i = idx as usize;
        let obs_value = observed[i];
        if !obs_value.is_finite() {
            idx += direction;
            continue;
        }

        let mut local_refs = Vec::with_capacity(3);
        if let Some(last) = last_valid {
            if i.abs_diff(last) <= max_gap && corrected[last].is_finite() {
                local_refs.push(corrected[last]);
                if let Some(last_two) = last_valid_two {
                    if corrected[last_two].is_finite() {
                        let slope = corrected[last] - corrected[last_two];
                        local_refs.push(corrected[last] + slope);
                    }
                }
            }
        }
        if let Some(ref_field) = reference {
            let ref_value = ref_field[i];
            if ref_value.is_finite() {
                local_refs.push(ref_value);
            }
        }

        if local_refs.is_empty() {
            corrected[i] = obs_value;
            confidence[i] = 0.15;
            last_valid_two = last_valid;
            last_valid = Some(i);
            idx += direction;
            continue;
        }

        let mut ref_value = nanmedian_small(&local_refs).unwrap_or(obs_value);
        let mut candidate = unfold_scalar(obs_value, ref_value, nyquist, 32);

        if let (Some(limit), Some(last)) = (max_abs_step, last_valid) {
            if corrected[last].is_finite() {
                let step = (candidate - corrected[last]).abs();
                if step > limit {
                    if let Some(ref_field) = reference {
                        let fallback_ref = ref_field[i];
                        if fallback_ref.is_finite() {
                            candidate = unfold_scalar(obs_value, fallback_ref, nyquist, 32);
                            ref_value = fallback_ref;
                        }
                    }
                }
            }
        }

        corrected[i] = candidate;
        confidence[i] = gaussian_confidence_scalar((candidate - ref_value).abs(), 0.45 * nyquist);
        last_valid_two = last_valid;
        last_valid = Some(i);
        idx += direction;
    }
}

pub fn dealias_radial_es90(
    observed: ArrayView1<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView1<'_, f64>>,
    seed_index: Option<usize>,
    max_gap: usize,
    max_abs_step: Option<f64>,
) -> Result<Es90RadialResult> {
    validate_nyquist(nyquist)?;
    if let Some(ref_field) = reference {
        if ref_field.len() != observed.len() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }

    let len = observed.len();
    let mut corrected = vec![f64::NAN; len];
    let mut confidence = vec![0.0f64; len];
    let reference_out = if let Some(ref_field) = reference {
        ArrayD::from_shape_vec(IxDyn(&[len]), ref_field.iter().copied().collect())
            .expect("shape preserved")
    } else {
        ArrayD::from_elem(IxDyn(&[len]), f64::NAN)
    };

    let chosen_seed = if let Some(seed) = seed_index {
        Some(seed)
    } else {
        pick_seed(observed, reference)
    };
    if let Some(seed) = chosen_seed {
        let obs_seed = observed[seed];
        if !obs_seed.is_finite() {
            return Err(DealiasError::ShapeMismatch(
                "seed_index points to a non-finite gate",
            ));
        }
        if let Some(ref_field) = reference {
            let ref_seed = ref_field[seed];
            if ref_seed.is_finite() {
                corrected[seed] = unfold_scalar(obs_seed, ref_seed, nyquist, 32);
                confidence[seed] = 0.98;
            } else {
                corrected[seed] = obs_seed;
                confidence[seed] = 0.80;
            }
        } else {
            corrected[seed] = obs_seed;
            confidence[seed] = 0.80;
        }

        walk_radial(
            observed,
            nyquist,
            &mut corrected,
            &mut confidence,
            reference,
            seed,
            1,
            max_gap,
            max_abs_step,
        );
        walk_radial(
            observed,
            nyquist,
            &mut corrected,
            &mut confidence,
            reference,
            seed,
            -1,
            max_gap,
            max_abs_step,
        );
    }

    let corrected_array =
        ArrayD::from_shape_vec(IxDyn(&[len]), corrected).expect("shape preserved");
    let confidence_array =
        ArrayD::from_shape_vec(IxDyn(&[len]), confidence).expect("shape preserved");
    let folds = fold_counts(corrected_array.view(), observed.into_dyn(), nyquist)?;

    Ok(Es90RadialResult {
        velocity: corrected_array,
        folds,
        confidence: confidence_array,
        reference: reference_out,
        seed_index: chosen_seed,
    })
}

pub fn dealias_sweep_es90(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    max_gap: usize,
    max_abs_step: Option<f64>,
) -> Result<Es90SweepResult> {
    validate_nyquist(nyquist)?;
    if let Some(ref_field) = reference {
        if ref_field.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }

    let (rows, cols) = observed.dim();
    let mut corrected = Array2::from_elem((rows, cols), f64::NAN);
    let mut confidence = Array2::from_elem((rows, cols), 0.0f64);

    for row in 0..rows {
        let mut combined_ref = vec![f64::NAN; cols];
        for col in 0..cols {
            let prev_value = if row > 0 {
                corrected[(row - 1, col)]
            } else {
                f64::NAN
            };
            let ref_value = reference
                .map(|ref_field| ref_field[(row, col)])
                .unwrap_or(f64::NAN);
            combined_ref[col] = match (prev_value.is_finite(), ref_value.is_finite()) {
                (true, true) => 0.5 * (prev_value + ref_value),
                (true, false) => prev_value,
                (false, true) => ref_value,
                (false, false) => f64::NAN,
            };
        }
        let combined_ref_array =
            ArrayD::from_shape_vec(IxDyn(&[cols]), combined_ref).expect("shape preserved");
        let radial = dealias_radial_es90(
            observed.row(row),
            nyquist,
            Some(
                combined_ref_array
                    .view()
                    .into_dimensionality()
                    .expect("1d view"),
            ),
            None,
            max_gap,
            max_abs_step,
        )?;
        let radial_velocity = radial
            .velocity
            .into_dimensionality::<ndarray::Ix1>()
            .expect("1d");
        let radial_confidence = radial
            .confidence
            .into_dimensionality::<ndarray::Ix1>()
            .expect("1d");
        corrected.row_mut(row).assign(&radial_velocity);
        confidence.row_mut(row).assign(&radial_confidence);
    }

    let corrected_array = corrected.into_dyn();
    let folds = fold_counts(corrected_array.view(), observed.into_dyn(), nyquist)?;
    let reference_out = if let Some(ref_field) = reference {
        ref_field.to_owned().into_dyn()
    } else {
        ArrayD::from_elem(IxDyn(&[rows, cols]), f64::NAN)
    };

    Ok(Es90SweepResult {
        velocity: corrected_array,
        folds,
        confidence: confidence.into_dyn(),
        reference: reference_out,
    })
}

pub fn dealias_sweep_zw06(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    weak_threshold_fraction: f64,
    wrap_azimuth: bool,
    max_iterations_per_pass: usize,
    include_diagonals: bool,
    recenter_without_reference: bool,
) -> Result<Zw06Result> {
    validate_nyquist(nyquist)?;
    if let Some(ref_field) = reference {
        if ref_field.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }

    let (rows, cols) = observed.dim();
    let valid = observed.mapv(|value| value.is_finite());
    let mut corrected = Array2::from_elem((rows, cols), f64::NAN);
    let mut confidence = Array2::from_elem((rows, cols), 0.0f64);
    let texture = texture_3x3(observed, wrap_azimuth)?;

    if let Some(ref_field) = reference {
        for row in 0..rows {
            for col in 0..cols {
                let obs_value = observed[(row, col)];
                let ref_value = ref_field[(row, col)];
                if obs_value.is_finite() && ref_value.is_finite() {
                    let candidate = unfold_scalar(obs_value, ref_value, nyquist, 32);
                    let mismatch = (candidate - ref_value).abs();
                    if mismatch <= 0.80 * nyquist {
                        corrected[(row, col)] = candidate;
                        confidence[(row, col)] =
                            gaussian_confidence_scalar(mismatch, 0.45 * nyquist);
                    }
                }
            }
        }
    }

    let weak_threshold = weak_threshold_fraction * nyquist;
    let texture_cut =
        quantile_linear(texture.iter().copied().collect(), 0.25).unwrap_or(weak_threshold);
    for row in 0..rows {
        for col in 0..cols {
            let unresolved = valid[(row, col)] && !corrected[(row, col)].is_finite();
            if unresolved {
                let obs_value = observed[(row, col)];
                let tex_value = texture[(row, col)];
                if obs_value.abs() <= weak_threshold || tex_value <= texture_cut {
                    corrected[(row, col)] = obs_value;
                    confidence[(row, col)] = 0.72;
                }
            }
        }
    }

    if !corrected.iter().any(|value| value.is_finite()) && valid.iter().any(|value| *value) {
        let mut best_index = None;
        let mut best_abs = f64::INFINITY;
        for row in 0..rows {
            for col in 0..cols {
                if valid[(row, col)] {
                    let value = observed[(row, col)].abs();
                    if value < best_abs {
                        best_abs = value;
                        best_index = Some((row, col));
                    }
                }
            }
        }
        if let Some((row, col)) = best_index {
            corrected[(row, col)] = observed[(row, col)];
            confidence[(row, col)] = 0.55;
        }
    }

    let passes: [(usize, f64, bool); 3] = [(3, 0.35, false), (2, 0.60, true), (1, 1.10, true)];
    let mut iterations_used = 0usize;

    for (min_neighbors, mismatch_fraction, reference_only) in passes {
        for _ in 0..max_iterations_per_pass {
            iterations_used += 1;
            let unresolved_any = corrected
                .iter()
                .zip(valid.iter())
                .any(|(corrected_value, valid_value)| *valid_value && !corrected_value.is_finite());
            if !unresolved_any {
                break;
            }

            let neigh = neighbor_stack(corrected.view(), include_diagonals, wrap_azimuth)?;
            let mut neigh_count = Array2::from_elem((rows, cols), 0usize);
            let mut neigh_ref = Array2::from_elem((rows, cols), f64::NAN);
            for row in 0..rows {
                for col in 0..cols {
                    let mut values = Vec::with_capacity(neigh.shape()[0]);
                    for layer in 0..neigh.shape()[0] {
                        let value = neigh[(layer, row, col)];
                        if value.is_finite() {
                            values.push(value);
                        }
                    }
                    neigh_count[(row, col)] = values.len();
                    if let Some(median) = nanmedian_small(&values) {
                        neigh_ref[(row, col)] = median;
                    }
                }
            }

            let mut assign_count = 0usize;
            for row in 0..rows {
                for col in 0..cols {
                    let unresolved = valid[(row, col)] && !corrected[(row, col)].is_finite();
                    if !unresolved {
                        continue;
                    }

                    let mut combined_ref = neigh_ref[(row, col)];
                    if let Some(ref_field) = reference {
                        let ref_value = ref_field[(row, col)];
                        let sparse = !combined_ref.is_finite()
                            || neigh_count[(row, col)] < usize::max(2, min_neighbors);
                        if sparse && ref_value.is_finite() {
                            combined_ref = ref_value;
                        }
                        let blend = combined_ref.is_finite()
                            && ref_value.is_finite()
                            && neigh_count[(row, col)] < (min_neighbors + 1);
                        if blend {
                            combined_ref = 0.5 * (neigh_ref[(row, col)] + ref_value);
                        }
                    }
                    if !combined_ref.is_finite() {
                        continue;
                    }

                    let candidate = unfold_scalar(observed[(row, col)], combined_ref, nyquist, 32);
                    let mismatch = (candidate - combined_ref).abs();
                    let mut enough_neighbors = neigh_count[(row, col)] >= min_neighbors;
                    if reference_only {
                        if let Some(ref_field) = reference {
                            enough_neighbors =
                                enough_neighbors || ref_field[(row, col)].is_finite();
                        }
                    }
                    if enough_neighbors && mismatch <= mismatch_fraction * nyquist {
                        corrected[(row, col)] = candidate;
                        confidence[(row, col)] = confidence[(row, col)]
                            .max(gaussian_confidence_scalar(mismatch, 0.40 * nyquist));
                        assign_count += 1;
                    }
                }
            }

            if assign_count == 0 {
                break;
            }
        }
    }

    if let Some(ref_field) = reference {
        for row in 0..rows {
            for col in 0..cols {
                let unresolved = valid[(row, col)] && !corrected[(row, col)].is_finite();
                if unresolved {
                    let ref_value = ref_field[(row, col)];
                    if ref_value.is_finite() {
                        let candidate = unfold_scalar(observed[(row, col)], ref_value, nyquist, 32);
                        let mismatch = (candidate - ref_value).abs();
                        if mismatch <= 1.10 * nyquist {
                            corrected[(row, col)] = candidate;
                            confidence[(row, col)] = confidence[(row, col)].max(0.45);
                        }
                    }
                }
            }
        }
    }

    let neigh = neighbor_stack(corrected.view(), include_diagonals, wrap_azimuth)?;
    for row in 0..rows {
        for col in 0..cols {
            let unresolved = valid[(row, col)] && !corrected[(row, col)].is_finite();
            if unresolved {
                let mut values = Vec::with_capacity(neigh.shape()[0]);
                for layer in 0..neigh.shape()[0] {
                    let value = neigh[(layer, row, col)];
                    if value.is_finite() {
                        values.push(value);
                    }
                }
                if let Some(neigh_median) = nanmedian_small(&values) {
                    corrected[(row, col)] =
                        unfold_scalar(observed[(row, col)], neigh_median, nyquist, 32);
                    confidence[(row, col)] = confidence[(row, col)].max(0.25);
                }
            }
        }
    }

    if let Some(ref_field) = reference {
        if corrected.iter().any(|value| value.is_finite()) {
            let mut cleanup_ref = Array2::from_elem((rows, cols), f64::NAN);
            for row in 0..rows {
                for col in 0..cols {
                    cleanup_ref[(row, col)] = match (
                        corrected[(row, col)].is_finite(),
                        ref_field[(row, col)].is_finite(),
                    ) {
                        (true, true) => 0.5 * (corrected[(row, col)] + ref_field[(row, col)]),
                        (true, false) => corrected[(row, col)],
                        (false, true) => ref_field[(row, col)],
                        (false, false) => f64::NAN,
                    };
                }
            }
            let radial = dealias_sweep_es90(observed, nyquist, Some(cleanup_ref.view()), 3, None)?;
            corrected = radial
                .velocity
                .into_dimensionality::<ndarray::Ix2>()
                .expect("2d");
            let radial_confidence = radial
                .confidence
                .into_dimensionality::<ndarray::Ix2>()
                .expect("2d");
            for row in 0..rows {
                for col in 0..cols {
                    confidence[(row, col)] =
                        confidence[(row, col)].max(radial_confidence[(row, col)]);
                }
            }
        }
    } else if recenter_without_reference && corrected.iter().any(|value| value.is_finite()) {
        let fold_array = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?;
        let finite_folds: Vec<f64> = corrected
            .iter()
            .zip(fold_array.iter())
            .filter_map(|(corrected_value, fold_value)| {
                corrected_value.is_finite().then_some(*fold_value as f64)
            })
            .collect();
        if let Some(median_fold) = nanmedian_small(&finite_folds) {
            let shift = median_fold.round_ties_even() as i32;
            if shift != 0 {
                let delta = 2.0 * nyquist * shift as f64;
                corrected.mapv_inplace(|value| {
                    if value.is_finite() {
                        value - delta
                    } else {
                        value
                    }
                });
            }
        }
    }

    let corrected_array = corrected.into_dyn();
    let folds = fold_counts(corrected_array.view(), observed.into_dyn(), nyquist)?;
    let seeded_gates = confidence.iter().filter(|value| **value >= 0.70).count();
    let assigned_gates = corrected_array
        .iter()
        .filter(|value| value.is_finite())
        .count();
    let reference_out = if let Some(ref_field) = reference {
        ref_field.to_owned().into_dyn()
    } else {
        ArrayD::from_elem(IxDyn(&[rows, cols]), f64::NAN)
    };

    Ok(Zw06Result {
        velocity: corrected_array,
        folds,
        confidence: confidence.into_dyn(),
        reference: reference_out,
        seeded_gates,
        assigned_gates,
        iterations_used,
    })
}

pub fn dealias_sweep_variational_refine(
    observed: ArrayView2<'_, f64>,
    initial_corrected: ArrayView2<'_, f64>,
    reference: Option<ArrayView2<'_, f64>>,
    nyquist: f64,
    max_abs_fold: i16,
    neighbor_weight: f64,
    reference_weight: f64,
    smoothness_weight: f64,
    max_iterations: usize,
    wrap_azimuth: bool,
) -> Result<VariationalResult> {
    validate_nyquist(nyquist)?;
    if initial_corrected.dim() != observed.dim() {
        return Err(DealiasError::ShapeMismatch(
            "initial_corrected must match observed shape",
        ));
    }
    if let Some(ref_field) = reference {
        if ref_field.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }

    let (rows, cols) = observed.dim();
    let mut corrected = initial_corrected.to_owned();
    for row in 0..rows {
        for col in 0..cols {
            if !observed[(row, col)].is_finite() {
                corrected[(row, col)] = f64::NAN;
            }
        }
    }

    let mut folds_array = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
    let mut iterations_used = 0usize;
    let mut changed_total = 0usize;
    let mut last_neigh_mean = Array2::from_elem((rows, cols), f64::NAN);

    for _ in 0..max_iterations {
        iterations_used += 1;
        let neigh = neighbor_stack(corrected.view(), true, wrap_azimuth)?;
        last_neigh_mean.fill(f64::NAN);
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for layer in 0..neigh.shape()[0] {
                    let value = neigh[(layer, row, col)];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
                if count > 0 {
                    last_neigh_mean[(row, col)] = sum / count as f64;
                }
            }
        }

        let mut changed = 0usize;
        for row in 0..rows {
            for col in 0..cols {
                let obs_value = observed[(row, col)];
                if !obs_value.is_finite() {
                    continue;
                }
                let center_fold = folds_array[(row, col)] as i32;
                let start_fold = i32::max(-(max_abs_fold as i32), center_fold - 2);
                let end_fold = i32::min(max_abs_fold as i32, center_fold + 2);
                let mut best_fold = center_fold;
                let mut best_value = corrected[(row, col)];
                let mut best_score = f64::INFINITY;

                let mut local_neighbors = Vec::with_capacity(neigh.shape()[0]);
                for layer in 0..neigh.shape()[0] {
                    let value = neigh[(layer, row, col)];
                    if value.is_finite() {
                        local_neighbors.push(value);
                    }
                }
                let local_ref = reference.and_then(|ref_field| {
                    let value = ref_field[(row, col)];
                    value.is_finite().then_some(value)
                });
                let local_mean = {
                    let value = last_neigh_mean[(row, col)];
                    value.is_finite().then_some(value)
                };

                for fold in start_fold..=end_fold {
                    let candidate = obs_value + 2.0 * nyquist * fold as f64;
                    let mut score = smoothness_weight * (fold.unsigned_abs() as f64);
                    if !local_neighbors.is_empty() {
                        let mse = local_neighbors
                            .iter()
                            .map(|neighbor| {
                                let diff = candidate - *neighbor;
                                diff * diff
                            })
                            .sum::<f64>()
                            / local_neighbors.len() as f64;
                        score += neighbor_weight * mse;
                    } else if let Some(mean) = local_mean {
                        let diff = candidate - mean;
                        score += neighbor_weight * diff * diff;
                    }
                    if let Some(reference_value) = local_ref {
                        let diff = candidate - reference_value;
                        score += reference_weight * diff * diff;
                    }
                    if score < best_score {
                        best_score = score;
                        best_fold = fold;
                        best_value = candidate;
                    }
                }

                if best_fold != center_fold {
                    folds_array[(row, col)] = best_fold as i16;
                    corrected[(row, col)] = best_value;
                    changed += 1;
                }
            }
        }

        changed_total += changed;
        if changed == 0 {
            break;
        }
    }

    let mut confidence = Array2::from_elem((rows, cols), 0.0f64);
    for row in 0..rows {
        for col in 0..cols {
            let obs_value = observed[(row, col)];
            if !obs_value.is_finite() {
                continue;
            }
            let target_value = if let Some(ref_field) = reference {
                let ref_value = ref_field[(row, col)];
                if ref_value.is_finite() {
                    ref_value
                } else {
                    last_neigh_mean[(row, col)]
                }
            } else {
                last_neigh_mean[(row, col)]
            };
            confidence[(row, col)] = gaussian_confidence_scalar(
                (corrected[(row, col)] - target_value).abs(),
                0.45 * nyquist,
            );
        }
    }

    Ok(VariationalResult {
        velocity: corrected.into_dyn(),
        folds: folds_array.into_dyn(),
        confidence: confidence.into_dyn(),
        iterations_used,
        changed_gates: changed_total,
    })
}

pub fn dealias_sweep_variational(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    block_shape: Option<(usize, usize)>,
    bootstrap_reference_weight: f64,
    bootstrap_iterations: usize,
    bootstrap_max_abs_fold: i16,
    bootstrap_min_region_area: usize,
    bootstrap_min_valid_fraction: f64,
    max_abs_fold: i16,
    neighbor_weight: f64,
    reference_weight: f64,
    smoothness_weight: f64,
    max_iterations: usize,
    wrap_azimuth: bool,
) -> Result<VariationalSweepResult> {
    let bootstrap = build_variational_bootstrap(
        observed,
        nyquist,
        reference,
        block_shape,
        bootstrap_reference_weight,
        bootstrap_iterations,
        bootstrap_max_abs_fold,
        bootstrap_min_region_area,
        bootstrap_min_valid_fraction,
        wrap_azimuth,
    )?;
    let refined = dealias_sweep_variational_refine(
        observed,
        bootstrap.initial.view(),
        reference,
        nyquist,
        max_abs_fold,
        neighbor_weight,
        reference_weight,
        smoothness_weight,
        max_iterations,
        wrap_azimuth,
    )?;

    Ok(VariationalSweepResult {
        velocity: refined.velocity,
        folds: refined.folds,
        confidence: refined.confidence,
        reference: bootstrap.reference.into_dyn(),
        method: if bootstrap.method == "2d_multipass" {
            "zw06_bootstrap_then_coordinate_descent"
        } else {
            "region_graph_bootstrap_then_coordinate_descent"
        },
        bootstrap_method: bootstrap.method,
        bootstrap_region_count: bootstrap.region_count,
        bootstrap_unresolved_regions: bootstrap.unresolved_regions,
        bootstrap_skipped_sparse_blocks: bootstrap.skipped_sparse_blocks,
        bootstrap_assigned_gates: bootstrap.assigned_gates,
        bootstrap_iterations_used: bootstrap.iterations_used,
        bootstrap_safety_fallback_applied: bootstrap.safety_fallback_applied,
        bootstrap_safety_fallback_reason: bootstrap.safety_fallback_reason,
        iterations_used: refined.iterations_used,
        changed_gates: refined.changed_gates,
    })
}

pub fn dealias_dual_prf(
    low: ArrayViewD<'_, f64>,
    high: ArrayViewD<'_, f64>,
    low_nyquist: f64,
    high_nyquist: f64,
    reference: Option<ArrayViewD<'_, f64>>,
    max_abs_fold: i16,
) -> Result<DualPrfResult> {
    validate_nyquist(low_nyquist)?;
    validate_nyquist(high_nyquist)?;
    if max_abs_fold < 0 {
        return Err(DealiasError::InvalidMaxAbsFold(max_abs_fold));
    }
    if low.shape() != high.shape() {
        return Err(DealiasError::ShapeMismatch(
            "low and high observations must have the same shape",
        ));
    }
    if let Some(ref ref_field) = reference {
        if ref_field.shape() != low.shape() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match the observed shape",
            ));
        }
    }

    let scale = low_nyquist.max(high_nyquist).max(1.0);
    let pair_scale = 0.40 * low_nyquist.min(high_nyquist);
    let ref_scale = 0.55 * low_nyquist.max(high_nyquist);
    let dim = low.raw_dim();
    let len = low.len();
    let low_values: Vec<f64> = low.iter().copied().collect();
    let high_values: Vec<f64> = high.iter().copied().collect();
    let reference_values: Option<Vec<f64>> =
        reference.map(|ref_field| ref_field.iter().copied().collect());

    let mut low_best = vec![f64::NAN; len];
    let mut high_best = vec![f64::NAN; len];
    let mut low_fold_values = vec![0i16; len];
    let mut high_fold_values = vec![0i16; len];
    let mut pair_gap_full = vec![f64::NAN; len];
    let mut combined = vec![f64::NAN; len];
    let mut confidence = vec![0.0f64; len];
    let mut branch_reference = vec![f64::NAN; len];

    let mut low_valid_gates = 0usize;
    let mut high_valid_gates = 0usize;
    let mut paired_gates = 0usize;
    let mut low_fold_sum = 0.0f64;
    let mut high_fold_sum = 0.0f64;
    let mut pair_gap_sum = 0.0f64;
    let mut pair_gap_max = f64::NAN;

    for idx in 0..len {
        let low_value = low_values[idx];
        let high_value = high_values[idx];
        let ref_value = reference_values
            .as_ref()
            .map(|values| values[idx])
            .unwrap_or(f64::NAN);
        let has_ref = ref_value.is_finite();
        let has_low = low_value.is_finite();
        let has_high = high_value.is_finite();

        if has_low {
            low_valid_gates += 1;
        }
        if has_high {
            high_valid_gates += 1;
        }

        let mut low_candidate = f64::NAN;
        let mut high_candidate = f64::NAN;
        let mut low_fold = 0i16;
        let mut high_fold = 0i16;

        if has_low && has_high {
            paired_gates += 1;

            let mut best_low_score = f64::INFINITY;
            let mut best_high_score = f64::INFINITY;

            for k in -max_abs_fold..=max_abs_fold {
                let candidate = low_value + 2.0 * low_nyquist * f64::from(k);
                let partner_gap = wrap_scalar(candidate - high_value, high_nyquist).abs();
                let mut score = partner_gap + 0.02 * f64::from(k.abs());
                if has_ref {
                    score += 0.10 * (candidate - ref_value).abs() / scale;
                }
                let better = score < best_low_score - 1e-12;
                let tied = (score - best_low_score).abs() <= 1e-12;
                let tie_break = if has_ref {
                    (candidate - ref_value).abs() < (low_candidate - ref_value).abs()
                } else {
                    candidate.abs() < low_candidate.abs()
                };
                if better || (tied && tie_break) {
                    best_low_score = score;
                    low_candidate = candidate;
                    low_fold = k;
                    pair_gap_full[idx] = partner_gap;
                }
            }

            for k in -max_abs_fold..=max_abs_fold {
                let candidate = high_value + 2.0 * high_nyquist * f64::from(k);
                let partner_gap = wrap_scalar(candidate - low_value, low_nyquist).abs();
                let mut score = partner_gap + 0.02 * f64::from(k.abs());
                if has_ref {
                    score += 0.10 * (candidate - ref_value).abs() / scale;
                }
                let better = score < best_high_score - 1e-12;
                let tied = (score - best_high_score).abs() <= 1e-12;
                let tie_break = if has_ref {
                    (candidate - ref_value).abs() < (high_candidate - ref_value).abs()
                } else {
                    candidate.abs() < high_candidate.abs()
                };
                if better || (tied && tie_break) {
                    best_high_score = score;
                    high_candidate = candidate;
                    high_fold = k;
                }
            }

            let branch_values = if has_ref {
                [low_candidate, high_candidate, ref_value]
            } else {
                [low_candidate, high_candidate, f64::NAN]
            };
            if let Some(median) = nanmedian_small(&branch_values) {
                combined[idx] = median;
            }
            let confidence_from_pair = gaussian_confidence_scalar(pair_gap_full[idx], pair_scale);
            let confidence_from_ref = if has_ref && combined[idx].is_finite() {
                gaussian_confidence_scalar((combined[idx] - ref_value).abs(), ref_scale)
            } else {
                0.0
            };
            confidence[idx] = confidence_from_pair.max(confidence_from_ref);
            pair_gap_sum += pair_gap_full[idx];
            pair_gap_max = if pair_gap_max.is_finite() {
                pair_gap_max.max(pair_gap_full[idx])
            } else {
                pair_gap_full[idx]
            };
            low_fold_sum += f64::from(low_fold);
            high_fold_sum += f64::from(high_fold);
        } else if has_low {
            low_candidate = if has_ref {
                let fold = ((ref_value - low_value) / (2.0 * low_nyquist))
                    .round_ties_even()
                    .clamp(-(max_abs_fold as f64), max_abs_fold as f64);
                low_value + 2.0 * low_nyquist * fold
            } else {
                low_value
            };
            combined[idx] = low_candidate;
            confidence[idx] = if has_ref { 0.80 } else { 0.55 };
        } else if has_high {
            high_candidate = if has_ref {
                let fold = ((ref_value - high_value) / (2.0 * high_nyquist))
                    .round_ties_even()
                    .clamp(-(max_abs_fold as f64), max_abs_fold as f64);
                high_value + 2.0 * high_nyquist * fold
            } else {
                high_value
            };
            combined[idx] = high_candidate;
            confidence[idx] = if has_ref { 0.80 } else { 0.55 };
        }

        low_best[idx] = low_candidate;
        high_best[idx] = high_candidate;
        low_fold_values[idx] = low_fold;
        high_fold_values[idx] = high_fold;

        if let Some(median) = nanmedian_small(&[low_candidate, high_candidate]) {
            branch_reference[idx] = median;
        }
    }

    let combined_array = ArrayD::from_shape_vec(dim.clone(), combined).expect("shape preserved");
    let folds = fold_counts(combined_array.view(), low, low_nyquist)?;
    let confidence_array =
        ArrayD::from_shape_vec(dim.clone(), confidence).expect("shape preserved");
    let reference_out = if let Some(ref_values) = reference_values {
        ArrayD::from_shape_vec(dim.clone(), ref_values).expect("shape preserved")
    } else {
        ArrayD::from_shape_vec(dim.clone(), branch_reference).expect("shape preserved")
    };

    Ok(DualPrfResult {
        velocity: combined_array,
        folds,
        confidence: confidence_array,
        reference: reference_out,
        low_valid_gates,
        high_valid_gates,
        paired_gates,
        low_branch_mean_fold: if paired_gates > 0 {
            low_fold_sum / paired_gates as f64
        } else {
            0.0
        },
        high_branch_mean_fold: if paired_gates > 0 {
            high_fold_sum / paired_gates as f64
        } else {
            0.0
        },
        mean_pair_gap: if paired_gates > 0 {
            pair_gap_sum / paired_gates as f64
        } else {
            f64::NAN
        },
        max_pair_gap: pair_gap_max,
    })
}

fn nanmedian_view2(view: ArrayView2<'_, f64>) -> Option<f64> {
    let mut values = Vec::with_capacity(view.len());
    for value in view.iter().copied() {
        if value.is_finite() {
            values.push(value);
        }
    }
    quantile_linear(values, 0.5)
}

fn choose_block_shape(
    shape: (usize, usize),
    block_shape: Option<(usize, usize)>,
) -> (usize, usize) {
    let (rows, cols) = shape;
    if let Some((br, bc)) = block_shape {
        return (br.max(1).min(rows), bc.max(1).min(cols));
    }

    let mut br = if rows >= 18 { rows / 18 } else { rows };
    let mut bc = if cols >= 18 { cols / 18 } else { cols };
    br = br.max(4).min(16);
    bc = bc.max(4).min(16);
    br = br.min(rows);
    bc = bc.min(cols);
    (br, bc)
}

const DEFAULT_REGION_GRAPH_MIN_REGION_AREA: usize = 4;
const DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION: f64 = 0.15;
const REGION_GRAPH_SAFETY_WEAK_THRESHOLD_FRACTION: f64 = 0.35;
const REGION_GRAPH_SAFETY_MAX_ITERATIONS_PER_PASS: usize = 12;
const REGION_GRAPH_SAFETY_COST_REL_MARGIN: f64 = 0.05;
const REGION_GRAPH_SAFETY_COST_ABS_MARGIN_FRACTION: f64 = 0.03;
const REGION_GRAPH_SAFETY_DISAGREEMENT_FRACTION: f64 = 0.02;
const REGION_GRAPH_SAFETY_COMPONENT_FRACTION: f64 = 0.01;
const DEFAULT_VARIATIONAL_BOOTSTRAP_REFERENCE_WEIGHT: f64 = 0.75;
const DEFAULT_VARIATIONAL_BOOTSTRAP_ITERATIONS: usize = 6;
const DEFAULT_VARIATIONAL_BOOTSTRAP_MAX_ABS_FOLD: i16 = 8;

#[derive(Clone, Debug)]
struct RegionGraphRegion {
    region_id: usize,
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
    mean_obs: f64,
    texture: f64,
    area: usize,
    density: f64,
    seedable: bool,
    reference_mean: Option<f64>,
    neighbors: Vec<usize>,
    boundary_weight: HashMap<usize, usize>,
}

fn push_neighbor(region: &mut RegionGraphRegion, neighbor_id: usize, weight: usize) {
    if !region.neighbors.contains(&neighbor_id) {
        region.neighbors.push(neighbor_id);
    }
    region
        .boundary_weight
        .entry(neighbor_id)
        .and_modify(|existing| *existing = (*existing).max(weight))
        .or_insert(weight);
}

fn add_region_edge(regions: &mut [RegionGraphRegion], a: usize, b: usize, weight: usize) {
    if a == b {
        return;
    }
    if a < b {
        let (left, right) = regions.split_at_mut(b);
        let region_a = &mut left[a];
        let region_b = &mut right[0];
        push_neighbor(region_a, b, weight);
        push_neighbor(region_b, a, weight);
    } else {
        let (left, right) = regions.split_at_mut(a);
        let region_a = &mut right[0];
        let region_b = &mut left[b];
        push_neighbor(region_a, b, weight);
        push_neighbor(region_b, a, weight);
    }
}

fn region_median(
    view: ArrayView2<'_, f64>,
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
) -> Option<f64> {
    let mut values = Vec::new();
    for row in row0..row1 {
        for col in col0..col1 {
            let value = view[(row, col)];
            if value.is_finite() {
                values.push(value);
            }
        }
    }
    quantile_linear(values, 0.5)
}

fn build_region_graph(
    observed: ArrayView2<'_, f64>,
    reference: Option<ArrayView2<'_, f64>>,
    block_shape: Option<(usize, usize)>,
    wrap_azimuth: bool,
    min_region_area: usize,
    min_valid_fraction: f64,
) -> Result<(Vec<RegionGraphRegion>, Array2<isize>, usize)> {
    let (rows, cols) = observed.dim();
    let (block_rows, block_cols) = choose_block_shape((rows, cols), block_shape);
    let texture_map = texture_3x3(observed, wrap_azimuth)?;

    let n_row_blocks = usize::max(1, (rows + block_rows - 1) / block_rows);
    let n_col_blocks = usize::max(1, (cols + block_cols - 1) / block_cols);
    let mut block_ids = Array2::from_elem((n_row_blocks, n_col_blocks), -1isize);
    let mut regions: Vec<RegionGraphRegion> = Vec::new();
    let mut skipped_sparse_blocks = 0usize;

    for bi in 0..n_row_blocks {
        let r0 = bi * block_rows;
        let r1 = usize::min(rows, r0 + block_rows);
        for bj in 0..n_col_blocks {
            let c0 = bj * block_cols;
            let c1 = usize::min(cols, c0 + block_cols);
            let block = observed.slice(ndarray::s![r0..r1, c0..c1]);
            let finite_count = block.iter().filter(|value| value.is_finite()).count();
            if finite_count == 0 {
                continue;
            }
            let total_cells = (r1 - r0) * (c1 - c0);
            let valid_fraction = finite_count as f64 / total_cells as f64;
            let mean_obs = nanmedian_view2(block).unwrap_or(f64::NAN);
            let seedable = finite_count >= min_region_area
                && valid_fraction >= min_valid_fraction
                && mean_obs.is_finite();
            if !seedable {
                skipped_sparse_blocks += 1;
                continue;
            }

            let region_id = regions.len();
            block_ids[(bi, bj)] = region_id as isize;
            let mut texture =
                nanmedian_view2(texture_map.slice(ndarray::s![r0..r1, c0..c1])).unwrap_or(0.0);
            if !texture.is_finite() {
                texture = 0.0;
            }
            let reference_mean = reference.and_then(|reference| region_median(reference, r0, r1, c0, c1));
            regions.push(RegionGraphRegion {
                region_id,
                row0: r0,
                row1: r1,
                col0: c0,
                col1: c1,
                mean_obs,
                texture,
                area: finite_count,
                density: valid_fraction,
                seedable: true,
                reference_mean,
                neighbors: Vec::new(),
                boundary_weight: HashMap::new(),
            });
        }
    }

    for bi in 0..n_row_blocks {
        for bj in 0..n_col_blocks {
            let region_id = block_ids[(bi, bj)];
            if region_id < 0 {
                continue;
            }
            let region_id = region_id as usize;
            if bj + 1 < n_col_blocks {
                let right_id = block_ids[(bi, bj + 1)];
                if right_id >= 0 && right_id as usize != region_id {
                    let right_id = right_id as usize;
                    let edge = usize::min(
                        regions[region_id].row1 - regions[region_id].row0,
                        regions[right_id].row1 - regions[right_id].row0,
                    );
                    add_region_edge(&mut regions, region_id, right_id, edge);
                }
            }
            if bi + 1 < n_row_blocks {
                let down_id = block_ids[(bi + 1, bj)];
                if down_id >= 0 && down_id as usize != region_id {
                    let down_id = down_id as usize;
                    let edge = usize::min(
                        regions[region_id].col1 - regions[region_id].col0,
                        regions[down_id].col1 - regions[down_id].col0,
                    );
                    add_region_edge(&mut regions, region_id, down_id, edge);
                }
            }
            if wrap_azimuth && n_row_blocks > 1 && bi == 0 {
                let wrap_id = block_ids[(n_row_blocks - 1, bj)];
                if wrap_id >= 0 && wrap_id as usize != region_id {
                    let wrap_id = wrap_id as usize;
                    let edge = usize::min(
                        regions[region_id].col1 - regions[region_id].col0,
                        regions[wrap_id].col1 - regions[wrap_id].col0,
                    );
                    add_region_edge(&mut regions, region_id, wrap_id, edge);
                }
            }
        }
    }

    if reference.is_some() {
        for region in &mut regions {
            if let Some(region_ref) = region.reference_mean {
                region.texture = region
                    .texture
                    .max(0.5 * (region.mean_obs - region_ref).abs());
            }
        }
    }

    Ok((regions, block_ids, skipped_sparse_blocks))
}

fn pick_seed_region(
    regions: &[RegionGraphRegion],
    reference: Option<ArrayView2<'_, f64>>,
) -> Option<usize> {
    if regions.is_empty() {
        return None;
    }

    let mut best_index: Option<usize> = None;
    let mut best_score = f64::INFINITY;
    for (index, region) in regions.iter().enumerate() {
        if !region.seedable {
            continue;
        }
        let score = if reference.is_some() {
            if let Some(region_ref) = region.reference_mean {
                (region.mean_obs - region_ref).abs() + 0.1 * region.texture
                    - 0.01 * region.area as f64
            } else {
                region.mean_obs.abs() + 0.1 * region.texture - 0.01 * region.area as f64
            }
        } else {
            region.mean_obs.abs() + 0.2 * region.texture - 0.01 * region.area as f64
        };
        if score < best_score {
            best_score = score;
            best_index = Some(index);
        }
    }
    best_index
}

fn active_seedable_region_ids(
    regions: &[RegionGraphRegion],
    seed: usize,
    reference: Option<ArrayView2<'_, f64>>,
) -> HashSet<usize> {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut components: Vec<HashSet<usize>> = Vec::new();
    for region in regions {
        if !region.seedable || seen.contains(&region.region_id) {
            continue;
        }
        let mut component: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::from([region.region_id]);
        seen.insert(region.region_id);
        while let Some(region_id) = queue.pop_front() {
            component.insert(region_id);
            for neighbor_id in &regions[region_id].neighbors {
                if seen.contains(neighbor_id) || !regions[*neighbor_id].seedable {
                    continue;
                }
                seen.insert(*neighbor_id);
                queue.push_back(*neighbor_id);
            }
        }
        components.push(component);
    }

    let mut active: HashSet<usize> = HashSet::new();
    let mut seed_component_index: Option<usize> = None;
    for (index, component) in components.iter().enumerate() {
        if component.contains(&seed) {
            active.extend(component.iter().copied());
            seed_component_index = Some(index);
            break;
        }
    }

    if reference.is_some() {
        for (index, component) in components.iter().enumerate() {
            if Some(index) == seed_component_index {
                continue;
            }
            let has_reference = component
                .iter()
                .any(|region_id| regions[*region_id].reference_mean.is_some());
            if has_reference {
                active.extend(component.iter().copied());
            }
        }
    }

    active
}

fn best_fold_for_region(
    region: &RegionGraphRegion,
    fold_map: &HashMap<usize, i16>,
    regions: &[RegionGraphRegion],
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    reference_weight: f64,
    max_abs_fold: i16,
) -> (i16, f64, f64) {
    let mut neighbor_means = Vec::new();
    let mut neighbor_weights = Vec::new();
    for neighbor_id in &region.neighbors {
        if let Some(fold) = fold_map.get(neighbor_id) {
            let neighbor = &regions[*neighbor_id];
            let corrected_mean = neighbor.mean_obs + 2.0 * nyquist * f64::from(*fold);
            neighbor_means.push(corrected_mean);
            neighbor_weights.push(*region.boundary_weight.get(neighbor_id).unwrap_or(&1) as f64);
        }
    }

    let region_ref = if reference.is_some() {
        region.reference_mean
    } else {
        None
    };
    let mut center = if !neighbor_means.is_empty() {
        let weight_sum: f64 = neighbor_weights.iter().sum::<f64>().max(1e-6);
        let target = neighbor_means
            .iter()
            .zip(neighbor_weights.iter())
            .map(|(mean, weight)| mean * weight)
            .sum::<f64>()
            / weight_sum;
        ((target - region.mean_obs) / (2.0 * nyquist)).round_ties_even() as i16
    } else if let Some(region_ref) = region_ref {
        ((region_ref - region.mean_obs) / (2.0 * nyquist)).round_ties_even() as i16
    } else {
        0
    };
    center = center.clamp(-max_abs_fold, max_abs_fold);

    let start = i16::max(-max_abs_fold, center - 3);
    let end = i16::min(max_abs_fold, center + 3);
    let mut best_fold = center;
    let mut best_score = f64::INFINITY;
    let mut best_mean = region.mean_obs + 2.0 * nyquist * f64::from(center);
    for fold in start..=end {
        let candidate_mean = region.mean_obs + 2.0 * nyquist * f64::from(fold);
        let mut score = 0.35 * f64::from(fold.abs());
        score -= 0.05 * region.density;
        for (neighbor_mean, weight) in neighbor_means.iter().zip(neighbor_weights.iter()) {
            score += weight * (candidate_mean - neighbor_mean).abs();
        }
        if let Some(region_ref) = region_ref {
            score += reference_weight * (candidate_mean - region_ref).abs();
        }
        if score < best_score {
            best_score = score;
            best_fold = fold;
            best_mean = candidate_mean;
        }
    }

    (best_fold, best_mean, best_score)
}

fn region_reference_value(
    region: &RegionGraphRegion,
    reference: Option<ArrayView2<'_, f64>>,
) -> Option<f64> {
    reference.and_then(|reference| {
        region_median(
            reference,
            region.row0,
            region.row1,
            region.col0,
            region.col1,
        )
    })
}

fn propagate_region_folds(
    regions: &[RegionGraphRegion],
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    reference_weight: f64,
    max_abs_fold: i16,
    max_iterations: usize,
) -> (
    HashMap<usize, i16>,
    HashMap<usize, f64>,
    HashMap<usize, f64>,
    usize,
) {
    let mut fold_map: HashMap<usize, i16> = HashMap::new();
    let mut mean_map: HashMap<usize, f64> = HashMap::new();
    let mut score_map: HashMap<usize, f64> = HashMap::new();
    if regions.is_empty() {
        return (fold_map, mean_map, score_map, 0);
    }

    let Some(seed) = pick_seed_region(regions, reference) else {
        return (fold_map, mean_map, score_map, 0);
    };
    let active_region_ids = active_seedable_region_ids(regions, seed, reference);
    let pruned_seedable_regions = regions
        .iter()
        .filter(|region| region.seedable && !active_region_ids.contains(&region.region_id))
        .count();
    let (seed_fold, seed_mean, seed_score) = best_fold_for_region(
        &regions[seed],
        &fold_map,
        regions,
        nyquist,
        reference,
        reference_weight,
        max_abs_fold,
    );
    fold_map.insert(seed, seed_fold);
    mean_map.insert(seed, seed_mean);
    score_map.insert(seed, seed_score);

    let mut queue: VecDeque<usize> = VecDeque::from([seed]);
    while let Some(region_id) = queue.pop_front() {
        for neighbor_id in &regions[region_id].neighbors {
            if fold_map.contains_key(neighbor_id) {
                continue;
            }
            if !active_region_ids.contains(neighbor_id) {
                continue;
            }
            if !regions[*neighbor_id].seedable {
                continue;
            }
            if !regions[*neighbor_id]
                .neighbors
                .iter()
                .any(|parent| fold_map.contains_key(parent))
            {
                continue;
            }
            let (fold, mean, score) = best_fold_for_region(
                &regions[*neighbor_id],
                &fold_map,
                regions,
                nyquist,
                reference,
                reference_weight,
                max_abs_fold,
            );
            fold_map.insert(*neighbor_id, fold);
            mean_map.insert(*neighbor_id, mean);
            score_map.insert(*neighbor_id, score);
            queue.push_back(*neighbor_id);
        }
    }

    for region in regions {
        if fold_map.contains_key(&region.region_id) {
            continue;
        }
        if !active_region_ids.contains(&region.region_id) {
            continue;
        }
        if !region.seedable {
            continue;
        }
        if region_reference_value(region, reference).is_none() {
            continue;
        }
        let (fold, mean, score) = best_fold_for_region(
            region,
            &fold_map,
            regions,
            nyquist,
            reference,
            reference_weight,
            max_abs_fold,
        );
        fold_map.insert(region.region_id, fold);
        mean_map.insert(region.region_id, mean);
        score_map.insert(region.region_id, score);
        queue.push_back(region.region_id);
    }

    while let Some(region_id) = queue.pop_front() {
        for neighbor_id in &regions[region_id].neighbors {
            if fold_map.contains_key(neighbor_id) {
                continue;
            }
            if !active_region_ids.contains(neighbor_id) {
                continue;
            }
            if !regions[*neighbor_id].seedable {
                continue;
            }
            if !regions[*neighbor_id]
                .neighbors
                .iter()
                .any(|parent| fold_map.contains_key(parent))
            {
                continue;
            }
            let (fold, mean, score) = best_fold_for_region(
                &regions[*neighbor_id],
                &fold_map,
                regions,
                nyquist,
                reference,
                reference_weight,
                max_abs_fold,
            );
            fold_map.insert(*neighbor_id, fold);
            mean_map.insert(*neighbor_id, mean);
            score_map.insert(*neighbor_id, score);
            queue.push_back(*neighbor_id);
        }
    }

    for _ in 0..max_iterations {
        let mut changes = 0usize;
        for region in regions {
            if !active_region_ids.contains(&region.region_id) {
                continue;
            }
            if !fold_map.contains_key(&region.region_id) {
                continue;
            }
            let current_fold = *fold_map.get(&region.region_id).unwrap();
            let current_mean = *mean_map.get(&region.region_id).unwrap();
            let current_score = *score_map.get(&region.region_id).unwrap();
            let (best_fold, best_mean, best_score) = best_fold_for_region(
                region,
                &fold_map,
                regions,
                nyquist,
                reference,
                reference_weight,
                max_abs_fold,
            );
            if best_fold != current_fold && best_score + 1e-8 < current_score {
                fold_map.insert(region.region_id, best_fold);
                mean_map.insert(region.region_id, best_mean);
                score_map.insert(region.region_id, best_score);
                changes += 1;
            } else {
                mean_map.insert(region.region_id, current_mean);
                score_map.insert(region.region_id, current_score);
            }
        }
        if changes == 0 {
            break;
        }
    }

    (fold_map, mean_map, score_map, pruned_seedable_regions)
}

fn expand_region_solution(
    observed: ArrayView2<'_, f64>,
    regions: &[RegionGraphRegion],
    _fold_map: &HashMap<usize, i16>,
    mean_map: &HashMap<usize, f64>,
    score_map: &HashMap<usize, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    wrap_azimuth: bool,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let (rows, cols) = observed.dim();
    let mut coarse = Array2::from_elem((rows, cols), f64::NAN);
    let mut confidence = Array2::from_elem((rows, cols), 0.0f64);
    let mut covered = Array2::from_elem((rows, cols), false);

    for region in regions {
        let Some(&corrected_mean) = mean_map.get(&region.region_id) else {
            continue;
        };
        let Some(&score) = score_map.get(&region.region_id) else {
            continue;
        };
        for row in region.row0..region.row1 {
            for col in region.col0..region.col1 {
                coarse[(row, col)] = corrected_mean;
                covered[(row, col)] = true;
            }
        }
        let density_penalty = region.density.max(0.25);
        let scale = f64::max(0.30 * nyquist, 1.0 + 0.12 * region.texture) / density_penalty;
        let conf = (-0.5 * (score / scale.max(1e-6)).powi(2))
            .exp()
            .clamp(0.05, 0.99);
        for row in region.row0..region.row1 {
            for col in region.col0..region.col1 {
                confidence[(row, col)] = conf;
            }
        }
    }

    let mut reference_field = Array2::from_elem((rows, cols), f64::NAN);
    let finite_observed = observed.iter().filter(|value| value.is_finite()).count().max(1);
    let finite_covered = observed
        .iter()
        .zip(covered.iter())
        .filter(|(value, is_covered)| value.is_finite() && **is_covered)
        .count();
    let covered_fraction = finite_covered as f64 / finite_observed as f64;
    if covered_fraction >= 0.85 {
        let coarse_neighbors = neighbor_stack(coarse.view(), true, wrap_azimuth)?;
        let mut smooth = Array2::from_elem((rows, cols), f64::NAN);
        for row in 0..rows {
            for col in 0..cols {
                let mut values = Vec::with_capacity(coarse_neighbors.shape()[0]);
                for layer in 0..coarse_neighbors.shape()[0] {
                    let value = coarse_neighbors[(layer, row, col)];
                    if value.is_finite() {
                        values.push(value);
                    }
                }
                if let Some(median) = quantile_linear(values, 0.5) {
                    smooth[(row, col)] = median;
                }
            }
        }
        for row in 0..rows {
            for col in 0..cols {
                let mut values = Vec::with_capacity(3);
                let coarse_value = coarse[(row, col)];
                if coarse_value.is_finite() {
                    values.push(coarse_value);
                }
                let smooth_value = smooth[(row, col)];
                if smooth_value.is_finite() {
                    values.push(smooth_value);
                }
                if let Some(reference) = reference {
                    let reference_value = reference[(row, col)];
                    if reference_value.is_finite() {
                        values.push(reference_value);
                    }
                }
                if let Some(median) = quantile_linear(values, 0.5) {
                    reference_field[(row, col)] = median;
                }
            }
        }
    } else {
        for row in 0..rows {
            for col in 0..cols {
                let mut values = Vec::with_capacity(2);
                let coarse_value = coarse[(row, col)];
                if coarse_value.is_finite() {
                    values.push(coarse_value);
                }
                if let Some(reference) = reference {
                    let reference_value = reference[(row, col)];
                    if reference_value.is_finite() {
                        values.push(reference_value);
                    }
                }
                if let Some(median) = quantile_linear(values, 0.5) {
                    reference_field[(row, col)] = median;
                }
            }
        }
    }
    let mut corrected = Array2::from_elem((rows, cols), f64::NAN);
    for row in 0..rows {
        for col in 0..cols {
            if !covered[(row, col)] {
                continue;
            }
            let observed_value = observed[(row, col)];
            let reference_value = reference_field[(row, col)];
            if !observed_value.is_finite() || !reference_value.is_finite() {
                confidence[(row, col)] = 0.0;
                continue;
            }
            let fold = ((reference_value - observed_value) / (2.0 * nyquist))
                .round_ties_even()
                .clamp(-32.0, 32.0);
            corrected[(row, col)] = observed_value + 2.0 * nyquist * fold;
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            if !observed[(row, col)].is_finite() || !covered[(row, col)] {
                corrected[(row, col)] = f64::NAN;
                confidence[(row, col)] = 0.0;
            }
        }
    }

    if reference_field.iter().any(|value| value.is_finite()) {
        for row in 0..rows {
            for col in 0..cols {
                let mismatch = (corrected[(row, col)] - reference_field[(row, col)]).abs();
                confidence[(row, col)] = confidence[(row, col)]
                    .max(gaussian_confidence_scalar(mismatch, 0.40 * nyquist));
            }
        }
    }

    Ok((corrected, confidence))
}

fn dealias_sweep_region_graph_raw(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    block_shape: Option<(usize, usize)>,
    reference_weight: f64,
    max_iterations: usize,
    max_abs_fold: i16,
    wrap_azimuth: bool,
    min_region_area: usize,
    min_valid_fraction: f64,
) -> Result<RegionGraphResult> {
    validate_nyquist(nyquist)?;
    if max_abs_fold < 0 {
        return Err(DealiasError::InvalidMaxAbsFold(max_abs_fold));
    }
    if !(0.0..=1.0).contains(&min_valid_fraction) {
        return Err(DealiasError::ShapeMismatch(
            "min_valid_fraction must be between 0 and 1",
        ));
    }
    if let Some(reference) = reference {
        if reference.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }

    let (rows, cols) = observed.dim();
    let has_valid = observed.iter().any(|value| value.is_finite());
    if !has_valid {
        let velocity = Array2::from_elem((rows, cols), f64::NAN).into_dyn();
        let folds = Array2::from_elem((rows, cols), 0i16).into_dyn();
        let confidence = Array2::from_elem((rows, cols), 0.0f64).into_dyn();
        let reference_out = if let Some(reference) = reference {
            reference.to_owned().into_dyn()
        } else {
            Array2::from_elem((rows, cols), f64::NAN).into_dyn()
        };
        return Ok(RegionGraphResult {
            velocity,
            folds,
            confidence,
            reference: reference_out,
            method: "region_graph_sweep",
            region_count: 0,
            seedable_region_count: 0,
            assigned_regions: 0,
            unresolved_regions: 0,
            seed_region: None,
            block_shape: choose_block_shape((rows, cols), block_shape),
            merge_iterations: 0,
            wrap_azimuth,
            average_fold: 0.0,
            regions_with_reference: 0,
            block_grid_shape: (0, 0),
            min_region_area,
            min_valid_fraction,
            skipped_sparse_blocks: 0,
            pruned_disconnected_seedable_regions: 0,
            safety_fallback_applied: false,
            safety_fallback_reason: None,
            candidate_cost: f64::INFINITY,
            fallback_cost: None,
            disagreement_fraction: 0.0,
            largest_disagreement_component: 0,
        });
    }

    let (regions, block_ids, skipped_sparse_blocks) = build_region_graph(
        observed,
        reference,
        block_shape,
        wrap_azimuth,
        min_region_area,
        min_valid_fraction,
    )?;
    if regions.is_empty() {
        let velocity = Array2::from_elem((rows, cols), f64::NAN).into_dyn();
        let folds = Array2::from_elem((rows, cols), 0i16).into_dyn();
        let confidence = Array2::from_elem((rows, cols), 0.0f64).into_dyn();
        let reference_out = if let Some(reference) = reference {
            reference.to_owned().into_dyn()
        } else {
            Array2::from_elem((rows, cols), f64::NAN).into_dyn()
        };
        return Ok(RegionGraphResult {
            velocity,
            folds,
            confidence,
            reference: reference_out,
            method: "region_graph_sweep",
            region_count: 0,
            seedable_region_count: 0,
            assigned_regions: 0,
            unresolved_regions: 0,
            seed_region: None,
            block_shape: choose_block_shape((rows, cols), block_shape),
            merge_iterations: 0,
            wrap_azimuth,
            average_fold: 0.0,
            regions_with_reference: 0,
            block_grid_shape: block_ids.dim(),
            min_region_area,
            min_valid_fraction,
            skipped_sparse_blocks,
            pruned_disconnected_seedable_regions: 0,
            safety_fallback_applied: false,
            safety_fallback_reason: None,
            candidate_cost: f64::INFINITY,
            fallback_cost: None,
            disagreement_fraction: 0.0,
            largest_disagreement_component: 0,
        });
    }
    let (fold_map, mean_map, score_map, pruned_disconnected_seedable_regions) = propagate_region_folds(
        &regions,
        nyquist,
        reference,
        reference_weight,
        max_abs_fold,
        max_iterations,
    );
    let (corrected, confidence) = expand_region_solution(
        observed,
        &regions,
        &fold_map,
        &mean_map,
        &score_map,
        nyquist,
        reference,
        wrap_azimuth,
    )?;
    let folds = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?;
    let mut average_fold_sum = 0.0f64;
    let mut average_fold_count = 0usize;
    for (corrected_value, fold_value) in corrected.iter().zip(folds.iter()) {
        if corrected_value.is_finite() {
            average_fold_sum += f64::from(*fold_value);
            average_fold_count += 1;
        }
    }
    let average_fold = if average_fold_count > 0 {
        average_fold_sum / average_fold_count as f64
    } else {
        0.0
    };
    let regions_with_reference = if reference.is_some() {
        regions
            .iter()
            .filter(|region| region.reference_mean.is_some())
            .count()
    } else {
        0
    };
    let seed_region = pick_seed_region(&regions, reference);
    let seedable_region_count = regions.iter().filter(|region| region.seedable).count();
    let reference_out = if let Some(reference) = reference {
        reference.to_owned().into_dyn()
    } else {
        Array2::from_elem((rows, cols), f64::NAN).into_dyn()
    };

    Ok(RegionGraphResult {
        velocity: corrected.into_dyn(),
        folds,
        confidence: confidence.into_dyn(),
        reference: reference_out,
        method: "region_graph_sweep",
        region_count: regions.len(),
        seedable_region_count,
        assigned_regions: fold_map.len(),
        unresolved_regions: regions.len().saturating_sub(fold_map.len()),
        seed_region,
        block_shape: choose_block_shape((rows, cols), block_shape),
        merge_iterations: max_iterations,
        wrap_azimuth,
        average_fold,
        regions_with_reference,
        block_grid_shape: block_ids.dim(),
        min_region_area,
        min_valid_fraction,
        skipped_sparse_blocks,
        pruned_disconnected_seedable_regions,
        safety_fallback_applied: false,
        safety_fallback_reason: None,
        candidate_cost: f64::NAN,
        fallback_cost: None,
        disagreement_fraction: 0.0,
        largest_disagreement_component: 0,
    })
}

pub fn dealias_sweep_region_graph(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    block_shape: Option<(usize, usize)>,
    reference_weight: f64,
    max_iterations: usize,
    max_abs_fold: i16,
    wrap_azimuth: bool,
    min_region_area: usize,
    min_valid_fraction: f64,
) -> Result<RegionGraphResult> {
    let mut result = dealias_sweep_region_graph_raw(
        observed,
        nyquist,
        reference,
        block_shape,
        reference_weight,
        max_iterations,
        max_abs_fold,
        wrap_azimuth,
        min_region_area,
        min_valid_fraction,
    )?;

    if result.region_count == 0 || result.velocity.iter().all(|value| !value.is_finite()) {
        return Ok(result);
    }

    let candidate_velocity = result
        .velocity
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let candidate_folds = result
        .folds
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();

    let suspicious = result.skipped_sparse_blocks > 0
        || result.unresolved_regions > 0
        || result.seedable_region_count < result.region_count;

    if result.region_count > 0 && suspicious {
        let (fallback, candidate_cost, fallback_cost, disagreement_fraction, largest_component) =
            prefer_zw06_region_graph_fallback(
                observed,
                nyquist,
                reference,
                &candidate_velocity,
                &candidate_folds,
                result.skipped_sparse_blocks,
                min_region_area,
                wrap_azimuth,
            )?;
        result.candidate_cost = candidate_cost;
        result.fallback_cost = fallback_cost;
        result.disagreement_fraction = disagreement_fraction;
        result.largest_disagreement_component = largest_component;

        if let Some((fallback_result, reason)) = fallback {
            result.velocity = fallback_result.velocity;
            result.folds = fallback_result.folds;
            result.confidence = fallback_result.confidence;
            result.reference = fallback_result.reference;
            result.method = "zw06_safety_fallback";
            result.safety_fallback_applied = true;
            result.safety_fallback_reason = Some(reason);
        }
    }

    Ok(result)
}

#[derive(Clone, Debug)]
struct RecursiveNode {
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
    depth: usize,
    mean_obs: f64,
    texture: f64,
    area: usize,
    children: Vec<RecursiveNode>,
    leaf_ids: Vec<usize>,
}

fn axis_median_profile(block: ArrayView2<'_, f64>, axis: usize) -> Vec<f64> {
    let (rows, cols) = block.dim();
    if axis == 0 {
        let mut profile = Vec::with_capacity(rows);
        for row in 0..rows {
            profile.push(
                nanmedian_view2(block.slice(ndarray::s![row..row + 1, ..])).unwrap_or(f64::NAN),
            );
        }
        profile
    } else {
        let mut profile = Vec::with_capacity(cols);
        for col in 0..cols {
            profile.push(
                nanmedian_view2(block.slice(ndarray::s![.., col..col + 1])).unwrap_or(f64::NAN),
            );
        }
        profile
    }
}

fn node_stats(
    observed: ArrayView2<'_, f64>,
    row0: usize,
    row1: usize,
    col0: usize,
    col1: usize,
    wrap_azimuth: bool,
) -> (f64, f64, usize) {
    let block = observed.slice(ndarray::s![row0..row1, col0..col1]);
    let texture_map = texture_3x3(block, wrap_azimuth).ok();
    let mean_obs = nanmedian_view2(block).unwrap_or(f64::NAN);
    let texture = texture_map
        .as_ref()
        .and_then(|map| nanmedian_view2(map.view()))
        .unwrap_or(0.0);
    let texture = if texture.is_finite() { texture } else { 0.0 };
    let area = block.iter().filter(|value| value.is_finite()).count();
    (mean_obs, texture, area)
}

fn profile_energy(block: ArrayView2<'_, f64>, axis: usize, nyquist: f64) -> (f64, isize) {
    let profile = axis_median_profile(block, axis);
    if profile.len() < 2 {
        return (0.0, -1);
    }
    let mut diffs = Vec::new();
    for pair in profile.windows(2) {
        if pair[0].is_finite() && pair[1].is_finite() {
            diffs.push(wrap_scalar(pair[1] - pair[0], nyquist).abs());
        }
    }
    if diffs.is_empty() {
        return (0.0, -1);
    }
    let mut best_idx = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for (idx, value) in diffs.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_idx = idx + 1;
        }
    }
    let energy = nanmedian_view2(
        Array2::from_shape_vec((1, diffs.len()), diffs.clone())
            .expect("shape preserved")
            .view(),
    )
    .unwrap_or(0.0);
    (energy, best_idx as isize)
}

fn split_node(
    node: &mut RecursiveNode,
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    depth_limit: usize,
    min_leaf_cells: usize,
    split_texture_fraction: f64,
    wrap_azimuth: bool,
) {
    let block = observed.slice(ndarray::s![node.row0..node.row1, node.col0..node.col1]);
    let (mean_obs, texture, area) = node_stats(
        observed,
        node.row0,
        node.row1,
        node.col0,
        node.col1,
        wrap_azimuth,
    );
    node.mean_obs = mean_obs;
    node.texture = texture;
    node.area = area;
    let rows = node.row1 - node.row0;
    let cols = node.col1 - node.col0;
    if node.depth >= depth_limit || node.area <= min_leaf_cells || rows <= 2 || cols <= 2 {
        return;
    }
    if node.texture <= split_texture_fraction * nyquist
        && usize::min(rows, cols) <= usize::max(6, usize::min(rows, cols) / 2)
    {
        return;
    }

    let (row_energy, row_cut) = profile_energy(block, 0, nyquist);
    let (col_energy, col_cut) = profile_energy(block, 1, nyquist);
    let mut axis = if row_energy >= col_energy { 0 } else { 1 };
    let mut cut = if axis == 0 { row_cut } else { col_cut };
    if cut <= 0
        || cut
            >= if axis == 0 {
                rows as isize
            } else {
                cols as isize
            }
    {
        axis = 1 - axis;
        cut = if axis == 0 { row_cut } else { col_cut };
        if cut <= 0
            || cut
                >= if axis == 0 {
                    rows as isize
                } else {
                    cols as isize
                }
        {
            cut = if axis == 0 {
                (rows / 2) as isize
            } else {
                (cols / 2) as isize
            };
        }
    }

    if axis == 0 {
        let mut split_row = node.row0 + cut as usize;
        let top = observed.slice(ndarray::s![node.row0..split_row, node.col0..node.col1]);
        let bottom = observed.slice(ndarray::s![split_row..node.row1, node.col0..node.col1]);
        if top.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
            || bottom.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
        {
            split_row = node.row0 + rows / 2;
            let top = observed.slice(ndarray::s![node.row0..split_row, node.col0..node.col1]);
            let bottom = observed.slice(ndarray::s![split_row..node.row1, node.col0..node.col1]);
            if top.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
                || bottom.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
            {
                return;
            }
        }
        let mut top_node = RecursiveNode {
            row0: node.row0,
            row1: split_row,
            col0: node.col0,
            col1: node.col1,
            depth: node.depth + 1,
            mean_obs: 0.0,
            texture: 0.0,
            area: 0,
            children: Vec::new(),
            leaf_ids: Vec::new(),
        };
        let mut bottom_node = RecursiveNode {
            row0: split_row,
            row1: node.row1,
            col0: node.col0,
            col1: node.col1,
            depth: node.depth + 1,
            mean_obs: 0.0,
            texture: 0.0,
            area: 0,
            children: Vec::new(),
            leaf_ids: Vec::new(),
        };
        split_node(
            &mut top_node,
            observed,
            nyquist,
            depth_limit,
            min_leaf_cells,
            split_texture_fraction,
            wrap_azimuth,
        );
        split_node(
            &mut bottom_node,
            observed,
            nyquist,
            depth_limit,
            min_leaf_cells,
            split_texture_fraction,
            wrap_azimuth,
        );
        node.children = vec![top_node, bottom_node];
    } else {
        let mut split_col = node.col0 + cut as usize;
        let left = observed.slice(ndarray::s![node.row0..node.row1, node.col0..split_col]);
        let right = observed.slice(ndarray::s![node.row0..node.row1, split_col..node.col1]);
        if left.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
            || right.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
        {
            split_col = node.col0 + cols / 2;
            let left = observed.slice(ndarray::s![node.row0..node.row1, node.col0..split_col]);
            let right = observed.slice(ndarray::s![node.row0..node.row1, split_col..node.col1]);
            if left.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
                || right.iter().filter(|value| value.is_finite()).count() < min_leaf_cells
            {
                return;
            }
        }
        let mut left_node = RecursiveNode {
            row0: node.row0,
            row1: node.row1,
            col0: node.col0,
            col1: split_col,
            depth: node.depth + 1,
            mean_obs: 0.0,
            texture: 0.0,
            area: 0,
            children: Vec::new(),
            leaf_ids: Vec::new(),
        };
        let mut right_node = RecursiveNode {
            row0: node.row0,
            row1: node.row1,
            col0: split_col,
            col1: node.col1,
            depth: node.depth + 1,
            mean_obs: 0.0,
            texture: 0.0,
            area: 0,
            children: Vec::new(),
            leaf_ids: Vec::new(),
        };
        split_node(
            &mut left_node,
            observed,
            nyquist,
            depth_limit,
            min_leaf_cells,
            split_texture_fraction,
            wrap_azimuth,
        );
        split_node(
            &mut right_node,
            observed,
            nyquist,
            depth_limit,
            min_leaf_cells,
            split_texture_fraction,
            wrap_azimuth,
        );
        node.children = vec![left_node, right_node];
    }
}

fn collect_leaves(node: &mut RecursiveNode, leaves: &mut Vec<RecursiveNode>) {
    if node.children.is_empty() {
        node.leaf_ids = vec![leaves.len()];
        leaves.push(node.clone());
        return;
    }
    node.leaf_ids.clear();
    for child in &mut node.children {
        collect_leaves(child, leaves);
        node.leaf_ids.extend(child.leaf_ids.iter().copied());
    }
}

fn touches(a: &RegionGraphRegion, b: &RegionGraphRegion, rows: usize, wrap_azimuth: bool) -> bool {
    let row_overlap = usize::min(a.row1, b.row1).saturating_sub(usize::max(a.row0, b.row0));
    let col_overlap = usize::min(a.col1, b.col1).saturating_sub(usize::max(a.col0, b.col0));
    if a.col1 == b.col0 || b.col1 == a.col0 {
        return row_overlap > 0;
    }
    if a.row1 == b.row0 || b.row1 == a.row0 {
        return col_overlap > 0;
    }
    if wrap_azimuth
        && rows > 1
        && ((a.row0 == 0 && b.row1 == rows) || (b.row0 == 0 && a.row1 == rows))
    {
        return col_overlap > 0;
    }
    false
}

fn build_leaf_regions(
    leaves: &[RecursiveNode],
    rows: usize,
    wrap_azimuth: bool,
) -> Vec<RegionGraphRegion> {
    let mut regions = Vec::with_capacity(leaves.len());
    for (rid, leaf) in leaves.iter().enumerate() {
        regions.push(RegionGraphRegion {
            region_id: rid,
            row0: leaf.row0,
            row1: leaf.row1,
            col0: leaf.col0,
            col1: leaf.col1,
            mean_obs: leaf.mean_obs,
            texture: leaf.texture,
            area: leaf.area,
            density: 1.0,
            seedable: true,
            reference_mean: None,
            neighbors: Vec::new(),
            boundary_weight: HashMap::new(),
        });
    }
    for i in 0..regions.len() {
        for j in (i + 1)..regions.len() {
            if !touches(&regions[i], &regions[j], rows, wrap_azimuth) {
                continue;
            }
            let weight = if regions[i].col1 == regions[j].col0 || regions[j].col1 == regions[i].col0
            {
                usize::max(
                    1,
                    usize::min(regions[i].row1, regions[j].row1)
                        - usize::max(regions[i].row0, regions[j].row0),
                )
            } else {
                usize::max(
                    1,
                    usize::min(regions[i].col1, regions[j].col1)
                        - usize::max(regions[i].col0, regions[j].col0),
                )
            };
            add_region_edge(&mut regions, i, j, weight);
        }
    }
    regions
}

fn leaf_mean(
    leaves: &[RecursiveNode],
    fold_map: &HashMap<usize, i16>,
    nyquist: f64,
    ids: &[usize],
) -> f64 {
    let mut values = Vec::new();
    let mut weights = Vec::new();
    for lid in ids {
        if let Some(fold) = fold_map.get(lid) {
            let leaf = &leaves[*lid];
            values.push(leaf.mean_obs + 2.0 * nyquist * f64::from(*fold));
            weights.push(usize::max(1, leaf.area) as f64);
        }
    }
    if values.is_empty() {
        return f64::NAN;
    }
    let weight_sum: f64 = weights.iter().sum::<f64>().max(1e-6);
    values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
        / weight_sum
}

fn node_anchor(
    node: &RecursiveNode,
    leaves: &[RecursiveNode],
    fold_map: &HashMap<usize, i16>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
) -> f64 {
    let child_mean = leaf_mean(leaves, fold_map, nyquist, &node.leaf_ids);
    if let Some(reference) = reference {
        let ref_mean = region_median(reference, node.row0, node.row1, node.col0, node.col1)
            .unwrap_or(f64::NAN);
        let values = [child_mean, ref_mean];
        let mut finite = Vec::new();
        for value in values {
            if value.is_finite() {
                finite.push(value);
            }
        }
        if finite.is_empty() {
            f64::NAN
        } else {
            quantile_linear(finite, 0.5).unwrap_or(f64::NAN)
        }
    } else {
        child_mean
    }
}

fn shift_subtree(node: &RecursiveNode, fold_map: &mut HashMap<usize, i16>, delta: i16) {
    for lid in &node.leaf_ids {
        let entry = fold_map.entry(*lid).or_insert(0);
        *entry += delta;
    }
}

fn best_fold_from_targets(
    mean_obs: f64,
    target_means: &[f64],
    target_weights: &[f64],
    ref_mean: Option<f64>,
    nyquist: f64,
    reference_weight: f64,
    max_abs_fold: i16,
) -> (i16, f64, f64) {
    let mut center = if !target_means.is_empty() {
        let weight_sum: f64 = target_weights.iter().sum::<f64>().max(1e-6);
        let target = target_means
            .iter()
            .zip(target_weights.iter())
            .map(|(mean, weight)| mean * weight)
            .sum::<f64>()
            / weight_sum;
        ((target - mean_obs) / (2.0 * nyquist)).round_ties_even() as i16
    } else if let Some(ref_mean) = ref_mean {
        if ref_mean.is_finite() {
            ((ref_mean - mean_obs) / (2.0 * nyquist)).round_ties_even() as i16
        } else {
            0
        }
    } else {
        0
    };
    center = center.clamp(-max_abs_fold, max_abs_fold);

    let start = i16::max(-max_abs_fold, center - 3);
    let end = i16::min(max_abs_fold, center + 3);
    let mut best_fold = center;
    let mut best_mean = mean_obs + 2.0 * nyquist * f64::from(center);
    let mut best_score = f64::INFINITY;
    for fold in start..=end {
        let candidate_mean = mean_obs + 2.0 * nyquist * f64::from(fold);
        let mut score = 0.35 * f64::from(fold.abs());
        for (target, weight) in target_means.iter().zip(target_weights.iter()) {
            score += weight * (candidate_mean - target).abs();
        }
        if let Some(ref_mean) = ref_mean {
            if ref_mean.is_finite() {
                score += reference_weight * (candidate_mean - ref_mean).abs();
            }
        }
        if score < best_score {
            best_score = score;
            best_fold = fold;
            best_mean = candidate_mean;
        }
    }
    (best_fold, best_mean, best_score)
}

fn directional_refine(
    leaves: &[RecursiveNode],
    fold_map: &mut HashMap<usize, i16>,
    mean_map: &mut HashMap<usize, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    axis: usize,
    reference_weight: f64,
    max_abs_fold: i16,
    reverse: bool,
) -> usize {
    let mut order: Vec<usize> = (0..leaves.len()).collect();
    order.sort_by(|a, b| {
        let left = &leaves[*a];
        let right = &leaves[*b];
        let primary_left = if axis == 1 {
            (left.col0 + left.col1) as f64 / 2.0
        } else {
            (left.row0 + left.row1) as f64 / 2.0
        };
        let primary_right = if axis == 1 {
            (right.col0 + right.col1) as f64 / 2.0
        } else {
            (right.row0 + right.row1) as f64 / 2.0
        };
        let secondary_left = if axis == 1 {
            (left.row0 + left.row1) as f64 / 2.0
        } else {
            (left.col0 + left.col1) as f64 / 2.0
        };
        let secondary_right = if axis == 1 {
            (right.row0 + right.row1) as f64 / 2.0
        } else {
            (right.col0 + right.col1) as f64 / 2.0
        };
        let ord = primary_left
            .partial_cmp(&primary_right)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                secondary_left
                    .partial_cmp(&secondary_right)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if reverse {
            ord.reverse()
        } else {
            ord
        }
    });

    let mut changes = 0usize;
    for lid in &order {
        let leaf = &leaves[*lid];
        let mut target_means = Vec::new();
        let mut target_weights = Vec::new();
        for other_id in &order {
            if *other_id == *lid {
                continue;
            }
            if !fold_map.contains_key(other_id) {
                continue;
            }
            let other = &leaves[*other_id];
            let overlap = if axis == 1 {
                if reverse {
                    if other.col0 < leaf.col1 {
                        continue;
                    }
                } else if other.col1 > leaf.col0 {
                    continue;
                }
                usize::min(leaf.row1, other.row1).saturating_sub(usize::max(leaf.row0, other.row0))
            } else {
                if reverse {
                    if other.row0 < leaf.row1 {
                        continue;
                    }
                } else if other.row1 > leaf.row0 {
                    continue;
                }
                usize::min(leaf.col1, other.col1).saturating_sub(usize::max(leaf.col0, other.col0))
            };
            if overlap == 0 {
                continue;
            }
            let distance = if axis == 1 {
                ((leaf.col0 + leaf.col1) as f64 / 2.0 - (other.col0 + other.col1) as f64 / 2.0)
                    .abs()
            } else {
                ((leaf.row0 + leaf.row1) as f64 / 2.0 - (other.row0 + other.row1) as f64 / 2.0)
                    .abs()
            };
            let corrected = *mean_map.get(other_id).unwrap();
            target_means.push(corrected);
            target_weights.push((usize::max(1, overlap) as f64) / (1.0 + distance));
        }

        let ref_mean = reference.and_then(|reference| {
            region_median(reference, leaf.row0, leaf.row1, leaf.col0, leaf.col1)
        });
        let (best_fold, best_mean, best_score) = best_fold_from_targets(
            leaf.mean_obs,
            &target_means,
            &target_weights,
            ref_mean,
            nyquist,
            reference_weight,
            max_abs_fold,
        );
        let current_fold = *fold_map.get(lid).unwrap_or(&0);
        let current_mean = *mean_map
            .get(lid)
            .unwrap_or(&(leaf.mean_obs + 2.0 * nyquist * f64::from(current_fold)));
        let current_score = best_fold_from_targets(
            leaf.mean_obs,
            &[current_mean],
            &[1.0],
            ref_mean,
            nyquist,
            reference_weight,
            max_abs_fold,
        )
        .2;
        if best_fold != current_fold && best_score + 1e-8 < current_score {
            fold_map.insert(*lid, best_fold);
            mean_map.insert(*lid, best_mean);
            changes += 1;
        }
    }
    changes
}

fn refine_tree(
    node: &RecursiveNode,
    leaves: &[RecursiveNode],
    fold_map: &mut HashMap<usize, i16>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
) -> usize {
    if node.children.is_empty() {
        return 0;
    }
    let mut changes = 0usize;
    let anchor = node_anchor(node, leaves, fold_map, nyquist, reference);
    if anchor.is_finite() {
        for child in &node.children {
            let child_mean = leaf_mean(leaves, fold_map, nyquist, &child.leaf_ids);
            if !child_mean.is_finite() {
                continue;
            }
            let delta = ((anchor - child_mean) / (2.0 * nyquist)).round_ties_even() as i16;
            if delta != 0
                && (child_mean + 2.0 * nyquist * f64::from(delta) - anchor).abs()
                    < (child_mean - anchor).abs()
            {
                shift_subtree(child, fold_map, delta);
                changes += 1;
            }
        }
    }
    for child in &node.children {
        changes += refine_tree(child, leaves, fold_map, nyquist, reference);
    }
    changes
}

fn make_recursive_reference_field(
    observed: ArrayView2<'_, f64>,
    leaves: &[RecursiveNode],
    fold_map: &HashMap<usize, i16>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    wrap_azimuth: bool,
) -> Array2<f64> {
    let mut coarse = Array2::from_elem(observed.dim(), f64::NAN);
    for (lid, leaf) in leaves.iter().enumerate() {
        let corrected_mean =
            leaf.mean_obs + 2.0 * nyquist * f64::from(*fold_map.get(&lid).unwrap_or(&0));
        for row in leaf.row0..leaf.row1 {
            for col in leaf.col0..leaf.col1 {
                coarse[(row, col)] = corrected_mean;
            }
        }
    }
    let smooth = neighbor_stack(coarse.view(), true, wrap_azimuth)
        .ok()
        .and_then(|stack| {
            let (layers, rows, cols) = stack.dim();
            let mut out = Array2::from_elem((rows, cols), f64::NAN);
            for row in 0..rows {
                for col in 0..cols {
                    let mut values = Vec::with_capacity(layers);
                    for layer in 0..layers {
                        let value = stack[(layer, row, col)];
                        if value.is_finite() {
                            values.push(value);
                        }
                    }
                    out[(row, col)] = quantile_linear(values, 0.5).unwrap_or(f64::NAN);
                }
            }
            Some(out)
        })
        .unwrap_or_else(|| Array2::from_elem(observed.dim(), f64::NAN));
    let mut reference_field = Array2::from_elem(observed.dim(), f64::NAN);
    for row in 0..observed.nrows() {
        for col in 0..observed.ncols() {
            let mut values = Vec::new();
            let coarse_value = coarse[(row, col)];
            if coarse_value.is_finite() {
                values.push(coarse_value);
            }
            let smooth_value = smooth[(row, col)];
            if smooth_value.is_finite() {
                values.push(smooth_value);
            }
            if let Some(reference) = reference {
                let ref_value = reference[(row, col)];
                if ref_value.is_finite() {
                    values.push(ref_value);
                }
            }
            if let Some(median) = quantile_linear(values, 0.5) {
                reference_field[(row, col)] = median;
            }
        }
    }
    reference_field
}

fn solution_cost(
    candidate: ArrayView2<'_, f64>,
    observed: ArrayView2<'_, f64>,
    reference: Option<ArrayView2<'_, f64>>,
    nyquist: f64,
) -> f64 {
    let mut valid = 0usize;
    let mut continuity_sum = 0.0f64;
    if candidate.ncols() >= 2 {
        for row in 0..candidate.nrows() {
            for col in 1..candidate.ncols() {
                let a = candidate[(row, col)];
                let b = candidate[(row, col - 1)];
                if a.is_finite()
                    && b.is_finite()
                    && observed[(row, col)].is_finite()
                    && observed[(row, col - 1)].is_finite()
                {
                    continuity_sum += wrap_scalar(a - b, nyquist).abs();
                    valid += 1;
                }
            }
        }
    }
    if candidate.nrows() >= 2 {
        for row in 1..candidate.nrows() {
            for col in 0..candidate.ncols() {
                let a = candidate[(row, col)];
                let b = candidate[(row - 1, col)];
                if a.is_finite()
                    && b.is_finite()
                    && observed[(row, col)].is_finite()
                    && observed[(row - 1, col)].is_finite()
                {
                    continuity_sum += wrap_scalar(a - b, nyquist).abs();
                    valid += 1;
                }
            }
        }
    }
    if valid == 0 {
        return f64::INFINITY;
    }
    let continuity = continuity_sum / valid as f64;
    let ref_cost = if let Some(reference) = reference {
        let mut mismatch_sum = 0.0f64;
        let mut mismatch_count = 0usize;
        for row in 0..candidate.nrows() {
            for col in 0..candidate.ncols() {
                let a = candidate[(row, col)];
                let r = reference[(row, col)];
                if a.is_finite() && r.is_finite() {
                    mismatch_sum += (a - r).abs();
                    mismatch_count += 1;
                }
            }
        }
        if mismatch_count > 0 {
            mismatch_sum / mismatch_count as f64
        } else {
            0.0
        }
    } else {
        0.0
    };
    continuity + 0.35 * ref_cost
}

fn fold_disagreement_metrics(
    candidate_folds: ArrayView2<'_, i16>,
    fallback_folds: ArrayView2<'_, i16>,
    candidate_velocity: ArrayView2<'_, f64>,
    fallback_velocity: ArrayView2<'_, f64>,
) -> (usize, usize, usize) {
    let (rows, cols) = candidate_folds.dim();
    let mut valid = Array2::from_elem((rows, cols), false);
    let mut mismatch_count = 0usize;
    let mut valid_count = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let comparable = candidate_velocity[(row, col)].is_finite()
                && fallback_velocity[(row, col)].is_finite();
            if comparable {
                valid[(row, col)] = candidate_folds[(row, col)] != fallback_folds[(row, col)];
                valid_count += 1;
                if valid[(row, col)] {
                    mismatch_count += 1;
                }
            }
        }
    }

    let mut visited = Array2::from_elem((rows, cols), false);
    let mut largest_component = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            if !valid[(row, col)] || visited[(row, col)] {
                continue;
            }
            let mut queue = VecDeque::from([(row, col)]);
            visited[(row, col)] = true;
            let mut size = 0usize;
            while let Some((r, c)) = queue.pop_front() {
                size += 1;
                if r > 0 && valid[(r - 1, c)] && !visited[(r - 1, c)] {
                    visited[(r - 1, c)] = true;
                    queue.push_back((r - 1, c));
                }
                if r + 1 < rows && valid[(r + 1, c)] && !visited[(r + 1, c)] {
                    visited[(r + 1, c)] = true;
                    queue.push_back((r + 1, c));
                }
                if c > 0 && valid[(r, c - 1)] && !visited[(r, c - 1)] {
                    visited[(r, c - 1)] = true;
                    queue.push_back((r, c - 1));
                }
                if c + 1 < cols && valid[(r, c + 1)] && !visited[(r, c + 1)] {
                    visited[(r, c + 1)] = true;
                    queue.push_back((r, c + 1));
                }
            }
            largest_component = largest_component.max(size);
        }
    }

    (mismatch_count, valid_count, largest_component)
}

fn should_prefer_region_graph_fallback(
    candidate_cost: f64,
    fallback_cost: f64,
    mismatch_count: usize,
    valid_count: usize,
    largest_component: usize,
    skipped_sparse_blocks: usize,
    min_region_area: usize,
    nyquist: f64,
    has_reference: bool,
) -> Option<String> {
    if !fallback_cost.is_finite() {
        return None;
    }
    if !candidate_cost.is_finite() {
        return Some("zw06_finite_solution_fallback".to_string());
    }
    if valid_count == 0 || mismatch_count == 0 {
        return None;
    }

    let mismatch_fraction = mismatch_count as f64 / valid_count as f64;
    let component_fraction = largest_component as f64 / valid_count as f64;
    let absolute_margin = REGION_GRAPH_SAFETY_COST_ABS_MARGIN_FRACTION * nyquist;
    let clearly_worse = candidate_cost > fallback_cost + absolute_margin
        && candidate_cost > fallback_cost * (1.0 + REGION_GRAPH_SAFETY_COST_REL_MARGIN);
    let reference_worse = has_reference && candidate_cost > fallback_cost + 1e-8;
    if !(clearly_worse || reference_worse) {
        return None;
    }

    let min_component = usize::max(min_region_area.max(16), valid_count / 200);
    let has_large_disagreement = mismatch_fraction >= REGION_GRAPH_SAFETY_DISAGREEMENT_FRACTION
        && component_fraction >= REGION_GRAPH_SAFETY_COMPONENT_FRACTION
        && largest_component >= min_component;
    let reference_guided_disagreement = has_reference
        && mismatch_fraction >= 0.01
        && largest_component >= usize::max(4, min_region_area);
    if !(has_large_disagreement || reference_guided_disagreement) {
        return None;
    }

    if has_reference {
        return Some("zw06_reference_consistency_fallback".to_string());
    }
    if skipped_sparse_blocks > 0 {
        return Some("zw06_sparse_gap_fallback".to_string());
    }
    Some("zw06_lower_cost_on_large_disagreement".to_string())
}

fn prefer_zw06_region_graph_fallback(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    candidate_velocity: &Array2<f64>,
    candidate_folds: &Array2<i16>,
    skipped_sparse_blocks: usize,
    min_region_area: usize,
    wrap_azimuth: bool,
) -> Result<(Option<(Zw06Result, String)>, f64, Option<f64>, f64, usize)> {
    let candidate_cost = solution_cost(candidate_velocity.view(), observed, reference, nyquist);
    let fallback = dealias_sweep_zw06(
        observed,
        nyquist,
        reference,
        REGION_GRAPH_SAFETY_WEAK_THRESHOLD_FRACTION,
        wrap_azimuth,
        REGION_GRAPH_SAFETY_MAX_ITERATIONS_PER_PASS,
        true,
        true,
    )?;
    let fallback_velocity = fallback
        .velocity
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
    let fallback_folds = fallback
        .folds
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
    let fallback_cost = solution_cost(fallback_velocity, observed, reference, nyquist);
    let (mismatch_count, valid_count, largest_component) = fold_disagreement_metrics(
        candidate_folds.view(),
        fallback_folds,
        candidate_velocity.view(),
        fallback_velocity,
    );
    let disagreement_fraction = if valid_count > 0 {
        mismatch_count as f64 / valid_count as f64
    } else {
        0.0
    };

    let fallback_result = should_prefer_region_graph_fallback(
        candidate_cost,
        fallback_cost,
        mismatch_count,
        valid_count,
        largest_component,
        skipped_sparse_blocks,
        min_region_area,
        nyquist,
        reference.is_some(),
    )
    .map(|reason| (fallback, reason));

    Ok((
        fallback_result,
        candidate_cost,
        Some(fallback_cost),
        disagreement_fraction,
        largest_component,
    ))
}

struct VariationalBootstrap {
    initial: Array2<f64>,
    reference: Array2<f64>,
    method: &'static str,
    region_count: usize,
    unresolved_regions: usize,
    skipped_sparse_blocks: usize,
    assigned_gates: usize,
    iterations_used: usize,
    safety_fallback_applied: bool,
    safety_fallback_reason: Option<String>,
}

fn use_region_graph_variational_bootstrap(
    block_shape: Option<(usize, usize)>,
    bootstrap_reference_weight: f64,
    bootstrap_iterations: usize,
    bootstrap_max_abs_fold: i16,
    bootstrap_min_region_area: usize,
    bootstrap_min_valid_fraction: f64,
) -> bool {
    block_shape.is_some()
        || (bootstrap_reference_weight - DEFAULT_VARIATIONAL_BOOTSTRAP_REFERENCE_WEIGHT).abs()
            > 1e-12
        || bootstrap_iterations != DEFAULT_VARIATIONAL_BOOTSTRAP_ITERATIONS
        || bootstrap_max_abs_fold != DEFAULT_VARIATIONAL_BOOTSTRAP_MAX_ABS_FOLD
        || bootstrap_min_region_area != DEFAULT_REGION_GRAPH_MIN_REGION_AREA
        || (bootstrap_min_valid_fraction - DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION).abs() > 1e-12
}

fn build_variational_bootstrap(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    block_shape: Option<(usize, usize)>,
    bootstrap_reference_weight: f64,
    bootstrap_iterations: usize,
    bootstrap_max_abs_fold: i16,
    bootstrap_min_region_area: usize,
    bootstrap_min_valid_fraction: f64,
    wrap_azimuth: bool,
) -> Result<VariationalBootstrap> {
    let (rows, cols) = observed.dim();
    if use_region_graph_variational_bootstrap(
        block_shape,
        bootstrap_reference_weight,
        bootstrap_iterations,
        bootstrap_max_abs_fold,
        bootstrap_min_region_area,
        bootstrap_min_valid_fraction,
    ) {
        let region = dealias_sweep_region_graph(
            observed,
            nyquist,
            reference,
            block_shape,
            bootstrap_reference_weight,
            bootstrap_iterations,
            bootstrap_max_abs_fold,
            wrap_azimuth,
            bootstrap_min_region_area,
            bootstrap_min_valid_fraction,
        )?;
        let initial = region
            .velocity
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d")
            .to_owned();
        let reference_field = if let Some(reference) = reference {
            reference.to_owned()
        } else {
            region
                .reference
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("2d")
                .to_owned()
        };
        return Ok(VariationalBootstrap {
            initial,
            reference: reference_field,
            method: region.method,
            region_count: region.region_count,
            unresolved_regions: region.unresolved_regions,
            skipped_sparse_blocks: region.skipped_sparse_blocks,
            assigned_gates: region.assigned_regions,
            iterations_used: region.merge_iterations,
            safety_fallback_applied: region.safety_fallback_applied,
            safety_fallback_reason: region.safety_fallback_reason.clone(),
        });
    }

    let zw = dealias_sweep_zw06(
        observed,
        nyquist,
        reference,
        REGION_GRAPH_SAFETY_WEAK_THRESHOLD_FRACTION,
        wrap_azimuth,
        REGION_GRAPH_SAFETY_MAX_ITERATIONS_PER_PASS,
        true,
        true,
    )?;
    let initial = zw
        .velocity
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let reference_field = if let Some(reference) = reference {
        reference.to_owned()
    } else {
        Array2::from_elem((rows, cols), f64::NAN)
    };
    Ok(VariationalBootstrap {
        initial,
        reference: reference_field,
        method: "2d_multipass",
        region_count: 0,
        unresolved_regions: 0,
        skipped_sparse_blocks: 0,
        assigned_gates: zw.assigned_gates,
        iterations_used: zw.iterations_used,
        safety_fallback_applied: false,
        safety_fallback_reason: None,
    })
}

pub fn dealias_sweep_recursive(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    reference: Option<ArrayView2<'_, f64>>,
    max_depth: usize,
    min_leaf_cells: usize,
    split_texture_fraction: f64,
    reference_weight: f64,
    max_abs_fold: i16,
    wrap_azimuth: bool,
) -> Result<RecursiveResult> {
    validate_nyquist(nyquist)?;
    if max_abs_fold < 0 {
        return Err(DealiasError::InvalidMaxAbsFold(max_abs_fold));
    }
    if let Some(reference) = reference {
        if reference.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }

    let (rows, cols) = observed.dim();
    let bootstrap = dealias_sweep_region_graph(
        observed,
        nyquist,
        reference,
        None,
        0.75,
        6,
        max_abs_fold,
        wrap_azimuth,
        DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
        DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
    )?;
    let RegionGraphResult {
        velocity: bootstrap_velocity_dyn,
        folds: bootstrap_folds_dyn,
        confidence: bootstrap_confidence_dyn,
        method: bootstrap_method,
        region_count: bootstrap_region_count,
        ..
    } = bootstrap;
    let bootstrap_velocity = bootstrap_velocity_dyn
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d")
        .to_owned();
    let bootstrap_reference = if let Some(reference) = reference {
        let mut combined = Array2::from_elem((rows, cols), f64::NAN);
        for row in 0..rows {
            for col in 0..cols {
                let mut values = Vec::new();
                let ref_value = reference[(row, col)];
                if ref_value.is_finite() {
                    values.push(ref_value);
                }
                let boot_value = bootstrap_velocity[(row, col)];
                if boot_value.is_finite() {
                    values.push(boot_value);
                }
                if let Some(median) = quantile_linear(values, 0.5) {
                    combined[(row, col)] = median;
                }
            }
        }
        Some(combined)
    } else {
        Some(bootstrap_velocity.clone())
    };

    let (root_mean, root_texture, root_area) = node_stats(observed, 0, rows, 0, cols, wrap_azimuth);
    let mut root = RecursiveNode {
        row0: 0,
        row1: rows,
        col0: 0,
        col1: cols,
        depth: 0,
        mean_obs: root_mean,
        texture: root_texture,
        area: root_area,
        children: Vec::new(),
        leaf_ids: Vec::new(),
    };
    split_node(
        &mut root,
        observed,
        nyquist,
        max_depth,
        min_leaf_cells,
        split_texture_fraction,
        wrap_azimuth,
    );
    let mut leaves = Vec::new();
    collect_leaves(&mut root, &mut leaves);
    let mut regions = build_leaf_regions(&leaves, rows, wrap_azimuth);
    let (mut fold_map, mut mean_map, mut score_map, _) = propagate_region_folds(
        &regions,
        nyquist,
        bootstrap_reference.as_ref().map(|field| field.view()),
        reference_weight,
        max_abs_fold,
        max_depth + 1,
    );

    for _ in 0..2 {
        let mut changes = 0usize;
        changes += directional_refine(
            &leaves,
            &mut fold_map,
            &mut mean_map,
            nyquist,
            bootstrap_reference.as_ref().map(|field| field.view()),
            1,
            reference_weight,
            max_abs_fold,
            false,
        );
        changes += directional_refine(
            &leaves,
            &mut fold_map,
            &mut mean_map,
            nyquist,
            bootstrap_reference.as_ref().map(|field| field.view()),
            1,
            reference_weight,
            max_abs_fold,
            true,
        );
        if changes == 0 {
            break;
        }
    }

    for _ in 0..2 {
        if refine_tree(
            &root,
            &leaves,
            &mut fold_map,
            nyquist,
            bootstrap_reference.as_ref().map(|field| field.view()),
        ) == 0
        {
            break;
        }
    }

    for (lid, leaf) in leaves.iter().enumerate() {
        regions[lid].mean_obs = leaf.mean_obs;
        regions[lid].texture = leaf.texture;
        regions[lid].area = leaf.area;
        mean_map.insert(
            lid,
            leaf.mean_obs + 2.0 * nyquist * f64::from(*fold_map.get(&lid).unwrap_or(&0)),
        );
        score_map.entry(lid).or_insert(0.0);
    }

    let reference_field = make_recursive_reference_field(
        observed,
        &leaves,
        &fold_map,
        nyquist,
        bootstrap_reference.as_ref().map(|field| field.view()),
        wrap_azimuth,
    );
    let corrected = unfold_to_reference(
        observed.into_dyn(),
        reference_field.view().into_dyn(),
        nyquist,
        32,
    )?
    .into_dimensionality::<ndarray::Ix2>()
    .expect("2d");
    let mut corrected = corrected;
    for row in 0..rows {
        for col in 0..cols {
            if !observed[(row, col)].is_finite() {
                corrected[(row, col)] = f64::NAN;
            }
        }
    }
    let mut confidence = Array2::from_elem((rows, cols), 0.0f64);
    for row in 0..rows {
        for col in 0..cols {
            let target = if let Some(reference) = bootstrap_reference.as_ref() {
                let ref_value = reference[(row, col)];
                if ref_value.is_finite() {
                    ref_value
                } else {
                    reference_field[(row, col)]
                }
            } else {
                reference_field[(row, col)]
            };
            confidence[(row, col)] =
                gaussian_confidence_scalar((corrected[(row, col)] - target).abs(), 0.38 * nyquist);
        }
    }
    let folds = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?;
    let recursive_cost = solution_cost(
        corrected.view(),
        observed,
        bootstrap_reference.as_ref().map(|field| field.view()),
        nyquist,
    );
    let bootstrap_cost = solution_cost(
        bootstrap_velocity.view(),
        observed,
        bootstrap_reference.as_ref().map(|field| field.view()),
        nyquist,
    );
    let (final_velocity, final_folds, final_confidence, method) =
        if recursive_cost > bootstrap_cost + 1e-8 {
            let fallback_method = if bootstrap_method == "zw06_safety_fallback" {
                "recursive_region_refinement_fallback_zw06"
            } else {
                "recursive_region_refinement_fallback_region_graph"
            };
            (
                bootstrap_velocity,
                bootstrap_folds_dyn
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("2d")
                    .to_owned(),
                bootstrap_confidence_dyn
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("2d")
                    .to_owned(),
                fallback_method,
            )
        } else {
            (
                corrected,
                folds.into_dimensionality::<ndarray::Ix2>().expect("2d"),
                confidence,
                "recursive_region_refinement",
            )
        };

    let final_reference =
        bootstrap_reference.unwrap_or_else(|| Array2::from_elem((rows, cols), f64::NAN));
    Ok(RecursiveResult {
        velocity: final_velocity.into_dyn(),
        folds: final_folds.into_dyn(),
        confidence: final_confidence.into_dyn(),
        reference: final_reference.into_dyn(),
        leaf_count: leaves.len(),
        max_depth,
        split_texture_fraction,
        reference_weight,
        wrap_azimuth,
        root_texture,
        bootstrap_method,
        bootstrap_region_count,
        method,
    })
}

fn resolve_volume_nyquist(nyquist: &[f64], n_sweeps: usize) -> Result<Vec<f64>> {
    if nyquist.len() != n_sweeps {
        return Err(DealiasError::ShapeMismatch(
            "nyquist must be scalar-expanded to length n_sweeps",
        ));
    }
    if let Some(value) = nyquist
        .iter()
        .copied()
        .find(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(DealiasError::InvalidNyquist(value));
    }
    Ok(nyquist.to_vec())
}

fn count_finite_2d(field: ArrayView2<'_, f64>) -> usize {
    field.iter().filter(|value| value.is_finite()).count()
}

fn choose_seed_sweep(
    observed: ArrayView3<'_, f64>,
    reference_volume: Option<ArrayView3<'_, f64>>,
) -> usize {
    let n_sweeps = observed.len_of(Axis(0));
    let center = n_sweeps / 2;
    let valid_counts: Vec<usize> = (0..n_sweeps)
        .map(|sweep| count_finite_2d(observed.index_axis(Axis(0), sweep)))
        .collect();

    if let Some(reference_volume) = reference_volume.as_ref() {
        let reference_counts: Vec<usize> = (0..n_sweeps)
            .map(|sweep| count_finite_2d(reference_volume.index_axis(Axis(0), sweep)))
            .collect();
        if reference_counts.iter().any(|value| *value > 0) {
            let mut ranked: Vec<usize> = (0..n_sweeps).collect();
            ranked.sort_by_key(|&sweep| (std::cmp::Reverse(reference_counts[sweep]), sweep));
            return ranked[0];
        }
    }

    if valid_counts.iter().any(|value| *value > 0) {
        let mut ranked: Vec<usize> = (0..n_sweeps).collect();
        ranked.sort_by_key(|&sweep| {
            (
                std::cmp::Reverse(valid_counts[sweep]),
                sweep.abs_diff(center),
                sweep,
            )
        });
        return ranked[0];
    }

    center
}

fn sweep_order_from_seed(seed: usize, n_sweeps: usize) -> Vec<usize> {
    let mut order = vec![seed];
    let mut step = 1usize;
    while seed >= step || seed + step < n_sweeps {
        if seed + step < n_sweeps {
            order.push(seed + step);
        }
        if seed >= step {
            order.push(seed - step);
        }
        step += 1;
    }
    order
}

fn sweep_reference_field(
    corrected: ArrayView3<'_, f64>,
    sweep_index: usize,
    reference_volume: Option<ArrayView3<'_, f64>>,
) -> Option<Array2<f64>> {
    let (n_sweeps, rows, cols) = corrected.dim();
    let mut reference = Array2::from_elem((rows, cols), f64::NAN);
    let mut any = false;

    for row in 0..rows {
        for col in 0..cols {
            let mut values = Vec::with_capacity(4);
            if let Some(reference_volume) = reference_volume.as_ref() {
                let value = reference_volume[(sweep_index, row, col)];
                if value.is_finite() {
                    values.push(value);
                }
            }
            if sweep_index > 0 {
                let value = corrected[(sweep_index - 1, row, col)];
                if value.is_finite() {
                    values.push(value);
                }
            }
            if sweep_index + 1 < n_sweeps {
                let value = corrected[(sweep_index + 1, row, col)];
                if value.is_finite() {
                    values.push(value);
                }
            }
            let value = corrected[(sweep_index, row, col)];
            if value.is_finite() {
                values.push(value);
            }
            if let Some(median) = quantile_linear(values, 0.5) {
                reference[(row, col)] = median;
                any = true;
            }
        }
    }

    any.then_some(reference)
}

fn fold_counts_volume(
    corrected: ArrayView3<'_, f64>,
    observed: ArrayView3<'_, f64>,
    nyquist: &[f64],
) -> Result<ArrayD<i16>> {
    if corrected.dim() != observed.dim() {
        return Err(DealiasError::ShapeMismatch(
            "corrected and observed must match",
        ));
    }
    let (n_sweeps, rows, cols) = corrected.dim();
    let nyq = resolve_volume_nyquist(nyquist, n_sweeps)?;
    let mut folds = Array3::from_elem((n_sweeps, rows, cols), 0i16);
    for sweep in 0..n_sweeps {
        let nyquist = nyq[sweep];
        for row in 0..rows {
            for col in 0..cols {
                let unfolded = corrected[(sweep, row, col)];
                let observed = observed[(sweep, row, col)];
                let ratio = (unfolded - observed) / (2.0 * nyquist);
                let rounded = ratio.round_ties_even();
                folds[(sweep, row, col)] = if rounded.is_finite() {
                    rounded.clamp(i16::MIN as f64, i16::MAX as f64) as i16
                } else {
                    0
                };
            }
        }
    }
    Ok(folds.into_dyn())
}

pub fn dealias_volume_3d(
    observed_volume: ArrayView3<'_, f64>,
    nyquist: &[f64],
    reference_volume: Option<ArrayView3<'_, f64>>,
    wrap_azimuth: bool,
    max_iterations: usize,
) -> Result<Volume3DResult> {
    let (n_sweeps, rows, cols) = observed_volume.dim();
    let nyq = resolve_volume_nyquist(nyquist, n_sweeps)?;
    if let Some(reference_volume) = reference_volume.as_ref() {
        if reference_volume.dim() != observed_volume.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference_volume must match observed_volume shape",
            ));
        }
    }

    let mut corrected = Array3::from_elem((n_sweeps, rows, cols), f64::NAN);
    let mut confidence = Array3::from_elem((n_sweeps, rows, cols), 0.0f64);
    let mut reference = Array3::from_elem((n_sweeps, rows, cols), f64::NAN);
    let mut per_sweep_valid_gates = vec![0usize; n_sweeps];
    let mut per_sweep_seeded_gates = vec![0usize; n_sweeps];
    let mut per_sweep_assigned_gates = vec![0usize; n_sweeps];
    let mut per_sweep_iterations_used = vec![0usize; n_sweeps];

    let seed = choose_seed_sweep(observed_volume, reference_volume);
    let seed_obs = observed_volume.index_axis(Axis(0), seed);
    per_sweep_valid_gates[seed] = count_finite_2d(seed_obs);
    let seed_ref = reference_volume
        .as_ref()
        .map(|field| field.index_axis(Axis(0), seed));
    let seed_result = match seed_ref {
        Some(ref_field) => dealias_sweep_zw06(
            seed_obs,
            nyq[seed],
            Some(ref_field),
            0.35,
            wrap_azimuth,
            12,
            true,
            true,
        )?,
        None => dealias_sweep_zw06(
            seed_obs,
            nyq[seed],
            None,
            0.35,
            wrap_azimuth,
            12,
            true,
            true,
        )?,
    };
    let seed_velocity = seed_result
        .velocity
        .into_dimensionality::<ndarray::Ix2>()
        .expect("seed result is 2D");
    let seed_confidence = seed_result
        .confidence
        .into_dimensionality::<ndarray::Ix2>()
        .expect("seed result is 2D");
    let seed_reference = seed_result
        .reference
        .into_dimensionality::<ndarray::Ix2>()
        .expect("seed result is 2D");
    corrected
        .index_axis_mut(Axis(0), seed)
        .assign(&seed_velocity);
    confidence
        .index_axis_mut(Axis(0), seed)
        .assign(&seed_confidence);
    reference
        .index_axis_mut(Axis(0), seed)
        .assign(&seed_reference);
    per_sweep_seeded_gates[seed] = seed_result.seeded_gates;
    per_sweep_assigned_gates[seed] = seed_result.assigned_gates;
    per_sweep_iterations_used[seed] = seed_result.iterations_used;

    let order = sweep_order_from_seed(seed, n_sweeps);
    let reverse_order: Vec<usize> = order.iter().copied().rev().collect();
    let mut iterations_used = 1usize;

    for iteration in 0..max_iterations {
        let current_order = if iteration % 2 == 0 {
            &order
        } else {
            &reverse_order
        };
        let mut changed = 0usize;
        for &sweep in current_order {
            let obs_slice = observed_volume.index_axis(Axis(0), sweep);
            per_sweep_valid_gates[sweep] = count_finite_2d(obs_slice);

            let current_reference =
                sweep_reference_field(corrected.view(), sweep, reference_volume).or_else(|| {
                    let current = corrected.index_axis(Axis(0), sweep);
                    current
                        .iter()
                        .any(|value| value.is_finite())
                        .then_some(current.to_owned())
                });
            if current_reference.is_none() && !obs_slice.iter().any(|value| value.is_finite()) {
                continue;
            }

            let result = match current_reference.as_ref() {
                Some(ref_field) => dealias_sweep_zw06(
                    obs_slice,
                    nyq[sweep],
                    Some(ref_field.view()),
                    0.35,
                    wrap_azimuth,
                    12,
                    true,
                    true,
                )?,
                None => dealias_sweep_zw06(
                    obs_slice,
                    nyq[sweep],
                    None,
                    0.35,
                    wrap_azimuth,
                    12,
                    true,
                    true,
                )?,
            };

            let result_velocity = result
                .velocity
                .into_dimensionality::<ndarray::Ix2>()
                .expect("2d");
            let result_confidence = result
                .confidence
                .into_dimensionality::<ndarray::Ix2>()
                .expect("2d");

            let current_slice = corrected.index_axis(Axis(0), sweep);
            for row in 0..rows {
                for col in 0..cols {
                    let a = result_velocity[(row, col)];
                    let b = current_slice[(row, col)];
                    let differs = if a.is_nan() && b.is_nan() {
                        false
                    } else if a.is_finite() && b.is_finite() {
                        (a - b).abs() > 1e-12
                    } else {
                        a.is_finite() != b.is_finite()
                    };
                    if differs {
                        changed += 1;
                    }
                }
            }
            corrected
                .index_axis_mut(Axis(0), sweep)
                .assign(&result_velocity);
            for row in 0..rows {
                for col in 0..cols {
                    let updated = result_confidence[(row, col)];
                    if updated > confidence[(sweep, row, col)] {
                        confidence[(sweep, row, col)] = updated;
                    }
                }
            }
            per_sweep_seeded_gates[sweep] = result.seeded_gates;
            per_sweep_assigned_gates[sweep] = result.assigned_gates;
            per_sweep_iterations_used[sweep] = result.iterations_used;
        }

        iterations_used += 1;
        if changed == 0 {
            break;
        }
    }

    for sweep in 0..n_sweeps {
        let sweep_reference = sweep_reference_field(corrected.view(), sweep, reference_volume);
        if sweep_reference.is_none() {
            continue;
        }
        let sweep_reference = sweep_reference.expect("checked above");
        let obs_slice = observed_volume.index_axis(Axis(0), sweep);
        if count_finite_2d(obs_slice) > 0 {
            let unfolded = unfold_to_reference(
                obs_slice.into_dyn(),
                sweep_reference.view().into_dyn(),
                nyq[sweep],
                32,
            )?
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
            let mut filled_sweep = corrected.index_axis(Axis(0), sweep).to_owned();
            for row in 0..rows {
                for col in 0..cols {
                    if !filled_sweep[(row, col)].is_finite() {
                        filled_sweep[(row, col)] = unfolded[(row, col)];
                    }
                }
            }
            corrected
                .index_axis_mut(Axis(0), sweep)
                .assign(&filled_sweep);
            for row in 0..rows {
                for col in 0..cols {
                    let mismatch = (filled_sweep[(row, col)] - sweep_reference[(row, col)]).abs();
                    let updated = gaussian_confidence_scalar(mismatch, 0.45 * nyq[sweep]);
                    if updated > confidence[(sweep, row, col)] {
                        confidence[(sweep, row, col)] = updated;
                    }
                }
            }
        } else {
            corrected
                .index_axis_mut(Axis(0), sweep)
                .assign(&sweep_reference);
            for row in 0..rows {
                for col in 0..cols {
                    let updated = gaussian_confidence_scalar(0.0, 0.45 * nyq[sweep]);
                    if updated > confidence[(sweep, row, col)] {
                        confidence[(sweep, row, col)] = updated;
                    }
                }
            }
        }
    }

    let folds = fold_counts_volume(corrected.view(), observed_volume, &nyq)?;
    Ok(Volume3DResult {
        velocity: corrected.into_dyn(),
        folds,
        confidence: confidence.into_dyn(),
        reference: reference.into_dyn(),
        seed_sweep: seed,
        iterations_used,
        sweep_order: order,
        per_sweep_valid_gates,
        per_sweep_seeded_gates,
        per_sweep_assigned_gates,
        per_sweep_iterations_used,
    })
}

fn build_reference_from_uv_core(
    azimuth_deg: ArrayView1<'_, f64>,
    n_range: usize,
    u: f64,
    v: f64,
    elevation_deg: f64,
    offset: f64,
    sign: f64,
) -> Array2<f64> {
    let rows = azimuth_deg.len();
    let el = elevation_deg.to_radians();
    let mut out = Array2::from_elem((rows, n_range), 0.0f64);
    for row in 0..rows {
        let az = azimuth_deg[row].to_radians();
        let radial = sign * (el.cos() * (u * az.sin() + v * az.cos()) + offset);
        for col in 0..n_range {
            out[(row, col)] = radial;
        }
    }
    out
}

fn combine_reference_fields_2d(fields: &[ArrayView2<'_, f64>]) -> Option<Array2<f64>> {
    if fields.is_empty() {
        return None;
    }
    let shape = fields[0].dim();
    let mut out = Array2::from_elem(shape, f64::NAN);
    for row in 0..shape.0 {
        for col in 0..shape.1 {
            let mut values = Vec::with_capacity(fields.len());
            for field in fields {
                let value = field[(row, col)];
                if value.is_finite() {
                    values.push(value);
                }
            }
            out[(row, col)] = quantile_linear(values, 0.5).unwrap_or(f64::NAN);
        }
    }
    Some(out)
}

fn nanmedian_rows(field: ArrayView2<'_, f64>) -> Vec<f64> {
    let (rows, cols) = field.dim();
    let mut out = vec![f64::NAN; rows];
    for row in 0..rows {
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            let value = field[(row, col)];
            if value.is_finite() {
                values.push(value);
            }
        }
        out[row] = quantile_linear(values, 0.5).unwrap_or(f64::NAN);
    }
    out
}

fn solve_linear_system(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = a[pivot * n + pivot].abs();
        for row in (pivot + 1)..n {
            let value = a[row * n + pivot].abs();
            if value > pivot_abs {
                pivot_abs = value;
                pivot_row = row;
            }
        }
        if pivot_abs <= 1e-12 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot_row != pivot {
            for col in 0..n {
                a.swap(pivot * n + col, pivot_row * n + col);
            }
            b.swap(pivot, pivot_row);
        }
        let pivot_value = a[pivot * n + pivot];
        for row in (pivot + 1)..n {
            let factor = a[row * n + pivot] / pivot_value;
            if !factor.is_finite() {
                return None;
            }
            for col in pivot..n {
                a[row * n + col] -= factor * a[pivot * n + col];
            }
            b[row] -= factor * b[pivot];
        }
    }

    let mut x = vec![0.0f64; n];
    for row_rev in 0..n {
        let row = n - 1 - row_rev;
        let mut value = b[row];
        for col in (row + 1)..n {
            value -= a[row * n + col] * x[col];
        }
        let diag = a[row * n + row];
        if diag.abs() <= 1e-12 || !diag.is_finite() {
            return None;
        }
        x[row] = value / diag;
    }
    Some(x)
}

fn least_squares_three_column(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    keep: &[bool],
) -> Option<[f64; 3]> {
    let mut xtx = vec![0.0f64; 9];
    let mut xty = vec![0.0f64; 3];
    let mut used = 0usize;
    for idx in 0..y.len() {
        if !keep[idx] {
            continue;
        }
        let row = [x1[idx], x2[idx], 1.0];
        let target = y[idx];
        if !target.is_finite() || !row[0].is_finite() || !row[1].is_finite() {
            continue;
        }
        used += 1;
        for r in 0..3 {
            xty[r] += row[r] * target;
            for c in 0..3 {
                xtx[r * 3 + c] += row[r] * row[c];
            }
        }
    }
    if used < 3 {
        return None;
    }
    solve_linear_system(xtx, xty, 3).map(|beta| [beta[0], beta[1], beta[2]])
}

fn gate_stats_2d(
    field: ArrayView2<'_, f64>,
    observed: ArrayView2<'_, f64>,
) -> (usize, usize, usize, f64) {
    let (rows, cols) = observed.dim();
    let mut valid = 0usize;
    let mut assigned = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            if observed[(row, col)].is_finite() {
                valid += 1;
                if field[(row, col)].is_finite() {
                    assigned += 1;
                }
            }
        }
    }
    let unresolved = valid.saturating_sub(assigned);
    let resolved_fraction = if valid > 0 {
        assigned as f64 / valid as f64
    } else {
        0.0
    };
    (valid, assigned, unresolved, resolved_fraction)
}

fn gate_stats_3d(
    field: ArrayView3<'_, f64>,
    observed: ArrayView3<'_, f64>,
) -> (usize, usize, usize, f64) {
    let (sweeps, rows, cols) = observed.dim();
    let mut valid = 0usize;
    let mut assigned = 0usize;
    for sweep in 0..sweeps {
        for row in 0..rows {
            for col in 0..cols {
                if observed[(sweep, row, col)].is_finite() {
                    valid += 1;
                    if field[(sweep, row, col)].is_finite() {
                        assigned += 1;
                    }
                }
            }
        }
    }
    let unresolved = valid.saturating_sub(assigned);
    let resolved_fraction = if valid > 0 {
        assigned as f64 / valid as f64
    } else {
        0.0
    };
    (valid, assigned, unresolved, resolved_fraction)
}

fn fold_counts_by_sweep(
    corrected: ArrayView3<'_, f64>,
    observed: ArrayView3<'_, f64>,
    nyquist: &[f64],
) -> Result<ArrayD<i16>> {
    let (sweeps, rows, cols) = corrected.dim();
    let mut out = Array3::from_elem((sweeps, rows, cols), 0i16);
    let nyq = resolve_volume_nyquist(nyquist, sweeps)?;
    for sweep in 0..sweeps {
        let folds = fold_counts(
            corrected.index_axis(Axis(0), sweep).into_dyn(),
            observed.index_axis(Axis(0), sweep).into_dyn(),
            nyq[sweep],
        )?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
        out.index_axis_mut(Axis(0), sweep).assign(&folds);
    }
    Ok(out.into_dyn())
}

pub fn estimate_uniform_wind_vad(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    azimuth_deg: ArrayView1<'_, f64>,
    elevation_deg: f64,
    sign: f64,
    max_iterations: usize,
    trim_quantile: f64,
    search_radius: Option<f64>,
) -> Result<VadFitResult> {
    validate_nyquist(nyquist)?;
    if observed.dim().0 != azimuth_deg.len() {
        return Err(DealiasError::ShapeMismatch(
            "azimuth_deg must match observed.shape[0]",
        ));
    }

    let (rows, cols) = observed.dim();
    let el = elevation_deg.to_radians();
    let mut x1 = vec![0.0f64; rows];
    let mut x2 = vec![0.0f64; rows];
    for row in 0..rows {
        let az = azimuth_deg[row].to_radians();
        x1[row] = sign * el.cos() * az.sin();
        x2[row] = sign * el.cos() * az.cos();
    }

    let y_wrapped = nanmedian_rows(observed);
    let valid_count = y_wrapped.iter().filter(|value| value.is_finite()).count();
    if valid_count < 8 {
        return Ok(VadFitResult {
            u: 0.0,
            v: 0.0,
            offset: 0.0,
            rms: f64::INFINITY,
            iterations: 0,
            reference: Array2::from_elem((rows, cols), 0.0).into_dyn(),
        });
    }

    let radius = if let Some(radius) = search_radius {
        radius
    } else {
        let mut amp = nyquist;
        for value in observed.iter().copied() {
            if value.is_finite() {
                amp = amp.max(value.abs());
            }
        }
        (3.0 * nyquist).max(amp + 2.0 * nyquist).max(25.0)
    };

    let mut coarse = Vec::new();
    let mut coarse_value = -radius;
    while coarse_value <= radius + 1e-9 {
        coarse.push(coarse_value);
        coarse_value += 1.0;
    }

    let mut best_score = f64::INFINITY;
    let mut u = 0.0;
    let mut v = 0.0;
    for &uc in &coarse {
        for &vc in &coarse {
            let mut score_sum = 0.0;
            let mut used = 0usize;
            for row in 0..rows {
                let wrapped = y_wrapped[row];
                if !wrapped.is_finite() {
                    continue;
                }
                let pred = x1[row] * uc + x2[row] * vc;
                let residual = wrap_scalar(wrapped - pred, nyquist);
                score_sum += residual * residual;
                used += 1;
            }
            if used == 0 {
                continue;
            }
            let score = score_sum / used as f64;
            if score < best_score {
                best_score = score;
                u = uc;
                v = vc;
            }
        }
    }

    let mut best_fine_score = f64::INFINITY;
    let mut best_u = u;
    let mut best_v = v;
    let mut fine = -2.0;
    while fine <= 2.0001 {
        let uc = u + fine;
        let mut fine_v = -2.0;
        while fine_v <= 2.0001 {
            let vc = v + fine_v;
            let mut score_sum = 0.0;
            let mut used = 0usize;
            for row in 0..rows {
                let wrapped = y_wrapped[row];
                if !wrapped.is_finite() {
                    continue;
                }
                let pred = x1[row] * uc + x2[row] * vc;
                let residual = wrap_scalar(wrapped - pred, nyquist);
                score_sum += residual * residual;
                used += 1;
            }
            if used > 0 {
                let score = score_sum / used as f64;
                if score < best_fine_score {
                    best_fine_score = score;
                    best_u = uc;
                    best_v = vc;
                }
            }
            fine_v += 0.25;
        }
        fine += 0.25;
    }
    u = best_u;
    v = best_v;
    let mut offset = 0.0;
    let mut rms = f64::INFINITY;
    let mut iterations_done = 0usize;

    for iteration in 1..=max_iterations {
        iterations_done = iteration;
        let reference =
            build_reference_from_uv_core(azimuth_deg, cols, u, v, elevation_deg, offset, sign);
        let unfolded = unfold_to_reference(
            observed.into_dyn(),
            reference.view().into_dyn(),
            nyquist,
            32,
        )?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
        let y = nanmedian_rows(unfolded.view());
        let valid: Vec<bool> = y.iter().map(|value| value.is_finite()).collect();
        if valid.iter().filter(|keep| **keep).count() < 8 {
            break;
        }

        let mut keep = valid.clone();
        let beta = if let Some(beta) = least_squares_three_column(&x1, &x2, &y, &keep) {
            beta
        } else {
            break;
        };
        let mut residual_abs = Vec::new();
        for row in 0..rows {
            if !valid[row] {
                continue;
            }
            let pred = x1[row] * beta[0] + x2[row] * beta[1] + beta[2];
            residual_abs.push((y[row] - pred).abs());
        }
        let cut = quantile_linear(residual_abs, trim_quantile).unwrap_or(0.0);
        for row in 0..rows {
            if !valid[row] {
                keep[row] = false;
                continue;
            }
            let pred = x1[row] * beta[0] + x2[row] * beta[1] + beta[2];
            keep[row] = (y[row] - pred).abs() <= cut;
        }
        let final_beta = if keep.iter().filter(|value| **value).count() >= 8 {
            least_squares_three_column(&x1, &x2, &y, &keep).unwrap_or(beta)
        } else {
            beta
        };
        let mut residual_sq_sum = 0.0;
        let mut residual_count = 0usize;
        for row in 0..rows {
            if !keep[row] {
                continue;
            }
            let pred = x1[row] * final_beta[0] + x2[row] * final_beta[1] + final_beta[2];
            let residual = y[row] - pred;
            residual_sq_sum += residual * residual;
            residual_count += 1;
        }
        rms = if residual_count > 0 {
            (residual_sq_sum / residual_count as f64).sqrt()
        } else {
            0.0
        };

        let new_u = final_beta[0];
        let new_v = final_beta[1];
        let new_offset = final_beta[2];
        let delta = (new_u - u)
            .abs()
            .max((new_v - v).abs())
            .max((new_offset - offset).abs());
        u = new_u;
        v = new_v;
        offset = new_offset;
        if delta < 1e-3 {
            break;
        }
    }

    let reference =
        build_reference_from_uv_core(azimuth_deg, cols, u, v, elevation_deg, offset, sign);
    Ok(VadFitResult {
        u,
        v,
        offset,
        rms,
        iterations: iterations_done,
        reference: reference.into_dyn(),
    })
}

pub fn dealias_sweep_xu11(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    azimuth_deg: ArrayView1<'_, f64>,
    elevation_deg: f64,
    external_reference: Option<ArrayView2<'_, f64>>,
    sign: f64,
    refine_with_multipass: bool,
) -> Result<Xu11Result> {
    validate_nyquist(nyquist)?;
    if observed.dim().0 != azimuth_deg.len() {
        return Err(DealiasError::ShapeMismatch(
            "azimuth_deg must match observed.shape[0]",
        ));
    }
    if let Some(reference) = external_reference {
        if reference.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "external_reference must match observed shape",
            ));
        }
    }

    let fit = estimate_uniform_wind_vad(
        observed,
        nyquist,
        azimuth_deg,
        elevation_deg,
        sign,
        6,
        0.85,
        None,
    )?;
    let fit_reference = fit
        .reference
        .clone()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
    let combined_reference = if let Some(reference) = external_reference {
        let mut combined = Array2::from_elem(fit_reference.dim(), f64::NAN);
        for row in 0..fit_reference.dim().0 {
            for col in 0..fit_reference.dim().1 {
                let mut values = Vec::with_capacity(2);
                let fit_value = fit_reference[(row, col)];
                if fit_value.is_finite() {
                    values.push(fit_value);
                }
                let ref_value = reference[(row, col)];
                if ref_value.is_finite() {
                    values.push(ref_value);
                }
                combined[(row, col)] = quantile_linear(values, 0.5).unwrap_or(f64::NAN);
            }
        }
        combined
    } else {
        fit_reference.clone()
    };

    if !refine_with_multipass {
        let corrected = unfold_to_reference(
            observed.into_dyn(),
            combined_reference.view().into_dyn(),
            nyquist,
            32,
        )?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
        let mut confidence = Array2::from_elem(observed.dim(), 0.0f64);
        for row in 0..observed.dim().0 {
            for col in 0..observed.dim().1 {
                if corrected[(row, col)].is_finite() {
                    confidence[(row, col)] = 0.80;
                }
            }
        }
        let folds = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?;
        return Ok(Xu11Result {
            velocity: corrected.into_dyn(),
            folds,
            confidence: confidence.into_dyn(),
            reference: combined_reference.into_dyn(),
            u: fit.u,
            v: fit.v,
            offset: fit.offset,
            vad_rms: fit.rms,
            vad_iterations: fit.iterations,
            method: "vad_reference_only",
        });
    }

    let result = dealias_sweep_zw06(
        observed,
        nyquist,
        Some(combined_reference.view()),
        0.35,
        true,
        12,
        true,
        true,
    )?;
    Ok(Xu11Result {
        velocity: result.velocity,
        folds: result.folds,
        confidence: result.confidence,
        reference: combined_reference.into_dyn(),
        u: fit.u,
        v: fit.v,
        offset: fit.offset,
        vad_rms: fit.rms,
        vad_iterations: fit.iterations,
        method: "vad_seeded_multipass",
    })
}

pub fn dealias_sweep_jh01(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    previous_corrected: Option<ArrayView2<'_, f64>>,
    background_reference: Option<ArrayView2<'_, f64>>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
    refine_with_multipass: bool,
) -> Result<Jh01SweepResult> {
    validate_nyquist(nyquist)?;
    if let Some(previous) = previous_corrected {
        if previous.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "previous_corrected must match observed shape",
            ));
        }
    }
    if let Some(background) = background_reference {
        if background.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "background_reference must match observed shape",
            ));
        }
    }

    let shifted_previous = if let Some(previous) = previous_corrected {
        Some(shift2d(previous, shift_az, shift_range, wrap_azimuth)?)
    } else {
        None
    };
    let reference = if let Some(previous) = shifted_previous.as_ref() {
        if let Some(background) = background_reference {
            let mut combined = Array2::from_elem(previous.dim(), f64::NAN);
            for row in 0..previous.dim().0 {
                for col in 0..previous.dim().1 {
                    let mut values = Vec::with_capacity(2);
                    let prev_value = previous[(row, col)];
                    if prev_value.is_finite() {
                        values.push(prev_value);
                    }
                    let bg_value = background[(row, col)];
                    if bg_value.is_finite() {
                        values.push(bg_value);
                    }
                    combined[(row, col)] = quantile_linear(values, 0.5).unwrap_or(f64::NAN);
                }
            }
            combined
        } else {
            previous.clone()
        }
    } else if let Some(background) = background_reference {
        background.to_owned()
    } else {
        return Err(DealiasError::ShapeMismatch(
            "previous_corrected or background_reference is required",
        ));
    };

    if !refine_with_multipass {
        let corrected = unfold_to_reference(
            observed.into_dyn(),
            reference.view().into_dyn(),
            nyquist,
            32,
        )?
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
        let mut confidence = Array2::from_elem(observed.dim(), 0.0f64);
        for row in 0..observed.dim().0 {
            for col in 0..observed.dim().1 {
                if corrected[(row, col)].is_finite() {
                    confidence[(row, col)] = 0.85;
                }
            }
        }
        let folds = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?;
        let stats = gate_stats_2d(corrected.view(), observed);
        return Ok(Jh01SweepResult {
            velocity: corrected.into_dyn(),
            folds,
            confidence: confidence.into_dyn(),
            reference: reference.into_dyn(),
            method: "temporal_reference_only",
            valid_gates: stats.0,
            assigned_gates: stats.1,
            unresolved_gates: stats.2,
            resolved_fraction: stats.3,
        });
    }

    let result = dealias_sweep_zw06(
        observed,
        nyquist,
        Some(reference.view()),
        0.35,
        wrap_azimuth,
        12,
        true,
        true,
    )?;
    let velocity = result
        .velocity
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
    let confidence = result
        .confidence
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2d");
    let stats = gate_stats_2d(velocity.view(), observed);
    Ok(Jh01SweepResult {
        velocity: velocity.into_dyn(),
        folds: result.folds,
        confidence: confidence.into_dyn(),
        reference: reference.into_dyn(),
        method: "temporal_multipass",
        valid_gates: stats.0,
        assigned_gates: stats.1,
        unresolved_gates: stats.2,
        resolved_fraction: stats.3,
    })
}

pub fn dealias_volume_jh01(
    observed_volume: ArrayView3<'_, f64>,
    nyquist: &[f64],
    azimuth_deg: ArrayView1<'_, f64>,
    elevation_deg: ArrayView1<'_, f64>,
    previous_volume: Option<ArrayView3<'_, f64>>,
    background_u: Option<&[f64]>,
    background_v: Option<&[f64]>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> Result<Jh01VolumeResult> {
    let (n_sweeps, rows, cols) = observed_volume.dim();
    let nyq = resolve_volume_nyquist(nyquist, n_sweeps)?;
    if elevation_deg.len() != n_sweeps {
        return Err(DealiasError::ShapeMismatch(
            "elevation_deg must have one value per sweep",
        ));
    }
    if azimuth_deg.len() != rows {
        return Err(DealiasError::ShapeMismatch(
            "azimuth_deg must match observed.shape[1]",
        ));
    }
    if let Some(previous) = previous_volume {
        if previous.dim() != observed_volume.dim() {
            return Err(DealiasError::ShapeMismatch(
                "previous_volume must match observed_volume shape",
            ));
        }
    }
    if let Some(values) = background_u {
        if values.len() != n_sweeps {
            return Err(DealiasError::ShapeMismatch(
                "background_u must match n_sweeps",
            ));
        }
    }
    if let Some(values) = background_v {
        if values.len() != n_sweeps {
            return Err(DealiasError::ShapeMismatch(
                "background_v must match n_sweeps",
            ));
        }
    }

    let shifted_previous = if let Some(previous) = previous_volume {
        Some(shift3d(previous, shift_az, shift_range, wrap_azimuth)?)
    } else {
        None
    };
    let mut order: Vec<usize> = (0..n_sweeps).collect();
    order.sort_by(|a, b| elevation_deg[*b].partial_cmp(&elevation_deg[*a]).unwrap());

    let mut corrected = Array3::from_elem((n_sweeps, rows, cols), f64::NAN);
    let mut confidence = Array3::from_elem((n_sweeps, rows, cols), 0.0f64);
    let mut reference = Array3::from_elem((n_sweeps, rows, cols), f64::NAN);
    let mut per_sweep_valid_gates = vec![0usize; n_sweeps];
    let mut per_sweep_assigned_gates = vec![0usize; n_sweeps];
    let mut per_sweep_unresolved_gates = vec![0usize; n_sweeps];
    let mut per_sweep_resolved_fraction = vec![0.0f64; n_sweeps];

    for (rank, &sweep) in order.iter().enumerate() {
        let background_reference = if let (Some(us), Some(vs)) = (background_u, background_v) {
            Some(build_reference_from_uv_core(
                azimuth_deg,
                cols,
                us[sweep],
                vs[sweep],
                elevation_deg[sweep],
                0.0,
                1.0,
            ))
        } else {
            None
        };

        let mut previous_refs: Vec<Array2<f64>> = Vec::new();
        if let Some(previous) = shifted_previous.as_ref() {
            previous_refs.push(previous.index_axis(Axis(0), sweep).to_owned());
        }
        if rank > 0 {
            previous_refs.push(corrected.index_axis(Axis(0), order[rank - 1]).to_owned());
        }
        let previous_combined = if previous_refs.is_empty() {
            None
        } else {
            let views: Vec<ArrayView2<'_, f64>> =
                previous_refs.iter().map(|field| field.view()).collect();
            combine_reference_fields_2d(&views)
        };

        let result = dealias_sweep_jh01(
            observed_volume.index_axis(Axis(0), sweep),
            nyq[sweep],
            previous_combined.as_ref().map(|field| field.view()),
            background_reference.as_ref().map(|field| field.view()),
            0,
            0,
            wrap_azimuth,
            true,
        )?;
        let velocity = result
            .velocity
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
        let conf = result
            .confidence
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
        let ref_field = result
            .reference
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
        corrected.index_axis_mut(Axis(0), sweep).assign(&velocity);
        confidence.index_axis_mut(Axis(0), sweep).assign(&conf);
        reference.index_axis_mut(Axis(0), sweep).assign(&ref_field);
        per_sweep_valid_gates[sweep] = result.valid_gates;
        per_sweep_assigned_gates[sweep] = result.assigned_gates;
        per_sweep_unresolved_gates[sweep] = result.unresolved_gates;
        per_sweep_resolved_fraction[sweep] = result.resolved_fraction;
    }

    let folds = fold_counts_by_sweep(corrected.view(), observed_volume, &nyq)?;
    let stats = gate_stats_3d(corrected.view(), observed_volume);
    Ok(Jh01VolumeResult {
        velocity: corrected.into_dyn(),
        folds,
        confidence: confidence.into_dyn(),
        reference: reference.into_dyn(),
        elevation_order_desc: order,
        per_sweep_valid_gates,
        per_sweep_assigned_gates,
        per_sweep_unresolved_gates,
        per_sweep_resolved_fraction,
        valid_gates: stats.0,
        assigned_gates: stats.1,
        unresolved_gates: stats.2,
        resolved_fraction: stats.3,
    })
}

fn ml_feature_names(include_reference: bool, include_azimuth: bool) -> Vec<String> {
    let mut names = vec![
        "bias".to_string(),
        "observed".to_string(),
        "neighbor_median".to_string(),
        "texture".to_string(),
        "row_frac".to_string(),
        "col_frac".to_string(),
    ];
    if include_azimuth {
        names.push("sin_az".to_string());
        names.push("cos_az".to_string());
    }
    if include_reference {
        names.push("reference".to_string());
    } else {
        names.push("reference".to_string());
    }
    names
}

fn build_ml_feature_arrays(
    observed: ArrayView2<'_, f64>,
    reference: Option<ArrayView2<'_, f64>>,
    azimuth_deg: Option<ArrayView1<'_, f64>>,
) -> Result<(Vec<Array2<f64>>, Vec<String>)> {
    let (rows, cols) = observed.dim();
    if let Some(reference) = reference {
        if reference.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }
    if let Some(azimuth) = azimuth_deg {
        if azimuth.len() != rows {
            return Err(DealiasError::ShapeMismatch(
                "azimuth_deg must have length n_azimuth",
            ));
        }
    }

    let texture = texture_3x3(observed, true)?;
    let neighbors = neighbor_stack(observed, true, true)?;
    let mut neigh_med = Array2::from_elem((rows, cols), 0.0f64);
    for row in 0..rows {
        for col in 0..cols {
            let mut values = Vec::with_capacity(neighbors.shape()[0]);
            for layer in 0..neighbors.shape()[0] {
                let value = neighbors[(layer, row, col)];
                if value.is_finite() {
                    values.push(value);
                }
            }
            neigh_med[(row, col)] = quantile_linear(values, 0.5).unwrap_or(0.0);
        }
    }

    let mut features = Vec::new();
    features.push(Array2::from_elem((rows, cols), 1.0));
    features.push(observed.mapv(|value| if value.is_finite() { value } else { 0.0 }));
    features.push(neigh_med);
    features.push(texture.mapv(|value| if value.is_finite() { value } else { 0.0 }));

    let mut row_frac = Array2::from_elem((rows, cols), 0.0f64);
    let mut col_frac = Array2::from_elem((rows, cols), 0.0f64);
    for row in 0..rows {
        let row_value = if rows > 1 {
            row as f64 / (rows - 1) as f64
        } else {
            0.0
        };
        for col in 0..cols {
            row_frac[(row, col)] = row_value;
            col_frac[(row, col)] = if cols > 1 {
                col as f64 / (cols - 1) as f64
            } else {
                0.0
            };
        }
    }
    features.push(row_frac);
    features.push(col_frac);

    if let Some(azimuth) = azimuth_deg {
        let mut sin_az = Array2::from_elem((rows, cols), 0.0f64);
        let mut cos_az = Array2::from_elem((rows, cols), 0.0f64);
        for row in 0..rows {
            let az = azimuth[row].to_radians();
            let sin_value = az.sin();
            let cos_value = az.cos();
            for col in 0..cols {
                sin_az[(row, col)] = sin_value;
                cos_az[(row, col)] = cos_value;
            }
        }
        features.push(sin_az);
        features.push(cos_az);
    }

    let ref_feature = if let Some(reference) = reference {
        reference
            .to_owned()
            .mapv(|value| if value.is_finite() { value } else { 0.0 })
    } else {
        Array2::from_elem((rows, cols), 0.0f64)
    };
    features.push(ref_feature);

    Ok((features, ml_feature_names(true, azimuth_deg.is_some())))
}

pub fn fit_ml_reference_model(
    observed: ArrayView2<'_, f64>,
    target_velocity: ArrayView2<'_, f64>,
    nyquist: Option<f64>,
    reference: Option<ArrayView2<'_, f64>>,
    azimuth_deg: Option<ArrayView1<'_, f64>>,
    ridge: f64,
) -> Result<MlModelState> {
    if target_velocity.dim() != observed.dim() {
        return Err(DealiasError::ShapeMismatch(
            "observed and target_velocity must be 2D with the same shape",
        ));
    }
    let (features, names) = build_ml_feature_arrays(observed, reference, azimuth_deg)?;
    let p = features.len();
    let reg = ridge.max(1e-8);

    let mut xtx = vec![0.0f64; p * p];
    let mut xty = vec![0.0f64; p];
    let (rows, cols) = observed.dim();
    let mut sample_count = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let obs_value = observed[(row, col)];
            let target = target_velocity[(row, col)];
            if !obs_value.is_finite() || !target.is_finite() {
                continue;
            }
            sample_count += 1;
            let mut x = vec![0.0f64; p];
            for idx in 0..p {
                x[idx] = features[idx][(row, col)];
            }
            let target_value = if let Some(nyq) = nyquist {
                ((target - obs_value) / (2.0 * nyq)).round_ties_even()
            } else {
                target
            };
            for i in 0..p {
                xty[i] += x[i] * target_value;
                for j in 0..p {
                    xtx[i * p + j] += x[i] * x[j];
                }
            }
        }
    }
    if sample_count < p {
        return Err(DealiasError::ShapeMismatch(
            "not enough finite training gates to fit ML reference model",
        ));
    }
    for diag in 0..p {
        xtx[diag * p + diag] += reg;
    }
    let weights = solve_linear_system(xtx, xty, p).ok_or(DealiasError::ShapeMismatch(
        "unable to solve ML ridge system",
    ))?;

    let mut mse_sum = 0.0f64;
    let mut mse_count = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let obs_value = observed[(row, col)];
            let target = target_velocity[(row, col)];
            if !obs_value.is_finite() || !target.is_finite() {
                continue;
            }
            let mut pred = 0.0f64;
            for idx in 0..p {
                pred += features[idx][(row, col)] * weights[idx];
            }
            let pred_velocity = if let Some(nyq) = nyquist {
                obs_value + 2.0 * nyq * pred.round_ties_even()
            } else {
                pred
            };
            let err = target - pred_velocity;
            mse_sum += err * err;
            mse_count += 1;
        }
    }
    let rmse = if mse_count > 0 {
        (mse_sum / mse_count as f64).sqrt()
    } else {
        0.0
    };
    Ok(MlModelState {
        weights,
        feature_names: names,
        ridge: reg,
        train_rmse: rmse,
        mode: if nyquist.is_some() {
            "fold".to_string()
        } else {
            "velocity".to_string()
        },
        nyquist,
    })
}

fn predict_ml_reference(
    observed: ArrayView2<'_, f64>,
    model: &MlModelState,
    reference: Option<ArrayView2<'_, f64>>,
    azimuth_deg: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>> {
    let (features, names) = build_ml_feature_arrays(observed, reference, azimuth_deg)?;
    if names != model.feature_names {
        return Err(DealiasError::ShapeMismatch(
            "feature set does not match the supplied model",
        ));
    }
    let (rows, cols) = observed.dim();
    let mut out = Array2::from_elem((rows, cols), 0.0f64);
    for row in 0..rows {
        for col in 0..cols {
            let mut pred = 0.0f64;
            for idx in 0..model.weights.len() {
                pred += features[idx][(row, col)] * model.weights[idx];
            }
            if model.mode == "fold" {
                let nyq = model
                    .nyquist
                    .ok_or(DealiasError::ShapeMismatch("fold model is missing nyquist"))?;
                out[(row, col)] = observed[(row, col)] + 2.0 * nyq * pred.round_ties_even();
            } else {
                out[(row, col)] = pred;
            }
        }
    }
    Ok(out)
}

pub fn dealias_sweep_ml(
    observed: ArrayView2<'_, f64>,
    nyquist: f64,
    model: Option<&MlModelState>,
    training_target: Option<ArrayView2<'_, f64>>,
    reference: Option<ArrayView2<'_, f64>>,
    azimuth_deg: Option<ArrayView1<'_, f64>>,
    ridge: f64,
    refine_with_variational: bool,
) -> Result<MlDealiasResult> {
    validate_nyquist(nyquist)?;
    if let Some(reference) = reference {
        if reference.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "reference must match observed shape",
            ));
        }
    }
    if let Some(target) = training_target {
        if target.dim() != observed.dim() {
            return Err(DealiasError::ShapeMismatch(
                "training_target must match observed shape",
            ));
        }
    }

    let (trained_from, model_state) = if let Some(model) = model {
        ("supplied_model".to_string(), model.clone())
    } else if let Some(target) = training_target {
        (
            "training_target".to_string(),
            fit_ml_reference_model(
                observed,
                target,
                Some(nyquist),
                reference,
                azimuth_deg,
                ridge,
            )?,
        )
    } else if let Some(reference_field) = reference {
        let target =
            unfold_to_reference(observed.into_dyn(), reference_field.into_dyn(), nyquist, 32)?
                .into_dimensionality::<ndarray::Ix2>()
                .expect("2d");
        (
            "reference_pseudo_target".to_string(),
            fit_ml_reference_model(
                observed,
                target.view(),
                Some(nyquist),
                reference,
                azimuth_deg,
                ridge,
            )?,
        )
    } else {
        let bootstrap = build_variational_bootstrap(
            observed,
            nyquist,
            None,
            None,
            DEFAULT_VARIATIONAL_BOOTSTRAP_REFERENCE_WEIGHT,
            DEFAULT_VARIATIONAL_BOOTSTRAP_ITERATIONS,
            DEFAULT_VARIATIONAL_BOOTSTRAP_MAX_ABS_FOLD,
            DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
            DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
            true,
        )?;
        let bootstrap = dealias_sweep_variational_refine(
            observed,
            bootstrap.initial.view(),
            None,
            nyquist,
            8,
            1.0,
            0.50,
            0.20,
            8,
            true,
        )?;
        let target = bootstrap
            .velocity
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
        (
            "variational_pseudo_target".to_string(),
            fit_ml_reference_model(
                observed,
                target.view(),
                Some(nyquist),
                None,
                azimuth_deg,
                ridge,
            )?,
        )
    };

    let predicted_reference = predict_ml_reference(observed, &model_state, reference, azimuth_deg)?;
    let mut corrected = unfold_to_reference(
        observed.into_dyn(),
        predicted_reference.view().into_dyn(),
        nyquist,
        32,
    )?
    .into_dimensionality::<ndarray::Ix2>()
    .expect("2d");
    let (rows, cols) = observed.dim();
    let mut confidence = Array2::from_elem((rows, cols), 0.0f64);
    for row in 0..rows {
        for col in 0..cols {
            if observed[(row, col)].is_finite() {
                confidence[(row, col)] = gaussian_confidence_scalar(
                    (corrected[(row, col)] - predicted_reference[(row, col)]).abs(),
                    0.50 * nyquist,
                );
            } else {
                corrected[(row, col)] = f64::NAN;
            }
        }
    }

    let mut folds = fold_counts(corrected.view().into_dyn(), observed.into_dyn(), nyquist)?;
    let mut refine_method = None;
    let mut refine_iterations = None;
    if refine_with_variational {
        let bootstrap = build_variational_bootstrap(
            observed,
            nyquist,
            Some(predicted_reference.view()),
            None,
            DEFAULT_VARIATIONAL_BOOTSTRAP_REFERENCE_WEIGHT,
            DEFAULT_VARIATIONAL_BOOTSTRAP_ITERATIONS,
            DEFAULT_VARIATIONAL_BOOTSTRAP_MAX_ABS_FOLD,
            DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
            DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
            true,
        )?;
        let refined = dealias_sweep_variational_refine(
            observed,
            bootstrap.initial.view(),
            Some(predicted_reference.view()),
            nyquist,
            8,
            1.0,
            0.50,
            0.20,
            8,
            true,
        )?;
        corrected = refined
            .velocity
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
        let refined_confidence = refined
            .confidence
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2d");
        for row in 0..rows {
            for col in 0..cols {
                if refined_confidence[(row, col)] > confidence[(row, col)] {
                    confidence[(row, col)] = refined_confidence[(row, col)];
                }
            }
        }
        folds = refined.folds;
        refine_method = Some("coordinate_descent".to_string());
        refine_iterations = Some(refined.iterations_used);
    }

    Ok(MlDealiasResult {
        velocity: corrected.into_dyn(),
        folds,
        confidence: confidence.into_dyn(),
        reference: predicted_reference.into_dyn(),
        trained_from,
        train_rmse: model_state.train_rmse,
        ridge: model_state.ridge,
        feature_names: model_state.feature_names.clone(),
        refine_method,
        refine_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn wrap_matches_nyquist_semantics() {
        let velocity = array![
            -31.0,
            -10.0,
            -0.0,
            0.0,
            9.999,
            10.0,
            29.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY
        ]
        .into_dyn();
        let wrapped = wrap_to_nyquist(velocity.view(), 10.0).unwrap();
        let expected = array![
            9.0,
            -10.0,
            0.0,
            0.0,
            9.999,
            -10.0,
            9.5,
            f64::NAN,
            f64::NAN,
            f64::NAN
        ]
        .into_dyn();
        assert_eq!(wrapped.shape(), expected.shape());
        for (actual, expected) in wrapped.iter().zip(expected.iter()) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fold_counts_rounds_ties_to_even() {
        let unfolded = array![-15.0, -5.0, 5.0, 15.0, f64::NAN].into_dyn();
        let observed = array![-5.0, -5.0, -5.0, -5.0, -5.0].into_dyn();
        let counts = fold_counts(unfolded.view(), observed.view(), 10.0).unwrap();
        assert_eq!(counts, array![0i16, 0, 0, 1, 0].into_dyn());
    }

    #[test]
    fn unfold_to_reference_clips_and_preserves_missing() {
        let observed = array![2.0, 2.0, 2.0, f64::NAN].into_dyn();
        let reference = array![102.0, -98.0, 22.0, 0.0].into_dyn();
        let out = unfold_to_reference(observed.view(), reference.view(), 10.0, 2).unwrap();
        let expected = array![42.0, -38.0, 22.0, f64::NAN].into_dyn();
        for (actual, expected) in out.iter().zip(expected.iter()) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn shift2d_matches_roll_and_nan_fill() {
        let field = array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
        let shifted = shift2d(field.view(), 1, -1, false).unwrap();
        let expected = array![
            [f64::NAN, f64::NAN, f64::NAN],
            [1.0, 2.0, f64::NAN],
            [4.0, 5.0, f64::NAN]
        ];
        for (actual, expected) in shifted.iter().zip(expected.iter()) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn shift3d_applies_slice_by_slice() {
        let volume = array![
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0]
            ],
            [
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0]
            ]
        ];
        let shifted = shift3d(volume.view(), -1, 1, true).unwrap();
        let expected = array![
            [
                [f64::NAN, 4.0, 5.0, 6.0],
                [f64::NAN, 8.0, 9.0, 10.0],
                [f64::NAN, 0.0, 1.0, 2.0]
            ],
            [
                [f64::NAN, 16.0, 17.0, 18.0],
                [f64::NAN, 20.0, 21.0, 22.0],
                [f64::NAN, 12.0, 13.0, 14.0]
            ]
        ];
        assert_eq!(shifted.shape(), expected.shape());
        for (actual, expected) in shifted.iter().zip(expected.iter()) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn dual_prf_recovers_known_branch() {
        let truth = array![-26.0, -14.0, -2.0, 11.0, 23.0, 35.0].into_dyn();
        let low = wrap_to_nyquist(truth.view(), 10.0).unwrap();
        let high = wrap_to_nyquist(truth.view(), 16.0).unwrap();
        let result =
            dealias_dual_prf(low.view(), high.view(), 10.0, 16.0, Some(truth.view()), 32).unwrap();

        assert_eq!(result.paired_gates, truth.len());
        for (actual, expected) in result.velocity.iter().zip(truth.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn region_graph_recovers_reference_branch() {
        let truth = array![
            [-26.0, -14.0, -2.0, 11.0],
            [23.0, 35.0, 47.0, 59.0],
            [-8.0, 4.0, 16.0, 28.0],
            [40.0, 52.0, 64.0, 76.0]
        ];
        let observed = wrap_to_nyquist(truth.view().into_dyn(), 10.0)
            .unwrap()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let result = dealias_sweep_region_graph(
            observed.view(),
            10.0,
            Some(truth.view()),
            Some((2, 2)),
            0.75,
            6,
            8,
            true,
            DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
            DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
        )
        .unwrap();

        assert_eq!(result.region_count, 4);
        assert_eq!(result.assigned_regions, 4);
        assert_eq!(result.block_shape, (2, 2));
        assert_eq!(result.block_grid_shape, (2, 2));
        assert_eq!(result.merge_iterations, 6);
        assert!(result.wrap_azimuth);
        assert_eq!(result.regions_with_reference, 4);
        assert!(!result.safety_fallback_applied);
        assert_eq!(result.method, "region_graph_sweep");
        assert_eq!(result.safety_fallback_reason, None);
        let velocity = result
            .velocity
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        assert!(velocity.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn region_graph_skips_sparse_blocks_and_leaves_them_unresolved() {
        let observed = array![
            [
                5.0,
                6.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
            [
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN
            ],
        ];
        let result = dealias_sweep_region_graph(
            observed.view(),
            10.0,
            None,
            Some((4, 4)),
            0.75,
            6,
            8,
            true,
            DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
            DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
        )
        .unwrap();

        assert_eq!(result.region_count, 0);
        assert_eq!(result.seedable_region_count, 0);
        assert_eq!(result.assigned_regions, 0);
        assert_eq!(result.unresolved_regions, 0);
        assert!(result.velocity.iter().all(|value| value.is_nan()));
        assert_eq!(result.skipped_sparse_blocks, 1);
        assert_eq!(result.min_region_area, 4);
        assert!((result.min_valid_fraction - 0.15).abs() < 1e-12);
    }

    #[test]
    fn region_graph_leaves_disconnected_component_unresolved_without_reference() {
        let observed = array![
            [
                5.0,
                6.0,
                7.0,
                8.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                -9.0,
                -8.0,
                -7.0,
                -6.0
            ],
            [
                5.5,
                6.5,
                7.5,
                8.5,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                -8.5,
                -7.5,
                -6.5,
                -5.5
            ],
            [
                4.5,
                5.5,
                6.5,
                7.5,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                -9.5,
                -8.5,
                -7.5,
                -6.5
            ],
            [
                4.0,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                -10.0,
                -9.0,
                -8.0,
                -7.0
            ],
        ];
        let result = dealias_sweep_region_graph(
            observed.view(),
            10.0,
            None,
            Some((4, 4)),
            0.75,
            6,
            8,
            true,
            DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
            DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
        )
        .unwrap();

        assert_eq!(result.region_count, 2);
        assert_eq!(result.assigned_regions, 1);
        assert_eq!(result.unresolved_regions, 1);
        assert_eq!(result.pruned_disconnected_seedable_regions, 1);
        let finite_velocity = result
            .velocity
            .iter()
            .filter(|value| value.is_finite())
            .count();
        assert_eq!(finite_velocity, 16);
    }

    #[test]
    fn recursive_bootstrap_path_matches_region_graph_when_unsplit() {
        let truth = array![
            [-26.0, -14.0, -2.0, 11.0],
            [23.0, 35.0, 47.0, 59.0],
            [-8.0, 4.0, 16.0, 28.0],
            [40.0, 52.0, 64.0, 76.0]
        ];
        let observed = wrap_to_nyquist(truth.view().into_dyn(), 10.0)
            .unwrap()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let recursive = dealias_sweep_recursive(
            observed.view(),
            10.0,
            Some(truth.view()),
            0,
            999,
            0.60,
            0.70,
            8,
            true,
        )
        .unwrap();

        let velocity = recursive
            .velocity
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let observed_mae = observed
            .iter()
            .zip(truth.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum::<f64>()
            / observed.len() as f64;
        let recursive_mae = velocity
            .iter()
            .zip(truth.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .sum::<f64>()
            / velocity.len() as f64;
        assert!(recursive_mae < observed_mae);
        assert!(matches!(
            recursive.bootstrap_method,
            "region_graph_sweep" | "zw06_safety_fallback"
        ));
        assert_eq!(recursive.bootstrap_region_count, 1);
        assert!(matches!(
            recursive.method,
            "recursive_region_refinement"
                | "recursive_region_refinement_fallback_region_graph"
                | "recursive_region_refinement_fallback_zw06"
        ));
    }

    #[test]
    fn variational_defaults_to_zw06_bootstrap() {
        let truth = array![
            [-26.0, -14.0, -2.0, 11.0],
            [23.0, 35.0, 47.0, 59.0],
            [-8.0, 4.0, 16.0, 28.0],
            [40.0, 52.0, 64.0, 76.0]
        ];
        let observed = wrap_to_nyquist(truth.view().into_dyn(), 10.0)
            .unwrap()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let result = dealias_sweep_variational(
            observed.view(),
            10.0,
            Some(truth.view()),
            None,
            DEFAULT_VARIATIONAL_BOOTSTRAP_REFERENCE_WEIGHT,
            DEFAULT_VARIATIONAL_BOOTSTRAP_ITERATIONS,
            DEFAULT_VARIATIONAL_BOOTSTRAP_MAX_ABS_FOLD,
            DEFAULT_REGION_GRAPH_MIN_REGION_AREA,
            DEFAULT_REGION_GRAPH_MIN_VALID_FRACTION,
            8,
            1.0,
            0.50,
            0.20,
            8,
            true,
        )
        .unwrap();

        assert_eq!(result.bootstrap_method, "2d_multipass");
        assert_eq!(result.method, "zw06_bootstrap_then_coordinate_descent");
        let velocity = result
            .velocity
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        assert!(velocity.iter().all(|value| value.is_finite()));
        assert!(result.changed_gates > 0);
    }

    #[test]
    fn volume3d_recovers_reference_volume() {
        let truth = array![
            [[-26.0, -14.0, -2.0], [11.0, 23.0, 35.0]],
            [[-8.0, 4.0, 16.0], [28.0, 40.0, 52.0]],
            [[-20.0, -8.0, 4.0], [16.0, 28.0, 40.0]]
        ];
        let observed = wrap_to_nyquist(truth.view().into_dyn(), 10.0)
            .unwrap()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();
        let nyquist = [10.0, 10.0, 10.0];
        let result =
            dealias_volume_3d(observed.view(), &nyquist, Some(truth.view()), true, 4).unwrap();

        assert_eq!(result.seed_sweep, 0);
        assert_eq!(result.sweep_order, vec![0, 1, 2]);
        assert_eq!(result.per_sweep_valid_gates, vec![6, 6, 6]);
        assert_eq!(result.per_sweep_seeded_gates, vec![6, 4, 4]);
        assert_eq!(result.per_sweep_assigned_gates, vec![6, 6, 6]);
        assert_eq!(result.per_sweep_iterations_used, vec![3, 3, 3]);
        assert_eq!(result.iterations_used, 4);

        let expected_velocity = array![
            [[-26.0, -14.0, -2.0], [-9.0, 3.0, -5.0]],
            [[-8.0, 4.0, -4.0], [8.0, 20.0, 12.0]],
            [[-20.0, -8.0, 4.0], [-4.0, 8.0, 0.0]]
        ];
        let expected_folds = array![
            [[-1i16, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1]],
            [[-1, 0, 0], [0, 0, 0]]
        ];
        for (actual, expected) in result.velocity.iter().zip(expected_velocity.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(result.folds, expected_folds.into_dyn());
        assert!(result.confidence.iter().all(|value| value.is_finite()));
    }
}
