use crate::common::{
    array_to_vec_f32, array_to_vec_f64, array_to_vec_i16, js_error, metadata_json, optional_2d,
    optional_3d, require_1d, require_2d, require_3d, resolve_nyquist_per_sweep,
    FlatDealiasResult2D, FlatDealiasResult3D, FlatVelocityResult2D,
};
use serde_json::json;
use wasm_bindgen::prelude::*;

fn result_2d(
    velocity: ndarray::ArrayD<f64>,
    folds: ndarray::ArrayD<i16>,
    confidence: ndarray::ArrayD<f64>,
    reference: ndarray::ArrayD<f64>,
    rows: usize,
    cols: usize,
    metadata: serde_json::Value,
) -> FlatDealiasResult2D {
    FlatDealiasResult2D {
        velocity: array_to_vec_f64(velocity),
        folds: array_to_vec_i16(folds),
        confidence: array_to_vec_f64(confidence),
        reference: array_to_vec_f64(reference),
        rows,
        cols,
        metadata_json: metadata_json(metadata),
    }
}

fn result_velocity_2d(
    velocity: ndarray::ArrayD<f64>,
    rows: usize,
    cols: usize,
    metadata: serde_json::Value,
) -> FlatVelocityResult2D {
    FlatVelocityResult2D {
        velocity: array_to_vec_f32(velocity),
        rows,
        cols,
        metadata_json: metadata_json(metadata),
    }
}

fn result_3d(
    velocity: ndarray::ArrayD<f64>,
    folds: ndarray::ArrayD<i16>,
    confidence: ndarray::ArrayD<f64>,
    reference: ndarray::ArrayD<f64>,
    sweeps: usize,
    rows: usize,
    cols: usize,
    metadata: serde_json::Value,
) -> FlatDealiasResult3D {
    FlatDealiasResult3D {
        velocity: array_to_vec_f64(velocity),
        folds: array_to_vec_i16(folds),
        confidence: array_to_vec_f64(confidence),
        reference: array_to_vec_f64(reference),
        sweeps,
        rows,
        cols,
        metadata_json: metadata_json(metadata),
    }
}

#[wasm_bindgen(js_name = dealiasDualPrfPacked)]
pub fn dealias_dual_prf_packed(
    low_observed: Vec<f64>,
    high_observed: Vec<f64>,
    rows: usize,
    cols: usize,
    low_nyquist: f64,
    high_nyquist: f64,
    reference: Vec<f64>,
    max_abs_fold: i16,
) -> Result<FlatDealiasResult2D, JsValue> {
    let low_observed = require_2d(low_observed, rows, cols, "low_observed")?;
    let high_observed = require_2d(high_observed, rows, cols, "high_observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let result = open_dealias_core::dealias_dual_prf(
        low_observed.view().into_dyn(),
        high_observed.view().into_dyn(),
        low_nyquist,
        high_nyquist,
        reference.as_ref().map(|value| value.view().into_dyn()),
        max_abs_fold,
    )
    .map_err(js_error)?;
    Ok(result_2d(
        result.velocity,
        result.folds,
        result.confidence,
        result.reference,
        rows,
        cols,
        json!({
            "paper_family": "DualPRF",
            "method": "dual_prf_pair_search",
            "low_nyquist": low_nyquist,
            "high_nyquist": high_nyquist,
            "max_abs_fold": max_abs_fold,
            "low_valid_gates": result.low_valid_gates,
            "high_valid_gates": result.high_valid_gates,
            "paired_gates": result.paired_gates,
            "low_branch_mean_fold": result.low_branch_mean_fold,
            "high_branch_mean_fold": result.high_branch_mean_fold,
            "mean_pair_gap": result.mean_pair_gap,
            "max_pair_gap": result.max_pair_gap,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepJH01Packed)]
pub fn dealias_sweep_jh01_packed(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    previous_corrected: Vec<f64>,
    background_reference: Vec<f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
    refine_with_multipass: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let previous_corrected = optional_2d(previous_corrected, rows, cols, "previous_corrected")?;
    let background_reference = optional_2d(background_reference, rows, cols, "background_reference")?;
    let result = open_dealias_core::dealias_sweep_jh01(
        observed.view(),
        nyquist,
        previous_corrected.as_ref().map(|value| value.view()),
        background_reference.as_ref().map(|value| value.view()),
        shift_az,
        shift_range,
        wrap_azimuth,
        refine_with_multipass,
    )
    .map_err(js_error)?;
    Ok(result_2d(
        result.velocity,
        result.folds,
        result.confidence,
        result.reference,
        rows,
        cols,
        json!({
            "paper_family": if refine_with_multipass { "JamesHouze2001+ZhangWang2006" } else { "JamesHouze2001" },
            "method": result.method,
            "shift_az": shift_az,
            "shift_range": shift_range,
            "valid_gates": result.valid_gates,
            "assigned_gates": result.assigned_gates,
            "unresolved_gates": result.unresolved_gates,
            "resolved_fraction": result.resolved_fraction,
            "wrap_azimuth": wrap_azimuth,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepJH01Velocity)]
pub fn dealias_sweep_jh01_velocity(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    previous_corrected: Vec<f64>,
    background_reference: Vec<f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
    refine_with_multipass: bool,
) -> Result<FlatVelocityResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let previous_corrected = optional_2d(previous_corrected, rows, cols, "previous_corrected")?;
    let background_reference = optional_2d(background_reference, rows, cols, "background_reference")?;
    let result = open_dealias_core::dealias_sweep_jh01(
        observed.view(),
        nyquist,
        previous_corrected.as_ref().map(|value| value.view()),
        background_reference.as_ref().map(|value| value.view()),
        shift_az,
        shift_range,
        wrap_azimuth,
        refine_with_multipass,
    )
    .map_err(js_error)?;
    Ok(result_velocity_2d(
        result.velocity,
        rows,
        cols,
        json!({
            "paper_family": if refine_with_multipass { "JamesHouze2001+ZhangWang2006" } else { "JamesHouze2001" },
            "method": result.method,
            "shift_az": shift_az,
            "shift_range": shift_range,
            "valid_gates": result.valid_gates,
            "assigned_gates": result.assigned_gates,
            "unresolved_gates": result.unresolved_gates,
            "resolved_fraction": result.resolved_fraction,
            "wrap_azimuth": wrap_azimuth,
            "output": "velocity_only",
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasVolumeJH01Packed)]
pub fn dealias_volume_jh01_packed(
    observed_volume: Vec<f64>,
    sweeps: usize,
    rows: usize,
    cols: usize,
    nyquist: Vec<f64>,
    azimuth_deg: Vec<f64>,
    elevation_deg: Vec<f64>,
    previous_volume: Vec<f64>,
    background_u: Vec<f64>,
    background_v: Vec<f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> Result<FlatDealiasResult3D, JsValue> {
    let observed_volume = require_3d(observed_volume, sweeps, rows, cols, "observed_volume")?;
    let nyquist = resolve_nyquist_per_sweep(nyquist, sweeps, "nyquist")?;
    let azimuth_deg = require_1d(azimuth_deg, rows, "azimuth_deg")?;
    let elevation_deg = require_1d(elevation_deg, sweeps, "elevation_deg")?;
    let previous_volume = optional_3d(previous_volume, sweeps, rows, cols, "previous_volume")?;
    let background_u = if background_u.is_empty() {
        None
    } else {
        Some(require_1d(background_u, sweeps, "background_u")?)
    };
    let background_v = if background_v.is_empty() {
        None
    } else {
        Some(require_1d(background_v, sweeps, "background_v")?)
    };
    let result = open_dealias_core::dealias_volume_jh01(
        observed_volume.view(),
        &nyquist,
        azimuth_deg.view(),
        elevation_deg.view(),
        previous_volume.as_ref().map(|value| value.view()),
        background_u.as_ref().map(|value| value.as_slice().expect("contiguous")),
        background_v.as_ref().map(|value| value.as_slice().expect("contiguous")),
        shift_az,
        shift_range,
        wrap_azimuth,
    )
    .map_err(js_error)?;
    Ok(result_3d(
        result.velocity,
        result.folds,
        result.confidence,
        result.reference,
        sweeps,
        rows,
        cols,
        json!({
            "paper_family": "JamesHouze2001",
            "method": "descending_volume_4dd_lite",
            "elevation_order_desc": result.elevation_order_desc,
            "shift_az": shift_az,
            "shift_range": shift_range,
            "valid_gates": result.valid_gates,
            "assigned_gates": result.assigned_gates,
            "unresolved_gates": result.unresolved_gates,
            "resolved_fraction": result.resolved_fraction,
            "per_sweep_valid_gates": result.per_sweep_valid_gates,
            "per_sweep_assigned_gates": result.per_sweep_assigned_gates,
            "per_sweep_unresolved_gates": result.per_sweep_unresolved_gates,
            "per_sweep_resolved_fraction": result.per_sweep_resolved_fraction,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasVolume3DPacked)]
pub fn dealias_volume_3d_packed(
    observed_volume: Vec<f64>,
    sweeps: usize,
    rows: usize,
    cols: usize,
    nyquist: Vec<f64>,
    reference_volume: Vec<f64>,
    wrap_azimuth: bool,
    max_iterations: usize,
) -> Result<FlatDealiasResult3D, JsValue> {
    let observed_volume = require_3d(observed_volume, sweeps, rows, cols, "observed_volume")?;
    let nyquist = resolve_nyquist_per_sweep(nyquist, sweeps, "nyquist")?;
    let reference_volume = optional_3d(reference_volume, sweeps, rows, cols, "reference_volume")?;
    let result = open_dealias_core::dealias_volume_3d(
        observed_volume.view(),
        &nyquist,
        reference_volume.as_ref().map(|value| value.view()),
        wrap_azimuth,
        max_iterations,
    )
    .map_err(js_error)?;
    Ok(result_3d(
        result.velocity,
        result.folds,
        result.confidence,
        result.reference,
        sweeps,
        rows,
        cols,
        json!({
            "paper_family": "UNRAVEL-style-3D",
            "method": "volume_3d_continuity",
            "seed_sweep": result.seed_sweep,
            "iterations_used": result.iterations_used,
            "wrap_azimuth": wrap_azimuth,
            "sweep_order": result.sweep_order,
            "per_sweep_valid_gates": result.per_sweep_valid_gates,
            "per_sweep_seeded_gates": result.per_sweep_seeded_gates,
            "per_sweep_assigned_gates": result.per_sweep_assigned_gates,
            "per_sweep_iterations_used": result.per_sweep_iterations_used,
        }),
    ))
}
