use crate::common::{
    array_to_vec_f32, array_to_vec_f64, array_to_vec_i16, js_error, metadata_json, optional_2d,
    require_1d, require_2d, FlatDealiasResult1D, FlatDealiasResult2D, FlatVelocityResult2D,
    WasmVadFit2D, WasmMlModel,
};
use serde_json::json;
use wasm_bindgen::prelude::*;

fn result_1d(
    velocity: ndarray::ArrayD<f64>,
    folds: ndarray::ArrayD<i16>,
    confidence: ndarray::ArrayD<f64>,
    reference: ndarray::ArrayD<f64>,
    len: usize,
    metadata: serde_json::Value,
) -> FlatDealiasResult1D {
    FlatDealiasResult1D {
        velocity: array_to_vec_f64(velocity),
        folds: array_to_vec_i16(folds),
        confidence: array_to_vec_f64(confidence),
        reference: array_to_vec_f64(reference),
        len,
        metadata_json: metadata_json(metadata),
    }
}

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

#[wasm_bindgen(js_name = dealiasRadialEs90)]
pub fn dealias_radial_es90(
    observed: Vec<f64>,
    nyquist: f64,
    reference: Vec<f64>,
    seed_index: Option<usize>,
    max_gap: usize,
    max_abs_step: Option<f64>,
) -> Result<FlatDealiasResult1D, JsValue> {
    let len = observed.len();
    let observed = require_1d(observed, len, "observed")?;
    let reference = if reference.is_empty() {
        None
    } else {
        Some(require_1d(reference, len, "reference")?)
    };
    let result = open_dealias_core::dealias_radial_es90(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        seed_index,
        max_gap,
        max_abs_step,
    )
    .map_err(js_error)?;
    Ok(result_1d(
        result.velocity,
        result.folds,
        result.confidence,
        result.reference,
        len,
        json!({
            "paper_family": "EiltsSmith1990",
            "method": "radial_continuity",
            "seed_index": result.seed_index,
            "max_gap": max_gap,
            "max_abs_step": max_abs_step,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepEs90)]
pub fn dealias_sweep_es90(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    max_gap: usize,
    max_abs_step: Option<f64>,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let result = open_dealias_core::dealias_sweep_es90(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        max_gap,
        max_abs_step,
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
            "paper_family": "EiltsSmith1990",
            "method": "sweep_radial_continuity",
            "max_gap": max_gap,
            "max_abs_step": max_abs_step,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepZw06)]
pub fn dealias_sweep_zw06(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    weak_threshold_fraction: f64,
    wrap_azimuth: bool,
    max_iterations_per_pass: usize,
    include_diagonals: bool,
    recenter_without_reference: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let result = open_dealias_core::dealias_sweep_zw06(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        weak_threshold_fraction,
        wrap_azimuth,
        max_iterations_per_pass,
        include_diagonals,
        recenter_without_reference,
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
            "paper_family": "JingWiener1993+ZhangWang2006",
            "method": "2d_multipass",
            "seeded_gates": result.seeded_gates,
            "assigned_gates": result.assigned_gates,
            "iterations_used": result.iterations_used,
            "weak_threshold_fraction": weak_threshold_fraction,
            "wrap_azimuth": wrap_azimuth,
            "include_diagonals": include_diagonals,
            "recenter_without_reference": recenter_without_reference,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepZw06Velocity)]
pub fn dealias_sweep_zw06_velocity(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    weak_threshold_fraction: f64,
    wrap_azimuth: bool,
    max_iterations_per_pass: usize,
    include_diagonals: bool,
    recenter_without_reference: bool,
) -> Result<FlatVelocityResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let result = open_dealias_core::dealias_sweep_zw06(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        weak_threshold_fraction,
        wrap_azimuth,
        max_iterations_per_pass,
        include_diagonals,
        recenter_without_reference,
    )
    .map_err(js_error)?;
    Ok(result_velocity_2d(
        result.velocity,
        rows,
        cols,
        json!({
            "paper_family": "JingWiener1993+ZhangWang2006",
            "method": "2d_multipass",
            "seeded_gates": result.seeded_gates,
            "assigned_gates": result.assigned_gates,
            "iterations_used": result.iterations_used,
            "weak_threshold_fraction": weak_threshold_fraction,
            "wrap_azimuth": wrap_azimuth,
            "include_diagonals": include_diagonals,
            "recenter_without_reference": recenter_without_reference,
            "output": "velocity_only",
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepRegionGraph)]
pub fn dealias_sweep_region_graph(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    block_rows: Option<usize>,
    block_cols: Option<usize>,
    reference_weight: f64,
    max_iterations: usize,
    max_abs_fold: i16,
    wrap_azimuth: bool,
    min_region_area: usize,
    min_valid_fraction: f64,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let block_shape = match (block_rows, block_cols) {
        (Some(r), Some(c)) => Some((r, c)),
        (None, None) => None,
        _ => return Err(JsValue::from_str("block_rows and block_cols must be provided together")),
    };
    let result = open_dealias_core::dealias_sweep_region_graph(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        block_shape,
        reference_weight,
        max_iterations,
        max_abs_fold,
        wrap_azimuth,
        min_region_area,
        min_valid_fraction,
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
            "paper_family": "PyARTRegionGraphLite",
            "method": result.method,
            "region_count": result.region_count,
            "assigned_regions": result.assigned_regions,
            "seed_region": result.seed_region,
            "block_shape": [result.block_shape.0, result.block_shape.1],
            "merge_iterations": result.merge_iterations,
            "wrap_azimuth": result.wrap_azimuth,
            "average_fold": result.average_fold,
            "regions_with_reference": result.regions_with_reference,
            "block_grid_shape": [result.block_grid_shape.0, result.block_grid_shape.1],
            "min_region_area": result.min_region_area,
            "min_valid_fraction": result.min_valid_fraction,
            "skipped_sparse_blocks": result.skipped_sparse_blocks,
            "safety_fallback_applied": result.safety_fallback_applied,
            "safety_fallback_reason": result.safety_fallback_reason,
            "candidate_cost": result.candidate_cost,
            "fallback_cost": result.fallback_cost,
            "disagreement_fraction": result.disagreement_fraction,
            "largest_disagreement_component": result.largest_disagreement_component,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepRegionGraphVelocity)]
pub fn dealias_sweep_region_graph_velocity(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    block_rows: Option<usize>,
    block_cols: Option<usize>,
    reference_weight: f64,
    max_iterations: usize,
    max_abs_fold: i16,
    wrap_azimuth: bool,
    min_region_area: usize,
    min_valid_fraction: f64,
) -> Result<FlatVelocityResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let block_shape = match (block_rows, block_cols) {
        (Some(r), Some(c)) => Some((r, c)),
        (None, None) => None,
        _ => return Err(JsValue::from_str("block_rows and block_cols must be provided together")),
    };
    let result = open_dealias_core::dealias_sweep_region_graph(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        block_shape,
        reference_weight,
        max_iterations,
        max_abs_fold,
        wrap_azimuth,
        min_region_area,
        min_valid_fraction,
    )
    .map_err(js_error)?;
    Ok(result_velocity_2d(
        result.velocity,
        rows,
        cols,
        json!({
            "paper_family": "PyARTRegionGraphLite",
            "method": result.method,
            "region_count": result.region_count,
            "assigned_regions": result.assigned_regions,
            "seed_region": result.seed_region,
            "block_shape": [result.block_shape.0, result.block_shape.1],
            "merge_iterations": result.merge_iterations,
            "wrap_azimuth": result.wrap_azimuth,
            "average_fold": result.average_fold,
            "regions_with_reference": result.regions_with_reference,
            "block_grid_shape": [result.block_grid_shape.0, result.block_grid_shape.1],
            "min_region_area": result.min_region_area,
            "min_valid_fraction": result.min_valid_fraction,
            "skipped_sparse_blocks": result.skipped_sparse_blocks,
            "safety_fallback_applied": result.safety_fallback_applied,
            "safety_fallback_reason": result.safety_fallback_reason,
            "candidate_cost": result.candidate_cost,
            "fallback_cost": result.fallback_cost,
            "disagreement_fraction": result.disagreement_fraction,
            "largest_disagreement_component": result.largest_disagreement_component,
            "output": "velocity_only",
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepRecursive)]
pub fn dealias_sweep_recursive(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    max_depth: usize,
    min_leaf_cells: usize,
    split_texture_fraction: f64,
    reference_weight: f64,
    max_abs_fold: i16,
    wrap_azimuth: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let result = open_dealias_core::dealias_sweep_recursive(
        observed.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        max_depth,
        min_leaf_cells,
        split_texture_fraction,
        reference_weight,
        max_abs_fold,
        wrap_azimuth,
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
            "paper_family": "R2D2StyleRecursiveLite",
            "method": result.method,
            "leaf_count": result.leaf_count,
            "max_depth": result.max_depth,
            "split_texture_fraction": result.split_texture_fraction,
            "reference_weight": result.reference_weight,
            "wrap_azimuth": result.wrap_azimuth,
            "root_texture": result.root_texture,
            "bootstrap_method": result.bootstrap_method,
            "bootstrap_region_count": result.bootstrap_region_count,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepVariationalRefine)]
pub fn dealias_sweep_variational_refine(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    initial_corrected: Vec<f64>,
    nyquist: f64,
    reference: Vec<f64>,
    max_abs_fold: i16,
    neighbor_weight: f64,
    reference_weight: f64,
    smoothness_weight: f64,
    max_iterations: usize,
    wrap_azimuth: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let initial_corrected = require_2d(initial_corrected, rows, cols, "initial_corrected")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let result = open_dealias_core::dealias_sweep_variational_refine(
        observed.view(),
        initial_corrected.view(),
        reference.as_ref().map(|value| value.view()),
        nyquist,
        max_abs_fold,
        neighbor_weight,
        reference_weight,
        smoothness_weight,
        max_iterations,
        wrap_azimuth,
    )
    .map_err(js_error)?;
    Ok(result_2d(
        result.velocity,
        result.folds,
        result.confidence,
        reference
            .map(|value| value.into_dyn())
            .unwrap_or_else(|| ndarray::Array2::from_elem((rows, cols), f64::NAN).into_dyn()),
        rows,
        cols,
        json!({
            "paper_family": "VariationalLite",
            "method": "coordinate_descent",
            "iterations_used": result.iterations_used,
            "changed_gates": result.changed_gates,
            "max_abs_fold": max_abs_fold,
            "wrap_azimuth": wrap_azimuth,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepVariational)]
pub fn dealias_sweep_variational(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    block_rows: Option<usize>,
    block_cols: Option<usize>,
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
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed_array = require_2d(observed, rows, cols, "observed")?;
    let reference_array = optional_2d(reference, rows, cols, "reference")?;
    let block_shape = match (block_rows, block_cols) {
        (Some(r), Some(c)) => Some((r, c)),
        (None, None) => None,
        _ => return Err(JsValue::from_str("block_rows and block_cols must be provided together")),
    };
    let result = open_dealias_core::dealias_sweep_variational(
        observed_array.view(),
        nyquist,
        reference_array.as_ref().map(|value| value.view()),
        block_shape,
        bootstrap_reference_weight,
        bootstrap_iterations,
        bootstrap_max_abs_fold,
        bootstrap_min_region_area,
        bootstrap_min_valid_fraction,
        max_abs_fold,
        neighbor_weight,
        reference_weight,
        smoothness_weight,
        max_iterations,
        wrap_azimuth,
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
            "paper_family": "VariationalLite",
            "method": result.method,
            "bootstrap": {
                "method": result.bootstrap_method,
                "region_count": result.bootstrap_region_count,
                "skipped_sparse_blocks": result.bootstrap_skipped_sparse_blocks,
                "assigned_gates": result.bootstrap_assigned_gates,
                "iterations_used": result.bootstrap_iterations_used,
                "safety_fallback_applied": result.bootstrap_safety_fallback_applied,
                "safety_fallback_reason": result.bootstrap_safety_fallback_reason,
            },
            "iterations_used": result.iterations_used,
            "changed_gates": result.changed_gates,
            "max_abs_fold": max_abs_fold,
            "wrap_azimuth": wrap_azimuth,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepVariationalVelocity)]
pub fn dealias_sweep_variational_velocity(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    reference: Vec<f64>,
    block_rows: Option<usize>,
    block_cols: Option<usize>,
    bootstrap_min_region_area: usize,
    bootstrap_min_valid_fraction: f64,
    max_abs_fold: i16,
    neighbor_weight: f64,
    reference_weight: f64,
    smoothness_weight: f64,
    max_iterations: usize,
    wrap_azimuth: bool,
) -> Result<FlatVelocityResult2D, JsValue> {
    let observed_array = require_2d(observed, rows, cols, "observed")?;
    let reference_array = optional_2d(reference, rows, cols, "reference")?;
    let block_shape = match (block_rows, block_cols) {
        (Some(r), Some(c)) => Some((r, c)),
        (None, None) => None,
        _ => return Err(JsValue::from_str("block_rows and block_cols must be provided together")),
    };
    let result = open_dealias_core::dealias_sweep_variational(
        observed_array.view(),
        nyquist,
        reference_array.as_ref().map(|value| value.view()),
        block_shape,
        0.75,
        6,
        8,
        bootstrap_min_region_area,
        bootstrap_min_valid_fraction,
        max_abs_fold,
        neighbor_weight,
        reference_weight,
        smoothness_weight,
        max_iterations,
        wrap_azimuth,
    )
    .map_err(js_error)?;
    Ok(result_velocity_2d(
        result.velocity,
        rows,
        cols,
        json!({
            "paper_family": "VariationalLite",
            "method": result.method,
            "iterations_used": result.iterations_used,
            "changed_gates": result.changed_gates,
            "max_abs_fold": max_abs_fold,
            "neighbor_weight": neighbor_weight,
            "reference_weight": reference_weight,
            "smoothness_weight": smoothness_weight,
            "wrap_azimuth": wrap_azimuth,
            "bootstrap_method": result.bootstrap_method,
            "bootstrap_region_count": result.bootstrap_region_count,
            "bootstrap_skipped_sparse_blocks": result.bootstrap_skipped_sparse_blocks,
            "bootstrap_assigned_gates": result.bootstrap_assigned_gates,
            "bootstrap_iterations_used": result.bootstrap_iterations_used,
            "bootstrap_safety_fallback_applied": result.bootstrap_safety_fallback_applied,
            "bootstrap_safety_fallback_reason": result.bootstrap_safety_fallback_reason,
            "output": "velocity_only",
        }),
    ))
}

#[wasm_bindgen(js_name = estimateUniformWindVad)]
pub fn estimate_uniform_wind_vad(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    azimuth_deg: Vec<f64>,
    elevation_deg: f64,
    sign: f64,
    max_iterations: usize,
    trim_quantile: f64,
    search_radius: Option<f64>,
) -> Result<WasmVadFit2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let azimuth_deg = require_1d(azimuth_deg, rows, "azimuth_deg")?;
    let result = open_dealias_core::estimate_uniform_wind_vad(
        observed.view(),
        nyquist,
        azimuth_deg.view(),
        elevation_deg,
        sign,
        max_iterations,
        trim_quantile,
        search_radius,
    )
    .map_err(js_error)?;
    Ok(WasmVadFit2D {
        u: result.u,
        v: result.v,
        offset: result.offset,
        rms: result.rms,
        iterations: result.iterations,
        reference: array_to_vec_f64(result.reference),
        rows,
        cols,
    })
}

#[wasm_bindgen(js_name = dealiasSweepXu11)]
pub fn dealias_sweep_xu11(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    azimuth_deg: Vec<f64>,
    elevation_deg: f64,
    external_reference: Vec<f64>,
    sign: f64,
    refine_with_multipass: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let azimuth_deg = require_1d(azimuth_deg, rows, "azimuth_deg")?;
    let external_reference = optional_2d(external_reference, rows, cols, "external_reference")?;
    let result = open_dealias_core::dealias_sweep_xu11(
        observed.view(),
        nyquist,
        azimuth_deg.view(),
        elevation_deg,
        external_reference.as_ref().map(|value| value.view()),
        sign,
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
            "paper_family": if refine_with_multipass { "Xu2011+ZhangWang2006" } else { "Xu2011" },
            "method": result.method,
            "u": result.u,
            "v": result.v,
            "offset": result.offset,
            "vad_rms": result.vad_rms,
            "vad_iterations": result.vad_iterations,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepXu11Velocity)]
pub fn dealias_sweep_xu11_velocity(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    azimuth_deg: Vec<f64>,
    elevation_deg: f64,
    external_reference: Vec<f64>,
    search_limit: f64,
    refine_with_multipass: bool,
) -> Result<FlatVelocityResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let azimuth_deg = require_1d(azimuth_deg, rows, "azimuth_deg")?;
    let external_reference = optional_2d(external_reference, rows, cols, "external_reference")?;
    let result = open_dealias_core::dealias_sweep_xu11(
        observed.view(),
        nyquist,
        azimuth_deg.view(),
        elevation_deg,
        external_reference.as_ref().map(|value| value.view()),
        search_limit,
        refine_with_multipass,
    )
    .map_err(js_error)?;
    Ok(result_velocity_2d(
        result.velocity,
        rows,
        cols,
        json!({
            "paper_family": if refine_with_multipass { "XuEtAl2011+ZhangWang2006" } else { "XuEtAl2011" },
            "method": result.method,
            "u": result.u,
            "v": result.v,
            "offset": result.offset,
            "vad_rms": result.vad_rms,
            "vad_iterations": result.vad_iterations,
            "search_limit": search_limit,
            "elevation_deg": elevation_deg,
            "output": "velocity_only",
        }),
    ))
}

#[wasm_bindgen(js_name = fitMlReferenceModel)]
pub fn fit_ml_reference_model(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    target_velocity: Vec<f64>,
    nyquist: Option<f64>,
    reference: Vec<f64>,
    azimuth_deg: Vec<f64>,
    ridge: f64,
) -> Result<WasmMlModel, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let target_velocity = require_2d(target_velocity, rows, cols, "target_velocity")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let azimuth_deg = if azimuth_deg.is_empty() {
        None
    } else {
        Some(require_1d(azimuth_deg, rows, "azimuth_deg")?)
    };
    let result = open_dealias_core::fit_ml_reference_model(
        observed.view(),
        target_velocity.view(),
        nyquist,
        reference.as_ref().map(|value| value.view()),
        azimuth_deg.as_ref().map(|value| value.view()),
        ridge,
    )
    .map_err(js_error)?;
    Ok(WasmMlModel::from_state(result))
}

#[wasm_bindgen(js_name = dealiasSweepMlWithModel)]
pub fn dealias_sweep_ml_with_model(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    model: &WasmMlModel,
    reference: Vec<f64>,
    azimuth_deg: Vec<f64>,
    refine_with_variational: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let azimuth_deg = if azimuth_deg.is_empty() {
        None
    } else {
        Some(require_1d(azimuth_deg, rows, "azimuth_deg")?)
    };
    let result = open_dealias_core::dealias_sweep_ml(
        observed.view(),
        nyquist,
        Some(model.as_state()),
        None,
        reference.as_ref().map(|value| value.view()),
        azimuth_deg.as_ref().map(|value| value.view()),
        model.as_state().ridge,
        refine_with_variational,
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
            "paper_family": "MLAssistLite",
            "method": "ridge_reference_predictor",
            "trained_from": result.trained_from,
            "train_rmse": result.train_rmse,
            "ridge": result.ridge,
            "feature_names": result.feature_names,
            "refine_method": result.refine_method,
            "refine_iterations": result.refine_iterations,
        }),
    ))
}

#[wasm_bindgen(js_name = dealiasSweepMlTrain)]
pub fn dealias_sweep_ml_train(
    observed: Vec<f64>,
    rows: usize,
    cols: usize,
    nyquist: f64,
    training_target: Vec<f64>,
    reference: Vec<f64>,
    azimuth_deg: Vec<f64>,
    ridge: f64,
    refine_with_variational: bool,
) -> Result<FlatDealiasResult2D, JsValue> {
    let observed = require_2d(observed, rows, cols, "observed")?;
    let training_target = require_2d(training_target, rows, cols, "training_target")?;
    let reference = optional_2d(reference, rows, cols, "reference")?;
    let azimuth_deg = if azimuth_deg.is_empty() {
        None
    } else {
        Some(require_1d(azimuth_deg, rows, "azimuth_deg")?)
    };
    let result = open_dealias_core::dealias_sweep_ml(
        observed.view(),
        nyquist,
        None,
        Some(training_target.view()),
        reference.as_ref().map(|value| value.view()),
        azimuth_deg.as_ref().map(|value| value.view()),
        ridge,
        refine_with_variational,
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
            "paper_family": "MLAssistLite",
            "method": "ridge_reference_predictor",
            "trained_from": result.trained_from,
            "train_rmse": result.train_rmse,
            "ridge": result.ridge,
            "feature_names": result.feature_names,
            "refine_method": result.refine_method,
            "refine_iterations": result.refine_iterations,
        }),
    ))
}
