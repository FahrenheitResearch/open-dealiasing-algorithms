use crate::common::{
    array_to_vec_f32, array_to_vec_f64, array_to_vec_i16, js_error, metadata_json, optional_2d,
    require_1d, require_2d, FlatDealiasResult1D, FlatDealiasResult2D, FlatVelocityResult2D,
    WasmMlModel, WasmVadFit2D,
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

fn view2_from_slice<'a>(
    data: &'a [f64],
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<ndarray::ArrayView2<'a, f64>, JsValue> {
    ndarray::ArrayView2::from_shape((rows, cols), data)
        .map_err(|_| JsValue::from_str(&format!("{name} must match shape ({rows}, {cols})")))
}

#[wasm_bindgen]
pub struct SweepVelocityWorkspace {
    rows: usize,
    cols: usize,
    observed: Vec<f32>,
    reference: Vec<f32>,
    observed_work: Vec<f64>,
    reference_work: Vec<f64>,
    velocity: Vec<f32>,
    metadata_json: String,
}

impl SweepVelocityWorkspace {
    fn len_internal(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    fn reset_shape(&mut self, rows: usize, cols: usize) {
        let len = rows.saturating_mul(cols);
        self.rows = rows;
        self.cols = cols;
        self.observed.resize(len, f32::NAN);
        self.reference.resize(len, f32::NAN);
        self.observed_work.resize(len, f64::NAN);
        self.reference_work.resize(len, f64::NAN);
        self.velocity.resize(len, f32::NAN);
        self.metadata_json.clear();
    }

    fn sync_inputs(&mut self) {
        for (dst, src) in self.observed_work.iter_mut().zip(self.observed.iter()) {
            *dst = f64::from(*src);
        }
        for (dst, src) in self.reference_work.iter_mut().zip(self.reference.iter()) {
            *dst = f64::from(*src);
        }
    }

    fn observed_view(&self) -> Result<ndarray::ArrayView2<'_, f64>, JsValue> {
        view2_from_slice(&self.observed_work, self.rows, self.cols, "observed")
    }

    fn reference_view(&self) -> Result<Option<ndarray::ArrayView2<'_, f64>>, JsValue> {
        if !self.reference_work.iter().any(|value| value.is_finite()) {
            return Ok(None);
        }
        view2_from_slice(&self.reference_work, self.rows, self.cols, "reference").map(Some)
    }

    fn store_velocity(
        &mut self,
        velocity: ndarray::ArrayD<f64>,
        metadata: serde_json::Value,
    ) -> Result<(), JsValue> {
        let velocity = velocity
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| JsValue::from_str("velocity result must be 2D"))?;
        if velocity.len() != self.len_internal() {
            return Err(JsValue::from_str(
                "velocity result length does not match workspace",
            ));
        }
        for (dst, src) in self.velocity.iter_mut().zip(velocity.iter()) {
            *dst = *src as f32;
        }
        self.metadata_json = metadata_json(metadata);
        Ok(())
    }
}

#[wasm_bindgen]
impl SweepVelocityWorkspace {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize) -> Result<SweepVelocityWorkspace, JsValue> {
        let len = rows.saturating_mul(cols);
        Ok(SweepVelocityWorkspace {
            rows,
            cols,
            observed: vec![f32::NAN; len],
            reference: vec![f32::NAN; len],
            observed_work: vec![f64::NAN; len],
            reference_work: vec![f64::NAN; len],
            velocity: vec![f32::NAN; len],
            metadata_json: String::new(),
        })
    }

    #[wasm_bindgen(js_name = resize)]
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.reset_shape(rows, cols);
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.len_internal()
    }

    #[wasm_bindgen(js_name = observedPtr)]
    pub fn observed_ptr(&mut self) -> *mut f32 {
        self.observed.as_mut_ptr()
    }

    #[wasm_bindgen(js_name = referencePtr)]
    pub fn reference_ptr(&mut self) -> *mut f32 {
        self.reference.as_mut_ptr()
    }

    #[wasm_bindgen(js_name = velocityPtr)]
    pub fn velocity_ptr(&self) -> *const f32 {
        self.velocity.as_ptr()
    }

    #[wasm_bindgen(js_name = clearReference)]
    pub fn clear_reference(&mut self) {
        self.reference.fill(f32::NAN);
    }

    #[wasm_bindgen(js_name = metadataJson)]
    pub fn metadata_json(&self) -> String {
        self.metadata_json.clone()
    }

    #[wasm_bindgen(js_name = runZw06VelocityOnly)]
    pub fn run_zw06_velocity_only(
        &mut self,
        nyquist: f64,
        weak_threshold_fraction: f64,
        wrap_azimuth: bool,
        max_iterations_per_pass: usize,
        include_diagonals: bool,
        recenter_without_reference: bool,
    ) -> Result<(), JsValue> {
        self.sync_inputs();
        let observed = self.observed_view()?;
        let reference = self.reference_view()?;
        let result = open_dealias_core::dealias_sweep_zw06(
            observed,
            nyquist,
            reference,
            weak_threshold_fraction,
            wrap_azimuth,
            max_iterations_per_pass,
            include_diagonals,
            recenter_without_reference,
        )
        .map_err(js_error)?;
        self.store_velocity(
            result.velocity,
            json!({
                "paper_family": "JingWiener1993+ZhangWang2006",
                "method": "2d_multipass",
                "seeded_gates": result.seeded_gates,
                "assigned_gates": result.assigned_gates,
                "iterations_used": result.iterations_used,
                "output": "velocity_only_workspace",
            }),
        )
    }

    #[wasm_bindgen(js_name = runRegionGraphVelocityOnly)]
    pub fn run_region_graph_velocity_only(
        &mut self,
        nyquist: f64,
        block_rows: Option<usize>,
        block_cols: Option<usize>,
        reference_weight: f64,
        max_iterations: usize,
        max_abs_fold: i16,
        wrap_azimuth: bool,
        min_region_area: usize,
        min_valid_fraction: f64,
    ) -> Result<(), JsValue> {
        self.sync_inputs();
        let observed = self.observed_view()?;
        let reference = self.reference_view()?;
        let block_shape = match (block_rows, block_cols) {
            (Some(r), Some(c)) => Some((r, c)),
            (None, None) => None,
            _ => {
                return Err(JsValue::from_str(
                    "block_rows and block_cols must be provided together",
                ))
            }
        };
        let result = open_dealias_core::dealias_sweep_region_graph(
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
        )
        .map_err(js_error)?;
        self.store_velocity(
            result.velocity,
            json!({
                "paper_family": "PyARTRegionGraphLite",
                "method": result.method,
                "region_count": result.region_count,
                "seedable_region_count": result.seedable_region_count,
                "assigned_regions": result.assigned_regions,
                "unresolved_regions": result.unresolved_regions,
                "skipped_sparse_blocks": result.skipped_sparse_blocks,
                "pruned_disconnected_seedable_regions": result.pruned_disconnected_seedable_regions,
                "safety_fallback_applied": result.safety_fallback_applied,
                "safety_fallback_reason": result.safety_fallback_reason,
                "output": "velocity_only_workspace",
            }),
        )
    }

    #[wasm_bindgen(js_name = runVariationalVelocityOnly)]
    pub fn run_variational_velocity_only(
        &mut self,
        nyquist: f64,
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
    ) -> Result<(), JsValue> {
        self.sync_inputs();
        let observed = self.observed_view()?;
        let reference = self.reference_view()?;
        let block_shape = match (block_rows, block_cols) {
            (Some(r), Some(c)) => Some((r, c)),
            (None, None) => None,
            _ => {
                return Err(JsValue::from_str(
                    "block_rows and block_cols must be provided together",
                ))
            }
        };
        let result = open_dealias_core::dealias_sweep_variational(
            observed,
            nyquist,
            reference,
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
        self.store_velocity(
            result.velocity,
            json!({
                "paper_family": "VariationalLite",
                "method": result.method,
                "bootstrap_method": result.bootstrap_method,
                "bootstrap_region_count": result.bootstrap_region_count,
                "bootstrap_unresolved_regions": result.bootstrap_unresolved_regions,
                "bootstrap_skipped_sparse_blocks": result.bootstrap_skipped_sparse_blocks,
                "bootstrap_safety_fallback_applied": result.bootstrap_safety_fallback_applied,
                "bootstrap_safety_fallback_reason": result.bootstrap_safety_fallback_reason,
                "iterations_used": result.iterations_used,
                "changed_gates": result.changed_gates,
                "output": "velocity_only_workspace",
            }),
        )
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
        _ => {
            return Err(JsValue::from_str(
                "block_rows and block_cols must be provided together",
            ))
        }
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
            "seedable_region_count": result.seedable_region_count,
            "assigned_regions": result.assigned_regions,
            "unresolved_regions": result.unresolved_regions,
            "pruned_disconnected_seedable_regions": result.pruned_disconnected_seedable_regions,
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
        _ => {
            return Err(JsValue::from_str(
                "block_rows and block_cols must be provided together",
            ))
        }
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
            "seedable_region_count": result.seedable_region_count,
            "assigned_regions": result.assigned_regions,
            "unresolved_regions": result.unresolved_regions,
            "pruned_disconnected_seedable_regions": result.pruned_disconnected_seedable_regions,
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
        _ => {
            return Err(JsValue::from_str(
                "block_rows and block_cols must be provided together",
            ))
        }
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
                "unresolved_regions": result.bootstrap_unresolved_regions,
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
        _ => {
            return Err(JsValue::from_str(
                "block_rows and block_cols must be provided together",
            ))
        }
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
            "bootstrap_unresolved_regions": result.bootstrap_unresolved_regions,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_zw06_velocity_matches_shape_and_metadata() {
        let mut workspace = SweepVelocityWorkspace::new(2, 4).unwrap();
        workspace
            .observed
            .copy_from_slice(&[5.0, -4.0, 3.0, -2.0, 4.0, -3.0, 2.0, -1.0]);
        workspace.clear_reference();
        workspace
            .run_zw06_velocity_only(10.0, 0.35, true, 4, true, true)
            .unwrap();
        assert_eq!(workspace.len(), 8);
        assert!(workspace.velocity.iter().any(|value| value.is_finite()));
        assert!(workspace
            .metadata_json()
            .contains("\"method\":\"2d_multipass\""));
    }

    #[test]
    fn workspace_region_graph_keeps_sparse_case_unresolved() {
        let mut workspace = SweepVelocityWorkspace::new(8, 8).unwrap();
        workspace.observed.copy_from_slice(&[
            5.0,
            6.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ]);
        workspace.clear_reference();
        workspace
            .run_region_graph_velocity_only(10.0, Some(4), Some(4), 0.75, 6, 8, true, 4, 0.15)
            .unwrap();
        assert!(workspace.velocity.iter().all(|value| value.is_nan()));
        assert!(workspace
            .metadata_json()
            .contains("\"region_count\":0"));
    }
}
