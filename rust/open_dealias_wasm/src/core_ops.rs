use crate::common::{
    array_to_vec_f64, array_to_vec_i16, js_error, require_2d, require_3d,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = wrapToNyquistFlat)]
pub fn wrap_to_nyquist_flat(data: Vec<f64>, nyquist: f64) -> Result<Vec<f64>, JsValue> {
    let array = ndarray::Array1::from_vec(data).into_dyn();
    open_dealias_core::wrap_to_nyquist(array.view(), nyquist)
        .map(array_to_vec_f64)
        .map_err(js_error)
}

#[wasm_bindgen(js_name = foldCountsFlat)]
pub fn fold_counts_flat(
    unfolded: Vec<f64>,
    observed: Vec<f64>,
    nyquist: f64,
) -> Result<Vec<i16>, JsValue> {
    if unfolded.len() != observed.len() {
        return Err(JsValue::from_str("unfolded and observed must have the same length"));
    }
    let unfolded = ndarray::Array1::from_vec(unfolded).into_dyn();
    let observed = ndarray::Array1::from_vec(observed).into_dyn();
    open_dealias_core::fold_counts(unfolded.view(), observed.view(), nyquist)
        .map(array_to_vec_i16)
        .map_err(js_error)
}

#[wasm_bindgen(js_name = unfoldToReferenceFlat)]
pub fn unfold_to_reference_flat(
    observed: Vec<f64>,
    reference: Vec<f64>,
    nyquist: f64,
    max_abs_fold: i16,
) -> Result<Vec<f64>, JsValue> {
    if observed.len() != reference.len() {
        return Err(JsValue::from_str("observed and reference must have the same length"));
    }
    let observed = ndarray::Array1::from_vec(observed).into_dyn();
    let reference = ndarray::Array1::from_vec(reference).into_dyn();
    open_dealias_core::unfold_to_reference(
        observed.view(),
        reference.view(),
        nyquist,
        max_abs_fold,
    )
    .map(array_to_vec_f64)
    .map_err(js_error)
}

#[wasm_bindgen(js_name = shift2d)]
pub fn shift2d_2d(
    field: Vec<f64>,
    rows: usize,
    cols: usize,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> Result<Vec<f64>, JsValue> {
    let field = require_2d(field, rows, cols, "field")?;
    open_dealias_core::shift2d(field.view(), shift_az, shift_range, wrap_azimuth)
        .map(|array| array.iter().copied().collect())
        .map_err(js_error)
}

#[wasm_bindgen(js_name = shift3d)]
pub fn shift3d_3d(
    volume: Vec<f64>,
    sweeps: usize,
    rows: usize,
    cols: usize,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> Result<Vec<f64>, JsValue> {
    let volume = require_3d(volume, sweeps, rows, cols, "volume")?;
    open_dealias_core::shift3d(volume.view(), shift_az, shift_range, wrap_azimuth)
        .map(|array| array.iter().copied().collect())
        .map_err(js_error)
}

#[wasm_bindgen(js_name = neighborStack)]
pub fn neighbor_stack_2d(
    field: Vec<f64>,
    rows: usize,
    cols: usize,
    include_diagonals: bool,
    wrap_azimuth: bool,
) -> Result<Vec<f64>, JsValue> {
    let field = require_2d(field, rows, cols, "field")?;
    open_dealias_core::neighbor_stack(field.view(), include_diagonals, wrap_azimuth)
        .map(|array| array.iter().copied().collect())
        .map_err(js_error)
}

#[wasm_bindgen(js_name = texture3x3)]
pub fn texture_3x3_2d(
    field: Vec<f64>,
    rows: usize,
    cols: usize,
    wrap_azimuth: bool,
) -> Result<Vec<f64>, JsValue> {
    let field = require_2d(field, rows, cols, "field")?;
    open_dealias_core::texture_3x3(field.view(), wrap_azimuth)
        .map(|array| array.iter().copied().collect())
        .map_err(js_error)
}
