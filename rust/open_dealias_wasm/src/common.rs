use ndarray::{Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};
use open_dealias_core as core;
use serde_json::Value;
use wasm_bindgen::prelude::*;

pub(crate) fn js_error(err: core::DealiasError) -> JsValue {
    JsValue::from_str(&err.to_string())
}

pub(crate) fn metadata_json(value: Value) -> String {
    value.to_string()
}

pub(crate) fn require_1d(data: Vec<f64>, expected_len: usize, name: &str) -> Result<Array1<f64>, JsValue> {
    if data.len() != expected_len {
        return Err(JsValue::from_str(&format!(
            "{name} length must be {expected_len}, got {}",
            data.len()
        )));
    }
    ArrayD::from_shape_vec(ndarray::IxDyn(&[expected_len]), data)
        .and_then(|array| array.into_dimensionality::<Ix1>())
        .map_err(|_| JsValue::from_str(&format!("{name} must be 1D")))
}

pub(crate) fn require_2d(
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<Array2<f64>, JsValue> {
    let expected_len = rows.saturating_mul(cols);
    if data.len() != expected_len {
        return Err(JsValue::from_str(&format!(
            "{name} length must be {expected_len} for shape ({rows}, {cols}), got {}",
            data.len()
        )));
    }
    ArrayD::from_shape_vec(ndarray::IxDyn(&[rows, cols]), data)
        .and_then(|array| array.into_dimensionality::<Ix2>())
        .map_err(|_| JsValue::from_str(&format!("{name} must be 2D")))
}

pub(crate) fn optional_2d(
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<Option<Array2<f64>>, JsValue> {
    if data.is_empty() {
        return Ok(None);
    }
    require_2d(data, rows, cols, name).map(Some)
}

pub(crate) fn require_3d(
    data: Vec<f64>,
    sweeps: usize,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<Array3<f64>, JsValue> {
    let expected_len = sweeps.saturating_mul(rows).saturating_mul(cols);
    if data.len() != expected_len {
        return Err(JsValue::from_str(&format!(
            "{name} length must be {expected_len} for shape ({sweeps}, {rows}, {cols}), got {}",
            data.len()
        )));
    }
    ArrayD::from_shape_vec(ndarray::IxDyn(&[sweeps, rows, cols]), data)
        .and_then(|array| array.into_dimensionality::<Ix3>())
        .map_err(|_| JsValue::from_str(&format!("{name} must be 3D")))
}

pub(crate) fn optional_3d(
    data: Vec<f64>,
    sweeps: usize,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<Option<Array3<f64>>, JsValue> {
    if data.is_empty() {
        return Ok(None);
    }
    require_3d(data, sweeps, rows, cols, name).map(Some)
}

pub(crate) fn require_positive_nyquist(nyquist: f64, name: &str) -> Result<f64, JsValue> {
    if nyquist.is_finite() && nyquist > 0.0 {
        Ok(nyquist)
    } else {
        Err(JsValue::from_str(&format!("{name} must be positive, got {nyquist}")))
    }
}

pub(crate) fn resolve_nyquist_per_sweep(
    nyquist: Vec<f64>,
    sweeps: usize,
    name: &str,
) -> Result<Vec<f64>, JsValue> {
    if nyquist.is_empty() {
        return Err(JsValue::from_str(&format!("{name} must not be empty")));
    }
    let values = if nyquist.len() == 1 {
        vec![nyquist[0]; sweeps]
    } else if nyquist.len() == sweeps {
        nyquist
    } else {
        return Err(JsValue::from_str(&format!(
            "{name} must have length 1 or {sweeps}, got {}",
            nyquist.len()
        )));
    };
    for value in &values {
        require_positive_nyquist(*value, name)?;
    }
    Ok(values)
}

pub(crate) fn array_to_vec_f64(array: ArrayD<f64>) -> Vec<f64> {
    array.iter().copied().collect()
}

pub(crate) fn array_to_vec_f32(array: ArrayD<f64>) -> Vec<f32> {
    array.iter().map(|value| *value as f32).collect()
}

pub(crate) fn array_to_vec_i16(array: ArrayD<i16>) -> Vec<i16> {
    array.iter().copied().collect()
}

#[wasm_bindgen(getter_with_clone)]
pub struct FlatDealiasResult1D {
    pub velocity: Vec<f64>,
    pub folds: Vec<i16>,
    pub confidence: Vec<f64>,
    pub reference: Vec<f64>,
    pub len: usize,
    pub metadata_json: String,
}

#[wasm_bindgen(getter_with_clone)]
pub struct FlatDealiasResult2D {
    pub velocity: Vec<f64>,
    pub folds: Vec<i16>,
    pub confidence: Vec<f64>,
    pub reference: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub metadata_json: String,
}

#[wasm_bindgen(getter_with_clone)]
pub struct FlatVelocityResult2D {
    pub velocity: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub metadata_json: String,
}

#[wasm_bindgen(getter_with_clone)]
pub struct FlatDealiasResult3D {
    pub velocity: Vec<f64>,
    pub folds: Vec<i16>,
    pub confidence: Vec<f64>,
    pub reference: Vec<f64>,
    pub sweeps: usize,
    pub rows: usize,
    pub cols: usize,
    pub metadata_json: String,
}

#[wasm_bindgen(getter_with_clone)]
pub struct WasmVadFit2D {
    pub u: f64,
    pub v: f64,
    pub offset: f64,
    pub rms: f64,
    pub iterations: usize,
    pub reference: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[wasm_bindgen]
pub struct WasmMlModel {
    state: core::MlModelState,
}

#[wasm_bindgen]
impl WasmMlModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights: Vec<f64>,
        feature_names_json: String,
        ridge: f64,
        train_rmse: f64,
        mode: String,
        nyquist: Option<f64>,
    ) -> Result<WasmMlModel, JsValue> {
        let feature_names: Vec<String> = serde_json::from_str(&feature_names_json)
            .map_err(|err| JsValue::from_str(&format!("invalid feature_names_json: {err}")))?;
        if !weights.is_empty() && weights.len() != feature_names.len() {
            return Err(JsValue::from_str(&format!(
                "weights length {} does not match feature_names length {}",
                weights.len(),
                feature_names.len()
            )));
        }
        Ok(WasmMlModel {
            state: core::MlModelState {
                weights,
                feature_names,
                ridge,
                train_rmse,
                mode,
                nyquist,
            },
        })
    }

    #[wasm_bindgen(getter)]
    pub fn weights(&self) -> Vec<f64> {
        self.state.weights.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn feature_names_json(&self) -> String {
        serde_json::to_string(&self.state.feature_names).unwrap_or_else(|_| "[]".to_string())
    }

    #[wasm_bindgen(getter)]
    pub fn ridge(&self) -> f64 {
        self.state.ridge
    }

    #[wasm_bindgen(getter)]
    pub fn train_rmse(&self) -> f64 {
        self.state.train_rmse
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        self.state.mode.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn nyquist(&self) -> Option<f64> {
        self.state.nyquist
    }
}

impl WasmMlModel {
    pub(crate) fn from_state(state: core::MlModelState) -> WasmMlModel {
        WasmMlModel { state }
    }

    pub(crate) fn as_state(&self) -> &core::MlModelState {
        &self.state
    }
}
