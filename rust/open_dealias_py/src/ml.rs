use crate::common::run_without_gil;
use ndarray::{Array1, ArrayView2};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

#[pyfunction]
#[pyo3(signature = (observed, target_velocity, nyquist=None, reference=None, azimuth_deg=None, ridge=1.0))]
fn fit_ml_reference_model<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    target_velocity: PyReadonlyArrayDyn<'py, f64>,
    nyquist: Option<f64>,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    azimuth_deg: Option<PyReadonlyArrayDyn<'py, f64>>,
    ridge: f64,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Vec<String>,
    f64,
    f64,
    String,
    Option<f64>,
)> {
    let observed = observed.as_array();
    let target = target_velocity.as_array();
    if observed.ndim() != 2 || target.ndim() != 2 {
        return Err(PyValueError::new_err(
            "observed and target_velocity must be 2D",
        ));
    }
    let observed: ArrayView2<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let target: ArrayView2<'_, f64> = target
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("target_velocity must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let azimuth_view = azimuth_deg
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::fit_ml_reference_model(
            observed,
            target,
            nyquist,
            reference_view,
            azimuth_view,
            ridge,
        )
    })?;
    Ok((
        Array1::from(result.weights.clone())
            .into_dyn()
            .into_pyarray(py),
        result.feature_names,
        result.ridge,
        result.train_rmse,
        result.mode,
        result.nyquist,
    ))
}

fn extract_ml_model_state(
    model: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<core::MlModelState>> {
    let Some(model) = model else {
        return Ok(None);
    };
    let weights_any = model.getattr("weights")?;
    let weights = if let Ok(values) = weights_any.extract::<Vec<f64>>() {
        values
    } else if let Ok(values) = weights_any.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        let arr = values.as_array();
        if arr.ndim() != 1 {
            return Err(PyValueError::new_err("model.weights must be 1D"));
        }
        arr.iter().copied().collect()
    } else {
        return Err(PyValueError::new_err(
            "model.weights must be a 1D float array",
        ));
    };
    let feature_names = model.getattr("feature_names")?.extract::<Vec<String>>()?;
    if !weights.is_empty() && weights.len() != feature_names.len() {
        return Err(PyValueError::new_err(format!(
            "model.weights length {} does not match feature_names length {}",
            weights.len(),
            feature_names.len()
        )));
    }
    let ridge = model.getattr("ridge")?.extract::<f64>()?;
    let train_rmse = model.getattr("train_rmse")?.extract::<f64>()?;
    let mode = model.getattr("mode")?.extract::<String>()?;
    let nyquist = model.getattr("nyquist")?.extract::<Option<f64>>()?;
    Ok(Some(core::MlModelState {
        weights,
        feature_names,
        ridge,
        train_rmse,
        mode,
        nyquist,
    }))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, model=None, training_target=None, reference=None, azimuth_deg=None, ridge=1.0, refine_with_variational=true))]
fn dealias_sweep_ml<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    model: Option<&Bound<'py, PyAny>>,
    training_target: Option<PyReadonlyArrayDyn<'py, f64>>,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    azimuth_deg: Option<PyReadonlyArrayDyn<'py, f64>>,
    ridge: f64,
    refine_with_variational: bool,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "observed must be 2D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView2<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let model_state = extract_ml_model_state(model)?;
    let training_target_view = training_target
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("training_target must be 2D"))
        })
        .transpose()?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let azimuth_view = azimuth_deg
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_ml(
            observed,
            nyquist,
            model_state.as_ref(),
            training_target_view,
            reference_view,
            azimuth_view,
            ridge,
            refine_with_variational,
        )
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "MLAssistLite")?;
    metadata.set_item("method", "ridge_reference_predictor")?;
    metadata.set_item("trained_from", result.trained_from.clone())?;
    metadata.set_item("train_rmse", result.train_rmse)?;
    metadata.set_item("ridge", result.ridge)?;
    metadata.set_item("feature_names", result.feature_names.clone())?;
    if let Some(refine_method) = result.refine_method.as_ref() {
        metadata.set_item("refine_method", refine_method.clone())?;
    }
    if let Some(iterations) = result.refine_iterations {
        metadata.set_item("refine_iterations", iterations)?;
    }
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_ml_reference_model, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_ml, m)?)?;
    Ok(())
}
