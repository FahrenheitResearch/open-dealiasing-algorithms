use crate::common::{map_error, run_without_gil};
use ndarray::{ArrayD, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

#[pyfunction]
fn wrap_to_nyquist<'py>(
    py: Python<'py>,
    velocity: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let out: ArrayD<f64> =
        core::wrap_to_nyquist(velocity.as_array(), nyquist).map_err(map_error)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn fold_counts<'py>(
    py: Python<'py>,
    unfolded: PyReadonlyArrayDyn<'py, f64>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
) -> PyResult<Bound<'py, PyArrayDyn<i16>>> {
    let out =
        core::fold_counts(unfolded.as_array(), observed.as_array(), nyquist).map_err(map_error)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn unfold_to_reference<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    reference: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    max_abs_fold: i16,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let out = core::unfold_to_reference(
        observed.as_array(),
        reference.as_array(),
        nyquist,
        max_abs_fold,
    )
    .map_err(map_error)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn shift2d<'py>(
    py: Python<'py>,
    field: PyReadonlyArrayDyn<'py, f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let field = field.as_array();
    if field.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "shift2d expects a 2D array, got {}D",
            field.ndim()
        )));
    }
    let field: ArrayView2<'_, f64> = field
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("shift2d expects a 2D array"))?;
    let out = core::shift2d(field, shift_az, shift_range, wrap_azimuth).map_err(map_error)?;
    Ok(out.into_dyn().into_pyarray(py))
}

#[pyfunction]
fn shift3d<'py>(
    py: Python<'py>,
    volume: PyReadonlyArrayDyn<'py, f64>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let volume = volume.as_array();
    if volume.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "shift3d expects a 3D array, got {}D",
            volume.ndim()
        )));
    }
    let volume: ArrayView3<'_, f64> = volume
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("shift3d expects a 3D array"))?;
    let out = core::shift3d(volume, shift_az, shift_range, wrap_azimuth).map_err(map_error)?;
    Ok(out.into_dyn().into_pyarray(py))
}

#[pyfunction]
fn neighbor_stack<'py>(
    py: Python<'py>,
    field: PyReadonlyArrayDyn<'py, f64>,
    include_diagonals: bool,
    wrap_azimuth: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let field = field.as_array();
    if field.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "neighbor_stack expects a 2D array, got {}D",
            field.ndim()
        )));
    }
    let field: ArrayView2<'_, f64> = field
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("neighbor_stack expects a 2D array"))?;
    let out = core::neighbor_stack(field, include_diagonals, wrap_azimuth).map_err(map_error)?;
    Ok(out.into_dyn().into_pyarray(py))
}

#[pyfunction]
fn texture_3x3<'py>(
    py: Python<'py>,
    field: PyReadonlyArrayDyn<'py, f64>,
    wrap_azimuth: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let field = field.as_array();
    if field.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "texture_3x3 expects a 2D array, got {}D",
            field.ndim()
        )));
    }
    let field: ArrayView2<'_, f64> = field
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("texture_3x3 expects a 2D array"))?;
    let out = core::texture_3x3(field, wrap_azimuth).map_err(map_error)?;
    Ok(out.into_dyn().into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (low_observed, high_observed, low_nyquist, high_nyquist, reference=None, max_abs_fold=32))]
fn dealias_dual_prf<'py>(
    py: Python<'py>,
    low_observed: PyReadonlyArrayDyn<'py, f64>,
    high_observed: PyReadonlyArrayDyn<'py, f64>,
    low_nyquist: f64,
    high_nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    max_abs_fold: i16,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let low_observed = low_observed.as_array();
    let high_observed = high_observed.as_array();
    let reference_array = reference.as_ref().map(|value| value.as_array());
    let result = run_without_gil(py, move || {
        core::dealias_dual_prf(
            low_observed,
            high_observed,
            low_nyquist,
            high_nyquist,
            reference_array,
            max_abs_fold,
        )
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "DualPRF")?;
    metadata.set_item("method", "dual_prf_pair_search")?;
    metadata.set_item("low_nyquist", low_nyquist)?;
    metadata.set_item("high_nyquist", high_nyquist)?;
    metadata.set_item("max_abs_fold", max_abs_fold)?;
    metadata.set_item("low_valid_gates", result.low_valid_gates)?;
    metadata.set_item("high_valid_gates", result.high_valid_gates)?;
    metadata.set_item("paired_gates", result.paired_gates)?;
    metadata.set_item("low_branch_mean_fold", result.low_branch_mean_fold)?;
    metadata.set_item("high_branch_mean_fold", result.high_branch_mean_fold)?;
    metadata.set_item("mean_pair_gap", result.mean_pair_gap)?;
    metadata.set_item("max_pair_gap", result.max_pair_gap)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wrap_to_nyquist, m)?)?;
    m.add_function(wrap_pyfunction!(fold_counts, m)?)?;
    m.add_function(wrap_pyfunction!(unfold_to_reference, m)?)?;
    m.add_function(wrap_pyfunction!(shift2d, m)?)?;
    m.add_function(wrap_pyfunction!(shift3d, m)?)?;
    m.add_function(wrap_pyfunction!(neighbor_stack, m)?)?;
    m.add_function(wrap_pyfunction!(texture_3x3, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_dual_prf, m)?)?;
    Ok(())
}
