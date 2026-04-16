use crate::common::run_without_gil;
use ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyfunction]
#[pyo3(signature = (velocity, reflectivity=None, texture=None, min_reflectivity=None, max_texture=None, min_gate_fraction_in_ray=0.02, wrap_azimuth=true))]
fn build_velocity_qc_mask<'py>(
    py: Python<'py>,
    velocity: PyReadonlyArrayDyn<'py, f64>,
    reflectivity: Option<PyReadonlyArrayDyn<'py, f64>>,
    texture: Option<PyReadonlyArrayDyn<'py, f64>>,
    min_reflectivity: Option<f64>,
    max_texture: Option<f64>,
    min_gate_fraction_in_ray: f64,
    wrap_azimuth: bool,
) -> PyResult<Bound<'py, PyArrayDyn<bool>>> {
    let velocity = velocity.as_array();
    if velocity.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "velocity must be 2D, got {}D",
            velocity.ndim()
        )));
    }
    let velocity: ArrayView2<'_, f64> = velocity
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("velocity must be 2D"))?;
    let reflectivity_view = reflectivity
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reflectivity must be 2D"))
        })
        .transpose()?;
    let texture_view = texture
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("texture must be 2D"))
        })
        .transpose()?;
    let out = run_without_gil(py, move || {
        core::build_velocity_qc_mask(
            velocity,
            reflectivity_view,
            texture_view,
            min_reflectivity,
            max_texture,
            min_gate_fraction_in_ray,
            wrap_azimuth,
        )
    })?;
    Ok(out.into_dyn().into_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_velocity_qc_mask, m)?)?;
    Ok(())
}
