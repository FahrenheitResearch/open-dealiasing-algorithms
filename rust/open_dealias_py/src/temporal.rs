use crate::common::run_without_gil;
use ndarray::{ArrayView1, ArrayView3};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PySequence};

#[pyfunction]
#[pyo3(signature = (observed, nyquist, azimuth_deg, elevation_deg=0.0, sign=1.0, max_iterations=6, trim_quantile=0.85, search_radius=None))]
fn estimate_uniform_wind_vad<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    azimuth_deg: PyReadonlyArrayDyn<'py, f64>,
    elevation_deg: f64,
    sign: f64,
    max_iterations: usize,
    trim_quantile: f64,
    search_radius: Option<f64>,
) -> PyResult<(f64, f64, f64, f64, usize, Bound<'py, PyArrayDyn<f64>>)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "observed must be 2D, got {}D",
            observed.ndim()
        )));
    }
    let observed = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let azimuth = azimuth_deg.as_array();
    if azimuth.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "azimuth_deg must be 1D, got {}D",
            azimuth.ndim()
        )));
    }
    let azimuth: ArrayView1<'_, f64> = azimuth
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))?;
    let result = run_without_gil(py, move || {
        core::estimate_uniform_wind_vad(
            observed,
            nyquist,
            azimuth,
            elevation_deg,
            sign,
            max_iterations,
            trim_quantile,
            search_radius,
        )
    })?;
    Ok((
        result.u,
        result.v,
        result.offset,
        result.rms,
        result.iterations,
        result.reference.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, azimuth_deg, elevation_deg=0.0, external_reference=None, sign=1.0, refine_with_multipass=true))]
fn dealias_sweep_xu11<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    azimuth_deg: PyReadonlyArrayDyn<'py, f64>,
    elevation_deg: f64,
    external_reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    sign: f64,
    refine_with_multipass: bool,
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
    let observed = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let azimuth = azimuth_deg.as_array();
    if azimuth.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "azimuth_deg must be 1D, got {}D",
            azimuth.ndim()
        )));
    }
    let azimuth: ArrayView1<'_, f64> = azimuth
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))?;
    let external_reference_view = external_reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("external_reference must be 2D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_xu11(
            observed,
            nyquist,
            azimuth,
            elevation_deg,
            external_reference_view,
            sign,
            refine_with_multipass,
        )
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item(
        "paper_family",
        if refine_with_multipass {
            "Xu2011+ZhangWang2006"
        } else {
            "Xu2011"
        },
    )?;
    metadata.set_item("method", result.method)?;
    metadata.set_item("u", result.u)?;
    metadata.set_item("v", result.v)?;
    metadata.set_item("offset", result.offset)?;
    metadata.set_item("vad_rms", result.vad_rms)?;
    metadata.set_item("vad_iterations", result.vad_iterations)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, previous_corrected=None, background_reference=None, shift_az=0, shift_range=0, wrap_azimuth=true, refine_with_multipass=true))]
fn dealias_sweep_jh01<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    previous_corrected: Option<PyReadonlyArrayDyn<'py, f64>>,
    background_reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
    refine_with_multipass: bool,
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
    let observed = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let previous_view = previous_corrected
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("previous_corrected must be 2D"))
        })
        .transpose()?;
    let background_view = background_reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("background_reference must be 2D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_jh01(
            observed,
            nyquist,
            previous_view,
            background_view,
            shift_az,
            shift_range,
            wrap_azimuth,
            refine_with_multipass,
        )
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item(
        "paper_family",
        if refine_with_multipass {
            "JamesHouze2001+ZhangWang2006"
        } else {
            "JamesHouze2001"
        },
    )?;
    metadata.set_item("method", result.method)?;
    metadata.set_item("shift_az", shift_az)?;
    metadata.set_item("shift_range", shift_range)?;
    metadata.set_item(
        "fill_policy",
        if refine_with_multipass {
            "temporal_reference_then_multipass_cleanup"
        } else {
            "temporal_reference_only"
        },
    )?;
    metadata.set_item("valid_gates", result.valid_gates)?;
    metadata.set_item("assigned_gates", result.assigned_gates)?;
    metadata.set_item("unresolved_gates", result.unresolved_gates)?;
    metadata.set_item("resolved_fraction", result.resolved_fraction)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

fn resolve_background_uv(
    background_uv: Option<&Bound<'_, PyAny>>,
    n_sweeps: usize,
) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
    let Some(background_uv) = background_uv else {
        return Ok(None);
    };
    if let Ok((u, v)) = background_uv.extract::<(f64, f64)>() {
        return Ok(Some((vec![u; n_sweeps], vec![v; n_sweeps])));
    }
    let seq = background_uv.downcast::<PySequence>()?;
    if seq.len()? != 2 {
        return Err(PyValueError::new_err("background_uv must have length 2"));
    }
    let u_item = seq.get_item(0)?;
    let v_item = seq.get_item(1)?;

    let parse_component = |item: Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
        if let Ok(value) = item.extract::<f64>() {
            return Ok(vec![value; n_sweeps]);
        }
        if let Ok(values) = item.extract::<Vec<f64>>() {
            if values.len() != n_sweeps {
                return Err(PyValueError::new_err(format!(
                    "background_uv arrays must have length {n_sweeps}"
                )));
            }
            return Ok(values);
        }
        if let Ok(values) = item.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            let arr = values.as_array();
            if arr.ndim() != 1 || arr.len() != n_sweeps {
                return Err(PyValueError::new_err(format!(
                    "background_uv arrays must be 1D length {n_sweeps}"
                )));
            }
            return Ok(arr.iter().copied().collect());
        }
        Err(PyValueError::new_err(
            "background_uv elements must be scalars or 1D arrays",
        ))
    };

    Ok(Some((parse_component(u_item)?, parse_component(v_item)?)))
}

fn resolve_volume_nyquist<'py>(nyquist: &Bound<'py, PyAny>, n_sweeps: usize) -> PyResult<Vec<f64>> {
    if let Ok(value) = nyquist.extract::<f64>() {
        if value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "nyquist must be positive, got {value}"
            )));
        }
        return Ok(vec![value; n_sweeps]);
    }
    if let Ok(values) = nyquist.extract::<Vec<f64>>() {
        if values.len() == 1 {
            let value = values[0];
            if value <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "nyquist must be positive, got {value}"
                )));
            }
            return Ok(vec![value; n_sweeps]);
        }
        if values.len() != n_sweeps {
            return Err(PyValueError::new_err(format!(
                "nyquist must have length {n_sweeps}, got {}",
                values.len()
            )));
        }
        if let Some(value) = values
            .iter()
            .copied()
            .find(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "nyquist values must be positive, got {value}"
            )));
        }
        return Ok(values);
    }
    if let Ok(values) = nyquist.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        let arr = values.as_array();
        if arr.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "nyquist must be scalar or 1D, got {}D",
                arr.ndim()
            )));
        }
        let nyq: Vec<f64> = arr.iter().copied().collect();
        if nyq.len() != n_sweeps {
            return Err(PyValueError::new_err(format!(
                "nyquist must have length {n_sweeps}, got {}",
                nyq.len()
            )));
        }
        if let Some(value) = nyq
            .iter()
            .copied()
            .find(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "nyquist values must be positive, got {value}"
            )));
        }
        return Ok(nyq);
    }
    Err(PyValueError::new_err(
        "nyquist must be a positive scalar or 1D array",
    ))
}

#[pyfunction]
#[pyo3(signature = (observed_volume, nyquist, azimuth_deg, elevation_deg, previous_volume=None, background_uv=None, shift_az=0, shift_range=0, wrap_azimuth=true))]
fn dealias_volume_jh01<'py>(
    py: Python<'py>,
    observed_volume: PyReadonlyArrayDyn<'py, f64>,
    nyquist: &Bound<'py, PyAny>,
    azimuth_deg: PyReadonlyArrayDyn<'py, f64>,
    elevation_deg: PyReadonlyArrayDyn<'py, f64>,
    previous_volume: Option<PyReadonlyArrayDyn<'py, f64>>,
    background_uv: Option<&Bound<'py, PyAny>>,
    shift_az: isize,
    shift_range: isize,
    wrap_azimuth: bool,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed_volume.as_array();
    if observed.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "observed_volume must be 3D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView3<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed_volume must be 3D"))?;
    let nyq = resolve_volume_nyquist(nyquist, observed.shape()[0])?;
    let azimuth = azimuth_deg.as_array();
    let elevation = elevation_deg.as_array();
    if azimuth.ndim() != 1 || elevation.ndim() != 1 {
        return Err(PyValueError::new_err(
            "azimuth_deg and elevation_deg must be 1D",
        ));
    }
    let azimuth: ArrayView1<'_, f64> = azimuth
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))?;
    let elevation: ArrayView1<'_, f64> = elevation
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("elevation_deg must be 1D"))?;
    let previous_view = previous_volume
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| PyValueError::new_err("previous_volume must be 3D"))
        })
        .transpose()?;
    let background = resolve_background_uv(background_uv, observed.shape()[0])?;
    let (background_u, background_v) = match background {
        Some((u, v)) => (Some(u), Some(v)),
        None => (None, None),
    };
    let result = run_without_gil(py, move || {
        core::dealias_volume_jh01(
            observed,
            &nyq,
            azimuth,
            elevation,
            previous_view,
            background_u.as_deref(),
            background_v.as_deref(),
            shift_az,
            shift_range,
            wrap_azimuth,
        )
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "JamesHouze2001")?;
    metadata.set_item("method", "descending_volume_4dd_lite")?;
    metadata.set_item("elevation_order_desc", result.elevation_order_desc.clone())?;
    metadata.set_item("shift_az", shift_az)?;
    metadata.set_item("shift_range", shift_range)?;
    metadata.set_item("fill_policy", "descending_volume_reference_then_cleanup")?;
    metadata.set_item("valid_gates", result.valid_gates)?;
    metadata.set_item("assigned_gates", result.assigned_gates)?;
    metadata.set_item("unresolved_gates", result.unresolved_gates)?;
    metadata.set_item("resolved_fraction", result.resolved_fraction)?;
    let per_sweep = PyList::empty(py);
    for sweep in 0..result.per_sweep_valid_gates.len() {
        let item = PyDict::new(py);
        item.set_item("valid_gates", result.per_sweep_valid_gates[sweep])?;
        item.set_item("assigned_gates", result.per_sweep_assigned_gates[sweep])?;
        item.set_item("unresolved_gates", result.per_sweep_unresolved_gates[sweep])?;
        item.set_item(
            "resolved_fraction",
            result.per_sweep_resolved_fraction[sweep],
        )?;
        per_sweep.append(item)?;
    }
    metadata.set_item("per_sweep", per_sweep)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

#[pyfunction]
#[pyo3(signature = (observed_volume, nyquist, reference_volume=None, wrap_azimuth=true, max_iterations=4))]
fn dealias_volume_3d<'py>(
    py: Python<'py>,
    observed_volume: PyReadonlyArrayDyn<'py, f64>,
    nyquist: &Bound<'py, PyAny>,
    reference_volume: Option<PyReadonlyArrayDyn<'py, f64>>,
    wrap_azimuth: bool,
    max_iterations: usize,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed_volume.as_array();
    if observed.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "observed_volume must be 3D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView3<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed_volume must be 3D"))?;

    let reference_view = reference_volume
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| PyValueError::new_err("reference_volume must be 3D"))
        })
        .transpose()?;

    let nyq = resolve_volume_nyquist(nyquist, observed.shape()[0])?;
    let result = run_without_gil(py, move || {
        core::dealias_volume_3d(observed, &nyq, reference_view, wrap_azimuth, max_iterations)
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "UNRAVEL-style-3D")?;
    metadata.set_item("method", "volume_3d_continuity")?;
    metadata.set_item("seed_sweep", result.seed_sweep)?;
    metadata.set_item("iterations_used", result.iterations_used)?;
    metadata.set_item("wrap_azimuth", wrap_azimuth)?;
    metadata.set_item("sweep_order", result.sweep_order.clone())?;
    let per_sweep = PyList::empty(py);
    for sweep in 0..result.per_sweep_valid_gates.len() {
        let item = PyDict::new(py);
        let valid = result.per_sweep_valid_gates[sweep];
        let assigned = result.per_sweep_assigned_gates[sweep];
        item.set_item("paper_family", "JingWiener1993+ZhangWang2006")?;
        item.set_item("method", "2d_multipass")?;
        item.set_item("valid_gates", valid)?;
        item.set_item("seeded_gates", result.per_sweep_seeded_gates[sweep])?;
        item.set_item("assigned_gates", assigned)?;
        item.set_item(
            "resolved_fraction",
            if valid > 0 {
                assigned as f64 / valid as f64
            } else {
                0.0
            },
        )?;
        item.set_item("iterations_used", result.per_sweep_iterations_used[sweep])?;
        item.set_item("wrap_azimuth", wrap_azimuth)?;
        per_sweep.append(item)?;
    }
    metadata.set_item("per_sweep", per_sweep)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_uniform_wind_vad, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_xu11, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_jh01, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_volume_jh01, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_volume_3d, m)?)?;
    Ok(())
}
