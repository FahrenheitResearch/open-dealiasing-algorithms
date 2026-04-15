use ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PySequence};

fn map_error(err: core::DealiasError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn wrap_to_nyquist<'py>(
    py: Python<'py>,
    velocity: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let out: ArrayD<f64> = core::wrap_to_nyquist(velocity.as_array(), nyquist).map_err(map_error)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn fold_counts<'py>(
    py: Python<'py>,
    unfolded: PyReadonlyArrayDyn<'py, f64>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
) -> PyResult<Bound<'py, PyArrayDyn<i16>>> {
    let out = core::fold_counts(unfolded.as_array(), observed.as_array(), nyquist).map_err(map_error)?;
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
        return Err(PyValueError::new_err(format!("shift2d expects a 2D array, got {}D", field.ndim())));
    }
    let field: ArrayView2<'_, f64> = field.into_dimensionality().map_err(|_| PyValueError::new_err("shift2d expects a 2D array"))?;
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
        return Err(PyValueError::new_err(format!("shift3d expects a 3D array, got {}D", volume.ndim())));
    }
    let volume: ArrayView3<'_, f64> = volume.into_dimensionality().map_err(|_| PyValueError::new_err("shift3d expects a 3D array"))?;
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
        return Err(PyValueError::new_err(format!("neighbor_stack expects a 2D array, got {}D", field.ndim())));
    }
    let field: ArrayView2<'_, f64> = field.into_dimensionality().map_err(|_| PyValueError::new_err("neighbor_stack expects a 2D array"))?;
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
        return Err(PyValueError::new_err(format!("texture_3x3 expects a 2D array, got {}D", field.ndim())));
    }
    let field: ArrayView2<'_, f64> = field.into_dimensionality().map_err(|_| PyValueError::new_err("texture_3x3 expects a 2D array"))?;
    let out = core::texture_3x3(field, wrap_azimuth).map_err(map_error)?;
    Ok(out.into_dyn().into_pyarray(py))
}

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
        return Err(PyValueError::new_err(format!("velocity must be 2D, got {}D", velocity.ndim())));
    }
    let velocity: ArrayView2<'_, f64> = velocity.into_dimensionality().map_err(|_| PyValueError::new_err("velocity must be 2D"))?;
    let reflectivity_view = reflectivity
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reflectivity must be 2D")))
        .transpose()?;
    let texture_view = texture
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("texture must be 2D")))
        .transpose()?;
    let out = core::build_velocity_qc_mask(
        velocity,
        reflectivity_view,
        texture_view,
        min_reflectivity,
        max_texture,
        min_gate_fraction_in_ray,
        wrap_azimuth,
    )
    .map_err(map_error)?;
    Ok(out.into_dyn().into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, reference=None, seed_index=None, max_gap=3, max_abs_step=None))]
fn dealias_radial_es90<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    seed_index: Option<usize>,
    max_gap: usize,
    max_abs_step: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Option<usize>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 1 {
        return Err(PyValueError::new_err(format!("observed must be 1D, got {}D", observed.ndim())));
    }
    let observed: ArrayView1<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 1D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix1>().map_err(|_| PyValueError::new_err("reference must be 1D")))
        .transpose()?;
    let result = core::dealias_radial_es90(observed, nyquist, reference_view, seed_index, max_gap, max_abs_step).map_err(map_error)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        result.seed_index,
    ))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, reference=None, max_gap=3, max_abs_step=None))]
fn dealias_sweep_es90<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    max_gap: usize,
    max_abs_step: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_es90(observed, nyquist, reference_view, max_gap, max_abs_step).map_err(map_error)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, reference=None, weak_threshold_fraction=0.35, wrap_azimuth=true, max_iterations_per_pass=12, include_diagonals=true, recenter_without_reference=true))]
fn dealias_sweep_zw06<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    weak_threshold_fraction: f64,
    wrap_azimuth: bool,
    max_iterations_per_pass: usize,
    include_diagonals: bool,
    recenter_without_reference: bool,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_zw06(
        observed,
        nyquist,
        reference_view,
        weak_threshold_fraction,
        wrap_azimuth,
        max_iterations_per_pass,
        include_diagonals,
        recenter_without_reference,
    )
    .map_err(map_error)?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "JingWiener1993+ZhangWang2006")?;
    metadata.set_item("method", "2d_multipass")?;
    metadata.set_item("seeded_gates", result.seeded_gates)?;
    metadata.set_item("assigned_gates", result.assigned_gates)?;
    metadata.set_item("iterations_used", result.iterations_used)?;
    metadata.set_item("wrap_azimuth", wrap_azimuth)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

#[pyfunction]
#[pyo3(signature = (observed, initial_corrected, nyquist, reference=None, max_abs_fold=8, neighbor_weight=1.0, reference_weight=0.50, smoothness_weight=0.20, max_iterations=8, wrap_azimuth=true))]
fn dealias_sweep_variational_refine<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    initial_corrected: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    max_abs_fold: i16,
    neighbor_weight: f64,
    reference_weight: f64,
    smoothness_weight: f64,
    max_iterations: usize,
    wrap_azimuth: bool,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed.as_array();
    let initial_corrected = initial_corrected.as_array();
    if observed.ndim() != 2 || initial_corrected.ndim() != 2 {
        return Err(PyValueError::new_err("observed and initial_corrected must be 2D"));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let initial_corrected: ArrayView2<'_, f64> = initial_corrected.into_dimensionality().map_err(|_| PyValueError::new_err("initial_corrected must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_variational_refine(
        observed,
        initial_corrected,
        reference_view,
        nyquist,
        max_abs_fold,
        neighbor_weight,
        reference_weight,
        smoothness_weight,
        max_iterations,
        wrap_azimuth,
    )
    .map_err(map_error)?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "VariationalLite")?;
    metadata.set_item("method", "coordinate_descent")?;
    metadata.set_item("iterations_used", result.iterations_used)?;
    metadata.set_item("changed_gates", result.changed_gates)?;
    metadata.set_item("max_abs_fold", max_abs_fold)?;
    metadata.set_item("wrap_azimuth", wrap_azimuth)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        metadata,
    ))
}

#[pyfunction]
#[pyo3(signature = (observed, nyquist, reference=None, max_depth=5, min_leaf_cells=24, split_texture_fraction=0.60, reference_weight=0.70, max_abs_fold=8, wrap_azimuth=true))]
fn dealias_sweep_recursive<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    max_depth: usize,
    min_leaf_cells: usize,
    split_texture_fraction: f64,
    reference_weight: f64,
    max_abs_fold: i16,
    wrap_azimuth: bool,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_recursive(
        observed,
        nyquist,
        reference_view,
        max_depth,
        min_leaf_cells,
        split_texture_fraction,
        reference_weight,
        max_abs_fold,
        wrap_azimuth,
    )
    .map_err(map_error)?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "R2D2StyleRecursiveLite")?;
    metadata.set_item("method", result.method)?;
    metadata.set_item("leaf_count", result.leaf_count)?;
    metadata.set_item("max_depth", result.max_depth)?;
    metadata.set_item("split_texture_fraction", result.split_texture_fraction)?;
    metadata.set_item("reference_weight", result.reference_weight)?;
    metadata.set_item("wrap_azimuth", result.wrap_azimuth)?;
    metadata.set_item("root_texture", result.root_texture)?;
    metadata.set_item("bootstrap_method", result.bootstrap_method)?;
    metadata.set_item("bootstrap_region_count", result.bootstrap_region_count)?;
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

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
) -> PyResult<(
    f64,
    f64,
    f64,
    f64,
    usize,
    Bound<'py, PyArrayDyn<f64>>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let azimuth = azimuth_deg.as_array();
    if azimuth.ndim() != 1 {
        return Err(PyValueError::new_err(format!("azimuth_deg must be 1D, got {}D", azimuth.ndim())));
    }
    let azimuth: ArrayView1<'_, f64> = azimuth.into_dimensionality().map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))?;
    let result = core::estimate_uniform_wind_vad(
        observed,
        nyquist,
        azimuth,
        elevation_deg,
        sign,
        max_iterations,
        trim_quantile,
        search_radius,
    )
    .map_err(map_error)?;
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
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let azimuth = azimuth_deg.as_array();
    if azimuth.ndim() != 1 {
        return Err(PyValueError::new_err(format!("azimuth_deg must be 1D, got {}D", azimuth.ndim())));
    }
    let azimuth: ArrayView1<'_, f64> = azimuth.into_dimensionality().map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))?;
    let external_reference_view = external_reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("external_reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_xu11(
        observed,
        nyquist,
        azimuth,
        elevation_deg,
        external_reference_view,
        sign,
        refine_with_multipass,
    )
    .map_err(map_error)?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", if refine_with_multipass { "Xu2011+ZhangWang2006" } else { "Xu2011" })?;
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
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let previous_view = previous_corrected
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("previous_corrected must be 2D")))
        .transpose()?;
    let background_view = background_reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("background_reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_jh01(
        observed,
        nyquist,
        previous_view,
        background_view,
        shift_az,
        shift_range,
        wrap_azimuth,
        refine_with_multipass,
    )
    .map_err(map_error)?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", if refine_with_multipass { "JamesHouze2001+ZhangWang2006" } else { "JamesHouze2001" })?;
    metadata.set_item("method", result.method)?;
    metadata.set_item("shift_az", shift_az)?;
    metadata.set_item("shift_range", shift_range)?;
    metadata.set_item("fill_policy", if refine_with_multipass { "temporal_reference_then_multipass_cleanup" } else { "temporal_reference_only" })?;
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

fn resolve_background_uv(background_uv: Option<&Bound<'_, PyAny>>, n_sweeps: usize) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
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
                return Err(PyValueError::new_err(format!("background_uv arrays must have length {n_sweeps}")));
            }
            return Ok(values);
        }
        if let Ok(values) = item.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            let arr = values.as_array();
            if arr.ndim() != 1 || arr.len() != n_sweeps {
                return Err(PyValueError::new_err(format!("background_uv arrays must be 1D length {n_sweeps}")));
            }
            return Ok(arr.iter().copied().collect());
        }
        Err(PyValueError::new_err("background_uv elements must be scalars or 1D arrays"))
    };

    Ok(Some((parse_component(u_item)?, parse_component(v_item)?)))
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
        return Err(PyValueError::new_err(format!("observed_volume must be 3D, got {}D", observed.ndim())));
    }
    let observed: ArrayView3<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed_volume must be 3D"))?;
    let nyq = resolve_volume_nyquist(nyquist, observed.shape()[0])?;
    let azimuth = azimuth_deg.as_array();
    let elevation = elevation_deg.as_array();
    if azimuth.ndim() != 1 || elevation.ndim() != 1 {
        return Err(PyValueError::new_err("azimuth_deg and elevation_deg must be 1D"));
    }
    let azimuth: ArrayView1<'_, f64> = azimuth.into_dimensionality().map_err(|_| PyValueError::new_err("azimuth_deg must be 1D"))?;
    let elevation: ArrayView1<'_, f64> = elevation.into_dimensionality().map_err(|_| PyValueError::new_err("elevation_deg must be 1D"))?;
    let previous_view = previous_volume
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix3>().map_err(|_| PyValueError::new_err("previous_volume must be 3D")))
        .transpose()?;
    let background = resolve_background_uv(background_uv, observed.shape()[0])?;
    let (background_u, background_v) = match background {
        Some((u, v)) => (Some(u), Some(v)),
        None => (None, None),
    };
    let result = core::dealias_volume_jh01(
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
    .map_err(map_error)?;
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
        item.set_item("resolved_fraction", result.per_sweep_resolved_fraction[sweep])?;
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
        return Err(PyValueError::new_err("observed and target_velocity must be 2D"));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let target: ArrayView2<'_, f64> = target.into_dimensionality().map_err(|_| PyValueError::new_err("target_velocity must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let azimuth_view = azimuth_deg
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix1>().map_err(|_| PyValueError::new_err("azimuth_deg must be 1D")))
        .transpose()?;
    let result = core::fit_ml_reference_model(observed, target, nyquist, reference_view, azimuth_view, ridge).map_err(map_error)?;
    Ok((
        ndarray::Array1::from(result.weights.clone()).into_dyn().into_pyarray(py),
        result.feature_names,
        result.ridge,
        result.train_rmse,
        result.mode,
        result.nyquist,
    ))
}

fn extract_ml_model_state(model: Option<&Bound<'_, PyAny>>) -> PyResult<Option<core::MlModelState>> {
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
        return Err(PyValueError::new_err("model.weights must be a 1D float array"));
    };
    let feature_names = model.getattr("feature_names")?.extract::<Vec<String>>()?;
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
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let model_state = extract_ml_model_state(model)?;
    let training_target_view = training_target
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("training_target must be 2D")))
        .transpose()?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let azimuth_view = azimuth_deg
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix1>().map_err(|_| PyValueError::new_err("azimuth_deg must be 1D")))
        .transpose()?;
    let result = core::dealias_sweep_ml(
        observed,
        nyquist,
        model_state.as_ref(),
        training_target_view,
        reference_view,
        azimuth_view,
        ridge,
        refine_with_variational,
    )
    .map_err(map_error)?;
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
    let reference_array = reference.as_ref().map(|value| value.as_array());
    let result = core::dealias_dual_prf(
        low_observed.as_array(),
        high_observed.as_array(),
        low_nyquist,
        high_nyquist,
        reference_array,
        max_abs_fold,
    )
    .map_err(map_error)?;
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

#[pyfunction]
#[pyo3(signature = (observed, nyquist, reference=None, block_shape=None, reference_weight=0.75, max_iterations=6, max_abs_fold=8, wrap_azimuth=true))]
fn dealias_sweep_region_graph<'py>(
    py: Python<'py>,
    observed: PyReadonlyArrayDyn<'py, f64>,
    nyquist: f64,
    reference: Option<PyReadonlyArrayDyn<'py, f64>>,
    block_shape: Option<(usize, usize)>,
    reference_weight: f64,
    max_iterations: usize,
    max_abs_fold: i16,
    wrap_azimuth: bool,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<i16>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyDict>,
)> {
    let observed = observed.as_array();
    if observed.ndim() != 2 {
        return Err(PyValueError::new_err(format!("observed must be 2D, got {}D", observed.ndim())));
    }
    let observed: ArrayView2<'_, f64> = observed.into_dimensionality().map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix2>().map_err(|_| PyValueError::new_err("reference must be 2D")))
        .transpose()?;
    let result = core::dealias_sweep_region_graph(
        observed,
        nyquist,
        reference_view,
        block_shape,
        reference_weight,
        max_iterations,
        max_abs_fold,
        wrap_azimuth,
    )
    .map_err(map_error)?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "PyARTRegionGraphLite")?;
    metadata.set_item("method", "region_graph_sweep")?;
    metadata.set_item("region_count", result.region_count)?;
    metadata.set_item("merge_iterations", result.merge_iterations)?;
    if result.region_count > 0 {
        metadata.set_item("assigned_regions", result.assigned_regions)?;
        metadata.set_item("seed_region", result.seed_region)?;
        metadata.set_item("block_shape", vec![result.block_shape.0, result.block_shape.1])?;
        metadata.set_item("wrap_azimuth", result.wrap_azimuth)?;
        metadata.set_item("average_fold", result.average_fold)?;
        metadata.set_item("regions_with_reference", result.regions_with_reference)?;
        metadata.set_item("block_grid_shape", vec![result.block_grid_shape.0, result.block_grid_shape.1])?;
    }
    Ok((
        result.velocity.into_pyarray(py),
        result.folds.into_pyarray(py),
        result.confidence.into_pyarray(py),
        result.reference.into_pyarray(py),
        metadata,
    ))
}

fn resolve_volume_nyquist<'py>(nyquist: &Bound<'py, PyAny>, n_sweeps: usize) -> PyResult<Vec<f64>> {
    if let Ok(value) = nyquist.extract::<f64>() {
        if value <= 0.0 {
            return Err(PyValueError::new_err(format!("nyquist must be positive, got {value}")));
        }
        return Ok(vec![value; n_sweeps]);
    }
    if let Ok(values) = nyquist.extract::<Vec<f64>>() {
        if values.len() == 1 {
            let value = values[0];
            if value <= 0.0 {
                return Err(PyValueError::new_err(format!("nyquist must be positive, got {value}")));
            }
            return Ok(vec![value; n_sweeps]);
        }
        if values.len() != n_sweeps {
            return Err(PyValueError::new_err(format!("nyquist must have length {n_sweeps}, got {}", values.len())));
        }
        if let Some(value) = values.iter().copied().find(|value| !value.is_finite() || *value <= 0.0) {
            return Err(PyValueError::new_err(format!("nyquist values must be positive, got {value}")));
        }
        return Ok(values);
    }
    if let Ok(values) = nyquist.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        let arr = values.as_array();
        if arr.ndim() != 1 {
            return Err(PyValueError::new_err(format!("nyquist must be scalar or 1D, got {}D", arr.ndim())));
        }
        let nyq: Vec<f64> = arr.iter().copied().collect();
        if nyq.len() != n_sweeps {
            return Err(PyValueError::new_err(format!("nyquist must have length {n_sweeps}, got {}", nyq.len())));
        }
        if let Some(value) = nyq.iter().copied().find(|value| !value.is_finite() || *value <= 0.0) {
            return Err(PyValueError::new_err(format!("nyquist values must be positive, got {value}")));
        }
        return Ok(nyq);
    }
    Err(PyValueError::new_err("nyquist must be a positive scalar or 1D array"))
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
        return Err(PyValueError::new_err(format!("observed_volume must be 3D, got {}D", observed.ndim())));
    }
    let observed: ArrayView3<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed_volume must be 3D"))?;

    let reference_view = reference_volume
        .as_ref()
        .map(|value| value.as_array().into_dimensionality::<ndarray::Ix3>().map_err(|_| PyValueError::new_err("reference_volume must be 3D")))
        .transpose()?;

    let nyq = resolve_volume_nyquist(nyquist, observed.shape()[0])?;
    let result = core::dealias_volume_3d(observed, &nyq, reference_view, wrap_azimuth, max_iterations).map_err(map_error)?;
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
        item.set_item("resolved_fraction", if valid > 0 { assigned as f64 / valid as f64 } else { 0.0 })?;
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

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wrap_to_nyquist, m)?)?;
    m.add_function(wrap_pyfunction!(fold_counts, m)?)?;
    m.add_function(wrap_pyfunction!(unfold_to_reference, m)?)?;
    m.add_function(wrap_pyfunction!(shift2d, m)?)?;
    m.add_function(wrap_pyfunction!(shift3d, m)?)?;
    m.add_function(wrap_pyfunction!(neighbor_stack, m)?)?;
    m.add_function(wrap_pyfunction!(texture_3x3, m)?)?;
    m.add_function(wrap_pyfunction!(build_velocity_qc_mask, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_radial_es90, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_es90, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_zw06, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_variational_refine, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_uniform_wind_vad, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_xu11, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_jh01, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_volume_jh01, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ml_reference_model, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_ml, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_dual_prf, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_region_graph, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_recursive, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_volume_3d, m)?)?;
    Ok(())
}
