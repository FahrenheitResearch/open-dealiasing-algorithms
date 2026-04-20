use crate::common::{map_error, run_without_gil};
use ndarray::{ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

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
        return Err(PyValueError::new_err(format!(
            "observed must be 1D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView1<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 1D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| PyValueError::new_err("reference must be 1D"))
        })
        .transpose()?;
    let result = core::dealias_radial_es90(
        observed,
        nyquist,
        reference_view,
        seed_index,
        max_gap,
        max_abs_step,
    )
    .map_err(map_error)?;
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
        return Err(PyValueError::new_err(format!(
            "observed must be 2D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView2<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let result = core::dealias_sweep_es90(observed, nyquist, reference_view, max_gap, max_abs_step)
        .map_err(map_error)?;
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
        return Err(PyValueError::new_err(format!(
            "observed must be 2D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView2<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_zw06(
            observed,
            nyquist,
            reference_view,
            weak_threshold_fraction,
            wrap_azimuth,
            max_iterations_per_pass,
            include_diagonals,
            recenter_without_reference,
        )
    })?;
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
        return Err(PyValueError::new_err(
            "observed and initial_corrected must be 2D",
        ));
    }
    let observed: ArrayView2<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let initial_corrected: ArrayView2<'_, f64> = initial_corrected
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("initial_corrected must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_variational_refine(
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
    })?;
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
        return Err(PyValueError::new_err(format!(
            "observed must be 2D, got {}D",
            observed.ndim()
        )));
    }
    let observed: ArrayView2<'_, f64> = observed
        .into_dimensionality()
        .map_err(|_| PyValueError::new_err("observed must be 2D"))?;
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_recursive(
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
    })?;
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
#[pyo3(signature = (observed, nyquist, reference=None, block_shape=None, reference_weight=0.75, max_iterations=6, max_abs_fold=8, wrap_azimuth=true, min_region_area=4, min_valid_fraction=0.15))]
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
    min_region_area: usize,
    min_valid_fraction: f64,
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
    let reference_view = reference
        .as_ref()
        .map(|value| {
            value
                .as_array()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| PyValueError::new_err("reference must be 2D"))
        })
        .transpose()?;
    let result = run_without_gil(py, move || {
        core::dealias_sweep_region_graph(
            observed,
            nyquist,
            reference_view,
            block_shape,
            reference_weight,
            max_iterations,
            max_abs_fold,
            wrap_azimuth,
            min_region_area,
            min_valid_fraction,
        )
    })?;
    let metadata = PyDict::new(py);
    metadata.set_item("paper_family", "PyARTRegionGraphLite")?;
    metadata.set_item("method", result.method)?;
    metadata.set_item("region_count", result.region_count)?;
    metadata.set_item("merge_iterations", result.merge_iterations)?;
    metadata.set_item("min_region_area", result.min_region_area)?;
    metadata.set_item("min_valid_fraction", result.min_valid_fraction)?;
    metadata.set_item("skipped_sparse_blocks", result.skipped_sparse_blocks)?;
    metadata.set_item("safety_fallback_applied", result.safety_fallback_applied)?;
    metadata.set_item("safety_fallback_reason", result.safety_fallback_reason.clone())?;
    metadata.set_item("candidate_cost", result.candidate_cost)?;
    metadata.set_item("fallback_cost", result.fallback_cost)?;
    metadata.set_item("disagreement_fraction", result.disagreement_fraction)?;
    metadata.set_item(
        "largest_disagreement_component",
        result.largest_disagreement_component,
    )?;
    if result.region_count > 0 {
        metadata.set_item("assigned_regions", result.assigned_regions)?;
        metadata.set_item("seed_region", result.seed_region)?;
        metadata.set_item(
            "block_shape",
            vec![result.block_shape.0, result.block_shape.1],
        )?;
        metadata.set_item("wrap_azimuth", result.wrap_azimuth)?;
        metadata.set_item("average_fold", result.average_fold)?;
        metadata.set_item("regions_with_reference", result.regions_with_reference)?;
        metadata.set_item(
            "block_grid_shape",
            vec![result.block_grid_shape.0, result.block_grid_shape.1],
        )?;
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
    m.add_function(wrap_pyfunction!(dealias_radial_es90, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_es90, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_zw06, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_variational_refine, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_recursive, m)?)?;
    m.add_function(wrap_pyfunction!(dealias_sweep_region_graph, m)?)?;
    Ok(())
}
