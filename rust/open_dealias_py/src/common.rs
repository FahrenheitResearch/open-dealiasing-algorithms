use open_dealias_core as core;
use pyo3::exceptions::PyValueError;
use pyo3::marker::Ungil;
use pyo3::prelude::*;

pub(crate) fn map_error(err: core::DealiasError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(crate) fn run_without_gil<T, F>(py: Python<'_>, op: F) -> PyResult<T>
where
    T: Ungil + Send,
    F: Ungil + FnOnce() -> core::Result<T>,
{
    py.allow_threads(op).map_err(map_error)
}
