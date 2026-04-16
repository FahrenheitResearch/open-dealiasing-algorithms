mod common;
mod core_ops;
mod ml;
mod qc;
mod sweep;
mod temporal;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    core_ops::register(m)?;
    qc::register(m)?;
    sweep::register(m)?;
    temporal::register(m)?;
    ml::register(m)?;
    Ok(())
}
