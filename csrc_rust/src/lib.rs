use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence, PyTuple};
use pyo3::PyTryFrom;

fn torch_min<'py>(py: Python<'py>, tensor: &'py PyAny) -> PyResult<&'py PyAny> {
    tensor.call_method0("min").or_else(|_| {
        let torch = py.import("torch")?;
        torch.call_method1("min", (tensor,))
    })
}

fn torch_max<'py>(py: Python<'py>, tensor: &'py PyAny) -> PyResult<&'py PyAny> {
    tensor.call_method0("max").or_else(|_| {
        let torch = py.import("torch")?;
        torch.call_method1("max", (tensor,))
    })
}

fn torch_zeros_like<'py>(py: Python<'py>, tensor: &'py PyAny) -> PyResult<&'py PyAny> {
    let torch = py.import("torch")?;
    torch.call_method1("zeros_like", (tensor,))
}

fn tensor_truthy(value: &PyAny) -> PyResult<bool> {
    value.call_method0("item")?.is_true()
}

fn normalize_tensor(py: Python<'_>, score: &PyAny) -> PyResult<PyObject> {
    let min_score = torch_min(py, score)?;
    let max_score = torch_max(py, score)?;
    let denom = max_score.call_method1("__sub__", (min_score,))?;
    let is_zero = denom.call_method1("__eq__", (0.0,))?;

    if tensor_truthy(is_zero)? {
        return Ok(torch_zeros_like(py, score)?.into_py(py));
    }

    let shifted = score.call_method1("__sub__", (min_score,))?;
    Ok(shifted.call_method1("__truediv__", (denom,))?.into_py(py))
}

#[pyfunction]
fn normalize_score(py: Python<'_>, score: &PyAny) -> PyResult<PyObject> {
    normalize_tensor(py, score)
}

#[pyfunction]
fn average_counts(py: Python<'_>, counts: &PyAny, num_views: usize) -> PyResult<PyObject> {
    if num_views == 0 {
        return Err(PyValueError::new_err("num_views must be greater than zero"));
    }

    let torch = py.import("torch")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("rounding_mode", "floor")?;
    Ok(torch
        .call_method("div", (counts, num_views), Some(kwargs))?
        .into_py(py))
}

#[pyfunction]
fn compute_scores_from_accumulators(
    py: Python<'_>,
    accum_metric_counts: &PyAny,
    photometric_losses: &PyAny,
    densify: bool,
) -> PyResult<PyObject> {
    let counts_seq = <PySequence as PyTryFrom>::try_from(accum_metric_counts)?;
    let losses_seq = <PySequence as PyTryFrom>::try_from(photometric_losses)?;
    let num_views = counts_seq.len()?;

    if num_views == 0 {
        return Err(PyValueError::new_err(
            "accum_metric_counts must contain at least one tensor",
        ));
    }
    if losses_seq.len()? != num_views {
        return Err(PyValueError::new_err(
            "accum_metric_counts and photometric_losses must have the same length",
        ));
    }

    let mut full_metric_counts: Option<PyObject> = None;
    let mut full_metric_score: Option<PyObject> = None;

    for index in 0..num_views {
        let counts = counts_seq.get_item(index)?;
        let loss = losses_seq.get_item(index)?;

        if densify {
            full_metric_counts = Some(match full_metric_counts {
                Some(current) => current
                    .as_ref(py)
                    .call_method1("__add__", (counts,))?
                    .into_py(py),
                None => counts.call_method0("clone")?.into_py(py),
            });
        }

        let weighted = loss.call_method1("__mul__", (counts.call_method0("clone")?,))?;
        full_metric_score = Some(match full_metric_score {
            Some(current) => current
                .as_ref(py)
                .call_method1("__add__", (weighted,))?
                .into_py(py),
            None => weighted.into_py(py),
        });
    }

    let score = full_metric_score.ok_or_else(|| PyTypeError::new_err("missing score tensor"))?;
    let pruning_score = normalize_tensor(py, score.as_ref(py))?;
    let importance_score = if densify {
        let counts =
            full_metric_counts.ok_or_else(|| PyTypeError::new_err("missing count tensor"))?;
        average_counts(py, counts.as_ref(py), num_views)?
    } else {
        py.None()
    };

    Ok(PyTuple::new(py, [importance_score, pruning_score]).into_py(py))
}

#[pyfunction]
fn extension_info(py: Python<'_>) -> PyResult<PyObject> {
    let info = PyDict::new(py);
    info.set_item("name", "fastgs_rust")?;
    info.set_item("backend", "pyo3-python-tensor-ops")?;
    info.set_item("libtorch_linked", false)?;
    Ok(info.into_py(py))
}

#[pymodule]
fn fastgs_rust(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_score, module)?)?;
    module.add_function(wrap_pyfunction!(average_counts, module)?)?;
    module.add_function(wrap_pyfunction!(compute_scores_from_accumulators, module)?)?;
    module.add_function(wrap_pyfunction!(extension_info, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__backend__", "pyo3-python-tensor-ops")?;

    let _ = py;
    Ok(())
}
