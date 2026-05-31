use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use ndarray::{Array1, Array2, Array3};
use crate::{
    quantize_int8_simd, dequantize_int8_simd, matmul_optimized, softmax_fast,
    layer_norm_fast, attention_parallel, relu_fast, gelu_fast, gelu_fast_batch,
    dot_product_parallel, cosine_similarity_batch, variance_fast, histogram_parallel,
};

/// Python module for native performance-critical operations
#[pymodule]
fn sdx_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_quantize_int8, m)?)?;
    m.add_function(wrap_pyfunction!(py_dequantize_int8, m)?)?;
    m.add_function(wrap_pyfunction!(py_softmax_fast, m)?)?;
    m.add_function(wrap_pyfunction!(py_layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(py_relu, m)?)?;
    m.add_function(wrap_pyfunction!(py_gelu_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(py_variance, m)?)?;
    m.add_function(wrap_pyfunction!(py_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(py_matmul, m)?)?;
    Ok(())
}

#[pyfunction]
fn py_quantize_int8(data: PyReadonlyArray1<f32>, scale: f32, py: Python) -> PyResult<PyObject> {
    let array = data.as_array();
    let result = quantize_int8_simd(array.to_slice().unwrap(), scale);
    Ok(result.into_py(py))
}

#[pyfunction]
fn py_dequantize_int8(data: PyReadonlyArray1<i8>, scale: f32, py: Python) -> PyResult<PyObject> {
    let array = data.as_array();
    let result = dequantize_int8_simd(array.to_slice().unwrap(), scale);
    Ok(result.into_py(py))
}

#[pyfunction]
fn py_softmax_fast(data: PyReadonlyArray1<f32>, py: Python) -> PyResult<PyObject> {
    let array = data.as_array();
    let result = softmax_fast(array.to_slice().unwrap());
    Ok(result.into_py(py))
}

#[pyfunction]
fn py_layer_norm(
    data: PyReadonlyArray1<f32>,
    gamma: PyReadonlyArray1<f32>,
    beta: PyReadonlyArray1<f32>,
    eps: f32,
) -> PyResult<Vec<f32>> {
    let d = data.as_array().to_vec();
    let g = gamma.as_array().to_vec();
    let b = beta.as_array().to_vec();
    Ok(layer_norm_fast(&d, eps, &g, &b))
}

#[pyfunction]
fn py_relu(mut data: PyReadonlyArray1<f32>) -> PyResult<()> {
    let mut array = data.as_array_mut().to_owned();
    relu_fast(&mut array);
    Ok(())
}

#[pyfunction]
fn py_gelu_batch(data: PyReadonlyArray1<f32>, py: Python) -> PyResult<PyObject> {
    let array = data.as_array();
    let result = gelu_fast_batch(array.to_slice().unwrap());
    Ok(result.into_py(py))
}

#[pyfunction]
fn py_dot_product(a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let a_arr = a.as_array().to_vec();
    let b_arr = b.as_array().to_vec();
    Ok(dot_product_parallel(&a_arr, &b_arr))
}

#[pyfunction]
fn py_variance(data: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let array = data.as_array();
    Ok(variance_fast(array.to_slice().unwrap()))
}

#[pyfunction]
fn py_histogram(data: PyReadonlyArray1<f32>, bins: usize, py: Python) -> PyResult<PyObject> {
    let array = data.as_array();
    let result = histogram_parallel(array.to_slice().unwrap(), bins);
    Ok(result.into_py(py))
}

#[pyfunction]
fn py_matmul(a: PyReadonlyArray2<f32>, b: PyReadonlyArray2<f32>, py: Python) -> PyResult<PyObject> {
    let a_arr = a.as_array().to_owned();
    let b_arr = b.as_array().to_owned();
    let result = matmul_optimized(&a_arr, &b_arr);
    Ok(result.into_py(py))
}
