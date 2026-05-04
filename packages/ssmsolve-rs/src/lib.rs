use numpy::ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

// CBLAS order and transpose constants.
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;

// Raw CBLAS bindings — symbols resolved at link time by the backend
// discovered in build.rs (system OpenBLAS, conda, or macOS Accelerate).
#[allow(non_snake_case)]
unsafe extern "C" {
    /// y = alpha * A @ x + beta * y
    fn cblas_sgemv(
        order: i32, trans: i32, m: i32, n: i32,
        alpha: f32, a: *const f32, lda: i32,
        x: *const f32, incx: i32,
        beta: f32, y: *mut f32, incy: i32,
    );
    /// y = alpha * A @ x + beta * y  (double precision)
    fn cblas_dgemv(
        order: i32, trans: i32, m: i32, n: i32,
        alpha: f64, a: *const f64, lda: i32,
        x: *const f64, incx: i32,
        beta: f64, y: *mut f64, incy: i32,
    );
}

/// Compute `y = alpha * A @ x + beta * y` via BLAS SGEMV.
///
/// Handles both Fortran-order (column-major) and C-order (row-major) `A`
/// by selecting the matching CBLAS order flag and leading dimension.
#[inline]
unsafe fn sgemv(
    a: &ArrayView2<f32>,
    x: &ArrayView1<f32>,
    y: &mut ArrayViewMut1<f32>,
    alpha: f32,
    beta: f32,
) {
    let m = a.shape()[0] as i32;
    let n = a.shape()[1] as i32;
    // ndarray strides are in element units.
    // F-order: strides = [1, m]  → row-stride == 1  → column-major.
    // C-order: strides = [n, 1]  → col-stride == 1  → row-major.
    let (order, lda) = if a.strides()[0] == 1 {
        (CBLAS_COL_MAJOR, a.strides()[1] as i32) // lda = col stride (= m)
    } else {
        (CBLAS_ROW_MAJOR, a.strides()[0] as i32) // lda = row stride (= n)
    };
    // incx/incy must match the actual element strides of x and y.
    let incx = x.strides()[0] as i32;
    let incy = y.strides()[0] as i32;
    unsafe { cblas_sgemv(order, CBLAS_NO_TRANS, m, n, alpha, a.as_ptr(), lda, x.as_ptr(), incx, beta, y.as_mut_ptr(), incy) };
}
/// Compute `y = alpha * A @ x + beta * y` via BLAS DGEMV.
#[inline]
unsafe fn dgemv(
    a: &ArrayView2<f64>,
    x: &ArrayView1<f64>,
    y: &mut ArrayViewMut1<f64>,
    alpha: f64,
    beta: f64,
) {
    let m = a.shape()[0] as i32;
    let n = a.shape()[1] as i32;
    let (order, lda) = if a.strides()[0] == 1 {
        (CBLAS_COL_MAJOR, a.strides()[1] as i32)
    } else {
        (CBLAS_ROW_MAJOR, a.strides()[0] as i32)
    };
    let incx = x.strides()[0] as i32;
    let incy = y.strides()[0] as i32;
    unsafe { cblas_dgemv(order, CBLAS_NO_TRANS, m, n, alpha, a.as_ptr(), lda, x.as_ptr(), incx, beta, y.as_mut_ptr(), incy) };
}

/// Run the discrete-time state-space recursion for `f32` arrays.
///
/// For each sample index `i`:
///   out[:, i] = C @ x + D @ sig[:, i]
///   x         = A @ x + B @ sig[:, i]
///
/// `x` (the system state) is updated **in place** so the caller retains the
/// final state after the call returns.
fn solve_f32_inner(
    mut out: ArrayViewMut2<f32>,
    mut x: ArrayViewMut1<f32>,
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    c: ArrayView2<f32>,
    d: ArrayView2<f32>,
    sig: ArrayView2<f32>,
) {
    let n_samples = sig.shape()[1];
    let n_states = x.len();
    let n_outputs = out.shape()[0];

    let mut x_cur: Array1<f32> = x.to_owned();
    let mut x_nxt: Array1<f32> = Array1::zeros(n_states);
    let mut y_buf: Array1<f32> = Array1::zeros(n_outputs);

    for i in 0..n_samples {
        let sig_i = sig.column(i);

        // y_buf = C @ x_cur + D @ sig_i
        unsafe {
            sgemv(&c, &x_cur.view(), &mut y_buf.view_mut(), 1.0, 0.0);
            sgemv(&d, &sig_i, &mut y_buf.view_mut(), 1.0, 1.0);
        }
        out.column_mut(i).assign(&y_buf);

        // x_nxt = A @ x_cur + B @ sig_i
        unsafe {
            sgemv(&a, &x_cur.view(), &mut x_nxt.view_mut(), 1.0, 0.0);
            sgemv(&b, &sig_i, &mut x_nxt.view_mut(), 1.0, 1.0);
        }

        std::mem::swap(&mut x_cur, &mut x_nxt);
    }

    x.assign(&x_cur);
}

/// Run the discrete-time state-space recursion for `f64` arrays.
fn solve_f64_inner(
    mut out: ArrayViewMut2<f64>,
    mut x: ArrayViewMut1<f64>,
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    c: ArrayView2<f64>,
    d: ArrayView2<f64>,
    sig: ArrayView2<f64>,
) {
    let n_samples = sig.shape()[1];
    let n_states = x.len();
    let n_outputs = out.shape()[0];

    let mut x_cur: Array1<f64> = x.to_owned();
    let mut x_nxt: Array1<f64> = Array1::zeros(n_states);
    let mut y_buf: Array1<f64> = Array1::zeros(n_outputs);

    for i in 0..n_samples {
        let sig_i = sig.column(i);

        // y_buf = C @ x_cur + D @ sig_i
        unsafe {
            dgemv(&c, &x_cur.view(), &mut y_buf.view_mut(), 1.0, 0.0);
            dgemv(&d, &sig_i, &mut y_buf.view_mut(), 1.0, 1.0);
        }
        out.column_mut(i).assign(&y_buf);

        // x_nxt = A @ x_cur + B @ sig_i
        unsafe {
            dgemv(&a, &x_cur.view(), &mut x_nxt.view_mut(), 1.0, 0.0);
            dgemv(&b, &sig_i, &mut x_nxt.view_mut(), 1.0, 1.0);
        }

        std::mem::swap(&mut x_cur, &mut x_nxt);
    }

    x.assign(&x_cur);
}

/// Python-callable solver for `float32` state-space systems.
///
/// Parameters
/// ----------
/// out : numpy.ndarray, shape (p, T), float32, writable
///     Output array filled in place.
/// x : numpy.ndarray, shape (n,), float32, writable
///     System state vector; updated in place to the state after the last sample.
/// a, b, c, d : numpy.ndarray, float32
///     System matrices (shapes (n,n), (n,m), (p,n), (p,m)).
/// sig : numpy.ndarray, shape (m, T), float32
///     Input signal.
#[pyfunction]
fn solve_f32<'py>(
    mut out: PyReadwriteArray2<'py, f32>,
    mut x: PyReadwriteArray1<'py, f32>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
    c: PyReadonlyArray2<'py, f32>,
    d: PyReadonlyArray2<'py, f32>,
    sig: PyReadonlyArray2<'py, f32>,
) -> PyResult<()> {
    solve_f32_inner(
        out.as_array_mut(),
        x.as_array_mut(),
        a.as_array(),
        b.as_array(),
        c.as_array(),
        d.as_array(),
        sig.as_array(),
    );
    Ok(())
}

/// Python-callable solver for `float64` state-space systems.
///
/// Parameters
/// ----------
/// out : numpy.ndarray, shape (p, T), float64, writable
///     Output array filled in place.
/// x : numpy.ndarray, shape (n,), float64, writable
///     System state vector; updated in place to the state after the last sample.
/// a, b, c, d : numpy.ndarray, float64
///     System matrices (shapes (n,n), (n,m), (p,n), (p,m)).
/// sig : numpy.ndarray, shape (m, T), float64
///     Input signal.
#[pyfunction]
fn solve_f64<'py>(
    mut out: PyReadwriteArray2<'py, f64>,
    mut x: PyReadwriteArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    sig: PyReadonlyArray2<'py, f64>,
) -> PyResult<()> {
    solve_f64_inner(
        out.as_array_mut(),
        x.as_array_mut(),
        a.as_array(),
        b.as_array(),
        c.as_array(),
        d.as_array(),
        sig.as_array(),
    );
    Ok(())
}

#[pymodule]
fn ssmsolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_f32, m)?)?;
    m.add_function(wrap_pyfunction!(solve_f64, m)?)?;
    Ok(())
}

