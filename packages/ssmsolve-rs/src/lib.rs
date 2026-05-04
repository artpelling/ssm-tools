use numpy::ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use numpy::{PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// BLAS bindings
// ---------------------------------------------------------------------------

// CBLAS — used for C-order (row-major) arrays.
#[allow(non_snake_case)]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32, trans: i32, m: i32, n: i32,
        alpha: f32, a: *const f32, lda: i32,
        x: *const f32, incx: i32,
        beta: f32, y: *mut f32, incy: i32,
    );
    fn cblas_dgemv(
        order: i32, trans: i32, m: i32, n: i32,
        alpha: f64, a: *const f64, lda: i32,
        x: *const f64, incx: i32,
        beta: f64, y: *mut f64, incy: i32,
    );
}

// Fortran BLAS — used for F-order (column-major) arrays; no layout translation.
unsafe extern "C" {
    fn sgemv_(
        trans: *const u8, m: *const i32, n: *const i32,
        alpha: *const f32, a: *const f32, lda: *const i32,
        x: *const f32, incx: *const i32,
        beta: *const f32, y: *mut f32, incy: *const i32,
    );
    fn dgemv_(
        trans: *const u8, m: *const i32, n: *const i32,
        alpha: *const f64, a: *const f64, lda: *const i32,
        x: *const f64, incx: *const i32,
        beta: *const f64, y: *mut f64, incy: *const i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

// ---------------------------------------------------------------------------
// BLAS wrappers — one per layout, no branching
// ---------------------------------------------------------------------------

/// `y = alpha * A @ x + beta * y`  — A is F-order (column-major).
#[inline]
unsafe fn sgemv_f(a: &ArrayView2<f32>, x: &ArrayView1<f32>, y: &mut ArrayViewMut1<f32>, alpha: f32, beta: f32) {
    let m = a.shape()[0] as i32;
    let n = a.shape()[1] as i32;
    let lda = a.strides()[1] as i32; // column stride = number of rows
    let incx = x.strides()[0] as i32;
    let incy = y.strides()[0] as i32;
    unsafe { sgemv_(b"N".as_ptr(), &m, &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta, y.as_mut_ptr(), &incy) };
}

/// `y = alpha * A @ x + beta * y`  — A is C-order (row-major).
#[inline]
unsafe fn sgemv_c(a: &ArrayView2<f32>, x: &ArrayView1<f32>, y: &mut ArrayViewMut1<f32>, alpha: f32, beta: f32) {
    let m = a.shape()[0] as i32;
    let n = a.shape()[1] as i32;
    let lda = a.strides()[0] as i32; // row stride = number of columns
    let incx = x.strides()[0] as i32;
    let incy = y.strides()[0] as i32;
    unsafe { cblas_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, m, n, alpha, a.as_ptr(), lda, x.as_ptr(), incx, beta, y.as_mut_ptr(), incy) };
}

/// `y = alpha * A @ x + beta * y`  — A is F-order (column-major).
#[inline]
unsafe fn dgemv_f(a: &ArrayView2<f64>, x: &ArrayView1<f64>, y: &mut ArrayViewMut1<f64>, alpha: f64, beta: f64) {
    let m = a.shape()[0] as i32;
    let n = a.shape()[1] as i32;
    let lda = a.strides()[1] as i32;
    let incx = x.strides()[0] as i32;
    let incy = y.strides()[0] as i32;
    unsafe { dgemv_(b"N".as_ptr(), &m, &n, &alpha, a.as_ptr(), &lda, x.as_ptr(), &incx, &beta, y.as_mut_ptr(), &incy) };
}

/// `y = alpha * A @ x + beta * y`  — A is C-order (row-major).
#[inline]
unsafe fn dgemv_c(a: &ArrayView2<f64>, x: &ArrayView1<f64>, y: &mut ArrayViewMut1<f64>, alpha: f64, beta: f64) {
    let m = a.shape()[0] as i32;
    let n = a.shape()[1] as i32;
    let lda = a.strides()[0] as i32;
    let incx = x.strides()[0] as i32;
    let incy = y.strides()[0] as i32;
    unsafe { cblas_dgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, m, n, alpha, a.as_ptr(), lda, x.as_ptr(), incx, beta, y.as_mut_ptr(), incy) };
}

// ---------------------------------------------------------------------------
// Inner solvers — templated by BLAS variant via function pointer
// ---------------------------------------------------------------------------

macro_rules! make_inner_solver {
    ($name:ident, $T:ty, $gemv:ident) => {
        fn $name(
            mut out: ArrayViewMut2<$T>,
            mut x:   ArrayViewMut1<$T>,
            a: ArrayView2<$T>,
            b: ArrayView2<$T>,
            c: ArrayView2<$T>,
            d: ArrayView2<$T>,
            sig: ArrayView2<$T>,
        ) {
            let n_samples = sig.shape()[1];
            let n_states  = x.len();
            let n_outputs = out.shape()[0];

            let mut x_cur: Array1<$T> = x.to_owned();
            let mut x_nxt: Array1<$T> = Array1::zeros(n_states);
            let mut y_buf: Array1<$T> = Array1::zeros(n_outputs);

            for i in 0..n_samples {
                let sig_i = sig.column(i);
                // y_buf = C @ x_cur + D @ sig_i
                unsafe {
                    $gemv(&c, &x_cur.view(), &mut y_buf.view_mut(), 1.0, 0.0);
                    $gemv(&d, &sig_i,        &mut y_buf.view_mut(), 1.0, 1.0);
                }
                out.column_mut(i).assign(&y_buf);
                // x_nxt = A @ x_cur + B @ sig_i
                unsafe {
                    $gemv(&a, &x_cur.view(), &mut x_nxt.view_mut(), 1.0, 0.0);
                    $gemv(&b, &sig_i,        &mut x_nxt.view_mut(), 1.0, 1.0);
                }
                std::mem::swap(&mut x_cur, &mut x_nxt);
            }
            x.assign(&x_cur);
        }
    };
}

make_inner_solver!(solve_f32_f_inner, f32, sgemv_f);
make_inner_solver!(solve_f32_c_inner, f32, sgemv_c);
make_inner_solver!(solve_f64_f_inner, f64, dgemv_f);
make_inner_solver!(solve_f64_c_inner, f64, dgemv_c);

// ---------------------------------------------------------------------------
// Python-callable functions — dispatch on array layout
// ---------------------------------------------------------------------------

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
    mut x:   PyReadwriteArray1<'py, f32>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
    c: PyReadonlyArray2<'py, f32>,
    d: PyReadonlyArray2<'py, f32>,
    sig: PyReadonlyArray2<'py, f32>,
) -> PyResult<()> {
    let a_arr = a.as_array();
    // Dispatch on A's layout; all system matrices share the same storage order.
    if a_arr.strides()[0] == 1 {
        solve_f32_f_inner(out.as_array_mut(), x.as_array_mut(), a_arr, b.as_array(), c.as_array(), d.as_array(), sig.as_array());
    } else {
        solve_f32_c_inner(out.as_array_mut(), x.as_array_mut(), a_arr, b.as_array(), c.as_array(), d.as_array(), sig.as_array());
    }
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
    mut x:   PyReadwriteArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray2<'py, f64>,
    d: PyReadonlyArray2<'py, f64>,
    sig: PyReadonlyArray2<'py, f64>,
) -> PyResult<()> {
    let a_arr = a.as_array();
    if a_arr.strides()[0] == 1 {
        solve_f64_f_inner(out.as_array_mut(), x.as_array_mut(), a_arr, b.as_array(), c.as_array(), d.as_array(), sig.as_array());
    } else {
        solve_f64_c_inner(out.as_array_mut(), x.as_array_mut(), a_arr, b.as_array(), c.as_array(), d.as_array(), sig.as_array());
    }
    Ok(())
}

#[pymodule]
fn ssmsolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_f32, m)?)?;
    m.add_function(wrap_pyfunction!(solve_f64, m)?)?;
    Ok(())
}
