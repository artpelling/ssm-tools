#!/usr/bin/env python3

from pyfar import Signal
from pyfar.classes.filter import StateSpaceModel
from numba import jit
import numpy as np
import scipy.linalg as spla
from time import perf_counter
import pyfar as pf


def solver(sys, signal, solver="basic"):
    out = np.zeros((sys.n_outputs, signal.n_samples), sys.dtype, order="F")
    sig = np.asfortranarray(signal.time)
    if solver == "basic":
        basic_solver(out, sys.state, sys._A, sys._B, sys._C, sys._D, sig)
    elif solver == "blas":
        blas_solver(out, sys.state, sys._A, sys._B, sys._C, sys._D, sig)
    return out


@jit(nopython=True, cache=True)
def basic_solver(out, x, A, B, C, D, sig):
    for i in range(out.shape[1]):
        out[:, i] = C @ x + D @ sig[:, i]
        x = A @ x + B @ sig[:, i]


def blas_solver(out, x, A, B, C, D, sig):
    gemv = spla.get_blas_funcs("gemv", arrays=(A, B, C, D))
    for i in range(out.shape[1]):
        # out[:, i] =
        out[:, i] = gemv(1.0, C, x, beta=1, y=gemv(1.0, D, sig[:, i], y=out[:, i]))
        # x =
        x = gemv(1.0, B, sig[:, i], beta=1, y=gemv(1.0, A, x, y=x))


if __name__ == "__main__":
    n, m, p = 1000, 12, 10
    fs, N = 1, 2048
    rng = np.random.default_rng(0)
    A, B, C, D = 0.8 * np.eye(n), rng.normal(size=(n, m)), rng.normal(size=(p, n)), None
    sys = StateSpaceModel(A, B, C, D, sampling_rate=fs, dtype=np.float64)
    sig = Signal(rng.normal(size=(m, N)), sampling_rate=fs)
    print(sig.time.flags)

    print("Convolution reference:")
    tic = perf_counter()
    conv = pf.dsp.convolve(sig.reshape((m, 1)), sys.impulse_response(sig.n_samples))
    conv = pf.Signal(conv.time[..., :N].sum(axis=0), sampling_rate=fs)
    print(f"Elapsed time: {perf_counter() - tic}")

    sys.init_state()
    print("\nReference solver:")
    tic = perf_counter()
    ref = sys.process(sig)
    print(f"Elapsed time: {perf_counter() - tic}")
    print(spla.norm(conv.time - ref.time))

    print("\nBasic solver:")
    solver(sys, sig)
    sys.init_state()
    tic = perf_counter()
    out1 = solver(sys, sig, solver="basic")
    print(f"Elapsed time: {perf_counter() - tic}")
    print(spla.norm(conv.time - out1))
