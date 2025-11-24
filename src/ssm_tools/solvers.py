#!/usr/bin/env python3

from pyfar import Signal
from pyfar.classes.filter import StateSpaceModel
from numba import jit
import numpy as np
import scipy.linalg as spla
from time import perf_counter
import pyfar as pf


def solver(sys, signal):
    out = np.zeros((sys.n_outputs, signal.n_samples), sys.dtype)
    print(sys.state.shape)
    basic_solver(out, sys.state, sys._A, sys._B, sys._C, sys._D, signal.time)
    return out


@jit(nopython=True, cache=False)
def basic_solver(out, x, A, B, C, D, sig):
    for i in range(out.shape[1]):
        out[:, i] = C @ x + D @ sig[:, i]
        x = A @ x + B @ sig[:, i]


if __name__ == "__main__":
    n, m, p = 20, 3, 2
    fs, N = 1, 1000
    A, B, C, D = 0.8 * np.eye(n), np.random.randn(n, m), np.random.randn(p, n), None
    sys = StateSpaceModel(A, B, C, D, sampling_rate=fs, dtype=np.float64)
    sig = Signal(np.random.randn(m, N), sampling_rate=fs)

    conv = pf.dsp.convolve(sig.reshape((m, 1)), sys.impulse_response(sig.n_samples))
    conv = pf.Signal(conv.time[..., :N].sum(axis=0), sampling_rate=fs)

    sys.init_state()
    tic = perf_counter()
    ref = sys.process(sig)
    print(f"Elapsed time: {perf_counter() - tic}")
    solver(sys, sig)

    sys.init_state()
    tic = perf_counter()
    out1 = solver(sys, sig)
    print(f"Elapsed time: {perf_counter() - tic}")
    print(spla.norm(ref.time - out1))
