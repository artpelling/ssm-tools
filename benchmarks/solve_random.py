# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from ssm_tools.solvers import solver
from pyfar.classes.filter import StateSpaceModel
from pyfar import Signal
import numpy as np
import pyfar as pf
import scipy.linalg as spla
from itertools import product


rng = np.random.default_rng(0)

M = (1, 8)
P = (1, 8)
N = (10, 100)
T = 512
DTYPE = (np.float32, np.float64)

DATA = dict()
for m, p, n, dtype in product(M, P, N, DTYPE):
    A = 0.8 * np.eye(n)
    B = rng.random(size=(n, m), dtype=dtype)
    C = rng.random(size=(p, n), dtype=dtype)
    D = None
    sys = StateSpaceModel(A, B, C, D, sampling_rate=1, dtype=dtype)
    sig = Signal(rng.normal(size=(m, T)).astype(dtype), sampling_rate=1)
    conv = pf.dsp.convolve(sig.reshape((m, 1)), sys.impulse_response(sig.n_samples))
    ref = conv.time[..., :T].sum(axis=0)
    sys.init_state()
    DATA[f"{m}-{p}-{n}-{dtype}"] = (sys, sig, ref)


class _SolveRandom:
    params = (M, P, N, DTYPE)
    param_names = ["m", "p", "n", "dtype"]

    def setup(self, m, p, n, dtype):
        self.sys, self.sig, self.ref = DATA[f"{m}-{p}-{n}-{dtype}"]


class _SolveRandomNumba(_SolveRandom):
    def setup(self, m, p, n, dtype):
        super().setup(m, p, n, dtype)
        # Warm up numba JIT
        solver(self.sys, self.sig)

    def track_error(self, m, p, n, dtype):
        out = solver(self.sys, self.sig)
        return float(spla.norm(self.ref - out))


class SolveRandomPyfar(_SolveRandom):
    track_error_unit = "norm"

    def time_solver(self, *args):
        self.sys.process(self.sig).time

    def track_error(self, m, p, n, dtype):
        out = self.sys.process(self.sig).time
        return float(spla.norm(self.ref - out))


class SolveRandomNumbaBasic(_SolveRandomNumba):
    def time_solver(self, *args):
        solver(self.sys, self.sig)
