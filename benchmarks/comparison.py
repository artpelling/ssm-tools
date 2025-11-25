# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from ssm_tools.models import (
    NumbaStateSpaceModel,
    TriangularStateSpaceModel,
    DiagonalStateSpaceModel,
)
from pyfar.classes.filter import StateSpaceModel
from pyfar import Signal
import numpy as np
import pyfar as pf
import scipy.linalg as spla

rng = np.random.default_rng(0)

DTYPE = (np.float64, np.float32)
n, m, p = 512, 5, 12
T = 4096

DATA = dict()
for dtype in DTYPE:
    A = 0.8 * np.eye(n)
    B = rng.random(size=(n, m), dtype=dtype)
    C = rng.random(size=(p, n), dtype=dtype)
    D = None
    sys = StateSpaceModel(A, B, C, D, sampling_rate=1, dtype=dtype)
    sig = Signal(rng.normal(size=(m, T)).astype(dtype), sampling_rate=1)
    conv = pf.dsp.convolve(sig.reshape((m, 1)), sys.impulse_response(sig.n_samples))
    ref = conv.time[..., :T].sum(axis=0)
    DATA[f"{dtype}"] = (sys, sig, ref)


class Compare:
    params = (
        (
            StateSpaceModel,
            NumbaStateSpaceModel,
            TriangularStateSpaceModel,
            DiagonalStateSpaceModel,
        ),
        (np.float64, np.float32),
    )
    param_names = ("class", "dtype")
    track_error_unit = "norm"

    def track_error(self, *args):
        out = self.sys.process(self.sig).time
        return float(spla.norm(self.ref - out))

    def setup(self, cls, dtype):
        self.sys, self.sig, self.ref = DATA[f"{dtype}"]
        if cls != StateSpaceModel:
            self.sys = cls.from_pyfar(self.sys)
            self.sys.init_state()
            self.sys.process(Signal(self.sig.time[:, :2], sampling_rate=1))
        self.sys.init_state()

    def time_solver(self, *args):
        self.sys.process(self.sig).time
