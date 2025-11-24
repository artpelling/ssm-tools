# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from ssm_tools.systems import NumbaStateSpaceModel
from pyfar.classes.filter import StateSpaceModel
from pyfar import Signal
import numpy as np
import pyfar as pf
import scipy.linalg as spla


DTYPE = (np.float64, np.float32)
n, m, p = 512, 5, 12
T = 4096

DATA = dict()
for dtype in DTYPE:
    A = 0.8 * np.eye(n)
    B = np.random.randn(n, m).astype(dtype)
    C = np.random.randn(p, n).astype(dtype)
    D = None
    sys = StateSpaceModel(A, B, C, D, sampling_rate=1, dtype=dtype)
    sig = Signal(np.random.randn(m, T).astype(dtype), sampling_rate=1)
    conv = pf.dsp.convolve(sig.reshape((m, 1)), sys.impulse_response(sig.n_samples))
    ref = conv.time[..., :T].sum(axis=0)
    DATA[f"{dtype}"] = (sys, sig, ref)


class Compare:
    params = ((StateSpaceModel, NumbaStateSpaceModel), (np.float64, np.float32))
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
            self.sys.process(Signal(self.sig.time[:, :1], sampling_rate=1))
        self.sys.init_state()

    def time_solver(self, *args):
        self.sys.process(self.sig).time
