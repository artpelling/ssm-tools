#!/usr/bin/env python3

import numpy as np
from numba import jit
from pyfar.classes.filter import StateSpaceModel
from pyfar import Signal
import scipy.linalg as spla


class NumbaStateSpaceModel(StateSpaceModel):
    solver = "numba"
    def __init__(self, A, B, C, D, sampling_rate=44100, dtype=np.float64, solver: str = "numba"):
        assert solver in ("numba", "blas"), f"Solver '{solver}' not implemented."
        super().__init__(A, B, C, D, sampling_rate=sampling_rate, dtype=dtype)
        self.solver = solver

    @classmethod
    def from_pyfar(cls, sys: StateSpaceModel, solver: str = "numba"):
        return cls(
            A=sys._A,
            B=sys._B,
            C=sys._C,
            D=sys._D,
            sampling_rate=sys.sampling_rate,
            dtype=sys.dtype,
            solver=solver,
        )

    def process(self, signal):
        out = self._process(signal)
        return Signal(out, sampling_rate=signal.sampling_rate)

    def _process(self, signal):
        out = np.zeros((self.n_outputs, signal.n_samples), self.dtype)
        match self.solver:
            case "numba":
                solver = self._basic_solver
            case "blas":
                solver = self._blas_solver
        solver(out, self.state, self._A, self._B, self._C, self._D, signal.time)
        return out

    @staticmethod
    @jit(nopython=True, cache=True)
    def _basic_solver(out, x, A, B, C, D, sig):
        for i in range(out.shape[1]):
            out[:, i] = C @ x + D @ sig[:, i]
            x = A @ x + B @ sig[:, i]

    @staticmethod
    def _blas_solver(out, x, A, B, C, D, sig):
        gemv = spla.get_blas_funcs("gemv", arrays=(A, B, C, D))
        for i in range(out.shape[1]):
            out[:, i] = gemv(1., D, sig[:, i], beta=0, y=out[:, i])
            out[:, i] = gemv(1., C, x, beta=1, y=out[:, i])
            x = gemv(1., A, x, beta=0, y=x)
            x = gemv(1., B, sig[:, i], beta=1, y=x)


class TriangularStateSpaceModel(NumbaStateSpaceModel):
    packed = False

    def process(self, signal):
        raise NotImplementedError("TriangularStateSpaceModel is not implemented yet.")


class DiagonalStateSpaceModel(NumbaStateSpaceModel):
    def process(self, signal):
        raise NotImplementedError("DiagonalStateSpaceModel is not implemented yet.")
