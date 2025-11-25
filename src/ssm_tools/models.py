#!/usr/bin/env python3

import numpy as np
from numba import jit, float64, float32
from pyfar.classes.filter import StateSpaceModel
from pyfar import Signal


class NumbaStateSpaceModel(StateSpaceModel):
    @classmethod
    def from_pyfar(cls, sys: StateSpaceModel):
        return cls(
            A=sys._A,
            B=sys._B,
            C=sys._C,
            D=sys._D,
            sampling_rate=sys.sampling_rate,
            dtype=sys.dtype,
        )

    def _process(self, u):
        y = np.zeros((self.n_outputs, u.shape[1]), self.dtype, order="C")
        u = np.ascontiguousarray(u)
        self._solver(y, self.state, self._A, self._B, self._C, self._D, u)
        return y

    @staticmethod
    @jit(
        [
            (
                float32[:, ::1],
                float32[::1],
                float32[:, ::1],
                float32[:, ::1],
                float32[:, ::1],
                float32[:, ::1],
                float32[:, ::1],
            ),
            (
                float64[:, ::1],
                float64[::1],
                float64[:, ::1],
                float64[:, ::1],
                float64[:, ::1],
                float64[:, ::1],
                float64[:, ::1],
            ),
        ],
        nopython=True,
        cache=True,
    )
    def _solver(out, x, A, B, C, D, sig):
        for i in range(out.shape[1]):
            out[:, i] = C @ x + D @ sig[:, i]
            x = A @ x + B @ sig[:, i]


class TriangularStateSpaceModel(StateSpaceModel):
    packed = False

    @classmethod
    def from_pyfar(cls, sys: StateSpaceModel):
        raise NotImplementedError("TriangularStateSpaceModel is not implemented yet.")

    def process(self, signal):
        raise NotImplementedError("TriangularStateSpaceModel is not implemented yet.")


class DiagonalStateSpaceModel(StateSpaceModel):
    @classmethod
    def from_pyfar(cls, sys: StateSpaceModel):
        raise NotImplementedError("DiagonalStateSpaceModel is not implemented yet.")

    def process(self, signal):
        raise NotImplementedError("DiagonalStateSpaceModel is not implemented yet.")
