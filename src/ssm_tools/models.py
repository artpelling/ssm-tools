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

    def process(self, signal):
        out = self._process(signal)
        return Signal(out, sampling_rate=signal.sampling_rate)

    def _process(self, signal):
        out = np.zeros((self.n_outputs, signal.n_samples), self.dtype, order="F")
        sig = np.asfortranarray(signal.time)
        self._solver(out, np.asfortranarray(self.state), self._A, self._B, self._C, self._D, sig)
        return out

    @staticmethod
    @jit(
        [
            (
                float32[::1, :],
                float32[::1],
                float32[::1, :],
                float32[::1, :],
                float32[::1, :],
                float32[::1, :],
                float32[::1, :],
            ),
            (
                float64[::1, :],
                float64[::1],
                float64[::1, :],
                float64[::1, :],
                float64[::1, :],
                float64[::1, :],
                float64[::1, :],
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
