#!/usr/bin/env python3

import numpy as np
from numba import float32, float64, jit
from pyfar.classes.filter import StateSpaceModel


class NumbaStateSpaceModel(StateSpaceModel):
    def __init__(self, A, B, C, D=None, sampling_rate=None, state=None, dtype=None, storage="F", comment=""):
        D = np.zeros((C.shape[0], B.shape[1])) if D is None else D
        assert all([isinstance(M, np.ndarray) and (M.ndim == 2) for M in (A, B, C, D)])
        assert A.shape[1] == A.shape[0], "A needs to be square."
        assert B.shape[0] == A.shape[0], f"B needs to be of shape ({A.shape[0]}, m)."
        assert C.shape[1] == A.shape[0], f"C needs to be of shape (p, {A.shape[0]})."
        assert D.shape == (C.shape[0], B.shape[1]), f"D needs to be of shape ({C.shape[0], B.shape[1]})."
        dtype = np.result_type(A, B, C, D) if dtype is None else dtype
        A, B, C, D = (
            A.astype(dtype, order=storage),
            B.astype(dtype, order=storage),
            C.astype(dtype, order=storage),
            D.astype(dtype, order=storage),
        )
        super(StateSpaceModel, self).__init__(sampling_rate=sampling_rate, state=state, comment=comment)
        self._A, self._B, self._C, self._D, self._dtype, self._storage = A, B, C, D, dtype, storage

    @property
    def storage(self):
        """The memory layout of the internal matrices."""
        return self._storage

    @storage.setter
    def storage(self, value):
        assert value in ("F", "C"), "Storage must be either 'F' (Fortran) or 'C' (Contiguous)."
        self._storage = value

    @classmethod
    def from_pyfar(cls, sys: StateSpaceModel, storage="F"):
        return cls(
            A=sys._A,
            B=sys._B,
            C=sys._C,
            D=sys._D,
            sampling_rate=sys.sampling_rate,
            dtype=sys.dtype,
            storage=storage,
        )

    def _process(self, u):
        y = np.zeros((self.n_outputs, u.shape[1]), self.dtype, order=self.storage)
        if self.storage == "F":
            u = np.asfortranarray(u)
            solver = self._solver_F
        else:
            u = np.ascontiguousarray(u)
            solver = self._solver_C
        solver(y, self.state, self._A, self._B, self._C, self._D, u)
        return y

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
    def _solver_F(out, x, A, B, C, D, sig):
        for i in range(out.shape[1]):
            out[:, i] = C @ x + D @ sig[:, i]
            x = A @ x + B @ sig[:, i]

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
    def _solver_C(out, x, A, B, C, D, sig):
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
