#!/usr/bin/env python3

import numpy as np
from pyfar.classes.filter import StateSpaceModel as PyfarStateSpaceModel

from ssmsolve_rs import solve_f32, solve_f64


class StateSpaceModel(PyfarStateSpaceModel):
    """State-space model with a Rust solver backend.

    The solver computes the discrete-time state equations:

        y[:, i] = C @ x + D @ u[:, i]
        x       = A @ x + B @ u[:, i]

    The system state ``x`` is updated in-place after each call to :meth:`process`, so sequential
    calls carry state across chunk boundaries.

    Parameters
    ----------
    A : numpy.ndarray, shape (n, n)
        State transition matrix.
    B : numpy.ndarray, shape (n, m)
        Input matrix.
    C : numpy.ndarray, shape (p, n)
        Output matrix.
    D : numpy.ndarray, shape (p, m), optional
        Feed-through matrix.  Defaults to zeros.
    sampling_rate : float, optional
        Sampling rate in Hz.
    state : numpy.ndarray, shape (n,), optional
        Initial state vector.  Defaults to zeros (via :meth:`init_state`).
    dtype : numpy.dtype, optional
        Working dtype; must be ``float32`` or ``float64``. Inferred from the matrices when not
        provided.
    storage : {'F', 'C'}, optional
        Memory layout for the system matrices (``'F'`` for column-major, ``'C'`` for row-major).
        Defaults to ``'F'``.
    comment : str, optional
        Any comment.
    """

    _SUPPORTED_DTYPES = (np.float32, np.float64)

    def __init__(self, A, B, C, D=None, sampling_rate=None, state=None, dtype=None, storage="F", comment=""):
        D = np.zeros((C.shape[0], B.shape[1])) if D is None else D
        assert all([isinstance(M, np.ndarray) and (M.ndim == 2) for M in (A, B, C, D)])
        assert A.shape[1] == A.shape[0], "A needs to be square."
        assert B.shape[0] == A.shape[0], f"B needs to be of shape ({A.shape[0]}, m)."
        assert C.shape[1] == A.shape[0], f"C needs to be of shape (p, {A.shape[0]})."
        assert D.shape == (C.shape[0], B.shape[1]), f"D needs to be of shape ({C.shape[0], B.shape[1]})."
        dtype = np.result_type(A, B, C, D) if dtype is None else np.dtype(dtype)
        if dtype not in self._SUPPORTED_DTYPES:
            raise TypeError(f"StateSpaceModel only supports float32 and float64, got {dtype}.")
        A, B, C, D = (
            A.astype(dtype, order=storage),
            B.astype(dtype, order=storage),
            C.astype(dtype, order=storage),
            D.astype(dtype, order=storage),
        )
        super(StateSpaceModel, self).__init__(A, B, C, D, sampling_rate=sampling_rate, state=state, dtype=dtype, comment=comment)
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
    def from_pyfar(cls, sys: PyfarStateSpaceModel, storage="F"):
        """Construct a :class:`StateSpaceModel` from a pyfar :class:`StateSpaceModel`.

        Parameters
        ----------
        sys : pyfar.classes.filter.StateSpaceModel
            Source system.
        storage : {'F', 'C'}, optional
            Memory layout for the internal matrices.

        Returns
        -------
        StateSpaceModel
        """
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
            u = np.asfortranarray(u, dtype=self.dtype)
        else:
            u = np.ascontiguousarray(u, dtype=self.dtype)
        if self._dtype == np.dtype(np.float32):
            solve_f32(y, self.state, self._A, self._B, self._C, self._D, u)
        else:
            solve_f64(y, self.state, self._A, self._B, self._C, self._D, u)
        return y


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
