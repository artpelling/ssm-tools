import numba as nb
import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.fft import fft, ifft, rfft, irfft

from pymor.operators.numpy import NumpyCirculantOperator, NumpyHankelOperator


class NumbaCirculantOperator(NumpyCirculantOperator):
    @staticmethod
    @nb.njit(
        [
            nb.float32[::1, :](
                nb.int64, nb.int64, nb.int64, nb.int64, nb.float32[:, ::1], nb.float32[:, ::1], nb.complex64[:, :, ::1]
            ),
            nb.float64[::1, :](
                nb.int64, nb.int64, nb.int64, nb.int64, nb.float64[:, ::1], nb.float64[:, ::1], nb.complex128[:, :, ::1]
            ),
        ],
        parallel=True,
        fastmath=True,
    )
    def _real_ops(m, p, n, d, vec, y, C):
        dim = d // p
        for j in range(m):
            x = vec[j::m]
            X = rfft(x, axis=0)
            for i in nb.prange(p):
                Y = X * C[j, i].reshape(-1, 1)
                # setting n=n below is necessary to allow uneven lengths but considerably slower
                # Hankel operator will always pad to even length to avoid that
                y[i::p] += irfft(Y, n=n, axis=0)[:dim]
        return y.T

    @staticmethod
    def _complex_ops(m, p, n, d, vec, y, C):
        for j in range(m):
            x = vec[j::m]
            X = fft(x, axis=0)
            for i in range(p):
                Y = X * C[:, i, j].reshape(-1, 1)
                Y = ifft(Y, axis=0, overwrite_x=True)
                y[i::p] += Y[: d // p]
        return y.T

    def _circular_matvec(self, vec):
        n, p, m = self._arr.shape
        s, k = vec.shape
        d = self.range.dim

        # use real arithmetic if possible
        isreal = np.isrealobj(self._arr) and np.isrealobj(vec)
        ismixed = np.isrealobj(self._arr) and np.iscomplexobj(vec)

        C = self._circulant()
        if ismixed:
            l = s // m - C.shape[0] + 1
            C = np.concatenate([C, C[1:l].conj()[::-1]])
        C = np.ascontiguousarray(C.T)
        y = np.zeros((self.range.dim, k), dtype=np.promote_types(self._arr.dtype, vec.dtype))
        return self._real_ops(m, p, n, d, vec, y, C) if isreal else self._complex_ops(m, p, n, d, vec, y, C)


class NumbaHankelOperator(NumpyHankelOperator):
    def __init__(self, c, r=None, name=None):
        super().__init__(c, r=r, name=name)
        k, l = self.c.shape[0], self.r.shape[0]
        n = k + l - 1
        # zero pad to even length if real to avoid slow irfft
        z = int(np.isrealobj(self.c) and np.isrealobj(self.r) and n % 2)
        h = np.concatenate((self.c, self.r[1:], np.zeros([z, *self.c.shape[1:]], dtype=c.dtype)))
        shift = n // 2 + int(np.ceil((k - l) / 2)) + (n % 2) + z  # this works
        self._circulant = NumbaCirculantOperator(np.roll(h, shift, axis=0), name=self.name + " (implicit circulant)")

    def to_scipy_linear_operator(self):
        def matvec(x):
            return self.apply(self.source.from_numpy(x.T)).to_numpy().T

        def rmatvec(x):
            return self.apply_adjoint(self.range.from_numpy(x.T)).to_numpy().T

        return LinearOperator(shape=(self.range.dim, self.source.dim), matvec=matvec, rmatvec=rmatvec)
