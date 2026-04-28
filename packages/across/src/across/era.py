import numpy as np

from pymor.reductors.era import ERAReductor, RandomizedERAReductor
from pymor.algorithms.rand_la import RandomizedRangeFinder

from across.fastoperators import NumbaHankelOperator


class ERA:
    """Eigensystem Realization Algorithm (ERA).

    Wraps pymor's ERAReductor to provide a simple interface for reducing impulse responses in pyFAR
    format.

    Parameters
    ----------
    ir : pyfar.Signal
        The impulse response to be reduced. Must be a `pyfar.Signal` with `cdim=2`, where
        `ir.cshape=(n_inputs, n_outputs)`.

    """

    def __init__(self, ir):
        self.reductor = ERAReductor(
            ir.time.T[1:],
            sampling_time=1 / ir.sampling_rate,
            feedthrough=ir.time[..., -1].T,
            force_stability=False,
        )

    def reduce(self, order):
        """Reduce the impulse response to a state-space model of given order.

        Parameters
        ----------
        order : int
            The desired order of the reduced model.

        Returns
        -------
        A, B, C, D : ndarray
            The state-space matrices of the reduced model.

        """
        return self.reductor.reduce(order).to_matrices()[:4]


class _NumbaRandomizedERAReductor(RandomizedERAReductor):
    def __init__(
        self,
        data,
        sampling_time,
        force_stability=True,
        feedthrough=None,
        allow_transpose=True,
        rrf_opts={},
        num_left=None,
        num_right=None,
    ):
        super(RandomizedERAReductor, self).__init__(
            data, sampling_time, force_stability=force_stability, feedthrough=feedthrough
        )
        self.__auto_init(locals())
        # data = data.copy()
        if num_left is not None or num_right is not None:
            self.logger.info("Computing the projected Markov parameters ...")
            data = self._project_markov_parameters(num_left, num_right)
        if self.force_stability:
            data = np.concatenate([data, np.zeros_like(data)[1:]], axis=0)
        s = (data.shape[0] + 1) // 2
        self._transpose = (data.shape[1] < data.shape[2]) if allow_transpose else False
        self._H = NumbaHankelOperator(data[:s], r=data[s - 1 :])
        if self._transpose:
            self.logger.info("Using transposed formulation.")
            self._H = self._H.H

        # monkey patch RRF with dtype of data for memory efficiency
        dtype = data.dtype
        self._last_sv_U_V = None
        self._rrf = RandomizedRangeFinder(self._H, **rrf_opts)
        self._rrf.Omega = self._rrf.A.range.make_array(np.empty((0, self._rrf.A.range.dim), dtype=dtype))
        self._rrf.estimator_last_basis_size, self.last_estimated_error = 0, np.inf
        self._rrf.Q = [
            self._rrf.A.range.make_array(np.empty((0, self._rrf.A.range.dim), dtype=dtype))
            for _ in range(self._rrf.power_iterations + 1)
        ]
        self._rrf.R = [np.empty((0, 0), dtype=dtype) for _ in range(self._rrf.power_iterations + 1)]
        self._rrf._draw_samples = self._draw_samples

    def _draw_samples(self, num):
        # faster way of computing the random samples for Hankel matrices
        self._rrf.logger.info(f"Taking {num} samples ...")
        dtype = self.data.dtype
        V = np.zeros((self._H._circulant.source.dim, num), dtype=dtype)
        V[: self._H.source.dim] = self._H.source.random(num, distribution="normal").to_numpy().T
        return self._H.range.make_array(self._H._circulant._circular_matvec(V)[:, : self._H.range.dim])


class RandomizedERA(ERA):
    """Randomized Eigensystem Realization Algorithm (ERA).

    Wraps pymor's RandomizedERAReductor and adds numba acceleration to provide a
    simple interface for reducing impulse responses in pyFAR format.

    Parameters
    ----------
    ir : pyfar.Signal
        The impulse response to be reduced. Must be a `pyfar.Signal` with `cdim=2`, where
        `ir.cshape=(n_inputs, n_outputs)`.

    """

    def __init__(self, ir, dtype=np.float32):
        self.reductor = _NumbaRandomizedERAReductor(
            ir.time.T[1:].astype(dtype),
            sampling_time=1 / ir.sampling_rate,
            feedthrough=ir.time[..., -1].T,
            force_stability=True,
            rrf_opts={
                "block_size": 50,
                "power_iterations": 2,
                "qr_method": "shifted_chol_qr",
                "error_estimator": "loo",
                "qr_opts": {"orth_tol": 1e-6, "maxiter": 10},
            },
        )
