import numpy as np
from pyfar.classes.filter import StateSpaceModel
from pymor.algorithms.rand_la import RandomizedSVD
from pymor.core.defaults import set_defaults
from pymor.reductors.era import ERAReductor
from pymor.reductors.era import RandomizedERAReductor as pyMORRandomizedERAReductor

from across.fastoperators import NumbaHankelOperator

set_defaults({"pymor.bindings.scipy.svd_lapack_driver.driver": "gesdd"})


class ERA:
    """Eigensystem Realization Algorithm (ERA).

    Wraps pymor's ERAReductor to provide a simple interface for reducing impulse responses in pyFAR
    format.

    Parameters
    ----------
    ir : pyfar.Signal
        The impulse response to be reduced. Must be a `pyfar.Signal` with `cdim=2`, where
        `ir.cshape=(n_inputs, n_outputs)`.
    pad_zero : bool
        Whether to pad the impulse response with zeros before constructing the Hankel operator.

    """

    def __init__(self, ir, pad_zero=False):
        self.reductor = ERAReductor(
            ir.time.T[1:],
            sampling_time=1 / ir.sampling_rate,
            feedthrough=ir.time[..., 0].T,
            force_stability=pad_zero,
        )

    def reduce(self, order, num_left=None, num_right=None):
        """Reduce the impulse response to a state-space model of given order.

        Parameters
        ----------
        order : int
            The desired order of the reduced model.

        Returns
        -------
        sys : pyfar.StateSpaceModel
            The reduced order state-space model.
        num_left : int or None
            Number of left (output) directions for tangential projection.
        num_right: int or None
            Number of right (input) directions for tangential projection.
        """
        return StateSpaceModel(
            *self.reductor.reduce(order, num_left=num_left, num_right=num_right).to_matrices()[:4],
            1 / self.reductor.sampling_time,
        )


class _NumbaRandomizedERAReductor(pyMORRandomizedERAReductor):
    def __init__(
        self,
        data,
        sampling_time,
        force_stability=True,
        feedthrough=None,
        allow_transpose=True,
        power_iterations=2,
        rrf_args=None,
        num_left=None,
        num_right=None,
    ):
        super(pyMORRandomizedERAReductor, self).__init__(
            data, sampling_time, force_stability=force_stability, feedthrough=feedthrough
        )
        self.__auto_init(locals())
        if rrf_args is not None and "error_estimator" in rrf_args:
            assert rrf_args["error_estimator"] == "loo", "Only the leave-one-out error estimator is supported."
        if num_left is not None or num_right is not None:
            self.logger.info("Computing the projected Markov parameters ...")
            data = self._project_markov_parameters(num_left, num_right)
        s = data.shape[0] if self.force_stability else (data.shape[0] + 1) // 2
        self._transpose = (data.shape[1] < data.shape[2]) if allow_transpose else False
        self._H = NumbaHankelOperator(data) if self.force_stability else NumbaHankelOperator(data[:s], r=data[s - 1 :])
        if self._transpose:
            self.logger.info("Using transposed formulation.")
            self._H = self._H.H
        rrf_args = {"qr_method": "shifted_chol_qr", **(rrf_args or {}), "error_estimator": "loo"}
        self.randomized_svd = RandomizedSVD(
            self._H, power_iterations=power_iterations, low_rank_svd_method="scipy_svd", rrf_args=rrf_args
        )
        range_finder = self.randomized_svd.range_finder
        range_finder.Omega = self._H.range.make_array(np.empty((self._H.range.dim, 0), dtype=data.dtype, order="F"))
        range_finder.Q = [
            self._H.range.make_array(np.empty((self._H.range.dim, 0), dtype=data.dtype, order="F"))
            for _ in range(power_iterations + 1)
        ]
        range_finder.R = [np.empty((0, 0), dtype=data.dtype, order="F") for _ in range(power_iterations + 1)]
        self.randomized_svd.B = self._H.source.make_array(
            np.empty((self._H.source.dim, 0), dtype=data.dtype, order="F")
        )
        range_finder._draw_samples = self._draw_samples

    def _draw_samples(self, num):
        self.randomized_svd.range_finder.logger.info(f"Taking {num} samples ...")
        V = np.zeros((self._H._circulant.source.dim, num), dtype=self.data.dtype, order="F")
        V[: self._H.source.dim] = self._H.source.random(num, distribution="normal").to_numpy()
        return self._H.range.make_array(self._H._circulant._circular_matvec(V, self._H.range.dim))


class RandomizedERA(ERA):
    """Randomized Eigensystem Realization Algorithm (ERA).

    Wraps pymor's RandomizedERAReductor and adds numba acceleration to provide a
    simple interface for reducing impulse responses in pyFAR format.

    Parameters
    ----------
    ir : pyfar.Signal
        The impulse response to be reduced. Must be a `pyfar.Signal` with `cdim=2`, where
        `ir.cshape=(n_inputs, n_outputs)`.
    pad_zero : bool
        Whether to pad the impulse response with zeros before constructing the Hankel operator.

    """

    def __init__(self, ir, pad_zero=False, dtype=np.float32):
        self.reductor = _NumbaRandomizedERAReductor(
            ir.time.T[1:].astype(dtype),
            sampling_time=1 / ir.sampling_rate,
            feedthrough=ir.time[..., 0].T,
            force_stability=pad_zero,
            power_iterations=2,
            rrf_args={
                "block_size": 50,
                "qr_method": "shifted_chol_qr",
                "error_estimator": "loo",
                "qr_opts": {"orth_tol": 1e-6, "maxiter": 10},
            },
        )
