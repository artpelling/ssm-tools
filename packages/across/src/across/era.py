import numpy as np
from pyfar.classes.filter import StateSpaceModel
from pymor.algorithms.rand_la import RandomizedSVD
from pymor.core.defaults import set_defaults
from pymor.reductors.era import ERAReductor
from pymor.reductors.era import RandomizedERAReductor as pyMORRandomizedERAReductor

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


class ProjectedERA(ERA):
    """ERA with predefined tangential directions and original input/output channels.

    The non-direct Markov parameters are projected before passing them to pyMOR.
    The resulting B and C matrices are lifted back to the original channels,
    while the original feedthrough is preserved without projection.

    Parameters
    ----------
    ir : pyfar.Signal
        Signal with ``cdim=2`` and ``cshape=(n_inputs, n_outputs)``.
    pad_zero : bool, optional
        Whether to zero-pad the Signal when constructing the Hankel operator.
    left_basis : array_like, optional
        Finite, real, column-orthonormal output basis of shape
        ``(n_outputs, n_left)``. Omit to leave the output channels unprojected.
    right_basis : array_like, optional
        Finite, real, column-orthonormal input basis of shape
        ``(n_inputs, n_right)``. Omit to leave the input channels unprojected.

    Notes
    -----
    At least one basis is required. Bases are copied and fixed at construction;
    their column counts determine the number of tangential directions. Full-width
    bases are supported. No additional direction counts are accepted by ``reduce``.
    """

    def __init__(self, ir, pad_zero=False, *, left_basis=None, right_basis=None):
        if ir.cdim != 2:
            raise ValueError("ir must have cdim=2 with cshape=(n_inputs, n_outputs).")
        if left_basis is None and right_basis is None:
            raise ValueError("At least one of left_basis or right_basis must be supplied.")
        self.left_basis = self._validate_basis(left_basis, ir.cshape[1], "left_basis")
        self.right_basis = self._validate_basis(right_basis, ir.cshape[0], "right_basis")
        self._feedthrough = ir.time[..., 0].T.copy()
        data = self._project_markov_parameters(ir.time.T[1:], self.left_basis, self.right_basis)
        self.reductor = ERAReductor(data, sampling_time=1 / ir.sampling_rate, force_stability=pad_zero)

    @staticmethod
    def _project_markov_parameters(data, left_basis, right_basis):
        """Use time-contiguous matrix products and the lower-FLOP contraction order."""
        _, p, m = data.shape
        # pyfar stores time contiguously; keep it inside each BLAS matrix, not as a batch axis.
        data = data.transpose(2, 1, 0)  # (inputs, outputs, time)
        if right_basis is None:
            data = left_basis.T @ data
        elif left_basis is None:
            data = (right_basis.T @ data.swapaxes(0, 1)).swapaxes(0, 1)
        else:
            n_left, n_right = left_basis.shape[1], right_basis.shape[1]
            # The common time dimension and factor of two cancel from both FLOP estimates.
            left_cost = n_left * m * (p + n_right)
            right_cost = p * n_right * (m + n_left)
            if (left_cost, n_left * m) <= (right_cost, p * n_right):
                data = left_basis.T @ data
                data = (right_basis.T @ data.swapaxes(0, 1)).swapaxes(0, 1)
            else:
                data = right_basis.T @ data.swapaxes(0, 1)
                data = left_basis.T @ data.swapaxes(0, 1)
        return data.transpose(2, 1, 0)

    @staticmethod
    def _validate_basis(basis, dimension, name):
        if basis is None:
            return None
        basis = np.array(basis, copy=True)
        if basis.ndim != 2 or basis.shape[0] != dimension or not 0 < basis.shape[1] <= dimension:
            raise ValueError(f"{name} must have shape ({dimension}, k) with 1 <= k <= {dimension}.")
        if not np.issubdtype(basis.dtype, np.number) or not np.isrealobj(basis) or not np.isfinite(basis).all():
            raise ValueError(f"{name} must contain finite, real numbers.")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=1e-5, atol=1e-6):
            raise ValueError(f"{name} must have orthonormal columns.")
        basis.setflags(write=False)
        return basis

    def reduce(self, order):
        """Construct a state-space model of the given order in the original channels.

        Parameters
        ----------
        order : int
            Desired order of the reduced model.

        Returns
        -------
        pyfar.StateSpaceModel
            State-space model with the original feedthrough and sampling rate.
        """
        A, B, C = self.reductor.reduce(order).to_matrices()[:3]
        if self.right_basis is not None:
            B = B @ self.right_basis.T
        if self.left_basis is not None:
            C = self.left_basis @ C
        return StateSpaceModel(A, B, C, self._feedthrough, 1 / self.reductor.sampling_time)


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
        from across.fastoperators import NumbaHankelOperator

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
