from pymor.reductors.era import ERAReductor


class ERA:
    """Eigensystem Realization Algorithm (ERA).

    Wraps pymor's ERAReductor to provide a simple interface for reducing impulse responses in pyFAR format.

    Parameters
    ----------
    ir : pyfar.Signal
        The impulse response to be reduced. Must be a `pyfar.Signal` with `cdim=2`, where `ir.cshape=(n_inputs, n_outputs)`.
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
        A, B, C : ndarray
            The state-space matrices of the reduced model.
        """
        return self.reductor.reduce(order).to_matrices()[:4]
