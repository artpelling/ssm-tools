from pymor.reductors.era import ERAReductor


class ERA:
    """Eigensystem Realization Algorithm (ERA).

    Wraps pymor's ERAReductor to provide a simple interface for reducing impulse resonses in pyFAR format.

    Parameters
    ----------
    ir : pyfar.Signal
        The impulse response to be reduced.
    """

    def __init__(self, ir):
        self.reductor = ERAReductor(
            ir.time.T, sampling_time=1 / ir.sampling_rate, force_stability=False
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
        return self.reductor.reduce(order).to_matrices()[:3]
