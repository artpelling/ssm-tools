#!/usr/bin/env python3

import numpy as np

from pyfar.classes.filter import StateSpaceModel
from pyfar import Signal

from ssm_tools.solvers import basic_solver


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
        out = np.zeros((self.n_outputs, signal.n_samples), self.dtype)
        basic_solver(out, self.state, self._A, self._B, self._C, self._D, signal.time)
        return Signal(out, sampling_rate=signal.sampling_rate)


class TriangularStateSpaceModel:
    packed = False
    pass


class DiagonalStateSpaceModel:
    pass
