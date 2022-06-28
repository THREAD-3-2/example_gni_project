"""
Abstract time stepping routine.
"""

import numpy as np

from example_gni_project.containers import SimulationResult

__all__ = ['simulate']


def simulate(ivp, scheme, h, duration):
    """compute discrete trajectory for an initial-value problem

    Parameters
    ----------
    ivp: IVP
        initial value problem
    scheme: function
        function defining the integration scheme
    h: float
        time step size
    duration: float
        duration of simulation

    Returns
    -------
    simulation: SimulationResult
        simulation result
    """

    dim = len(ivp.variables)
    update = scheme(dim, ivp.vector_field, h)
    steps = round(duration / h)
    trajectory = np.empty((1 + steps, dim))
    trajectory[0] = ivp.initial_condition
    for i in range(steps):
        trajectory[i + 1] = update(trajectory[i])
    return SimulationResult(ivp, scheme, h, duration, steps, trajectory)
