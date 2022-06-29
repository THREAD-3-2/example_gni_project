"""
Abstract time stepping routine.
"""

import numpy as np

from example_gni_project.containers import SimulationResult

__all__ = ['simulate']


def simulate(ivp, scheme, h, duration):
    """Compute a discrete trajectory for an initial-value problem.

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

    Example
    -------
    Compute the evolution of a given initial value problem `ivp`
    for 10 seconds
    using the explicit Euler method
    with a time step of 0.1 seconds.

    >>> import autograd.numpy as np
    >>> from example_gni_project import *
    >>> def vector_field(x):
    ...     q, p = x
    ...     return np.array([p, -q])
    >>> ivp = IVP(("q", "p"), vector_field, np.array([1., 0.]))
    >>> simulation = simulate(ivp, explicit_euler, 0.1, 0.5)
    >>> simulation.trajectory[-1, :]
    array([ 0.9005 , -0.49001])
    """

    dim = len(ivp.variables)
    update = scheme(dim, ivp.vector_field, h)
    steps = round(duration / h)
    trajectory = np.empty((1 + steps, dim))
    trajectory[0] = ivp.initial_condition
    for i in range(steps):
        trajectory[i + 1] = update(trajectory[i])
    return SimulationResult(ivp, scheme, h, duration, steps, trajectory)
