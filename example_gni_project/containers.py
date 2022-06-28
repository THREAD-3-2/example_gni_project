"""
Containers for of initial value problems and simulation results.
"""

import collections


__all__ = ['IVP', 'SimulationResult']


IVP = collections.namedtuple("IVP", ("variables", "vector_field", "initial_condition"))
IVP.__doc__ = """Type defining an initial value problem.

Attributes
----------
variables: tuple of strings
    names of the state variables
vector_field: function
    right hand side of the ODE
initial_condition: np.ndarray
    initial condition
"""


SimulationResult = collections.namedtuple(
    "SimulationResult", ("ivp", "scheme", "h", "duration", "steps", "trajectory")
)
SimulationResult.__doc__ = """Type defining a simulation result.

Attributes
----------
ivp: IVP
    initial value problem
scheme: function
    utilized integration scheme
h: float
    time step size
duration: float
    duration of simulation
steps: int
    number of time steps
trajectory: ndarray
    discrete trajectory
    numpy array with shape (1+steps, len(variables))
"""
