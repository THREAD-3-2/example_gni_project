"""
Functions for plotting trajectories, phase portraits, and arbitrary functions of the state.
"""

from example_gni_project.containers import SimulationResult
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['plot_trajectory', 'plot_phase_portrait', 'plot_observable']


def plot_trajectory(simulation: SimulationResult, filename=None):
    """Plot the entire discrete trajectory.
    The figure contains one line plot for every component of the state space.

    Parameters
    ----------
    simulation: SimulationResult
    filename: str
        If a filename (path) is provided, the plot is saved.
    """
    dim = len(simulation.ivp.variables)
    t = np.arange(0, 1 + simulation.steps) * simulation.h
    fig, ax = plt.subplots(nrows=dim, sharex=True)
    ax[0].set_title("trajectory")
    for i in range(dim):
        ax[i].plot(t, simulation.trajectory[:, i])
        ax[i].set_ylabel("$" + simulation.ivp.variables[i] + "$")
    ax[dim - 1].set_xlabel("time")
    if filename:
        fig.savefig(filename)


def plot_phase_portrait(simulation: SimulationResult, x=0, y=1, filename=None):
    """Plot the phase portrait.

    Parameters
    ----------
    simulation: SimulationResult
    x: int
        Index of the component that is plotted on the x-axis.
    y: int
        Index of the component that is plotted on the y-axis.
    filename: str
        If a filename (path) is provided, the plot is saved.
    """
    fig, ax = plt.subplots()
    ax.set_title("phase portrait")
    ax.plot(simulation.trajectory[:, x], simulation.trajectory[:, y])
    ax.set_aspect("equal")
    ax.set_xlabel("$" + simulation.ivp.variables[x] + "$")
    ax.set_ylabel("$" + simulation.ivp.variables[y] + "$")
    ax.scatter(
        simulation.ivp.initial_condition[x],
        simulation.ivp.initial_condition[y],
        color="red",
    )
    if filename:
        fig.savefig(filename)


def plot_observable(simulation: SimulationResult, name: str, observable, filename=None):
    """Plot the evolution of an observable (scalar-valued function of the state).
    
    Parameters
    ----------
    simulation: SimulationResult
    name: str
        name / plot title
    observable: function
        scalar-valued function of the state (e.g. energy)
    filename: str
        If a filename (path) is provided, the plot is saved.

    Example
    -------
    Plot just the first component of the state:

    >>> import autograd.numpy as np
    >>> from example_gni_project import *
    >>> def vector_field(x):
    ...     q, p = x
    ...     return np.array([p, -q])
    >>> ivp = IVP(("q", "p"), vector_field, np.array([1., 0.]))
    >>> simulation = simulate(ivp, explicit_euler, 0.1, 0.5)
    >>> plot_observable(simulation, "q", lambda x: x[0])
    """
    n = 1 + simulation.steps
    t = np.arange(0, n) * simulation.h
    values = np.empty(n, dtype=np.float64)
    for i in range(n):
        values[i] = observable(simulation.trajectory[i])
    fig, ax = plt.subplots()
    ax.set_title(name)
    ax.plot(t, values)
    ax.set_xlabel("time")
    ax.set_ylabel(name)
    if filename:
        fig.savefig(filename)

