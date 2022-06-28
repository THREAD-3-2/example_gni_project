"""
Numerical integration schemes for solving initial value problems of ODEs.
"""

import autograd
import autograd.numpy as np
import scipy.optimize

__all__ = ['explicit_euler', 'implicit_euler', 'symplectic_euler', 'symplectic_euler2', 'störmer_verlet']


def explicit_euler(dim, vector_field, h):
    """function defining the explicit Euler scheme

    Parameters
    ----------
    dim: int
        dimension of the state vector
    vector_field: function
        right hand side of the ODE
    h: float
        time step size

    Returns
    -------
    update: function
        called by the simulate function
    """

    def update(x0):
        return x0 + h * vector_field(x0)

    return update


def implicit_euler(dim, vector_field, h):
    """function defining the implicit Euler scheme

    Parameters
    ----------
    dim: int
        dimension of the state vector
    vector_field: function
        right hand side of the ODE
    h: float
        time step size

    Returns
    -------
    update: function
        called by the simulate function
    """

    def residual_function(x, x0):
        return x - (x0 + h * vector_field(x))

    def jacobian_function(x, x0):
        return autograd.jacobian(lambda x: residual_function(x, x0))(x)

    def update(x0):
        solution = scipy.optimize.root(
            residual_function, x0, args=(x0), jac=jacobian_function
        )
        return solution.x

    return update


def symplectic_euler(dim, vector_field, h):
    """function defining the symplectic Euler scheme

    Parameters
    ----------
    dim: int
        dimension of the state vector
        Must be an even natural number.
    vector_field: function
        right hand side of the ODE
        First dim/2 components belong to first partition (explicit Euler).
        Remaining dim/2 components belong to second partition (implicit Euler).
    h: float
        time step size

    Returns
    -------
    update: function
        called by the simulate function
    """

    assert dim % 2 == 0

    def residual_function(x, x0):
        return x - (x0 + h * vector_field(np.hstack((x0[: dim // 2], x[dim // 2:]))))

    def jacobian_function(x, x0):
        return autograd.jacobian(lambda x: residual_function(x, x0))(x)

    def update(x0):
        solution = scipy.optimize.root(
            residual_function, x0, args=(x0), jac=jacobian_function
        )
        return solution.x

    return update


def symplectic_euler2(dim, vector_field, h):
    """function defining the other symplectic Euler scheme

    Parameters
    ----------
    dim: int
        dimension of the state vector
        Must be an even natural number.
    vector_field: function
        right hand side of the ODE
        First dim/2 components belong to first partition (explicit Euler).
        Remaining dim/2 components belong to second partition (implicit Euler).
    h: float
        time step size

    Returns
    -------
    update: function
        called by the simulate function
    """

    assert dim % 2 == 0

    def residual_function(x, x0):
        return x - (x0 + h * vector_field(np.hstack((x[: dim // 2], x0[dim // 2:]))))

    def jacobian_function(x, x0):
        return autograd.jacobian(lambda x: residual_function(x, x0))(x)

    def update(x0):
        solution = scipy.optimize.root(
            residual_function, x0, args=(x0), jac=jacobian_function
        )
        return solution.x

    return update


def störmer_verlet(dim, vector_field, h):
    """function defining the (explicit) Störmer-Verlet scheme

    Assumes separable Hamiltonian.

    Parameters
    ----------
    dim: int
        dimension of the state vector
        Must be an even natural number.
    vector_field: function
        right hand side of the ODE
        First dim/2 components belong to first partition (explicit Euler).
        Remaining dim/2 components belong to second partition (implicit Euler).
    h: float
        time step size

    Returns
    -------
    update: function
        called by the simulate function
    """

    assert dim % 2 == 0

    def update(x0):
        q0 = x0[: dim // 2]
        p0 = x0[dim // 2 :]
        p12 = p0 + h / 2 * vector_field(x0)[dim // 2 :]  # p0 is just a dummy input
        q1 = q0 + h * p12
        p1 = (
            p12 + h / 2 * vector_field(np.hstack((q1, p0)))[dim // 2 :]
        )  # p0 is just a dummy input
        return np.hstack((q1, p1))

    return update
