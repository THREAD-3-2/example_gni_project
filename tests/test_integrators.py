"""tests for integrators.py"""

import autograd.numpy as np
from example_gni_project.integrators import *


def _test_symmetry(scheme):
    """
    Assert the symmetry of the given scheme based on the pendulum example."""

    def pendulum_vector_field(x):
        q, p = x
        return np.array([p, -np.sin(q)])

    dim = 2
    h = 0.1
    update_f = scheme(dim, pendulum_vector_field, +h)
    update_b = scheme(dim, pendulum_vector_field, -h)

    ic = np.array([1., 0.])

    assert np.allclose(ic, update_b(update_f(ic)))


def test_symmetry_störmer_verlet():
    """
    Assert the symmetry of the Störmer-Verlet scheme.
    """
    _test_symmetry(störmer_verlet)


def test_composition_störmer_verlet():
    """
    Assert that the symplectic Euler schemes
    result from the composition of
    the explicit Euler scheme and the implicit Euler scheme.
    """

    def pendulum_vector_field(x):
        q, p = x
        return np.array([p, -np.sin(q)])

    dim = 2
    h = 0.1
    update_se1 = symplectic_euler(dim, pendulum_vector_field, h/2)
    update_se2 = symplectic_euler2(dim, pendulum_vector_field, h/2)
    update_sv = störmer_verlet(dim, pendulum_vector_field, h)

    ic = np.array([1., 0.])

    assert np.allclose(update_se2(update_se1(ic)), update_sv(ic))
