"""Tests for ideal generators."""

import numpy as np
import pytest
import sympy as sp

from deepgroebner.ideals import *


R1, x, y, z = sp.ring('x,y,z', sp.FF(32003), 'grevlex')
R2, a, b, c, d = sp.ring('a,b,c,d', sp.QQ, 'lex')
R3, t, u, v = sp.ring('t,u,v', sp.FF(101), 'grlex')


def test_cyclic():
    F = cyclic(3)
    x, y, z = F[0].ring.gens
    assert F == [x + y + z, x*y + y*z + z*x, x*y*z - 1]


@pytest.mark.parametrize("ring, d, B", [
    (R1, 0, [R1.one]),
    (R2, 1, [a, b, c, d]),
    (R1, 2, [x**2, x*y, x*z, y**2, y*z, z**2]),
    (R3, 3, [t**3, t**2*u, t**2*v, t*u**2, t*u*v, t*v**2, u**3, u**2*v, u*v**2, v**3]),
])
def test_basis(ring, d, B):
    assert set(basis(ring, d)) == set(B)


@pytest.mark.parametrize("d, dist, constants, D", [
    (1, "weighted", False, np.array([0.0, 1.0])),
    (1, "weighted", True, np.array([0.5, 0.5])),
    (1, "uniform", True, np.array([0.25, 0.75])),
    (5, "weighted", False, np.array([0] + 5 * [0.2])),
    (5, "weighted", True, np.ones(6)/6),
    (5, "uniform", True, np.array([1, 3, 6, 10, 15, 21]) / 56.0),
    (3, "maximum", True, np.array([0.5, 0, 0, 0.5])),
    (3, "maximum", False, np.array([0, 0, 0, 1.0])),
    (3, 'uniform', False, np.array([0, 3, 6, 10]) / 19.0),
    (3, 'weighted', False, np.array([0, 1, 1, 1]) / 3.0),
])
def test_degree_distribution(d, dist, constants, D):
    degree_dist = degree_distribution(R1, d, dist=dist, constants=constants)
    assert np.array_equal(degree_dist, D)


def test_RandomBinomialIdealGenerator():
    ideal_gen = RandomBinomialIdealGenerator(3, 5, 5)
    ideal_gen.seed(123)
    x0, x1, x2 = ideal_gen.ring.gens
    F = [x0**2*x2 + 495*x2,
         x1*x2**4 + 5901*x0*x1*x2,
         x0**5 + 14384*x0**3*x2**2,
         x0**3*x1*x2 + 16417*x1**2*x2,
         x0**3*x1*x2 + 13109*x1**3*x2**2]    
    assert next(ideal_gen) == F


def test_RandomIdealGenerator():
    ideal_gen = RandomIdealGenerator(3, 5, 5, 0.5)
    ideal_gen.seed(123)
    x0, x1, x2 = ideal_gen.ring.gens
    F = [x0**3*x2**2 + 10689*x0**2*x1 + 12547*x1*x2**2,
         x1*x2**4 + 15388*x1**2*x2 + 22355*x1*x2**2,
         x0*x1*x2**2 + 4665*x0**3 + 15800*x0**2*x1,
         x0**3*x2**2 + 8782*x1**2*x2**3 + 15890*x1*x2**2,
         x1**2*x2 + 30687*x1**2]
    assert next(ideal_gen) == F
