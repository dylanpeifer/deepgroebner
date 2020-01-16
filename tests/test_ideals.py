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


def test_FixedIdealGenerator_0():
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    assert [y - x**2, z - x**3] == next(ideal_gen)


def test_FixedIdealGenerator_1():
    """Check that generated ideals are distinct objects from each other."""
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    G = next(ideal_gen)
    G[0] = x
    assert [y - x**2, z - x**3] == next(ideal_gen)


def test_FixedIdealGenerator_2():
    """Check that generated ideals are distinct objects from input."""
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    F[0] = x
    assert [y - x**2, z - x**3] == next(ideal_gen)

@pytest.mark.parametrize("ring, c, seed", [
    (R1, 19967, 123),
    (R1, 29314, 99),
    (R3, 67, 123),
])
def test_random_nonzero_coeff(ring, c, seed):
    np.random.seed(seed)
    assert random_nonzero_coeff(ring) == c


@pytest.mark.parametrize("ring, d, B", [
    (R1, 0, [R1.one]),
    (R2, 1, [a, b, c, d]),
    (R1, 2, [x**2, x*y, x*z, y**2, y*z, z**2]),
    (R3, 3, [t**3, t**2*u, t**2*v, t*u**2, t*u*v, t*v**2, u**3, u**2*v, u*v**2, v**3]),
])
def test_basis(ring, d, B):
    assert set(basis(ring, d)) == set(B)


def test_random_binomial_0():
    """Check that monomials are ordered in grevlex, not lex."""
    np.random.seed(10)
    b = random_binomial(np.ones(7)/7, R3)
    assert b == v**3 + 10*t*u


@pytest.mark.parametrize("D, ring, b", [
    (np.array([0.5, 0.5]), R1, z + 19967),
    (np.array([0, 0.2, 0.3, 0.5]), R1, x**2*z + 19967*x*y),
    (np.ones(11)/11, R3, t**2*u**4 + 67*u**2),
    (np.ones(5)/5, R3, u**2 + 67*v),
])
def test_random_binomial_1(D, ring, b):
    np.random.seed(123)
    assert random_binomial(D, ring) == b


@pytest.mark.parametrize("D, ring, b", [
    (np.array([0.5, 0.5]), R1, x + 20963*y),
    (np.array([0, 0, 0.2, 0, 0.8]), R1, x**3*y + 20963*x**2*y*z),
    (np.ones(12)/12, R1, x**9*y + 20963*x**2*y**8),
    (np.ones(7)/7, R3, t**5*u + 99*t**4*u*v),
])
def test_random_binomial_2(D, ring, b):
    np.random.seed(12345)
    assert random_binomial(D, ring, homogeneous=True) == b

@pytest.mark.parametrize("D, ring, b", [
    (np.array([0.2, 0.8]), R1, x + 32002),
    (np.ones(11)/11, R1, x**8*y*z + 32002*x*y),
    (np.array([0]*20 + [1]), R1, x**6*y**14 + 32002*x**5*y**3*z**12),
    (np.ones(7)/7, R3, t**4*u*v + 100*u),
    ([0.1, 0.1, 0, 0, 0, 0.8], R3, t**3*u*v + 100*u),
    (np.array([0.5, 0.5]), R3, t + 100),
])
def test_random_binomial_3(D, ring, b):
    np.random.seed(543)
    assert random_binomial(D, ring, pure=True) == b

@pytest.mark.parametrize("D, ring, b", [
    (np.ones(12)/12, R1, x**2 + 32002*y**2),
    (np.ones(9)/9, R2, a - d),
    ([0, 0.8, 0.2], R3, t + 100*u),
])
def test_random_binomial_4(D, ring, b):
    np.random.seed(11)
    assert random_binomial(D, ring, homogeneous=True, pure=True) == b

@pytest.mark.parametrize("D, ring, b", [
    ([0, 0.8, 0.2], R1, y + 19967*z),
    (np.ones(10)/10, R1, y*z**6 + 19967*x*y**3),
    ([0]*10 + [1], R3, t*u**7*v**2 + 67*u**8*v**2),
])
def test_random_binomial_5(D, ring, b):
    np.random.seed(123)
    bases = [basis(ring, i) for i in range(len(D))]
    assert random_binomial(D, ring, bases=bases) == b


def test_RandomBinomialIdealGenerator_0():
    np.random.seed(123)
    ideal_gen = RandomBinomialIdealGenerator(3, 5, 5)
    x0, x1, x2 = ideal_gen.ring.gens
    I = [x0**3*x2 + 19967*x1**3,
         x0*x1**3 + 15378*x0**2*x1,
         x0**3 + 97*x2,
         x1*x2**3 + 22753*x0*x1*x2,
         x0**3*x2 + 14945*x0*x1**2]
    assert next(ideal_gen) == I


@pytest.mark.parametrize("d, consts, degs, D", [
    (1, False, "uniform", np.array([0.0, 1.0])),
    (1, True, "uniform", np.array([0.5, 0.5])),
    (1, True, "weighted", np.array([0.25, 0.75])),
    (5, False, "uniform", np.array([0] + 5 * [0.2])),
    (5, True, "uniform", np.ones(6)/6),
    (5, True, "weighted", np.array([1, 3, 6, 10, 15, 21])/56.0),
    (3, True, "maximum", np.array([0.5, 0, 0, 0.5])),
    (3, False, "maximum", np.array([0, 0, 0, 1.0])),
])
def test_RandomBinomialIdealGenerator_1(d, consts, degs, D):
    ideal_gen = RandomBinomialIdealGenerator(3, d, 3, constants=consts, degrees=degs)
    assert np.array_equal(ideal_gen.dist, D)


def test_RandomBinomialIdealGenerator_2():
    np.random.seed(681)
    ideal_gen = RandomBinomialIdealGenerator(3, 3, 2, degrees='weighted', order='lex')
    x0, x1, x2 = ideal_gen.ring.gens
    assert next(ideal_gen) == [x1 + 11414*x2**3, x0 + 11229*x1*x2**2]


def test_RandomBinomialIdealGenerator_3():
    np.random.seed(10)
    ideal_gen = RandomBinomialIdealGenerator(5, 10, 5, degrees='weighted',
                                             homogeneous=True, coefficient_ring=sp.FF(101))
    x0, x1, x2, x3, x4 = ideal_gen.ring.gens
    I = [x0**5*x1**3 + 10*x0*x2*x3*x4**5,
         x1**8*x2 + 29*x0**7*x1*x4,
         x0**4*x1**2*x2 + 74*x0**3*x1**2*x2*x3,
         x0**4*x3**3*x4**2 + 12*x0*x1**4*x3*x4**3,
         x1*x3**3 + 63*x1*x2*x3*x4]
    assert next(ideal_gen) == I
