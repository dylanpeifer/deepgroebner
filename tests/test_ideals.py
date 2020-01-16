"""Tests for ideal generators."""

import pytest
import sympy as sp

from deepgroebner.ideals import *


def test_cyclic():
    F = cyclic(3)
    x, y, z = F[0].ring.gens
    assert F == [x + y + z, x*y + y*z + z*x, x*y*z - 1]


def test_FixedIdealGenerator_0():
    R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    assert [y - x**2, z - x**3] == next(ideal_gen)


def test_FixedIdealGenerator_1():
    """Check that generated ideals are distinct objects from each other."""
    R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    G = next(ideal_gen)
    G[0] = x
    assert [y - x**2, z - x**3] == next(ideal_gen)


def test_FixedIdealGenerator_2():
    """Check that generated ideals are distinct objects from input."""
    R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    F[0] = x
    assert [y - x**2, z - x**3] == next(ideal_gen)
