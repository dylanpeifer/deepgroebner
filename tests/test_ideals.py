# test_ideals.py
# Dylan Peifer
# 24 Sep 2019
"""Tests for ideal generators."""

import pytest
import sympy as sp

from environments.ideals import *


def test_FixedIdealGenerator_0():
    R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    assert [y - x**2, z - x**3] == next(ideal_gen)


def test_FixedIdealGenerator_1():
    R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    G = next(ideal_gen)
    G[0] = x
    assert [y - x**2, z - x**3] == next(ideal_gen)


def test_FixedIdealGenerator_2():
    R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    F[0] = x
    assert [y - x**2, z - x**3] == next(ideal_gen)
