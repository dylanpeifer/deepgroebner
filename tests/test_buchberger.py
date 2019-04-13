# test_buchberger.py
# Dylan Peifer
# 12 Apr 2019
"""Tests for Buchberger environments."""

import numpy as np
import sympy as sp
import pytest

from environments.buchberger import *


x, y, z = sp.symbols('x y z')


@pytest.mark.parametrize("f,lf,order", [
    (sp.poly(x**2 + x*y**2), sp.poly(x*y**2, x, y), 'grevlex'),
    (sp.poly(x**2 + x*y**2), sp.poly(x**2, x, y), 'lex'),
    (sp.poly(x**2, modulus=32003), sp.poly(x**2, modulus=32003), 'grevlex'),
])
def test_lm(f, lf, order):
    assert lm(f, order) == lf


@pytest.mark.parametrize("f,lf,order", [
    (sp.poly(x**2 + 2*x*y**2), sp.poly(2*x*y**2, x, y), 'grevlex'),
    (sp.poly(x**2 + x*y**2), sp.poly(x**2, x, y), 'lex'),
    (sp.poly(-x**2, modulus=32003), sp.poly(-x**2, modulus=32003), 'grevlex'),
])
def test_lt(f, lf, order):
    assert lt(f, order) == lf


def test_buchberger():
    F = [y-x**2, z-x**3]
    G = [sp.poly(x**2 - y, x, y, z, modulus=32003),
         sp.poly(x*y - z, x, y, z, modulus=32003),
         sp.poly(-x*z + y**2, x, y, z, modulus=32003)]
    assert buchberger(F, [x, y, z], sp.FF(32003), 'grevlex') == G
