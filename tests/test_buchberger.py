# test_buchberger.py
# Dylan Peifer
# 31 Mar 2019
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
