# test_pg.py
# Dylan Peifer
# 06 May 2019
"""Tests for policy gradient agent."""

import numpy as np
import pytest

from agents.pg import *


@pytest.mark.parametrize("r, gamma, r_", [
    ([], 0.9, []),
    ([1, 2, 3], 1, [6, 5, 3]),
    ([1, 1, 1, 1], 0.9, [3.439, 2.71, 1.9, 1.]),
    (np.array([]), 0.9, np.array([])),
    (np.array([1, 2, 3]), 1, np.array([6, 5, 3])),
    (np.array([1., 1., 1., 1.]), 0.9, np.array([3.439, 2.71, 1.9, 1.])),
])
def test_discount_rewards(r, gamma, r_):
    assert np.array_equal(discount_rewards(r, gamma), r_)
