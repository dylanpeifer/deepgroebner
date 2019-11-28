"""Tests for the proximal policy optimization agent.

"""

import numpy as np
import pytest

from agents.ppo import *


@pytest.mark.parametrize("r, gam, r_", [
    ([], 0.9, []),
    ([1, 2, 3], 1, [6, 5, 3]),
    ([1, 1, 1, 1], 0.9, [3.439, 2.71, 1.9, 1.]),
    (np.array([]), 0.9, np.array([])),
    (np.array([1, 2, 3]), 1, np.array([6, 5, 3])),
    (np.array([1., 1., 1., 1.]), 0.9, np.array([3.439, 2.71, 1.9, 1.])),
])
def test_discount_rewards_0(r, gam, r_):
    assert np.array_equal(discount_rewards(r, gam), r_)


def test_discount_rewards_1():
    L = [1, 2, 3, 4, 5]
    L[2:] = discount_rewards(L[2:], 0.5)
    assert L == [1, 2, 6.25, 6.5, 5]


def test_discount_rewards_2():
    L = np.array([1., 2., 3., 4., 5.])
    discount_rewards(L[2:], 0.5)
    assert np.array_equal(L, np.array([1, 2, 6.25, 6.5, 5]))
