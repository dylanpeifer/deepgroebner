# test_pg.py
# Dylan Peifer
# 07 May 2019
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
def test_discount_rewards_0(r, gamma, r_):
    assert np.array_equal(discount_rewards(r, gamma), r_)


def test_discount_rewards_1():
    L = [1, 2, 3, 4, 5]
    L[2:] = discount_rewards(L[2:], 0.5)
    assert L == [1, 2, 6.25, 6.5, 5]


def test_discount_rewards_2():
    L = np.array([1., 2., 3., 4., 5.])
    discount_rewards(L[2:], 0.5)
    assert np.array_equal(L, np.array([1, 2, 6.25, 6.5, 5]))


@pytest.mark.parametrize("gam, lam, val, adv", [
    (1.0, 1.0, np.array([[5.], [4.], [3.], [2.], [1.]]),
     np.array([[0., 5., 0.],
               [4., 0., 0.],
               [0., 3., 0.],
               [0., 0., 2.],
               [1., 0., 0.]])),
    (0.5, 1.0, np.array([[1.9375], [1.875], [1.75], [1.5], [1.]]),
     np.array([[0., 1.9375, 0.],
               [1.875, 0., 0.],
               [0., 1.75, 0.],
               [0., 0., 1.5],
               [1., 0., 0.]])),
    (1.0, 0.5, np.array([[5.], [4.], [3.], [2.], [1.]]),
     np.array([[0., 1.9375, 0.],
               [1.875, 0., 0.],
               [0., 1.75, 0.],
               [0., 0., 1.5],
               [1., 0., 0.]])),
    (0.5, 0.5, np.array([[1.9375], [1.875], [1.75], [1.5], [1.]]),
     np.array([[0., 1.33203125, 0.],
               [1.328125, 0., 0.],
               [0., 1.3125, 0.],
               [0., 0., 1.25],
               [1., 0., 0.]])),
])
def test_TrajectoryBuffer_0(gam, lam, val, adv):
    buf = TrajectoryBuffer(gam, lam)
    tau = [(np.array([1, 2]), 1, 1, 0),
           (np.array([1, 3]), 0, 1, 0),
           (np.array([1, 4]), 1, 1, 0),
           (np.array([1, 5]), 2, 1, 0),
           (np.array([1, 7]), 0, 1, 0)]
    for t in tau:
        buf.store(*t)
    buf.finish()
    batches = buf.getBatches(lambda s: 3, normalize=False)
    s, v, a = batches[(2,)]
    assert np.array_equal(s, np.array([[1., 2.], [1., 3.], [1., 4.], [1., 5.], [1., 7.]]))
    assert np.array_equal(v, val)
    assert np.array_equal(a, adv)
