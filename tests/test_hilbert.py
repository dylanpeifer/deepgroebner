# test_hilbert.py
# Dylan Peifer
# 12 Apr 2019
"""Tests for Hilbert environments."""

import numpy as np
import pytest

from environments.hilbert import *


def test_pivot():
    p = np.array([1, 2, 5])
    A = np.array([[3, 7, 2],
                  [1, 3, 6],
                  [8, 2, 6],
                  [3, 5, 3],
                  [0, 4, 8]])
    l = np.array([[3, 7, 2],
                  [3, 5, 3],
                  [0, 4, 8],
                  [1, 2, 5]])
    r = np.array([[0, 1, 1],
                  [7, 0, 1],
                  [2, 3, 0]])
    lp, rp = pivot(A, p)
    assert np.array_equal(lp, l)
    assert np.array_equal(rp, r)

    
def test_minimalize():
    A = np.array([[2, 5, 0],
                  [0, 1, 1],
                  [7, 0, 1],
                  [2, 3, 2],
                  [0, 2, 3]])
    B = np.array([[2, 5, 0],
                  [0, 1, 1],
                  [7, 0, 1]])
    assert np.array_equal(minimalize(A), B)


@pytest.mark.parametrize("A,done", [
    (np.eye(4), True),
    (np.array([[1, 0, 0], [1, 2, 5], [0, 1, 0]]), True),
    (np.array([[2, 5, 0], [0, 1, 1], [7, 0, 1]]), False),
])
def test_terminal(A, done):
    assert terminal(A) == done


def test_strategy():
    B = np.array([[2, 5, 0],
                  [0, 1, 1],
                  [7, 0, 1],
                  [2, 3, 2],
                  [0, 2, 3]])
    assert strategy(B) == 1


def test_base_cases():
    A = np.array([[3, 7, 2],
                  [1, 3, 6],
                  [8, 2, 6],
                  [3, 5, 3],
                  [0, 4, 8]])
    assert base_cases(A) == 9
