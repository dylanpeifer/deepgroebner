# test_networks.py
# Dylan Peifer
# 12 May 2019
"""Tests for agent networks."""

import numpy as np
import pytest

from agents.networks import *


def test_ParallelMultilayerPerceptron():
    policy = ParallelMultilayerPerceptron(6, [24])
    X = np.random.randn(10, 15, 6)
    assert np.allclose(policy.predict(X), policy.network.predict(X))


@pytest.mark.parametrize("gam, states, values", [
    (1.0, np.zeros((10, 5, 6)), np.full((10, 1), -5)),
    (0.9, np.zeros((10, 5, 6)), np.full((10, 1), -4.0951)),
])
def test_PairsLeft(gam, states, values):
    value = PairsLeft(gam=gam)
    assert np.allclose(value.predict(states), values)
    