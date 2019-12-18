"""Tests for agent networks."""

import numpy as np
import pytest

from agents.networks import *


def test_MultilayerPerceptron_0():
    policy = MultilayerPerceptron(4, [128], 2, final_activation='softmax')
    X = np.random.randn(10, 4)
    assert np.allclose(policy.predict(X), policy.network.predict(X))


def test_MultilayerPerceptron_1():
    policy = MultilayerPerceptron(4, [128], 2, final_activation='linear')
    X = np.random.randn(10, 4)
    assert np.allclose(policy.predict(X), policy.network.predict(X))


def test_ParallelMultilayerPerceptron():
    policy = ParallelMultilayerPerceptron(6, [24])
    X = np.random.randn(10, 15, 6)
    assert np.allclose(policy.predict(X), policy.network.predict(X))


@pytest.mark.parametrize("gam, states, values", [
    (1.0, np.zeros((10, 5, 6)), np.full((10, 1), -5)),
    (0.9, np.zeros((10, 5, 6)), np.full((10, 1), -4.0951)),
])
def test_PairsLeftBaseline(gam, states, values):
    value = PairsLeftBaseline(gam=gam)
    assert np.allclose(value.predict(states), values)
    