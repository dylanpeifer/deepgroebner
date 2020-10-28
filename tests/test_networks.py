"""Tests for agent networks."""

import numpy as np
import pytest

from deepgroebner.networks import *


def test_MultilayerPerceptron_0():
    mlp = MultilayerPerceptron(2, [128], final_activation='log_softmax')
    states = tf.random.uniform((64, 4))
    logprobs = mlp(states)
    assert logprobs.shape == [64, 2]
    assert np.allclose(np.sum(np.exp(logprobs), axis=-1), 1)
    actions = tf.random.categorical(logprobs, 1)
    assert actions.shape == [64, 1]


def test_ParallelMultilayerPerceptron():
    policy = ParallelMultilayerPerceptron(6, [24])
    np.random.seed(123)
    X = np.random.randn(10, 15, 6)
    assert np.allclose(policy.predict(X), policy.network.predict(X)[0])


@pytest.mark.parametrize("gam, states, values", [
    (1.0, np.zeros((10, 5, 6)), np.full((10, 1), -5)),
    (0.9, np.zeros((10, 5, 6)), np.full((10, 1), -4.0951)),
])
def test_PairsLeftBaseline(gam, states, values):
    value = PairsLeftBaseline(gam=gam)
    assert np.allclose(value.predict(states), values)
    