"""Tests for agent networks."""

import numpy as np
import pytest

from deepgroebner.networks import *


def test_MultilayerPerceptron_0():
    tf.random.set_seed(123)
    mlp = MultilayerPerceptron(2, [128], final_activation='log_softmax')
    states = tf.random.uniform((64, 4))
    logprobs = mlp(states)
    assert logprobs.shape == [64, 2]
    assert np.allclose(np.sum(np.exp(logprobs), axis=-1), 1)
    actions = tf.random.categorical(logprobs, 1)
    assert actions.shape == [64, 1]


def test_ParallelMultilayerPerceptron():
    tf.random.set_seed(123)
    pmlp = ParallelMultilayerPerceptron([128])
    states = tf.constant([
        [[ 0,  1],
         [ 3,  0],
         [-1, -1]],
        [[ 8,  5],
         [ 3,  3],
         [ 3,  5]],
        [[ 6,  7],
         [ 6,  8],
         [-1, -1]],
    ])
    logprobs = pmlp(states)
    assert logprobs.shape == [3, 3]
    assert np.allclose(np.sum(np.exp(logprobs), axis=-1), 1)
    assert np.isclose(np.exp(logprobs)[0, 2], 0) and np.isclose(np.exp(logprobs)[2, 2], 0)
    actions = tf.random.categorical(logprobs, 1)
    assert actions.shape == [3, 1]


@pytest.mark.parametrize("gam, states, values", [
    (1.0, np.zeros((10, 5, 6)), np.full((10, 1), -5)),
    (0.9, np.zeros((10, 5, 6)), np.full((10, 1), -4.0951)),
])
def test_PairsLeftBaseline(gam, states, values):
    value = PairsLeftBaseline(gam=gam)
    assert np.allclose(value.predict(states), values)
    