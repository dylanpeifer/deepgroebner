"""Tests for agent networks."""

import numpy as np
import pytest
import tensorflow as tf

from deepgroebner.networks import *


def test_MultilayerPerceptron():
    tf.random.set_seed(123)
    mlp = MultilayerPerceptron(2, [128], final_activation='log_softmax')
    states = tf.random.uniform((64, 4))
    logprobs = mlp(states)
    assert logprobs.shape == [64, 2]
    assert np.allclose(np.sum(np.exp(logprobs), axis=-1), 1)
    actions = tf.random.categorical(logprobs, 1)
    assert actions.shape == [64, 1]


@pytest.mark.parametrize("hidden_layers", [[], [32], [10, 10]])
def test_ParallelEmbeddingLayer(hidden_layers):
    tf.random.set_seed(123)
    embed = ParallelEmbeddingLayer(12, hidden_layers)
    batch = tf.constant([
        [[0,  1],
         [3,  0],
         [-1, -1]],
        [[8,  5],
         [3,  3],
         [3,  5]],
        [[6,  7],
         [6,  8],
         [-1, -1]],
    ])
    X = embed(batch)
    mask = X._keras_mask
    assert X.shape == [3, 3, 12]
    assert mask is not None and mask.shape == [3, 3]
    assert np.allclose(mask, np.array(
        [[True, True, False], [True, True, True], [True, True, False]]))


@pytest.mark.parametrize("hidden_layers", [[], [32], [10, 10]])
def test_DenseProcessingLayer(hidden_layers):
    tf.random.set_seed(123)
    embed = ParallelEmbeddingLayer(12, hidden_layers)
    process = DenseProcessingLayer(32, hidden_layers)
    batch = tf.constant([
        [[0,  1],
         [3,  0],
         [-1, -1]],
        [[8,  5],
         [3,  3],
         [3,  5]],
        [[6,  7],
         [6,  8],
         [-1, -1]],
    ])
    X = embed(batch)
    X = process(X)
    mask = X._keras_mask
    assert X.shape == [3, 3, 32]
    assert mask is not None and mask.shape == [3, 3]
    assert np.allclose(mask, np.array(
        [[True, True, False], [True, True, True], [True, True, False]]))


@pytest.mark.parametrize("hidden_layers", [[], [32], [10, 10]])
def test_ParallelDecidingLayer(hidden_layers):
    tf.random.set_seed(123)
    embed = ParallelEmbeddingLayer(12, hidden_layers)
    process = DenseProcessingLayer(32, hidden_layers)
    decide = ParallelDecidingLayer(hidden_layers)
    batch = tf.constant([
        [[0,  1],
         [3,  0],
         [-1, -1]],
        [[8,  5],
         [3,  3],
         [3,  5]],
        [[6,  7],
         [-1, -1],
         [-1, -1]],
    ])
    X = embed(batch)
    X = process(X)
    X = decide(X)
    assert X.shape == [3, 3]
    assert np.allclose(np.sum(np.exp(X), axis=1), [1., 1., 1.])
    assert np.allclose(np.exp(X)[[0, 2, 2], [2, 1, 1]], [0., 0., 0.])


@pytest.mark.parametrize("gam, states, values", [
    (1.0, np.zeros((10, 5, 6)), np.full((10, 1), -5)),
    (0.9, np.zeros((10, 5, 6)), np.full((10, 1), -4.0951)),
])
def test_PairsLeftBaseline(gam, states, values):
    value = PairsLeftBaseline(gam=gam)
    assert np.allclose(value.predict(states), values)
