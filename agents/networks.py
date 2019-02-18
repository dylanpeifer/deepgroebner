# networks.py
# Dylan Peifer
# 18 Feb 2019
"""Networks for agents."""

from math import factorial
import tensorflow as tf


def MultilayerPerceptron(input_dim, hidden_layers, output_dim, activation='relu', final_activation='softmax', dropout=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for hidden in hidden_layers:
        model.add(tf.keras.layers.Dense(hidden, activation=activation))
        if dropout is not None:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_dim, activation=final_activation))
    return model


def AtariNetSmall(input_shape, action_size, final_activation='linear'):
    """Return the network from the first DQN paper."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: x / 255.0))
    model.add(tf.keras.layers.Conv2D(16, 8, strides=4, activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size, activation=final_activation))
    return model


def AtariNetLarge(input_shape, action_size, final_activation='linear'):
    """Return the network from the second DQN paper."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: x / 255.0))
    model.add(tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size, activation=final_activation))
    return model


def binom(n, m):
    """Return the value of (n choose m)."""
    return factorial(n) // factorial(m) // factorial(n - m)


def KInteractions(k, m):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(m, binom(m-1, k-1), k*k)))

    # phi applies to each submatrix
    channels = k * k
    for i in range(5):
        model.add(tf.keras.layers.Conv2D(2 * channels, 1, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        channels *= 2

    # accumulate submatrix information for each row
    model.add(tf.keras.layers.Permute((0, 2, 1, 3)))
    model.add(tf.keras.layers.Reshape((binom(m-1, k-1), -1)))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Reshape((m, 1, -1)))

    # F applies to each row
    for i in range(5):
        model.add(tf.keras.layers.Conv2D(channels // 2, 1, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        channels //= 2
    model.add(tf.keras.layers.Conv2D(1, 1, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('linear'))

    model.add(tf.keras.layers.Reshape((m,)))

    return model
