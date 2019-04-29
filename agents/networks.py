# networks.py
# Dylan Peifer
# 29 Apr 2019
"""Neural networks for agents."""

from math import factorial
import tensorflow as tf


def MultilayerPerceptron(input_dim, hidden_layers, output_dim, activation='relu', final_activation='softmax'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for hidden in hidden_layers:
        model.add(tf.keras.layers.Dense(hidden, activation=activation))
    model.add(tf.keras.layers.Dense(output_dim, activation=final_activation))
    return model


def ParallelMultilayerPerceptron(input_dim, hidden_layers, activation='relu', final_activation='softmax'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(None, 1, input_dim)))
    for hidden in hidden_layers:
        model.add(tf.keras.layers.Conv2D(hidden, 1, activation=activation))
    model.add(tf.keras.layers.Conv2D(1, 1, activation='linear'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Activation(final_activation))
    return model


def CountTensor(input_shape, i):
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.fill((tf.shape(x)[0], 1), tf.cast(tf.shape(x)[i], tf.float32)),
                               input_shape=input_shape)
    ])
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
