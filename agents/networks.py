"""Neural networks for agents.

The two network classes are designed to be fast wrappers around tf.keras models.
In particular, they store their weights in NumPy arrays and do predict calls in
pure NumPy, which in testing is at least on order of magnitude faster than
TensorFlow when called repeatedly.
"""

import numpy as np
from scipy.special import softmax
import tensorflow as tf


class MultilayerPerceptron:
    """A multilayer perceptron network with fast predict calls."""

    def __init__(self, input_dim, hidden_layers, output_dim, final_activation='softmax'):
        self.network = self._build_network(input_dim, hidden_layers, output_dim, final_activation)
        self.weights = self.get_weights()
        self.trainable_variables = self.network.trainable_variables
        self.final_activation = final_activation

    def predict(self, X, **kwargs):
        for i, (m, b) in enumerate(self.weights):
            X = np.dot(X, m) + b
            if i == len(self.weights)-1:
                if self.final_activation == 'softmax':
                    X = softmax(X, axis=1)
            else:
                X = np.maximum(X, 0, X)
        return X

    def __call__(self, inputs):
        return self.network(inputs)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)
        self.weights = self.get_weights()

    def get_weights(self):
        network_weights = self.network.get_weights()
        self.weights = []
        for i in range(len(network_weights)//2):
            m = network_weights[2*i]
            b = network_weights[2*i + 1]
            self.weights.append((m, b))
        return self.weights

    def _build_network(self, input_dim, hidden_layers, output_dim, final_activation):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
        for hidden in hidden_layers:
            model.add(tf.keras.layers.Dense(hidden, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation=final_activation))
        return model


class ParallelMultilayerPerceptron:
    """A parallel multilayer perceptron network with fast predict calls."""

    def __init__(self, input_dim, hidden_layers):
        self.network = self._build_network(input_dim, hidden_layers)
        self.weights = self.get_weights()
        self.trainable_variables = self.network.trainable_variables

    def predict(self, X, **kwargs):
        for i, (m, b) in enumerate(self.weights):
            X = np.dot(X, m) + b
            if i == len(self.weights)-1:
                X = softmax(X, axis=1).squeeze(axis=-1)
            else:
                X = np.maximum(X, 0, X)
        return X

    def __call__(self, inputs):
        return self.network(inputs)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)
        self.weights = self.get_weights()

    def get_weights(self):
        network_weights = self.network.get_weights()
        self.weights = []
        for i in range(len(network_weights)//2):
            m = network_weights[2*i].squeeze(axis=0)
            b = network_weights[2*i + 1]
            self.weights.append((m, b))
        return self.weights

    def _build_network(self, input_dim, hidden_layers):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(None, input_dim)))
        for hidden in hidden_layers:
            model.add(tf.keras.layers.Conv1D(hidden, 1, activation='relu'))
        model.add(tf.keras.layers.Conv1D(1, 1, activation='linear'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Activation('softmax'))
        return model


class PairsLeft:
    """A Buchberger value network that returns discounted pairs left."""

    def __init__(self, gam=0.99):
        self.gam = gam

    def predict(self, tensor):
        states = tensor.shape[0]
        pairs = tensor.shape[1]
        if self.gam == 1:
            fill_value = - pairs
        else:
            fill_value = - (1 - self.gam ** pairs) / (1 - self.gam)
        return np.full((states, 1), fill_value)

    def fit(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        pass

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass


class PairsLeftBaseline(tf.keras.Model):
    """A value network for BuchbergerEnv."""

    def __init__(self, gam=0.99, **kwargs):
        super().__init__(**kwargs)
        self.gam = gam

    def call(self, inputs):
        states = tf.shape(inputs)[0]
        pairs = tf.cast(tf.shape(inputs)[1], tf.float32)
        if self.gam == 1:
            value = -pairs
        else:
            value = -(1 - self.gam ** pairs) / (1 - self.gam)
        return tf.fill([states, 1], value)


class AgentBaseline:
    """A Buchberger value network that returns an agent's performance."""

    def __init__(self, agent, gam=0.99):
        self.agent = agent
        self.gam = gam

    def predict(self, env):
        env = env.copy()
        R = 0.0
        discount = 1.0
        state = (env.G, env.P)
        done = False
        while not done:
            action = self.agent.act(state)
            state, reward, done, _ = env.step(action)
            R += reward * discount
            discount *= self.gam
        return R

    def fit(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        pass

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass


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
