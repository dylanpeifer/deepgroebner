# networks.py
# Dylan Peifer
# 12 May 2019
"""Neural networks for agents."""

import numpy as np
from scipy.special import softmax
import tensorflow as tf


def MultilayerPerceptron(input_dim, hidden_layers, output_dim, activation='relu', final_activation='softmax'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for hidden in hidden_layers:
        model.add(tf.keras.layers.Dense(hidden, activation=activation))
    model.add(tf.keras.layers.Dense(output_dim, activation=final_activation))
    return model


class ParallelMultilayerPerceptron():

    def __init__(self, input_dim, hidden_layers):
        self.network = self._build_network(input_dim, hidden_layers)
        self.weights = self.get_weights()

    def predict(self, X, **kwargs):
        for i, (m, b) in enumerate(self.weights):
            X = np.dot(X, m) + b
            if i == len(self.weights)-1:
                X = softmax(X, axis=1).squeeze(axis=-1)
            else:
                X = np.maximum(X, 0, X)
        return X

    def fit(self, X, y, **kwargs):
        self.network.fit(X, y, **kwargs)
        self.weights = self.get_weights()

    def compile(self, **kwargs):
        self.network.compile(**kwargs)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)
        self.weights = self.get_weights()

    def get_weights(self):
        network_weights = self.network.get_weights()
        weights = []
        for i in range(len(network_weights)//2):
            m = network_weights[2*i].squeeze(axis=0)
            b = network_weights[2*i + 1]
            weights.append((m, b))
        return weights

    def _build_network(self, input_dim, hidden_layers):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(None, input_dim)))
        for hidden in hidden_layers:
            model.add(tf.keras.layers.Conv1D(hidden, 1, activation='relu'))
        model.add(tf.keras.layers.Conv1D(1, 1, activation='linear'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Activation('softmax'))
        return model


class PairsLeft():
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


class AgentBaseline():
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


def ValueRNN(input_dim, size, cell='lstm', final_activation='linear', gpu=False):
    model = tf.keras.models.Sequential()
    if gpu:
        if cell == 'lstm':
            model.add(tf.keras.layers.CuDNNLSTM(size, input_shape=(None, input_dim)))
        elif cell == 'gru':
            model.add(tf.keras.layers.CuDNNGRU(size, input_shape=(None, input_dim)))
    else:
        if cell == 'lstm':
            model.add(tf.keras.layers.LSTM(size, input_shape=(None, input_dim)))
        elif cell == 'gru':
            model.add(tf.keras.layers.GRU(size, input_shape=(None, input_dim)))
    model.add(tf.keras.layers.Dense(1, activation=final_activation))
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
