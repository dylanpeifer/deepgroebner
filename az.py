from collections import deque
import time
import random
import numpy as np
import tensorflow as tf

from deepgroebner.wrapped import CLeadMonomialsEnv as LeadMonomialsEnv
from deepgroebner.networks import ParallelMultilayerPerceptron, ParallelEmbeddingLayer


class RecurrentValueModel(tf.keras.Model):

    def __init__(self, units):
        super(RecurrentValueModel, self).__init__()
        self.embedding = ParallelEmbeddingLayer(units, [])
        self.rnn = tf.keras.layers.LSTM(units)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, batch):
        return self.dense(self.rnn(self.embedding(batch)))


class GlobalSumPooling1D(tf.keras.layers.Layer):

    def __init__(self):
        super(GlobalSumPooling1D, self).__init__()

    def call(self, batch, mask=None):
        if mask is not None:
            batch = batch * tf.cast(tf.expand_dims(mask, -1), tf.float32)
        return tf.reduce_sum(batch, axis=-2)


class PoolingValueModel(tf.keras.Model):

    def __init__(self, hidden_layers1, hidden_layers2, method='max'):
        super(PoolingValueModel, self).__init__()
        self.embedding = ParallelEmbeddingLayer(
            hidden_layers1[-1], hidden_layers1[:-1])
        if method == 'max':
            self.pooling = tf.keras.layers.GlobalMaxPooling1D()
        elif method == 'mean':
            self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        elif method == 'sum':
            self.pooling = GlobalSumPooling1D()
        else:
            raise ValueError('invalid method')
        self.hidden_layers = [tf.keras.layers.Dense(
            u, activation='relu') for u in hidden_layers2]
        self.final_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, batch):
        X = self.pooling(self.embedding(batch))
        for layer in self.hidden_layers:
            X = layer(X)
        return self.final_layer(X)


class AZWrapper:
    """A wrapper for LeadMonomialsEnv environments to interact with the AZAgent."""

    def __init__(self, env):
        self.env = env
        self.players = 1
        self.turn = 0
        self.state = None
        self.done = None
        self.actions = []

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.actions = list(range(len(self.state)))
        return self.state

    def step(self, action):
        self.state, reward, self.done, info = self.env.step(action)
        self.actions = list(range(len(self.state)))
        return self.state, reward, self.done, info

    def copy(self):
        copy = AZWrapper(self.env.copy())
        copy.state = self.state.copy()
        copy.done = self.done
        copy.actions = self.actions.copy()
        return copy


class AZTreeNode:
    """A tree node for AlphaZero tree search."""

    def __init__(self, parent, action, reward, env, logpi, value):
        self.parent = parent
        self.children = []
        self.action = action
        self.reward = reward
        self.env = env
        self.visits = 0
        self.logpi = logpi
        self.value = value


def az_ucb(c=np.sqrt(2)):
    """Return an upper confidence bound tree policy for AlphaZero tree search."""
    def policy(node):
        def value(child):
            prob = np.exp(node.logpi[child.action])
            return child.value[child.env.turn] + c * prob * np.sqrt(node.visits)/(1 + child.visits)
        return max(node.children, key=value)
    return policy


class AZBuffer:

    def __init__(self):
        self.states = []
        self.probs = []
        self.values = []

    def store(self, state, prob, value):
        self.states.append(state)
        self.probs.append(prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.probs.clear()
        self.values.clear()

    def get(self, batch_size=32, drop_remainder=False):
        """Return tf.data.Dataset of training data."""
        if self.states and self.states[0].ndim == 2:

            # filter out any states with only one action available
            indices = [i for i in range(
                len(self.states)) if self.states[i].shape[0] != 1]
            states = [self.states[i].astype(np.int32) for i in indices]
            probs = [self.probs[i].astype(np.float32) for i in indices]
            values = np.array([self.values[i]
                               for i in indices], dtype=np.float32)

            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_generator(lambda: states, tf.int32),
                tf.data.Dataset.from_generator(lambda: probs, tf.float32),
                tf.data.Dataset.from_tensor_slices(values),
            ))
            if batch_size is None:
                batch_size = len(states)
            padded_shapes = ([None, self.states[0].shape[1]],
                             [None, ], [None, ])
            padding_values = (tf.constant(-1, dtype=tf.int32),
                              tf.constant(0.0, dtype=tf.float32),
                              tf.constant(0.0, dtype=tf.float32))
            dataset = dataset.padded_batch(batch_size,
                                           padded_shapes=padded_shapes,
                                           padding_values=padding_values,
                                           drop_remainder=drop_remainder)
            return dataset

    def __len__(self):
        return len(self.states)


class AZAgent:
    """An AlphaZero tree search agent.

    Parameters
    ----------
    policy_network : network
        The network that maps states to policies.
    value_network : network
        The network that maps states to values.
    tree_policy : function
        A function which maps node to child node.
    timeout : float, optional
        The amount of time in seconds to search before choosing an action.

    """

    def __init__(self, policy_network, value_network, tree_policy=az_ucb(), timeout=1.0,
                 policy_lr=1e-3, policy_updates=40, value_lr=1e-3, value_updates=40):
        self.tree_policy = tree_policy
        self.timeout = timeout

        self.policy_model = policy_network
        self.policy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = tf.keras.losses.MSE
        self.value_optimizer = tf.keras.optimizers.Adam(lr=value_lr)
        self.value_updates = value_updates

    def act(self, env, root=None):
        """Return a chosen action for the env.

        Parameters
        ----------
        env : environment
            The current environment.
        root : AZTreeNode
            The root of the tree corresponding to env state if a subtree has already been explored.

        """
        if root is None:
            root = AZTreeNode(None, None, 0.0, env.copy(),
                              self.policy(env.state),
                              self.value(env.state))
        limit = time.time() + self.timeout
        counter = 0
        while time.time() < limit:
            leaf = self.expand(root)
            value = 0.0 if leaf.env.done else self.value(leaf.env.state)
            self.backup(leaf, value)
            counter += 1
        return max(root.children, key=lambda node: node.visits).action

    @tf.function
    def policy(self, state):
        """Return the predicted policy for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.

        """
        return self.policy_model(state[tf.newaxis])[0]

    @tf.function
    def value(self, state):
        """Return the predicted value for the given state using the value model.

        Parameters
        ----------
        state : np.array
            The state of the environment.

        """
        return self.value_model(state[tf.newaxis])[0]

    def run_episode(self, env, buffer=None):
        env.reset()
        root = AZTreeNode(None, None, 0.0, env.copy(),
                          self.policy(env.state),
                          self.value(env.state))
        total_reward = 0.0
        length = 0
        while not env.done:
            action = self.act(env, root=root)
            if buffer is not None:
                visits = np.array([child.visits for child in root.children])
                probs = (visits / np.sum(visits)).astype(np.float32)
                buffer.store(root.env.state, probs, root.value)
            _, reward, _, _ = env.step(action)
            root = root.children[action]
            root.parent = None
            total_reward += reward
            length += 1
        return total_reward, length

    def run_episodes(self, env, episodes=100, buffer=None):
        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}
        for i in range(episodes):
            R, L = self.run_episode(env, buffer=buffer)
            history['returns'][i] = R
            history['lengths'][i] = L
        return history

    def train(self, env, episodes=100, epochs=1):
        buffer = AZBuffer()
        history = {'mean_returns': np.zeros(epochs)}
        for epoch in range(epochs):
            run_history = self.run_episodes(
                env, episodes=episodes, buffer=buffer)
            dataset = buffer.get()
            self._fit_policy_model(dataset, epochs=self.policy_updates)
            self._fit_value_model(dataset, epochs=self.value_updates)
            history['mean_returns'] = np.mean(run_history['returns'])
            buffer.clear()
        return history

    def expand(self, node):
        """Return an unvisited or terminal leaf node following the tree policy.

        Before returning, this function performs all possible actions from the
        leaf node and adds new nodes for them to the tree as children of the
        leaf node.
        """
        while node.visits != 0 and len(node.children) > 0:
            node = self.tree_policy(node)
        if not node.env.done:
            for action in node.env.actions:
                env = node.env.copy()
                _, reward, _, _ = env.step(action)
                node.children.append(AZTreeNode(node, action, reward, env,
                                                self.policy(env.state),
                                                self.value(env.state)))
        return node

    def backup(self, node, value):
        """Backup the value from a new leaf node."""
        while node is not None:
            value += node.reward
            node.visits += 1
            node.value = (node.visits - 1)/node.visits * \
                node.value + value/node.visits
            node = node.parent

    def _fit_policy_model(self, dataset, epochs=1):
        """Fit value model using data from dataset."""
        history = {'loss': []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for states, probs, _ in dataset:
                batch_loss = self._fit_policy_model_step(states, probs)
                loss += batch_loss
                batches += 1
            history['loss'].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    @tf.function(experimental_relax_shapes=True)
    def _fit_policy_model_step(self, states, probs):
        """Fit value model on one batch of data."""
        with tf.GradientTape() as tape:
            logpis = self.policy_model(states, training=True)
            loss = tf.reduce_mean(self.policy_loss(probs, logpis))
        varis = self.policy_model.trainable_variables
        grads = tape.gradient(loss, varis)
        self.policy_optimizer.apply_gradients(zip(grads, varis))
        return loss

    def _fit_value_model(self, dataset, epochs=1):
        """Fit value model using data from dataset."""
        history = {'loss': []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for states, _, values in dataset:
                batch_loss = self._fit_value_model_step(states, values)
                loss += batch_loss
                batches += 1
            history['loss'].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    @tf.function(experimental_relax_shapes=True)
    def _fit_value_model_step(self, states, values):
        """Fit value model on one batch of data."""
        with tf.GradientTape() as tape:
            pred_values = self.value_model(states, training=True)
            loss = tf.reduce_mean(self.value_loss(values, pred_values))
        varis = self.value_model.trainable_variables
        grads = tape.gradient(loss, varis)
        self.value_optimizer.apply_gradients(zip(grads, varis))
        return loss
