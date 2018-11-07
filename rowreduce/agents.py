# agents.py
# Dylan Peifer
# 10 May 2018
"""Agents for reinforcemnt learning in computer algebra.

The structure of Q-learning involves an agent and an environment. The agent
gives the environment actions and gets from the environment states, rewards,
and if the current episode is done. The agent has a memory of state
transitions that it accumulates and trains on.
"""

import numpy as np
import random
from math import factorial
from itertools import combinations
from keras.models import Sequential, Model
from keras.layers import InputLayer, Flatten, Dense, Conv2D, Input, Concatenate, Reshape, GlobalAveragePooling1D, Activation, BatchNormalization, Lambda
import keras.backend as K


class Memory:
    """A cyclic buffer to store transition memories. Adapted from
    http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MatrixAgent:
    """A deep Q network agent adapted from
    https://keon.io/deep-q-learning/. Takes in a matrix.
    """

    def __init__(self, matrix_shape, action_size):
        self.matrix_shape = matrix_shape
        self.action_size = action_size
        self.memory = Memory(1000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.model = self._build_model()

    def remember(self, state, action, reward, next_state, done):
        """Add a state transition to memory."""
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        """Choose an action for the given state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        tensor = np.expand_dims(np.expand_dims(state, axis=2), axis=0)  # has shape (1,N,M,1)
        return np.argmax(self.model.predict(tensor)[0])

    def replay(self, batch_size):
        """Train on batch_size transitions from memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.expand_dims(np.array(states), axis=3)
        next_states = np.expand_dims(np.array(next_states), axis=3)
        dones = np.array(dones)

        targets = np.array(rewards)
        targets = targets + (dones == False) * self.gamma * \
                  np.max(self.model.predict(next_states), axis=1)
        targets_f = self.model.predict(states)
        targets_f[np.arange(targets_f.shape[0]), actions] = targets

        self.model.fit(states, targets_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.matrix_shape + (1,)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='logcosh', optimizer='adam')
        return model


class RowChoiceAgent:
    """A deep Q network agent adapted from
    https://keon.io/deep-q-learning/. Takes in a matrix and an
    array denoting which rows have not been used.
    """

    def __init__(self, matrix_shape, action_size):
        self.matrix_shape = matrix_shape
        self.action_size = action_size
        self.memory = Memory(1000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.model = self._build_model()

    def remember(self, state, action, reward, next_state, done):
        """Add a state transition to memory."""
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        """Choose an action for the given state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        rows = state[0]
        matrix = state[1]
        # make inputs have shape (1, N) and (1, N, M, 1)
        rows_batch = np.expand_dims(np.squeeze(rows, axis=1), axis=0)
        matrix_batch = np.expand_dims(np.expand_dims(matrix, axis=2), axis=0)
        return np.argmax(self.model.predict([matrix_batch, rows_batch])[0])

    def replay(self, batch_size):
        """Train on batch_size transitions from memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        rows, matrices = zip(*states)
        next_rows, next_matrices = zip(*next_states)
        rows_batch = np.squeeze(np.array(rows), axis=2)
        matrices_batch = np.expand_dims(np.array(matrices), axis=3)
        next_rows_batch = np.squeeze(np.array(next_rows), axis=2)
        next_matrices_batch = np.expand_dims(np.array(next_matrices), axis=3)
        dones = np.array(dones)

        targets = np.array(rewards)
        targets = targets + (dones == False) * self.gamma * \
                  np.max(self.model.predict([next_matrices_batch, next_rows_batch]), axis=1)
        targets_f = self.model.predict([matrices_batch, rows_batch])
        targets_f[np.arange(targets_f.shape[0]), actions] = targets

        self.model.fit([matrices_batch, rows_batch], targets_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def _build_model(self):
        matrix = Input(shape=self.matrix_shape + (1,))
        rows = Input(shape=(self.matrix_shape[0],))
        X = Conv2D(16, (5, 5), activation='relu')(matrix)
        X = Flatten()(X)
        X = Dense(self.matrix_shape[0], activation='relu')(X)
        X = Concatenate()([X, rows])
        output = Dense(self.action_size, activation='linear')(X)

        model = Model(inputs=[matrix, rows], outputs=output)
        model.compile(loss='logcosh', optimizer='adam')
        return model


def binom(n, m):
    """Return the value of (n choose m)."""
    return factorial(n) // factorial(m) // factorial(n - m)


def leading_submatrix(matrix, rows, k):
    """Return the matrix consisting of the first k nonzero columns in the
    given rows. Pad with zeros at end if necessary."""
    submatrix = np.zeros((len(rows), k), dtype=int)
    input_index = 0
    output_index = 0
    while input_index < matrix.shape[1] and output_index < k:
        col = matrix[rows, input_index]
        if np.any(col):
            submatrix[:, output_index] = col
            output_index += 1
        input_index += 1
    return submatrix


def state_tensor(matrix, k):
    """Return the k-way interactions state tensor for given matrix. If matrix
    is mxn then the return is m x (m-1 choose k-1) x k^2."""
    m, n = matrix.shape
    tensor = np.zeros((m, binom(m - 1, k - 1), k * k))
    for row in range(m):
        other_rows = [x for x in range(m) if x != row]
        i = 0
        for rows in combinations(other_rows, k - 1):
            lead = leading_submatrix(matrix, [row] + list(rows), k)
            tensor[row, i, :] = np.reshape(lead, (1, 1, -1))
            i += 1
    return tensor


class KInteractionsAgent:
    """A Q network agent that uses k-way interactions to decide which row of a
    matrix to choose as the pivot."""

    def __init__(self, k, rows):
        self.k = k
        self.action_size = rows
        self.memory = Memory(1000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self._build_model()

    def remember(self, state, action, reward, next_state, done):
        """Add a state transition to memory. Store the state tensor, not the
        matrix."""
        self.memory.push(state_tensor(state, self.k),
                         action, reward,
                         state_tensor(next_state, self.k),
                         done)

    def act(self, state):
        """Choose an action (row) for the given state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        tensor = np.expand_dims(state_tensor(state, self.k), axis=0)
        return np.argmax(self.model.predict(tensor)[0])

    def replay(self, batch_size, verbose=0):
        """Train on batch_size transitions from memory."""
        if len(self.memory) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        next_states = np.array(next_states)
        dones = np.array(dones)

        targets = np.array(rewards)
        targets = targets + (dones == False) * self.gamma * \
                  np.max(self.model.predict(next_states), axis=1)
        targets_f = self.model.predict(states)
        targets_f[np.arange(targets_f.shape[0]), actions] = targets

        self.model.fit(states, targets_f, epochs=1, verbose=verbose)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name, by_name=True)

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.action_size,
                                          binom(self.action_size - 1, self.k - 1),
                                          self.k * self.k)))

        # phi applies to each submatrix
        channels = self.k * self.k
        for i in range(5):
            model.add(Conv2D(channels * 2, (1, 1), use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            channels *= 2

        # accumulate submatrix information for each row
        model.add(Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3))))
        model.add(Reshape((binom(self.action_size-1, self.k-1), -1)))
        model.add(GlobalAveragePooling1D())
        model.add(Reshape((self.action_size, 1, -1)))

        # F applies to each row
        for i in range(5):
            model.add(Conv2D(channels // 2, (1, 1), use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            channels //= 2
        model.add(Conv2D(1, (1, 1), use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Reshape((self.action_size,)))
        
        model.compile(loss='logcosh', optimizer='adam')
        return model
