# kinteractions.py
# Dylan Peifer
# 17 Sep 2018
"""Attempt at Q-function composed of k-way interactions of rows.

Here the Q-function is of the form

    Q(state, action) = \sum_{i=1}^\infty f_k(state, action)

where f_k contains all size k subsets of the rows containing row action.
"""

import numpy as np
import random
from math import factorial
from itertools import combinations
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Reshape, Lambda, GlobalAveragePooling1D
import keras.backend as K
from statistics import mean, median


class RowChoiceEnvironment:
    """An environment for matrix row reduction over F2. Agents choose a row
    to use, and this row is then used as a pivot.
    """

    def __init__(self, shape, density):
        self.N = shape[0]
        self.M = shape[1]
        self.density = density
        self.matrix = np.zeros((self.N, self.M))
        self.action_size = self.N

    def reset(self):
        """Reset the state of the environment to a matrix that is not
        reduced.
        """
        self.matrix = self._random_matrix()
        while self._is_reduced():
            self.matrix = self._random_matrix()
        return np.copy(self.matrix)

    def step(self, action):
        """Perform a step from current state using action."""
        lead = next((i for i, x in enumerate(self.matrix[action, :]) if x != 0), None)
        if lead is None:
            return (np.copy(self.matrix),
                    -1,
                    self._is_reduced())
        moves = 0
        for i in range(self.N):
            if i != action and self.matrix[i, lead] != 0:
                self.matrix[i, :] = (self.matrix[i, :] + self.matrix[action, :]) % 2
                moves += 1
        return (np.copy(self.matrix),
                -1 - moves,
                self._is_reduced())

    def _is_reduced(self):
        """Return true if the current matrix is reduced."""
        for row in range(self.N):
            # find index of lead term in this row
            lead = next((i for i, x in enumerate(self.matrix[row, :]) if x != 0), None)
            # if this row has lead then everything in lead's column must be 0
            if lead is not None:
                for i in range(self.N):
                    if i != row and self.matrix[i, lead] != 0:
                        return False
        return True

    def _random_matrix(self):
        """Return a new random matrix."""
        return 1 * (np.random.rand(self.N, self.M) > (1 - self.density))


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
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
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

    def replay(self, batch_size):
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

        self.model.fit(states, targets_f, epochs=1, verbose=0)

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

        channels = self.k * self.k
        while channels // 2 > 0:
            model.add(Conv2D(channels // 2, (1, 1), activation='relu'))
            channels = channels // 2

        model.add(Reshape((self.action_size, -1)))
        model.add(Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1))))
        model.add(GlobalAveragePooling1D())

        model.add(Lambda(
            lambda x: x * self.action_size * self.action_size
            / binom(self.action_size - 1, self.k - 1)))

        model.compile(loss='logcosh', optimizer='adam')
        return model


env = RowChoiceEnvironment((5, 10), 0.5)
agent = KInteractionsAgent(2, 5)


def train(episodes, batch_size):
    """Train the agent on env for given episodes and using batch_size replays
    per episode.
    """
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        agent.replay(batch_size)


def test(episodes):
    """Test the agent and get average reward over given episodes."""
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        r = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            r += reward
        rewards.append(r)
    return min(rewards), median(rewards), max(rewards), mean(rewards)


if __name__ == "__main__":
    for i in range(100):
        train(100, 1024)
        result = test(100)
        print(i, ": ", result)
