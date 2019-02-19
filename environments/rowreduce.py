# rowreduce.py
# Dylan Peifer
# 18 Feb 2019
"""An environment for matrix row reduction."""

from itertools import combinations
from math import factorial
import numpy as np


class RowEchelonEnv:
    """A simple environment for matrix row reduction. Agents can add or swap
    rows, and the environment is done when the matrix is in row echelon form.
    """

    def __init__(self, shape, modulus):
        self.N = shape[0]
        self.M = shape[1]
        self.F = modulus
        self.matrix = np.zeros((self.N, self.M))
        self.action_tuples = [('swap', i, j) for i in range(self.N) for j in range(i)] \
            + [('add', i, j) for i in range(self.N) for j in range(self.N) if i != j]
        self.action_size = len(self.action_tuples)

    def reset(self):
        """Reset the state of the environment to a matrix that is not row
        reduced.
        """
        self.matrix = self._random_matrix()
        while self._is_row_echelon():
            self.matrix = self._random_matrix()
        return np.copy(self.matrix)

    def step(self, action):
        """Perform a step from current state using action."""
        action = self.action_tuples[action]
        if action[0] == 'add':
            self._add_rows(action[1:])
        else:
            self._swap_rows(action[1:])
        return np.copy(self.matrix), -1, self._is_row_echelon(), {}

    def _add_rows(self, pair):
        """Add the rows given by pair."""
        self.matrix[pair[1], :] = (self.matrix[pair[1], :] + self.matrix[pair[0], :]) % self.F

    def _swap_rows(self, pair):
        """Swap the rows given by pair."""
        self.matrix[pair, :] = self.matrix[(pair[1], pair[0]), :]

    def _is_row_echelon(self):
        """Return true if the matrix is in row echelon form."""
        prev_lead = -1
        for row in range(self.N):
            next_lead = next((i for i, x in enumerate(self.matrix[row,:]) if x != 0), None)
            if next_lead is not None:
                if prev_lead is None or prev_lead >= next_lead:
                    return False
            prev_lead = next_lead
        return True

    def _random_matrix(self):
        """Return a new random matrix."""
        return np.random.randint(self.F, size=(self.N, self.M))


class RowChoiceEnv:
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
                    -100,
                    self._is_reduced(),
                    {})
        moves = 0
        for i in range(self.N):
            if i != action and self.matrix[i, lead] != 0:
                self.matrix[i, :] = (self.matrix[i, :] + self.matrix[action, :]) % 2
                moves += 1
        if moves == 0:
            return (np.copy(self.matrix),
                    -100,
                    self._is_reduced(),
                    {})
        else:
            return (np.copy(self.matrix),
                    -moves,
                    self._is_reduced(),
                    {})

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


class KInteractionsEnv(RowChoiceEnv):

    def __init__(self, k, shape, density):
        RowChoiceEnv.__init__(self, shape, density)
        self.k = k

    def reset(self):
        state = RowChoiceEnv.reset(self)
        return state_tensor(state, self.k)

    def step(self, action):
        state, reward, done, info = RowChoiceEnv.step(self, action)
        return state_tensor(state, self.k), reward, done, info
