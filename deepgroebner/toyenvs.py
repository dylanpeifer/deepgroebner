import numpy as np


class DumbTicTacToeEnv():
    def __init__(self, dim):
        self.state = np.zeros((dim, dim))
        self.chosen = dict()
        self.actions = [i for i in range(dim)]
        self.done = False
        self.dim = dim
        self.players = 1
        self.turn = 0

    def check_status(self):
        """
        Returns:
        True if the game is won,
        False if the game hasn't ended.
        """

        dim = self.dim
        st = self.state

        # check rows, cols
        for i in range(dim):
            counter_r = 0
            counter_c = 0
            for j in range(dim):
                if st[i][j] == 1:
                    counter_r += 1
                if st[j][i] == 1:
                    counter_c += 1
            if counter_r == dim:
                return True
            if counter_c == dim:
                return True
            counter_r = 0
            counter_c = 0

        # check diags
        counter_d1 = 0
        for k in range(dim):
            if st[k][k] == 1:
                counter_d1 += 1
        if counter_d1 == dim:
            return True

        counter_d2 = 0
        for l in range(dim):
            if st[l][dim - 1 - l] == 1:
                counter_d2 += 1
        if counter_d2 == dim:
            return True
        return False

    def step(self, pos):
        """
        Precondition: pos must be a tuple (m, n) where m, n < dim

        Returns:
        updated state (np.array)
        reward (int): -1
        done (boolean)
        empty dictionary
        """
        reward = -1
        if pos in self.chosen:
            self.chosen[pos] += 1
        else:
            self.chosen[pos] = 1
        if self.chosen[pos] >= 10:
            return self.state, -99999, True, dict()
        self.state[pos % self.dim][pos // self.dim] = 1
#         self.actions = [(i, j) for i in range(self.dim)
#                         for j in range(self.dim) if self.state[i][j] != 1]
        self.done = self.check_status()
        return self.state, reward, self.done, dict()

    def to_string(self):
        return np.array_str(self.state) + "\nDimension: " + str(self.dim)

    def reset(self):
        self.state = np.zeros((self.dim, self.dim))
        self.done = False
        self.actions = [i for i in range(self.dim)]
        self.chosen = dict()
        return self.state

    def copy(self):
        copy = DumbTicTacToeEnv(self.dim)
        copy.state = self.state.copy()
        copy.done = self.done
        copy.actions = self.actions.copy()
        copy.chosen = self.chosen.copy()
        return copy


class VectorSortEnv():
    def __init__(self, dim, k=50, norm=2):
        self.state = []
        self.norm = norm
        self.k = k
        self.actions = [i for i in range(k)]
        self.dim = dim
        self.done = False
        self.players = 1
        self.turn = 0
        self.vecs = [np.random.rand(dim) for _ in range(k)]
        self.correct_sequence = sorted(
            self.vecs, key=lambda x: np.linalg.norm(x, self.norm))
        self.saved_actions = []

    def check(self):
        reward = -1
        # for i in range(len(self.state)):
        #     reward -= abs(np.linalg.norm(np.subtract(
        #         self.state[i], self.correct_sequence[i]), self.norm))
        if len(self.state) == self.k:
            return reward, True
        return reward, False

    def step(self, action):
        """
        Precondition: pos must be a tuple (m, n) where m, n < dim

        Returns:
        updated state (np.array)
        reward (int)
        done (boolean)
        empty dictionary
        """
        self.state.append(self.vecs[action])
        self.actions.remove(action)
        self.saved_actions.append(action)
        reward, self.done = self.check()
        return self.state, reward, self.done, dict()

    def reset(self):
        self.state = []
        self.vecs = [np.random.rand(self.dim) for _ in range(self.k)]
        self.done = False
        self.actions = [i for i in range(self.k)]
        self.correct_sequence = sorted(
            self.vecs, key=lambda x: np.linalg.norm(x))
        self.saved_actions = []

    def copy(self):
        copy = VectorSortEnv(self.dim)
        copy.vecs = [np.copy(i) for i in self.vecs]
        copy.state = [np.copy(i) for i in self.state]
        copy.k = self.k
        copy.done = self.done
        copy.actions = self.actions.copy()
        copy.correct_sequence = sorted(
            copy.vecs, key=lambda x: np.linalg.norm(x, self.norm))
        copy.saved_actions = self.saved_actions.copy()
        return copy

class AlphabeticalEnv():
    def __init__(self, number_of_words = 10, dim = 12):
        self.dim = dim
        self.sample_size = number_of_words
        self.correct_sequence = []
        self.index = 0
        self.state = None
        self.words = []

    def set_correct_sequence(self, sample):
        sample = list(enumerate(sample))
        sample = sorted(sample, key = lambda x : x[1])
        self.correct_sequence = [w[0] for w in sample]

    def get_correct_sequence(self):
        return self.correct_sequence
    
    def update_correct_sequence(self, action):
        for index, correct_action in enumerate(self.correct_sequence):
            if correct_action > action:
                self.correct_sequence[index] = correct_action - 1

    def reset(self):
        mat = np.zeros((self.sample_size, self.dim)) # (sample_size, vector dimension)
        sample = np.random.choice([i for i in range(self.dim)], size = self.sample_size, replace = False) # sample of 10 words
        for index, w in list(enumerate(sample)):
            mat[index, w] = 1
        self.words = sample
        self.set_correct_sequence(sample)
        self.index = 0
        self.state = mat
        return mat

    def step(self, action):
        """
        Remove word from matrix if it is the correct minimum, else do nothing

        Returns
        -------
        new matrix
        reward
        done
        info
        """
        reward = -10
        done = False
        if action == self.correct_sequence[self.index]:
            self.state = np.delete(self.state, action, 0)
            self.update_correct_sequence(action)
            self.index += 1
            if self.index == self.sample_size:
                done = True
            reward = -1
        return self.state, reward, done, {}

    def seed(self, seed = None):
        pass
    
    def copy(self):
        copy = AlphabeticalEnv(self.sample_size, self.dim)
        copy.correct_sequence = self.correct_sequence.copy() # this is correct right?
        copy.index = self.index
        copy.state = self.state
        copy.words = self.words