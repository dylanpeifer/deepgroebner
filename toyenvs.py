import numpy as np


class DumbTicTacToeEnv():
    def __init__(self, dim):
        self.state = np.zeros((dim, dim))
        self.actions = [(i, j) for i in range(dim)
                        for j in range(dim) if self.state[i][j] != 1]
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
        self.state[pos[0], pos[1]] = 1
        self.actions = [(i, j) for i in range(self.dim)
                        for j in range(self.dim) if self.state[i][j] != 1]
        self.done = self.check_status()
        return self.state, reward, self.done, dict()

    def to_string(self):
        return np.array_str(self.state) + "\nDimension: " + str(self.dim)

    def reset(self):
        self.state = np.zeros((self.dim, self.dim))
        self.done = False
        self.actions = [(i, j) for i in range(self.dim)
                        for j in range(self.dim)]
        return self.state

    def copy(self):
        copy = DumbTicTacToeEnv(self.dim)
        copy.state = self.state.copy()
        copy.done = self.done
        copy.actions = self.actions.copy()
        return copy


def simple_test():
    """
    Tests basic functionality the environment - stepping,
    checking status, resetting.
    """
    env = DumbTicTacToeEnv(4)

    # check first diagonal
    env.step((3, 3))
    assert not env.check_status()
    env.step((2, 2))
    env.step((2, 1))
    env.step((1, 1))
    env.step((0, 0))
    assert env.check_status()
    env.reset()

    # check row
    env.step((0, 1))
    env.step((0, 2))
    env.step((0, 3))
    assert not env.check_status()
    env.step((0, 0))
    assert env.check_status()
    env.reset()

    # check col
    env.step((1, 1))
    env.step((2, 1))
    env.step((0, 1))
    assert not env.check_status()
    env.step((3, 1))
    assert env.check_status()
    env.reset()

    # check second diagonal
    env.step((0, 3))
    env.step((1, 2))
    assert not env.check_status()
    env.step((2, 1))
    env.step((3, 0))
    assert env.check_status()


def run_episode_test(env):
    _ = env.reset()
    pass


simple_test()


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
            copy.vecs, key=lambda x: np.linalg.norm(x, 2))
        copy.saved_actions = self.saved_actions.copy()
        return copy
