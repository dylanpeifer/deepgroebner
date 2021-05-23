import os
import random
import csv

import numpy as np

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

class VectorEnv():
    def __init__(self, k = 10, vector_length = 64, norm = 1):
        self.set_size = k
        self.vector_length = vector_length
        self.norm = norm
        self.state = []
        self.correct_sequence = []
        self.index = 0

        # Debugging lists
        self.predicted_seq = []
        self.correct = []

    def set_correct_sequence(self):
        norms = list(np.linalg.norm(self.state, ord = self.norm, axis = 1))
        sorted_norms = sorted(list(enumerate(norms)), key=lambda x:x[1])
        self.correct_sequence = [x[0] for x in sorted_norms]
        self.correct = [self.state[i] for i in self.correct_sequence]

    def update_correct_sequence(self, action):
        for index, correct_action in enumerate(self.correct_sequence):
            if correct_action > action:
                self.correct_sequence[index] = correct_action - 1

    def set_state(self, mat):
        self.state = mat

    def step(self, action):
        reward = -10
        done = False
        if action == self.correct_sequence[self.index]:
            self.predicted_seq.append(self.state[action])
            self.state = np.delete(self.state, action, 0)
            self.update_correct_sequence(action)
            self.index += 1
            if self.index == len(self.correct_sequence):
                done = True
            reward = -1
        return self.state, reward, done, {}

    def reset(self):
        mat = np.random.rand(self.set_size, self.vector_length)
        self.set_state(mat)
        self.set_correct_sequence()
        self.index = 0
        return mat

    def seed(self, seed = None):
        pass

def run_episode_test(env):
    _ = env.reset()

    gold_standard = env.correct
    sorted_gs = []
    if isinstance(env, VectorEnv):
        norms = list(np.linalg.norm(env.state, ord = env.norm, axis = 1))
        sorted_norms = sorted(list(enumerate(norms)), key=lambda x:x[1])
        correct_sequence = [x[0] for x in sorted_norms]
        sorted_gs = [env.state[i] for i in correct_sequence]   
        for gs, sgs in list(zip(gold_standard, sorted_gs)):
            if np.sum(gs == sgs) == env.set_size:
                raise RuntimeError("Correct sequence was not actually correct")   

    index = 0
    done = False
    action = env.correct_sequence
    while not done:
        _, _, done, _ = env.step(action[index])
        index += 1
    if isinstance(env, AlphabeticalEnv):
        for p, c in list(zip(env.predicted_seq, env.correct)):
            if p != c:
                raise RuntimeError('One of the predicted element was not equal to the corresponding correct element')
    elif isinstance(env, VectorEnv):
        for p, c in list(zip(env.predicted_seq, env.correct)):
            if np.sum(p != c) == env.set_size:
                raise RuntimeError('One of the predicted element was not equal to the corresponding correct element')
    print("Done")

def run_episode_agent():
    pass

def main():
    pass

if __name__ == '__main__':
    main()