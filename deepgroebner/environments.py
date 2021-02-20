import os
import random
import csv

import numpy as np
from nltk.tokenize import word_tokenize

FILENAME = '/Users/christianvarner/Research/deepgroebner/deepgroebner/booksummaries/booksummaries.txt'

def get_data(filename):
    vocab = []
    with open(filename, 'r+', encoding='utf-8') as data:
        line = data.readline()
        while line and len(vocab) < 1000:
            line_data = line.split('\t')
            summary = line_data[6]
            words = word_tokenize(summary.strip('\n'))
            no_dup = []
            [no_dup.append(w.lower()) for w in words if not w.lower() in no_dup and w.isalpha()]
            vocab += [w for w in no_dup if not w in vocab]
            line = data.readline()
    return list(enumerate(sorted(vocab)))

def save_data(loc, vocab):
    with open(loc, 'w+') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(vocab)

def get_saved_data(loc):
    with open(loc, 'r+') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            vocab = row
    return vocab

class AlphabeticalEnv():
    def __init__(self, number_of_words = 10):
        self.data = get_data(filename = FILENAME)
        self.sample_size = number_of_words
        self.correct_sequence = []
        self.index = 0
        self.state = None
        self.words = []

        # Debugging sequence
        self.predicted_seq = []
        self.correct = []

    def set_correct_sequence(self, sample):
        sample = list(enumerate(sample))
        sample = sorted(sample, key = lambda x : x[1][0])
        self.correct_sequence = [w[0] for w in sample]
        self.correct = [s[1][1] for s in sample]

    def get_correct_sequence(self):
        return self.correct_sequence
    
    def update_correct_sequence(self, action):
        for index, correct_action in enumerate(self.correct_sequence):
            if correct_action > action:
                self.correct_sequence[index] = correct_action - 1

    def reset(self):
        mat = np.zeros((self.sample_size, len(self.data)))
        sample = random.sample(self.data, self.sample_size) # sample of 10 words
        for index, w in list(enumerate(sample)):
            mat[index, w[0]] = 1
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
        reward = -1
        done = False
        if action == self.correct_sequence[self.index]:
            self.predicted_seq.append(self.data[list(self.state[action]).index(1)][1])
            self.state = np.delete(self.state, action, 0)
            self.update_correct_sequence(action)
            self.index += 1
            if self.index == self.sample_size:
                done = True
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
        self.picked_sequence = []
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
        reward = -1
        done = False
        if action == self.correct_sequence[self.index]:
            self.picked_sequence.append(self.state[action])
            self.state = np.delete(self.state, action, 0)
            self.update_correct_sequence(action)
            self.index += 1
            if self.index == len(self.correct_sequence):
                done = True
        return self.state, reward, done, {}

    def reset(self):
        mat = np.random.rand(self.set_size, self.vector_length)
        self.set_state(mat)
        self.set_correct_sequence()
        self.index = 0
        return mat

    def seed(self, seed = None):
        pass

def run_episode(env):
    _ = env.reset()

    gold_standard = env.correct
    sorted_gs = []
    if isinstance(env, AlphabeticalEnv):
        sorted_gs = sorted(gold_standard) # sort the correct sequence should result in the same sequence
    elif isinstance(env, VectorEnv):
        norms = list(np.linalg.norm(env.original_state, ord = env.norm, axis = 1))
        sorted_norms = sorted(list(enumerate(norms)), key=lambda x:x[1])
        correct_sequence = [x[0] for x in sorted_norms]
        sorted_gs = [env.original_state[i] for i in correct_sequence]      

    for gs, sgs in list(zip(gold_standard, sorted_gs)):
        if gs != sgs:
            raise RuntimeError("Correct sequence was not actually correct")

    index = 0
    done = False
    action = env.correct_sequence
    while not done:
        _, _, done, _ = env.step(action[index])
        index += 1
        
    for p, c in list(zip(env.predicted_seq, env.correct)):
        if p != c:
            raise RuntimeError('One of the predicted element was not equal to the corresponding correct element')
    print("Done")


def main():
    env = AlphabeticalEnv(number_of_words=10)
    run_episode(env)

if __name__ == '__main__':
    main()