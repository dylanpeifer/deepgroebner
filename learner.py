import numpy as np
import deepgroebner.buchberger as dpb
from deepgroebner.ideals import RandomBinomialIdealGenerator

import tensorflow as tf
from tensorflow import keras

import os

class SupervisedLearner():
    """
    Used to test train new neural net models in 

    Parameters
    ----------
    learner: neural net architecture
        Should output log probabilities
    env: Buchberger environment
    """

    def __init__(self, n, d, s, learner, optimizer,selection_strategy = 'normal'):
        self.model = learner
        self.dataset = PolynomialDataset(n, d, s, selection_strategy=selection_strategy)
        self.optimizer = optimizer

    def loss_fn(self, correct_action, log_prob):
        '''
        Implementation of negative log likelihood loss 
        '''
        return tf.linalg.matmul(correct_action, log_prob, transpose_b = True)
    

    def train(self, filename, model_path, epochs = 5, batch_size = 100):
        '''
        Train the learner using certain strategy 
        '''
        dataset, keys = self.dataset.load(filename)
        history = {'loss':[]} # in case we want to add anything else
        for epoch in range(epochs):
            losses = []
            batch = self.dataset.batch(dataset, keys, batch_size) # Get random sample of data
            with tf.GradientTape() as tape:
                for _, datum in enumerate(batch):
                    datum_tensor = tf.expand_dims(tf.convert_to_tensor(datum[0]), axis = 0) #(1, num_poly, feature_size)
                    logprob = self.model(datum_tensor)
                    correct_action = tf.expand_dims(tf.one_hot(datum[1], datum[0].shape[0]), axis = 0)
                    losses.append(self.loss_fn(correct_action, logprob))
                loss = tf.reduce_mean(tf.concat(losses, axis=0))
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                print('Epoch {}/{} - Loss was: {}').format(epoch, epochs, loss)
                history['loss'] = (epoch, loss)
        self.model.save_weights(model_path)
        return history

class PolynomialDataset():
    '''
    Polynomial dataset that generates, save, load, and batch examples based on selection strategy.

    Parameters
    ----------
    n: Number of variables in monomial
    d: Max degree
    s: Number of generators
    selection_strategy: Can be "first", "degree", "normal", and "random"

    '''
    def __init__(self, n, d, s, selection_strategy = 'normal'):
        ideal_gen = RandomBinomialIdealGenerator(n, d, s, degrees='weighted') 
        self.env = dpb.LeadMonomialsWrapper(dpb.BuchbergerEnv(ideal_gen), k=2)
        self.agent = dpb.LeadMonomialsAgent(selection = selection_strategy, k = 2)
    
    def run_episode(self, dataset):
        '''
        Run episode using LeadMonomialAgent and LeadMonomialsWrapper

        Parameters
        ----------
        dataset: list of tuples (matrix set, action) that we keep updating
        '''
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.act(state)
            next_state, _, done, _ = self.env.step(action)
            dataset.append((state, action))
            state = next_state
        return dataset
            
    def generate_dataset(self, filename, num_episodes = 1000, num_data_points = 10000):
        '''
        Generate dataset of size number of episodes. File saved in learner_data.

        Parameters
        ----------
        filename: dataset name
        num_episodes: number of episodes to run
        num_data_points: number of data points NOTE: this doesn't do anything
        '''

        # Check if learner_data exists and if filename in learner_data
        if 'learner_data' in os.listdir('.'):
            if filename in os.listdir('learner_data'):
                print('File already created!')
                return None
        else:
            os.mkdir('learner_data') # If learner_data doesn't exist create it
        filename = os.path.join('learner_data', filename) # Filename
        dataset = []

        # Run episodes and record matrix and move
        for _ in range(num_episodes):
            dataset = self.run_episode(dataset)
        np.savez(filename, allow_pickle = True, *dataset)
    
    def load(self, filename):
        '''
        Load dataset from filename
        '''
        if 'learner_data' in os.listdir('.'):
            if filename in os.listdir('learner_data'):
                filename = os.path.join('learner_data', filename)
            else:
                print('No file of that name!')
                return None
        else:
            print('No data file!')
            return None

        container = np.load(filename, allow_pickle=True)
        keys = [key for key in container]
        return container, keys[1:len(keys)]

    def batch(self, data, keys, batch_size):
        '''
        Randomly sample from the dataset
        '''
        return [data[key] for key in np.random.choice(keys, batch_size)]
