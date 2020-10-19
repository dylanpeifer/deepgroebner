import numpy as np
import deepgroebner.buchberger as dpb
from deepgroebner.ideals import RandomBinomialIdealGenerator
from deepgroebner.pg import PPOAgent

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
            loss = 0
            batch = self.dataset.batch(dataset, keys, batch_size) # Get random sample of data
            with tf.GradientTape() as tape:
                for _, datum in enumerate(batch):
                    datum_tensor = tf.expand_dims(tf.convert_to_tensor(datum[0]), axis = 0) #(1, num_poly, feature_size)
                    logprob = self.model(datum_tensor)
                    correct_action = tf.expand_dims(tf.one_hot(datum[1], datum[0].shape[0]), axis = 0)
                    loss += self.loss_fn(correct_action, -logprob)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                print('Epoch {}/{} - Loss was: {}'.format(epoch, epochs, loss))
                history['loss'] = (epoch, loss)
        self.model.save_weights(model_path, save_format='tf')
        return history

    def model_summary(self):
        self.model.summary()

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

#NOTE: everything below still needs to be tested!!

class Evaluator:
    '''
    Evaluate learners.

    Parameters
    ----------
    learners: {'ModelName': [learner, model_path]}
    '''
    def __init__(self, learners:dict, lead_agent, env):
        self.pretrained_model = {}
        self.performance_tracker = {}
        for key in learners.keys():
            pmodel = learners[key][0]
            model_path = learners[key][1]
            pmodel.load_weights(model_path)
            self.pretrained_model[key] = PPOAgent(pmodel)
            self.performance_tracker[key] = [0]
        self.lead_agent = lead_agent
        self.env = env

    def eval(self, num_episodes=1):
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            total_step = 0
            while not done:
                lead_action = self.lead_agent.act(state)
                for key in self.pretrained_model.keys():
                    action = self.pretrained_model[key].act(state, greedy = True)
                    if action == lead_action:
                        self.performance_tracker[key][i] = self.performance_tracker[key][i] + 1
                next_state,_,done,_ = self.env.step(lead_action)
                state = next_state
                total_step += 1
            for key in self.performance_tracker.keys():
                temp = self.performance_tracker[key]
                temp[i] = temp[i]/total_step
                if i != num_episodes-1:
                    temp.append(0)
                self.performance_tracker[key] = temp
        return self.performance_tracker

class Teacher:
    def __init__(self, dataset_path, strategy, model, model_save_path, optimizer, n, d, s, epochs, batch_size):
        pd = PolynomialDataset(n,d,s,strategy)
        pd.generate_dataset(dataset_path, num_episodes=1000)
        self.learner = SupervisedLearner(n,d,s,model,optimizer, selection_strategy=strategy)
        self.model_path = model_save_path
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size

    def teach(self):
        self.learner.train(self.dataset_path, self.model_path, self.epochs, self.batch_size)
