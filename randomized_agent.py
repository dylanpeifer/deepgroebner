import numpy as np
from random import sample
import tensorflow as tf
import os 
import datetime
import matplotlib.pyplot as plt
import math

from deepgroebner.wrapped import CLeadMonomialsEnv as LeadMonomialsEnv
from deepgroebner.pg import PPOAgent
from deepgroebner.networks import ParallelMultilayerPerceptron, TransformerPMLP, TransformerPMLP_Score_MHA, TransformerPMLP, PointerNetwork

class trajectory:

    def __init__(self):
        self.trajectory = []

    def get_trajectory(self):
        return self.trajectory

    def store(self, state, action, reward, seed):
        self.trajectory.append([state, action, reward, seed])

    def clear(self):
        self.trajectory = []

class episode:
    def __init__(self, ):
        pass

class RandomAgent:
    """A agent for LeadMonomialsEnv that selects at random."""

    def act(self, state):
        actions = state.shape[0]
        return np.random.randint(actions)

    def run_episode(self, env, buffer:trajectory, seed):
        state = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            action = self.act(state)
            new_state, reward, done, _ = env.step(action)
            buffer.store(state, action, reward, seed) # update our trajectory
            total_reward += reward
            length += 1
            state = new_state
        return total_reward, length

class SupervisedLearner():
    """
    Used to test train new neural net models in 
    Parameters
    ----------
    learner: neural net architecture
        Should output log probabilities
    env: Buchberger environment
    """

    def __init__(self, learner, optimizer, filename):
        self.model = learner
        container = np.load(filename, allow_pickle = True)
        self.dataset = [container[key] for key in container][2:]
        print("Done loading dataset!")
        self.optimizer = optimizer
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def get_episode(self):
        pass
    
    def train(self, model_path, epochs = 5, batch_size = 100):
        '''
        Train the learner using certain strategy 
        '''
        history = {'loss':[]} # in case we want to add anything else
        number_of_batches = math.floor(len(self.dataset)/batch_size)+1
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(number_of_batches): # change this to find the best batch_size 
                loss = 0
                batch = sample(self.dataset, batch_size)
                with tf.GradientTape() as tape:
                    for _, datum in enumerate(batch):
                        datum_tensor = tf.expand_dims(tf.convert_to_tensor(datum[0]), axis = 0) #(1, num_poly, feature_size)
                        logprob = self.model(datum_tensor)
                        correct_action = [datum[1]]
                        loss += self.loss(correct_action, logprob)
                    loss = loss/batch_size
                    epoch_loss += loss
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            print('Epoch {}/{} - Loss was: {}'.format(epoch+1, epochs, epoch_loss/(number_of_batches-1)))
        self.model.save_weights(model_path)
        return history

    def model_summary(self):
        self.model.summary()

def test_dataset(dataset):
    container = np.load(dataset, allow_pickle= True)
    dataset = [container[key] for key in container][2:]
    env = LeadMonomialsEnv('3-20-10-weighted', k=2) 
    seeds = list(range(10, 3000))
    index = 0
    num_correct = 0
    episode_indexes = []
    start = 0
    for num_seed, seed in enumerate(seeds):
        env.seed(seed)
        state = env.reset()
        done = False
        total_reward = 0
        dataset_reward = 0
        while not done:
            state, reward, done, _ = env.step(dataset[index][1])
            total_reward += reward
            dataset_reward += dataset[index][2]           
            index += 1

        episode_indexes.append([start, index]) # record when the episode starts and ends+1
        start = index # start of the new episode

        if done and dataset_reward == total_reward:
            num_correct += 1
        print('We are done with {}'.format(num_seed))
    print('Seeds: {} and Number correct {}'.format(len(seeds), num_correct))

def test_episode(dataset):
    pass

def evaluate(agent, env, samples, seed=123):
    """Run the agent repeatedly on env initialized by seed."""

    buffer = trajectory()
    best_reward=-10000
    dataset = []
    returns = []
    for _ in range(samples):
        env.seed(seed)

        if isinstance(agent, RandomAgent):
            reward,_ = agent.run_episode(env, buffer, seed)
        else:
            reward,_ = agent.run_episode(env)

        if reward > best_reward:
            dataset = buffer.get_trajectory()
            best_reward = reward
        buffer.clear()
        returns.append(reward)



    # Error checking
    dataset_return = 0
    seed = dataset[0][3]
    for datum in dataset:
        dataset_return += datum[2]
        new_seed = datum[3]
        if seed != new_seed:
            print('Not all seeds are the same!')
            raise Exception
        else:
            seed = new_seed

    if best_reward != max(returns) and dataset_return != max(returns):
        print('Something went wrong')
        raise Exception

    return returns, dataset

def get_dataset(env, agent, seeds, samples = 1000):

    def find_difficulty(ideal):
        pass
    
    all_results = {}
    for seed in seeds:
        returns, dataset = evaluate(agent, env, samples, seed)

        if isinstance(agent, RandomAgent):
            all_results[seed] = (returns, dataset)
        else:
            all_results[seed] = returns
        print('Done with seed {}'.format(seed))
    return all_results

def find_better_trajectories(random_agent, trained_agent, seeds):
    better = {}
    for seed in seeds:
        if max(random_agent[seed][0]) > max(trained_agent[seed]):
            better[seed] = (max(random_agent[seed][0]), random_agent[seed][1]) # (best, best trajectories)
    return better

def save(dataset, filename):
    np.savez(filename, allow_pickle = True, dtype = object, *dataset)

def generate_dataset(filename = "random_path_dataset_large.npz"):
    seeds = list(range(10, 3000))
    env = LeadMonomialsEnv('3-20-10-weighted', k=2) 
    random_agent = RandomAgent()
    samples = 1500
    ran_results = get_dataset(env, random_agent, seeds, samples = samples)

    dataset = []
    for k in ran_results.keys():
        for data in ran_results[k][1]:
            dataset.append(data)
    save(dataset, filename)
    print('Done')

def train_on_random_dataset(dataset_name):
    dataset_loc = dataset_name
    if not dataset_loc in os.listdir():
        generate_dataset(dataset_loc)

    optimizer = tf.keras.optimizers.Adam()
    tpmlp = TransformerPMLP(128, 128)
    learner = SupervisedLearner(tpmlp, optimizer, dataset_loc)

    # save the model
    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = 'pmlp-larger'
    time_string += name
    logdir = os.path.join('random_agent_test', time_string)
    os.mkdir(logdir)
    learner.train(os.path.join(logdir, 'policy_random.h5'), epochs = 100, batch_size = 64)

def main():
    dataset_loc = "ran_agent_dataset.npz"
    train_on_random_dataset(dataset_loc)

if __name__ == '__main__':
    main()