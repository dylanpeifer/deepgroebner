import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import tensorflow as tf

from environments.binomials import BinomialBuchbergerEnv, LeadMonomialWrapper
from agents.networks import ParallelMultilayerPerceptron


variables = sp.symbols('x y z')
domain = sp.FF(32003)
order = 'grevlex'
degree = 7
size = 5

env = LeadMonomialWrapper(BinomialBuchbergerEnv(degree, size, len(variables)), k=2)


def discounted_rewards(rewards, gamma):
    out = np.empty(len(rewards))
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        out[i] = cumulative_reward
    return list(out)


class PGAgent:
    """A policy gradient agent."""

    def __init__(self, network, learning_rate=0.00025, gamma=0.99):
        self.model = self._buildModel(network, learning_rate)
        self.gamma = gamma

    def act(self, state):
        """Choose an action for the given state."""
        probs = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(len(probs), p=probs)

    def train(self, env, episodes):
        """Train the agent using policy gradients."""
        reward_out = np.zeros(episodes)

        total_states = []
        total_actions = []
        total_rewards = []
        total_baselines = []

        # generate rollouts and discounted rewards
        for i in range(episodes):
            state = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            baselines = []
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                reward_out[i] += reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                baselines.append(state.shape[0])
                state = next_state
            rewards = discounted_rewards(rewards, self.gamma)

            total_states += states
            total_actions += actions
            total_rewards += rewards
            total_baselines += baselines

        # produce and normalize advantages
        advantages = np.array(total_rewards) + np.array(total_baselines)
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages)

        # fit to advantages to perform policy gradient step
        for i in range(len(total_states)):
            state = np.expand_dims(total_states[i], axis=0)
            advantage = np.zeros((1, state.shape[1]))
            advantage[0, total_actions[i]] = advantages[i]  
            self.model.fit(state, advantage, verbose=0)

        return reward_out

    def train_batched(self, env, episodes):
        """Train the agent using policy gradients."""
        reward_out = np.zeros(episodes)

        total_states = []
        total_actions = []
        total_rewards = []
        total_baselines = []

        # generate rollouts and discounted rewards
        for i in range(episodes):
            state = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            baselines = []
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                reward_out[i] += reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                baselines.append(state.shape[0])
                state = next_state
            rewards = discounted_rewards(rewards, self.gamma)

            total_states += states
            total_actions += actions
            total_rewards += rewards
            total_baselines += baselines

        # produce and normalize advantages
        total_advantages = np.array(total_rewards) + np.array(total_baselines)
        total_advantages -= np.mean(total_advantages)
        total_advantages /= np.std(total_advantages)
        
        # process into batches
        batches = {}
        for i in range(len(total_states)):
            size = total_states[i].shape[0]
            if size not in batches:
                batches[size] = [[], [], []]
            batches[size][0].append(total_states[i])
            batches[size][1].append(total_actions[i])
            batches[size][2].append(total_advantages[i])

        # fit to advantages to perform policy gradient step
        for size in batches:
            states = np.stack(batches[size][0])
            actions = np.array(batches[size][1])
            advantages = np.zeros((len(states), states[0].shape[0]))
            advantages[np.arange(len(states)), actions] = np.array(batches[size][2])
            self.model.fit(states, advantages, verbose=0)

        return reward_out

    def loadModel(self, filename):
        self.model.load_weights(filename)
        
    def saveModel(self, filename):
        self.model.save_weights(filename)

    def _buildModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model
    

if __name__ == "__main__":
    n = len(variables)
    network = ParallelMultilayerPerceptron(4*n, [16*n, 16*n])
    agent = PGAgent(network, learning_rate=0.00001, gamma=1.0)
    
    i = 0
    while True:
        r = np.mean(agent.train_batched(env, 1000))
        with open('rewards.txt', 'a') as f:
            f.write(str(r) + '\n')
        if i % 25 == 0:
            agent.saveModel('models/nonhombinom/' + str(i) + '.h5')
        i += 1
        