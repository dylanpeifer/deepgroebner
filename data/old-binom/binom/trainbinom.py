# trainbinom.py
# 03 Mar 2019
# Dylan Peifer
"""Train a policy gradient agent on a binomial Buchberger environment.

The environment is wrapped to return a state tensor of shape (len(P), 1, 2*n),
where P is the set of pairs and n is the number of variables. Each row in the
tensor is the concatenation of the exponent vectors corresponding to the lead
monomials of a pair. Random selection in this environment is about -35, while
degree selection is -20. The network can be manually set to weights that result
in -20. The goal is to learn this from scratch.
"""

import numpy as np
import sympy as sp
import tensorflow as tf

from environments.buchberger import BinomialBuchbergerEnv, LeadMonomialWrapper
from agents.networks import ParallelMultilayerPerceptron

variables = sp.symbols('x y z')
domain = sp.FF(32003)
order = 'grevlex'
degree = 2
size = 5

ENV = LeadMonomialWrapper(BinomialBuchbergerEnv(degree, size, variables, domain=domain, order=order))


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

        # generate rollouts and discounted rewards
        for i in range(episodes):
            state = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                reward_out[i] += reward
                states += [state]
                actions += [action]
                rewards += [reward]
                state = next_state
            rewards = discounted_rewards(rewards, self.gamma)

            total_states += states
            total_actions += actions
            total_rewards += rewards

        # fit to advantage vectors
        for i in range(len(total_states)):
            state = np.expand_dims(total_states[i], axis=0)
            advantage = np.zeros((1, state.shape[1]))
            advantage[0, total_actions[i]] = total_rewards[i] + state.shape[1]
            
            self.model.fit(state, advantage, verbose=0)

        return reward_out

    def saveModel(self, filename):
        self.model.save_weights(filename)

    def loadModel(self, filename):
        self.model.load_weights(filename)

    def _buildModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model


if __name__ == "__main__":
    n = len(variables)
    network = ParallelMultilayerPerceptron(2*n, [8*n])
    agent = PGAgent(network, learning_rate=0.00001, gamma=1.0)
    i = 0
    while True:
        r = np.mean(agent.train(ENV, 1000))
        with open('rewards.txt', 'a') as f:
            f.write(str(r) + '\n')
        agent.saveModel('models/binom/' + str(i) + '.h5')
        i += 1
