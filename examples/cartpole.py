# cartpole.py
# Dylan Peifer
# 06 Nov 2018
"""Experimenting with the CartPole environment from OpenAI Gym.

Our agent is represented by a single 1x4 array of parameter floats. Actions are
0 or 1, and we choose by dotting the parameters with the state and choosing 0
if negative and 1 otherwise.
"""

import gym
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense

env = gym.make('CartPole-v0')


class Agent:
    """A simple agent to solve cartpole."""

    def __init__(self):
        self.model = self._build_model()

    def act(self, state):
        pvals = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(2, p=pvals)

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(2, input_dim=4, activation='softmax'))
        return model


def random_weights(n=100):
    """Return a random set of weights from [-n,  n] for the agent's model."""
    return [2 * n * np.random.rand(4, 2) - n, 2 * n * np.random.rand(2) - n]


def random_search(agent, iterations=10000):
    """Set the agent's weights to best weights found by randomly searching."""
    best_weights = agent.model.get_weights()
    best_reward = run_episode(agent)
    for _ in range(iterations):
        weights = random_weights()
        agent.model.set_weights(weights)
        reward = run_episode(agent)
        if reward > best_reward:
            best_reward = reward
            best_weights = weights
    agent.model.set_weights(best_weights)


def hill_climb(agent, rate=0.1, iterations=10000):
    """Set the agent's weights to best weights found by hill climbing."""
    weights = agent.model.get_weights()
    reward = run_episode(agent)
    for _ in range(iterations):
        step = random_weights()
        new_weights = [weights[0] + step[0] * rate,
                       weights[1] + step[1] * rate]
        agent.model.set_weights(new_weights)
        new_reward = run_episode(agent)
        if new_reward > reward:
            reward = new_reward
            weights = new_weights
    agent.model.set_weights(weights)


class DQNAgent:
    pass


def run_episode(agent):
    """Run an episode to completion and return the total reward."""
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def run_episodes(agent, episodes):
    """Run several episodes to completion and return the average total
    reward.
    """
    total_reward = 0
    for _ in range(episodes):
        total_reward += run_episode(agent)
    return total_reward / episodes


def show_episode(agent):
    """Show the frames and states for a single episode run with given
    parameters. To close the window that contains the frames call env.close()
    """
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        env.render()
        time.sleep(0.1)
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print("Total reward: {}".format(total_reward))
