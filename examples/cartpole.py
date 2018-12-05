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
import random
from keras.models import Sequential
from keras.layers import Dense

env = gym.make('CartPole-v0')


class Parameters:
    """An even simpler agent to solve cartpole. Parameters were found by
    random search. The agent reliably gets to 200.
    """

    def __init__(self):
        self.params = np.array([0.20, 0.23, 0.52, 0.52])

    def act(self, state):
        return 0 if np.matmul(self.params, state) < 0 else 1


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


class Memory:
    """A cyclic buffer to store transition memories. Adapted from
    http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a random sample from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def discounted_rewards(rewards, gamma):
    out = np.empty(len(rewards))
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        out[i] = cumulative_reward
    return list(out)


class PGAgent:
    """A policy gradient agent for cartpole."""

    def __init__(self):
        self.gamma = 0.95
        self.model = self._build_model()

    def act(self, state):
        probs = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(2, p=probs)

    def train(self, episodes):
        """Train the agent using policy gradients."""
        total_states = []
        total_actions = []
        total_rewards = []

        # generate rollouts and discounted rewards
        for _ in range(episodes):
            state = env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                states += [state]
                actions += [action]
                rewards += [reward]
                state = next_state
            rewards = discounted_rewards(rewards, self.gamma)

            total_states += states
            total_actions += actions
            total_rewards += rewards

        # normalize the rewards and produce the advantage vectors
        total_rewards = np.array(total_rewards)
        total_rewards -= np.mean(total_rewards)
        total_rewards /= np.std(total_rewards)
        advantages = np.zeros((len(total_rewards), 2))
        for i in range(len(total_rewards)):
            advantages[i][total_actions[i]] = total_rewards[i]

        # fitting to advantages performs policy gradient step
        self.model.fit(np.array(total_states), advantages, epochs=1, verbose=0)

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=4, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model


class DQNAgent:
    """A Deep Q Network agent for cartpole."""

    def __init__(self):
        self.memory = Memory(1000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self._build_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(2)
        return np.argmax(self.model.predict(np.expand_dims(state, axis=0))[0])

    def replay(self, batch_size, verbose=0):
        if len(self.memory) < batch_size:
            return

        minibatch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        next_states = np.array(next_states)
        dones = np.array(dones)

        targets = np.array(rewards)
        targets = targets + (dones == False) * self.gamma * \
                  np.max(self.model.predict(next_states), axis=1)
        targets_f = self.model.predict(states)
        targets_f[np.arange(targets_f.shape[0]), actions] = targets

        self.model.fit(states, targets_f, epochs=1, verbose=verbose)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=4, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='linear'))

        model.compile(loss='logcosh', optimizer='adam')
        return model


def explore(agent, env, episodes):
    """Run the agent on env for given episodes and save transitions."""
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state


def train(agent, epochs, batch_size, verbose=0):
    """Train the agent epochs times on batches of batch_size from its
    memory.
    """
    for _ in range(epochs):
        agent.replay(batch_size, verbose=verbose)


def copy(leader, follower, env, episodes):
    """Store episodes from env acted on by leader in follower's memory."""
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = leader.act(state)
            next_state, reward, done, _ = env.step(action)
            follower.remember(state, action, reward, next_state, done)
            state = next_state


def main(agent, env, episodes, epochs, batch_size):
    for _ in range(epochs):
        explore(agent, env, episodes)
        train(agent, episodes, batch_size, verbose=1)
        print(run_episodes(agent, 100))


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
