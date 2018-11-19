# main.py
# Dylan Peifer
# 05 Nov 2018
"""Doing computer algebra with reinforcement learning.

Running this file with

     $ python3 main.py

will create a new agent, train it, and save it. You can then observe the agent
on a random environment by entering the python interpreter and loading the
saved model.

     $ python3
     >>> from main import *
     >>> agent.load("models/agent.h5")
     >>> play()

Hyperparameters can be adjusted in the init method for the agent, and matrix
size and options can be changed at the bottom of the file.
"""

from environments import RowChoiceEnvironment
from agents import KInteractionsAgent, state_tensor

from statistics import mean, median
import numpy as np

agent = KInteractionsAgent(2, 2)
env = RowChoiceEnvironment((2, 2), 0.5)


def explore(agent, env, episodes):
    """Run the agent on env for given episodes and save transitions."""
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state


def train(agent, epochs, batch_size, verbose=0):
    """Train the agent epochs times on batches of batch_size from its
    memory.
    """
    for _ in range(epochs):
        agent.replay(batch_size, verbose=verbose)


def remember_stupid_strat(agent, env, episodes):
    """Fill the agents memory with episodes of the strategy that picks each
    row in order."""
    for _ in range(episodes):
        state = env.reset()
        done = False
        i = 0
        while not done:
            action = i
            next_state, reward, done = env.step(action)
            if reward != -1:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            i += 1


def remember_min_strat(agent, env, episodes):
    """Fill the agent's memory with episodes of the strategy that picks then
    row with min (but not 0) other 1s in it's lead column."""
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:

            # score each row by subtractions used if we pick it
            scores = []
            for row in state:
                lead = next((i for i, x in enumerate(row) if x != 0), None)
                if lead is not None:
                    score = sum(state[i, lead] for i in range(len(state))) - 1
                if lead is None or score == 0:
                    score = np.inf
                scores.append(score)

            # pick the min nonzero row
            action = np.argmin(scores)

            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state


def min_strat_V(state, gamma):
    """Return the actual V value for the min strat."""
    total_reward = 0
    n = 0  # power of gamma
    done = is_reduced(state)
    while not done:

        # score each row by subtractions used if we pick it
        scores = []
        for row in state:
            lead = next((i for i, x in enumerate(row) if x != 0), None)
            if lead is not None:
                score = sum(state[i, lead] for i in range(len(state))) - 1
            if lead is None or score == 0:
                score = np.inf
            scores.append(score)

        # pick the min nonzero row
        action = np.argmin(scores)

        # perform the action
        lead = next((i for i, x in enumerate(state[action, :]) if x != 0), None)
        if lead is None:
            reward = -100
        else:
            moves = 0
            for i in range(len(state)):
                if i != action and state[i, lead] != 0:
                    state[i, :] = (state[i, :] + state[action, :]) % 2
                    moves += 1
            reward = -100 - moves

        # record reward
        total_reward += (gamma ** n) * reward

        n += 1
        done = is_reduced(state)

    return total_reward


def min_strat_Q(state, gamma):
    """Return the actual Q values for the min strat."""
    Q_values = np.zeros(len(state))
    for action in range(len(state)):
        matrix = np.copy(state)
        lead = next((i for i, x in enumerate(matrix[action, :]) if x != 0), None)
        if lead is None:
            reward = -100
        else:
            moves = 0
            for i in range(len(state)):
                if i != action and matrix[i, lead] != 0:
                    matrix[i, :] = (matrix[i, :] + matrix[action, :]) % 2
                    moves += 1
            reward = -100 - moves
        Q_values[action] = reward + gamma * min_strat_V(matrix, gamma)

    return Q_values


def train_Q(agent, episodes, epochs, verbose=0):
    """Train the agent directly to the Q-function."""
    states = np.array([1 * (np.random.rand(agent.action_size, agent.action_size) < 0.5)
                       for _ in range(episodes)])
    targets = np.array([min_strat_Q(state, agent.gamma) for state in states])
    tensors = np.array([state_tensor(state, agent.k) for state in states])
    for _ in range(epochs):
        agent.model.fit(tensors, targets, verbose=verbose)


def is_reduced(matrix):
    """Return true if the matrix is reduced."""
    for row in range(len(matrix)):
        # find index of lead term in this row
        lead = next((i for i, x in enumerate(matrix[row, :]) if x != 0), None)
        # if this row has lead then everything in lead's column must be 0
        if lead is not None:
            for i in range(len(matrix)):
                if i != row and matrix[i, lead] != 0:
                    return False
    return True


def test(agent, env, episodes):
    """Test the agent and get average reward over given episodes."""
    eps = agent.epsilon
    agent.epsilon = 0.01
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        r = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            r += reward
        rewards.append(r)
    agent.epsilon = eps
    return min(rewards), median(rewards), max(rewards), mean(rewards)


def play(agent, env):
    """Show the steps as the agent solves a matrix."""
    epsilon = agent.epsilon  # save current epsilon
    agent.epsilon = 0  # don't explore during a play
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        print(state, action)
        next_state, reward, done = env.step(action)
        state = next_state
    print(state)
    agent.epsilon = epsilon  # reset epsilon


def main(agent, env, episodes, epochs, batch_size):
    for i in range(epochs):
        explore(agent, env, episodes)
        train(agent, episodes, batch_size, verbose=1)
        print(str(i) + ": ", test(agent, env, 100), agent.epsilon, len(agent.memory))
