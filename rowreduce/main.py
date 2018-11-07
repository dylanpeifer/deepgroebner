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
from agents import KInteractionsAgent

from statistics import mean, median

agent = KInteractionsAgent(2, 3)
env = RowChoiceEnvironment((3, 3), 0.5)


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
