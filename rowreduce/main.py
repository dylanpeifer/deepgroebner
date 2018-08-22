# main.py
# Dylan Peifer
# 10 May 2018
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
from agents import RowChoiceAgent

from statistics import mean, median


def train(episodes, batch_size):
    """Train the agent on env for given episodes and using batch_size replays
    per episode.
    """
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        agent.replay(batch_size)


def test(episodes):
    """Test the agent and get average reward over given episodes."""
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
    return min(rewards), median(rewards), max(rewards), mean(rewards)


def play():
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


def stupid_strat():
    env.reset()
    total_reward = 0
    for i in range(5):
        next_state, reward, done = env.step(i)
        total_reward += reward
        if done:
            break
    return total_reward


N = 5  # number of rows in matrix
M = 15  # number of cols in matrix
env = RowChoiceEnvironment((N, M), 0.25)
agent = RowChoiceAgent((N, M), env.action_size)

if __name__ == "__main__":
    print("Moves needed before training: ", test(100))
    print("training batch || min || median || max || mean || epsilon")
    for i in range(10):
        train(100, 1024)
        print(str(i), test(100), agent.epsilon, len(agent.memory))
    agent.save("models/agent.h5")
    print("Agent saved to models/agent.h5")
