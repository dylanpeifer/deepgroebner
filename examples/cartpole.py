# cartpole.py
# Dylan Peifer
# 30 Oct 2018
"""Experimenting with the CartPole environment from OpenAI Gym.

Our agent is represented by a single 1x4 array of parameter floats. Actions are
0 or 1, and we choose by dotting the parameters with the state and choosing 0
if negative and 1 otherwise.
"""

import gym
import numpy as np
import time


env = gym.make('CartPole-v0')
parameters = np.array([0.0, 0.0, 0.0, 0.0])


def random_parameters():
    """Return a random array of 4 parameters between -1 and 1."""
    return 2 * (np.random.rand(4) - 0.5)


def choose_action(parameters, state):
    """Return 0 or 1 based on the sign of the dot product of parameters and
    state."""
    return 0 if np.dot(parameters, state) < 0 else 1


def run_episode(parameters):
    "Run an episode to completion and return the total reward."""
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = choose_action(parameters, state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def run_episodes(parameters, episodes):
    """Run several episodes to completion and return the average total
    reward."""
    total_reward = 0
    for _ in range(episodes):
        total_reward += run_episode(parameters)
    return total_reward / episodes


def show_episode(parameters):
    """Show the frames and states for a single episode run with given
    parameters. To close the window that contains the frames call env.close()
    """
    state = env.reset()
    done = False
    t = 0
    while not done:
        env.render()
        time.sleep(0.1)
        action = choose_action(parameters, state)
        state, _, done, _ = env.step(action)
        t += 1
    print("Episode finished after {} timesteps".format(t))


def random_search(iterations=10000):
    """Find best parameters by randomly searching."""
    best_parameters = random_parameters()
    best_reward = run_episode(best_parameters)
    for _ in range(iterations):
        parameters = random_parameters()
        reward = run_episode(parameters)
        if reward > best_reward:
            best_reward = reward
            best_parameters = parameters
    return best_parameters


def hill_climb(rate=0.1, iterations=10000):
    """Find best parameters by hill climbing."""
    parameters = random_parameters()
    reward = run_episode(parameters)
    for _ in range(iterations):
        new_parameters = parameters + random_parameters() * rate
        new_reward = run_episode(new_parameters)
        if new_reward > reward:
            reward = new_reward
            parameters = new_parameters
    return parameters
