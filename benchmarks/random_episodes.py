#!/usr/bin/env python
"""Benchmark running random episodes on the environment."""

import numpy as np
import sys

from deepgroebner.buchberger import LeadMonomialsEnv
from deepgroebner.wrapped import CLeadMonomialsEnv


def run_episode(agent, env):
    """Return total reward from one episode of agent in env."""
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


class RandomAgent:
    """Agent which selects randomly in LeadMonomialsWrapper environment."""

    def act(self, state):
        return np.random.choice(len(state))


if __name__ == "__main__":
    agent = RandomAgent()
    if len(sys.argv) == 4 and sys.argv[-1] == '--python':
        env = LeadMonomialsEnv(sys.argv[1], k=2)
        returns = [run_episode(agent, env) for _ in range(int(sys.argv[2]))]
        print(np.mean(returns))
    elif len(sys.argv) == 3:
        env = CLeadMonomialsEnv(sys.argv[1], k=2)
        returns = [run_episode(agent, env) for _ in range(int(sys.argv[2]))]
        print(np.mean(returns))
    else:
        print("Usage: random_episodes.py <dist> <episodes>")
