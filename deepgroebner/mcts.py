from collections import deque
import time
import random
import numpy as np
import sys

from deepgroebner.wrapped import CLeadMonomialsEnv as LeadMonomialsEnv

def epsilon_greedy(epsilon=0.09):
    """Return an epsilon-greedy tree policy."""
    def policy(node):
        if random.random() < epsilon:
            return random.choice(node.children)
        else:
            return max(node.children, key=lambda n: n.value[node.env.turn])
    return policy


def ucb(c=np.sqrt(2)):
    """Return an upper confidence bound tree policy."""
    def policy(node):
        def v(n):
            if n.visits == 0:
                return np.inf
            else:
                return n.value[node.env.turn] + c * np.sqrt(np.log(node.visits)/n.visits)
        return max(node.children, key=v)
    return policy


class TreeNode:
    """A tree node for Monte Carlo tree search."""

    def __init__(self, parent, action, reward, env):
        self.parent = parent
        self.children = []
        self.action = action
        self.reward = reward
        self.env = env
        self.visits = 0
        self.value = np.zeros(env.players)


class MCTSAgent:
    """A Monte Carlo tree search agent.

    Parameters
    ----------
    tree_policy : function
        A function which maps node to child node.
    timeout : float, optional
        The amount of time in seconds to perform rollouts before choosing an action.

    """

    def __init__(self, tree_policy=ucb(), timeout=1.0):
        self.tree_policy = tree_policy
        self.timeout = timeout
        self.root = None

    def act(self, env):
        """Return a chosen action for the env.

        Parameters
        ----------
        env : environment
            The current environment.

        """
        self.root = self.find_root(env)
        limit = time.time() + self.timeout
        while time.time() < limit:
            leaf = self.expand(self.root)
            value = self.simulate(leaf)
            self.backup(leaf, value)
        return max(self.root.children, key=lambda node: node.visits).action

    def expand(self, node):
        """Return an unvisited or terminal leaf node following the tree policy.

        Before returning, this function performs all possible actions from the
        leaf node and adds new nodes for them to the tree as children of the
        leaf node.
        """
        while node.visits != 0 and len(node.children) > 0:
            node = self.tree_policy(node)
        if not node.env.done:
            for action in node.env.actions:
                env = node.env.copy()
                _, reward, _, _ = env.step(action)
                node.children.append(TreeNode(node, action, reward, env))
        return node

    def simulate(self, node):
        """Return one total reward from node following uniform random policy."""
        env = node.env.copy()
        total_rewards = np.zeros(env.players)
        while not env.done:
            action = random.choice(env.actions)
            _, rewards, _, _ = env.step(action)
            total_rewards += rewards
        return total_rewards

    def backup(self, node, value):
        """Backup the return from a rollout from node."""
        while node is not None:
            value += node.reward
            node.visits += 1
            node.value = (node.visits - 1)/node.visits * node.value + value/node.visits
            node = node.parent

    def find_root(self, env):
        """Return node corresponding to env in current tree using BFS."""
        if self.root is not None:
            q = deque(self.root.children)
            while q:
                node = q.popleft()
                if node.env == env:
                    return node
                q.extend(node.children)
        return TreeNode(None, None, np.zeros(env.players), env)


class MCTSWrapper:
    """A wrapper for LeadMonomialsEnv environments to interact with the MCTSAgent."""

    def __init__(self, env):
        self.env = env
        self.players = 1
        self.turn = 0
        self.state = None
        self.done = None
        self.actions = []

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.actions = list(range(len(self.state)))
        return self.state

    def step(self, action):
        self.state, reward, self.done, info = self.env.step(action)
        self.actions = list(range(len(self.state)))
        return self.state, reward, self.done, info

    def copy(self):
        copy = MCTSWrapper(self.env.copy())
        copy.state = self.state.copy()
        copy.done = self.done
        copy.actions = self.actions.copy()
        return copy


def run_episode(agent, env):
    env.reset()
    total_reward = 0.0
    while not env.done:
        action = agent.act(env)
        _, reward, _, _ = env.step(action)
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    dist = sys.argv[1]
    agent = MCTSAgent(timeout=1)
    env = MCTSWrapper(LeadMonomialsEnv(dist))
    with open(f'mcts-{dist}.csv', 'a') as f:
        f.write(str(run_episode(agent, env)) + "\n")
