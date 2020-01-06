"""A Monte Carlo tree search agent.

"""

import numpy as np


class TreeNode:
    """A tree node for Monte Carlo tree search."""
    
    def __init__(self, parent, action, reward, done, env):
        self.parent = parent
        self.action = action
        self.reward = reward
        self.done = done
        self.env = env
        self.visits = 0
        self.value = 0
        self.children = []


class MCTSAgent:
    """A Monte Carlo tree search agent.
    
    The tree policy is epsilon-greedy and the off-tree policy is
    uniform random.

    Parameters
    ----------
    env_fn : function
        A function which maps states to new environments.
    epsilon : float, optional
        The fraction of uniform random choices in epsilon-greedy.
    rollouts : int, optional
        The number of rollouts to perform before choosing an action.

    """
    
    def __init__(self, env_fn, epsilon=0.05, rollouts=100):
        self.env_fn = env_fn
        self.epsilon = epsilon
        self.rollouts = rollouts
    
    def act(self, state):
        env = self.env_fn(state)
        root = TreeNode(None, None, 0, False, env)
        for _ in range(self.rollouts):
            leaf = self.expand(root)
            value = self.simulate(leaf)
            self.backup(leaf, value)      
        return max(root.children, key=lambda node: node.value).action
    
    def expand(self, node):
        """Return an unvisited or terminal leaf node following epsilon-greedy.
        
        Before returning, this function performs all possible actions from the
        leaf node and adds new nodes for them to the tree as children of the
        leaf node.
        """
        while node.visits != 0 and len(node.children) > 0:
            if np.random.rand() < self.epsilon:
                node = np.random.choice(node.children)
            else:
                node = max(node.children, key=lambda node: node.value)
        for action in node.env.actions:
            env = node.env.copy()
            state, reward, done, _ = env.step(action)
            node.children.append(TreeNode(node, action, reward, done, env))
        return node
    
    def simulate(self, node):
        """Return one total reward from node following uniform random policy."""
        env = node.env.copy()
        done = node.done
        total_reward = 0
        while not done:
            action = env.actions[np.random.choice(len(env.actions))]
            state, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward
    
    def backup(self, node, value):
        """Backup the return from a rollout from node."""
        while node != None:
            node.visits += 1
            node.value = (node.visits - 1)/node.visits * node.value + value/node.visits
            value += node.reward
            node = node.parent