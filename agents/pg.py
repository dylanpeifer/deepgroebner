# pg.py
# Dylan Peifer
# 18 Feb 2019
"""A policy gradient agent."""

import numpy as np
import tensorflow as tf
import time


def discounted_rewards(rewards, gamma):
    out = np.empty(len(rewards))
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        out[i] = cumulative_reward
    return list(out)


class PGAgent:
    """A policy gradient agent."""

    def __init__(self, network):
        self.action_size = network.output_shape[1]
        self.model = self._buildModel(network)
        self.gamma = 0.99

    def act(self, state):
        """Choose an action (row) for the given state."""
        probs = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(self.action_size, p=probs)

    def train(self, env, episodes):
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
        advantages = np.zeros((len(total_rewards), self.action_size))
        for i in range(len(total_rewards)):
            advantages[i][total_actions[i]] = total_rewards[i]

        # fitting to advantages performs policy gradient step
        self.model.fit(np.array(total_states), advantages, verbose=0)
        
    def test(self, env, episodes, render=False):
        """Test the agent for given episodes on given environment."""
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                rewards[i] += reward
                state = next_state
                if render:
                    env.render()
                    time.sleep(0.05)
        return rewards

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def _buildModel(self, network):
        model = tf.keras.models.clone_model(network)
        loss = 'categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(0.00025)
        model.compile(loss=loss, optimizer=optimizer)
        return model
