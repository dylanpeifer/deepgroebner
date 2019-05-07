# pg.py
# Dylan Peifer
# 06 May 2019
"""Policy gradient agent that supports changing state shapes."""

import numpy as np
import tensorflow as tf


def discount_rewards(rewards, gamma):
    """Discount the list or array of rewards by gamma in-place."""
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gamma * cumulative_reward
        rewards[i] = cumulative_reward
    return rewards


class TrajectoryBuffer:
    """A buffer to store and compute with trajectories."""

    def __init__(self, gam, lam):
        self.gam = gam
        self.lam = lam
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.start = 0
        self.end = 0

    def store(self, state, action, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.end += 1

    def finish(self):
        """Finish an episode and compute advantages and discounted rewards."""
        tau = slice(self.start, self.end)
        delta = np.array(self.rewards[tau], dtype=np.float)
        delta -= np.array(self.values[tau])
        delta[:-1] += self.gam * np.array(self.values[self.start+1:self.end])
        self.values[tau] = list(discount_rewards(delta, self.gam * self.lam))
        self.rewards[tau] = discount_rewards(self.rewards[tau], self.gam)
        self.start = self.end

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.start = 0
        self.end = 0

    def makeBatches(self, action_dim_fn, normalize=True):
        """Return a dictionary of state shapes to (states, values, advantages) batches."""
        adv = np.array(self.values[:self.start])
        if normalize:
            m, s = np.mean(adv), np.std(adv)
            adv = (adv - m) / s
        shapes = {}
        for i, state in enumerate(self.states[:self.start]):
            shapes.setdefault(state.shape, []).append(i)
        batches = {}
        for shape, indices in shapes.items():
            states = np.array([self.states[i] for i in indices])
            values = np.expand_dims(np.array([self.rewards[i] for i in indices]), axis=1)
            advantages = np.zeros((len(indices), action_dim_fn(shape)))
            actions = [self.actions[i] for i in indices]
            advantages[np.arange(len(indices)), actions] = [adv[i] for i in indices]
            batches[shape] = (states, values, advantages)
        return batches


class PGAgent:
    """A policy gradient agent."""

    def __init__(self, policy_network, policy_learning_rate=0.00025,
                 value_network=None, value_learning_rate=0.001,
                 gam=0.99, lam=0.97, normalize=True,
                 value_updates_per_epoch=25, action_dim_fn=lambda s: s[0]):
        self.policyModel = self._buildPolicyModel(policy_network, policy_learning_rate)
        if value_network is None:
            self.valueModel = None
        else:
            self.valueModel = self._buildValueModel(value_network, value_learning_rate)
        self.gam = gam
        self.lam = lam
        self.normalize = normalize
        self.value_updates_per_epoch = value_updates_per_epoch
        self.action_dim_fn = action_dim_fn

    def act(self, state, greedy=False):
        """Return an action for the given state."""
        probs = self.policyModel.predict(np.expand_dims(state, axis=0))[0]
        if greedy:
            return np.argmax(probs)
        else:
            return np.random.choice(len(probs), p=probs)

    def train(self, env, episodes, epochs=1, verbose=0, savedir=None, savefreq=1):
        """Train the agent using policy gradients."""
        buf = TrajectoryBuffer(self.gam, self.lam)
        for i in range(1, epochs + 1):

            buf.clear()
            total_reward = 0
            for _ in range(episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.act(state)
                    next_state, reward, done, _ = env.step(action)
                    if self.valueModel is None:
                        value = 0
                    else:
                        value = self.valueModel.predict(np.expand_dims(state, axis=0))[0][0]
                    buf.store(state, action, reward, value)
                    total_reward += reward
                    state = next_state
                buf.finish()

            batches = buf.makeBatches(self.action_dim_fn, normalize=self.normalize)
            for shape in batches:
                self.policyModel.fit(batches[shape][0], batches[shape][2], verbose=0)
            for _ in range(self.value_updates_per_epoch):
                for shape in batches:
                    self.valueModel.fit(batches[shape][0], batches[shape][1], verbose=0)

            if verbose == 1:
                print("\rEpoch: {}/{} - avg_reward: {}"
                      .format(i, epochs, total_reward / episodes), end="")

            if savedir is not None:
                with open(savedir + '/rewards.txt', 'a') as f:
                    f.write(str(i) + ',' + str(r) + '\n')
                if i % savefreq == 0:
                    self.savePolicyModel(savedir + "/policy-" + str(i) + ".h5")
                    self.saveValueModel(savedir + "/value-" + str(i) + ".h5")

    def test(self, env, episodes=1, greedy=False):
        """Test the agent for episodes on env and return array of total rewards."""
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state, greedy=greedy)
                state, reward, done, _ = env.step(action)
                rewards[i] += reward
        return rewards

    def loadPolicyModel(self, filename):
        self.policyModel.load_weights(filename)

    def savePolicyModel(self, filename):
        self.policyModel.save_weights(filename)

    def loadValueModel(self, filename):
        self.valueModel.load_weights(filename)

    def saveValueModel(self, filename):
        self.valueModel.save_weights(filename)

    def _buildPolicyModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def _buildValueModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'mse'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model
