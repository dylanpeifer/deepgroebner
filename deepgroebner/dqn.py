"""A deep Q-Network agent."""

import numpy as np
import tensorflow as tf
import time


class CyclicMemory:
    """A cyclic buffer to store transition memories."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(shape=capacity, dtype=np.object)
        self.position = 0
        self.size = 0

    def push(self, transition):
        """Save a transition."""
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Return batch_size transitions taken uniformly at random with replacement."""
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]

    def __len__(self):
        return self.size


class DQNAgent:
    """A deep Q network agent."""

    def __init__(self, network, learning_rate=0.00025,
                 memory_capacity=100000, batch_size=32,
                 gamma=0.99,
                 epsilon_min=0.01, decay_rate=0.999, decay_mode='exponential',
                 start_steps=1000, replay_freq=1, target_update_freq=100, epsilon_decay_freq=1,
                 double=False):
        self.action_size = network.output_shape[1]
        self.onlineModel = self._buildModel(network, learning_rate)
        self.targetModel = self._buildModel(network, learning_rate)
        
        self.memory = CyclicMemory(memory_capacity)
        self.batch_size = batch_size
        
        self.gamma = gamma
        
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.decay_mode = decay_mode

        self.steps = 0
        self.start_steps = start_steps
        self.replay_freq = replay_freq
        self.target_update_freq = target_update_freq
        self.epsilon_decay_freq = epsilon_decay_freq
        
        self.double = double

    def act(self, state, epsilon=None):
        """Choose an action for the given state using an epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.predict(state))

    def predict(self, state, model='online'):
        """Return the predicted Q-values for the given state."""
        if model == 'online':
            return self.onlineModel.predict(np.expand_dims(state, axis=0))[0]
        elif model == 'target':
            return self.targetModel.predict(np.expand_dims(state, axis=0))[0]
        else:
            raise ValueError

    def predictBatch(self, states, model='online'):
        """Return the predicted Q-values for a batch of states."""
        if model == 'online':
            return self.onlineModel.predict(states)
        elif model == 'target':
            return self.targetModel.predict(states)
        else:
            raise ValueError

    def remember(self, state, action, reward, next_state, done):
        """Add a state transition to memory."""
        self.memory.push((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Train on batch_size transitions from memory."""
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        pred_online = self.predictBatch(states, model='online')
        next_pred_target = self.predictBatch(next_states, model='target')     
        if self.double:
            next_pred_online = self.predictBatch(next_states, model='online')
            update = rewards + (dones == False) * self.gamma * next_pred_target[np.arange(next_pred_target.shape[0]), np.argmax(next_pred_online, axis=1)]
        else:
            update = rewards + (dones == False) * self.gamma * np.max(next_pred_target, axis=1)   
        pred_online[np.arange(pred_online.shape[0]), actions] = update

        self.onlineModel.fit(states, pred_online, verbose=0)

    def decayEpsilon(self):
        """Decrease the value of epsilon."""
        if self.decay_mode == 'linear':
            self.epsilon = max(self.epsilon_min, self.epsilon - self.decay_rate)
        elif self.decay_mode == 'exponential':
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * self.decay_rate
        else:
            raise ValueError

    def updateTargetModel(self):
        """Copy online model weights to target model."""
        self.targetModel.set_weights(self.onlineModel.get_weights())
        
    def test(self, env, episodes, epsilon=None, render=False):
        """Test the agent for given episodes on given environment."""
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state, epsilon=epsilon)
                state, reward, done, _ = env.step(action)
                rewards[i] += reward
                if render:
                    env.render()
                    time.sleep(0.05)
        return rewards
    
    def explore(self, env, steps, epsilon=None):
        """Add transitions to the agent's memory without training."""
        while steps > 0:
            state = env.reset()
            done = False
            while not done:
                action = self.act(state, epsilon=epsilon)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                steps -= 1
                if steps <= 0:
                    break

    def train(self, env, steps, epochs=1, verbose=0, test_env=None, test_episodes=None, test_epsilon=None, savefile=None):
        """Train the agent for given steps on given environment."""
        if self.steps < self.start_steps:
            self.explore(env, self.start_steps - self.steps)
            self.steps = self.start_steps

        rewards = np.zeros(epochs)
        state = env.reset()
        done = False
        for epoch in range(epochs):
            epoch_steps = 0
            while epoch_steps < steps:
                if done:
                    state = env.reset()
                    done = False
                while not done:
                    action = self.act(state)
                    next_state, reward, done, _ = env.step(action) 
                    self.remember(state, action, reward, next_state, done)
                    state = next_state

                    self.steps += 1
                    if self.steps % self.replay_freq == 0:
                        self.replay(self.batch_size)
                    if self.steps % self.target_update_freq == 0:
                        self.updateTargetModel()
                    if self.steps % self.epsilon_decay_freq == 0:
                        self.decayEpsilon()

                    epoch_steps += 1
                    if epoch_steps >= steps:
                        break
                        
            if test_episodes is not None:
                rewards[epoch] = np.mean(self.test(test_env, test_episodes, epsilon=test_epsilon))       
            if savefile is not None:
                self.saveOnlineModel(str(epoch) + savefile)     
            if verbose == 1:
                print("\rEpoch {}/{} - avg_reward: {}".format(epoch+1, epochs, rewards[epoch]), end="")
                
        return rewards

    def saveOnlineModel(self, filename):
        self.onlineModel.save_weights(filename)

    def loadOnlineModel(self, filename):
        self.onlineModel.load_weights(filename)

    def _buildModel(self, network, learning_rate):
        model = tf.keras.models.clone_model(network)
        loss = 'mse'
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(loss=loss, optimizer=optimizer)
        return model
