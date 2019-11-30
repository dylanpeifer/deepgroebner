"""A proximal policy optimization agent that supports changing state shapes.

"""

import numpy as np
import tensorflow as tf


def discount_rewards(rewards, gam):
    """Discount the list or array of rewards by gamma in-place.

    Parameters
    ----------
    rewards : list or ndarray of ints or floats
        A list of rewards from a single complete trajectory.
    gam : float
        The discount rate.

    Returns
    -------
    rewards : list or ndarray
        The input array with each reward replaced by discounted reward-to-go.

    Examples
    --------
    >>> rewards = [1, 2, 3, 4, 5]
    >>> discount_rewards(rewards, 0.5)
    [1, 2, 6.25, 6.5, 5]

    Note that the input rewards list is modified in place. The return
    value is just a reference to the original list.

    >>> rewards = [1, 2, 3, 4, 5]
    >>> discounted_rewards = discount_rewards(rewards, 0.5)
    >>> rewards
    [1, 2, 6.25, 6.5, 5]

    """
    cumulative_reward = 0
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gam * cumulative_reward
        rewards[i] = cumulative_reward
    return rewards


class TrajectoryBuffer:
    """A buffer to store and compute with trajectories.

    The buffer is used to store information from each step of interaction
    between the agent and environment. When a trajectory is finished it
    computes the discounted rewards and generalized advantage estimates. After
    some number of trajectories are finished it can return batches grouped by
    state shape with normalized advantage estimates.

    Parameters
    ----------
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.

    See Also
    --------
    discount_rewards : Discount the list or array of rewards by gamma in-place.

    Notes
    -----
    The implementation is based on the implementation of buffers used in the
    policy gradient algorithms from OpenAI Spinning Up. Formulas for
    generalized advantage estimation are from [1]_. The major implementation
    difference is that we allow for different sized states and action
    dimensions, and only assume that each state shape corresponds to some fixed
    action dimension.

    References
    ----------
    .. [1] Schulman et al, "High-Dimensional Continuous Control Using
       Generalized Advantage Estimation," ICLR 2016.

    Examples
    --------
    >>> buffer = TrajectoryBuffer()
    >>> tau = [(np.array([3]), np.array([0.4, 0.3, 0.3]), 3, 1, 1),
    ...        (np.array([1, 3, 7]), np.array([0.1, 0.9]), 2, 0, 0),
    ...        (np.array([1, 4, 2]), np.array([0.3, 0.7]), 1, 2, 2),
    ...        (np.array([2, 5]), np.array([0.4, 0.6]), 2, 1, 1),
    ...        (np.array([1, 7]), np.array([0.9, 0.1]), 0, 0, 1)]
    >>> for t in tau:
    ...     buffer.store(*t)
    >>> buffer.finish()
    >>> buffer.get()

    """

    def __init__(self, gam=0.99, lam=0.97):
        self.gam = gam
        self.lam = lam
        self.states = []
        self.probas = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.start = 0
        self.end = 0

    def store(self, state, proba, value, action, reward):
        """Store the information from one interaction with the environment.

        Parameters
        ----------
        state : ndarray
           The observation of the state.
        proba : ndarray
           The agent's computed probability distribution on actions.
        value : float
           The agent's computed value of the state.
        action : int
           The chosen action in this trajectory.
        reward : float
           The reward received in the next transition.

        """
        self.states.append(state)
        self.probas.append(proba)
        self.values.append(value)
        self.actions.append(action)
        self.rewards.append(reward)
        self.end += 1

    def finish(self):
        """Finish an episode and compute advantages and discounted rewards in-place.

        Advantages are stored in place of `values` and discounted rewards are
        stored in place of `rewards` for the current trajectory.
        """
        tau = slice(self.start, self.end)
        rewards = np.array(self.rewards[tau], dtype=np.float)
        values = np.array(self.values[tau], dtype=np.float)
        delta = rewards - values
        delta[:-1] += self.gam * values[1:]
        self.rewards[tau] = list(discount_rewards(rewards, self.gam))
        self.values[tau] = list(discount_rewards(delta, self.gam * self.lam))
        self.start = self.end

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.probas.clear()
        self.values.clear()
        self.actions.clear()
        self.rewards.clear()
        self.start = 0
        self.end = 0

    def get(self, normalize_advantages=True):
        """Return a dictionary of state shapes to training data.

        Parameters
        ----------
        normalize_advantages : bool, optional
            Whether to normalize the returned advantages.

        Returns
        -------
        data : dict
            A dictionary mapping state shape to training data.

            Each value of the dictionary is a dictionary with keys
            'states', 'probas', 'values', 'actions', 'advants', and values
            ndarrays.

        """
        advantages = np.array(self.values[:self.start])
        if normalize_advantages:
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages)
        shapes = {}
        for i, state in enumerate(self.states[:self.start]):
            shapes.setdefault(state.shape, []).append(i)
        data = {}
        for shape, indices in shapes.items():
            data[shape] = {
                'states': np.array([self.states[i] for i in indices],
                                   dtype=np.float32),
                'probas': np.array([self.probas[i] for i in indices],
                                   dtype=np.float32),
                'values': np.array([[self.rewards[i]] for i in indices],
                                   dtype=np.float32),
                'actions': np.array([self.actions[i] for i in indices],
                                    dtype=np.int),
                'advants': np.array([advantages[i] for i in indices],
                                    dtype=np.float32),
            }
        return data

    def __len__(self):
        return len(self.states)


@tf.function(experimental_relax_shapes=True)
def pg_surrogate_loss(new_probs, old_probs, actions, advantages):
    """Return loss with gradient for policy gradient.

    Parameters
    ----------
    new_probs : Tensor (batch_dim, action_dim)
        The output of the current model.
    old_probs : Tensor (batch_dim, action_dim)
        The stored output from interaction.
    actions : Tensor (batch_dim,)
        The chosen actions.
    advantages : Tensor (batch_dim,)
        The computed advantages.

    Returns
    -------
    loss : Tensor (batch_dim,)
        The loss for each interaction.

    """
    action_dim = tf.shape(new_probs)[1]
    new_pi = tf.reduce_sum(tf.one_hot(actions, action_dim) * new_probs, axis=1)
    return -tf.math.log(new_pi) * advantages


def ppo_surrogate_loss(eps=0.2):
    """Return loss function with gradient for proximal policy optimization.

    Parameters
    ----------
    eps : float
        The clip ratio for PPO.

    """
    @tf.function(experimental_relax_shapes=True)
    def loss(new_probs, old_probs, actions, advantages):
        """Return loss with gradient for proximal policy optimization.

        Parameters
        ----------
        new_probs : Tensor (batch_dim, action_dim)
            The output of the current model.
        old_probs : Tensor (batch_dim, action_dim)
            The stored output from interaction.
        actions : Tensor (batch_dim,)
            The chosen actions.
        advantages : Tensor (batch_dim,)
            The computed advantages.

        Returns
        -------
        loss : Tensor (batch_dim,)
            The loss for each interaction.
        """
        action_dim = tf.shape(new_probs)[1]
        pi_new = tf.reduce_sum(tf.one_hot(actions, action_dim) * new_probs, axis=1)
        pi_old = tf.reduce_sum(tf.one_hot(actions, action_dim) * old_probs, axis=1)
        min_adv = tf.where(advantages > 0, (1 + eps) * advantages, (1 - eps) * advantages)
        return -tf.minimum(pi_new / pi_old * advantages, min_adv)
    return loss


def print_status_bar(i, total, history, verbose=1):
    """Print a formatted status line."""
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in history])
    end = "\n" if verbose == 2 or i == total else ""
    print("\rEpoch {}/{}".format(i, total) + metrics, end=end)


class PPOAgent:
    """A proximal policy optimization agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    policy_lr : float, optional
        The learning rate for the policy model.
    policy_updates : int, optional
        The number of policy updates per epoch of training.
    value_network : network, optional
        The network for the value model.
    value_lr : float, optional
        The learning rate for the value model.
    value_updates : int, optional
        The number of value updates per epoch of training.
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.
    normalize : bool, optional
        Whether to normalize advantages.
    eps : float, optional
        The clip ratio for PPO.
    action_dim_fn : function, optional
        The function that maps state shape to action dimension.
    """

    def __init__(self,
                 policy_network, policy_lr=1e-4, policy_updates=1,
                 value_network=None, value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 action_dim_fn=lambda s: s[0]):
        self.policy_model = policy_network
        self.policy_loss = ppo_surrogate_loss(eps)
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = tf.keras.losses.mse
        self.value_optimizer = tf.keras.optimizers.Adam(lr=value_lr)
        self.value_updates = value_updates

        self.buffer = TrajectoryBuffer(gam=gam, lam=lam)
        self.normalize_advantages = normalize_advantages
        self.action_dim_fn = action_dim_fn

    def act(self, state, greedy=False, return_probs=False):
        """Return an action for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.
        greedy : bool, optional
            Whether to sample or pick the action with max probability.
        return_probs : bool, optional
            Whether to return the probability vector.
        """
        probs = self.policy_model.predict(state[np.newaxis])[0]
        action = np.argmax(probs) if greedy else np.random.choice(len(probs), p=probs)
        return (action, probs) if return_probs else action

    def train(self, env, episodes=10, epochs=1,
              max_episode_length=None, verbose=0, save_freq=1, logdir=None):
        """Train the agent.

        Parameters
        ----------
        env : environment
            The environment to train on.
        episodes : int, optional
            The number of episodes to perform per epoch of training.
        epochs : int, optional
            The number of epochs to train.
        max_episode_length : int, optional
            The maximum number of steps of interaction in an episode.
        verbose : int, optional
            How much information to print to the user.
        save_freq : int, optional
            How often to save the model weights, measured in epochs.
        logdir : str, optional
            The directory to store Tensorboard logs and model weights.

        Returns
        -------
        history : dict
            Dictionary with statistics from training.

        """
        tb_writer = None if logdir is None else tf.summary.create_file_writer(logdir)
        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs)}

        for i in range(epochs):
            self.buffer.clear()
            return_history = self.run_episodes(env, max_episode_length=max_episode_length, store=True)
            batches = self.buffer.get(normalize_advantages=self.normalize_advantages)
            policy_history = self._fit_policy_model(batches, epochs=self.policy_updates)
            value_history = self._fit_value_model(batches, epochs=self.value_updates)

            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])

            if logdir is not None and i % save_freq == 0:
                self.save_policy_weights(logdir + "/policy-" + str(i) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i) + ".h5")
            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar('mean_returns', history['mean_returns'][i], step=i)
                    tf.summary.scalar('min_returns', history['min_returns'][i], step=i)
                    tf.summary.scalar('max_returns', history['max_returns'][i], step=i)
                    tf.summary.scalar('std_returns', history['std_returns'][i], step=i)
                    tf.summary.histogram('returns', return_history['returns'], step=i)
                tb_writer.flush()
            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)
    
    def run_episode(self, env, max_episode_length=None, greedy=False, store=False):
        """Run an episode and return total reward and episode length.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        max_episode_length : int, optional
            The maximum number of interactions before the episode ends.
        greedy : bool, optional
            Whether to choose the maximum probability or sample.
        store : bool, optional
            Whether to store the interactions in the agent's buffer.

        Returns
        -------
        (total_reward, episode_length) : (float, int)
            The total nondiscounted reward obtained in this episode and the
            episode length.

        """
        state = env.reset()
        done = False
        episode_length = 0
        total_reward = 0
        while not done:
            action, probs = self.act(state, return_probs=True)
            next_state, reward, done, _ = env.step(action)
            value = 0 if self.value_model is None else self.value_model.predict(state[np.newaxis])[0][0]
            if store:
                self.buffer.store(state, probs, value, action, reward)
            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state
        self.buffer.finish()
        return total_reward, episode_length

    def run_episodes(self, env, episodes=100, max_episode_length=None, greedy=False, store=False):
        """Run several episodes, store interaction in buffer, and return history.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        episodes : int, optional
            The number of episodes to perform.
        max_episode_length : int, optional
            The maximum number of steps before the episode is terminated.
        verbose : int, optional
            The amount of information to print.
        
        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """
        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}
        for i in range(episodes):
            R, L = self.run_episode(env, max_episode_length=max_episode_length, greedy=greedy, store=store)
            history['returns'][i] = R
            history['lengths'][i] = L
        return history

    def _fit_policy_model(self, batches, epochs=1):
        """Fit policy model with one gradient update per epoch."""
        history = {'loss': np.zeros(epochs)}
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                losses = []
                for shape, data in batches.items():
                    if self.action_dim_fn(shape) == 1:
                        continue
                    probs = self.policy_model(data['states'])
                    losses.append(self.policy_loss(probs, data['probas'], data['actions'], data['advants']))
                loss = tf.reduce_mean(tf.concat(losses, axis=0))
            varis = self.policy_model.trainable_variables
            grads = tape.gradient(loss, varis)
            self.policy_optimizer.apply_gradients(zip(grads, varis))
            history['loss'][epoch] = loss
        return history

    def load_policy_weights(self, filename):
        """Load weights from filename into the policy model."""
        self.policy_model.load_weights(filename)

    def save_policy_weights(self, filename):
        """Save the current weights in the policy model to filename."""
        self.policy_model.save_weights(filename)

    def _fit_value_model(self, batches, epochs=1):
        """Fit value model with one gradient update per epoch."""
        if self.value_model is None:
            epochs = 0
        history = {'loss': np.zeros(epochs)}
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                losses = []
                for shape, data in batches.items():
                    values = self.value_model(data['states'])
                    losses.append(self.value_loss(values, data['values']))
                loss = tf.reduce_mean(tf.concat(losses, axis=0))
            varis = self.value_model.trainable_variables
            grads = tape.gradient(loss, varis)
            self.value_optimizer.apply_gradients(zip(grads, varis))
            history['loss'][epoch] = loss
        return history

    def load_value_weights(self, filename):
        """Load weights from filename into the value model."""
        if self.value_model is not None:
            self.value_model.load_weights(filename)

    def save_value_weights(self, filename):
        """Save the current weights in the value model to filename."""
        if self.value_model is not None:
            self.value_model.save_weights(filename)
