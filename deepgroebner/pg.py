"""Policy gradient agents that support changing state shapes.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy gradient) and proximal policy optimization
agent.
"""

import numpy as np
import multiprocessing as mp
import tensorflow as tf


PACKET_SIZE = 10 # this must divide the number of episodes, and ideally should divide (episodes)/(number of cores)


def discount_rewards(rewards, gam):
    """Return discounted rewards-to-go computed from inputs.

    Parameters
    ----------
    rewards : array_like
        List or 1D array of rewards from a single complete trajectory.
    gam : float
        Discount rate.

    Returns
    -------
    rewards : ndarray
        1D array of discounted rewards-to-go.

    Examples
    --------
    >>> rewards = [1, 2, 3, 4, 5]
    >>> discount_rewards(rewards, 0.5)
    [1, 2, 6.25, 6.5, 5]

    """
    cumulative_reward = 0
    discounted_rewards = np.zeros_like(rewards, dtype=np.float)
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gam * cumulative_reward
        discounted_rewards[i] = cumulative_reward
    return discounted_rewards


def compute_advantages(rewards, values, gam, lam):
    """Return generalized advantage estimates computed from inputs.

    Parameters
    ----------
    rewards : array_like
        List or 1D array of rewards from a single complete trajectory.
    values : array_like
        List or 1D array of value predictions from a single complete trajectory.
    gam : float
        Discount rate.
    lam : float
        Parameter for generalized advantage estimation.

    Returns
    -------
    advantages : ndarray
        1D array of computed advantage scores.

    References
    ----------
    .. [1] Schulman et al, "High-Dimensional Continuous Control Using
       Generalized Advantage Estimation," ICLR 2016.

    Examples
    --------
    >>> rewards = [1, 1, 1, 1, 1]
    >>> values = [0, 0, 0, 0, 0]
    >>> compute_advantages(rewards, values, 0.5, 0.5)
    array([1.33203125, 1.328125  , 1.3125    , 1.25      , 1.        ])

    """
    rewards = np.array(rewards, dtype=np.float)
    values = np.array(values, dtype=np.float)
    delta = rewards - values
    delta[:-1] += gam * values[1:]
    return discount_rewards(delta, gam * lam)


class TrajectoryBuffer:
    """A buffer to store and compute with trajectories.

    The buffer is used to store information from each step of interaction
    between the agent and environment. When a trajectory is finished it
    computes the discounted rewards and generalized advantage estimates. After
    some number of trajectories are finished it can return a tf.Dataset of the
    training data for policy gradient algorithms.

    Parameters
    ----------
    gam : float, optional
        Discount rate.
    lam : float, optional
        Parameter for generalized advantage estimation.

    See Also
    --------
    discount_rewards : Discount the list or array of rewards by gamma in-place.
    compute_advantages : Return generalized advantage estimates computed from inputs.

    """

    def __init__(self, gam=0.99, lam=0.97):
        self.gam = gam
        self.lam = lam
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.start = 0  # index to start of current episode
        self.end = 0  # index to one past end of current episode

    def store(self, state, action, reward, logprob, value):
        """Store the information from one interaction with the environment.

        Parameters
        ----------
        state : ndarray
           Observation of the state.
        action : int
           Chosen action in this trajectory.
        reward : float
           Reward received in the next transition.
        logprob : float
           Agent's logged probability of picking the chosen action.
        value : float
           Agent's computed value of the state.

        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.end += 1

    def finish(self):
        """Finish an episode and compute advantages and discounted rewards.

        Advantages are stored in place of `values` and discounted rewards are
        stored in place of `rewards` for the current trajectory.
        """
        tau = slice(self.start, self.end)
        rewards = discount_rewards(self.rewards[tau], self.gam)
        values = compute_advantages(self.rewards[tau], self.values[tau], self.gam, self.lam)
        self.rewards[tau] = rewards
        self.values[tau] = values
        self.start = self.end

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.values.clear()
        self.start = 0
        self.end = 0

    def get(self, batch_size=64, normalize_advantages=True, sort=False, drop_remainder=True):
        """Return a tf.Dataset of training data from this TrajectoryBuffer.

        Parameters
        ----------
        batch_size : int, optional
            Batch size in the returned tf.Dataset.
        normalize_advantages : bool, optional
            Whether to normalize the returned advantages.
        sort : bool, optional
            Whether to sort by state shape before batching to minimize padding.
        drop_remainder : bool, optional
            Whether to drop the last batch if it has fewer than batch_size elements.

        Returns
        -------
        dataset : tf.Dataset

        """
        actions = np.array(self.actions[:self.start], dtype=np.int32)
        logprobs = np.array(self.logprobs[:self.start], dtype=np.float32)
        advantages = np.array(self.values[:self.start], dtype=np.float32)
        values = np.array(self.rewards[:self.start], dtype=np.float32)

        if normalize_advantages:
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages)

        if self.states and self.states[0].ndim == 2:

            # filter out any states with only one action available
            indices = [i for i in range(len(self.states[:self.start])) if self.states[i].shape[0] != 1]
            states = [self.states[i].astype(np.int32) for i in indices]
            actions = actions[indices]
            logprobs = logprobs[indices]
            advantages = advantages[indices]
            values = values[indices]

            if sort:
                indices = np.argsort([s.shape[0] for s in states])
                states = [states[i] for i in indices]
                actions = actions[indices]
                logprobs = logprobs[indices]
                advantages = advantages[indices]
                values = values[indices]

            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_generator(lambda: states, tf.int32),
                tf.data.Dataset.from_tensor_slices(actions),
                tf.data.Dataset.from_tensor_slices(logprobs),
                tf.data.Dataset.from_tensor_slices(advantages),
                tf.data.Dataset.from_tensor_slices(values),
            ))
            if batch_size is None:
                batch_size = len(states)
            padded_shapes = ([None, self.states[0].shape[1]], [], [], [], [])
            padding_values = (tf.constant(-1, dtype=tf.int32),
                              tf.constant(0, dtype=tf.int32),
                              tf.constant(0.0, dtype=tf.float32),
                              tf.constant(0.0, dtype=tf.float32),
                              tf.constant(0.0, dtype=tf.float32))
            dataset = dataset.padded_batch(batch_size,
                                           padded_shapes=padded_shapes,
                                           padding_values=padding_values,
                                           drop_remainder=drop_remainder)

        else:
            states = np.array(self.states[:self.start], dtype=np.float32)
            dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(states),
                tf.data.Dataset.from_tensor_slices(actions),
                tf.data.Dataset.from_tensor_slices(logprobs),
                tf.data.Dataset.from_tensor_slices(advantages),
                tf.data.Dataset.from_tensor_slices(values),
            ))
            if batch_size is None:
                batch_size = len(states)
            dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        return dataset

    def __len__(self):
        return len(self.states)


def _merge_buffers(bufferlist):
    output = bufferlist[0]
    assert output.start == output.end, "Must apply self.finish() before merging buffers"
    for b in bufferlist[1:]:
        assert b.start == b.end, "Must apply self.finish() before merging buffers"
        output.states += b.states
        output.actions += b.actions
        output.rewards += b.rewards
        output.logprobs += b.logprobs
        output.values += b.values
    output.end = len(output.states)
    output.start = output.end
    return output


def print_status_bar(i, epochs, history, verbose=1):
    """Print a formatted status line."""
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in ['mean_returns']])
    end = "\n" if verbose == 2 or i+1 == epochs else ""
    print("\rEpoch {}/{}".format(i+1, epochs) + metrics, end=end)


class Agent:
    """Abstract base class for policy gradient agents.

    All functionality for policy gradient is implemented in this
    class. Derived classes must define the property `policy_loss`
    which is used to train the policy.

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
    normalize_advantages : bool, optional
        Whether to normalize advantages.
    kld_limit : float, optional
        The limit on KL divergence for early stopping policy updates.
    """

    def __init__(self,
                 policy_network, policy_lr=1e-4, policy_updates=1,
                 value_network=None, value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=0.01):
        self.policy_model = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = tf.keras.losses.mse
        self.value_optimizer = tf.keras.optimizers.Adam(lr=value_lr)
        self.value_updates = value_updates

        self.lam = lam
        self.gam = gam
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam)
        self.normalize_advantages = normalize_advantages
        self.kld_limit = kld_limit

    @tf.function(experimental_relax_shapes=True)
    def act(self, state, return_logprob=False):
        """Return an action for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.
        return_logp : bool, optional
            Whether to return the log probability of choosing the chosen action.

        """
        logpi = self.policy_model(state[tf.newaxis])
        action = tf.random.categorical(logpi, 1)[0, 0]
        if return_logprob:
            return action, logpi[:, action][0]
        else:
            return action

    @tf.function(experimental_relax_shapes=True)
    def value(self, state):
        """Return the predicted value for the given state using the value model.

        Parameters
        ----------
        state : np.array
            The state of the environment.

        """
        return self.value_model(state[tf.newaxis])[0][0]

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, parallel=True, batch_size=64):
        """Train the agent on env.

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
        parallel : bool, optional
            Whether to run parallel rollouts.
        batch_size : int or None, optional
            The batch sizes for training (None indicates one large batch).

        Returns
        -------
        history : dict
            Dictionary with statistics from training.

        """
        tb_writer = None if logdir is None else tf.summary.create_file_writer(logdir)
        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs),
                   'mean_ep_lens': np.zeros(epochs),
                   'min_ep_lens': np.zeros(epochs),
                   'max_ep_lens': np.zeros(epochs),
                   'std_ep_lens': np.zeros(epochs),
                   'policy_updates': np.zeros(epochs),
                   'delta_policy_loss': np.zeros(epochs),
                   'policy_ent': np.zeros(epochs),
                   'policy_kld': np.zeros(epochs)}

        for i in range(epochs):
            self.buffer.clear()
            return_history = self.run_episodes(
                env, episodes=episodes, max_episode_length=max_episode_length,
                store=True, parallel=parallel
            )
            dataset = self.buffer.get(normalize_advantages=self.normalize_advantages, batch_size=batch_size)
            policy_history = self._fit_policy_model(dataset, epochs=self.policy_updates)
            value_history = self._fit_value_model(dataset, epochs=self.value_updates)

            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])
            history['mean_ep_lens'][i] = np.mean(return_history['lengths'])
            history['min_ep_lens'][i] = np.min(return_history['lengths'])
            history['max_ep_lens'][i] = np.max(return_history['lengths'])
            history['std_ep_lens'][i] = np.std(return_history['lengths'])
            history['policy_updates'][i] = len(policy_history['loss'])
            history['delta_policy_loss'][i] = policy_history['loss'][-1] - policy_history['loss'][0]
            history['policy_ent'][i] = policy_history['ent'][-1]
            history['policy_kld'][i] = policy_history['kld'][-1]

            if logdir is not None and (i+1) % save_freq == 0:
                self.save_policy_weights(logdir + "/policy-" + str(i+1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i+1) + ".h5")
            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar('mean_returns', history['mean_returns'][i], step=i)
                    tf.summary.scalar('min_returns', history['min_returns'][i], step=i)
                    tf.summary.scalar('max_returns', history['max_returns'][i], step=i)
                    tf.summary.scalar('std_returns', history['std_returns'][i], step=i)
                    tf.summary.scalar('mean_ep_lens', history['mean_ep_lens'][i], step=i)
                    tf.summary.scalar('min_ep_lens', history['min_ep_lens'][i], step=i)
                    tf.summary.scalar('max_ep_lens', history['max_ep_lens'][i], step=i)
                    tf.summary.scalar('std_ep_lens', history['std_ep_lens'][i], step=i)
                    tf.summary.histogram('returns', return_history['returns'], step=i)
                    tf.summary.histogram('lengths', return_history['lengths'], step=i)
                    tf.summary.scalar('policy_updates', history['policy_updates'][i], step=i)
                    tf.summary.scalar('delta_policy_loss', history['delta_policy_loss'][i], step=i)
                    tf.summary.scalar('policy_ent', history['policy_ent'][i], step=i)
                    tf.summary.scalar('policy_kld', history['policy_kld'][i], step=i)
                tb_writer.flush()
            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """Run an episode and return total reward and episode length.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        max_episode_length : int, optional
            The maximum number of interactions before the episode ends.
        buffer : TrajectoryBuffer object, optional
            If included, it will store the whole rollout in the given buffer.

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
            if state.dtype == np.float64:
                state = state.astype(np.float32)
            action, logprob = self.act(state, return_logprob=True)
            if self.value_model is None:
                value = 0
            elif self.value_model == 'env':
                value = env.value(gamma=self.gam)
            else:
                value = self.value(state)
            next_state, reward, done, _ = env.step(action.numpy())
            if buffer is not None:
                buffer.store(state, action, reward, logprob, value)
            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state
        if buffer is not None:
            buffer.finish()
        return total_reward, episode_length

    def _parallel_run_episode(self, env, max_episode_length, random_seed, output, packet_size):
        np.random.seed(random_seed)
        buff = TrajectoryBuffer(gam=self.gam, lam=self.lam)
        results = []
        for i in range(packet_size):
            results.append(self.run_episode(env, max_episode_length=max_episode_length, buffer=buff))
        output.put((results,buff))

    def run_episodes(self, env, episodes=100, max_episode_length=None, store=False, parallel=True):
        """Run several episodes, store interaction in buffer, and return history.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        episodes : int, optional
            The number of episodes to perform.
        max_episode_length : int, optional
            The maximum number of steps before the episode is terminated.
        store : bool, optional
            Whether or not to store the rollout in self.buffer.
        parallel : bool, optional

        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """
        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}
        if parallel:
            output = mp.Queue()
            assert episodes % PACKET_SIZE == 0, "PACKET_SIZE must divide the number of episodes"
            num_processes = int(episodes / PACKET_SIZE)
            processes = [mp.Process(target=self._parallel_run_episode,
                                    args=(env, max_episode_length, seed, output, PACKET_SIZE))
                         for seed in np.random.randint(0, 4294967295, num_processes)]
            for p in processes:
                p.start()
            results = [output.get() for p in processes]
            for p in processes:
                p.join()
            self.buffer=_merge_buffers([b for (_, b) in results])
            returns = [x for (t,_) in results for x in t]
            for i in range(episodes):
                (history['returns'][i], history['lengths'][i]) = returns[i]
        else:
            for i in range(episodes):
                R, L = self.run_episode(env, max_episode_length=max_episode_length, buffer=self.buffer)
                history['returns'][i] = R
                history['lengths'][i] = L

        return history

    def _fit_policy_model(self, dataset, epochs=1):
        """Fit policy model using data from dataset."""
        history = {'loss': [], 'kld': [], 'ent': []}
        for epoch in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            for states, actions, logprobs, advantages, _ in dataset:
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(states, actions, logprobs, advantages)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1
            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            if self.kld_limit is not None and kld > self.kld_limit:
                break
        return {k: np.array(v) for k, v in history.items()}

    @tf.function(experimental_relax_shapes=True)
    def _fit_policy_model_step(self, states, actions, logprobs, advantages):
        """Fit policy model on one batch of data."""
        with tf.GradientTape() as tape:
            logpis = self.policy_model(states)
            new_logprobs = tf.reduce_sum(tf.one_hot(actions, tf.shape(logpis)[1]) * logpis, axis=1)
            loss = tf.reduce_mean(self.policy_loss(new_logprobs, logprobs, advantages))
            kld = tf.reduce_mean(logprobs - new_logprobs)
            ent = -tf.reduce_mean(new_logprobs)
        varis = self.policy_model.trainable_variables
        grads = tape.gradient(loss, varis)
        self.policy_optimizer.apply_gradients(zip(grads, varis))
        return loss, kld, ent

    def load_policy_weights(self, filename):
        """Load weights from filename into the policy model."""
        self.policy_model.load_weights(filename)

    def save_policy_weights(self, filename):
        """Save the current weights in the policy model to filename."""
        self.policy_model.save_weights(filename)

    def _fit_value_model(self, dataset, epochs=1):
        """Fit value model using data from dataset."""
        if self.value_model is None or self.value_model == 'env':
            epochs = 0
        history = {'loss': []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for states, _, _, _, values in dataset:
                batch_loss = self._fit_value_model_step(states, values)
                loss += batch_loss
                batches += 1
            history['loss'].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    @tf.function(experimental_relax_shapes=True)
    def _fit_value_model_step(self, states, values):
        """Fit value model on one batch of data."""
        with tf.GradientTape() as tape:
            pred_values = self.value_model(states)
            loss = tf.reduce_mean(self.value_loss(pred_values, values))
        varis = self.value_model.trainable_variables
        grads = tape.gradient(loss, varis)
        self.value_optimizer.apply_gradients(zip(grads, varis))
        return loss

    def load_value_weights(self, filename):
        """Load weights from filename into the value model."""
        if self.value_model is not None:
            self.value_model.load_weights(filename)

    def save_value_weights(self, filename):
        """Save the current weights in the value model to filename."""
        if self.value_model is not None:
            self.value_model.save_weights(filename)


@tf.function(experimental_relax_shapes=True)
def pg_surrogate_loss(new_logps, old_logps, advantages):
    """Return loss with gradient for policy gradient.

    Parameters
    ----------
    new_logps : Tensor (batch_dim,)
        The output of the current model for the chosen action.
    old_logps : Tensor (batch_dim,)
        The previous logged probability of the chosen action.
    advantages : Tensor (batch_dim,)
        The computed advantages.

    Returns
    -------
    loss : Tensor (batch_dim,)
        The loss for each interaction.

    """
    return -new_logps * advantages


class PGAgent(Agent):
    """A policy gradient agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.

    """

    def __init__(self, policy_network, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = pg_surrogate_loss


def ppo_surrogate_loss(eps=0.2):
    """Return loss function with gradient for proximal policy optimization.

    Parameters
    ----------
    eps : float
        The clip ratio for PPO.

    """
    @tf.function(experimental_relax_shapes=True)
    def loss(new_logps, old_logps, advantages):
        """Return loss with gradient for proximal policy optimization.

        Parameters
        ----------
        new_logps : Tensor (batch_dim,)
            The output of the current model for the chosen action.
        old_logps : Tensor (batch_dim,)
            The previous logged probability for the chosen action.
        advantages : Tensor (batch_dim,)
            The computed advantages.

        Returns
        -------
        loss : Tensor (batch_dim,)
            The loss for each interaction.
        """
        min_adv = tf.where(advantages > 0, (1 + eps) * advantages, (1 - eps) * advantages)
        return -tf.minimum(tf.exp(new_logps - old_logps) * advantages, min_adv)
    return loss


class PPOAgent(Agent):
    """A proximal policy optimization agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    eps : float, optional
        The clip ratio for PPO.

    """

    def __init__(self, policy_network, eps=0.2, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = ppo_surrogate_loss(eps)
