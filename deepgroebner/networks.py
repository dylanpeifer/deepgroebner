"""Neural networks for agents."""

import numpy as np
import tensorflow as tf


class MultilayerPerceptron(tf.keras.Model):
    """A basic multilayer perceptron network.

    Parameters
    ----------
    output_dim : int
        The output positive integer dimension of the network.
    hidden_layers : list
        The list of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        The activation used for the hidden layers.
    final_activation : {'log_softmax', 'softmax', 'linear', 'exponential'}
        The activation used for the final output layer.

    Examples
    --------
    >>> mlp = MultilayerPerceptron(2, [128])
    >>> states = tf.random.uniform((64, 4))
    >>> logprobs = mlp(states)
    >>> logprobs.shape
    TensorShape([64, 2])
    >>> actions = tf.random.categorical(logprobs, 1)
    >>> actions.shape
    TensorShape([64, 1])

    """

    def __init__(self, output_dim, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(MultilayerPerceptron, self).__init__()
        final_activation = tf.nn.log_softmax if final_activation == 'log_softmax' else final_activation
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(output_dim, activation=final_activation)

    def call(self, X):
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.final_layer(X)
        return X


class EmbeddingLayer(tf.keras.layers.Layer):
    """A layer for computing a nonlinear embedding of non-negative integer feature vectors.
    
    This layer expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor with shape
    (batch_dim, padded_dim, embed_dim) and a mask of shape (batch_dim, padded_dim) that
    indicates which rows were padded.

    Parameters
    ----------
    embed_dim : int
        The positive integer output dimension of the embedding.
    hidden_layers : list
        The list of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        The activation used for the hidden layers.
    final_activation : {'linear', 'exponential'}
        The activation used for the final output embedding layer.

    """

    def __init__(self, embed_dim, hidden_layers, activation='relu', final_activation='linear'):
        super(EmbeddingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(embed_dim, activation=final_activation)

    def call(self, X):
        mask = tf.cast(tf.math.equal(X[:, :, -1], -1), tf.float32)
        X = tf.cast(X, tf.float32)
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.final_layer(X)
        return X, mask


class DecidingLayer(tf.keras.layers.Layer):
    """A layer for computing softmaxed probability distributions over arbitrary numbers of rows.

    This layer expects input with shape (batch_dim, padded_dim, feature_dim) and a mask of
    shape (batch_dim, padded_dim) that indicates which rows were padded. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    hidden_layers : list
        The list of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        The activation for the hidden layers.
    final_activation : {'log_softmax', 'softmax'}, optional
        The activation for the final output embedding layer.

    """

    def __init__(self, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(DecidingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(1, activation='linear')
        self.final_activation = tf.nn.log_softmax if final_activation == 'log_softmax' else tf.nn.softmax

    def call(self, X, mask):
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.final_activation(tf.squeeze(self.final_layer(X), axis=-1) + mask * -1e9)
        return X


class ParallelMultilayerPerceptron(tf.keras.Model):
    """A parallel multilayer perceptron network.

    This model expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    hidden_layers : list
        The list of positive integer hidden layer dimensions.
    activation : {'relu', 'tanh'}, optional
        The activation for the hidden layers.
    final_activation : {'log_softmax', 'softmax'}, optional
        The activation for the final output layer.

    Examples
    --------
    >>> pmlp = ParallelMultilayerPerceptron([128])
    >>> states = tf.constant([
            [[ 0,  1],
             [ 3,  0],
             [-1, -1]],
            [[ 8,  5],
             [ 3,  3],
             [ 3,  5]],
            [[ 6,  7],
             [ 6,  8],
             [-1, -1]],
        ])
    >>> logprobs = pmlp(states)
    >>> logprobs.shape
    TensorShape([3, 3])
    >>> actions = tf.random.categorical(logprobs, 1)
    >>> actions.shape
    TensorShape([3, 1])

    """

    def __init__(self, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(ParallelMultilayerPerceptron, self).__init__()
        self.embedding = EmbeddingLayer(hidden_layers[-1], hidden_layers[:-1], activation=activation)
        self.decider = DecidingLayer([], final_activation=final_activation)

    def call(self, X):
        X, mask = self.embedding(X)
        X = self.decider(X, mask)
        return X


def scaled_dot_product_attention(Q, K, V, mask):
    """Return calculated vector and attention weights.

    Parameters
    ----------
    Q : tensor of shape (..., dq, d)
        Tensor of queries as rows.
    K : tensor of shape (..., dk, d)
        Tensor of keys as rows.
    V : tensor of shape (..., dk, dv)
        Tensor of values as rows.
    mask : boolean tensor of shape broadcastable to (..., dq, dk)
        The mask representing valid rows.

    Returns
    -------
    X : tensor
    attention_weights : tensor
    """

    QK = tf.matmul(Q, K, transpose_b=True)
    d = tf.cast(tf.shape(K)[-1], tf.float32)
    attention_logits = QK / tf.math.sqrt(d)
    if mask is not None:
        attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(attention_logits)
    X = tf.matmul(attention_weights, V)
    return X, attention_weights


class MultiHeadSelfAttentionLayer(tf.keras.layers.Layer):
    """A multi head self attention layer.

    Adapted from https://www.tensorflow.org/tutorials/text/transformer.

    """

    def __init__(self, dim, n_heads):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.depth = dim // n_heads
        self.Wq = tf.keras.layers.Dense(dim)
        self.Wk = tf.keras.layers.Dense(dim)
        self.Wv = tf.keras.layers.Dense(dim)
        self.dense = tf.keras.layers.Dense(dim)

    def split_heads(self, X, batch_size):
        X = tf.reshape(X, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(X, perm=[0, 2, 1, 3])

    def call(self, X, mask):
        batch_size = tf.shape(X)[0]
        Q = self.split_heads(self.Wq(X), batch_size)
        K = self.split_heads(self.Wk(X), batch_size)
        V = self.split_heads(self.Wv(X), batch_size)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dim))
        X = self.dense(concat_attention)
        return X, attention_weights


class TransformerPMLP(tf.keras.Model):
    """A parallel multilayer perceptron network with multihead self attention layer."""

    def __init__(self, embed_dim, embed_hl, num_heads, activation='relu', final_activation='log_softmax'):
        super(TransformerPMLP, self).__init__()
        self.embedding = EmbeddingLayer(embed_dim, embed_hl, activation=activation)
        self.mhsa = MultiHeadSelfAttentionLayer(embed_dim, num_heads)
        self.decider = DecidingLayer([], final_activation=final_activation)

    def call(self, X):
        X, mask = self.embedding(X)
        X, attention_weights = self.mhsa(X, mask)
        X = self.decider(X, mask)
        return X


class PairsLeftBaseline:
    """A Buchberger value network that returns discounted pairs left."""

    def __init__(self, gam=0.99):
        self.gam = gam
        self.trainable_variables = []

    def predict(self, X, **kwargs):
        states, pairs, *_ = X.shape
        if self.gam == 1:
            fill_value = - pairs
        else:
            fill_value = - (1 - self.gam ** pairs) / (1 - self.gam)
        return np.full((states, 1), fill_value)

    def __call__(self, inputs):
        return self.predict(inputs)

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass

    def get_weights(self):
        pass


class AgentBaseline:
    """A Buchberger value network that returns an agent's performance."""

    def __init__(self, agent, gam=0.99):
        self.agent = agent
        self.gam = gam
        self.trainable_variables = []

    def predict(self, env):
        env = env.copy()
        R = 0.0
        discount = 1.0
        state = (env.G, env.P) if hasattr(env, 'G') else env._matrix()
        done = False
        while not done:
            action = self.agent.act(state)
            state, reward, done, _ = env.step(action)
            R += reward * discount
            discount *= self.gam
        return R

    def __call__(self, inputs):
        return self.predict(inputs)

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass

    def get_weights(self):
        pass
