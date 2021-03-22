"""Neural networks for agents."""

import numpy as np
import tensorflow as tf


class MultilayerPerceptron(tf.keras.Model):
    """A basic multilayer perceptron network.

    This network is used for the policy and value models when training on CartPole-v0,
    CartPole-v1, and LunarLander-v2.

    Parameters
    ----------
    output_dim : int
        Positive integer output dimension of the network.
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation used for the hidden layers.
    final_activation : {'log_softmax', 'softmax', 'linear', 'exponential'}, optional
        Activation used for the final output layer.

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
        return self.final_layer(X)


class ParallelEmbeddingLayer(tf.keras.layers.Layer):
    """A layer for computing a nonlinear embedding of non-negative integer feature vectors.

    This layer is used with the LeadMonomialsEnv to embed the exponent vectors of pairs
    into feature vectors. Each vector is embedded independently by a single learned multilayer
    perceptron. A mask is generated and attached to the output based on padding by -1.

    Parameters
    ----------
    embed_dim : int
        Positive integer output dimension of the embedding.
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation used for the hidden layers.
    final_activation : {'relu', 'linear', 'exponential'}, optional
        Activation used for the final output embedding layer.

    """

    def __init__(self, embed_dim, hidden_layers, activation='relu', final_activation='relu'):
        super(ParallelEmbeddingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(embed_dim, activation=final_activation)

    def call(self, batch):
        """Return the embedding for this batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.int32` and shape (batch_dim, padded_dim, input_dim)
            Input batch, with padded rows indicated by -1 and all other values non-negative.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, embed_dim)
            Embedding of the input batch with attached mask indicating valid rows.

        """
        X = tf.cast(batch, tf.float32)
        for layer in self.hidden_layers:
            X = layer(X)
        output = self.final_layer(X)
        return output

    def compute_mask(self, batch, mask=None):
        return tf.math.not_equal(batch[:, :, -1], -1)


class RecurrentEmbeddingLayer(tf.keras.layers.Layer):
    """A layer for computing a nonlinear embedding of non-negative integer feature vectors.

    This layer is used with the LeadMonomialsEnv to embed the exponent vectors of pairs
    into feature vectors. An RNN is used so vector embeddings can depend on other vectors.
    A mask is generated and attached to the output based on padding by -1.

    Parameters
    ----------
    embed_dim : int
        Positive integer output dimension of the embedding.
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    cell : {'gru', 'lstm'}
        Type of recurrent cell.

    """

    def __init__(self, embed_dim, hidden_layers, cell='gru', need_mask = True):
        super(RecurrentEmbeddingLayer, self).__init__()
        cell_fn = tf.keras.layers.GRU if cell == 'gru' else tf.keras.layers.LSTM
        self.hidden_layers = [cell_fn(u, return_sequences=True) for u in hidden_layers]
        self.final_layer = cell_fn(embed_dim, return_sequences=True, return_state=True)
        self.supports_masking = need_mask

    def call(self, batch, initial_state = None):
        """Return embedding and final hidden states for this batch.

        Parameters
        ----------
        batch : `Tensor` of type tf.int32` and shape (batch_dim, padded_dim, feature_dim)
            Input batch, with padded rows indicated by -1 and all other values non-negative.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, embed_dim)
            Embedding of the input batch with mask indicating valid rows.
        state : `Tensor` or pair of `Tensor`s of type `tf.float32` and shape (batch_dim, embed_dim)

        """
        mask = self.compute_mask(batch) if self.supports_masking else None
        X = tf.cast(batch, tf.float32) if not batch.dtype == tf.float32 else batch

        for layer in self.hidden_layers:
            X, *state = layer(X, mask=mask)
        output, *state = self.final_layer(X, mask=mask, initial_state=initial_state)
        return (output, *state)

    def compute_mask(self, batch, mask=None):
        return tf.math.not_equal(batch[:, :, -1], -1)


class MonomialEncodingLayer(tf.keras.layers.Layer):
    """A layer for computing a vocab-like embedding of monomials.

    This layer is used with the LeadMonomialsEnv to map pairs to feature vectors.
    It does this by learning an embedding on monomials and concatenating the embeddings
    for the the monomials shown for each pair. Not all monomials are embeddable - any
    monomial with a variable raised to a power greater than `max_power` is mapped to the
    same fixed overflow value. This means that this layer has generally worse
    performance than the ParallelEmbeddingLayer.

    Parameters
    ----------
    dim : int
        Dimension of the embedded monomial vectors (output dim will be 2 * dim * k).
    n : int 
        Number of variables in the polynomials.
    k : int
        Number of lead terms shown in each polynomial.
    max_power : int
        Maximum power of a variable in an embeddable monomial.

    """

    def __init__(self, dim, n=3, k=1, max_power=10):
        super(MonomialEncodingLayer, self).__init__()
        self.dim = dim
        self.n = n
        self.k = k
        self.max_power = max_power
        self.embed = tf.keras.layers.Embedding((max_power + 1) ** n + 1, dim, input_length=2*k)

    def call(self, batch):
        """Return the embedding for this batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.int32` and shape (batch_dim, padded_dim, 2 * n * k)
            Input batch, with padded rows indicated by -1 and all other values non-negative.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, 2 * dim * k)
            Embedding of the input batch with attached mask indicating valid rows.

        """
        batch += tf.cast(tf.math.equal(batch, -1), tf.int32)
        batch_size = tf.shape(batch)[0]
        monomials = tf.reshape(batch, (batch_size, -1, self.n))
        powers = tf.math.cumprod(tf.fill((self.n, 1), self.max_power + 1), exclusive=True)
        values = tf.squeeze(tf.matmul(monomials, powers))
        valid = tf.cast(tf.reduce_max(monomials, axis=-1) <= self.max_power, tf.int32)
        encoded = values * valid + (1 - valid) * (self.max_power + 1) ** self.n
        encoded = tf.reshape(encoded, (batch_size, -1, 2 * self.k))
        return tf.reshape(self.embed(encoded), (batch_size, -1, 2 * self.k * self.dim))

    def compute_mask(self, batch, mask=None):
        return tf.math.not_equal(batch[:, :, -1], -1)


class DenseProcessingLayer(tf.keras.layers.Layer):
    """A simple processing stack that applies dense layers.

    Parameters
    ----------
    output_dim : int
       Positive integer output dimension of this layer.
    hidden_layers : list
       List of positive integer hidden layer dimensions
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation used for the hidden layers and output.    

    """

    def __init__(self, output_dim, hidden_layers, activation='relu'):
        super(DenseProcessingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(output_dim, activation=activation)
        self.supports_masking = True

    def call(self, batch):
        """Return the processed output for this batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, input_dim)
            Input batch with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, output_dim)
            Processed batch with attached mask passed through.

        """
        X = batch
        for layer in self.hidden_layers:
            X = layer(X)
        output = self.final_layer(X)
        return output


class SelfAttentionLayer(tf.keras.layers.Layer):
    """A multi head self attention layer.

    Adapted from https://www.tensorflow.org/tutorials/text/transformer.

    Parameters
    ----------
    dim : int
        Positive integer dimension.
    n_heads : int, optional
        Positive integer number of heads (must divide `dim`).

    """

    def __init__(self, dim, n_heads=1):
        super(SelfAttentionLayer, self).__init__()
        assert dim % n_heads == 0, "number of heads must divide dimension"
        self.dim = dim
        self.n_heads = n_heads
        self.depth = dim // n_heads
        self.Wq = tf.keras.layers.Dense(dim)
        self.Wk = tf.keras.layers.Dense(dim)
        self.Wv = tf.keras.layers.Dense(dim)
        self.dense = tf.keras.layers.Dense(dim)
        self.supports_masking = True

    def call(self, batch, mask=None):
        """Return the processed batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Input batch with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Processed batch with mask passed through.

        """
        batch_size = tf.shape(batch)[0]
        Q = self.split_heads(self.Wq(batch), batch_size)
        K = self.split_heads(self.Wk(batch), batch_size)
        V = self.split_heads(self.Wv(batch), batch_size)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        X, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        X = tf.transpose(X, perm=[0, 2, 1, 3])
        X = tf.reshape(X, (batch_size, -1, self.dim))
        output = self.dense(X)
        return output

    def split_heads(self, batch, batch_size):
        """Return batch reshaped for multihead attention."""
        X = tf.reshape(batch, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(X, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Return calculated vectors and attention weights.

        Parameters
        ----------
        Q : `Tensor` of type `tf.float32' and shape (..., dq, d1)
            Tensor of queries as rows.
        K : `Tensor` of type `tf.float32` and shape (..., dkv, d1)
            Tensor of keys as rows.
        V : `Tensor` of type `tf.float32` and shape (..., dkv, d2)
            Tensor of values as rows.
        mask : `Tensor of type `tf.bool' and shape (..., 1, dkv)
            The mask representing valid key/value rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (..., dq, d2)
            Processed batch of Q, K, V.
        attention_weights : `Tensor` of type `tf.float32` and shape (..., dq, dkv)
            Attention weights from intermediate step.

        """
        QK = tf.matmul(Q, K, transpose_b=True)
        d = tf.cast(tf.shape(K)[-1], tf.float32)
        attention_logits = QK / tf.math.sqrt(d)
        if mask is not None:
            attention_logits += tf.cast(~mask, tf.float32) * -1e9
        attention_weights = tf.nn.softmax(attention_logits)
        output = tf.matmul(attention_weights, V)
        return output, attention_weights


class TransformerLayer(tf.keras.layers.Layer):
    """A transformer encoder layer.

    Parameters
    ----------
    dim : int
        Positive integer dimension of the attention layer and output.
    hidden_dim : int
        Positive integer dimension of the feed forward hidden layer.
    n_heads : int, optional
        Positive integer number of heads in attention layer (must divide `dim`).
    dropout : float, optional
        Dropout rate.

    """

    def __init__(self, dim, hidden_dim, n_heads=1, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttentionLayer(dim, n_heads=n_heads)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(dim)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.supports_masking = True

    def call(self, batch, mask=None, training=False):
        """Return the processed batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Input batch with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, dim)
            Processed batch with mask passed through.

        """
        X1 = self.attention(batch, mask=mask)
        X1 = self.dropout1(X1, training=training)
        X1 = self.layer_norm1(batch + X1, training=training)
        X2 = self.dense2(self.dense1(X1))
        X2 = self.dropout2(X2, training=training)
        output = self.layer_norm2(X1 + X2, training=training)
        return output


class ParallelDecidingLayer(tf.keras.layers.Layer):
    """A layer for computing probability distributions over arbitrary numbers of rows.

    This layer is used following an embedding and processing of the state of a LeadMonomialsWrapper
    to produce the policy probabilities for each available action. The layer learns a single function
    implemented with a multilayer perceptron that computes scores independently for each row. These
    scores are softmaxed to produce the probabilites.

    Parameters
    ----------
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the hidden layers.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

    """

    def __init__(self, hidden_layers, activation='relu', final_activation='log_softmax'):
        super(ParallelDecidingLayer, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(u, activation=activation) for u in hidden_layers]
        self.final_layer = tf.keras.layers.Dense(1, activation='linear')
        self.final_activation = tf.nn.log_softmax if final_activation == 'log_softmax' else tf.nn.softmax

    def call(self, batch, mask=None):
        """Return probability distributions over rows of batch.

        Parameters
        ----------
        batch : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim, feature_dim)
            Batch of feature vectors with attached mask indicating valid rows.

        Returns
        -------
        output : `Tensor` of type `tf.float32` and shape (batch_dim, padded_dim)
            Softmaxed probability distributions over valid rows.

        """
        X = batch
        for layer in self.hidden_layers:
            X = layer(X)
        X = tf.squeeze(self.final_layer(X), axis=-1)
        if mask is not None:
            X += tf.cast(~mask, tf.float32) * -1e9
        output = self.final_activation(X)
        return output


class PointerDecidingLayer(tf.keras.layers.Layer):
    """Pointer network attention mechanism. This will also handle the one decode step
    
    input shape is (batch_dim, padded_dim, feature_dim)
    output shape is (batch_dim, padded_dim)

        Attribute -
            decoder_lstm: Only used for starting token which is a fixed random vector
            encoder_weight: Matrix to project encoder output during attention
            decode_weight: Matrix to project decode output (the start token)
            v: Used to project encode and decode into a 1d number
            tanh: activation function for attention
            input_size: size of input
            hidden_size: number of hidden nodes
    """

    def __init__(self, input_dim, embed_dim, hidden_layers, 
                        layer_type = 'gru', dot_product_attention = True, prob = 'log'):
        super(PointerDecidingLayer, self).__init__()
        self.decoder_layer = RecurrentEmbeddingLayer(embed_dim, hidden_layers, cell = layer_type, need_mask = False)
        self.softmax = tf.nn.log_softmax if prob == 'log' else tf.nn.softmax
        self.dot_prod_attention = dot_product_attention
        self.input_size = input_dim
        self.start_token = tf.ones([1, 1, self.input_size])
        if not dot_product_attention:
            self.encoder_weight = tf.keras.layers.Dense(embed_dim)
            self.decode_weight = tf.keras.layers.Dense(embed_dim)
            self.v = tf.keras.layers.Dense(1)
            self.tanh = tf.keras.activations.tanh

    def call(self, encoder_output, initial_states=None, mask = None):
        '''
        Calculate attention.
        
        Params:
            encoder_output: output of the encoder block
                size: (batch, seq_len, input_dim)
            initial_states: hidden and cell states from the encoder block
       '''
        batch_size = tf.shape(encoder_output)[0]
        start_token = self.initialize_start_token(batch_size)
        lstm_decoder_output,*state = self.decoder_layer(start_token, initial_state=initial_states) #(batch, 1, input_dim)
        if self.dot_prod_attention:
            attention_scores = tf.squeeze(tf.linalg.matmul(lstm_decoder_output, encoder_output, transpose_b = True), axis = 1) + tf.cast(~mask, tf.float32) * -1e9
            return self.softmax(attention_scores)
        else:
            pad_dim = encoder_output.shape[1]
            lstm_decoder_projection = self.decode_weight(lstm_decoder_output) # (batch_size, 1, embed_dim)
            encoder_project = self.encoder_weight(encoder_output) # (batch_size, padd_dim, embed_dim)
            similarity_score = self.v(self.tanh(encoder_project + tf.tile(lstm_decoder_projection, [1, pad_dim, 1])))
            return self.softmax(tf.squeeze(tf.reshape(similarity_score, [batch_size, 1, pad_dim]), axis = 1) + tf.cast(~mask, tf.float32) * -1e9)

    def initialize_start_token(self, batch_size):
        '''
        Initialize start token

        Params:
            batch_size: size of batch
        '''
        return tf.tile(self.start_token, [batch_size, 1, 1])
    
    
        #np.random.seed(42)
        #tf.random.set_seed(42)
        #start_token = tf.random.uniform([batch_size,1,self.input_size])
        #start_token = tf.convert_to_tensor(np.random.random([batch_size,1,self.input_size]).astype(np.float32))
        #np.random.seed()
        #tf.random.set_seed()
        #return start_token


class ParallelMultilayerPerceptron(tf.keras.Model):
    """A parallel multilayer perceptron network.

    This model expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    hidden_layers : list
        List of positive integer hidden layer dimensions.
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the hidden layers.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

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
        self.embedding = ParallelEmbeddingLayer(hidden_layers[-1], hidden_layers[:-1],
                                                activation=activation, final_activation=activation)
        self.deciding = ParallelDecidingLayer([], final_activation=final_activation)

    def call(self, batch):
        X = self.embedding(batch)
        X = self.deciding(X)
        return X


class AttentionPMLP(tf.keras.Model):
    """A parallel multilayer perceptron network with attention.

    This model expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    dim : int
        Positive integer dimension of the attention layer.
    n_heads : int, optional
        Positive integer number of heads in attention layer (must divide `dim`).
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the embedding.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

    """

    def __init__(self, dim, n_heads=1, activation='relu', final_activation='log_softmax'):
        super(AttentionPMLP, self).__init__()
        self.embedding = ParallelEmbeddingLayer(dim, [], final_activation=activation)
        self.trans = SelfAttentionLayer(dim, n_heads=n_heads)
        self.deciding = ParallelDecidingLayer([], final_activation=final_activation)

    def call(self, batch):
        X = self.embedding(batch)
        X = self.trans(X)
        X = self.deciding(X)
        return X


class TransformerPMLP(tf.keras.Model):
    """A parallel multilayer perceptron network with a transformer layer.

    This model expects an input with shape (batch_dim, padded_dim, feature_dim), where
    entries are non-negative integers and padding is by -1. It returns a tensor
    of shape (batch_dim, padded_dim) where each batch is a softmaxed distribution over the rows
    with zero probability on any padded row.

    Parameters
    ----------
    dim : int
        Positive integer dimension of the transformer attention layer.
    hidden_dim : int
        Positive integer dimension of the transformer hidden feedforward layer.
    n_heads : int, optional
        Positive integer number of heads in attention layer (must divide `dim`).
    activation : {'relu', 'selu', 'elu', 'tanh', 'sigmoid'}, optional
        Activation for the embedding.
    final_activation : {'log_softmax', 'softmax'}, optional
        Activation for the final output layer.

    """

    def __init__(self, dim, hidden_dim, n_heads=1, activation='relu', final_activation='log_softmax'):
        super(TransformerPMLP, self).__init__()
        self.embedding = ParallelEmbeddingLayer(dim, [], final_activation=activation)
        self.attn = TransformerLayer(dim, hidden_dim, n_heads=n_heads)
        self.deciding = ParallelDecidingLayer([], final_activation=final_activation)

    def call(self, batch, training=False):
        X = self.embedding(batch)
        X = self.attn(X, training=training)
        X = self.deciding(X)
        return X


class PointerNetwork(tf.keras.Model):
    """Recurrent embedding followed by pointer."""

    def __init__(self, input_dim, hidden_layers:list, embed_dim, 
                            cell_type = 'gru', dot_prod_attention=True, prob = 'log'):
        '''
        Params:
            input_dim: dimension of input
            hidden_layer: dimension of output layer
            input_layer: lstm or gru
            dot_prod_attention: dot product attention or traditional pointer network attention
        '''
        super(PointerNetwork, self).__init__()
        self.encoder = RecurrentEmbeddingLayer(embed_dim, hidden_layers)
        self.pointer = PointerDecidingLayer(input_dim, embed_dim, hidden_layers, cell_type, dot_prod_attention, prob)

    def call(self, input):
        '''
        '''
        X, *state = self.encoder(input)
        log_prob = self.pointer(X, state)
        return log_prob


class CustomLSTM(tf.keras.layers.Layer):
    def __init__(self, hl_out_size):
        super(CustomLSTM, self).__init__()
        self.Whi = tf.keras.layers.Dense(hl_out_size, activation='sigmoid')
        self.Whf = tf.keras.layers.Dense(hl_out_size, activation='sigmoid')
        self.Whg = tf.keras.layers.Dense(hl_out_size, activation='tanh')
        self.Who = tf.keras.layers.Dense(hl_out_size, activation='sigmoid')
    def __call__(self, input, cell_state):
        i_t = self.Whi(input)
        f_t = self.Whf(input)
        g_t = self.Whg(input)
        o_t = self.Who(input)
        c_t = tf.math.multiply(f_t,cell_state) + tf.math.multiply(i_t, g_t)
        h_t = tf.math.multiply(o_t, tf.tanh(c_t))
        return h_t, c_t


class ProcessBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_layer, num_step):
        '''
        Constructor for ProcessBlock.

        @Params:
            hidden_layer: Size of embeddings and LSTM
            num_step: Number of times I process the input
        '''
        super(ProcessBlock, self).__init__()
        self.embed = tf.keras.layers.Dense(hidden_layer)
        self.process_block = CustomLSTM(hidden_layer)
        self.hidden_size = hidden_layer
        self.num_step = num_step
        self.supports_masking = True

    def read_out(self, M, q, c,batch_size, mask = None):
        '''
        Perform the attention with the memory vectors and pass by the LSTM

        @Params:
            M - Polynomial Embeddings
            q - Query vector (last hidden state)
            c - Last cell state
            batch_size - size of batch
        '''
        logits = tf.squeeze(tf.linalg.matmul(M, q, transpose_b = True), axis = 2)
        attention = tf.expand_dims(tf.nn.softmax(logits + tf.cast(~mask, tf.float32) * -1e9), axis = 2)
        r_t = tf.linalg.matmul(attention, M, transpose_a = True) # sum of weight memory vector (by attention)
        q_star_t = tf.concat([q, r_t], axis = 2)
        mem_state, cell_state = self.process_block(q_star_t, c)
        return mem_state, cell_state

    def initHiddenState(self, batch_size):
        '''
        Initialize initial hidden states and cell state. 

        @Params:
            batch_size - size of batch
        '''
        np.random.seed(42)
        hidden_state = tf.convert_to_tensor(np.random.random([batch_size,1,self.hidden_size]).astype(np.float32)) # Change this
        cell_state = tf.convert_to_tensor(np.random.random([batch_size,1,self.hidden_size]).astype(np.float32)) # Change this
        np.random.seed()
        return hidden_state, cell_state

    def call(self, input_seq):
        '''
        Calculate embedding.

        @Params:
            input_seq - input sequence
        '''
        mask = self.compute_mask(input_seq)
        initial_state, cell_state = self.initHiddenState(input_seq.shape[0])

        # Start processing
        for _ in range(self.num_step):
            initial_state, cell_state = self.read_out(input_seq, initial_state, cell_state, input_seq.shape[0], mask)
        return initial_state, cell_state

    def compute_mask(self, batch, mask=None):
        return tf.math.not_equal(batch[:, :, -1], -1)


class PBPointerNet(tf.keras.Model):
    def __init__(self, input_dim, embed_dim, num_write_outs, hidden_layers = [], 
                        embedding_hidden_layers = [], layer_type = 'gru', dot_product_attention = True, prob = 'log'):
        super(PBPointerNet, self).__init__()
        self.embed = ParallelEmbeddingLayer(embed_dim, embedding_hidden_layers)
        self.pointer = PointerDecidingLayer(input_dim, embed_dim, hidden_layers, layer_type = 'lstm')
        self.processBlock = ProcessBlock(embed_dim, num_write_outs)

    def __call__(self, input):
        X = self.embed(input)
        hidden_state, cell_state = self.processBlock(X)
        hidden_state = tf.squeeze(hidden_state, axis = 1)
        cell_state = tf.squeeze(cell_state, axis = 1)
        log_prob = self.pointer(X, initial_states = [hidden_state, cell_state])
        return log_prob


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


class RecurrentValueModel(tf.keras.Model):

    def __init__(self, units):
        super(RecurrentValueModel, self).__init__()
        self.embedding = ParallelEmbeddingLayer(units, [])
        self.rnn = tf.keras.layers.LSTM(units)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, batch):
        return self.dense(self.rnn(self.embedding(batch)))


class GlobalSumPooling1D(tf.keras.layers.Layer):

    def __init__(self):
        super(GlobalSumPooling1D, self).__init__()

    def call(self, batch, mask=None):
        if mask is not None:
            batch = batch * tf.cast(tf.expand_dims(mask, -1), tf.float32)
        return tf.reduce_sum(batch, axis=-2)


class PoolingValueModel(tf.keras.Model):

    def __init__(self, hidden_layers1, hidden_layers2, method='max'):
        super(PoolingValueModel, self).__init__()
        self.embedding = ParallelEmbeddingLayer(hidden_layers1[-1], hidden_layers1[:-1])
        if method == 'max':
            self.pooling = tf.keras.layers.GlobalMaxPooling1D()
        elif method == 'mean':
            self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        elif method == 'sum':
            self.pooling = GlobalSumPooling1D()
        else:
            raise ValueError('invalid method')
        self.hidden_layers = [tf.keras.layers.Dense(u, activation='relu') for u in hidden_layers2]
        self.final_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, batch):
        X = self.pooling(self.embedding(batch))
        for layer in self.hidden_layers:
            X = layer(X)
        return self.final_layer(X)


@tf.function
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Return calculated vectors and attention weights.

    Parameters
    ----------
    Q : `Tensor` of type `tf.float32' and shape (..., dq, d1)
        Tensor of queries as rows.
    K : `Tensor` of type `tf.float32` and shape (..., dkv, d1)
        Tensor of keys as rows.
    V : `Tensor` of type `tf.float32` and shape (..., dkv, d2)
        Tensor of values as rows.
    mask : `Tensor of type `tf.bool' and shape (..., 1, dkv)
        The mask representing valid key/value rows.

    Returns
    -------
    output : `Tensor` of type `tf.float32` and shape (..., dq, d2)
        Processed batch of Q, K, V.
    attention_weights : `Tensor` of type `tf.float32` and shape (..., dq, dkv)
        Attention weights from intermediate step.

    """
    QK = tf.matmul(Q, K, transpose_b=True)
    d = tf.cast(tf.shape(K)[-1], tf.float32)
    attention_logits = QK / tf.math.sqrt(d)
    if mask is not None:
        attention_logits += tf.cast(~mask, tf.float32) * -1e9
    attention_weights = tf.nn.softmax(attention_logits)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights


class AttentionPoolingLayer(tf.keras.layers.Layer):
    
    def __init__(self, dim):
        super(AttentionPoolingLayer, self).__init__()
        self.dim = dim
        self.Wk = tf.keras.layers.Dense(dim)
        self.Wv = tf.keras.layers.Dense(dim)
        self.dense = tf.keras.layers.Dense(1)
    
    def build(self, batch_input_shape):
        self.Q = self.add_weight(name='query',
                                 shape=[1, self.dim],
                                 initializer='glorot_normal')
        super(AttentionPoolingLayer, self).build(batch_input_shape)

    def call(self, batch, mask=None):
        K = self.Wk(batch)
        V = self.Wv(batch)
        if mask is not None:
            mask = mask[:, tf.newaxis, tf.newaxis, :]
        X, attn_weights = scaled_dot_product_attention(self.Q, K, V, mask=mask)
        return tf.squeeze(self.dense(X), axis=-1)
