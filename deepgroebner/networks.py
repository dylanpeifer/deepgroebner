"""Neural networks for agents.

The two network classes are designed to be fast wrappers around tf.keras models.
In particular, they store their weights in NumPy arrays and do predict calls in
pure NumPy, which in testing is at least on order of magnitude faster than
TensorFlow when called repeatedly.
"""

import numpy as np
import scipy.special as sc
import tensorflow as tf

# NEW IMPORT
from threading import Thread

class MultilayerPerceptron:
    """A multilayer perceptron network with fast predict calls."""

    def __init__(self, input_dim, hidden_layers, output_dim, final_activation='softmax'):
        self.network = self._build_network(input_dim, hidden_layers, output_dim, final_activation)
        self.weights = self.get_weights()
        self.trainable_variables = self.network.trainable_variables
        self.final_activation = final_activation

    def predict(self, X, **kwargs):
        for i, (m, b) in enumerate(self.weights):
            X = np.dot(X, m) + b
            if i == len(self.weights)-1:
                if self.final_activation == 'softmax':
                    X = sc.softmax(X, axis=1)
            else:
                X = np.maximum(X, 0, X)
        return X

    def __call__(self, inputs):
        return self.network(inputs)

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)
        self.weights = self.get_weights()

    def get_weights(self):
        network_weights = self.network.get_weights()
        self.weights = []
        for i in range(len(network_weights)//2):
            m = network_weights[2*i]
            b = network_weights[2*i + 1]
            self.weights.append((m, b))
        return self.weights

    def _build_network(self, input_dim, hidden_layers, output_dim, final_activation):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
        for hidden in hidden_layers:
            model.add(tf.keras.layers.Dense(hidden, activation='relu'))
        model.add(tf.keras.layers.Dense(output_dim, activation=final_activation))
        return model


class ParallelMultilayerPerceptron:
    """A parallel multilayer perceptron network with fast predict calls."""

    def __init__(self, input_dim, hidden_layers):
        self.network = self._build_network(input_dim, hidden_layers)
        self.weights = self.get_weights()
        self.trainable_variables = self.network.trainable_variables

    def predict(self, X, **kwargs):
        for i, (m, b) in enumerate(self.weights):
            X = np.dot(X, m) + b
            if i == len(self.weights)-1:
                X = sc.softmax(X, axis=1).squeeze(axis=-1)
            else:
                X = np.maximum(X, 0, X)
        return X

    def __call__(self, inputs):
        return self.network(inputs)[0]

    def get_logits(self, inputs):
        return self.network(inputs)[1]

    def save_weights(self, filename):
        self.network.save_weights(filename)

    def load_weights(self, filename):
        self.network.load_weights(filename)
        self.weights = self.get_weights()

    def get_weights(self):
        network_weights = self.network.get_weights()
        self.weights = []
        for i in range(len(network_weights)//2):
            m = network_weights[2*i].squeeze(axis=0)
            b = network_weights[2*i + 1]
            self.weights.append((m, b))
        return self.weights

    def _build_network(self, input_dim, hidden_layers):
        inputs = tf.keras.Input(shape=(None, input_dim))
        x = inputs
        for hidden in hidden_layers:
            x = tf.keras.layers.Conv1D(hidden, 1, activation='relu')(x)
        outputs = tf.keras.layers.Conv1D(1, 1, activation='linear')(x)
        x = tf.keras.layers.Flatten()(outputs)
        probs = tf.keras.layers.Activation('softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=[probs, outputs])

#--------------------------------------------------------## Implementation for Transformers

# I assume batch length is always 1
# TODO: 
#   2) Decoding Block

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_layer):
        super(SelfAttention, self).__init__()
        self.query_weight = tf.keras.layers.Dense(hidden_layer)
        self.key_weight = tf.keras.layers.Dense(hidden_layer)
        self.value_weight = tf.keras.layers.Dense(hidden_layer)
        self.input_size = input_dim

    def attention(self, Q, K, V):
        similarities = tf.math.divide(tf.matmul(Q, tf.transpose(K)), tf.constant([self.input_size**(1/2)]))
        attention_weights = tf.nn.softmax(similarities)
        weighted_vectors = tf.matmul(attention_weights, V)
        return weighted_vectors

    def __call__(self, input_set):
        Q = tf.squeeze(self.query_weight(input_set), axis = 0)
        K = tf.squeeze(self.key_weight(input_set), axis = 0)
        V = tf.squeeze(self.value_weight(input_set), axis = 0)

        weighted_vectors = tf.expand_dims(self.attention(Q,K,V), axis = 0)
        
        return weighted_vectors

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, input_dim):
        super(MultiHeadSelfAttention, self).__init__()
        
        if input_dim % num_heads != 0:
            raise Exception('InputDimensionAndNumHeadError: input_dim must be divisble by num_heads')


        self.selfAttentionLayer = []
        for _ in range(num_heads):
            self.selfAttentionLayer.append(SelfAttention(input_dim, input_dim/num_heads))
        
        self.num_heads = num_heads
        self.final_layer = tf.keras.layers.Dense(input_dim)

    def attention(self, input_set, attention_layer, results, index):
        results[index] = tf.squeeze(attention_layer(input_set), axis = 0)

    def __call__(self, input_set):
        results = [None] * self.num_heads
        threads = []
        for i in range(self.num_heads):
            process = Thread(target = self.attention, args=[input_set, self.selfAttentionLayer[i], results, i])
            process.start()
            threads.append(process)

        for process in threads:
            process.join()

        head_concat = tf.expand_dims(tf.concat(results, 1), axis = 0)

        return self.final_layer(head_concat)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, input_dim, feed_forward_hidden_size, training: bool, rate = .1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(num_heads, input_dim)
        self.first_lt = tf.keras.layers.Dense(feed_forward_hidden_size, activation = 'relu')
        self.second_lt = tf.keras.layers.Dense(input_dim)

        self.layer_norm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.training = training

    def __call__(self, input_set):

        input_set = tf.cast(input_set, tf.float32)

        attention_output = self.attention(input_set)
        attention_output = self.dropout1(attention_output, training= self.training) # Only use drop out when training
        att_norm = self.layer_norm_mha(input_set + attention_output)

        ff = self.second_lt(self.first_lt(att_norm))
        ff = self.dropout1(ff, training = self.training)
        encoder_output = self.layer_norm_ff(input_set + ff)
        return encoder_output

class TransformersEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, input_dim, feed_forward_hidden_size):
        super(TransformersEncoder, self).__init__()
        self.encoding_layer = [EncoderLayer(num_heads, input_dim, feed_forward_hidden_size) for _ in range(num_layers)]
        self.num_layers = num_layers

    def __call__(self, input_set):
        for i in range(self.num_layers):
            input_set = self.encoding_layer[i](input_set)
        return input_set
        
#--------------------------------------------------------## End of Transformers


#--------------------------------------------------------## Implementation of Pointer Network
class pnetEncoder(tf.keras.layers.Layer):
    ''' 
    Encoder for pointer network. 
    '''

    def __init__(self, hidden_layer):
        '''
        Constructor for pointer network LSTM. 
        
        Args-
            hidden_layer - the output dimension of the output
        '''
        super(pnetEncoder, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_layer, return_sequences=True, return_state=True)
        self.hidden_size = hidden_layer
    
    def __call__(self, input_seq):
        '''
        Run LSTM

        Args -
            input_seq - input set

        Return output sequences, final memory state, final cell state
        '''
        return self.lstm(tf.cast(input_seq, tf.float32))

class pointer(tf.keras.layers.Layer):
    '''
    Pointer network attention mechanism. This will also handle the one decode step
    '''

    def __init__(self, input_dim, hidden_layer):
        '''
        Constructor for pointer based attention.

        Attribute -
            decoder_lstm: Only used for starting token which is a fixed random vector
            encoder_weight: Matrix to project encoder output during attention
            decode_weight: Matrix to project decode output (the start token)
            v: Used to project encode and decode into a 1d number
            tanh: activation function for attention
            input_size: size of input
            hidden_size: number of hidden nodes
        '''

        super(pointer, self).__init__()
        self.decoder_lstm = tf.keras.layers.LSTM(hidden_layer, return_sequences=True)
        self.encoder_weight = tf.keras.layers.Dense(hidden_layer)
        self.decode_weight = tf.keras.layers.Dense(hidden_layer)
        self.v = tf.keras.layers.Dense(1)
        self.tanh = tf.keras.activations.tanh
        self.input_size = input_dim
        self.hidden_size = hidden_layer
    
    def initStartToken(self, input_dim):
        '''
        '''
        np.random.seed(42)
        start_token = tf.convert_to_tensor(np.random.random([1,1,self.input_size]).astype(np.float32))
        return start_token

    def __call__(self, encoder_output, initial_states):
        '''
        '''
        start_token = self.initStartToken(self.input_size)
        lstm_decoder_output = self.decoder_lstm(start_token, initial_state=initial_states) # Start generating
        similarity_score = np.zeros([encoder_output.shape[1]])
        lstm_decoder_projection = self.decode_weight(lstm_decoder_output)
        for index, _ in enumerate(similarity_score):
            e_i = tf.expand_dims(encoder_output[0][index], axis = 0)
            intermediate_value = self.tanh(self.encoder_weight(e_i) + lstm_decoder_projection)
            u_i = self.v(intermediate_value)
            similarity_score[index] = u_i[0][0][0]
        return tf.nn.softmax(tf.convert_to_tensor(similarity_score))

class PointerNetwork(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_layer, input_edit = 'lstm'):
        '''
        '''
        super(PointerNetwork, self).__init__()
        if input_edit == 'lstm':
            self.encoder = pnetEncoder(hidden_layer)
        elif input_edit == 'self-attention':
            self.selfAttention = SelfAttention(input_dim, hidden_layer)
        self.point = pointer(input_dim, hidden_layer)
        self.hidden_size = hidden_layer
        self.encoding_type = input_edit

    def predict(self, input):
        '''
        '''
        if(self.encoding_type == 'lstm'):
            seq_output, mem_state, carry_state = self.encoder(input)
            initial_states = [mem_state, carry_state]
        elif(self.encoding_type == 'self-attention'):
            seq_output = self.selfAttention(input)
            h0 = tf.convert_to_tensor(np.random.random([1,self.hidden_size]).astype(np.float32))
            c0 = tf.convert_to_tensor(np.random.random([1,self.hidden_size]).astype(np.float32))
            initial_states = [h0, c0]

        prob_dist = tf.expand_dims(self.point(seq_output, initial_states), axis = 0)

        return prob_dist
#-------------------------------------------------------## End of Pointer Network


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



def ValueRNN(input_dim, units, cell='lstm', bidirectional=True):
    """Return an RNN value network for LeadMonomialsWrapper environments."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[None, input_dim]))
    if cell == 'simple':
        layer = tf.keras.layers.SimpleRNN(units)
    elif cell == 'lstm':
        layer = tf.keras.layers.LSTM(units)
    elif cell == 'gru':
        layer = tf.keras.layers.GRU(units)
    else:
        raise ValueError('unknown cell type')
    if bidirectional:
        model.add(tf.keras.layers.Bidirectional(layer))
    else:
        model.add(layer)
    model.add(tf.keras.layers.Dense(1))
    return model


def PolicyRNN(input_dim, units):
    """Return an RNN policy network for LeadMonomialsWrapper environments."""
    inputs = tf.keras.layers.Input(shape=[None, input_dim])
    X, h = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)(inputs)
    h = tf.keras.layers.Reshape([units, 1])(h)
    outputs = tf.nn.softmax(tf.squeeze(tf.matmul(X, h), axis=[-1]))
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def AtariNetSmall(input_shape, action_size, final_activation='linear'):
    """Return the network from the first DQN paper."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: x / 255.0))
    model.add(tf.keras.layers.Conv2D(16, 8, strides=4, activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size, activation=final_activation))
    return model


def AtariNetLarge(input_shape, action_size, final_activation='linear'):
    """Return the network from the second DQN paper."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Lambda(lambda x: x / 255.0))
    model.add(tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size, activation=final_activation))
    return model
