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
        '''
        Constructor for self attention.
        
        Params:
            input_dim: size of input size (number of features)
            hidden_layer: output size for dense layers (in transformers it will be input_dim / num_heads)
        '''
        super(SelfAttention, self).__init__()
        self.query_weight = tf.keras.layers.Dense(hidden_layer)
        self.key_weight = tf.keras.layers.Dense(hidden_layer)
        self.value_weight = tf.keras.layers.Dense(hidden_layer)
        self.input_size = input_dim

    def attention(self, Q, K, V):
        '''
        Calculate attention given Q, K, V matrix
        '''
        similarities = tf.math.divide(tf.linalg.matmul(Q, K, transpose_b=True), tf.constant([self.input_size**(1/2)]))
        attention_weights = tf.nn.softmax(similarities)
        weighted_vectors = tf.matmul(attention_weights, V)
        return weighted_vectors

    def __call__(self, q_input, k_input, v_input):
        '''
        Get weighted vectors.

        Params:
            q_input, k_input, v_input: Query, Key, Value matrix
            
        NOTE: For encoder stage all input will be the same. For decoder stage q_input will be from the start token
        and k_input = v_input from the encoder stage 
        '''
        Q = self.query_weight(q_input)
        K = self.key_weight(k_input)
        V = self.value_weight(v_input)

        weighted_vectors = self.attention(Q,K,V)
        
        return weighted_vectors

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, input_dim):
        '''
        Params:
            num_heads: Number of projections
            input_dim: Number of features of our input
        '''
        super(MultiHeadSelfAttention, self).__init__()
        
        if input_dim % num_heads != 0:
            raise Exception('InputDimensionAndNumHeadError: input_dim must be divisble by num_heads')


        self.selfAttentionLayer = []
        for _ in range(num_heads):
            self.selfAttentionLayer.append(SelfAttention(input_dim, input_dim/num_heads))
        
        self.num_heads = num_heads
        self.final_layer = tf.keras.layers.Dense(input_dim)

    def attention(self, q, k, v, attention_layer, results, index):
        '''
        Calculate the self attention of q, k, v (see __call__ in SelfAttention)

        Params:
            q, k, v: Query, Value, Key matrix
            attention_layer: ith SelfAttention layer
            results: list used to store the results of the threaded attention
            index: index that the output of attention_layer should go
        '''
        results[index] = attention_layer(q,k,v)

    def __call__(self, q, k, v):
        '''
        Get multi-headed attention, we thread the self attention layers

        Params:
            q, k, v: Query, Key, Value matrix
        '''
        results = [None] * self.num_heads
        threads = []
        for i in range(self.num_heads):
            process = Thread(target = self.attention, args=[q,k,v, self.selfAttentionLayer[i], results, i])
            process.start()
            threads.append(process)

        for process in threads:
            process.join()

        head_concat = tf.concat(results, 2) # Concatenate output

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

        attention_output = self.attention(input_set, input_set, input_set)
        attention_output = self.dropout1(attention_output, training= self.training) # Only use drop out when training
        att_norm = self.layer_norm_mha(input_set + attention_output)

        ff = self.second_lt(self.first_lt(att_norm))
        ff1 = self.dropout1(ff, training = self.training)
        encoder_output = self.layer_norm_ff(input_set + ff1)
        return encoder_output

class TransformersEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, input_dim, feed_forward_hidden_size, training):
        super(TransformersEncoder, self).__init__()
        self.encoding_layer = [EncoderLayer(num_heads, input_dim, feed_forward_hidden_size, training) for _ in range(num_layers)]
        self.num_layers = num_layers

    def __call__(self, input_set):
        for i in range(self.num_layers):
            input_set = self.encoding_layer[i](input_set)
        return input_set

# Useless stuff for right now
#-----------------------------------------------------------------------------------------------
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, input_dim, feed_forward_hidden_size):
        super(DecoderLayer, self).__init__()
        self.mha_1 = MultiHeadSelfAttention(num_heads, input_dim)
        self.mha_2 = MultiHeadSelfAttention(num_heads, input_dim)

        self.first_lt = tf.keras.layers.Dense(feed_forward_hidden_size, activation = 'relu')
        self.second_lt = tf.keras.layers.Dense(input_dim)

        self.layer_norm_mha_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_mah_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.layers.Dense(feed_forward_hidden_size, activation='relu')

        self.input_dim = input_dim

    def initDecodeInput(self, batch_size):
        np.random.seed(42)
        start_token = tf.convert_to_tensor(np.random.random([batch_size,1,self.input_dim]).astype(np.float32))
        np.random.seed()
        return start_token
        

    def __call__(self, key, value):
        batch_size = key.shape[0]
        decode_input = self.initDecodeInput(batch_size)
        
        mha_1_output = self.mha_1(decode_input, decode_input, decode_input)
        norm_mha_1_output = self.layer_norm_mha_1(mha_1_output + decode_input)
        mha_2_output = self.mha_1(mha_1_output, key, value)
        norm_mha_2_output = self.layer_norm_mah_2(mha_2_output + decode_input)

        ff_output = self.second_lt(self.first_lt(norm_mha_2_output))
        ff_norm = self.layer_norm_ff(ff_output + norm_mha_2_output)

        return ff_norm
           
class TransformersDecoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, input_dim, feed_forward_hidden_size):
        super(TransformersDecoder, self).__init__()
        self.decoder = DecoderLayer(num_heads, input_dim, feed_forward_hidden_size)
        self.linear = tf.keras.layers.Layer(input_dim)
    def __call__(self, key, value):
        decoder_output = self.decoder(key, value)
        return tf.squeeze(tf.nn.softmax(self.linear(decoder_output)), axis = 0)
#-----------------------------------------------------------------------------------------------

class Transformers(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, input_dim, feed_forward_hidden_size, training):
        '''
        Constructor for Transformers.

        Params:
            num_layers: number of encoder and decoder blocks
            num_heads: number of multi-attention heads. NOTE: It must divide input_dim
            input_dim: size of input
            feed_forward_hidden_size: Size of the feed forward network
            training: To indicate that we want dropout
        '''
        super(Transformers, self).__init__()
        self.encoder = TransformersEncoder(num_layers, num_heads, input_dim, feed_forward_hidden_size, training)
        self.decoder = pointer(input_dim, input_dim, layer_type='gru', dot_product_attention=True)
    def predict(self, input_set):
        output_encoder = self.encoder(input_set)
        return self.decoder(output_encoder)

class TPMP(tf.keras.layers.Layer):
    '''
    TPMP: TransformerParallelMultilayerPerceptron - using the transformer encoding step and the perceptron for the decoder
    '''
    def __init__(self, num_layers, num_heads, input_dim, feed_forward_hidden_size, training, perceptron_hidden_layer: list):
        '''
        Constructor for TPMP

        Params:
            num_layers: number of encoder and decoder blocks
            num_heads: number of multi-attention heads. NOTE: It must divide input_dim
            input_dim: size of input
            feed_forward_hidden_size: Size of the feed forward network
            training: To indicate that we want dropout
            perceptron_hidden_layer: dimension of the hidden layer for ParallelMultilayerPerceptron
        '''
        super(TPMP, self).__init__()
        self.encoder = TransformersEncoder(num_layers, num_heads, input_dim, feed_forward_hidden_size, training)
        self.decoder = ParallelMultilayerPerceptron(input_dim, perceptron_hidden_layer)
    def predict(self, input_set):
        '''
        Output probability over the input_set
        '''
        output_encoder = self.encoder(input_set)
        prob = self.decoder(output_encoder)
        return prob
        
#--------------------------------------------------------## End of Transformers


#--------------------------------------------------------## Implementation of Pointer Network
# TODO:
# - Check out probabilities

class pnetEncoder(tf.keras.layers.Layer):
    ''' 
    Encoder for pointer network. 
    '''

    def __init__(self, hidden_layer, layer_type = 'lstm'):
        '''
        Constructor for pointer network LSTM. 
        
        Args-
            hidden_layer - the output dimension of the output
        '''
        super(pnetEncoder, self).__init__()
        if layer_type == 'lstm':
            self.encode = tf.keras.layers.LSTM(hidden_layer, return_sequences=True, return_state=True)
        elif layer_type == 'gru':
            self.encode = tf.keras.layers.GRU(hidden_layer, return_sequences=True, return_state=True)
        else:
            raise Exception('LayerTypeError: Layer type entered is not supported')

        self.hidden_size = hidden_layer
    
    def __call__(self, input_seq):
        '''
        Run LSTM

        Args -
            input_seq - input set

        Return output sequences, final memory state, final cell state
        '''
        return self.encode(tf.cast(input_seq, dtype=tf.float32))

class pointer(tf.keras.layers.Layer):
    '''
    Pointer network attention mechanism. This will also handle the one decode step
    '''

    def __init__(self, input_dim, hidden_layer, layer_type = 'lstm', dot_product_attention = False):
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
        if layer_type == 'lstm':
            self.decoder_layer = tf.keras.layers.LSTM(hidden_layer, return_sequences=True)
        elif layer_type == 'gru':
            self.decoder_layer = tf.keras.layers.GRU(hidden_layer, return_sequences=True) 
        if not dot_product_attention:
            self.encoder_weight = tf.keras.layers.Dense(hidden_layer)
            self.decode_weight = tf.keras.layers.Dense(hidden_layer)
            self.v = tf.keras.layers.Dense(1)
            self.tanh = tf.keras.activations.tanh
        self.dot_prod_attention = dot_product_attention
        self.input_size = input_dim
        self.hidden_size = hidden_layer
    
    def initStartToken(self, batch_size):
        '''
        Initialize start token

        Params:
            batch_size: size of batch
        '''
        np.random.seed(42)
        start_token = tf.convert_to_tensor(np.random.random([batch_size,1,self.input_size]).astype(np.float32))
        np.random.seed()
        return start_token

    def __call__(self, encoder_output, initial_states = None):
        '''
        Calculate attention.
        
        Params:
            encoder_output: output of the encoder block
                size: (batch, seq_len, input_dim)
            initial_states: hidden and cell states from the encoder block
       '''
        batch_size = encoder_output.shape[0]
        start_token = self.initStartToken(batch_size)
        lstm_decoder_output = self.decoder_layer(start_token, initial_state=initial_states) #(batch, 1, input_dim)
        if self.dot_prod_attention:
            attention_scores = tf.squeeze(tf.linalg.matmul(lstm_decoder_output, encoder_output, transpose_b = True), axis = 1)
            return tf.nn.softmax(attention_scores)
        else:
            similarity_score = np.zeros([batch_size, encoder_output.shape[1]])
            lstm_decoder_projection = self.decode_weight(lstm_decoder_output)
            encoder_project = self.encoder_weight(encoder_output)
            for batch in range(batch_size):
                for index in range(similarity_score.shape[1]):
                    #e_i = tf.expand_dims(encoder_output[batch][index], axis = 0)
                    similarity_score[batch][index] = self.v(self.tanh(encoder_project[batch][index] + lstm_decoder_projection[batch]))[0][0] # change this
            return tf.nn.softmax(tf.convert_to_tensor(similarity_score))

class PointerNetwork(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_layer, input_layer = 'lstm', dot_prod_attention=False):
        '''
        Params:
            input_dim: dimension of input
            hidden_layer: dimension of output layer
            input_layer: lstm or gru
            dot_prod_attention: dot product attention or traditional pointer network attention
        '''
        super(PointerNetwork, self).__init__()
        self.encoder = pnetEncoder(hidden_layer, layer_type = input_layer)
        self.point = pointer(input_dim, hidden_layer, layer_type=input_layer, dot_product_attention=dot_prod_attention)
        self.hidden_size = hidden_layer
        self.layer = input_layer
        self.attention_type = dot_prod_attention

    def predict(self, input):
        '''
        '''
        if self.layer == 'lstm':
            seq_output, mem_state, carry_state = self.encoder(input)
            initial_states = [mem_state, carry_state]
        else:
            seq_output, mem_state = self.encoder(input)
            initial_states = mem_state
        prob_dist = self.point(seq_output, initial_states)

        return prob_dist

    def __call__(self, input):
        '''
        '''
        seq_output, mem_state, carry_state = self.encoder(input)
        initial_states = [mem_state, carry_state]
        prob_dist = self.point(seq_output, initial_states)

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
