import tensorflow as tf
import os
import datetime
import numpy as np
from deepgroebner.pg import PPOAgent, PGAgent
from deepgroebner.networks import ParallelMultilayerPerceptron, TransformerPMLP, Score, TransformerPMLP_Score_MHA
from deepgroebner.new_networks import TransformerPMLP_Score_Q, TransformerPMLP_DVal, TransformerPMLP_MHA_Q_Scorer
from deepgroebner.wrapped import CLeadMonomialsEnv
from deepgroebner.environments import VectorEnv, AlphabeticalEnv

env = AlphabeticalEnv()
model = TransformerPMLP(128, 128, softmax=True)
value_model = tf.keras.Sequential([tf.keras.layers.GRU(128), tf.keras.layers.Dense(1, activation='relu')])
agent = PPOAgent(model, value_network=value_model, pv_function=False)
agent.train(env, epochs = 500, episodes = 100, verbose=2, logdir = 'data/runs/shit1', parallel=False, max_episode_length=500)

