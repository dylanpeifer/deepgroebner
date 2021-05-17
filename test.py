import tensorflow as tf
import os
import datetime
import numpy as np
import csv
from deepgroebner.pg import PPOAgent, PGAgent, discount_rewards
from deepgroebner.networks import ParallelMultilayerPerceptron
from deepgroebner.dual_networks import DualTransformerPMLP
from deepgroebner.wrapped import CLeadMonomialsEnv
from deepgroebner.environments import VectorEnv, AlphabeticalEnv
from deepgroebner.transformer_value_network import TransformerValueModel


def run_episodes(env, agent, value_model, num_episodes = 10, max_cut_off = 500):
    p_values = []
    a_values = []
    for _ in range(num_episodes):
        predicted, actual = run_episode(env, agent, value_model, max_cut_off)
        p_values.append(predicted)
        a_values.append(actual)
    return p_values, a_values

def run_episode(env, agent, value_model, max_cut_off = 500):
    state = env.reset()
    step = 0
    done = False
    predicted_values = []
    rewards = []
    while not done and step < max_cut_off:
        if state.dtype == np.float64:
                state = state.astype(np.float32)

        logpi = agent(state[tf.newaxis])
        action = tf.random.categorical(logpi, 1)[0, 0]

        predicted_values.append(float(value_model(state[tf.newaxis])[0][0]))
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
        step += 1

    return predicted_values, discount_rewards(rewards, gam = 0.99)

def get_model(p_agent, v_agent, policy_path, value_path):
    batch = np.zeros((1, 10, 1000), dtype=np.int32)

    #build model
    p_agent(batch)
    v_agent(batch)

    #load weights
    p_agent.load_weights(policy_path)
    v_agent.load_weights(value_path)

    return p_agent, v_agent

def save_data(predict, actual, filename):
    with open(filename, "w+") as f:
        csvWriter = csv.writer(f)
        for trajectory in list(zip(predict, actual)):
            csvWriter.writerow(trajectory[0]) # predicted row
            csvWriter.writerow(trajectory[1]) # actual row

# env = AlphabeticalEnv()
# policy_path = '/Users/christianvarner/Research/deepgroebner/data/runs/pmlp_sig_val_function/policy-500.h5'
# value_path = '/Users/christianvarner/Research/deepgroebner/data/runs/pmlp_sig_val_function/value-500.h5'

# model = ParallelMultilayerPerceptron([128])
# value_model = TransformerValueModel([128, 64, 32], 128, softmax = False)
# batch = np.zeros((1, 10, 1000), dtype=np.int32)
# model(batch)
# value_model(batch)

# model.load_weights(policy_path)
# value_model.load_weights(value_path)

# predicted, actual = run_episodes(env, model, value_model)
# save_data(predicted, actual, 'value_model_data.csv')

# exit()

env = AlphabeticalEnv()
model = DualTransformerPMLP(128, 128, num_layers=4)
value_model = TransformerValueModel([],128,False)

# batch = np.zeros((1, 10, 1000), dtype=np.int32)
# model(batch)
# value_model(batch)

# policy_path = '/Users/christianvarner/Research/deepgroebner/data/runs/pmlp_sig_val_function_v4/policy-500.h5'
# value_path = '/Users/christianvarner/Research/deepgroebner/data/runs/pmlp_sig_val_function_v4/value-500.h5'

# model.load_weights(policy_path)
# value_model.load_weights(value_path)

agent = PPOAgent(model, value_network=value_model, policy_updates=10, pv_function=True)
#logdir = 'data/runs/pmlp_sig_val_function_v4.5'
agent.train(env, epochs = 500, episodes = 10, verbose=2, save_freq=25, parallel=False, max_episode_length=20)