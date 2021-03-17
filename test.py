import tensorflow as tf
import os
import datetime
import numpy as np
import csv
from deepgroebner.pg import PPOAgent, PGAgent, discount_rewards
from deepgroebner.networks import ParallelMultilayerPerceptron, TransformerPMLP, Score, TransformerPMLP_Score_MHA
from deepgroebner.new_networks import TransformerPMLP_Score_Q, TransformerPMLP_DVal, TransformerPMLP_MHA_Q_Scorer
from deepgroebner.wrapped import CLeadMonomialsEnv
from deepgroebner.environments import VectorEnv, AlphabeticalEnv
from deepgroebner.transformer_value_network import TransformerValueModel

env = AlphabeticalEnv()
model = ParallelMultilayerPerceptron([128])
value_model = TransformerValueModel([128, 64, 32], 128, softmax = False)
batch = np.zeros((1, 10, 1000), dtype=np.int32)
model(batch)
value_model(batch)

model.load_weights('/Users/christianvarner/Research/deepgroebner/data/runs/pmlp_sig_tran/policy-500.h5')
value_model.load_weights('/Users/christianvarner/Research/deepgroebner/data/runs/pmlp_sig_tran/value-500.h5')

agent = PPOAgent(model, policy_updates=10, value_network=value_model, pv_function=False)
agent.train(env, epochs = 500, episodes = 100, verbose=2, logdir = 'data/runs/pmlp_sig_val_function', parallel=False, max_episode_length=500)

def run_episodes(env, agent, value_model, num_episodes = 10000, max_cut_off = 500):
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
        action = agent(state)
        predicted_values.append(value_model(state))
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