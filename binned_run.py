# binned_run.py
# Dylan Peifer
# 16 Oct 2019
"""Entry point for training and testing the binomial model on pre-binned data."""

from datetime import datetime
import os, errno, json
import tensorflow as tf
import numpy as np
import sympy as sp

from environments.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import FromDirectoryIdealGenerator, RandomBinomialIdealGenerator
from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, PairsLeft

PARAMS = {
    'learning_rate' : 0.00001,
    'episodes_per_epoch' : 100,
    'gam' : 1.0, # discount rate for future rewards
    'lam' : 1.0, # how much to trust value network
    'k' : 2, # number of lead monomials to expose to the agent
    'elimination' : 'gebauermoeller', # options: 'none', 'lcm', 'gebauermoeller'
    'homogeneous' : False,
    'gen_degree' : 10,
    'gen_number' : 5,
    'hidden_layers' : [128, 128],
    'num_variables' : 5
    }

LOG_DIR = 'data/test' # where to save results, will append time of run

# create networks and environment
R = sp.ring('a,b,c,d,e', sp.FF(32003), 'grevlex')[0]
train_ideal_gen = FromDirectoryIdealGenerator('data/bins-5-10-5', R)
test_ideal_gen = RandomBinomialIdealGenerator(PARAMS['num_variables'],
                                              PARAMS['gen_degree'],
                                              PARAMS['gen_number'],
                                              homogeneous=PARAMS['homogeneous'])
train_env = LeadMonomialsWrapper(BuchbergerEnv(train_ideal_gen,
                                               elimination=PARAMS['elimination'],
                                               sort_reducers=True),
                                 k=PARAMS['k'])
test_env = LeadMonomialsWrapper(BuchbergerEnv(test_ideal_gen,
                                              elimination=PARAMS['elimination'],
                                              sort_reducers=True),
                                k=PARAMS['k'])
network = ParallelMultilayerPerceptron(2*PARAMS['num_variables']*PARAMS['k'],
                                       PARAMS['hidden_layers'])
agent = PGAgent(network,
                policy_learning_rate=PARAMS['learning_rate'],
                value_network=PairsLeft(gam=PARAMS['gam']),
                gam=PARAMS['gam'],
                lam=PARAMS['lam'])

# prepare to log results
now = datetime.now()
run_name = 'run_' + now.strftime("%Y_%m_%d_%H_%M_%S")
savedir = os.path.join(LOG_DIR, run_name)
try:
    os.makedirs(savedir)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(savedir):
        pass
    else:
        assert "Error making directory"
with open(os.path.join(savedir, 'params.txt'), 'w') as outfile:
    json.dump(PARAMS, outfile)

# run training
print('Beginning training for ' + run_name)

tb_writer = tf.compat.v1.summary.FileWriter(savedir)

for i in range(100000):

    avg_reward = np.mean(agent.test(test_env, episodes=100, max_episode_length=1000))
    summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='rewards',
                                          simple_value=avg_reward)])
    tb_writer.add_summary(summary, i)
    tb_writer.flush()    
    agent.train(train_env,
                PARAMS['episodes_per_epoch'],
                epochs=1,
                max_episode_length=1000,
                savedir=savedir,
                savefreq=100)
    train_ideal_gen.update()
