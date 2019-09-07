# basic_run.py
# Dan H-L
# 7 September 2019
"""Basic entry point for training and testing the binomial model"""

from datetime import datetime
import os, errno, json

from environments.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import random_binomial_ideal
from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, PairsLeft

PARAMS = {
    'learning_rate' : 0.0005, # learning rate
    'episodes_per_epoch' : 100,
    'gam' : 0.95, # how much to trust value network
    'lam' : 1.0, # discount rate for future rewards
    }

LOG_DIR = 'data/test' # where to save results, will append time of run

# create networks and environment

f = lambda R: random_binomial_ideal(R, 2, 5, homogeneous=True)
env = BuchbergerEnv(f, elimination='none')
env = LeadMonomialsWrapper(env, k=1)
network = ParallelMultilayerPerceptron(6, [24])
agent = PGAgent(network, policy_learning_rate=PARAMS['learning_rate'],
                value_network=PairsLeft(gam=PARAMS['gam']),
                gam=PARAMS['gam'], lam=PARAMS['lam'])

# prepare to log results

now = datetime.now()
savedir = os.path.join(LOG_DIR,'run_' + now.strftime("%Y_%m_%d_%H_%M_%S"))
try:
    os.makedirs(savedir)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(directory_name):
        pass
    else:
        assert "Error making directory"
with open(os.path.join(savedir,'params.txt'), 'w') as outfile:
    json.dump(PARAMS, outfile)

# run training

agent.train(env, PARAMS['episodes_per_epoch'], epochs=100000//PARAMS['episodes_per_epoch'],
            savedir=savedir, savefreq=25000//PARAMS['episodes_per_epoch'])
