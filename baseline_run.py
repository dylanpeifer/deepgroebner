# agentbaseline_run.py
# Dylan Peifer
# 13 Nov 2019
"""Entry point for training with actual agent baseline."""

from datetime import datetime
import os, errno, json

from environments.buchberger import BuchbergerEnv, BuchbergerAgent, LeadMonomialsWrapper
from environments.ideals import RandomBinomialIdealGenerator
from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, AgentBaseline

PARAMS = {
    'learning_rate' : 0.0001,
    'episodes_per_epoch' : 100,
    'gam' : 1.0, # discount rate for future rewards
    'lam' : 1.0, # how much to trust value network
    'k' : 2, # number of lead monomials to expose to the agent
    'elimination' : 'gebauermoeller', # options: 'none', 'lcm', 'gebauermoeller'
    'homogeneous' : False,
    'gen_degree' : 10,
    'gen_number' : 5,
    'hidden_layers' : [64, 64],
    'num_variables' : 5
    }

LOG_DIR = 'data/test' # where to save results, will append time of run

# create networks and environment
ideal_gen = RandomBinomialIdealGenerator(PARAMS['num_variables'],
                                         PARAMS['gen_degree'],
                                         PARAMS['gen_number'],
                                         homogeneous=PARAMS['homogeneous'])
env = BuchbergerEnv(ideal_gen, elimination=PARAMS['elimination'])
env = LeadMonomialsWrapper(env, k=PARAMS['k'])
network = ParallelMultilayerPerceptron(2*PARAMS['num_variables']*PARAMS['k'], PARAMS['hidden_layers'])
agent = PGAgent(network, policy_learning_rate=PARAMS['learning_rate'],
                value_network=AgentBaseline(BuchbergerAgent(selection='degree'),
                                            gam=PARAMS['gam']),
                gam=PARAMS['gam'], lam=PARAMS['lam'])

# prepare to log results
now = datetime.now()
run_name = 'run_' + now.strftime("%Y_%m_%d_%H_%M_%S")
savedir = os.path.join(LOG_DIR,run_name)
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
print('Beginning training for ' + run_name)
agent.train(env, PARAMS['episodes_per_epoch'], epochs=1000,
            savedir=savedir, savefreq=25, tensorboard_dir=savedir,
            max_episode_length=1000)
