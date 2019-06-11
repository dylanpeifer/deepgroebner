# run.py
# Dylan Peifer
# 03 Jun 2019
"""Run a hyperparameter search on degree 9 binomials."""

from itertools import product
from os import mkdir

from environments.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import random_binomial_ideal
from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, PairsLeft

mkdir('data/paramsearch2')

EPISODES_PER_EPOCHS = [10, 100, 1000]
LEARNING_RATES = [0.001, 0.0005, 0.0001]
GAMMAS = [1.0, 0.95]
LAMBDAS = [1.0, 0.97]
HYPERPARAMETERS = product(EPISODES_PER_EPOCHS, LEARNING_RATES, GAMMAS, LAMBDAS)

for episodes_per_epoch, learning_rate, gam, lam in HYPERPARAMETERS:
    for i in range(3):
        
        env = BuchbergerEnv(lambda R: random_binomial_ideal(R, 9, 5, homogeneous=False), elimination='gebauermoeller')
        env = LeadMonomialsWrapper(env, k=2)
        network = ParallelMultilayerPerceptron(12, [48, 48])
        agent = PGAgent(network, policy_learning_rate=learning_rate,
                        value_network=PairsLeft(gam=gam),
                        gam=gam, lam=lam)
        
        savedir = ('data/paramsearch2/'
                   + str(i) + "-"
                   + str(episodes_per_epoch) + "-"
                   + str(learning_rate) + "-"
                   + str(gam) + "-"
                   + str(lam))
        mkdir(savedir)
        agent.train(env, episodes_per_epoch, epochs=100000//episodes_per_epoch,
                    savedir=savedir, savefreq=25000//episodes_per_epoch)
