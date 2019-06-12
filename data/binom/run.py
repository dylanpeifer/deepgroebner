# run.py
# Dylan Peifer
# 10 Jun 2019
"""Replicate initial results on degree 2 binomials."""

from os import mkdir

from environments.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import random_binomial_ideal
from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, PairsLeft

mkdir('data/binom')

for i in range(5):

    f = lambda R: random_binomial_ideal(R, 2, 5, homogeneous=True)        
    env = BuchbergerEnv(f, elimination='none')
    env = LeadMonomialsWrapper(env, k=1)
    network = ParallelMultilayerPerceptron(6, [24])
    agent = PGAgent(network, policy_learning_rate=0.00001,
                    value_network=PairsLeft(gam=1.0),
                    gam=1.0, lam=1.0)

    savedir = 'data/binom/' + str(i)
    mkdir(savedir)
    agent.train(env, 1000, epochs=1000, savedir=savedir, savefreq=25)

