import numpy as np
import sympy as sp

from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, ValueRNN
from environments.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import random_binomial_ideal

R = sp.ring('x,y,z', sp.FF(32003), 'grevlex')[0]
f = lambda R: random_binomial_ideal(R, 7, 5)
env = LeadMonomialsWrapper(BuchbergerEnv(f, ring=R, elimination='gebauermoeller'), k=2)
policy = ParallelMultilayerPerceptron(12, [48])
value = ValueRNN(12, 32)
agent = PGAgent(policy, policy_learning_rate=0.0001, gam=1.0, lam=1.0)

r = agent.train(env, 1000, epochs=3, verbose=1)

print(np.mean(r))
