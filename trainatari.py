import gym
import numpy as np

from agents.dqn import DQNAgent
from agents.networks import AtariNetSmall
from environments.atari import AtariEnv


env = AtariEnv('BreakoutDeterministic-v4')
network = AtariNetSmall((105, 80, 4), 4)
agent = DQNAgent(network,
                 memory_capacity=1000000,
                 epsilon_min=0.1, decay_rate=0.0000009, decay_mode='linear',
                 start_steps=50000, replay_freq=4, target_update_freq=40000,
                 double=True)

while agent.steps < 50000000:
    rewards = agent.train(env, 10000, verbose=1)
    print(np.mean(rewards))
