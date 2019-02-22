import gym
import numpy as np

from agents.dqn import DQNAgent
from agents.networks import AtariNetSmall
from environments.atari import AtariEnv


env = AtariEnv('BreakoutDeterministic-v4')
test_env = AtariEnv('BreakoutDeterministic-v4', reset_on_death=False)
network = AtariNetSmall((105, 80, 4), 4)
agent = DQNAgent(network,
                 memory_capacity=1000000,
                 epsilon_min=0.1, decay_rate=0.0000009, decay_mode='linear',
                 start_steps=50000, replay_freq=4, target_update_freq=40000,
                 double=True)
agent.train(env, 1000000, epochs=25, verbose=1, test_env=test_env, test_episodes=100, test_epsilon=0.05, savefile='breakout.h5')
