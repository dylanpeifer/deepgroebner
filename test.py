import numpy as np
import sympy as sp
import tensorflow as tf
import csv
import time

from deepgroebner.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from deepgroebner.ideals import RandomBinomialIdealGenerator
from deepgroebner.networks import MultilayerPerceptron, PointerNetwork, pnetEncoder, Transformers, TPMP, ProcessBlock
import deepgroebner.networks as n
from deepgroebner.pg import PPOAgent

from pyinstrument import Profiler

def testing_pointer_network():
    # Ideal generator
    ideal_gen = RandomBinomialIdealGenerator(3, 20, 10, degrees='weighted') 

    # wrap it into env
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen), k=2)
    #network = ParallelMultilayerPerceptron(12, [128])
    #network = Transformers(1, 3, 12, 128, training=False)
    network = PointerNetwork(12, 128, input_layer='gru', dot_prod_attention=True)
    agent = PPOAgent(network)

    state = env.reset()
    # print('Probability Distribution')
    # print('---------------------------')
    # for_prob = tf.expand_dims(state, axis=0)
    # prob_dist = network.predict(for_prob)
    # print(prob_dist)
    # print('---------------------------')

    # start = time.time()
    # agent.run_episodes(env, episodes=5, parallel=False)
    # end = time.time()
    # print(end-start)]
    
    action = agent.act(state)

    #stuff = agent.act(state)

    print('My action is')
    print('---------------------------')
    print(action)
    #print('Time')
    # print(end-start)
    # print('---------------------------')

def time_test_episode(network, env, seed):

    agent = PPOAgent(network)
    start = time.time()
    run_episode(agent, env, seed)
    end = time.time()

    return end-start

def time_test_episodes(network, env, episodes):

    agent = PPOAgent(network)
    start = time.time()
    agent.run_episodes(env, episodes=episodes, parallel=False)
    end = time.time()

    return end-start

def save_time(data):
    with open('run_time_data.csv', 'w+') as rt:
        csvWriter = csv.writer(rt)
        for key in data.keys():
            row = [key[0], key[1]]
            for datum in data[key]:
                row.append(datum)
            csvWriter.writerow(row)

def run_mult_test():

    hidden_dim = [16, 32, 64, 128]
    network = [ParallelMultilayerPerceptron, PointerNetwork, PointerNetwork, Transformers]
    ideal_gen = RandomBinomialIdealGenerator(3, 20, 10, degrees='weighted') 
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen), k=2)

    times = {}
    
    for hd in hidden_dim:
        for i in range(25):
            seed = i
            for index,nn in enumerate(network):
                if index == 0:
                    time_trail = time_test_episode(nn(12, [hd]), env, seed)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
                elif index == 1:
                    time_trail = time_test_episode(nn(12, hd), env, seed)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
                elif index == 2:
                    time_trail = time_test_episode(nn(12, hd, input_layer = 'gru'), env, seed)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
                else:
                    time_trail = time_test_episode(nn(1, 3, 12, hd, training = False), env, seed)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
    save_time(times)

def time_test_agent():
    episodes = [25, 50, 75, 100]
    network = [ParallelMultilayerPerceptron, PointerNetwork, PointerNetwork, Transformers]
    ideal_gen = RandomBinomialIdealGenerator(3, 20, 10, degrees='weighted') 
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen), k=2)

    times = {}
    
    for hd in episodes:
        for i in range(1):
            for index,nn in enumerate(network):
                if index == 0:
                    time_trail = time_test_episodes(nn(12, [128]), env, hd)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
                elif index == 1:
                    time_trail = time_test_episodes(nn(12, 128), env, hd)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
                elif index == 2:
                    time_trail = time_test_episodes(nn(12, 128, input_layer = 'gru'), env, hd)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
                else:
                    time_trail = time_test_episodes(nn(1, 3, 12, 128, training = False), env, hd)
                    if (index, hd) in times.keys():
                        times[(index, hd)].append(time_trail)
                    else:
                        times[(index,hd)] = [time_trail]
    save_time(times)

def run_episode(agent, env, seed):
    states = []
    np.random.seed(seed)
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state, greedy = True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        states.append(state)
        state = next_state
    return total_reward, states

def test_multi_headed():
    # Ideal generator
    ideal_gen = RandomBinomialIdealGenerator(3, 20, 10, degrees='weighted') 

    # wrap it into env
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen), k=2)
    state = env.reset()
    state = tf.expand_dims(state, axis = 0)

    print('Before network: ')
    print(state.shape)

    network = n.TransformersEncoder(6, 2, 12, 128)
    result = network(state)

    print('After network: ')
    print(result.shape) # num_examples, 12

# Tests for processing_block
def test_process_block():

    #Environment Initializations
    ideal_gen = RandomBinomialIdealGenerator(3, 20, 10, degrees='weighted') 
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen), k=2)
    state = env.reset()
    state_shuffled = tf.random.shuffle(state) # shuffle state

    processBlock = ProcessBlock(128, 4) # Processing block

    state = tf.expand_dims(state, axis = 0)
    state_shuffled = tf.expand_dims(state_shuffled, axis = 0)
    blend_og = processBlock(state)
    blend_shuffled = processBlock(state_shuffled)

    #print(blend_og)


    
def main():
    #---------------------------------------------------------------------
    ideal_gen = RandomBinomialIdealGenerator(3, 20, 10, degrees='weighted') 
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen), k=2)

    #test_process_block()

    network = PointerNetwork(12, 128, input_layer='gru', dot_prod_attention=True)
    #network = ParallelMultilayerPerceptron(12, [128])
    #network = Transformers(1, 4, 20, 128, training=False)
    #print(network.non_trainable_variables)
    #network = TPMP(1, 3, 12, 128, False, [128])
    agent = PPOAgent(network)
    #agent.run_episodes(env, episodes=1, greedy=True, parallel = False)
    agent.train(env, episodes = 100, verbose = 1, parallel=True)
    #------------------------------------------------------------------

if __name__ == '__main__':
    main()