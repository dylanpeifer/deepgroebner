#!/usr/bin/env python
"""Entry point for all evaluation runs."""

import argparse
import datetime
import numpy as np
import os
import shutil
import json

import gym

import csv
from deepgroebner.pg import TrajectoryBuffer

from deepgroebner.buchberger import LeadMonomialsEnv
from deepgroebner.pg import PGAgent
from deepgroebner.networks import MultilayerPerceptron, ParallelMultilayerPerceptron, \
                                    AttentionPMLP, TransformerPMLP, TransformerPMLP_Score_MHA
from deepgroebner.new_networks import TransformerPMLP_Score_Q
from deepgroebner.wrapped import CLeadMonomialsEnv


def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Evaluate a model",
                                     fromfile_prefix_chars='@')

    env = parser.add_argument_group('environment', 'environment type')
    env.add_argument('--environment',
                     choices=['RandomBinomialIdeal', 'RandomIdeal',
                              'CartPole-v0', 'CartPole-v1', 'LunarLander-v2'],
                     default='RandomBinomialIdeal',
                     help='evaluation environment')
    env.add_argument('--env_seed',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=None,
                     help='seed for the environment')

    ideal = parser.add_argument_group('ideals', 'ideal distribution and environment options')
    ideal.add_argument('--distribution',
                       type=str,
                       default='3-20-10-weighted',
                       help='random ideal distribution')
    ideal.add_argument('--elimination',
                       choices=['gebauermoeller', 'lcm', 'none'],
                       default='gebauermoeller',
                       help='pair elimination strategy')
    ideal.add_argument('--rewards',
                       choices=['additions', 'reductions'],
                       default='additions',
                       help='reward given for each step')
    ideal.add_argument('--k',
                       type=int,
                       default=2,
                       help='number of lead monomials visible')
    ideal.add_argument('--use_cython',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to use the Cython environment')

    policy = parser.add_argument_group('policy model')
    policy.add_argument('--policy_model',
                        choices=['mlp', 'pmlp', 'apmlp', 'tpmlp', 'tpmlp_q_scorer', 'tpmlp_MHA_scorer'],
                        default='pmlp',
                        help='policy network type')
    policy.add_argument('--policy_kwargs',
                        type=json.loads,
                        default={"hidden_layers": [128]},
                        help='arguments to policy model constructor, passed through json.loads')
    policy.add_argument('--policy_weights',
                        type=str,
                        default="",
                        help='filename for initial policy weights')

    run = parser.add_argument_group('running')
    run.add_argument('--episodes',
                     type=int,
                     default=10000,
                     help='number of episodes per epoch')
    run.add_argument('--max_episode_length',
                     type=lambda x: int(x) if x.lower() != 'none' else None,
                     default=500,
                     help='max number of interactions per episode')
    run.add_argument('--use_gpu',
                     type=lambda x: str(x).lower() == 'true',
                     default=True,
                     help='whether to use a GPU if available')

    save = parser.add_argument_group('saving')
    save.add_argument('--name',
                       type=str,
                       default='run',
                       help='name of evaluation run')
    save.add_argument('--datetag',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to append current time to run name')
    save.add_argument('--logdir',
                       type=str,
                       default='data/eval',
                       help='base directory for evaluation runs')
    return parser


def make_env(args):
    """Return the evaluation environment for this run."""
    if args.environment in ['CartPole-v0', 'CartPole-v1', 'LunarLander-v2']:
        env = gym.make(args.environment)
    elif args.use_cython:
        env = CLeadMonomialsEnv(args.distribution, elimination=args.elimination, rewards=args.rewards, k=args.k)
    else:
        env = LeadMonomialsEnv(args.distribution, elimination=args.elimination, rewards=args.rewards, k=args.k)
    env.seed(args.env_seed)
    return env


def make_policy_network(args):
    """Return the policy network for this run."""
    kwargs = args.policy_kwargs
    if args.environment == 'LunarLander-v2':
        assert "output_dim" not in kwargs
        policy_network = MultilayerPerceptron(4, **kwargs)
        batch = np.zeros((1, 8), dtype=np.float32)
    elif args.environment in ['CartPole-v0', 'CartPole-v1']:
        assert "output_dim" not in kwargs
        policy_network = MultilayerPerceptron(2, **kwargs)
        batch = np.zeros((1, 4), dtype=np.float32)
    else:
        if args.policy_model == 'pmlp':
            policy_network = ParallelMultilayerPerceptron(**args.policy_kwargs)
        elif args.policy_model == 'apmlp':
            policy_network = AttentionPMLP(**args.policy_kwargs)
        elif args.policy_model == 'tpmlp':
            policy_network = TransformerPMLP(**args.policy_kwargs)
        elif args.policy_model == 'tpmlp_q_scorer':
            policy_network = TransformerPMLP_Score_Q(**args.policy_kwargs)
        else:
            policy_network = TransformerPMLP_Score_MHA(**args.policy_kwargs)
        batch = np.zeros((1, 10, 2 * args.k * int(args.distribution.split('-')[0])), dtype=np.int32)
    policy_network(batch)  # build network
    if args.policy_weights != "":
        policy_network.load_weights(args.policy_weights)
    return policy_network


def make_agent(args):
    """Return the agent for this run."""
    policy_network = make_policy_network(args)
    agent = PGAgent(policy_network=policy_network)
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    run_name = args.name
    if args.datetag:
        time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = time_string + '_' + run_name
    logdir = os.path.join(args.logdir, run_name)
    os.makedirs(logdir)
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write('--' + arg + '\n')
            if isinstance(value, dict):
                f.write(json.dumps(value) + "\n")
            else:
                f.write(str(value) + '\n')
    return logdir

def eval_analysis(dataset):
    returns = 0
    std_returns = 0
    total = len(dataset)
    one_std_dataset = []
    two_std_dataset = []
        
    for datum in dataset:
        returns += datum[1]
    mean_returns = returns/total

    for datum in dataset:
        std_return += (datum[1]-mean_returns)**2
    std_returns /= total

    for datum in dataset:
        if datum[1] >= mean_returns + 2*std_returns:
            two_std_dataset.append(datum)
        elif datum[1] >= mean_returns + std_returns:
            one_std_dataset.append(datum)
    return one_std_dataset, two_std_dataset

def save(logdir, name, dataset):
    np.savez(os.path.join(logdir, name), allow_pickle = True, dtype = object, *dataset)

if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    env = make_env(args)
    agent = make_agent(args)
    logdir = make_logdir(args)
    print("Saving run in", logdir)
    shutil.copy(args.policy_weights, os.path.join(logdir, "policy.h5"))
    dataset = []

    
    buffer = TrajectoryBuffer(gam = .99, lam = .97)
    
    with open(os.path.join(logdir, "results.csv"), 'w') as f:
        f.write("Return,Length\n")
        
    with open(os.path.join(logdir, "predicted_vs_value.csv"), "w") as f:
        f.write("Predicted,Discounted Value,Episode\n")

    for i in range(args.episodes):
        buffer.clear()
        reward, length, _, _, value_pairs, ideal = agent.run_episode_v2(env, max_episode_length=args.max_episode_length, buffer = buffer, num_episode=i)
        if i < 5000:
            with open(os.path.join(logdir, "predicted_vs_value.csv"), "a") as f:
                for pair in value_pairs:
                    predicted_val = float(pair[0])
                    f.write(f"{float(predicted_val)},{pair[1]},{pair[2]}\n")
        with open(os.path.join(logdir, "results.csv"), "a") as f:
            f.write(f"{reward},{length}\n")
        dataset.append((ideal, reward, length))
        if i in [2500, 5000, 7500, args.episodes - 1]:
            save(logdir, "fractional_dataset"+str(i)+".npz", dataset)
