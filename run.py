#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import os
import json

import gym

from deepgroebner.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from deepgroebner.ideals import RandomBinomialIdealGenerator
from deepgroebner.pg import PGAgent, PPOAgent
from deepgroebner.networks import MultilayerPerceptron, ParallelMultilayerPerceptron, PairsLeftBaseline

def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    # environment type
    parser.add_argument('--environment',
                        choices=['RandomBinomialIdeal', 'CartPole-v0', 'CartPole-v1', 'LunarLander-v2'],
                        default='RandomBinomialIdeal',
                        help='the training environment')
    
    # RandomBinomialIdeal parameters
    parser.add_argument('--variables',
                        type=int,
                        default=3,
                        help='the number of variables')
    parser.add_argument('--degree',
                        type=int,
                        default=5,
                        help='the maximal degree of monomials')
    parser.add_argument('--generators',
                        type=int,
                        default=5,
                        help='the number of generators')
    parser.add_argument('--constants',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether the generators can have constants')
    parser.add_argument('--degree_distribution',
                        choices=['uniform', 'weighted', 'maximum'],
                        default='uniform',
                        help='the probability distribution on degrees')
    parser.add_argument('--homogeneous',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether the ideals are homogeneous')
    parser.add_argument('--pure',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether the ideals are pure')
    parser.add_argument('--elimination',
                        choices=['gebauermoeller', 'lcm', 'none'],
                        default='gebauermoeller',
                        help='the elimination strategy')
    parser.add_argument('--rewards',
                        choices=['additions', 'reductions'],
                        default='additions',
                        help='the reward given for each step')
    parser.add_argument('--k',
                        type=int,
                        default=2,
                        help='the number of lead monomials visible')

    # agent parameters
    parser.add_argument('--algorithm',
                        choices=['ppo', 'pg'],
                        default='ppo',
                        help='the training algorithm')
    parser.add_argument('--policy_hl',
                        type=int, nargs='*',
                        default=[128],
                        help='the hidden layers in the policy model')
    parser.add_argument('--policy_lr',
                        type=float,
                        default=1e-4,
                        help='the policy model learning rate')
    parser.add_argument('--policy_updates',
                        type=int,
                        default=40,
                        help='policy model gradient updates per epoch')
    parser.add_argument('--policy_kld_limit',
                        type=float,
                        default=0.01,
                        help='the KL divergence limit')
    parser.add_argument('--value_model',
                        choices=['none', 'mlp', 'pairsleft'],
                        default='none',
                        help='the value network type')
    parser.add_argument('--value_hl',
                        type=int, nargs='*',
                        default=[128],
                        help='the hidden layers in the policy model')
    parser.add_argument('--value_lr',
                        type=float,
                        default=1e-4,
                        help='the value model learning rate')
    parser.add_argument('--value_updates',
                        type=int,
                        default=40,
                        help='value model gradient updates per epoch')
    parser.add_argument('--gam',
                        type=float,
                        default=1.0,
                        help='the discount rate')
    parser.add_argument('--lam',
                        type=float,
                        default=1.0,
                        help='the generalized advantage parameter')
    parser.add_argument('--eps',
                        type=float,
                        default=0.1,
                        help='the clip ratio for PPO')

    # training parameters
    parser.add_argument('--name',
                        type=str,
                        default='run',
                        help='name of training run')
    parser.add_argument('--datetag',
                        type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='whether to append current time to run name')
    parser.add_argument('--episodes',
                        type=int,
                        default=100,
                        help='the number of episodes per epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        help='the number of epochs')
    parser.add_argument('--max_episode_length',
                        type=lambda x: int(x) if x.lower() != 'none' else None,
                        default=None,
                        help='the max number of interactions per episode')
    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='how much information to print')
    parser.add_argument('--save_freq',
                        type=int,
                        default=25,
                        help='how often to save the weights')
    parser.add_argument('--logdir',
                        type=str,
                        default='data/runs',
                        help='the base directory for training runs')
    parser.add_argument('--binned',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether to train on binned ideals')
    parser.add_argument('--stacked',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether to use stacking')
    parser.add_argument('--stack_size',
                        type=int,
                        default=-1,
                        help='the stack batch size for stacking')
    parser.add_argument('--pad',
                        type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='whether to pad stacks')

    return parser


def make_env(args):
    """Return the environment for this run."""
    if args.environment in ['CartPole-v0', 'CartPole-v1', 'LunarLander-v2']:
        env = gym.make(args.environment)
    else:
        ideal_gen = RandomBinomialIdealGenerator(args.variables, args.degree, args.generators,
                                                 constants=args.constants, degrees=args.degree_distribution,
                                                 homogeneous=args.homogeneous, pure=args.pure)
        env = BuchbergerEnv(ideal_gen, elimination=args.elimination, rewards=args.rewards)
        env = LeadMonomialsWrapper(env, k=args.k)
    return env


def make_agent(args):
    """Return the agent for this run."""
    dims = {'CartPole-v0': (4, 2),
            'CartPole-v1': (4, 2),
            'LunarLander-v2': (8, 4),
            'RandomBinomialIdeal': (2 * args.variables * args.k, 1)}[args.environment]
    if args.environment == 'RandomBinomialIdeal':
        policy_network = ParallelMultilayerPerceptron(dims[0], args.policy_hl)
        value_network = PairsLeftBaseline(gam=args.gam)
        action_dim_fn = lambda s: s[0]
    else:
        policy_network = MultilayerPerceptron(dims[0], args.policy_hl, dims[1])
        value_network = MultilayerPerceptron(dims[0], args.value_hl, 1, final_activation='linear')
        action_dim_fn = lambda s: dims[1]
    if args.algorithm == 'pg':
        agent = PGAgent(policy_network=policy_network, policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                        value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                        gam=args.gam, lam=args.lam, action_dim_fn=action_dim_fn)
    else:
        agent = PPOAgent(policy_network=policy_network, policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                         value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                         gam=args.gam, lam=args.lam, eps=args.eps, action_dim_fn=action_dim_fn)
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.datetag:
        run_name = args.name + "_" + time_string
    else:
        run_name = args.name
    logdir = os.path.join(args.logdir, run_name)
    os.makedirs(logdir)
    return logdir


def save_args(logdir, args):
    """Save args as a text file in logdir."""
    with open(os.path.join(logdir,'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write('--' + arg +'\n')
            if isinstance(value, list):
                f.write("\n".join(str(v) for v in value) + "\n")
            else:
                f.write(str(value) + '\n')


if __name__ == '__main__':
    args = make_parser().parse_args()
    env = make_env(args)
    agent = make_agent(args)
    logdir = make_logdir(args)
    save_args(logdir, args)
    print(logdir)
    agent.train(env, episodes=args.episodes, epochs=args.epochs,
                stacked=args.stacked, stack_size=args.stack_size, pad=args.pad,
                save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                max_episode_length=args.max_episode_length)
