#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import os
import json

import gym

from environments.buchberger_cy import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import RandomBinomialIdealGenerator
from agents.ppo import PPOAgent
from agents.networks import MultilayerPerceptron, ParallelMultilayerPerceptron, PairsLeftBaseline

def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    # environment parameters
    parser.add_argument('--environment',
                        choices=['RandomBinomialIdeal', 'CartPole-v0', 'CartPole-v1', 'LunarLander-v2'],
                        default='RandomBinomialIdeal',
                        help='the training environment')
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
    parser.add_argument('--degree_distribution',
                        choices=['uniform', 'weighted', 'maximum'],
                        default='uniform',
                        help='the distribution of degrees')
    parser.add_argument('--elimination',
                        choices=['gebauermoeller', 'lcm', 'none'],
                        default='gebauermoeller',
                        help='the elimination strategy')
    parser.add_argument('--k',
                        type=int,
                        default=2,
                        help='the number of monomials visible')

    # agent parameters
    parser.add_argument('--hidden_layers',
                        type=int, nargs='*',
                        default=[48, 48],
                        help='the hidden layers in the models')
    parser.add_argument('--policy_lr',
                        type=float,
                        default=1e-4,
                        help='the policy model learning rate')
    parser.add_argument('--policy_updates',
                        type=int,
                        default=40,
                        help='policy model gradient updates per epoch')
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
                        help='run name, required')
    parser.add_argument('--episodes',
                        type=int,
                        default=100,
                        help='the number of episodes per epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='the number of epochs')
    parser.add_argument('--save_freq',
                        type=int,
                        default=25,
                        help='how often to save the weights')
    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='how much information to print')
    parser.add_argument('--logdir',
                        type=str,
                        default='data/runs',
                        help='the top level directory for training runs')

    return parser


def make_env(args):
    """Return the environment for this run."""
    if args.environment in ['CartPole-v0', 'CartPole-v1', 'LunarLander-v2']:
        env = gym.make(args.environment)
    else:
        ideal_gen = RandomBinomialIdealGenerator(args.variables, args.degree, args.generators)
        env = BuchbergerEnv(ideal_gen, elimination=args.elimination)
        env = LeadMonomialsWrapper(env, k=args.k)
    return env


def make_agent(args):
    """Return the agent for this run."""
    dims = {'CartPole-v0': (4, 2),
            'CartPole-v1': (4, 2),
            'LunarLander-v2': (8, 4),
            'RandomBinomialIdeal': (2 * args.variables * args.k, 1)}[args.environment]
    if args.environment == 'RandomBinomialIdeal':
        policy_network = ParallelMultilayerPerceptron(dims[0], args.hidden_layers)
        value_network = PairsLeftBaseline(gam=args.gam)
        action_dim_fn = lambda s: s[0]
    else:
        policy_network = MultilayerPerceptron(dims[0], args.hidden_layers, dims[1])
        value_network = None
        action_dim_fn = lambda s: dims[1]
    agent = PPOAgent(policy_network, policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                     value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                     gam=args.gam, lam=args.lam, eps=args.eps, action_dim_fn=action_dim_fn)
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    param_string = "".join([k + '=' + str(v) + ',' for k, v in vars(args).items()])
    logdir = os.path.join(args.logdir, args.name + '_' + time_string + '_' + str(abs(hash(param_string))))
    os.makedirs(logdir)
    return logdir


def save_args(logdir,args):
    with open(os.path.join(logdir,'args.txt'), 'w') as outfile:
        json.dump(vars(args), outfile)


if __name__ == '__main__':
    args = make_parser().parse_args()
    env = make_env(args)
    agent = make_agent(args)
    logdir = make_logdir(args)
    save_args(logdir,args)
    print(logdir)
    agent.train(env, episodes=args.episodes, epochs=args.epochs,
                save_freq=args.save_freq, logdir=logdir, verbose=args.verbose)
