#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import os

from environments.buchberger import BuchbergerEnv, LeadMonomialsWrapper
from environments.ideals import RandomBinomialIdealGenerator
from agents.pg import PGAgent
from agents.networks import ParallelMultilayerPerceptron, PairsLeft


def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    # environment parameters
    parser.add_argument('--n',
                        type=int,
                        default=3,
                        help='the number of variables')
    parser.add_argument('--d',
                        type=int,
                        default=5,
                        help='the maximal degree of monomials')
    parser.add_argument('--s',
                        type=int,
                        default=5,
                        help='the number of generators')
    parser.add_argument('--elimination',
                        choices=['gebauermoeller', 'lcm', 'none'],
                        default='gebauermoeller',
                        help='the elimination strategy')

    # agent parameters
    parser.add_argument('--k',
                        type=int,
                        default=2,
                        help='the number of monomials visible')
    parser.add_argument('--hidden_layers',
                        type=int, nargs='*',
                        default=[48, 48],
                        help='the hidden layers in the model')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='the learning rate')
    parser.add_argument('--gam',
                        type=float,
                        default=1.0,
                        help='the discount rate')
    parser.add_argument('--lam',
                        type=float,
                        default=1.0,
                        help='the generalized advantage parameter')
    
    # training parameters
    parser.add_argument('--episodes',
                        type=int,
                        default=100,
                        help='the number of episodes per epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='the number of epochs to train')
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
    ideal_gen = RandomBinomialIdealGenerator(args.n, args.d, args.s)
    env = LeadMonomialsWrapper(BuchbergerEnv(ideal_gen, elimination=args.elimination), k=args.k)
    return env


def make_agent(args):
    """Return the agent for this run."""
    network = ParallelMultilayerPerceptron(2 * args.n * args.k, args.hidden_layers)
    agent = PGAgent(network, policy_learning_rate=args.learning_rate,
                    value_network=PairsLeft(gam=args.gam),
                    gam=args.gam, lam=args.lam)
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    param_string = "".join([k + '=' + str(v) + ',' for k, v in vars(args).items()])
    logdir = os.path.join(args.logdir, param_string, time_string)
    return logdir


if __name__ == '__main__':
    args = make_parser().parse_args()
    env = make_env(args)
    agent = make_agent(args)
    logdir = make_logdir(args)
    agent.train(env, args.episodes, epochs=args.epochs, savefreq=args.save_freq, tensorboard_dir=logdir)
