#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import os
import json

import gym
import sympy as sp

from deepgroebner.buchberger import BuchbergerEnv, LeadMonomialsWrapper, BuchbergerAgent
from deepgroebner.ideals import RandomBinomialIdealGenerator, FromDirectoryIdealGenerator, RandomIdealGenerator
from deepgroebner.pg import PGAgent, PPOAgent
from deepgroebner.networks import MultilayerPerceptron, ParallelMultilayerPerceptron, TransformerPMLP, PairsLeftBaseline, AgentBaseline

def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    # environment type
    parser.add_argument('--environment',
                        choices=['RandomBinomialIdeal', 'MixedRandomBinomialIdeal', 'RandomPolynomialIdeal',
                                 'CartPole-v0', 'CartPole-v1', 'LunarLander-v2'],
                        default='RandomBinomialIdeal',
                        help='the training environment')
    
    # RandomBinomialIdeal parameters
    parser.add_argument('--variables',
                        type=int,
                        default=3,
                        help='the number of variables')
    parser.add_argument('--degree',
                        type=int,
                        default=20,
                        help='the maximal degree of monomials')
    parser.add_argument('--generators',
                        type=int,
                        default=10,
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
    parser.add_argument('--l',
                        type=float,
                        default=0.1,
                        help='the parameter for Poisson distribution')

    # agent parameters
    parser.add_argument('--algorithm',
                        choices=['ppo', 'pg'],
                        default='ppo',
                        help='the training algorithm')
    parser.add_argument('--gam',
                        type=float,
                        default=0.99,
                        help='the discount rate')
    parser.add_argument('--lam',
                        type=float,
                        default=0.97,
                        help='the generalized advantage parameter')
    parser.add_argument('--eps',
                        type=float,
                        default=0.2,
                        help='the clip ratio for PPO')

    # policy model
    parser.add_argument('--policy_model',
                        choices=['mlp', 'pmlp', 'transformer'],
                        default='pmlp',
                        help='the policy network type')
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
                        default=80,
                        help='policy model gradient updates per epoch')
    parser.add_argument('--policy_kld_limit',
                        type=float,
                        default=0.01,
                        help='the KL divergence limit')
    parser.add_argument('--policy_weights',
                        type=str,
                        default="",
                        help='filename for initial policy weights')

    # value model
    parser.add_argument('--value_model',
                        choices=['none', 'mlp', 'pairsleft', 'agent', 'degree'],
                        default='none',
                        help='the value network type')
    parser.add_argument('--value_hl',
                        type=int, nargs='*',
                        default=[128],
                        help='the hidden layers in the value model')
    parser.add_argument('--value_lr',
                        type=float,
                        default=1e-3,
                        help='the value model learning rate')
    parser.add_argument('--value_updates',
                        type=int,
                        default=80,
                        help='value model gradient updates per epoch')
    parser.add_argument('--value_weights',
                        type=str,
                        default="",
                        help='filename for initial value weights')

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
                        default=2500,
                        help='the number of epochs')
    parser.add_argument('--max_episode_length',
                        type=lambda x: int(x) if x.lower() != 'none' else None,
                        default=500,
                        help='the max number of interactions per episode')
    parser.add_argument('--batch_size',
                        type=lambda x: int(x) if x.lower() != 'none' else None,
                        default=64,
                        help='the size of batches in training')
    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='how much information to print')
    parser.add_argument('--save_freq',
                        type=int,
                        default=100,
                        help='how often to save the weights')
    parser.add_argument('--logdir',
                        type=str,
                        default='data/runs',
                        help='the base directory for training runs')
    parser.add_argument('--parallel',
                        type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='whether to parallelize rollouts')
    parser.add_argument('--use_gpu',
                        type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='whether to use a GPU if available')

    return parser


def make_env(args):
    """Return the training environment for this run."""
    if args.environment in ['CartPole-v0', 'CartPole-v1', 'LunarLander-v2']:
        env = gym.make(args.environment)
    elif args.environment == "RandomPolynomialIdeal":
        ideal_gen = RandomIdealGenerator(args.variables, args.degree, args.generators, args.l,
                                         constants=args.constants, degrees=args.degree_distribution)
        env = BuchbergerEnv(ideal_gen, elimination=args.elimination, rewards=args.rewards)
        env = LeadMonomialsWrapper(env, k=args.k)
    elif args.environment == "MixedRandomBinomialIdeal":
        ideal_gen = MixedRandomBinomialIdealGenerator(args.variables,
                                                      list(range(5, args.degree+1)),
                                                      list(range(4, args.generators+1)),
                                                      constants=args.constants, degrees=args.degree_distribution,
                                                      homogeneous=args.homogeneous, pure=args.pure)
        env = BuchbergerEnv(ideal_gen, elimination=args.elimination, rewards=args.rewards)
        env = LeadMonomialsWrapper(env, k=args.k)
    else:
        ideal_gen = RandomBinomialIdealGenerator(args.variables, args.degree, args.generators,
                                                 constants=args.constants, degrees=args.degree_distribution,
                                                 homogeneous=args.homogeneous, pure=args.pure)
        env = BuchbergerEnv(ideal_gen, elimination=args.elimination, rewards=args.rewards)
        env = LeadMonomialsWrapper(env, k=args.k)
    return env


def make_policy_network(args):
    """Return the policy network for this run."""
    dims = {'CartPole-v0': (4, 2),
            'CartPole-v1': (4, 2),
            'LunarLander-v2': (8, 4),
            'RandomBinomialIdeal': (2 * args.variables * args.k, 1),
            'MixedRandomBinomialIdeal': (2 * args.variables * args.k, 1),
            'RandomPolynomialIdeal': (2 * args.variables * args.k, 1)}[args.environment]

    if args.environment in ['RandomBinomialIdeal', 'MixedRandomBinomialIdeal', 'RandomPolynomialIdeal']:
        if args.policy_model == 'pmlp':
            policy_network = ParallelMultilayerPerceptron(args.policy_hl)
        else:
            policy_network = TransformerPMLP(128, [64], 8)
    else:
        policy_network = MultilayerPerceptron(dims[1], args.policy_hl)

    if args.policy_weights != "":
        policy_network.load_weights(args.policy_weights)

    return policy_network


def make_value_network(args):
    """Return the value network for this run."""
    dims = {'CartPole-v0': (4, 2),
            'CartPole-v1': (4, 2),
            'LunarLander-v2': (8, 4),
            'RandomBinomialIdeal': (2 * args.variables * args.k, 1),
            'MixedRandomBinomialIdeal': (2 * args.variables * args.k, 1),
            'RandomPolynomialIdeal': (2 * args.variables * args.k, 1)}[args.environment]

    if args.environment in ['CartPole-v0', 'CartPole-v1', 'LunarLander-v2']:
        if args.value_model == 'none':
            value_network = None
        else:
            value_network = MultilayerPerceptron(1, args.value_hl, final_activation='linear')
    else:
        if args.value_model == 'none':
            value_network = None
        elif args.value_model == 'pairsleft':
            value_network = PairsLeftBaseline(gam=args.gam)
        elif args.value_model == 'degree':
            value_network = AgentBaseline(BuchbergerAgent('degree'), gam=args.gam)
        elif args.value_model == 'agent':
            agent = PPOAgent(ParallelMultilayerPerceptron(args.policy_hl))
            agent.load_policy_weights(args.value_weights)
            value_network = AgentBaseline(agent, gam=args.gam)

    return value_network


def make_agent(args):
    """Return the agent for this run."""
    dims = {'CartPole-v0': (4, 2),
            'CartPole-v1': (4, 2),
            'LunarLander-v2': (8, 4),
            'RandomBinomialIdeal': (2 * args.variables * args.k, 1),
            'MixedRandomBinomialIdeal': (2 * args.variables * args.k, 1),
            'RandomPolynomialIdeal': (2 * args.variables * args.k, 1)}[args.environment]

    if args.environment in ['RandomBinomialIdeal', 'MixedRandomBinomialIdeal', 'RandomPolynomialIdeal']:
        action_dim_fn = lambda s: s[0]
    else:
        action_dim_fn = lambda s: dims[1]

    policy_network = make_policy_network(args)
    value_network = make_value_network(args)

    if args.algorithm == 'pg':
        agent = PGAgent(policy_network=policy_network, policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                        value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                        gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit)
    else:
        agent = PPOAgent(policy_network=policy_network, policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                         value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                         gam=args.gam, lam=args.lam, eps=args.eps, kld_limit=args.policy_kld_limit)
    return agent


def make_logdir(args):
    """Return the directory name for this run."""
    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.datetag:
        run_name = time_string + '_' + args.name
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
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    env = make_env(args)
    agent = make_agent(args)
    logdir = make_logdir(args)
    save_args(logdir, args)
    print(logdir)
    agent.train(env, episodes=args.episodes, epochs=args.epochs,
                save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                max_episode_length=args.max_episode_length, parallel=args.parallel,
                batch_size=args.batch_size)
