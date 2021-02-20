#!/usr/bin/env python
"""Entry point for all training runs."""

import argparse
import datetime
import numpy as np
import os
import json

import gym

from deepgroebner.buchberger import LeadMonomialsEnv, BuchbergerAgent
from deepgroebner.pg import PGAgent, PPOAgent
from deepgroebner.networks import MultilayerPerceptron, ParallelMultilayerPerceptron, AttentionPMLP, TransformerPMLP, PairsLeftBaseline, AgentBaseline, TransformerPMLP_Score_MHA
from deepgroebner.new_networks import TransformerPMLP_Score_Q, TransformerPMLP_DVal, TransformerPMLP_MHA_Q_Scorer
from deepgroebner.wrapped import CLeadMonomialsEnv

from deepgroebner.environments import VectorEnv, AlphabeticalEnv

def make_parser():
    """Return the command line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train a new model",
                                     fromfile_prefix_chars='@')

    env = parser.add_argument_group('environment', 'environment type')
    env.add_argument('--environment',
                     choices=['RandomBinomialIdeal', 'RandomIdeal',
                              'CartPole-v0', 'CartPole-v1', 'LunarLander-v2', 'VectorEnv', 'AlphabeticalEnv'],
                     default='RandomBinomialIdeal',
                     help='training environment')
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

    alg = parser.add_argument_group('algorithm', 'algorithm parameters')
    alg.add_argument('--algorithm',
                     choices=['ppo', 'pg'],
                     default='ppo',
                     help='training algorithm')
    alg.add_argument('--gam',
                     type=float,
                     default=0.99,
                     help='discount rate')
    alg.add_argument('--lam',
                     type=float,
                     default=0.97,
                     help='generalized advantage parameter')
    alg.add_argument('--eps',
                     type=float,
                     default=0.2,
                     help='clip ratio for PPO')

    policy = parser.add_argument_group('policy model')
    policy.add_argument('--policy_model',
                        choices=['mlp', 'pmlp', 'apmlp', 'tpmlp', 'tpmlp_q_scorer', 'tpmlp_MHA_scorer', 'dval', 'qmha'],
                        default='pmlp',
                        help='policy network type')
    policy.add_argument('--policy_kwargs',
                        type=json.loads,
                        default={"hidden_layers": [128]},
                        help='arguments to policy model constructor, passed through json.loads')
    policy.add_argument('--policy_lr',
                        type=float,
                        default=1e-4,
                        help='policy model learning rate')
    policy.add_argument('--policy_updates',
                        type=int,
                        default=40,
                        help='policy model updates per epoch')
    policy.add_argument('--policy_kld_limit',
                        type=float,
                        default=0.01,
                        help='KL divergence limit used for early stopping')
    policy.add_argument('--policy_weights',
                        type=str,
                        default="",
                        help='filename for initial policy weights')
    policy.add_argument('--scorer',
                        type = bool,
                        default = True,
                        help = 'have multi objective training')
    policy.add_argument('--score_weight',
                        type = float,
                        default=1e-3,
                        help='weight gradients of l2 loss')

    value = parser.add_argument_group('value model')
    value.add_argument('--value_model',
                       choices=['none', 'mlp', 'pairsleft', 'degree'],
                       default='none',
                       help='value network type')
    value.add_argument('--value_kwargs',
                       type=json.loads,
                       default={"hidden_layers": [128]},
                       help='arguments to value model constructor, passed through json.loads')
    value.add_argument('--value_lr',
                       type=float,
                       default=1e-3,
                       help='the value model learning rate')
    value.add_argument('--value_updates',
                       type=int,
                       default=1,
                       help='value model updates per epoch')
    value.add_argument('--value_weights',
                       type=str,
                       default="",
                       help='filename for initial value weights')

    train = parser.add_argument_group('training')
    train.add_argument('--episodes',
                       type=int,
                       default=100,
                       help='number of episodes per epoch')
    train.add_argument('--epochs',
                       type=int,
                       default=2500,
                       help='number of epochs')
    train.add_argument('--max_episode_length',
                       type=lambda x: int(x) if x.lower() != 'none' else None,
                       default=500,
                       help='max number of interactions per episode')
    train.add_argument('--batch_size',
                       type=lambda x: int(x) if x.lower() != 'none' else None,
                       default=64,
                       help='size of batches in training')
    train.add_argument('--parallel',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to parallelize rollouts')
    train.add_argument('--use_gpu',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to use a GPU if available')
    train.add_argument('--verbose',
                       type=int,
                       default=0,
                       help='how much information to print')

    save = parser.add_argument_group('saving')
    save.add_argument('--name',
                       type=str,
                       default='run',
                       help='name of training run')
    save.add_argument('--datetag',
                       type=lambda x: str(x).lower() == 'true',
                       default=True,
                       help='whether to append current time to run name')
    save.add_argument('--logdir',
                       type=str,
                       default='data/runs',
                       help='base directory for training runs')
    save.add_argument('--save_freq',
                       type=int,
                       default=100,
                       help='how often to save the models')

    return parser


def make_env(args):
    """Return the training environment for this run."""
    if args.environment == 'VectorEnv':
        env = VectorEnv()
    elif args.environment == 'AlphabeticalEnv':
        env = AlphabeticalEnv()
    elif args.environment in ['CartPole-v0', 'CartPole-v1', 'LunarLander-v2']:
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
        elif args.policy_model == 'tpmlp_MHA_scorer':
            policy_network = TransformerPMLP_Score_MHA(**args.policy_kwargs)
        elif args.policy_model == 'dval':
            policy_network = TransformerPMLP_DVal(**args.policy_kwargs)
        else:
            pass
        batch = np.zeros((1, 10, 2 * args.k * int(args.distribution.split('-')[0])), dtype=np.int32)
    if args.environment == 'VectorEnv':
        batch = np.zeros((1, 10, 64), dtype=np.int32)
    elif args.environment == 'AlphabeticalEnv':
        batch = np.zeros((1, 10, 1128), dtype=np.int32)
    policy_network(batch)  # build network
    if args.policy_weights != "":
        policy_network.load_weights(args.policy_weights)
    return policy_network


def make_value_network(args):
    """Return the value network for this run."""
    kwargs = args.value_kwargs
    if args.value_model == 'none':
        value_network = None
    elif args.environment == 'LunarLander-v2':
        assert "output_dim" not in kwargs and "final_activation" not in kwargs
        value_network = MultilayerPerceptron(1, final_activation='linear', **args.value_kwargs)
        batch = np.zeros((1, 8), dtype=np.float32)
        value_network(batch)  # build network
    elif args.environment in ['CartPole-v0', 'CartPole-v1']:
        assert "output_dim" not in kwargs and "final_activation" not in kwargs
        value_network = MultilayerPerceptron(1, final_activation='linear', **args.value_kwargs)
        batch = np.zeros((1, 4), dtype=np.float32)
        value_network(batch)  # build network
    else:
        if args.value_model == 'pairsleft':
            value_network = PairsLeftBaseline(gam=args.gam)
        else:
            value_network = 'env'
    if args.value_weights != "":
        value_network.load_weights(args.value_weights)
    return value_network


def make_agent(args):
    """Return the agent for this run."""
    policy_network = make_policy_network(args)
    value_network = make_value_network(args)
    agent_fn = PGAgent if args.algorithm == 'pg' else PPOAgent
    agent = agent_fn(policy_network=policy_network, policy_lr=args.policy_lr, policy_updates=args.policy_updates,
                     value_network=value_network, value_lr=args.value_lr, value_updates=args.value_updates,
                     gam=args.gam, lam=args.lam, kld_limit=args.policy_kld_limit)
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


if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.use_gpu or args.parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    env = make_env(args)
    agent = make_agent(args)
    logdir = make_logdir(args)
    print("Saving run in", logdir)
    agent.train(env, episodes=args.episodes, epochs=args.epochs,
                save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                max_episode_length=args.max_episode_length, parallel=args.parallel,
                batch_size=args.batch_size)
