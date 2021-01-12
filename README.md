# DeepGroebner

Applications of reinforcement learning to selection strategies in Buchberger's algorithm.

This is the main code repository for the paper

* [Dylan Peifer, Michael Stillman, and Daniel Halpern-Leistner. Learning selection strategies in Buchberger's
algorithm. In *Proceedings of the 37th International Conference on Machine Learning (ICML
2020).*](https://icml.cc/virtual/2020/poster/6742) [arXiv:2005.01917](https://arxiv.org/abs/2005.01917)

with some precomputed statistics and training data available at
[10.5281/zenodo.3676043](https://doi.org/10.5281/zenodo.3676043). For the exact version of the code used to generate
results, refer to the commit hashes in the file `experiments.txt` in the training data, and also see the
[releases](https://github.com/dylanpeifer/deepgroebner/releases). This repository is still in active development.

## Requirements and Installation

The main codebase is a Python package in the `deepgroebner/` directory. To install dependencies create a
[conda](https://www.anaconda.com/products/individual) environment from the included YAML file with

    conda env create -f deepgroebner.yml
    conda activate deepgroebner

and then install the `deepgroebner` package with

    pip install -e .

[Macaulay2](http://www2.macaulay2.com/Macaulay2/) packages in the `m2/` directory are used for generating statistics
and experimenting with known strategies. To generate the statistics, first follow the [installation
instructions](http://www2.macaulay2.com/Macaulay2/Downloads/) for Macaulay2 and then verify that you can enter the
Macaulay2 interpreter by typing

    M2

at the command line in this directory. Finally, type

    i1: needsPackage("Ideals", FileName => "m2/Ideals.m2");
    i2: needsPackage("SelectionStrategies", FileName => "m2/SelectionStrategies.m2");    

to verify that the packages are available.
    
## Tests

After installation, basic tests of the Python code can be run with

    python -m pytest
    
Basic tests for the Macaulay2 packages can be run inside the Macaulay2 interpreter as

    i1: needsPackage("Ideals", FileName => "m2/Ideals.m2");
    i2: needsPackage("SelectionStrategies", FileName => "m2/SelectionStrategies.m2");    
    i3: check Ideals
    i4: check SelectionStrategies

The main training script can be called on the OpenAI Gym environments `CartPole-v0`, `CartPole-v1`, and
`LunarLander-v2`, which provides a way to check the PPO implementation on known benchmarks. The commands

    python scripts/train.py --environment CartPole-v0 --value_model mlp --epochs 25 --verbose 2

    python scripts/train.py --environment CartPole-v1 --value_model mlp --epochs 50 --verbose 2

    python scripts/train.py --environment LunarLander-v2 --value_model mlp --epochs 500 --verbose 2
    
should take a couple minutes, several minutes, and a few hours respectively. Final performance should match or exceed
`mean_returns` of 195.0, 475.0, and 200.0 on these environments.

## Running Experiments

All experiments start with the `train.py` script in the `scripts` directory. For a list of arguments type
    
    python scripts/train.py --help

Defaults are provided for all arguments in the script.

For example, we can train an agent on 3-20-10-weighted using

    python scripts/train.py --distribution 3-20-10-weighted --value_model degree

By default, the script will create a subdirectory in `data/train/` where it will store TensorBoard logs, model
checkpoints, and a complete list of script arguments in the file `args.txt`. After copying this file to the top
directory, we can rerun the same experiment with

    python scripts/train.py @args.txt

Evaluation of trained models is performed with the `eval.py` script in the `scripts` directory, which has similar
arguments. In particular, `--policy_weights` should receive the file containing the trained model weights.

## Generating Statistics

To generate statistics, use the `make_dist.m2`, `make_stats.m2`, and `make_strat.m2` scripts in the `script`
directory. The basic workflow is to create a file of sampled ideals with

    M2 --script scripts/make_dist.m2 <distribution> <samples> <seed>

and then use this sample to compute values with

    M2 --script scripts/make_stats.m2 <distribution>

or compute performance of strategies with

    M2 --script scripts/make_strat.m2 <distribution> <strategy> <seed>

where seeding is only important for Random selection. Output is stored in CSV files in the `data/stats/` directory.

For example, we can compute statistics and run Degree selection on 10000 samples from 3-20-10-weighted with

    M2 --script scripts/make_dist.m2 3-20-10-weighted 10000 123
    M2 --script scripts/make_stats.m2 3-20-10-weighted
    M2 --script scripts/make_strat.m2 3-20-10-weighted degree

