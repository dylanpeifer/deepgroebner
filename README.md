# DeepGroebner

Applications of reinforcement learning to selection strategies
in Buchberger's algorithm.

## Requirements and Installation

The main codebase is a Python package in the `deepgroebner/` directory. To
install dependencies create a conda environment from the included YAML file
with

    conda env create -f deepgroebner.yml
    conda activate deepgroebner

and then install the `deepgroebner` package with

    pip install -e .

[Macaulay2](http://www2.macaulay2.com/Macaulay2/) packages and scripts in the
`m2/` directory are used for generating statistics and experimenting with known
strategies. To generate the statistics, first follow the [installation
instructions](http://www2.macaulay2.com/Macaulay2/Downloads/) for Macaulay2 and
then verify that you can enter the Macaulay2 interpreter by typing

    M2

at the command line. Then, place the files `SelectionStrategies.m2` and
`Ideals.m2` in the packages directory of your Macaulay2 install. Finally, type

    i1: needsPackage "SelectionStrategies";
    i2: needsPackage "Ideals";
    
to verify that the packages are available.
    
## Tests

Basic tests of the Python code can be run with

    python -m pytest
    
Basic tests for the Macaulay2 packages can be run inside the Macaulay2
interpreter as

    i1: needsPackage "SelectionStrategies";
    i2: check SelectionStrategies

and

    i3: needsPackage "Ideals";
    i4: check Ideals

The main script can be called on the OpenAI Gym environments `CartPole-v0`,
`CartPole-v1`, and `LunarLander-v2`, which provides a way to check the PPO
implementation on known benchmarks. The commands

    python run.py --environment CartPole-v0 --value_model mlp --epochs 25 --verbose 2

    python run.py --environment CartPole-v1 --value_model mlp --epochs 50 --verbose 2

    python run.py --environment LunarLander-v2 --value_model mlp --epochs 500 --verbose 2
    
should take a couple minutes, several minutes, and a few hours respectively.
Final performance should match or exceed `mean_returns` of 195.0, 475.0, and
200.0 on these environments.

## Running Experiments

All experiments start with the `run.py` script in this directory. For a list of
arguments type
    
    python run.py --help

Defaults are provided for all arguments in the script.

For example, we can train an agent on 3-20-10-uniform using

    python run.py --variables 3 --degree 20 --generators 10 \\
                  --degree_distribution uniform \\
                  --value_model agent --value_updates 0

By default, the script will create a subdirectory in `data/runs` where it will
store TensorBoard logs, model checkpoints, and a complete list of script
arguments in the file `args.txt`. After copying this file to the top
directory, we can rerun the same experiment with

    python run.py @args.txt

## Generating Statistics

To generate statistics, use the `make_stats.m2` and `make_stats2.m2` scripts in
the `m2/` directory.  Run these scripts as

    M2 --script make_stats.m2 n d s consts degs homog pure samples

where n, d, s, and samples are integers, consts, homog, and pure are 0 or 1
representing false or true, and degs is one of {uniform, weighted, maximum}.
Output is stored in the `data/stats/` directory, and files will be appended to
rather than replaced if they already exist.

## Viewing Results
TensorBoard can be used to visualize training runs with

    tensorboard --logdir data/runs

The statistics are aggregated and processed in Jupyter notebooks in the
`notebooks/` directory.
