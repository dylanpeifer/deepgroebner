# DeepGroebner
Applications of deep learning to selection strategies in Buchberger's
algorithm.

## Requirements and Installation
Environments depend on NumPy and SymPy, while agents use TensorFlow. To install
dependencies create a conda environment from the included YAML file with

    conda env create -f deepgroebner.yml
    conda activate deepgroebner

To remove (or update), first remove the `deepgroebner` environment:

    conda remove env --name deepgroebner --all
    
## Tests
Tests can be run with

    python -m pytest

## Command line interface
Here is an example of using the command line interface to train a network
    
    python run.py --help
    
    python run.py --environment CartPole-v0 --name MyCartPoleExample1 --epochs 20 --verbose 1
    
## To view results via tensorboard

    tensorboard --logdir data/runs    
    