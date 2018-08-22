# deepgroebner
applications of deep learning to Gaussian elimination and Buchberger's algorithm

## Selection Strategies

The Macaulay2 package SelectionStrategies implements a few tools for
experimenting with different selection strategies. It depends on the package
Reduction, which is an interface to some engine code that implements the
division algorithm (doing this at the top-level is really slow). This engine
code is NOT available in a standard Macaulay2 install - it lives on a branch of
my fork of the M2 repository on GitHub.  To get it to work you could replace
all calls to divAlg by equivalent calls to divide, compile the version of M2 on
my branch, or wait until I get around to putting this engine code in Macualay2
for real.

## Row Reduction

The files in `rowreduce` implement some experiments with row reduction using
keras/tensorflow/numpy. The top of `main.py` gives instructions.
