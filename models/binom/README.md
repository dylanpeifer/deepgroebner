# Trained models for binomial Buchberger

Models were trained with the script `trainbinom.py` in this directory using commit 7834901.
Hyperparameters are in the script. Training was performed on an AWS c5.xlarge running the
Deep Learning AMI (Ubuntu) Version 21.2 - ami-0a47106e391391252 and using the tensorflow_p36
environment. Average rewards over each epoch of training are in `rewards.txt` in this
directory. Total training time was roughly 72 hours.

