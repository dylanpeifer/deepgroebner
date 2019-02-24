# Trained models for Breakout

Models were trained with the script `trainatari.py` in this directory using commit 047ec409.
Hyperparameters are in the script. Training was performed on an AWS p2.xlarge running the
Deep Learning AMI (Ubuntu) Version 21.2 - ami-0a47106e391391252 and using the tensorflow_p36
environment. Average rewards over 100 episodes on the test environment were

 0:  14.59
 1:  51.89
 2:  90.62
 3: 112.25
 4: 172.97
 5: 191.41
 6: 182.70
 7: 160.30
 8: 232.27
 9: 139.05
10: 218.83
11: 235.24
12: 180.50
13:  97.05
14: 143.12
15: 111.02
16: 186.08
17: 168.59
18: 155.62
19: 172.85
20: 183.54
21: 185.86
22:  98.95
23: 181.88
24: 183.74

Total training time was roughly 50 hours.
