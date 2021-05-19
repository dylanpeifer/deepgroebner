#!/usr/bin/env python
"""The classic cyclic benchmark."""

import sys

from deepgroebner.ideals import cyclic
from deepgroebner.buchberger import buchberger


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: cyclic.py <number>")
    else:
        I = cyclic(int(sys.argv[1]))
        G = buchberger(I)
        print(len(G))
