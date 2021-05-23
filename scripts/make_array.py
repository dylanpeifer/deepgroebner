"""Generate and save the NumPy array of lead exponents for sampled ideals.

Run this script as

    python scripts/make_array.py <distribution>

Input is taken from

    data/stats/<distribution>/<distribution>.csv

and output is stored in the file

    data/stats/<distribution>/<distribution>.npy

This is script is slow and a bit hacky (note the use of exec and SymPy parsing),
but we only need to run it once on the distributions we care about. Consider
rewriting it or trying something else if you are using it more.
"""

import pandas as pd
import numpy as np
import sympy as sp
import sys


def parse_to_sympy(col):
    """Return list of SymPy ideals from the ideal column in df."""
    ideals = col.str.replace('|', ',').str.replace('{', '[').str.replace('}', ']').str.replace('^', '**')
    return [eval(ideal) for ideal in ideals]


def parse_to_numpy(col):
    """Return list of NumPy arrays from the ideal column in df."""
    ideals = parse_to_sympy(col)
    return np.array([[sum(f.monoms(), ()) for f in ideal] for ideal in ideals])


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Usage: python scripts/make_array.py <distribution>")
        sys.exit()

    dist = sys.argv[1]

    df = pd.read_csv(f"data/stats/{dist}/{dist}.csv")
    n = int(dist.split('-')[0])
    letters = [chr(i) for i in range(ord('a'), ord('a') + n)]
    R, *variables = sp.ring(",".join(letters), sp.FF(32003), 'grevlex')
    for i in range(len(letters)):
        exec(f"{letters[i]} = variables[{i}]")

    X = parse_to_numpy(df.Ideal)
    np.save(f"data/stats/{dist}/{dist}.npy", X)
