# badmatrices.py
# Dylan Peifer
# 17 Sep 2018
"""Computing matrices where order makes large differences in row reduction.

To row reduce a matrix we can choose the rows in some order and at each choice
use the chosen row's lead term as a pivot. For some matrices the order of these
choices makes a difference in how many moves are needed. Our standard example
is

    [[1, 1],
     [1, 0]]

over F2, where choosing the second row first ends in one move, but choosing the
first row first requires two. This file is a (dumb brute force) attempt to find
more examples like this.
"""

import numpy as np
from itertools import permutations, product


def step(matrix, row):
    """Choose row as pivot and perform reduction. Return moves needed and
    resulting matrix."""
    (M, N) = matrix.shape
    moves = 0
    lead = next((i for i, x in enumerate(matrix[row, :]) if x != 0), None)
    if lead is not None:
        for i in range(N):
            if i != row and matrix[i, lead] != 0:
                matrix[i, :] = (matrix[i, :] + matrix[row, :]) % 2
                moves += 1
    return moves


def reduce(matrix, rows):
    """Reduce a matrix and return the number of moves needed."""
    total_moves = 0
    for row in rows:
        total_moves += step(matrix, row)
    return total_moves


def minmax_reduce(matrix):
    """Find and return min and max moves to reduce the matrix."""
    (M, N) = matrix.shape
    min_moves = M
    min_rows = None
    max_moves = 0
    max_rows = None
    for rows in permutations(range(M)):
        m = np.copy(matrix)
        moves = reduce(m, rows)
        if moves < min_moves or min_rows is None:
            min_moves = moves
            min_rows = rows
        if moves > max_moves or max_rows is None:
            max_moves = moves
            max_rows = rows
    return (min_moves, min_rows), (max_moves, max_rows)


def max_difference(n):
    """Return nxn matrix over F2 with max difference between min and max
    moves."""
    best = None
    best_diff = 0
    for prod in product(range(2), repeat=n*n):
        m = np.reshape(np.array(prod), (n, n))
        result = minmax_reduce(m)
        diff = result[1][0] - result[0][0]
        if diff > best_diff or best is None:
            best = m
            best_diff = diff
    return best


def is_reduced(matrix):
    """Return true if the current matrix is reduced."""
    (M, N) = matrix.shape
    for row in range(N):
        # find index of lead term in this row
        lead = next((i for i, x in enumerate(matrix[row, :]) if x != 0), None)
        # if this row has lead term then zero everything else in column
        if lead is not None:
            for i in range(N):
                if i != row and matrix[i, lead] != 0:
                    return False
    return True


if __name__ == "__main__":
    for i in range(1, 10):
        m = max_difference(i)
        print(m)
        print(minmax_reduce(m))
        print()
