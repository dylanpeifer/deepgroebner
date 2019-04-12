# hilbert.py
# Dylan Peifer and Lily Silverstein
# 12 Apr 2019
"""An environment for computing Hilbert functions."""

import numpy as np


def pivot(A, p):
    """Pivots on 1xn p to reduce mxn A.
    
    Rows of A are monomials generating I, return left <I, p> and right <I:p>.
    """
    left = []
    right =[]
    for m in A:
        if not all(m >= p):
            left.append(m)
        right.append(np.where(m >= p, m - p, 0))
    left.append(p)
    return np.array(left), minimalize(np.array(right))


def minimalize(A):
    """Minimalize the generators given as rows of A."""
    Amin = []
    for m in A:
        if all([not all(m >= g) for g in Amin]):
            Amin = [g for g in Amin if not all(g >= m)]
            Amin.append(m)
    return np.array(Amin)


def terminal(A):
    """Return if A is a base case."""
    nontrivial = 0
    for m in A:
        if np.count_nonzero(m) > 1:
            nontrivial += 1
    return nontrivial <= 1


def strategy(A):
    """Return the column index with most nonzero entries."""
    return np.argmax(np.count_nonzero(A, axis=0))


def base_cases(A):
    if terminal(A):
        return 1
    else:
        p = np.zeros(A.shape[1])
        p[strategy(A)] = 1
        A1, A2 = pivot(A, p)
        print(A1, A2)
        return base_cases(A1) + base_cases(A2)
