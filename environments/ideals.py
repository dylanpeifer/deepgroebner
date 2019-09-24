# ideals.py
# Dylan Peifer
# 28 Apr 2019
"""Functions to generate ideals."""

import numpy as np
import sympy as sp


def random_partition(n, k):
    """Use bars and stars to get a random partition of integer n into k pieces."""
    bars = np.random.choice(n+k-1, k-1, replace=False)
    bars.sort()
    counts = np.empty(k, dtype=int)
    counts[0] = bars[0]
    for i in range(1, len(bars)):
        counts[i] = bars[i] - bars[i-1] - 1
    counts[k-1] = n + k - 1 - bars[-1] - 1
    return tuple(counts)


def random_coefficient(ring, nonzero=False):
    """Return a random coefficient from the coefficient ring of ring."""
    assert ring.domain.is_Field, "polynomial ring must be over a field"
    c = ring.domain.characteristic()
    if c == 0:
        # TODO: decide on something for non-finite fields
        return sp.Rational(np.random.randint(-99, 100), np.random.randint(100))
    elif nonzero:
        return ring.one * np.random.randint(1, c)
    else:
        return ring.one * np.random.randint(c)


def random_binomial(ring, degree, homogeneous=False, pure=False):
    """Return a random binomial from the ring in given degree."""
    if homogeneous:
        d1, d2 = degree, degree
    else:
        d1, d2 = np.random.randint(1, degree+1, size=2)
    n = len(ring.gens)
    e1, e2 = random_partition(d1, n), random_partition(d2, n)
    while e1 == e2:
        e2 = random_partition(d2, n)
    m1, m2 = (np.product([x**k for x, k in zip(ring.gens, e)]) for e in [e1, e2])
    if ring.order(e1) < ring.order(e2):
        m1, m2, = m2, m1
    if pure:
        return m1 - m2
    else:
        return m1 + random_coefficient(ring, nonzero=True) * m2


def random_binomial_ideal(ring, degree, size, homogeneous=False, pure=False):
    """Return a random binomial ideal in ring as a list."""
    return [random_binomial(ring, degree, homogeneous=homogeneous, pure=pure)
            for _ in range(size)]


def FixedIdealGenerator(F):
    """Yield repeated copies of the list of polynomials F."""
    while True:
        yield [f.copy() for f in F]


def FromFileIdealGenerator(filename, ring):
    """Yield ideals from the given file."""
    while True:
        with open(filename, 'r') as f:
            while True:
                try:
                    F = []
                    size = int(next(f).split()[0])
                    for _ in range(size):
                        c, *expvs = [int(i) for i in next(f).split()]
                        e1, e2 = expvs[:len(expvs)//2], expvs[len(expvs)//2:]
                        m1, m2 = (np.product([x**k for x, k in zip(ring.gens, e)]) for e in [e1, e2])
                        F.append(m1 + c * m2)
                    yield F
                except StopIteration:
                    break