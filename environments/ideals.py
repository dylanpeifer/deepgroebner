# ideals.py
# Dylan Peifer
# 28 Apr 2019
"""Functions to generate ideals."""

import numpy as np
from os import listdir
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


class FixedIdealGenerator:
    """Yield repeated copies of the list of polynomials F."""
    
    def __init__(self, F):
        self.F = [f.copy() for f in F]

    def __next__(self):
        return [f.copy() for f in self.F]

    def __iter__(self):
        return self


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


class FromDirectoryIdealGenerator:
    """Yield ideals from several different files."""
    
    def __init__(self, directory, ring, p='weighted'):
        self.gens = [FromFileIdealGenerator(directory + '/' + f, ring)
                     for f in listdir(directory)]
        self.p = self._make_probability(directory, p)
        self.current_gen = np.random.choice(self.gens, p=self.p)

    def __next__(self):
        return next(self.current_gen)

    def __iter__(self):
        return self
    
    def update(self):
        """Randomly select a new file generator from the directory."""
        self.current_gen = np.random.choice(self.gens, p=self.p)
    
    def _make_probability(self, directory, p):
        filenames = listdir(directory)
        n = len(filenames)
        if p == 'uniform':
            return np.full((n,), 1/n)
        elif p == 'weighted':
            weights = np.zeros((n,))
            for i, filename in enumerate(filenames):
                with open(directory + '/' + filename, 'r') as f:
                    weights[i] = sum(1 for line in f)
            return weights / np.sum(weights)
        else:
            assert len(p) == n and sum(p) == 1
            return p


class RandomBinomialIdealGenerator:
    """Yield random examples of binomial ideals."""

    def __init__(self, variables, degree, size,
                 domain=sp.FF(32003),
                 order='grevlex',
                 homogeneous=False,
                 pure=False,
                 seed=None):
        if isinstance(variables, int):
            variables = 'x:' + str(variables)
        R, _ = sp.xring(variables, domain, order)
        self.ring = R
        self.rand = np.random.RandomState(seed=seed)
        self.degree = degree
        self.size = size
        self.homogeneous = homogeneous
        self.pure = pure

    def __next__(self):
        return random_binomial_ideal(self.ring, self.degree, self.size,
                                     homogeneous=self.homogeneous,
                                     pure=self.pure)

    def __iter__(self):
        return self

    def seed(self, seed=None):
        self.rand.seed(seed)
