"""Generators of ideals."""

import numpy as np
import os
import sympy as sp


def cyclic(n, coefficient_ring=sp.FF(32003), order='grevlex'):
    """Return the cyclic-n ideal."""
    R, gens = sp.xring('x:' + str(n), coefficient_ring, order)
    F = [sum(np.prod([gens[(i+k) % n] for k in range(d)]) for i in range(n))
         for d in range(1, n)]
    return F + [np.product(gens) - 1]


class FixedIdealGenerator:
    """Generator of repeated copies of a fixed ideal.

    Parameters
    ----------
    F : list
        The list of polynomial generators for the ideal.

    Examples
    --------
    >>> import sympy as sp
    >>> R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    >>> F = [y - x**2, z - x**3]
    >>> ideal_gen = FixedIdealGenerator(F)
    >>> next(ideal_gen)
    [32002 mod 32003*x**2 + y, 32002 mod 32003*x**3 + z]

    """

    def __init__(self, F):
        self.F = [f.copy() for f in F]

    def __next__(self):
        return [f.copy() for f in self.F]

    def __iter__(self):
        return self


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


def random_monomial(ring, degree):
    pass


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


class RandomBinomialIdealGenerator:
    """Yield random examples of binomial ideals.

    Parameters
    ----------
    n : int
        The number of variables.
    d : int
        The maximum degree of a chosen monomial.
    s : int
        The number of generators of each ideal.
    coefficient_ring : ring, optional
        The coefficient ring for the polynomials.
    order : {'grevlex', 'lex', 'grlex'}, optional
        The monomial order.
    constants : bool, optional
        Whether to include constants as monomials.
    degrees : {'uniform', 'weighted', 'maximum'}, optional
        The distribution of degrees of monomials.
    homogeneous : bool, optional
        Whether the binomials are homogeneous.
    pure : bool, optional
        Whether the binomials are pure.
    seed : int, optional
        The seed for the random number generator.

    Examples
    --------

    """

    def __init__(self, n, d, s,
                 coefficient_ring=sp.FF(32003),
                 order='grevlex',
                 homogeneous=False,
                 pure=False,
                 seed=None):
        R, _ = sp.xring('x:' + str(n), coefficient_ring, order)
        self.ring = R
        self.degree = degree
        self.size = size
        self.homogeneous = homogeneous
        self.pure = pure
        self.rand = np.random.RandomState(seed=seed)

    def __next__(self):
        return random_binomial_ideal(self.ring, self.degree, self.size,
                                     homogeneous=self.homogeneous,
                                     pure=self.pure)

    def __iter__(self):
        return self

    def seed(self, seed=None):
        self.rand.seed(seed)


def FromFileIdealGenerator(filename, ring):
    """Yield ideals from the given file.

    The file is repeatedly looped over, so the generator is never exhausted.

    Ideal files must consist of entries with the format

        n m
          c1 e11 e12
          c2 e21 e22
          ...
          cn en1 en2

    where n is the number of generators, m is the number of reduced Groebner
    basis elements (unused except as a sanity check), the ci are coefficients
    and the eij are space-separated exponent vectors.  For example, the entry

        5 6
          2634 0 4 1 2 1 2
          9889 0 1 1 0 0 2
          -8019 2 3 0 3 1 1
          -12872 4 0 1 3 1 1
          8580 3 0 0 0 1 2

    represents the ideal

        [y**4*z + 2634*x**2*y*z**2,
         y*z + 9889*z**2,
         x**2*y**3 - 8019*x**3*y*z,
         x**4*z - 12872*x**3*y*z,
         x**3 + 8580*y*z**2]

    Note that the current format restrictions mean that ideal files can only
    contain binomial ideals.

    Ideal files can be generated using the Macaulay2 script make_bins.m2 in the
    m2/ directory, which generates ideals binned by the size of their reduced
    Groebner basis. Binned ideals are useful for variance reduction in
    training.

    Parameters
    ----------
    filename : str
        The file containing the ideals.
    ring : ring
        The polynomial ring that contains the ideals.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("x,y,z", sp.FF(32003), 'grevlex')
    >>> ideal_gen = FromFileIdealGenerator('data/bins/3-5-5-uniform-homog/bin18.txt', ring)
    >>> next(ideal_gen)
    [y**4*z + 2634 mod 32003*x**2*y*z**2,
     y*z + 9889 mod 32003*z**2,
     x**2*y**3 + 23984 mod 32003*x**3*y*z,
     x**4*z + 19131 mod 32003*x**3*y*z,
     x**3 + 8580 mod 32003*y*z**2]

    """
    while True:
        with open(filename, 'r') as f:
            while True:
                try:
                    F = []
                    size = int(next(f).split()[0])
                    for _ in range(size):
                        c, *expvs = [int(i) for i in next(f).split()]
                        e1, e2 = expvs[:len(expvs)//2], expvs[len(expvs)//2:]
                        m1, m2 = (np.prod([x**k for x, k in zip(ring.gens, e)])
                                  for e in [e1, e2])
                        F.append(m1 + c * m2)
                    yield F
                except StopIteration:
                    break


class FromDirectoryIdealGenerator:
    """Generator of ideals from several different files.

    Files are repeatedly looped over, so the generator is never exhausted.

    Ideal files must follow the format read by the class
    FromFileIdealGenerator, as this class simply creates generators for each
    file and then swaps between them when update is called.

    Parameters
    ----------
    directory : str
        The directory which contains the files of ideals.
    ring : ring
        The polynomial ring that contains the ideals.
    p : {'weighted', 'uniform'}, optional
        Whether to pick files uniformly at random or weighted by size.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("a,b,c,d,e", sp.FF(32003), 'grevlex')
    >>> ideal_gen = FromDirectoryIdealGenerator('data/bins/5-10-5-uniform', ring)
    >>> next(ideal_gen)
    [b**4*c**2*e**4 + 7134 mod 32003*a**2*b*c*d**2*e**2,
     b**3*e + 26763 mod 32003*c,
     c*d**5*e + 30236 mod 32003*a,
     c**2*d**2*e**6 + 31780 mod 32003*b**2*c**2,
     b*d**2 + 22128 mod 32003*c*d]
    >>> ideal_gen.update()
    >>> next(ideal_gen)
    [b**3*c**4 + 11665 mod 32003*a*b**2,
     a**2*b**4*c**2*e**2 + 4466 mod 32003*a**2*b**2*c*d*e**4,
     a**7*d*e + 19544 mod 32003*a**4*c,
     a**3*c*d**2 + 8049 mod 32003*c,
     b**2*d**2 + 1606 mod 32003*d*e]

    """

    def __init__(self, directory, ring, p='weighted'):
        self.gens = [FromFileIdealGenerator(directory + '/' + f, ring)
                     for f in os.listdir(directory)]
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
        filenames = os.listdir(directory)
        n = len(filenames)
        if p == 'uniform':
            return np.ones(n) / n
        elif p == 'weighted':
            weights = np.zeros(n)
            for i, filename in enumerate(filenames):
                with open(directory + '/' + filename, 'r') as f:
                    weights[i] = sum(1 for line in f)
            return weights / np.sum(weights)
        else:
            assert len(p) == n and sum(p) == 1
            return p
