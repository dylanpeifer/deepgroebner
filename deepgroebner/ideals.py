"""Generators of ideals."""

import itertools as it
import numpy as np
import sympy as sp


def cyclic(n, coefficient_ring=sp.FF(32003), order='grevlex'):
    """Return the cyclic-n ideal."""
    R, gens = sp.xring('x:' + str(n), coefficient_ring, order)
    F = [sum(np.prod([gens[(i+k) % n] for k in range(d)]) for i in range(n))
         for d in range(1, n)]
    return F + [np.product(gens) - 1]


def basis(ring, d):
    """Return the monomial basis of a polynomial ring in degree d.

    A monomial basis in degree d is just all monomials of degree d in the
    polynomial ring.

    Parameters
    ----------
    ring : polynomial ring
        Ring containing monomials.
    d : int
        Degree of the basis.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("x,y,z", sp.FF(32003), 'grevlex')
    >>> basis(ring, 2)
    [x**2, x*y, x*z, y**2, y*z, z**2]

    """
    if d == 0:
        return [ring.one]
    else:
        return [np.prod(c) for c in it.combinations_with_replacement(ring.gens, d)]


def degree_distribution(ring, d, dist='uniform', constants=False):
    """Return the probability distribution on degrees.

    Parameters
    ----------
    ring : polynomial ring
        Base ring for monomials.
    d : int
        Maximum degree included in the distribution.
    dist : {'uniform', 'weighted', 'maximum'}, optional
        Type of distribution on monomials.

        In a 'uniform' distribution, monomials are sampled uniformly at random
        from all monomials less than or equal to the maximum degree (which
        means higher degrees are more likely). In 'weighted', monomials are
        sampled with weights such that each degree has the same probability of
        appearing (i.e., degrees are sampled uniformly and then a monomial is
        picked uniformly at random from that degree). In 'maximum', monomials
        are sampled uniformly at random from all monomials of the maximum
        degree.
    constants : bool
        Whether to have nonzero probability of constants.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("x,y,z", sp.FF(32003), 'grevlex')
    >>> degree_distribution(ring, 3, dist='uniform')
    array([0.        , 0.15789474, 0.31578947, 0.52631579])
    >>> degree_distribution(ring, 3, dist='weighted')
    array([0.        , 0.33333333, 0.33333333, 0.33333333])

    """
    head = [1] if constants else [0]
    if dist == 'uniform':
        n = len(ring.gens)
        tail = [int(sp.binomial(n + i - 1, n - 1)) for i in range(1, d + 1)]
    elif dist == 'weighted':
        tail = d * [1]
    elif dist == 'maximum':
        tail = (d - 1) * [0] + [1]
    else:
        raise ValueError('unrecognized dist option')
    count = np.array(head + tail)
    return count / np.sum(count)


class IdealGenerator:
    """Abstract base class for all ideal generators.

    Derived classes must implement a __next__ method which returns
    a new list of SymPy polynomials representing an ideal. If randomness
    is used to generate polynomials then override the seed method.

    """

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        pass


def parse_ideal_dist(ideal_dist):
    """Return concrete IdealGenerator instance given by string ideal_dist."""
    dist_args = ideal_dist.split('-')
    if dist_args[0] == 'cyclic':
        n = int(dist_args[1])
        return FixedIdealGenerator(cyclic(n))
    elif dist_args[3] in ['uniform', 'weighted', 'maximum']:
        kwargs = {
            'n': int(dist_args[0]),
            'd': int(dist_args[1]),
            's': int(dist_args[2]),
            'dist': dist_args[3],
            'constants': 'consts' in dist_args,
            'homogeneous': 'homog' in dist_args,
            'pure': 'pure' in dist_args,
        }
        return RandomBinomialIdealGenerator(**kwargs)
    else:
        kwargs = {
            'n': int(dist_args[0]),
            'd': int(dist_args[1]),
            's': int(dist_args[2]),
            'lam': float(dist_args[3]),
            'dist': dist_args[4],
            'constants': 'consts' in dist_args,
            'homogeneous': 'homog' in dist_args,
        }
        return RandomIdealGenerator(**kwargs)


class FixedIdealGenerator(IdealGenerator):
    """Generator of repeated copies of a fixed ideal.

    Parameters
    ----------
    F : list
        List of polynomial generators for the ideal.

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


class RandomBinomialIdealGenerator(IdealGenerator):
    """Generator of random examples of binomial ideals.

    Parameters
    ----------
    n : int, optional
        Number of variables.
    d : int, optional
        Maximum degree of a chosen monomial.
    s : int, optional
        Number of generators of each ideal.
    dist : {'uniform', 'weighted', 'maximum'}, optional
        Type of distribution on monomials.
    constants : bool, optional
        Whether to include constants as monomials.
    homogeneous : bool, optional
        Whether the binomials are homogeneous.
    pure : bool, optional
        Whether the binomials are pure.
    coefficient_ring : ring, optional
        Coefficient ring for the polynomials.
    order : {'grevlex', 'lex', 'grlex'}, optional
        Monomial order.

    Examples
    --------
    >>> ideal_gen = RandomBinomialIdealGenerator(3, 5, 5)
    >>> ideal_gen.seed(123)
    >>> next(ideal_gen)
    [x0*x1**2*x2 + 3723 mod 32003*x2,
     x1*x2**2 + 26613 mod 32003*x1,
     x0*x1**3 + 23034 mod 32003*x1**2*x2,
     x0*x1**2 + 20977 mod 32003*x0*x1*x2,
     x0 + 18938 mod 32003*x2]

    """

    def __init__(self, n=3, d=20, s=10, dist='uniform',
                 constants=False, homogeneous=False, pure=False,
                 coefficient_ring=sp.FF(32003), order='grevlex'):
        ring = sp.xring('x:' + str(n), coefficient_ring, order)[0]
        self.s = s
        self.homogeneous = homogeneous
        self.pure = pure
        self.bases = [basis(ring, i) for i in range(d + 1)]
        self.rng = np.random.default_rng()
        self.degree_dist = degree_distribution(ring, d, dist=dist, constants=constants)
        self.ring = ring
        self.P = ring.domain.characteristic()

    def __next__(self):
        """Return a new random binomial ideal from the ring."""
        F = []
        for _ in range(self.s):

            c = -1 if self.pure else self.rng.integers(1, self.P)

            if self.homogeneous:
                d = self.rng.choice(len(self.degree_dist), p=self.degree_dist)
                d1, d2 = d, d
            else:
                d1, d2 = self.rng.choice(len(self.degree_dist), size=2, p=self.degree_dist)

            for _ in range(1000):
                m1 = self.rng.choice(self.bases[d1])
                m2 = self.rng.choice(self.bases[d2])
                # sympy always uses lex for m1 > m2, so get sort keys
                key1 = self.ring.order(m1.LM)
                key2 = self.ring.order(m2.LM)
                if key1 > key2:
                    F.append(m1 + c * m2)
                    break
                elif key1 < key2:
                    F.append(m2 + c * m1)
                    break
            else:
                raise RuntimeError('failed to generate two distinct random monomials after 1000 trials')

        return F

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)


class RandomIdealGenerator(IdealGenerator):
    """Generator of random examples of polynomial ideals.

    The number of terms generated for polynomial is two larger than
    a Poisson random variable sampled for each polynomial. Terms are
    not checked to be distinct, so could sum or cancel with other terms
    in producing the final polynomial.

    Parameters
    ----------
    n : int, optional
        Number of variables.
    d : int, optional
        Maximum degree of a chosen monomial.
    s : int, optional
        Number of generators of each ideal.
    lam : float, optional
        Parameter for the Poisson distribution on lengths.
    dist : {'uniform', 'weighted', 'maximum'}, optional
        Distribution of degrees of monomials.
    constants : bool, optional
        Whether to include constants as monomials.
    homogeneous : bool, optional
        Whether the binomials are homogeneous.
    coefficient_ring : ring, optional
        Coefficient ring for the polynomials.
    order : {'grevlex', 'lex', 'grlex'}, optional
        Monomial order.

    Examples
    --------
    >>> ideal_gen = RandomIdealGenerator(3, 5, 5, 0.5)
    >>> ideal_gen.seed(123)
    >>> next(ideal_gen)
    [x0**3*x2**2 + 10689 mod 32003*x0**2*x1 + 12547 mod 32003*x1*x2**2,
     x1*x2**4 + 15388 mod 32003*x1**2*x2 + 22355 mod 32003*x1*x2**2,
     x0*x1*x2**2 + 4665 mod 32003*x0**3 + 15800 mod 32003*x0**2*x1,
     x0**3*x2**2 + 8782 mod 32003*x1**2*x2**3 + 15890 mod 32003*x1*x2**2,
     x1**2*x2 + 30687 mod 32003*x1**2]

    """

    def __init__(self, n=3, d=20, s=10, lam=0.5, dist='uniform', constants=False, homogeneous=False,
                 coefficient_ring=sp.FF(32003), order='grevlex'):
        ring = sp.xring('x:' + str(n), coefficient_ring, order)[0]
        self.s = s
        self.lam = lam
        self.homogeneous = homogeneous
        self.bases = [basis(ring, i) for i in range(d + 1)]
        self.rng = np.random.default_rng()
        self.degree_dist = degree_distribution(ring, d, dist=dist, constants=constants)
        self.ring = ring
        self.P = ring.domain.characteristic()

    def __next__(self):
        """Return a new random ideal from the ring."""
        F = []
        for _ in range(self.s):
            f = 0
            terms = 2 + self.rng.poisson(self.lam)
            d = self.rng.choice(len(self.degree_dist), p=self.degree_dist)
            for _ in range(terms):
                c = self.rng.integers(1, self.P)
                f += c * self.rng.choice(self.bases[d])
                if not self.homogeneous:
                    d = self.rng.choice(len(self.degree_dist), p=self.degree_dist)
            F.append(f.monic())
        return F

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
