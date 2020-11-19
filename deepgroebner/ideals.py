"""Generators of ideals."""

import itertools as it
import numpy as np
import sympy as sp


class IdealGenerator:
    """Abstract base class for all ideal generators.

    Derived classes must implement a __next__ method which returns
    a list of SymPy polynomials. If randomness is used to generate
    polynomials then override the seed method.

    """

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        pass


def cyclic(n, coefficient_ring=sp.FF(32003), order='grevlex'):
    """Return the cyclic-n ideal."""
    R, gens = sp.xring('x:' + str(n), coefficient_ring, order)
    F = [sum(np.prod([gens[(i+k) % n] for k in range(d)]) for i in range(n))
         for d in range(1, n)]
    return F + [np.product(gens) - 1]


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


def random_nonzero_coeff(ring, rng=None):
    """Return a random nonzero coefficient from the ring.

    Parameters
    ----------
    ring : polynomial ring over a finite field
        Polynomial ring.
    rng : np.random.Generator or None, optional
        Random number generator.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("a,b", sp.FF(101), 'lex')
    >>> random_nonzero_coeff()
    11 mod 101

    """
    c = ring.domain.characteristic()
    if c <= 1:
        raise ValueError('ring of coefficients must be a finite field')
    else:
        rand = np.random.randint if rng is None else rng.integers
        return ring.one * rand(1, c)


def basis(ring, d):
    """Return a monomial basis of polynomial ring in degree d.

    A monomial basis in degree d is just all monomials of degree d in the
    polynomial ring.

    Parameters
    ----------
    ring : polynomial ring
        The polynomial ring.
    d : int
        The degree of the basis.

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


def random_monomial(D, ring, bases=None, rng=None):
    """Return a random monomial from the ring.

    Parameters
    ----------
    D : array_like
        The list of probabilities indexed by degree (i.e., D[i] = P(degree i))
    ring : polynomial ring
        The polynomial ring.
    bases : list, optional
        The list of precomputed monomial bases for efficiency.
    rng : np.random.Generator or None, optional
        Random number generator.

    """
    if bases is None:
        bases = [basis(ring, i) for i in range(len(D))]
    rng = np.random if rng is None else rng
    d = rng.choice(len(D), p=D)
    return rng.choice(bases[d])

    
def random_binomial(D, ring, homogeneous=False, pure=False, bases=None, rng=None):
    """Return a random binomial from the ring.

    This function will only return binomials. If after 1000 tries it has only
    generated two identical monomials it will raise a RuntimeError.

    Parameters
    ----------
    D : array_like
        List of probabilities indexed by degree (i.e., D[i] = P(degree i))
    ring : polynomial ring
        Polynomial ring containing the binomial.
    homogeneous : bool, optional
        Whether the binomial will be homogeneous.
    pure : bool, optional
        Whether the binomial will be pure.
    bases : list, optional
        List of precomputed monomial bases for efficiency.
    rng : np.random.Generator or None, optional
        Random number generator.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("x,y,z", sp.FF(32003), 'grevlex')
    >>> random_binomial([0, 0.5, 0.5], ring)
    x*y + 5982 mod 32003*z**2

    """
    if bases is None:
        bases = [basis(ring, i) for i in range(len(D))]
    c = -1 if pure else random_nonzero_coeff(ring, rng=rng)
    rng = np.random if rng is None else rng
    if homogeneous:
        d = rng.choice(len(D), p=D)
        d1, d2 = d, d
    else:
        d1, d2 = rng.choice(len(D), size=2, p=D)
    for i in range(1000):
        m1 = rng.choice(bases[d1])
        m2 = rng.choice(bases[d2])
        # sympy always uses lex for m1 > m2, so get sort keys
        key1 = ring.order(m1.LM)
        key2 = ring.order(m2.LM)
        if key1 > key2:
            return m1 + c * m2
        elif key1 < key2:
            return m2 + c * m1
    else:
        raise RuntimeError('failed to generate two distinct random monomials after 1000 trials')


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
    degrees : {'uniform', 'weighted', 'maximum'}, optional
        Distribution of degrees of monomials.
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
    seed : int or None, optional
        Seed for random number generator.

    Examples
    --------
    >>> ideal_gen = RandomBinomialIdealGenerator(3, 5, 5)
    >>> next(ideal_gen)
    [x0*x1**2*x2 + 3723 mod 32003*x2,
     x1*x2**2 + 26613 mod 32003*x1,
     x0*x1**3 + 23034 mod 32003*x1**2*x2,
     x0*x1**2 + 20977 mod 32003*x0*x1*x2,
     x0 + 18938 mod 32003*x2]

    """

    def __init__(self, n=3, d=20, s=10, degrees='uniform',
                 constants=False, homogeneous=False, pure=False,
                 coefficient_ring=sp.FF(32003), order='grevlex',
                 seed=None):
        self.D = self._make_dist(n, d, constants, degrees)
        self.ring = sp.xring('x:' + str(n), coefficient_ring, order)[0]
        self.s = s
        self.homogeneous = homogeneous
        self.pure = pure
        self.bases = [basis(self.ring, i) for i in range(d + 1)]
        self.seed(seed)

    def __next__(self):
        kwargs = {
            'homogeneous': self.homogeneous,
            'pure': self.pure,
            'bases': self.bases,
            'rng': self.rng,
        }
        return [random_binomial(self.D, self.ring, **kwargs) for _ in range(self.s)]

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _make_dist(self, n, d, constants, degrees):
        """Return the probability distribution of degrees."""
        head = [1] if constants else [0]
        if degrees == 'uniform':
             tail = [int(sp.binomial(n+i-1, n-1)) for i in range(1, d + 1)]
        elif degrees == 'weighted':
            tail = d * [1]
        elif degrees == 'maximum':
            tail = (d - 1) * [0] + [1]
        else:
            raise ValueError('unrecognized degree option')
        dist = np.array(head + tail)
        return dist / np.sum(dist)


def random_polynomial(D, lam, ring, bases=None, rng=None):
    """Return a random polynomial from the ring.

    The number of terms in the polynomial is one larger than
    a Poisson random variable with parameter lam.

    Parameters
    ----------
    D : array_like
        The list of probabilities indexed by degree (i.e., D[i] = P(degree i))
    lam : float
        The lambda parameter for the poisson distribution on length.
    ring : polynomial ring
        The polynomial ring.
    bases : list, optional
        The list of precomputed monomial bases for efficiency.
    rng : np.random.Generator or None, optional
        Random number generator.

    """
    if bases is None:
        bases = [basis(ring, i) for i in range(len(D))]
    rng = np.random if rng is None else rng
    t = 2 + rng.poisson(lam)
    f = 0
    for _ in range(t):
        c = random_nonzero_coeff(ring, rng=rng)
        f += c * random_monomial(D, ring, bases=bases, rng=rng)
    return f


class RandomIdealGenerator(IdealGenerator):
    """Generator of random examples of polynomial ideals.

    Parameters
    ----------
    n : int, optional
        Number of variables.
    d : int, optional
        Maximum degree of a chosen monomial.
    s : int, optional
        Number of generators of each ideal.
    lam : float, optional
        Lambda parameter for the poisson distribution on lengths.
    degrees : {'uniform', 'weighted', 'maximum'}, optional
        Distribution of degrees of monomials.
    constants : bool, optional
        Whether to include constants as monomials.
    coefficient_ring : ring, optional
        Coefficient ring for the polynomials.
    order : {'grevlex', 'lex', 'grlex'}, optional
        Monomial order.
    seed : int or None, optional
        Seed for random number generator.

    """

    def __init__(self, n=3, d=20, s=10, lam=0.5, degrees='uniform',
                 constants=False, coefficient_ring=sp.FF(32003), order='grevlex',
                 seed=None):
        self.ring = sp.xring('x:' + str(n), coefficient_ring, order)[0]
        self.dist = self._make_dist(n, d, constants, degrees)
        self.lam = lam
        self.generators = s
        self.bases = [basis(self.ring, i) for i in range(d + 1)]
        self.seed(seed)

    def __next__(self):
        return [random_polynomial(self.dist, self.lam, self.ring,
                                  bases=self.bases)
                for _ in range(self.generators)]

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _make_dist(self, n, d, constants, degrees):
        """Return the probability distribution of degrees."""
        head = [1] if constants else [0]
        if degrees == 'uniform':
            tail = [int(sp.binomial(n+i-1, n-1)) for i in range(1, d + 1)]
        elif degrees == 'weighted':
            tail = d * [1]
        elif degrees == 'maximum':
            tail = (d - 1) * [0] + [1]
        else:
            raise ValueError('unrecognized degree option')
        dist = np.array(head + tail)
        return dist / np.sum(dist)
