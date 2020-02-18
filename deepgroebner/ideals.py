"""Generators of ideals."""

import itertools as it
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


def random_nonzero_coeff(ring):
    """Return a random nonzero coefficient from the ring.

    Parameters
    ----------
    ring : polynomial ring over a finite field
        The polynomial ring.

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
        return ring.one * np.random.randint(1, c)


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


def random_monomial(D, ring, bases=None):
    """Return a random monomial from the ring.

    Parameters
    ----------
    D : array_like
        The list of probabilities indexed by degree (i.e., D[i] = P(degree i))
    ring : polynomial ring
        The polynomial ring.
    bases : list, optional
        The list of precomputed monomial bases for efficiency.
    """
    if bases is None:
        bases = [basis(ring, i) for i in range(len(D))]
    d = np.random.choice(len(D), p=D)
    return np.random.choice(bases[d])

    
def random_binomial(D, ring, homogeneous=False, pure=False, bases=None):
    """Return a random binomial from the ring.

    This function will only return binomials. If after 1000 tries it has only
    generated two identical monomials it will raise a RuntimeError.

    Parameters
    ----------
    D : array_like
        The list of probabilities indexed by degree (i.e., D[i] = P(degree i))
    ring : polynomial ring
        The polynomial ring.
    homogeneous : bool, optional
        Whether the binomial will be homogeneous.
    pure : bool, optional
        Whether the binomial will be pure.
    bases : list, optional
        The list of precomputed monomial bases for efficiency.

    Examples
    --------
    >>> import sympy as sp
    >>> ring, _ = sp.xring("x,y,z", sp.FF(32003), 'grevlex')
    >>> random_binomial([0, 0.5, 0.5], ring)
    x*y + 5982 mod 32003*z**2

    """
    if bases is None:
        bases = [basis(ring, i) for i in range(len(D))]
    c = -1 if pure else random_nonzero_coeff(ring)
    if homogeneous:
        d = np.random.choice(len(D), p=D)
        d1, d2 = d, d
    else:
        d1, d2 = np.random.choice(len(D), size=2, p=D)
    for i in range(1000):
        m1 = np.random.choice(bases[d1])
        m2 = np.random.choice(bases[d2])
        # sympy always uses lex for m1 > m2, so get sort keys
        key1 = ring.order(m1.LM)
        key2 = ring.order(m2.LM)
        if key1 > key2:
            return m1 + c * m2
        elif key1 < key2:
            return m2 + c * m1
    else:
        raise RuntimeError('failed to generate two distinct random monomials after 1000 trials')


def random_polynomial(D, lam, ring, bases=None):
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
    """
    if bases is None:
        bases = [basis(ring, i) for i in range(len(D))]
    t = 2 + np.random.poisson(lam)
    f = 0
    for _ in range(t):
        c = random_nonzero_coeff(ring)
        f += c * random_monomial(D, ring, bases=bases)
    return f


class RandomBinomialIdealGenerator:
    """Generator of random examples of binomial ideals.

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

    def __init__(self, n, d, s, coefficient_ring=sp.FF(32003), order='grevlex',
                 constants=False, degrees='uniform', homogeneous=False,
                 pure=False):
        self.ring = sp.xring('x:' + str(n), coefficient_ring, order)[0]
        self.dist = self._make_dist(n, d, constants, degrees)
        self.generators = s
        self.homogeneous = homogeneous
        self.pure = pure
        self.bases = [basis(self.ring, i) for i in range(d + 1)]

    def __next__(self):
        return [random_binomial(self.dist, self.ring,
                                homogeneous=self.homogeneous, pure=self.pure,
                                bases=self.bases)
                for _ in range(self.generators)]

    def __iter__(self):
        return self

    def _make_dist(self, n, d, constants, degrees):
        """Return the probability distribution of degrees."""
        head = [1] if constants else [0]
        if degrees == 'uniform':
            tail = d * [1]
        elif degrees == 'weighted':
            tail = [int(sp.binomial(n+i-1, n-1)) for i in range(1, d + 1)]
        elif degrees == 'maximum':
            tail = (d - 1) * [0] + [1]
        else:
            raise ValueError('unrecognized degree option')
        dist = np.array(head + tail)
        return dist / np.sum(dist)


class RandomIdealGenerator:
    """Generator of random examples of polynomial ideals.

    Parameters
    ----------
    n : int
        The number of variables.
    d : int
        The maximum degree of a chosen monomial.
    s : int
        The number of generators of each ideal.
    lam : float
        The lambda parameter for the poisson distribution on lengths.
    coefficient_ring : ring, optional
        The coefficient ring for the polynomials.
    order : {'grevlex', 'lex', 'grlex'}, optional
        The monomial order.
    constants : bool, optional
        Whether to include constants as monomials.
    degrees : {'uniform', 'weighted', 'maximum'}, optional
        The distribution of degrees of monomials.
    """

    def __init__(self, n, d, s, lam, coefficient_ring=sp.FF(32003), order='grevlex',
                 constants=False, degrees='uniform'):
        self.ring = sp.xring('x:' + str(n), coefficient_ring, order)[0]
        self.dist = self._make_dist(n, d, constants, degrees)
        self.lam = lam
        self.generators = s
        self.bases = [basis(self.ring, i) for i in range(d + 1)]

    def __next__(self):
        return [random_polynomial(self.dist, self.lam, self.ring,
                                  bases=self.bases)
                for _ in range(self.generators)]

    def __iter__(self):
        return self

    def _make_dist(self, n, d, constants, degrees):
        """Return the probability distribution of degrees."""
        head = [1] if constants else [0]
        if degrees == 'uniform':
            tail = d * [1]
        elif degrees == 'weighted':
            tail = [int(sp.binomial(n+i-1, n-1)) for i in range(1, d + 1)]
        elif degrees == 'maximum':
            tail = (d - 1) * [0] + [1]
        else:
            raise ValueError('unrecognized degree option')
        dist = np.array(head + tail)
        return dist / np.sum(dist)


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
