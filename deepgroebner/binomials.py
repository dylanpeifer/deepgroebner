"""A fast Buchberger environment that only supports binomials.

The BinomialBuchbergerEnv has the same interface and behavior as
the BuchbergerEnv, but is implemented using NumPy arrays for
speed.
"""

import numpy as np


CHAR = 32003 # for now, the characteristic is fixed


def grevlex(M1, M2):
    """Return 1 if M1 > M2 in the grevlex ordering, 0 if M1 = M2, and -1 if M1 < M2."""
    if np.sum(M1) > np.sum(M2):
        return 1
    elif np.sum(M1) < np.sum(M2):
        return -1
    else:
        for i in np.flip(M1 - M2):
            if i > 0:
                return -1
            if i < 0:
                return 1
        return 0 


def grevlex_key(M1):
    """Return the sort key for monomial M1 in grevlex."""
    return np.sum(M1), tuple(-np.flip(M1))


def euclidean(a, b):
    """Return GCD of a, b with coefficients from extended Euclidean algorithm."""
    if abs(b) > abs(a):
        x, y ,d = euclidean(b, a)
        return y, x, d

    if abs(b) == 0:
        return 1, 0, a

    x1, x2, y1, y2 = 0, 1, 1, 0
    while abs(b) > 0:
        q, r = divmod(a, b)
        x = x2 - q*x1
        y = y2 - q*y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y
    return x2, y2, a


def invert(a):
    """Return multiplicative inverse of a mod CHAR."""
    x, y, d = euclidean(a % CHAR, CHAR)
    return x % CHAR


class Binomial:

    def __init__(self, a1, a2, M1, M2):
        """
        a1, a2 should be integers,
        M1, M2 should be integer numpy arrays of the same length
        """
        assert M1.shape == M2.shape
        self.M1 = M1
        self.M2 = M2
        self.a1 = a1
        self.a2 = a2
        self.simplify()

    def __str__(self):
        return '%i x^%s + %i x^%s' % (self.a1, self.M1, self.a2, self.M2)

    def __repr__(self):
        return '%i x^%s + %i x^%s' % (self.a1, self.M1, self.a2, self.M2)
    
    def __eq__(self, other):
        return (self.a1 == other.a1 and
                self.a2 == other.a2 and
                np.array_equal(self.M1, other.M1) and
                np.array_equal(self.M2, other.M2))

    def simplify(self):
        """Arrange that M1 >= M2 in grevlex order and rescale to be monic."""
        self.a1 = self.a1 % CHAR
        if self.a1 == 0:
            self.M1 = np.zeros_like(self.M1)
        self.a2 = self.a2 % CHAR
        if self.a2 == 0:
            self.M2 = np.zeros_like(self.M2)

        order = grevlex(self.M1, self.M2)
        if order == -1:
            self.M1, self.M2 = self.M2, self.M1
            self.a1, self.a2 = self.a2, self.a1
        elif order == 0:
            self.M2 = np.zeros_like(self.M1)
            self.a1 = (self.a1 + self.a2) % CHAR
            self.a2 = 0

        if self.a1 is not 0:
            self.a2 = (self.a2 * invert(self.a1)) % CHAR
            self.a1 = 1


def to_binomial(f):
    """Return monic Binomial from SymPy polynomial."""
    terms = f.terms()
    while len(terms) < 2:
        terms.append(((0,) * f.ring.ngens, 0)) 
    (a1, M1), (a2, M2) = ((int(c), np.array(m, dtype=np.int)) for m, c in terms)
    return Binomial(a1, a2, M1, M2)


def spoly(f, g):
    """Return the s-polynomial of binomials f and g.

    Assumes both are simplified!
    """
    lcm = np.max([f.M1, g.M1], 0)
    return Binomial(f.a2, (-g.a2) % CHAR, lcm - f.M1 + f.M2, lcm - g.M1 + g.M2)


def reduce_monomial(a, M, F):
    """Return the remainder when monomial (a, M) is divided by binomials F."""
    r_a, r_M = a, M
    stats = {'steps': 0}
    found_divisor = True
    while found_divisor:
        if r_a == 0:
            return r_a, r_M, stats
        found_divisor = False
        for f in F:
            if all(r_M >= f.M1):
                assert f.a1 is not 0, 'No polynomial in the set %s can be 0' % F
                r_a = (- r_a * f.a2) % CHAR
                r_M = r_M - f.M1 + f.M2
                found_divisor = True
                stats['steps'] += 1
                break
    return r_a, r_M, stats


def reduce(g, F):
    """Return monic remainder when binomial g is divided by binomials F.
    
    Assumes g and the binomials in F are simplified.
    """
    # we reduce each monomial in g separately
    a1, M1, stats1 = reduce_monomial(g.a1, g.M1, F)
    a2, M2, stats2 = reduce_monomial(g.a2, g.M2, F)
    return Binomial(a1, a2, M1, M2), {'steps': stats1['steps'] + stats2['steps']}


def update(G, P, f, strategy='gebauermoeller'):
    """Return the new list of binomials and set of pairs when f is added to the basis G."""
    lmf = f.M1
    lmG = [g.M1 for g in G]

    if strategy == 'none':
        P_ = {(i, len(G)) for i in range(len(G))}
    elif strategy == 'lcm':
        P_ = {(i, len(G)) for i in range(len(G))
              if not np.array_equal(np.maximum(lmG[i], lmf), lmG[i] + lmf)}
    elif strategy == 'gebauermoeller':
        P = {p for p in P if (not np.all(np.maximum(lmG[p[0]], lmG[p[1]]) >= lmf) or
                              np.array_equal(np.maximum(lmG[p[0]], lmG[p[1]]), np.maximum(lmG[p[0]], lmf)) or
                              np.array_equal(np.maximum(lmG[p[0]], lmG[p[1]]), np.maximum(lmG[p[1]], lmf)))}
        lcm_dict = {}
        for i in range(len(G)):
            lcm_dict.setdefault(tuple(np.maximum(lmG[i], lmf)), []).append(i)
        minimalized_lcms = []
        for L in sorted(lcm_dict.keys(), key=grevlex_key):
            if all(not all(x >= y for x, y in zip(L, L_)) for L_ in minimalized_lcms):
                minimalized_lcms.append(L)
        P_ = set()
        for L in minimalized_lcms:
            if not any(np.array_equal(np.maximum(lmG[i], lmf), lmG[i] + lmf) for i in lcm_dict[L]):
                P_.add((min(lcm_dict[L]), len(G)))
    else:
        raise ValueError('unknown elimination strategy')

    return G + [f], P | P_


def minimalize(G):
    """Return a minimal Groebner basis from arbitrary Groebner basis G."""
    Gmin = []
    for f in G:
        if all([not all(f.M1 >= g.M1) for g in Gmin]):
            Gmin = [g for g in Gmin if not all(g.M1 >= f.M1)]
            Gmin.append(f)
    return Gmin


def interreduce(G):
    """Return a list of the polynomials in G reduced with respect to each other."""
    Gred = []
    for i in range(len(G)):
        g = reduce(G[i], G[:i] + G[i+1:])
        Gred.append(g) # the output of reduce is automatically monic
    return Gred


class BinomialBuchbergerEnv:
    """An environment for Groebner basis computation using Buchberger."""

    def __init__(self, ideal_gen, elimination='gebauermoeller',
                 sort_reducers=True, rewards='additions'):
        self.ideal_gen = ideal_gen
        self.elimination = elimination
        self.sort_reducers = sort_reducers
        self.rewards = rewards
        self.G = []
        self.P = set()
        self.reducers = []

    def reset(self, G=None):
        """Initialize the polynomial list and pair list from polynomials G."""
        F = [to_binomial(f) for f in next(self.ideal_gen)]
        self.G = []
        self.P = set()
        for f in F:
            self.G, self.P = update(self.G, self.P, f, strategy=self.elimination)
        if self.sort_reducers:
            self.reducers = sorted([f for f in F], key=lambda f: grevlex_key(f.M1))
        else:
            self.reducers = [f for f in F]
        return (self.G, self.P) if self.P else self.reset()

    def step(self, action):
        """Perform one reduction and return the new polynomial list and pair list."""
        i, j = action
        self.P.remove((i, j))
        s = spoly(self.G[i], self.G[j])
        r, stats = reduce(s, self.reducers)
        if not r.a1 == 0:
            self.G, self.P = update(self.G, self.P, r, strategy=self.elimination)
            if self.sort_reducers:
                self.reducers = sorted(self.reducers + [r], key=lambda f: grevlex_key(f.M1))
            else:
                self.reducers.append(r)
        reward = -(1 + stats['steps']) if self.rewards == 'additions' else -1
        return (self.G, self.P), reward, len(self.P) == 0, {}

    def render(self):
        print(self.P)
        print(self.G)
        print()


def monomial_matrix(state, k=1):
    """Return a (len(P), 2*num_variables) matrix of pairs of lead monomials."""
    G, P = state
    if not P:
        n = len(G[0].M1)
        return np.zeros((0, 2 * n * k), dtype=np.int)
    if k == 1:
        return np.array([np.concatenate((G[p[0]].M1, G[p[1]].M1)) for p in P])
    else:
        return np.array([np.concatenate((G[p[0]].M1, G[p[0]].M2, G[p[1]].M1, G[p[1]].M2)) for p in P])


class BinomialLeadMonomialsWrapper:
    """A wrapper for BinomialBuchbergerEnv that returns lead monomials as vectors."""

    def __init__(self, env, k=1):
        self.env = env
        self.k = k
        self.state = None
        self.pairs = []

    def reset(self):
        self.state = self.env.reset()
        self.pairs = list(self.state[1])
        return monomial_matrix(self.state, k=self.k)

    def step(self, action):
        action = self.pairs[action]
        self.state, reward, done, info = self.env.step(action)
        self.pairs = list(self.state[1])
        return monomial_matrix(self.state, k=self.k), reward, done, info

    def render(self):
        self.env.render()
